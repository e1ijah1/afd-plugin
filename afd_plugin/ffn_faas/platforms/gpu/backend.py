# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""FFN-only routed experts composed with the existing GPU M2N runtime.

Imported inside the externally launched F process, after the adapter has
validated the launch. No vLLM Worker, upstream ModelRunner, EngineCore,
scheduler, KV cache or complete CausalLM is constructed. Step, graph, DBO and
transport logic are not re-implemented: they belong to
``GPUFFNRoutedExpertsModelRunner`` and ``DeepEPM2NAFDConnector``.
"""

from __future__ import annotations

import threading
from argparse import Namespace

import torch
from vllm.config import set_current_vllm_config

from afd_plugin.connectors.gpu.deepep_m2n.connector import DeepEPM2NAFDConnector
from afd_plugin.ffn_faas.interfaces import FFNStepResult
from afd_plugin.ffn_faas.model_utils import parameter_storage_bytes
from afd_plugin.ffn_faas.platforms.gpu.environment import GPUFFNEnvironment
from afd_plugin.ffn_faas.platforms.gpu.settings import GPUFFNConfig
from afd_plugin.ffn_faas.service import FFNService, ServicePhase
from afd_plugin.v1.worker.ffn_routed_experts_model_runner import (
    GPUFFNRoutedExpertsModelRunner,
    RoutedExpertsModel,
)
from afd_plugin.validation import validate_gpu_model_runner_v2_config


class DeepEPM2NFFNBackend:
    """Keep weights and communication owners until receiver exit is proven.

    Implements the ``FFNBackend`` lifecycle over DeepEP M2N: ``prepare`` runs
    on the host thread; ``start_receiver``, ``execute_step`` and
    ``receiver_exited`` run on the receiver thread; ``close`` then releases
    the runtime, the compute groups and the weights, in that order.
    ``request_stop`` ends an idle receive through its NCCL control group.
    """

    def __init__(self, settings: GPUFFNConfig, args: Namespace) -> None:
        """Hold the CPU-side launch facts; native work starts in ``prepare``."""
        self.settings = settings
        self.args = args
        self.environment = GPUFFNEnvironment(settings)
        self.runner: GPUFFNRoutedExpertsModelRunner | None = None
        self.model: RoutedExpertsModel | None = None
        self._cancel_requested = threading.Event()
        self._cancel_thread: threading.Thread | None = None
        self._cancel_error: BaseException | None = None

    def prepare(self, service: FFNService) -> None:
        """Bring this F rank from launch to paired with Attention.

        Runs on the host thread, in the order that lets a bad deployment fail
        before the expert weights are read:

        1. build the F-only ``VllmConfig`` (AFD architecture, Triton MoE pin),
           which resolves ``--device-ids``, and admit the deployment with the
           Worker's GPU ModelRunnerV2 checks, which also resolve the CUDA
           graph policy; no device or weight is touched;
        2. select and check the GPU through that assignment, and the
           afdEP/DeepGEMM stack;
        3. create the F-only NCCL world and vLLM's model-parallel groups;
        4. construct the routed-experts runner with that graph policy, which
           creates the connector (no collective yet);
        5. load the FFN-only routed-experts shell through vLLM's native loader;
        6. start the connector: the collective M2N handshake with Attention,
           the model-info check, the NCCL control group, the lanes' DeepEP
           buffers and the dispatch-kernel precompile.

        Phases: ``loading`` through step 5 (with ``loaded_experts`` and
        ``total_experts`` after the load), ``connecting`` from step 5 on; the
        host marks ``ready`` afterwards. CUDA graphs are not captured here: they
        are captured lazily, driven by Attention's warmup and capture
        payloads, exactly as on the Worker path. Every owner is assigned
        before the next native step, so a failure leaves it reachable.
        """
        environment = self.environment
        config = environment.build_config(self.args)
        # No Worker runs here, so admit the deployment as AFDFFNWorker does.
        cudagraph_policy = validate_gpu_model_runner_v2_config(
            config, expected_role="ffn", device_type="cuda"
        )
        environment.initialize_device(config)
        environment.initialize_compute(config)
        with set_current_vllm_config(config):
            self.runner = GPUFFNRoutedExpertsModelRunner(
                config,
                environment.device,
                afd_cudagraph_policy=cudagraph_policy,
            )
            connector = self.runner.connector
            if not isinstance(connector, DeepEPM2NAFDConnector):
                raise TypeError(
                    f"standalone GPU FFN needs DeepEPM2NAFDConnector, got "
                    f"{type(connector).__name__}"
                )
            self.runner.load_model()
            model: RoutedExpertsModel = self.runner.get_model()
            self.model = model
            # vLLM's loader reports no per-expert progress; this rank's experts
            # are all loaded once it returns.
            total_experts = sum(
                len(self.runner.local_experts[layer_idx].expert_ids)
                for layer_idx in self.runner.moe_layer_indices
            )
            service.loading_progress(total_experts, total_experts)
            if not isinstance(model, torch.nn.Module):
                raise TypeError(
                    f"FFN-only model must be a torch module, got {type(model).__name__}"
                )
            parameter_bytes = parameter_storage_bytes(model)
            local_expert_ids = self.runner.local_experts[
                self.runner.moe_layer_indices[0]
            ].expert_ids
            service.phase(
                ServicePhase.CONNECTING,
                model_loaded=True,
                parameter_bytes=parameter_bytes,
                local_expert_first=local_expert_ids[0],
                local_expert_last=local_expert_ids[-1],
            )
            self.runner.initialize_afd_connector()
            assert connector.attn_tp_size is not None
            assert connector.capacity_tokens_per_rank is not None
            assert connector.bucket_ladder is not None
            assert connector.network_plan is not None
            service.phase(
                ServicePhase.CONNECTING,
                m2n_paired=True,
                attn_tp_size=connector.attn_tp_size,
                capacity_tokens_per_rank=connector.capacity_tokens_per_rank,
                num_buckets=len(connector.bucket_ladder),
                network_hybrid=connector.network_plan.hybrid,
                compute_ep_world_size=environment.ep_group.world_size,
            )

    def start_receiver(self) -> None:
        """Select the GPU once on the receiving thread, as the Worker does."""
        self.environment.activate_thread()

    def execute_step(self) -> FFNStepResult:
        """Run one synchronized payload, or stop after receive cancellation."""
        assert self.runner is not None
        completed = self.runner.execute_connector_driven_step(
            cancel_requested=self._cancel_requested.is_set
        )
        return FFNStepResult.WORK if completed else FFNStepResult.STOP

    def request_stop(self) -> None:
        """End an idle receive without blocking the host on NCCL teardown."""
        self._cancel_requested.set()
        self._cancel_thread = threading.Thread(target=self._cancel_receive, daemon=True)
        self._cancel_thread.start()

    def _cancel_receive(self) -> None:
        try:
            if self.runner is not None:
                control_plane = self.runner.connector.control_plane
                if control_plane is not None:
                    control_plane.close()
        except BaseException as error:
            self._cancel_error = error

    def receiver_exited(self) -> None:
        """Record on the receiver thread that its loop ended cleanly."""
        assert self.runner is not None
        self.runner.mark_receiver_exited()

    def close(self) -> None:
        """Release CUDA graphs, DeepEP buffers, compute groups and weights.

        Only after the receiver exit was proven. ``runner.shutdown`` resets
        the captured graphs, drops their pool, clears vLLM's forward-context
        registrations and the model reference, and closes the connector
        (control group and lanes); then the environment destroys the F-only
        groups, and the last reference to the FFN-only model is dropped. The
        connector's gloo bootstrap group outlives this, which only matters
        for an in-process restart, which is not supported.
        """
        assert self.runner is not None
        if not self.runner.receiver_exited:
            raise RuntimeError("GPU FFN receiver has not exited; retain resources")
        if self._cancel_thread is not None:
            self._cancel_thread.join(timeout=self.settings.shutdown_timeout)
            if self._cancel_thread.is_alive():
                raise RuntimeError("GPU FFN cancellation is pending; retain resources")
            if self._cancel_error is not None:
                raise RuntimeError("GPU FFN cancellation failed; retain resources") from self._cancel_error
        self.runner.shutdown()
        self.environment.close()
        self.model = None
