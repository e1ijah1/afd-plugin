# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Standalone Qwen MoE execution over the legacy GPU P2P connector."""

from __future__ import annotations

import threading
from argparse import Namespace

from vllm.config import set_current_vllm_config
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.layers.quantization.utils.quant_utils import kNvfp4Static

from afd_plugin.connectors.base import AFDControlPlaneClosedError
from afd_plugin.connectors.gpu.p2p import P2pNcclAFDConnector
from afd_plugin.ffn_faas.interfaces import FFNStepResult
from afd_plugin.ffn_faas.platforms.ffn_model.qwen3_6.gpu.model import QwenMoEModelLoader
from afd_plugin.ffn_faas.platforms.gpu.environment import GPUFFNEnvironment
from afd_plugin.ffn_faas.platforms.gpu.settings import GPUFFNSettings
from afd_plugin.ffn_faas.service import FFNService, ServicePhase
from afd_plugin.v1.worker.ffn_step import execute_ffn_forward_step


class GPUP2PFFNBackend:
    def __init__(self, settings: GPUFFNSettings, args: Namespace) -> None:
        self.settings = settings
        self.args = args
        self.environment = GPUFFNEnvironment(settings)
        self.config = None
        self.model = None
        self.connector = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._executing = False
        self._receiver_exited = False
        self._cancel_thread = None
        self._cancel_error = None

    def prepare(self, service: FFNService) -> None:
        config = EngineArgs.from_cli_args(self.args).create_engine_config()
        config.parallel_config.distributed_executor_backend = "external_launcher"
        self.config = config
        self.environment.initialize(config)
        with set_current_vllm_config(config):
            self.model = QwenMoEModelLoader(config.load_config).load_model(
                vllm_config=config, model_config=config.model_config
            )

            layers = self.model.num_layers
            nvfp4 = 0
            moe = 0
            for module in self.model.modules():
                method = getattr(module, "quant_method", None)
                spec = getattr(method, "spec", None)
                kernel = getattr(method, "kernel", None)
                if spec is not None and spec.weight == kNvfp4Static and spec.activation is None:
                    if type(kernel).__name__ != "MarlinNvFp4LinearKernel":
                        raise RuntimeError("standalone FFN NVFP4 linear kernel is not MARLIN")
                    nvfp4 += 1
                routed = getattr(getattr(module, "experts", None), "routed_experts", None)
                if routed is not None:
                    quant = routed.quant_method
                    if type(quant).__name__ != "ModelOptNvFp4FusedMoE" or quant.nvfp4_backend.name != "MARLIN":
                        raise RuntimeError("standalone FFN MoE kernel is not NVFP4 MARLIN")
                    moe += 1
            if (moe, nvfp4) != (layers, layers * 2):
                raise RuntimeError(f"standalone FFN kernel admission failed: moe={moe}, nvfp4={nvfp4}, layers={layers}")
            service.phase(ServicePhase.CONNECTING, model_loaded=True, marlin_moe_layers=moe, marlin_nvfp4_linears=nvfp4)
            self.connector = P2pNcclAFDConnector(
                rank=self.settings.rank,
                local_rank=self.settings.local_rank,
                vllm_config=config,
                afd_config=self.settings.afd_config,
                role_rank=self.settings.rank,
            )
            self.connector.init_afd_connector()
            service.phase(ServicePhase.CONNECTING, p2p_paired=True)

    def start_receiver(self) -> None:
        self.environment.activate_thread()

    def execute_step(self) -> FFNStepResult:
        assert self.connector is not None and self.model is not None and self.config is not None
        if self._stop.is_set():
            return FFNStepResult.STOP
        try:
            payload = self.connector.control_plane.recv_dp_metadata_list()
        except AFDControlPlaneClosedError:
            if self._stop.is_set():
                return FFNStepResult.STOP
            raise
        with self._lock:
            if self._stop.is_set():
                raise RuntimeError("P2P stop raced with incoming work; retain resources")
            self._executing = True
        try:
            execute_ffn_forward_step(
                vllm_config=self.config,
                afd_config=self.settings.afd_config,
                connector=self.connector,
                model=self.model,
                num_layers=self.model.num_layers,
                dp_metadata_list=payload.dp_metadata_list,
                is_graph_capturing=payload.is_graph_capturing,
                is_warmup=payload.is_warmup,
                is_graph_replaying=payload.is_graph_replaying,
            )
            self.environment.synchronize()
        finally:
            with self._lock:
                self._executing = False
        if payload.is_profile or payload.is_warmup:
            return FFNStepResult.WARMUP
        return FFNStepResult.WORK

    def request_stop(self) -> None:
        self._stop.set()
        self._cancel_thread = threading.Thread(target=self._cancel_receive, daemon=True)
        self._cancel_thread.start()

    def _cancel_receive(self) -> None:
        try:
            with self._lock:
                if self._executing:
                    raise RuntimeError("P2P stop requested during work; retain resources")
                assert self.connector is not None
                self.connector.control_plane.close()
        except BaseException as error:
            self._cancel_error = error

    def receiver_exited(self) -> None:
        self._receiver_exited = True

    def close(self) -> None:
        if not self._receiver_exited:
            raise RuntimeError("P2P receiver has not exited; retain resources")
        if self._cancel_thread is not None:
            self._cancel_thread.join(timeout=self.settings.shutdown_timeout)
            if self._cancel_thread.is_alive():
                raise RuntimeError("P2P cancellation is pending; retain resources")
            if self._cancel_error is not None:
                raise RuntimeError("P2P cancellation failed; retain resources") from self._cancel_error
        self.environment.synchronize()
        assert self.connector is not None
        self.connector.close()
        self.environment.close()
        self.model = None
        self.connector = None
        self.config = None
