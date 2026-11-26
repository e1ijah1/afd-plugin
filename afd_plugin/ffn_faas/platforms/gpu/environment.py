# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Own one standalone FFN rank's CUDA device, compute groups, and workspace."""

from __future__ import annotations

import os
from datetime import timedelta
from typing import TYPE_CHECKING

from afd_plugin.ffn_faas.platforms.gpu.settings import GPUFFNSettings

if TYPE_CHECKING:
    import torch
    from vllm.config import VllmConfig
    from vllm.distributed.parallel_state import GroupCoordinator


COMPUTE_GROUP_TIMEOUT_SECONDS = 180


class GPUFFNEnvironment:
    """Initialize and release only resources created by this F-only process."""

    def __init__(self, settings: GPUFFNSettings) -> None:
        self.settings = settings
        self._device: torch.device | None = None
        self._ep_group: GroupCoordinator | None = None
        self._groups_owned = False
        self._workspace_owned = False

    @property
    def device(self) -> torch.device:
        if self._device is None:
            raise RuntimeError("GPU FFN device has not been initialized")
        return self._device

    @property
    def ep_group(self) -> GroupCoordinator:
        if self._ep_group is None:
            raise RuntimeError("GPU FFN expert group has not been initialized")
        return self._ep_group

    @property
    def owns_resources(self) -> bool:
        return self._groups_owned or self._workspace_owned

    def initialize(self, config: VllmConfig) -> None:
        """Follow the pinned worker setup order without constructing a Worker."""
        if self.owns_resources or self._device is not None:
            raise RuntimeError("GPU FFN environment is already initialized")

        import torch
        from vllm import envs
        from vllm.config import set_current_vllm_config
        from vllm.distributed.parallel_state import (
            get_ep_group,
            model_parallel_is_initialized,
        )
        from vllm.platforms import current_platform
        from vllm.utils.torch_utils import set_random_seed
        from vllm.v1.worker.gpu_worker import init_worker_distributed_environment
        from vllm.v1.worker.workspace import (
            init_workspace_manager,
            is_workspace_manager_initialized,
        )

        settings = self.settings
        if (settings.world_size, settings.rank, settings.local_rank) != (1, 0, 0):
            raise ValueError("GPU FFN prototype environment requires one local rank")
        if torch.distributed.is_initialized() or model_parallel_is_initialized():
            raise RuntimeError("GPU FFN cannot borrow existing distributed groups")
        if is_workspace_manager_initialized():
            raise RuntimeError("GPU FFN cannot replace an existing workspace manager")
        if torch.cuda.device_count() < 1:
            raise RuntimeError("GPU FFN requires one visible CUDA device")

        self._device = torch.device("cuda:0")
        torch.cuda.set_device(self._device)
        current_platform.check_if_supports_dtype(config.model_config.dtype)
        torch.set_float32_matmul_precision(envs.VLLM_FLOAT32_MATMUL_PRECISION)

        try:
            with set_current_vllm_config(config):
                set_random_seed(config.model_config.seed)
                self._groups_owned = True
                init_worker_distributed_environment(
                    config,
                    rank=settings.rank,
                    distributed_init_method="env://",
                    local_rank=self._device.index or 0,
                    backend=current_platform.dist_backend,
                )
                group = get_ep_group()
                if group.world_size != settings.world_size or group.rank_in_group != settings.rank:
                    raise ValueError("native GPU expert group differs from the F-only world")
                self._ep_group = group
                os.environ.pop("TORCHELASTIC_USE_AGENT_STORE", None)
                init_workspace_manager(self._device, num_ubatches=1, num_lanes=1)
                self._workspace_owned = True
        except BaseException:
            try:
                self.close()
            except BaseException as cleanup_error:
                raise RuntimeError(
                    "GPU FFN initialization failed and owned resources could not be released"
                ) from cleanup_error
            raise

    def activate_thread(self) -> None:
        """Select the owned CUDA device on the actual receiver thread."""
        import torch

        torch.cuda.set_device(self.device)

    def synchronize(self) -> None:
        """Fence this device before model, connector, or group owners are released."""
        import torch

        torch.cuda.synchronize(self.device)

    def close(self) -> None:
        """Release owned groups/workspace after callers have stopped device work."""
        if not self.owns_resources:
            self._device = None
            self._ep_group = None
            return

        from vllm.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )
        from vllm.v1.worker.workspace import reset_workspace_manager

        self.synchronize()
        if self._groups_owned:
            destroy_model_parallel()
            destroy_distributed_environment()
            self._groups_owned = False
            self._ep_group = None
        if self._workspace_owned:
            reset_workspace_manager()
            self._workspace_owned = False
        self._device = None
