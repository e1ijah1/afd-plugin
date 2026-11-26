# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Select an implemented adapter without importing its tensor/runtime stack."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from afd_plugin.ffn_faas.interfaces import FFNPlatformAdapter

SUPPORTED_BACKENDS = ("ascend-cam", "gpu-p2p")


def get_platform_adapter(name: str) -> FFNPlatformAdapter:
    if name == "ascend-cam":
        from afd_plugin.ffn_faas.platforms.npu.adapter import AscendCAMAdapter

        return AscendCAMAdapter()
    if name == "gpu-p2p":
        from afd_plugin.ffn_faas.platforms.gpu.adapter import GPUP2PAdapter

        return GPUP2PAdapter()
    raise ValueError(f"unsupported standalone FFN backend: {name}")
