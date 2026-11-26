# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU-only launch contract for the single-rank Qwen3.6 P2P FFN prototype."""

from __future__ import annotations

import math
import os
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from afd_plugin.config import AFDConfig, parse_afd_config

if TYPE_CHECKING:
    from argparse import Namespace


P2P_CONNECTOR = "P2pNcclAFDConnector"
STRICT_KERNEL_CONFIG = {
    "enable_flashinfer_autotune": False,
    "linear_backend_per_quant": {"fp8_w8a8": "torch"},
}


@dataclass(frozen=True)
class GPUFFNSettings:
    """Validated one-rank launch facts; constructing settings touches no device."""

    model_path: Path
    rank: int
    local_rank: int
    world_size: int
    ffn_only_master_port: int
    rendezvous_host: str
    rendezvous_port: int
    management_host: str
    management_port: int
    attention_api_host: str
    attention_api_port: int
    shutdown_timeout: float
    afd_config: AFDConfig


def _required_loopback_port(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer port")
    try:
        port = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be an integer port") from error
    if not 0 < port < 65536:
        raise ValueError(f"{name} must be in 1..65535")
    return port


def _assert_tcp_port_available(host: str, port: int, *, name: str) -> None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind((host, port))
    except OSError as error:
        raise ValueError(f"{name} port {port} is unavailable on {host}") from error


def validate_gpu_ffn_launch(
    args: Namespace,
    *,
    environ: dict[str, str] | os._Environ[str] | None = None,
) -> GPUFFNSettings:
    """Reject unsupported launch configurations before any process-group init."""
    env = os.environ if environ is None else environ
    model_path = Path(args.model).expanduser()
    if not model_path.is_dir():
        raise ValueError(f"model path must be a local directory: {model_path}")

    afd_config = parse_afd_config(args, expected_role="ffn")
    if afd_config.connector != P2P_CONNECTOR or afd_config.async_dp:
        raise ValueError("gpu-p2p requires synchronous P2pNcclAFDConnector")
    if afd_config.compute_gate_on_attention:
        raise ValueError("gpu-p2p requires the complete MoE gate on the FFN role")
    if (afd_config.num_attention_ranks, afd_config.num_ffn_ranks) != (1, 1):
        raise ValueError("gpu-p2p prototype requires exactly one Attention and one FFN rank")
    if afd_config.host != "127.0.0.1":
        raise ValueError("gpu-p2p prototype requires loopback A/F rendezvous")
    if afd_config.connector_extra_config:
        raise ValueError("gpu-p2p requires empty connector_extra_config")

    if args.tensor_parallel_size != 1 or args.data_parallel_size != 1:
        raise ValueError("gpu-p2p prototype requires TP1/DP1")
    if args.pipeline_parallel_size != 1:
        raise ValueError("gpu-p2p prototype does not support pipeline parallelism")
    if args.prefill_context_parallel_size != 1 or args.decode_context_parallel_size != 1:
        raise ValueError("gpu-p2p prototype does not support context parallelism")
    if not args.enable_expert_parallel:
        raise ValueError("gpu-p2p requires static expert parallelism")
    if args.enable_elastic_ep or args.enable_eplb:
        raise ValueError("gpu-p2p does not support dynamic expert placement")
    if not args.enforce_eager or args.enable_dbo:
        raise ValueError("gpu-p2p prototype requires eager, non-DBO execution")
    graph_mode = args.compilation_config.cudagraph_mode
    if graph_mode is not None and graph_mode.name != "NONE":
        raise ValueError("gpu-p2p prototype does not support CUDA graphs")
    if args.speculative_config is not None:
        raise ValueError("gpu-p2p prototype does not support speculative decoding")
    if args.enable_lora:
        raise ValueError("gpu-p2p prototype does not support LoRA")
    if not args.language_model_only:
        raise ValueError("gpu-p2p prototype requires --language-model-only")
    if args.dtype != "bfloat16":
        raise ValueError("gpu-p2p prototype requires bfloat16 hidden-state transport")
    if args.max_model_len != 1024 or args.max_num_seqs != 1 or args.max_num_batched_tokens != 1024:
        raise ValueError("gpu-p2p prototype requires the frozen 1024/1/1024 short-request profile")
    if args.mamba_cache_mode != "align":
        raise ValueError("gpu-p2p prototype requires --mamba-cache-mode align")
    kernel_config = args.kernel_config
    if kernel_config is None:
        raise ValueError("gpu-p2p requires an explicit strict kernel config")
    if kernel_config.enable_flashinfer_autotune is not False:
        raise ValueError("gpu-p2p strict profile must disable FlashInfer autotune")
    if kernel_config.linear_backend != "auto":
        raise ValueError("gpu-p2p strict profile requires the global linear backend to remain auto")
    if kernel_config.linear_backend_per_quant != {"fp8_w8a8": "torch"}:
        raise ValueError("gpu-p2p strict profile requires fp8_w8a8=torch only")
    if args.distributed_executor_backend != "external_launcher":
        raise ValueError("gpu-p2p requires the external_launcher backend")
    if not math.isfinite(args.receiver_shutdown_timeout) or args.receiver_shutdown_timeout <= 0:
        raise ValueError("receiver shutdown timeout must be finite and positive")

    try:
        world_size = int(env["WORLD_SIZE"])
        rank = int(env["RANK"])
        local_rank = int(env["LOCAL_RANK"])
        master_port = _required_loopback_port(env["MASTER_PORT"], name="F-only MASTER_PORT")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("gpu-p2p must be launched with torchrun rank environment") from error
    if (world_size, rank, local_rank) != (1, 0, 0):
        raise ValueError("gpu-p2p prototype requires torchrun WORLD_SIZE=1 RANK=0 LOCAL_RANK=0")
    if env.get("MASTER_ADDR", "127.0.0.1") != "127.0.0.1":
        raise ValueError("gpu-p2p F-only process group must use loopback MASTER_ADDR")
    if args.data_parallel_rank not in (None, 0):
        raise ValueError("--data-parallel-rank must be zero for the single F-only rank")

    attention_api_host = getattr(args, "attention_api_host", "127.0.0.1")
    management_host = args.status_host
    if attention_api_host != "127.0.0.1" or management_host != "127.0.0.1":
        raise ValueError("gpu-p2p management and Attention API must bind to loopback")
    attention_api_port = _required_loopback_port(
        getattr(args, "attention_api_port", 19301), name="Attention API"
    )
    management_port = _required_loopback_port(args.status_port, name="FFN management")
    master_port = _required_loopback_port(master_port, name="F-only MASTER_PORT")
    ports = {
        "F-only MASTER_PORT": master_port,
        "A/F rendezvous": afd_config.port,
        "Attention API": attention_api_port,
        "FFN management": management_port,
    }
    if len(set(ports.values())) != len(ports):
        raise ValueError("F-only, rendezvous, Attention API, and management ports must differ")
    _assert_tcp_port_available(management_host, management_port, name="FFN management")

    return GPUFFNSettings(
        model_path=model_path,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        ffn_only_master_port=master_port,
        rendezvous_host=afd_config.host,
        rendezvous_port=afd_config.port,
        management_host=management_host,
        management_port=management_port,
        attention_api_host=attention_api_host,
        attention_api_port=attention_api_port,
        shutdown_timeout=args.receiver_shutdown_timeout,
        afd_config=afd_config,
    )
