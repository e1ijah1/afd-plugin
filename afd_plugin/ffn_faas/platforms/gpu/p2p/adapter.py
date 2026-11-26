# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU-only adapter for the standalone GPU P2P backend."""

from __future__ import annotations

from argparse import ArgumentParser, Namespace
from functools import partial
from typing import TYPE_CHECKING

from afd_plugin.ffn_faas.interfaces import FFNLaunch
from afd_plugin.ffn_faas.platforms.gpu.p2p.settings import (
    P2P_CONNECTOR,
    validate_gpu_ffn_launch,
)
from afd_plugin.ffn_faas.platforms.launch import create_ffn_engine_parser

if TYPE_CHECKING:
    from afd_plugin.ffn_faas.interfaces import FFNBackend
    from afd_plugin.ffn_faas.platforms.gpu.p2p.settings import GPUFFNSettings


def create_backend(settings: GPUFFNSettings, args: Namespace) -> FFNBackend:
    from afd_plugin.ffn_faas.platforms.gpu.p2p.backend import GPUP2PFFNBackend

    return GPUP2PFFNBackend(settings, args)


class GPUP2PAdapter:
    backend_name = "gpu-p2p"

    def create_parser(self) -> ArgumentParser:
        parser = create_ffn_engine_parser()
        parser.set_defaults(skip_tokenizer_init=True)
        return parser

    def configure(self, args: Namespace) -> FFNLaunch:
        from vllm.engine.arg_utils import EngineArgs

        if args.model_tag is not None:
            args.model = args.model_tag
        engine_args = EngineArgs.from_cli_args(args)
        engine_args.receiver_shutdown_timeout = args.receiver_shutdown_timeout
        engine_args.status_host = args.status_host
        engine_args.status_port = args.status_port
        engine_args.attention_api_host = getattr(args, "attention_api_host", "127.0.0.1")
        engine_args.attention_api_port = getattr(args, "attention_api_port", 19301)
        settings = validate_gpu_ffn_launch(engine_args)
        return FFNLaunch(
            rank=settings.rank,
            local_rank=settings.local_rank,
            model_path=str(settings.model_path),
            backend_name=P2P_CONNECTOR,
            shutdown_timeout=settings.shutdown_timeout,
            backend_factory=partial(create_backend, settings, args),
            initial_status={"p2p_paired": False},
        )
