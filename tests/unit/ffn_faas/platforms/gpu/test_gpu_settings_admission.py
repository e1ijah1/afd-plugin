# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU-side admission for the single-rank GPU P2P FFN prototype."""

from __future__ import annotations

import json
import socket
from argparse import Namespace

import pytest

pytest.importorskip("vllm")

from afd_plugin.ffn_faas.platforms.gpu.settings import validate_gpu_ffn_launch


@pytest.fixture
def launch_case(tmp_path):
    from vllm.engine.arg_utils import EngineArgs
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    parser.set_defaults(distributed_executor_backend="external_launcher")
    parsed = parser.parse_args(
        [
            "--model",
            str(tmp_path),
            "--tensor-parallel-size",
            "1",
            "--data-parallel-size",
            "1",
            "--enable-expert-parallel",
            "--enforce-eager",
            "--language-model-only",
            "--dtype",
            "bfloat16",
            "--max-model-len",
            "1024",
            "--max-num-seqs",
            "1",
            "--max-num-batched-tokens",
            "1024",
            "--mamba-cache-mode",
            "align",
            "--kernel-config",
            json.dumps(
                {
                    "linear_backend": "auto",
                    "linear_backend_per_quant": {"fp8_w8a8": "torch"},
                    "enable_flashinfer_autotune": False,
                }
            ),
            "--additional-config",
            json.dumps(
                {
                    "afd": {
                        "role": "ffn",
                        "connector": "P2pNcclAFDConnector",
                        "host": "127.0.0.1",
                        "port": 19339,
                        "num_attention_ranks": 1,
                        "num_ffn_ranks": 1,
                        "compute_gate_on_attention": False,
                        "afd_process_group_timeout_s": 180,
                        "connector_extra_config": {},
                    }
                }
            ),
        ]
    )
    args = EngineArgs.from_cli_args(parsed)
    args.receiver_shutdown_timeout = 30.0
    args.status_host = "127.0.0.1"
    args.status_port = _free_loopback_port()
    args.attention_api_host = "127.0.0.1"
    args.attention_api_port = _free_loopback_port()
    return args


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _launcher_env(**changes: str) -> dict[str, str]:
    return {
        "WORLD_SIZE": "1",
        "RANK": "0",
        "LOCAL_RANK": "0",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "19338",
        **changes,
    }


def test_valid_strict_single_rank_launch_is_cpu_only(launch_case):
    settings = validate_gpu_ffn_launch(launch_case, environ=_launcher_env())

    assert settings.rank == settings.local_rank == 0
    assert settings.world_size == 1
    assert settings.ffn_only_master_port == 19338
    assert settings.rendezvous_port == 19339
    assert settings.management_host == settings.attention_api_host == "127.0.0.1"
    assert settings.management_port != settings.attention_api_port
    assert settings.afd_config.connector_extra_config == {}


def test_omitted_connector_extra_config_is_accepted(launch_case):
    config = {**launch_case.additional_config}
    config["afd"] = {**config["afd"]}
    del config["afd"]["connector_extra_config"]
    launch_case.additional_config = config
    settings = validate_gpu_ffn_launch(launch_case, environ=_launcher_env())
    assert not settings.afd_config.connector_extra_config


@pytest.mark.parametrize(
    "change, message",
    [
        ({"tensor_parallel_size": 2}, "TP1/DP1"),
        ({"data_parallel_size": 2}, "TP1/DP1"),
        ({"pipeline_parallel_size": 2}, "pipeline parallelism"),
        ({"prefill_context_parallel_size": 2}, "context parallelism"),
        ({"enable_expert_parallel": False}, "expert parallelism"),
        ({"enable_elastic_ep": True}, "dynamic expert placement"),
        ({"enable_eplb": True}, "dynamic expert placement"),
        ({"enforce_eager": False}, "eager, non-DBO"),
        ({"enable_dbo": True}, "eager, non-DBO"),
        ({"speculative_config": {"method": "mtp"}}, "speculative decoding"),
        ({"language_model_only": False}, "language-model-only"),
        ({"dtype": "float16"}, "bfloat16"),
        ({"max_model_len": 2048}, "1024/1/1024"),
        ({"max_num_seqs": 2}, "1024/1/1024"),
        ({"max_num_batched_tokens": 2048}, "1024/1/1024"),
        ({"mamba_cache_mode": "none"}, "mamba-cache-mode align"),
        ({"distributed_executor_backend": "mp"}, "external_launcher"),
        ({"kernel_config": Namespace(enable_flashinfer_autotune=True)}, "FlashInfer autotune"),
        ({"kernel_config": Namespace(enable_flashinfer_autotune=False, linear_backend="torch", linear_backend_per_quant={"fp8_w8a8": "torch"})}, "global linear backend"),
        ({"kernel_config": Namespace(enable_flashinfer_autotune=False, linear_backend="auto", linear_backend_per_quant={"fp8_w8a8": "torch", "fp8_w4a8": "torch"})}, "fp8_w8a8=torch only"),
        ({"kernel_config": None}, "explicit strict kernel config"),
        ({"data_parallel_rank": 1}, "data-parallel-rank"),
        ({"receiver_shutdown_timeout": 0.0}, "finite and positive"),
    ],
)
def test_unsupported_engine_configuration_is_rejected_before_collective(
    launch_case, change, message
):
    for key, value in change.items():
        setattr(launch_case, key, value)

    with pytest.raises(ValueError, match=message):
        validate_gpu_ffn_launch(launch_case, environ=_launcher_env())


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("connector", "DeepEPM2NAFDConnector", "synchronous P2pNcclAFDConnector"),
        ("role", "attention", "AFD role mismatch"),
        ("compute_gate_on_attention", True, "complete MoE gate on the FFN role"),
        ("num_attention_ranks", 2, "exactly one Attention and one FFN rank"),
        ("num_ffn_ranks", 2, "num_attention_ranks >= num_ffn_ranks"),
        ("host", "0.0.0.0", "loopback A/F rendezvous"),
        ("connector_extra_config", {"control_protocol": "v1"}, "empty connector_extra_config"),
    ],
)
def test_unsupported_afd_configuration_is_rejected_before_collective(
    launch_case, field, value, message
):
    config = {**launch_case.additional_config}
    config["afd"] = {**config["afd"], field: value}
    launch_case.additional_config = config

    with pytest.raises(ValueError, match=message):
        validate_gpu_ffn_launch(launch_case, environ=_launcher_env())


@pytest.mark.parametrize(
    "environment",
    [
        {},
        _launcher_env(WORLD_SIZE="2"),
        _launcher_env(RANK="1"),
        _launcher_env(LOCAL_RANK="1"),
        _launcher_env(MASTER_ADDR="0.0.0.0"),
        _launcher_env(MASTER_PORT="bad"),
    ],
)
def test_invalid_torchrun_environment_is_rejected(launch_case, environment):
    with pytest.raises(ValueError):
        validate_gpu_ffn_launch(launch_case, environ=environment)


def test_rendezvous_management_and_fonly_ports_must_be_distinct(launch_case):
    launch_case.status_port = 19338

    with pytest.raises(ValueError, match="ports must differ"):
        validate_gpu_ffn_launch(launch_case, environ=_launcher_env())

def test_attention_api_port_defaults_to_the_g0_port(launch_case):
    del launch_case.attention_api_port

    settings = validate_gpu_ffn_launch(launch_case, environ=_launcher_env())

    assert settings.attention_api_port == 19301


def test_attention_api_port_conflict_is_rejected(launch_case):
    launch_case.attention_api_port = launch_case.status_port

    with pytest.raises(ValueError, match="ports must differ"):
        validate_gpu_ffn_launch(launch_case, environ=_launcher_env())


def test_occupied_management_port_is_rejected(launch_case):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        launch_case.status_port = listener.getsockname()[1]

        with pytest.raises(ValueError, match="unavailable"):
            validate_gpu_ffn_launch(launch_case, environ=_launcher_env())
