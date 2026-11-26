# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Shared synchronous eager execution for connector-driven GPU FFN steps."""

from __future__ import annotations

from typing import Any

import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.forward_context import DPMetadata, get_forward_context, set_forward_context

from afd_plugin.config import AFDConfig
from afd_plugin.connectors import AFDConnectorBase, AFDDPMetadata
from afd_plugin.connectors.metadata import AFDControlPayload
from afd_plugin.v1.worker.ffn_metadata import (
    _set_moe_layer_index,
    aggregate_ffn_token_counts,
    project_ffn_token_counts_to_dp,
)


def execute_ffn_forward_step(
    *,
    vllm_config: VllmConfig,
    afd_config: AFDConfig,
    connector: AFDConnectorBase,
    model: Any,
    num_layers: int,
    dp_metadata_list: dict[int, DPMetadata | AFDDPMetadata],
    is_graph_capturing: bool = False,
    is_warmup: bool = False,
    is_graph_replaying: bool = False,
    update_connector_state: bool = True,
) -> torch.Tensor | None:
    """Run one received metadata step through an explicitly supplied FFN model.

    The traditional worker uses this path. Metadata reception, graph dispatch,
    lifecycle, and ownership remain with the caller.
    """
    if update_connector_state:
        control_plane = connector.control_plane
        if control_plane is None:
            raise RuntimeError("GPU FFN step requires a connector control plane")
        control_plane.update_state_from_dp_metadata(
            AFDControlPayload(
                dp_metadata_list=dp_metadata_list,
                is_graph_capturing=is_graph_capturing,
                is_warmup=is_warmup,
                is_graph_replaying=is_graph_replaying,
            ),
        )

    experts_layer_indices = frozenset(model.get_experts_layer_indices())
    layer_indices = (
        tuple(sorted(experts_layer_indices))
        if afd_config.compute_gate_on_attention
        else tuple(range(max(int(num_layers or 0), 1)))
    )
    stage_ids = sorted(int(stage_idx) for stage_idx in dp_metadata_list) or [0]
    ffn_dp_metadata_list = {
        stage_idx: _make_ffn_dp_metadata(
            dp_metadata_list[stage_idx],
            connector=connector,
            vllm_config=vllm_config,
        )
        for stage_idx in stage_ids
    }
    recv_routing = getattr(model, "afd_requires_attention_routing", False)
    recv_input_ids = (
        False if recv_routing else getattr(model, "afd_requires_input_ids", False)
    )
    rank_ffn_output = None
    with (
        torch.inference_mode(),
        set_current_vllm_config(vllm_config),
        set_forward_context(attn_metadata=None, vllm_config=vllm_config),
    ):
        forward_context = get_forward_context()
        for layer_idx in layer_indices:
            uses_remote_experts = layer_idx in experts_layer_indices
            routing_spec = (
                model.get_experts_routing_spec(layer_idx)
                if uses_remote_experts and afd_config.compute_gate_on_attention
                else None
            )
            for stage_idx in stage_ids:
                recv_kwargs: dict[str, Any] = {}
                if routing_spec is not None:
                    recv_kwargs["routing_spec"] = routing_spec
                if recv_routing:
                    recv_kwargs["recv_routing"] = True
                elif recv_input_ids:
                    recv_kwargs["recv_input_ids"] = True
                payload = connector.recv_attn_output(
                    ubatch_idx=stage_idx,
                    **recv_kwargs,
                )
                hidden_states = payload.hidden_states
                context = payload.context
                metadata = context.metadata
                metadata.layer_idx = layer_idx
                metadata.stage_idx = stage_idx
                if forward_context is not None:
                    forward_context.dp_metadata = ffn_dp_metadata_list[
                        metadata.stage_idx
                    ]
                    forward_context.additional_kwargs["afd_metadata"] = metadata
                    _set_moe_layer_index(forward_context, layer_idx)
                if (
                    uses_remote_experts
                    and afd_config.compute_gate_on_attention
                ):
                    router_logits = payload.router_logits
                    assert router_logits is not None
                    rank_ffn_output = model.compute_experts_output(
                        hidden_states,
                        layer_idx,
                        router_logits,
                    )
                elif payload.input_ids is None:
                    rank_ffn_output = model.compute_ffn_output(
                        hidden_states,
                        layer_idx,
                    )
                else:
                    rank_ffn_output = model.compute_ffn_output(
                        hidden_states,
                        layer_idx,
                        input_ids=payload.input_ids,
                    )
                connector.send_ffn_output(rank_ffn_output, context)
    return rank_ffn_output


def _make_ffn_dp_metadata(
    dp_metadata: DPMetadata | AFDDPMetadata,
    *,
    connector: AFDConnectorBase,
    vllm_config: VllmConfig,
) -> AFDDPMetadata:
    attention_counts = tuple(int(count) for count in dp_metadata.num_tokens_across_dp_cpu)
    ffn_counts = aggregate_ffn_token_counts(
        attention_counts,
        attention_size=int(connector.attn_size),
        ffn_size=int(connector.ffn_size),
    )
    dp_counts = project_ffn_token_counts_to_dp(
        ffn_counts,
        dp_size=int(vllm_config.parallel_config.data_parallel_size),
    )
    return AFDDPMetadata(num_tokens_across_dp_cpu=dp_counts)
