# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Pure native Qwen MoE ownership for the standalone GPU FFN process."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.utils import initialize_model
from vllm.model_executor.models import qwen3_5 as native
from vllm.model_executor.models import qwen3_next as next_native
from vllm.model_executor.models.utils import AutoWeightsLoader

from afd_plugin.model_executor.models._weight_utils import filter_role_weights
from afd_plugin.model_executor.models.qwen3_5 import _checkpoint_weight_roles

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig


class QwenDecoderMoE(nn.Module):
    """Keep the native MoE under its checkpoint-compatible ``mlp`` prefix."""

    def __init__(self, vllm_config: VllmConfig, layer_idx: int, prefix: str) -> None:
        super().__init__()
        self.mlp = next_native.Qwen3NextSparseMoeBlock(
            vllm_config=vllm_config,
            prefix=f"{prefix}.mlp",
            reduce_results=False,
        )
        self.layer_idx = layer_idx

    def compute(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.mlp(hidden_states)


class QwenTextModelContainer(nn.Module):
    """Hold native MoE layers under ``model.layers.N.mlp``."""

    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()
        hf_config = vllm_config.model_config.hf_text_config
        self.layers = nn.ModuleDict(
            {
                str(layer_idx): QwenDecoderMoE(
                    vllm_config,
                    layer_idx,
                    prefix=f"{prefix}.layers.{layer_idx}",
                )
                for layer_idx in range(hf_config.num_hidden_layers)
            }
        )
        self.num_layers = int(hf_config.num_hidden_layers)

    def get_experts_layer_indices(self) -> tuple[int, ...]:
        return tuple(range(self.num_layers))

    def compute_ffn_output(self, hidden_states: torch.Tensor, layer_idx: int) -> torch.Tensor:
        if not 0 <= layer_idx < self.num_layers:
            raise IndexError(f"Qwen MoE layer index {layer_idx} is out of range")
        return self.layers[str(layer_idx)].compute(hidden_states)


class QwenLanguageModelContainer(nn.Module):
    """Map the native text checkpoint prefix once, without a CausalLM shell."""

    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__()
        self.model = QwenTextModelContainer(vllm_config, f"{prefix}.model")

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return AutoWeightsLoader(self).load_weights(
            weights,
            mapper=native.Qwen3_5Model.hf_to_vllm_mapper,
        )

    def get_experts_layer_indices(self) -> tuple[int, ...]:
        return self.model.get_experts_layer_indices()

    def compute_ffn_output(self, hidden_states: torch.Tensor, layer_idx: int) -> torch.Tensor:
        return self.model.compute_ffn_output(hidden_states, layer_idx)


class QwenMoEOwner(nn.Module):
    """Own only Qwen router, routed/shared experts, shared gate, and MoE output."""

    hf_to_vllm_mapper = native.Qwen3_5MoeForConditionalGeneration.hf_to_vllm_mapper
    packed_modules_mapping = native.Qwen3_5MoeForConditionalGeneration.packed_modules_mapping
    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "model") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.quant_config = vllm_config.quant_config
        if config.model_type != "qwen3_5_moe_text":
            raise ValueError(f"Qwen3.6 MoE owner requires qwen3_5_moe_text, got {config.model_type}")
        afd_config = vllm_config.additional_config.get("afd", {})
        if afd_config.get("compute_gate_on_attention", False):
            raise ValueError("standalone Qwen MoE owner requires the complete MoE gate on F")
        if vllm_config.parallel_config.use_sequence_parallel_moe:
            raise ValueError("standalone Qwen MoE owner does not support sequence-parallel MoE")
        if vllm_config.parallel_config.enable_eplb or vllm_config.parallel_config.enable_elastic_ep:
            raise ValueError("standalone Qwen MoE owner requires static expert placement")

        self.language_model = QwenLanguageModelContainer(vllm_config, native.maybe_prefix(prefix, "language_model"))
        self.num_layers = int(config.num_hidden_layers)

    def get_experts_layer_indices(self) -> tuple[int, ...]:
        return self.language_model.get_experts_layer_indices()

    def compute_ffn_output(self, hidden_states: torch.Tensor, layer_idx: int) -> torch.Tensor:
        return self.language_model.compute_ffn_output(hidden_states, layer_idx)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        config = self.model_config.hf_text_config
        quantized_layers = getattr(self.quant_config, "quantized_layers", {})
        required = set()
        for layer_idx in range(self.num_layers):
            source = f"model.language_model.layers.{layer_idx}.mlp"
            target = f"language_model.model.layers.{layer_idx}.mlp"
            required.update((f"{source}.gate.weight", f"{source}.shared_expert_gate.weight"))
            for group, count in (("experts", config.num_experts), ("shared_expert", 1)):
                for projection in ("gate_proj", "up_proj", "down_proj"):
                    quant_target = f"{target}.{group}"
                    if group == "shared_expert":
                        quant_target += f".{projection}"
                    recipe = quantized_layers.get(quant_target, {})
                    suffixes = ("weight",)
                    if recipe.get("quant_algo", "").upper() == "W4A16_NVFP4":
                        suffixes = ("weight", "weight_scale", "weight_scale_2", "input_scale")
                    for expert_idx in range(count):
                        expert = f".{expert_idx}" if group == "experts" else ""
                        required.update(f"{source}.{group}{expert}.{projection}.{suffix}" for suffix in suffixes)

        seen = set()

        def checked_weights():
            for name, weight in filter_role_weights(weights, role="ffn", classify=_checkpoint_weight_roles):
                if name in required:
                    seen.add(name)
                yield name, weight

        loaded = AutoWeightsLoader(self).load_weights(checked_weights(), mapper=self.hf_to_vllm_mapper)
        missing = required - seen
        if missing:
            raise ValueError(f"Missing required Qwen MoE checkpoint weights: {sorted(missing)}")
        return loaded


class QwenMoEModelLoader(DefaultModelLoader):
    """Delegate checkpoint reading and the single native post-load to vLLM."""

    def create_model(
        self,
        vllm_config: VllmConfig,
        model_config: ModelConfig,
        prefix: str = "",
    ) -> nn.Module:
        return initialize_model(
            vllm_config=vllm_config,
            model_config=model_config,
            prefix=prefix,
            model_class=QwenMoEOwner,
        )
