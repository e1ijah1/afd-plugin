# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""CPU-safe contracts for the standalone Qwen full-MoE owner."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
from torch import nn  # noqa: E402
from vllm.model_executor.models import qwen3_5 as native  # noqa: E402
from vllm.model_executor.models import qwen3_next  # noqa: E402
from vllm.model_executor.models.utils import AutoWeightsLoader  # noqa: E402

from afd_plugin.ffn_faas.platforms.ffn_model.qwen3_6.gpu.model import (  # noqa: E402
    QwenDecoderMoE,
    QwenLanguageModelContainer,
    QwenMoEOwner,
    QwenTextModelContainer,
)
from tests.unit.fakes.weights import OneShotWeights  # noqa: E402


def test_qwen_text_container_maps_text_weights_once_into_native_moe(monkeypatch):
    received = []

    class FakeAutoWeightsLoader:
        def __init__(self, module):
            self.module = module

        def load_weights(self, weights, *, mapper):
            mapped = list(mapper.apply(weights))
            received.extend(name for name, _ in mapped)
            assert self.module is container
            return {"model.layers.0.mlp.shared_expert.gate_up_proj.weight"}

    monkeypatch.setattr(
        "afd_plugin.ffn_faas.platforms.ffn_model.qwen3_6.gpu.model.AutoWeightsLoader",
        FakeAutoWeightsLoader,
    )
    container = object.__new__(QwenLanguageModelContainer)
    nn.Module.__init__(container)
    weights = OneShotWeights(
        [
            "model.layers.0.mlp.shared_expert.gate_proj.weight",
            "model.layers.0.mlp.shared_expert.up_proj.weight",
        ]
    )

    loaded = container.load_weights(weights)

    assert loaded == {"model.layers.0.mlp.shared_expert.gate_up_proj.weight"}
    assert received == [
        "model.layers.0.mlp.shared_expert.gate_up_proj.weight",
        "model.layers.0.mlp.shared_expert.gate_up_proj.weight",
    ]
    assert weights.iterations == 1


def test_qwen_root_filters_checkpoint_weights_and_applies_prefix_mapper_once(monkeypatch):
    received = []

    class FakeAutoWeightsLoader:
        def __init__(self, module):
            self.module = module

        def load_weights(self, weights, *, mapper):
            mapped = list(mapper.apply(weights))
            received.extend(name for name, _ in mapped)
            assert self.module is owner
            return {"language_model.model.layers.0.mlp.gate.weight"}

    monkeypatch.setattr(
        "afd_plugin.ffn_faas.platforms.ffn_model.qwen3_6.gpu.model.AutoWeightsLoader",
        FakeAutoWeightsLoader,
    )
    owner = object.__new__(QwenMoEOwner)
    nn.Module.__init__(owner)
    owner.model_config = SimpleNamespace(hf_text_config=SimpleNamespace(num_experts=1))
    owner.quant_config = None
    owner.num_layers = 1
    weights = OneShotWeights(
        [
            "model.language_model.embed_tokens.weight",
            "model.language_model.layers.0.mlp.gate.weight",
            "model.language_model.layers.0.mlp.experts.0.down_proj.weight",
            "model.language_model.layers.0.mlp.experts.0.gate_proj.weight",
            "model.language_model.layers.0.mlp.experts.0.up_proj.weight",
            "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight",
            "model.language_model.layers.0.mlp.shared_expert.up_proj.weight",
            "model.language_model.layers.0.mlp.shared_expert.down_proj.weight",
            "model.language_model.layers.0.mlp.shared_expert_gate.weight",
            "model.visual.patch_embed.proj.weight",
        ]
    )

    loaded = owner.load_weights(weights)

    assert loaded == {"language_model.model.layers.0.mlp.gate.weight"}
    assert received == [
        "language_model.model.layers.0.mlp.gate.weight",
        "language_model.model.layers.0.mlp.experts.0.down_proj.weight",
        "language_model.model.layers.0.mlp.experts.0.gate_proj.weight",
        "language_model.model.layers.0.mlp.experts.0.up_proj.weight",
        "language_model.model.layers.0.mlp.shared_expert.gate_proj.weight",
        "language_model.model.layers.0.mlp.shared_expert.up_proj.weight",
        "language_model.model.layers.0.mlp.shared_expert.down_proj.weight",
        "language_model.model.layers.0.mlp.shared_expert_gate.weight",
    ]
    assert weights.iterations == 1


def test_decoder_moe_calls_native_internal_router_and_combine():
    hidden = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    decoder = object.__new__(QwenDecoderMoE)
    nn.Module.__init__(decoder)

    class NativeMoE(nn.Module):
        def forward(self, value):
            return value + 1

    decoder.mlp = NativeMoE()

    assert torch.equal(decoder.compute(hidden), hidden + 1)


def test_pure_owner_constructs_only_the_moe_layer_hierarchy(monkeypatch):
    calls = []

    class FakeMoE(nn.Module):
        def __init__(self, *, vllm_config, prefix, reduce_results):
            super().__init__()
            calls.append((prefix, reduce_results))
            self.gate = nn.Linear(4, 8, bias=False)
            self.shared_expert_gate = nn.Linear(4, 1, bias=False)
            self.experts = nn.Linear(4, 4, bias=False)

        def forward(self, hidden_states):
            return hidden_states + 1

    monkeypatch.setattr(qwen3_next, "Qwen3NextSparseMoeBlock", FakeMoE)
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type="qwen3_5_moe_text", num_hidden_layers=4)
        ),
        quant_config=None,
        parallel_config=SimpleNamespace(
            use_sequence_parallel_moe=False,
            enable_eplb=False,
            enable_elastic_ep=False,
        ),
        additional_config={"afd": {"compute_gate_on_attention": False}},
    )

    owner = QwenMoEOwner(vllm_config=config, prefix="")

    assert owner.get_experts_layer_indices() == (0, 1, 2, 3)
    modules = dict(owner.named_modules())
    assert all(f"language_model.model.layers.{i}.mlp" in modules for i in range(4))
    assert all(f"language_model.model.layers.{i}.mlp.gate" in modules for i in range(4))
    assert calls == [
        (f"language_model.model.layers.{index}.mlp", False)
        for index in range(4)
    ]
    assert not any(
        token in name.lower()
        for name in dict(owner.named_modules())
        for token in ("attention", "tokenizer", "scheduler", "kv_cache", "lm_head", "causallm")
    )
    hidden = torch.zeros((2, 4))
    assert torch.equal(owner.compute_ffn_output(hidden, 2), hidden + 1)


def test_qwen_owner_rejects_attention_gate_and_dynamic_expert_placement(monkeypatch):
    base = {
        "model_config": SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type="qwen3_5_moe_text", num_hidden_layers=1)
        ),
        "quant_config": None,
        "additional_config": {"afd": {"compute_gate_on_attention": True}},
        "parallel_config": SimpleNamespace(
            use_sequence_parallel_moe=False,
            enable_eplb=False,
            enable_elastic_ep=False,
        ),
    }
    with pytest.raises(ValueError, match="complete MoE gate on F"):
        QwenMoEOwner(vllm_config=SimpleNamespace(**base))

    base["additional_config"] = {"afd": {"compute_gate_on_attention": False}}
    base["parallel_config"] = SimpleNamespace(
        use_sequence_parallel_moe=False,
        enable_eplb=True,
        enable_elastic_ep=False,
    )
    with pytest.raises(ValueError, match="static expert placement"):
        QwenMoEOwner(vllm_config=SimpleNamespace(**base))


@pytest.mark.parametrize("missing_suffix", ["weight", "weight_scale", "weight_scale_2", "input_scale"])
def test_quantized_four_layer_owner_requires_every_source_weight(monkeypatch, missing_suffix):
    class ConsumeLoader:
        def __init__(self, module):
            pass

        def load_weights(self, weights, *, mapper):
            list(weights)
            return set()

    monkeypatch.setattr(
        "afd_plugin.ffn_faas.platforms.ffn_model.qwen3_6.gpu.model.AutoWeightsLoader",
        ConsumeLoader,
    )
    owner = object.__new__(QwenMoEOwner)
    nn.Module.__init__(owner)
    owner.num_layers = 4
    owner.model_config = SimpleNamespace(hf_text_config=SimpleNamespace(num_experts=2))
    recipes = {}
    names = []
    for layer in range(4):
        source = f"model.language_model.layers.{layer}.mlp"
        target = f"language_model.model.layers.{layer}.mlp"
        names.extend((f"{source}.gate.weight", f"{source}.shared_expert_gate.weight"))
        recipes[f"{target}.experts"] = {"quant_algo": "W4A16_NVFP4"}
        for projection in ("gate_proj", "up_proj", "down_proj"):
            recipes[f"{target}.shared_expert.{projection}"] = {"quant_algo": "W4A16_NVFP4"}
            for group in ("experts.0", "experts.1", "shared_expert"):
                names.extend(f"{source}.{group}.{projection}.{suffix}" for suffix in
                             ("weight", "weight_scale", "weight_scale_2", "input_scale"))
    owner.quant_config = SimpleNamespace(quantized_layers=recipes)
    assert owner.load_weights((name, None) for name in names) == set()
    missing = f"model.language_model.layers.3.mlp.experts.1.up_proj.{missing_suffix}"
    with pytest.raises(ValueError, match=missing.replace(".", r"\.")):
        owner.load_weights((name, None) for name in names if name != missing)


def test_owner_exposes_native_quantization_mapper():
    assert QwenMoEOwner.hf_to_vllm_mapper is native.Qwen3_5MoeForConditionalGeneration.hf_to_vllm_mapper
    assert QwenMoEOwner.packed_modules_mapping is native.Qwen3_5MoeForConditionalGeneration.packed_modules_mapping
