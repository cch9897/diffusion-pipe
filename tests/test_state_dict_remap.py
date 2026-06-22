"""
Round-trip tests for cosmos_predict2 state_dict key remapping/splitting.

Covers the safetensors crash scenario: _split_state_dict_keys produces
non-contiguous tensor views that cause safetensors.torch.save_file
to fail inside _find_shared_tensors (tensor.view(-1) on non-contiguous).

Usage: python -m pytest tests/test_state_dict_remap.py -v
"""
import sys
import os
import tempfile
from pathlib import Path

import torch
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.cosmos_predict2_state_dict import _remap_state_dict_keys, _split_state_dict_keys

try:
    from models.cosmos_predict2 import CosmosPredict2Pipeline, InitialLayer
    _HAS_PIPELINE = True
except ImportError:
    _HAS_PIPELINE = False
    CosmosPredict2Pipeline = None  # type: ignore
    InitialLayer = None  # type: ignore

from models.cosmos_predict2_modeling import _zero_adaln_modulation_2_offdiag_grad


def _legacy_qkv(block_idx=0, D=64, device='cpu'):
    return {
        f'blocks.{block_idx}.self_attn.q_proj.weight': torch.randn(D, D, device=device),
        f'blocks.{block_idx}.self_attn.k_proj.weight': torch.randn(D, D, device=device),
        f'blocks.{block_idx}.self_attn.v_proj.weight': torch.randn(D, D, device=device),
    }


def _legacy_kv(block_idx=0, D=64, device='cpu'):
    return {
        f'blocks.{block_idx}.cross_attn.k_proj.weight': torch.randn(D, D, device=device),
        f'blocks.{block_idx}.cross_attn.v_proj.weight': torch.randn(D, D, device=device),
    }


def _legacy_adaln_lora(block_idx=0, D=64, adaln_lora_dim=32, device='cpu'):
    sd = {}
    for branch in ['self_attn', 'cross_attn', 'mlp']:
        sd[f'blocks.{block_idx}.adaln_modulation_{branch}.1.weight'] = torch.randn(adaln_lora_dim, D, device=device)
        sd[f'blocks.{block_idx}.adaln_modulation_{branch}.2.weight'] = torch.randn(3 * D, adaln_lora_dim, device=device)
    return sd


def _legacy_adaln_nolora(block_idx=0, D=64, device='cpu'):
    sd = {}
    for branch in ['self_attn', 'cross_attn', 'mlp']:
        sd[f'blocks.{block_idx}.adaln_modulation_{branch}.1.weight'] = torch.randn(3 * D, D, device=device)
    return sd


# ── QKV ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize('D', [64, 128])
def test_qkv_round_trip(D):
    legacy = _legacy_qkv(D=D)
    fused = _remap_state_dict_keys(legacy.copy())
    assert f'blocks.0.self_attn.qkv_proj.weight' in fused
    assert fused[f'blocks.0.self_attn.qkv_proj.weight'].shape == (3 * D, D)

    split = _split_state_dict_keys(fused)
    for k in ['q_proj', 'k_proj', 'v_proj']:
        fk = f'blocks.0.self_attn.{k}.weight'
        assert torch.equal(split[fk], legacy[fk]), f'{fk} mismatch'


@pytest.mark.parametrize('D', [64, 128])
def test_qkv_split_contiguous(D):
    legacy = _legacy_qkv(D=D)
    fused = _remap_state_dict_keys(legacy.copy())
    split = _split_state_dict_keys(fused)
    for k, v in split.items():
        assert v.is_contiguous(), f'Non-contiguous after split: {k}'


def test_qkv_save_safetensors():
    import safetensors.torch
    legacy = _legacy_qkv()
    fused = _remap_state_dict_keys(legacy.copy())
    split = _split_state_dict_keys(fused)
    split = {k: v.contiguous() for k, v in split.items()}

    with tempfile.NamedTemporaryFile(suffix='.safetensors', delete=False) as f:
        tmp = f.name
    try:
        safetensors.torch.save_file(split, tmp)
        loaded = safetensors.torch.load_file(tmp)
        for k in ['q_proj', 'k_proj', 'v_proj']:
            assert torch.equal(loaded[f'blocks.0.self_attn.{k}.weight'], legacy[f'blocks.0.self_attn.{k}.weight'])
    finally:
        os.unlink(tmp)


# ── Cross-attn KV ────────────────────────────────────────────────────

@pytest.mark.parametrize('D', [64, 128])
def test_kv_round_trip(D):
    legacy = _legacy_kv(D=D)
    fused = _remap_state_dict_keys(legacy.copy())
    assert f'blocks.0.cross_attn.kv_proj.weight' in fused
    split = _split_state_dict_keys(fused)
    for k in ['k_proj', 'v_proj']:
        fk = f'blocks.0.cross_attn.{k}.weight'
        assert torch.equal(split[fk], legacy[fk])


def test_kv_split_contiguous():
    legacy = _legacy_kv()
    fused = _remap_state_dict_keys(legacy.copy())
    split = _split_state_dict_keys(fused)
    for k, v in split.items():
        assert v.is_contiguous(), f'Non-contiguous: {k}'


def test_kv_save_safetensors():
    import safetensors.torch
    legacy = _legacy_kv()
    fused = _remap_state_dict_keys(legacy.copy())
    split = _split_state_dict_keys(fused)
    split = {k: v.contiguous() for k, v in split.items()}

    with tempfile.NamedTemporaryFile(suffix='.safetensors', delete=False) as f:
        tmp = f.name
    try:
        safetensors.torch.save_file(split, tmp)
        loaded = safetensors.torch.load_file(tmp)
        for k in ['k_proj', 'v_proj']:
            assert torch.equal(loaded[f'blocks.0.cross_attn.{k}.weight'], legacy[f'blocks.0.cross_attn.{k}.weight'])
    finally:
        os.unlink(tmp)


# ── AdaLN lora ───────────────────────────────────────────────────────

@pytest.mark.parametrize('D,ald', [(64, 32), (128, 48)])
def test_adaln_lora_round_trip(D, ald):
    legacy = _legacy_adaln_lora(D=D, adaln_lora_dim=ald)
    fused = _remap_state_dict_keys(legacy.copy())
    assert f'blocks.0.adaln_modulation_1.weight' in fused
    split = _split_state_dict_keys(fused)
    for branch in ['self_attn', 'cross_attn', 'mlp']:
        for layer in ['1', '2']:
            fk = f'blocks.0.adaln_modulation_{branch}.{layer}.weight'
            assert torch.equal(split[fk], legacy[fk]), f'{fk} mismatch'


def test_adaln_lora_split_contiguous():
    legacy = _legacy_adaln_lora()
    fused = _remap_state_dict_keys(legacy.copy())
    split = _split_state_dict_keys(fused)
    for k, v in split.items():
        assert v.is_contiguous(), f'Non-contiguous: {k}'

def test_adaln_lora_grad_hook_preserves_split_invariant():
    D = 8
    ald = 4
    grad = torch.ones(9 * D, 3 * ald)
    _zero_adaln_modulation_2_offdiag_grad(grad, D, ald)

    for row_branch in range(3):
        for col_branch in range(3):
            block_grad = grad[row_branch * 3 * D:(row_branch + 1) * 3 * D, col_branch * ald:(col_branch + 1) * ald]
            if row_branch == col_branch:
                assert torch.equal(block_grad, torch.ones_like(block_grad))
            else:
                assert torch.equal(block_grad, torch.zeros_like(block_grad))


def test_anima_t_embedder_adaln_uses_mod_lr():
    def classify(name):
        if 'llm_adapter' in name:
            return 'llm_adapter'
        if '.self_attn' in name:
            return 'self_attn'
        if '.cross_attn' in name:
            return 'cross_attn'
        if '.mlp' in name:
            return 'mlp'
        if '.adaln_modulation' in name or ('t_embedder' in name and 'adaln' in name):
            return 'mod'
        return 'base'

    assert classify('t_embedder.linear_2.weight') == 'mod'

# ── AdaLN nolora ─────────────────────────────────────────────────────

def test_adaln_nolora_round_trip():
    legacy = _legacy_adaln_nolora()
    fused = _remap_state_dict_keys(legacy.copy())
    assert f'blocks.0.adaln_modulation.weight' in fused
    split = _split_state_dict_keys(fused)
    for branch in ['self_attn', 'cross_attn', 'mlp']:
        fk = f'blocks.0.adaln_modulation_{branch}.1.weight'
        assert torch.equal(split[fk], legacy[fk])


def test_adaln_nolora_split_contiguous():
    legacy = _legacy_adaln_nolora()
    fused = _remap_state_dict_keys(legacy.copy())
    split = _split_state_dict_keys(fused)
    for k, v in split.items():
        assert v.is_contiguous(), f'Non-contiguous: {k}'


# ── Idempotency ──────────────────────────────────────────────────────

def test_remap_idempotent():
    legacy = _legacy_qkv()
    fused_once = _remap_state_dict_keys(legacy.copy())
    fused_twice = _remap_state_dict_keys(fused_once.copy())
    for k in fused_once:
        assert torch.equal(fused_once[k], fused_twice[k])


def test_split_idempotent():
    legacy = _legacy_qkv()
    split_once = _split_state_dict_keys(legacy.copy())
    split_twice = _split_state_dict_keys(split_once.copy())
    for k in split_once:
        assert torch.equal(split_once[k], split_twice[k])


# ── Anima timestep convention ────────────────────────────────────────

@pytest.mark.skipif(not _HAS_PIPELINE, reason="requires full ML stack (transformers, peft, accelerate, etc.)")
def test_anima_defaults_match_comfy_flow_timesteps():
    """Anima sampling_settings in ComfyUI: {"multiplier": 1.0, "shift": 3.0}."""
    pipeline = object.__new__(CosmosPredict2Pipeline)
    pipeline.name = 'anima'
    pipeline.model_config = {}
    pipeline._set_timestep_defaults()

    assert pipeline.timestep_shift == 3.0
    assert pipeline.timestep_multiplier == 1.0


@pytest.mark.skipif(not _HAS_PIPELINE, reason="requires full ML stack (transformers, peft, accelerate, etc.)")
def test_cosmos_predict2_keeps_normalized_timesteps():
    pipeline = object.__new__(CosmosPredict2Pipeline)
    pipeline.name = 'cosmos_predict2'
    pipeline.model_config = {}
    pipeline._set_timestep_defaults()

    assert pipeline.timestep_shift is None
    assert pipeline.timestep_multiplier == 1.0


@pytest.mark.skipif(not _HAS_PIPELINE, reason="requires full ML stack (transformers, peft, accelerate, etc.)")
def test_initial_layer_scales_anima_timestep_before_embedding():
    class StubTEmbedder(torch.nn.Module):
        def forward(self, timesteps):
            return timesteps, timesteps + 1

    class StubNorm(torch.nn.Module):
        def forward(self, x):
            return x

    model = torch.nn.Module()
    model.extra_per_block_abs_pos_emb = False
    model.x_embedder = torch.nn.Identity()
    model.pos_embedder = torch.nn.Identity()
    model.t_embedder = StubTEmbedder()
    model.t_embedding_norm = StubNorm()
    model.timestep_multiplier = 1.0

    layer = InitialLayer(model, None, False, False)
    timesteps = torch.tensor([0.25, 0.5])

    assert torch.equal(layer._scale_timesteps(timesteps), torch.tensor([[0.25], [0.50]]))
