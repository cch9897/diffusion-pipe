# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os.path
import re

import torch
from torch import nn
import torch.nn.functional as F
import safetensors
import transformers
from transformers import T5TokenizerFast, T5EncoderModel, AutoTokenizer, AutoModelForCausalLM
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device

from models.base import BasePipeline, PreprocessMediaFile, make_contiguous
from models.cosmos_predict2_modeling import MiniTrainDIT
from utils.common import load_state_dict, AUTOCAST_DTYPE, is_main_process, iterate_safetensors
from utils.offloading import ModelOffloader
from models.wan.vae2_1 import WanVAE_


KEEP_IN_HIGH_PRECISION = ['x_embedder', 't_embedder', 't_embedding_norm', 'final_layer']

MULTISCALE_LOSS_THRESHOLDS = [size * 0.9 for size in [1024]]
MULTISCALE_LOSS_THRESHOLDS.sort()


def _remap_state_dict_keys(state_dict):
    """Remap legacy checkpoint keys to the fused module layout used by MiniTrainDIT.

    Handles two fusions:
    1. QKV: self_attn q_proj/k_proj/v_proj (3 separate Linears) -> qkv_proj (single Linear).
       Weights are concatenated along output dim (dim 0): [q_w; k_w; v_w].
    2. AdaLN: three adaln_modulation_{self_attn,cross_attn,mlp} Sequentials -> two Linears
       (adaln_modulation_1, adaln_modulation_2). Layer-1 weights are concatenated along dim 0;
       layer-2 weights are placed block-diagonally (each branch keeps its own bottleneck slice).

    Idempotent: if the state_dict already uses fused keys, it is returned unchanged.
    """
    new_sd = {}
    # Collect keys we need to merge so we can skip the individual ones.
    skip_keys = set()

    # --- QKV fusion ---
    # Find all blocks.N.self_attn.q_proj.weight and merge with k_proj/v_proj
    qkv_pattern_re = re.compile(r'^(.+\.self_attn)\.q_proj\.weight$')
    for k in list(state_dict.keys()):
        m = qkv_pattern_re.match(k)
        if not m:
            continue
        prefix = m.group(1)
        qkv_key = f'{prefix}.qkv_proj.weight'
        if qkv_key in state_dict:
            # Already fused in checkpoint
            continue
        q_w = state_dict[f'{prefix}.q_proj.weight']
        k_w = state_dict[f'{prefix}.k_proj.weight']
        v_w = state_dict[f'{prefix}.v_proj.weight']
        new_sd[qkv_key] = torch.cat([q_w, k_w, v_w], dim=0)
        skip_keys.update({f'{prefix}.q_proj.weight', f'{prefix}.k_proj.weight', f'{prefix}.v_proj.weight'})

    # --- Cross-attn KV fusion ---
    # Find all blocks.N.cross_attn.k_proj.weight and merge with v_proj
    kv_pattern_re = re.compile(r'^(.+\.cross_attn)\.k_proj\.weight$')
    for k in list(state_dict.keys()):
        m = kv_pattern_re.match(k)
        if not m:
            continue
        prefix = m.group(1)
        kv_key = f'{prefix}.kv_proj.weight'
        if kv_key in state_dict:
            continue
        k_w = state_dict[f'{prefix}.k_proj.weight']
        v_w = state_dict[f'{prefix}.v_proj.weight']
        new_sd[kv_key] = torch.cat([k_w, v_w], dim=0)
        skip_keys.update({f'{prefix}.k_proj.weight', f'{prefix}.v_proj.weight'})

    # --- AdaLN fusion ---
    adaln_branches = ['self_attn', 'cross_attn', 'mlp']
    adaln_pattern_re = re.compile(r'^(.+)\.adaln_modulation_self_attn\.1\.weight$')
    for k in list(state_dict.keys()):
        m = adaln_pattern_re.match(k)
        if not m:
            continue
        prefix = m.group(1)
        mod1_key = f'{prefix}.adaln_modulation_1.weight'
        mod2_key = f'{prefix}.adaln_modulation_2.weight'
        if mod1_key in state_dict:
            # Already fused
            continue
        # use_adaln_lora=True path: two layers (1 and 2)
        layer1_keys = [f'{prefix}.adaln_modulation_{b}.1.weight' for b in adaln_branches]
        layer2_keys = [f'{prefix}.adaln_modulation_{b}.2.weight' for b in adaln_branches]
        if all(lk in state_dict for lk in layer1_keys + layer2_keys):
            # Layer 1: cat along dim 0 -> (3*adaln_lora_dim, D)
            new_sd[mod1_key] = torch.cat([state_dict[lk] for lk in layer1_keys], dim=0)
            # Layer 2: block-diagonal -> (9*D, 3*adaln_lora_dim)
            w2_list = [state_dict[lk] for lk in layer2_keys]  # each (3*D, adaln_lora_dim)
            D = w2_list[0].shape[0] // 3
            adaln_lora_dim = w2_list[0].shape[1]
            total_out = 9 * D
            total_mid = 3 * adaln_lora_dim
            w2 = torch.zeros(total_out, total_mid, dtype=w2_list[0].dtype, device=w2_list[0].device)
            for i, w in enumerate(w2_list):
                w2[i*3*D:(i+1)*3*D, i*adaln_lora_dim:(i+1)*adaln_lora_dim] = w
            new_sd[mod2_key] = w2
            for lk in layer1_keys + layer2_keys:
                skip_keys.add(lk)
        else:
            # use_adaln_lora=False path: single layer (1) -> adaln_modulation (9*D, D)
            single_keys = [f'{prefix}.adaln_modulation_{b}.1.weight' for b in adaln_branches]
            if all(sk in state_dict for sk in single_keys):
                new_sd[f'{prefix}.adaln_modulation.weight'] = torch.cat([state_dict[sk] for sk in single_keys], dim=0)
                for sk in single_keys:
                    skip_keys.add(sk)

    # Copy remaining keys
    for k, v in state_dict.items():
        if k not in skip_keys:
            new_sd.setdefault(k, v)

    return new_sd


def _split_state_dict_keys(state_dict):
    """Inverse of _remap_state_dict_keys: split fused keys back to legacy layout.

    Used by save_model/save_adapter so checkpoints stay compatible with ComfyUI
    (which uses the original 3-projection / 3-modulation layout).
    """
    new_sd = {}
    skip_keys = set()

    # --- QKV split ---
    qkv_pattern_re = re.compile(r'^(.+\.self_attn)\.qkv_proj\.weight$')
    for k in list(state_dict.keys()):
        m = qkv_pattern_re.match(k)
        if not m:
            continue
        prefix = m.group(1)
        qkv_w = state_dict[k]
        inner_dim = qkv_w.shape[0] // 3
        new_sd[f'{prefix}.q_proj.weight'] = qkv_w[:inner_dim]
        new_sd[f'{prefix}.k_proj.weight'] = qkv_w[inner_dim:2*inner_dim]
        new_sd[f'{prefix}.v_proj.weight'] = qkv_w[2*inner_dim:]
        skip_keys.add(k)

    # --- Cross-attn KV split ---
    kv_pattern_re = re.compile(r'^(.+\.cross_attn)\.kv_proj\.weight$')
    for k in list(state_dict.keys()):
        m = kv_pattern_re.match(k)
        if not m:
            continue
        prefix = m.group(1)
        kv_w = state_dict[k]
        inner_dim = kv_w.shape[0] // 2
        new_sd[f'{prefix}.k_proj.weight'] = kv_w[:inner_dim]
        new_sd[f'{prefix}.v_proj.weight'] = kv_w[inner_dim:]
        skip_keys.add(k)

    # --- AdaLN split ---
    adaln1_pattern_re = re.compile(r'^(.+)\.adaln_modulation_1\.weight$')
    adaln_pattern_re = re.compile(r'^(.+)\.adaln_modulation\.weight$')
    for k in list(state_dict.keys()):
        m = adaln1_pattern_re.match(k)
        if m:
            prefix = m.group(1)
            mod1_w = state_dict[k]  # (3*adaln_lora_dim, D)
            mod2_w = state_dict[f'{prefix}.adaln_modulation_2.weight']  # (9*D, 3*adaln_lora_dim)
            adaln_lora_dim = mod1_w.shape[0] // 3
            D = mod2_w.shape[0] // 9
            for i, b in enumerate(['self_attn', 'cross_attn', 'mlp']):
                new_sd[f'{prefix}.adaln_modulation_{b}.1.weight'] = mod1_w[i*adaln_lora_dim:(i+1)*adaln_lora_dim]
                new_sd[f'{prefix}.adaln_modulation_{b}.2.weight'] = mod2_w[i*3*D:(i+1)*3*D, i*adaln_lora_dim:(i+1)*adaln_lora_dim]
            skip_keys.add(k)
            skip_keys.add(f'{prefix}.adaln_modulation_2.weight')
            continue
        m = adaln_pattern_re.match(k)
        if m:
            prefix = m.group(1)
            mod_w = state_dict[k]  # (9*D, D)
            D = mod_w.shape[1]
            for i, b in enumerate(['self_attn', 'cross_attn', 'mlp']):
                new_sd[f'{prefix}.adaln_modulation_{b}.1.weight'] = mod_w[i*3*D:(i+1)*3*D]
            skip_keys.add(k)

    for k, v in state_dict.items():
        if k not in skip_keys:
            new_sd.setdefault(k, v)

    return new_sd


def _remap_lora_keys(state_dict):
    """Remap legacy LoRA adapter keys to the fused module layout.

    Handles two key formats:
    - PEFT state dict: ...self_attn.q_proj.lora_A.weight (no adapter name)
    - After load_adapter_weights transform: ...self_attn.q_proj.lora_A.default.weight

    QKV: q_proj/k_proj/v_proj -> qkv_proj. The old 3 LoRAs each have rank R;
    the fused qkv_proj LoRA also has rank R. lora_B cats naturally (3×(D,R) -> (3D,R)),
    but lora_A (3×(R,D) -> (R,D)) requires averaging — this is lossy but the best
    rank-R approximation. A warning is printed.

    AdaLN: adaln_modulation_{self,cross,mlp}.{1,2} -> adaln_modulation_{1,2}.
    Layer-1 lora_A cats along dim 0 (3×(R,D) -> (3R,D)) and layer-2 lora_B is
    block-diagonal (3×(3D,R) -> (9D,3R)) — this increases the fused rank to 3R,
    which requires the PEFT config to use rank 3R for these modules. Since
    adaln modulation typically uses a small rank, this is handled by padding
    lora_A to match: if the target rank < 3R, we average; if equal, we cat.
    """
    import warnings
    new_sd = {}
    skip_keys = set()
    
    _SUFFIX = r'(?:\.([^.]+))?\.weight$'

    # --- QKV LoRA remap ---
    qkv_lora_re = re.compile(r'^(.+\.self_attn)\.q_proj\.(lora_[AB])' + _SUFFIX)
    for k in list(state_dict.keys()):
        m = qkv_lora_re.match(k)
        if not m:
            continue
        prefix = m.group(1)
        lora_type = m.group(2)
        adapter_name = m.group(3)
        mid = f'.{adapter_name}' if adapter_name else ''
        fused_key = f'{prefix}.qkv_proj.{lora_type}{mid}.weight'
        if fused_key in state_dict or fused_key in new_sd:
            continue
        q_key = f'{prefix}.q_proj.{lora_type}{mid}.weight'
        k_key = f'{prefix}.k_proj.{lora_type}{mid}.weight'
        v_key = f'{prefix}.v_proj.{lora_type}{mid}.weight'
        if not all(kk in state_dict for kk in [q_key, k_key, v_key]):
            continue
        q_w, k_w, v_w = state_dict[q_key], state_dict[k_key], state_dict[v_key]
        if lora_type == 'lora_A':
            # Old: 3 × (R, D). New: (R, D). Average to preserve rank-R capacity.
            warnings.warn(
                f'Remapping legacy QKV LoRA: averaging 3 lora_A tensors for {prefix}. '
                f'This is lossy (rank 3R -> R). Consider retraining with fused qkv_proj.'
            )
            new_sd[fused_key] = (q_w + k_w + v_w) / 3.0
        else:
            # lora_B: Old 3 × (D, R). New: (3D, R). Cat along dim 0.
            new_sd[fused_key] = torch.cat([q_w, k_w, v_w], dim=0)
        skip_keys.update({q_key, k_key, v_key})

    # --- Cross-attn KV LoRA remap ---
    kv_lora_re = re.compile(r'^(.+\.cross_attn)\.k_proj\.(lora_[AB])' + _SUFFIX)
    for k in list(state_dict.keys()):
        m = kv_lora_re.match(k)
        if not m:
            continue
        prefix = m.group(1)
        lora_type = m.group(2)
        adapter_name = m.group(3)
        mid = f'.{adapter_name}' if adapter_name else ''
        fused_key = f'{prefix}.kv_proj.{lora_type}{mid}.weight'
        if fused_key in state_dict or fused_key in new_sd:
            continue
        k_key = f'{prefix}.k_proj.{lora_type}{mid}.weight'
        v_key = f'{prefix}.v_proj.{lora_type}{mid}.weight'
        if not all(kk in state_dict for kk in [k_key, v_key]):
            continue
        k_w, v_w = state_dict[k_key], state_dict[v_key]
        if lora_type == 'lora_A':
            # Old: 2 × (R, D). New: (R, D). Average (lossy — 2R → R).
            import warnings
            warnings.warn(
                f'Remapping legacy KV LoRA: averaging 2 lora_A tensors for {prefix}. '
                f'This is lossy (rank 2R → R). Consider retraining with fused kv_proj.'
            )
            new_sd[fused_key] = (k_w + v_w) / 2.0
        else:
            # lora_B: Old 2 × (D, R). New: (2D, R). Cat along dim 0.
            new_sd[fused_key] = torch.cat([k_w, v_w], dim=0)
        skip_keys.update({k_key, v_key})

    # --- AdaLN LoRA remap ---
    adaln_branches = ['self_attn', 'cross_attn', 'mlp']
    adaln_lora_re = re.compile(r'^(.+)\.adaln_modulation_self_attn\.1\.(lora_[AB])' + _SUFFIX)
    for k in list(state_dict.keys()):
        m = adaln_lora_re.match(k)
        if not m:
            continue
        prefix = m.group(1)
        lora_type = m.group(2)
        adapter_name = m.group(3)
        mid = f'.{adapter_name}' if adapter_name else ''
        fused_key_1 = f'{prefix}.adaln_modulation_1.{lora_type}{mid}.weight'
        fused_key_2 = f'{prefix}.adaln_modulation_2.{lora_type}{mid}.weight'
        if fused_key_1 in state_dict or fused_key_1 in new_sd:
            continue
        layer1_keys = [f'{prefix}.adaln_modulation_{b}.1.{lora_type}{mid}.weight' for b in adaln_branches]
        layer2_keys = [f'{prefix}.adaln_modulation_{b}.2.{lora_type}{mid}.weight' for b in adaln_branches]
        if all(kk in state_dict for kk in layer1_keys + layer2_keys):
            l1_list = [state_dict[kk] for kk in layer1_keys]
            l2_list = [state_dict[kk] for kk in layer2_keys]
            if lora_type == 'lora_A':
                # Old layer1: 3 × (R, D). New: (R, D). Average (lossy — 3R → R).
                # Old layer2: 3 × (R, adaln_lora_dim). New: (R, 3*adaln_lora_dim). Cat dim 1 (non-lossy).
                import warnings
                adaln_lora_dim = l2_list[0].shape[1]
                if adaln_lora_dim % 3 != 0:
                    warnings.warn(
                        f'Remapping legacy AdaLN LoRA for {prefix}: '
                        f'adaln_lora_dim={adaln_lora_dim} not divisible by 3. '
                        f'Layer-2 lora_A cat may produce unexpected shapes.'
                    )
                warnings.warn(
                    f'Remapping legacy AdaLN LoRA for {prefix}: '
                    f'layer-1 lora_A averaged (lossy, 3R→R); '
                    f'layer-2 lora_A concatenated dim 1 (non-lossy). '
                    f'Consider retraining with fused adaln_modulation for full fidelity.'
                )
                new_sd[fused_key_1] = sum(l1_list) / 3.0
                new_sd[fused_key_2] = torch.cat(l2_list, dim=1)
            else:
                # lora_B
                # Old layer1: 3 × (adaln_lora_dim, R). New: (3*adaln_lora_dim, R). Cat dim 0.
                new_sd[fused_key_1] = torch.cat(l1_list, dim=0)
                # Old layer2: 3 × (3D, R). New: (9D, R). Cat dim 0.
                new_sd[fused_key_2] = torch.cat(l2_list, dim=0)
            skip_keys.update(layer1_keys + layer2_keys)

    for k, v in state_dict.items():
        if k not in skip_keys:
            new_sd.setdefault(k, v)
    return new_sd


def _split_lora_keys(state_dict):
    """Inverse of _remap_lora_keys: split fused LoRA keys back to legacy layout.

    Used by save_adapter so saved LoRA files stay compatible with ComfyUI's
    3-projection / 3-modulation layout.

    QKV lora_A (R, D) is replicated 3× (each projection gets the same lora_A).
    QKV lora_B (3D, R) is sliced into 3 × (D, R).
    AdaLN lora_A is replicated 3×; lora_B is sliced/cat'd into 3 pieces.
    """
    new_sd = {}
    skip_keys = set()
    
    _SUFFIX = r'(?:\.([^.]+))?\.weight$'

    # --- QKV LoRA split ---
    qkv_lora_re = re.compile(r'^(.+\.self_attn)\.qkv_proj\.(lora_[AB])' + _SUFFIX)
    for k in list(state_dict.keys()):
        m = qkv_lora_re.match(k)
        if not m:
            continue
        prefix = m.group(1)
        lora_type = m.group(2)
        adapter_name = m.group(3)
        mid = f'.{adapter_name}' if adapter_name else ''
        qkv_w = state_dict[k]
        if lora_type == 'lora_A':
            # (R, D) -> replicate to 3 × (R, D)
            for proj in ['q_proj', 'k_proj', 'v_proj']:
                new_sd[f'{prefix}.{proj}.{lora_type}{mid}.weight'] = qkv_w.clone()
        else:
            # (3D, R) -> 3 × (D, R)
            out_dim = qkv_w.shape[0] // 3
            for i, proj in enumerate(['q_proj', 'k_proj', 'v_proj']):
                new_sd[f'{prefix}.{proj}.{lora_type}{mid}.weight'] = qkv_w[i*out_dim:(i+1)*out_dim]
        skip_keys.add(k)

    # --- Cross-attn KV LoRA split ---
    kv_lora_re = re.compile(r'^(.+\.cross_attn)\.kv_proj\.(lora_[AB])' + _SUFFIX)
    for k in list(state_dict.keys()):
        m = kv_lora_re.match(k)
        if not m:
            continue
        prefix = m.group(1)
        lora_type = m.group(2)
        adapter_name = m.group(3)
        mid = f'.{adapter_name}' if adapter_name else ''
        kv_w = state_dict[k]
        if lora_type == 'lora_A':
            # (R, D) -> replicate to 2 × (R, D)
            for proj in ['k_proj', 'v_proj']:
                new_sd[f'{prefix}.{proj}.{lora_type}{mid}.weight'] = kv_w.clone()
        else:
            # (2D, R) -> 2 × (D, R)
            out_dim = kv_w.shape[0] // 2
            for i, proj in enumerate(['k_proj', 'v_proj']):
                new_sd[f'{prefix}.{proj}.{lora_type}{mid}.weight'] = kv_w[i*out_dim:(i+1)*out_dim]
        skip_keys.add(k)

    # --- AdaLN LoRA split ---
    adaln_branches = ['self_attn', 'cross_attn', 'mlp']
    adaln1_lora_re = re.compile(r'^(.+)\.adaln_modulation_1\.(lora_[AB])' + _SUFFIX)
    for k in list(state_dict.keys()):
        m = adaln1_lora_re.match(k)
        if not m:
            continue
        prefix = m.group(1)
        lora_type = m.group(2)
        adapter_name = m.group(3)
        mid = f'.{adapter_name}' if adapter_name else ''
        mod1_w = state_dict[k]
        mod2_w = state_dict.get(f'{prefix}.adaln_modulation_2.{lora_type}{mid}.weight')
        if mod2_w is None:
            continue
        if lora_type == 'lora_A':
            # mod1: (R, D) -> replicate to 3 × (R, D)  (was averaged, all branches identical)
            # mod2: (R, 3*adaln_lora_dim) -> slice dim 1 to 3 × (R, adaln_lora_dim)
            ald = mod2_w.shape[1] // 3
            for i, b in enumerate(adaln_branches):
                new_sd[f'{prefix}.adaln_modulation_{b}.1.{lora_type}{mid}.weight'] = mod1_w.clone()
                new_sd[f'{prefix}.adaln_modulation_{b}.2.{lora_type}{mid}.weight'] = mod2_w[:, i*ald:(i+1)*ald].clone()
        else:
            # lora_B
            # mod1: (3*adaln_lora_dim, R) -> 3 × (adaln_lora_dim, R)
            mid_dim = mod1_w.shape[0] // 3
            for i, b in enumerate(adaln_branches):
                new_sd[f'{prefix}.adaln_modulation_{b}.1.{lora_type}{mid}.weight'] = mod1_w[i*mid_dim:(i+1)*mid_dim]
            # mod2: (9*D, R) -> 3 × (3*D, R)
            D = mod2_w.shape[0] // 9
            for i, b in enumerate(adaln_branches):
                new_sd[f'{prefix}.adaln_modulation_{b}.2.{lora_type}{mid}.weight'] = mod2_w[i*3*D:(i+1)*3*D]
        skip_keys.add(k)
        skip_keys.add(f'{prefix}.adaln_modulation_2.{lora_type}{mid}.weight')

    for k, v in state_dict.items():
        if k not in skip_keys:
            new_sd.setdefault(k, v)
    return new_sd


def time_shift(mu: float, sigma: float, t: torch.Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15):
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def _video_vae(pretrained_path=None, z_dim=None, device='cpu', **kwargs):
    """
    Autoencoder3d adapted from Stable Diffusion 1.x, 2.x and XL.
    """
    # params
    cfg = dict(
        dim=96,
        z_dim=z_dim,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        dropout=0.0)
    cfg.update(**kwargs)

    # init model
    with torch.device('meta'):
        model = WanVAE_(**cfg)

    # load checkpoint
    model.load_state_dict(
        load_state_dict(pretrained_path), assign=True)

    return model


class WanVAE:
    def __init__(self,
                 z_dim=16,
                 vae_pth=None,
                 dtype=torch.float,
                 device="cpu"):
        self.dtype = dtype
        self.device = device

        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        self.mean = torch.tensor(mean, dtype=dtype, device=device)
        self.std = torch.tensor(std, dtype=dtype, device=device)
        self.scale = [self.mean, 1.0 / self.std]

        # init model
        self.model = _video_vae(
            pretrained_path=vae_pth,
            z_dim=z_dim,
        ).eval().requires_grad_(False).to(device)


def vae_encode(tensor, vae):
    return vae.model.encode(tensor, vae.scale)


def get_dit_config(state_dict, key_prefix=''):
    dit_config = {}
    dit_config["max_img_h"] = 1024
    dit_config["max_img_w"] = 1024
    dit_config["max_frames"] = 128
    concat_padding_mask = True
    dit_config["in_channels"] = (state_dict['{}x_embedder.proj.1.weight'.format(key_prefix)].shape[1] // 4) - int(concat_padding_mask)
    dit_config["out_channels"] = 16
    dit_config["patch_spatial"] = 2
    dit_config["patch_temporal"] = 1
    dit_config["model_channels"] = state_dict['{}x_embedder.proj.1.weight'.format(key_prefix)].shape[0]
    dit_config["concat_padding_mask"] = concat_padding_mask
    dit_config["crossattn_emb_channels"] = 1024
    dit_config["pos_emb_cls"] = "rope3d"
    dit_config["pos_emb_learnable"] = True
    dit_config["pos_emb_interpolation"] = "crop"
    dit_config["min_fps"] = 1
    dit_config["max_fps"] = 30

    dit_config["use_adaln_lora"] = True
    dit_config["adaln_lora_dim"] = 256
    if dit_config["model_channels"] == 2048:
        dit_config["num_blocks"] = 28
        dit_config["num_heads"] = 16
    elif dit_config["model_channels"] == 5120:
        dit_config["num_blocks"] = 36
        dit_config["num_heads"] = 40
    elif dit_config["model_channels"] == 1280:
        dit_config["num_blocks"] = 20
        dit_config["num_heads"] = 20

    if dit_config["in_channels"] == 16:
        dit_config["extra_per_block_abs_pos_emb"] = False
        dit_config["rope_h_extrapolation_ratio"] = 4.0
        dit_config["rope_w_extrapolation_ratio"] = 4.0
        dit_config["rope_t_extrapolation_ratio"] = 1.0
    elif dit_config["in_channels"] == 17:
        dit_config["extra_per_block_abs_pos_emb"] = False
        dit_config["rope_h_extrapolation_ratio"] = 3.0
        dit_config["rope_w_extrapolation_ratio"] = 3.0
        dit_config["rope_t_extrapolation_ratio"] = 1.0

    dit_config["extra_h_extrapolation_ratio"] = 1.0
    dit_config["extra_w_extrapolation_ratio"] = 1.0
    dit_config["extra_t_extrapolation_ratio"] = 1.0
    dit_config["rope_enable_fps_modulation"] = False

    return dit_config


def _tokenize(tokenizer, prompts):
    return tokenizer(
        prompts,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=512,
    )

def _compute_text_embeddings(text_encoder, input_ids, attn_mask, is_generic_llm=False):
    input_ids = input_ids.to(text_encoder.device)
    attn_mask = attn_mask.to(text_encoder.device)

    outputs = text_encoder(input_ids=input_ids, attention_mask=attn_mask)
    encoded_text = outputs.last_hidden_state
    encoded_text[~attn_mask.bool()] = 0

    return encoded_text


class CosmosPredict2Pipeline(BasePipeline):
    name = 'cosmos_predict2'
    framerate = 16
    checkpointable_layers = ['TransformerLayer']
    adapter_target_modules = [
        'Block',
        'TransformerBlock',  # LLM adapter
    ]

    def __init__(self, config):
        self.config = config
        self.model_config = self.config['model']
        self.offloader = ModelOffloader('dummy', [], 0, 0, True, torch.device('cuda'), False, debug=False)
        dtype = self.model_config['dtype']
        self.cache_text_embeddings = self.model_config.get('cache_text_embeddings', True)
        self.multiscale_loss_weight = self.model_config.get('multiscale_loss_weight', None)

        # This isn't a nn.Module.
        self.vae = WanVAE(
            vae_pth=self.model_config['vae_path'],
            device='cpu',
            dtype=dtype,
        )
        # These need to be on the device the VAE will be moved to during caching.
        self.vae.mean = self.vae.mean.to('cuda')
        self.vae.std = self.vae.std.to('cuda')
        self.vae.scale = [self.vae.mean, 1.0 / self.vae.std]

        self.is_generic_llm = False
        self.t5_tokenizer = T5TokenizerFast(
            vocab_file='configs/t5_old/spiece.model',
            tokenizer_file='configs/t5_old/tokenizer.json',
        )

        if 't5_path' in self.model_config:
            self.tokenizer = self.t5_tokenizer
            t5_state_dict = load_state_dict(self.model_config['t5_path'])
            if self.model_config.get('text_encoder_nf4', False):
                quantization_config = transformers.BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type='nf4',
                    bnb_4bit_compute_dtype=dtype,
                )
            else:
                quantization_config = None
            self.text_encoder = T5EncoderModel.from_pretrained(
                None,
                config='configs/t5_old/config.json',
                state_dict=t5_state_dict,
                torch_dtype='auto',
                local_files_only=True,
                quantization_config=quantization_config,
            )
            if quantization_config is None and self.model_config.get('text_encoder_fp8', False):
                for name, p in self.text_encoder.named_parameters():
                    if p.ndim == 2 and not ('shared' in name or 'relative_attention_bias' in name):
                        p.data = p.data.to(torch.float8_e4m3fn)
        elif 'llm_path' in self.model_config:
            llm_path = self.model_config['llm_path']
            if os.path.isdir(llm_path):
                # generic Transformers LLM
                self.tokenizer = AutoTokenizer.from_pretrained(llm_path, local_files_only=True)
                text_encoder = AutoModelForCausalLM.from_pretrained(llm_path, dtype=dtype, local_files_only=True)
            else:
                # assume Qwen3-0.6b (Anima)
                self.tokenizer = AutoTokenizer.from_pretrained('configs/qwen3_06b', local_files_only=True)
                llm_config = transformers.Qwen3Config.from_pretrained('configs/qwen3_06b', local_files_only=True)
                with init_empty_weights():
                    text_encoder = transformers.Qwen3ForCausalLM(llm_config)
                for key, tensor in iterate_safetensors(llm_path):
                    set_module_tensor_to_device(text_encoder, key, device='cpu', dtype=dtype, value=tensor)
            self.text_encoder = text_encoder.model
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.text_encoder.config.use_cache = False
            self.is_generic_llm = True
            # text encoder is different from Cosmos, use a different cache dir
            self.name = 'anima'
        else:
            raise RuntimeError('Missing text encoder path')

        self.text_encoder.requires_grad_(False)

    def load_diffusion_model(self):
        dtype = self.model_config['dtype']
        transformer_dtype = self.model_config.get('transformer_dtype', dtype)

        state_dict = load_state_dict(self.model_config['transformer_path'])
        # Remove 'net.' prefix
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('net.'):
                k = k[len('net.'):]
            new_state_dict[k] = v
        state_dict = new_state_dict

        # Remap legacy checkpoint keys (q_proj/k_proj/v_proj -> qkv_proj,
        # adaln_modulation_{self,cross,mlp} -> adaln_modulation_1/2) to the fused layout.
        state_dict = _remap_state_dict_keys(state_dict)

        dit_config = get_dit_config(state_dict)
        dit_config['attention_backend'] = self.model_config.get('attention_backend')

        if 'llm_adapter_path' in self.model_config:
            self.use_llm_adapter = True
            dit_config['use_llm_adapter'] = True
            llm_adapter_state_dict = {
                k: v.to(dtype)
                for k, v in load_state_dict(self.model_config['llm_adapter_path']).items()
            }
        elif 'llm_adapter.out_proj.weight' in state_dict:
            self.use_llm_adapter = True
            dit_config['use_llm_adapter'] = True
            llm_adapter_state_dict = None  # llm_adapter gets loaded as part of the DiT
        else:
            self.use_llm_adapter = False

        with init_empty_weights():
            transformer = MiniTrainDIT(**dit_config)
            for name, p in transformer.named_parameters():
                if name not in state_dict:
                    continue
                dtype_to_use = dtype if (any(keyword in name for keyword in KEEP_IN_HIGH_PRECISION) or p.ndim == 1) else transformer_dtype
                set_module_tensor_to_device(transformer, name, device='cpu', dtype=dtype_to_use, value=state_dict[name])

        if self.use_llm_adapter and llm_adapter_state_dict is not None:
            llm_adapter = transformer.llm_adapter
            for name, p in llm_adapter.named_parameters():
                dtype_to_use = dtype if (any(keyword in name for keyword in KEEP_IN_HIGH_PRECISION) or p.ndim == 1) else transformer_dtype
                set_module_tensor_to_device(llm_adapter, name, device='cpu', dtype=dtype_to_use, value=llm_adapter_state_dict[name])

        self.transformer = transformer
        self.transformer.train()
        for name, p in self.transformer.named_parameters():
            p.original_name = name

    def get_vae(self):
        return self.vae.model

    def get_text_encoders(self):
        if self.cache_text_embeddings:
            return [self.text_encoder]
        else:
            return []

    def save_adapter(self, save_dir, peft_state_dict):
        self.peft_config.save_pretrained(save_dir)
        # Split fused LoRA keys back to legacy layout for ComfyUI compatibility.
        peft_state_dict = _split_lora_keys(peft_state_dict)
        # Make tensors contiguous — split slices may be non-contiguous views,
        # which breaks safetensors' tensor.view(-1) in _find_shared_tensors.
        peft_state_dict = {k: v.contiguous() for k, v in peft_state_dict.items()}
        # ComfyUI format.
        peft_state_dict = {'diffusion_model.'+k: v for k, v in peft_state_dict.items()}
        safetensors.torch.save_file(peft_state_dict, save_dir / 'adapter_model.safetensors', metadata={'format': 'pt'})

    def save_model(self, save_dir, state_dict):
        # Split fused keys back to legacy layout for ComfyUI compatibility.
        state_dict = _split_state_dict_keys(state_dict)
        # Make tensors contiguous — split slices may be non-contiguous views,
        # which breaks safetensors' tensor.view(-1) in _find_shared_tensors.
        state_dict = {k: v.contiguous() for k, v in state_dict.items()}
        state_dict = {'net.'+k: v for k, v in state_dict.items()}
        safetensors.torch.save_file(state_dict, save_dir / 'model.safetensors', metadata={'format': 'pt'})

    def _remap_adapter_state_dict(self, state_dict):
        """Remap legacy LoRA keys (q_proj/k_proj/v_proj -> qkv_proj,
        adaln_modulation_{self,cross,mlp} -> adaln_modulation_1/2) to the fused layout."""
        return _remap_lora_keys(state_dict)
    def remap_checkpoint_state_dict(self, layer_state_dict):
        """Remap legacy checkpoint keys (q_proj/k_proj/v_proj -> qkv_proj,
        adaln_modulation_{self,cross,mlp} -> adaln_modulation_1/2) to the fused layout."""
        return _remap_state_dict_keys(layer_state_dict)

    def get_preprocess_media_file_fn(self):
        return PreprocessMediaFile(
            self.config,
            support_video=True,
            framerate=self.framerate,
        )

    def get_call_vae_fn(self, vae):
        def fn(tensor):
            p = next(vae.parameters())
            tensor = tensor.to(p.device, p.dtype)
            latents = vae_encode(tensor, self.vae)
            return {'latents': latents}
        return fn

    def get_call_text_encoder_fn(self, text_encoder):
        def fn(captions, is_video):
            # args are lists
            batch_encoding = _tokenize(self.tokenizer, captions)
            t5_batch_encoding = _tokenize(self.t5_tokenizer, captions)
            encoded_text = _compute_text_embeddings(self.text_encoder, batch_encoding.input_ids, batch_encoding.attention_mask)
            return {'prompt_embeds': encoded_text, 'attn_mask': batch_encoding.attention_mask, 't5_input_ids': t5_batch_encoding.input_ids, 't5_attn_mask': t5_batch_encoding.attention_mask}
        return fn

    # Note to myself / future readers:
    # The timestep sampling, input construction, and loss function have a different formulation here than how Nvidia does it
    # in the official code. It wasn't obvious at first, but if you work through the math you will see the this model is just
    # a standard rectified flow model, the same as Flux, SD3, Lumina 2, etc. The ONLY difference is that in the way Nvidia
    # formulated it, you end up with an effective loss weighting of t**2 + (1-t)**2. This is a quadratic that is 1 at the endpoints
    # t=0 and t=1, and 0.5 at t=0.5. So, the middle timesteps are downweighted slightly. I left out this weighting because I don't
    # see any justification or point to doing it. As such, everything here aligns with the other rectified flow models.
    def prepare_inputs(self, inputs, timestep_quantile=None):
        latents = inputs['latents'].float()
        mask = inputs['mask']

        if self.cache_text_embeddings:
            prompt_embeds_or_batch_encoding = (inputs['prompt_embeds'], inputs['attn_mask'], inputs['t5_input_ids'], inputs['t5_attn_mask'])
        else:
            captions = inputs['caption']
            batch_encoding = _tokenize(self.tokenizer, captions)
            t5_batch_encoding = _tokenize(self.t5_tokenizer, captions)
            prompt_embeds_or_batch_encoding = (batch_encoding.input_ids, batch_encoding.attention_mask, t5_batch_encoding.input_ids, t5_batch_encoding.attention_mask)

        bs, channels, num_frames, h, w = latents.shape

        if mask is not None:
            mask = mask.unsqueeze(1)  # make mask (bs, 1, img_h, img_w)
            mask = F.interpolate(mask, size=(h, w), mode='nearest-exact')  # resize to latent spatial dimension
            mask = mask.unsqueeze(2)  # make mask same number of dims as target

        timestep_sample_method = self.model_config.get('timestep_sample_method', 'logit_normal')

        if timestep_sample_method == 'logit_normal':
            dist = torch.distributions.normal.Normal(0, 1)
        elif timestep_sample_method == 'uniform':
            dist = torch.distributions.uniform.Uniform(0, 1)
        else:
            raise NotImplementedError()

        if timestep_quantile is not None:
            t = dist.icdf(torch.full((bs,), timestep_quantile, device=latents.device))
        else:
            t = dist.sample((bs,)).to(latents.device)

        if timestep_sample_method == 'logit_normal':
            sigmoid_scale = self.model_config.get('sigmoid_scale', 1.0)
            t = t * sigmoid_scale
            t = torch.sigmoid(t)

        if shift := self.model_config.get('shift', None):
            t = (t * shift) / (1 + (shift - 1) * t)
        elif self.model_config.get('flux_shift', False):
            mu = get_lin_function(y1=0.5, y2=1.15)((h // 2) * (w // 2))
            t = time_shift(mu, 1.0, t)

        noise = torch.randn_like(latents)
        t_expanded = t.view(-1, 1, 1, 1, 1)
        noisy_latents = (1 - t_expanded)*latents + t_expanded*noise
        target = noise - latents
        t = t.view(-1, 1)

        return (noisy_latents, t, *prompt_embeds_or_batch_encoding), (target, mask)

    def to_layers(self):
        transformer = self.transformer
        text_encoder = None if self.cache_text_embeddings else self.text_encoder

        # Opt-in per-block torch.compile. We compile the inner Block.forward (not the
        # outer TransformerLayer.forward) so that ModelOffloader.wait_for_block /
        # submit_move_blocks_forward stay outside the compiled graph. PEFT has already
        # injected LoRA modules by this point (configure_adapter runs before to_layers),
        # so they are part of the traced graph.
        if self.model_config.get('torch_compile', False):
            compile_mode = self.model_config.get('torch_compile_mode', 'default')
            compile_dynamic = self.model_config.get('torch_compile_dynamic', False)
            # activation_checkpointing='unsloth' wraps the block forward in
            # @torch._disable_dynamo, making compile a silent no-op.
            if self.config.get('activation_checkpointing') == 'unsloth':
                raise ValueError(
                    'torch_compile=true is incompatible with activation_checkpointing="unsloth". '
                    'unsloth wraps the block forward in @torch._disable_dynamo, so compile becomes a no-op. '
                    'Set activation_checkpointing=true (native, keeps activations in VRAM) or =false to use torch_compile.'
                )
            # blocks_to_swap mutates module.weight.data on every step, invalidating
            # dynamo guards and triggering recompiles on every step.
            if self.config.get('blocks_to_swap', 0):
                raise ValueError(
                    'torch_compile=true is incompatible with blocks_to_swap > 0. '
                    'blocks_to_swap mutates module.weight.data on every step, which invalidates '
                    'dynamo guards and triggers recompilation. Set blocks_to_swap=0 to use torch_compile.'
                )
            if is_main_process():
                print(f'Compiling transformer blocks with torch.compile (mode={compile_mode!r}, dynamic={compile_dynamic}, fullgraph=False)...')
                if self.model_config.get('transformer_dtype') == torch.float8_e4m3fn:
                    print('  warning: transformer_dtype=float8 has limited dynamo support; expect graph breaks.')
                if not compile_dynamic:
                    print('  note: dynamic=False — every new (block, shape) pair will recompile. Set torch_compile_dynamic=true if training on multiple size buckets.')
            for block in transformer.blocks:
                block.forward = torch.compile(
                    block.forward,
                    mode=compile_mode,
                    fullgraph=False,
                    dynamic=compile_dynamic,
                )

        layers = [
            InitialLayer(transformer, text_encoder, self.is_generic_llm),
            LLMAdapterLayer(transformer.llm_adapter if self.use_llm_adapter else None),
        ]
        for i, block in enumerate(transformer.blocks):
            layers.append(TransformerLayer(block, i, self.offloader))
        layers.append(FinalLayer(transformer))
        return layers

    def enable_block_swap(self, blocks_to_swap):
        transformer = self.transformer
        blocks = transformer.blocks
        num_blocks = len(blocks)
        assert (
            blocks_to_swap <= num_blocks - 2
        ), f'Cannot swap more than {num_blocks - 2} blocks. Requested {blocks_to_swap} blocks to swap.'
        self.offloader = ModelOffloader(
            'TransformerBlock', blocks, num_blocks, blocks_to_swap, True, torch.device('cuda'), self.config['reentrant_activation_checkpointing']
        )
        transformer.blocks = None
        transformer.to('cuda')
        transformer.blocks = blocks
        self.prepare_block_swap_training()
        print(f'Block swap enabled. Swapping {blocks_to_swap} blocks out of {num_blocks} blocks.')

    def prepare_block_swap_training(self):
        self.offloader.enable_block_swap()
        self.offloader.set_forward_only(False)
        self.offloader.prepare_block_devices_before_forward()

    def prepare_block_swap_inference(self, disable_block_swap=False):
        if disable_block_swap:
            self.offloader.disable_block_swap()
        self.offloader.set_forward_only(True)
        self.offloader.prepare_block_devices_before_forward()

    def get_param_groups(self, parameters):
        base_params, self_attn_params, cross_attn_params, mlp_params, mod_params, llm_adapter_params = [], [], [], [], [], []
        for p in parameters:
            name = getattr(p, 'original_name', '')
            if 'llm_adapter' in name:
                llm_adapter_params.append(p)
            elif '.self_attn' in name:
                self_attn_params.append(p)
            elif '.cross_attn' in name:
                cross_attn_params.append(p)
            elif '.mlp' in name:
                mlp_params.append(p)
            elif '.adaln_modulation' in name:
                mod_params.append(p)
            else:
                base_params.append(p)

        base_lr = self.config['optimizer'].get('lr', None)
        self_attn_lr = self.model_config.get('self_attn_lr', base_lr)
        cross_attn_lr = self.model_config.get('cross_attn_lr', base_lr)
        mlp_lr = self.model_config.get('mlp_lr', base_lr)
        mod_lr = self.model_config.get('mod_lr', base_lr)
        llm_adapter_lr = self.model_config.get('llm_adapter_lr', base_lr)

        if is_main_process():
            print(f'Using base_lr={base_lr}, self_attn_lr={self_attn_lr}, cross_attn_lr={cross_attn_lr}, mlp_lr={mlp_lr}, mod_lr={mod_lr}, llm_adapter_lr={llm_adapter_lr}')
            print(f'Num base params: {len(base_params)}')
            print(f'Num self_attn params: {len(self_attn_params)}')
            print(f'Num cross_attn params: {len(cross_attn_params)}')
            print(f'Num mlp params: {len(mlp_params)}')
            print(f'Num mod params: {len(mod_params)}')
            print(f'Num llm_adapter params: {len(llm_adapter_params)}')

        param_groups = []
        for lr, params in [(base_lr, base_params), (self_attn_lr, self_attn_params), (cross_attn_lr, cross_attn_params), (mlp_lr, mlp_params), (mod_lr, mod_params), (llm_adapter_lr, llm_adapter_params)]:
            if lr == 0:
                for p in params:
                    p.requires_grad_(False)
            elif len(params) > 0:
                param_groups.append({'params': params, 'lr': lr})

        return param_groups

    def get_loss_fn(self):
        def loss_fn(output, label):
            target, mask = label
            with torch.autocast('cuda', enabled=False):
                output = output.to(torch.float32)
                target = target.to(output.device, torch.float32)
                if 'huber_delta' in self.config:
                    loss = F.huber_loss(output, target, reduction='none', delta=self.config['huber_delta'])
                elif 'smooth_l1_beta' in self.config:
                    loss = F.smooth_l1_loss(output, target, reduction='none', beta=self.config['smooth_l1_beta'])
                else:
                    loss = F.mse_loss(output, target, reduction='none')
                # empty tensor means no masking
                if mask.numel() > 0:
                    mask = mask.to(output.device, torch.float32)
                    loss *= mask
                loss = loss.mean()

                if weight := self.multiscale_loss_weight:
                    assert output.ndim == 5 and target.ndim == 5
                    output = output.squeeze(2)
                    target = target.squeeze(2)
                    terms = [loss]
                    total_weight = 1.0
                    h, w = target.shape[-2:]
                    side_length = math.sqrt(h*w) * 8
                    for thresh in MULTISCALE_LOSS_THRESHOLDS:
                        if side_length >= thresh:
                            output = F.avg_pool2d(output, 2)
                            target = F.avg_pool2d(target, 2)
                            additional_loss = F.mse_loss(output, target) * weight
                            terms.append(additional_loss)
                            total_weight += weight
                        else:
                            break
                    loss = sum(terms) / total_weight

            return loss
        return loss_fn


class InitialLayer(nn.Module):
    def __init__(self, model, text_encoder, is_generic_llm):
        super().__init__()
        self.x_embedder = model.x_embedder
        self.pos_embedder = model.pos_embedder
        if model.extra_per_block_abs_pos_emb:
            self.extra_pos_embedder = model.extra_pos_embedder
        self.t_embedder = model.t_embedder
        self.t_embedding_norm = model.t_embedding_norm
        self.text_encoder = text_encoder
        self.model = [model]
        self.is_generic_llm = is_generic_llm

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        x_B_C_T_H_W, timesteps_B_T, *prompt_embeds_or_batch_encoding = inputs

        if torch.is_floating_point(prompt_embeds_or_batch_encoding[0]):
            crossattn_emb, attn_mask, t5_input_ids, t5_attn_mask = prompt_embeds_or_batch_encoding
        else:
            with torch.no_grad():
                input_ids, attn_mask, t5_input_ids, t5_attn_mask = prompt_embeds_or_batch_encoding
                crossattn_emb = _compute_text_embeddings(self.text_encoder, input_ids, attn_mask, is_generic_llm=self.is_generic_llm)

        padding_mask = torch.zeros(x_B_C_T_H_W.shape[0], 1, x_B_C_T_H_W.shape[3], x_B_C_T_H_W.shape[4], dtype=x_B_C_T_H_W.dtype, device=x_B_C_T_H_W.device)
        x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = self.model[0].prepare_embedded_sequence(
            x_B_C_T_H_W,
            fps=None,
            padding_mask=padding_mask,
        )
        assert extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D is None
        assert rope_emb_L_1_1_D is not None
        # Cast rope_emb to autocast dtype once here so the per-block rotary apply
        # doesn't repeat the float32->bf16 cast 112 times/step (28 blocks × 2 tensors × 2).
        rope_emb_L_1_1_D = rope_emb_L_1_1_D.to(AUTOCAST_DTYPE)

        if timesteps_B_T.ndim == 1:
            timesteps_B_T = timesteps_B_T.unsqueeze(1)
        t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T)
        t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)

        # Only x needs contiguous — the other 8 tensors are read-only (int/bool
        # tensors like t5_input_ids/attn_mask are unaffected, and the float tensors
        # are computed fresh each step). TransformerLayer.forward already re-contiguifies
        # x after each block, and LLMAdapterLayer re-contiguifies crossattn_emb.
        x_B_T_H_W_D = x_B_T_H_W_D.contiguous()
        outputs = (x_B_T_H_W_D, t_embedding_B_T_D, crossattn_emb, t5_input_ids, attn_mask, t5_attn_mask, rope_emb_L_1_1_D, adaln_lora_B_T_3D, timesteps_B_T)
        for tensor in outputs:
            if torch.is_floating_point(tensor):
                tensor.requires_grad_(True)
        return outputs


class LLMAdapterLayer(nn.Module):
    def __init__(self, llm_adapter):
        super().__init__()
        self.llm_adapter = llm_adapter

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        x_B_T_H_W_D, t_embedding_B_T_D, crossattn_emb, t5_input_ids, attn_mask, t5_attn_mask, rope_emb_L_1_1_D, adaln_lora_B_T_3D, timesteps_B_T = inputs

        if self.llm_adapter is not None:
            crossattn_emb = self.llm_adapter(
                source_hidden_states=crossattn_emb,
                target_input_ids=t5_input_ids,
                target_attention_mask=t5_attn_mask,
                source_attention_mask=attn_mask,
            )
            crossattn_emb = crossattn_emb.masked_fill(~t5_attn_mask.bool().unsqueeze(-1), 0)

        # Only crossattn_emb changed; x and other tensors are unchanged.
        crossattn_emb = crossattn_emb.contiguous()
        return (x_B_T_H_W_D, t_embedding_B_T_D, crossattn_emb, rope_emb_L_1_1_D, adaln_lora_B_T_3D, timesteps_B_T)


class TransformerLayer(nn.Module):
    def __init__(self, block, block_idx, offloader):
        super().__init__()
        self.block = block
        self.block_idx = block_idx
        self.offloader = offloader

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        x_B_T_H_W_D, t_embedding_B_T_D, crossattn_emb, rope_emb_L_1_1_D, adaln_lora_B_T_3D, timesteps_B_T = inputs

        self.offloader.wait_for_block(self.block_idx)
        x_B_T_H_W_D = self.block(x_B_T_H_W_D, t_embedding_B_T_D, crossattn_emb, rope_emb_L_1_1_D=rope_emb_L_1_1_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D)
        self.offloader.submit_move_blocks_forward(self.block_idx)

        # Only x needs contiguous — the other 5 tensors are unchanged between blocks.
        x_B_T_H_W_D = x_B_T_H_W_D.contiguous()
        return (x_B_T_H_W_D, t_embedding_B_T_D, crossattn_emb, rope_emb_L_1_1_D, adaln_lora_B_T_3D, timesteps_B_T)


class FinalLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.final_layer = model.final_layer
        self.model = [model]

    def __getattr__(self, name):
        return getattr(self.model[0], name)

    @torch.autocast('cuda', dtype=AUTOCAST_DTYPE)
    def forward(self, inputs):
        x_B_T_H_W_D, t_embedding_B_T_D, crossattn_emb, rope_emb_L_1_1_D, adaln_lora_B_T_3D, timesteps_B_T = inputs
        x_B_T_H_W_O = self.final_layer(x_B_T_H_W_D, t_embedding_B_T_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D)
        net_output_B_C_T_H_W = self.unpatchify(x_B_T_H_W_O)
        return net_output_B_C_T_H_W
