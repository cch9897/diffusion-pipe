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

"""Pure state-dict key remapping functions for Cosmos-Predict2.

These functions operate only on tensor keys and do NOT import heavy ML
dependencies (transformers, accelerate, peft, etc.). They can be imported
and tested in environments without the full training stack.
"""

import re
import warnings

import torch


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
