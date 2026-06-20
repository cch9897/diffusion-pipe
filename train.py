import argparse
import os

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True,garbage_collection_threshold:0.6')
import glob
import inspect
import json
import random
import shutil
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import deepspeed
import toml
import torch
import wandb
from deepspeed import comm as dist
from deepspeed.runtime.pipe import module as ds_pipe_module

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import multiprocess as mp
import numpy as np
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import utils.saver
from utils import common
from utils import dataset as dataset_util
from utils.common import DTYPE_MAP, empty_cuda_cache, get_rank, is_main_process
from utils.isolate_rng import isolate_rng
from utils.patches import apply_patches
from utils.pipeline import ManualPipelineModule
from utils.unsloth_utils import unsloth_checkpoint

# needed for broadcasting Queue in dataset.py
mp.current_process().authkey = b'afsaskgfdjh4'

wandb_enable = False

TIMESTEP_QUANTILES_FOR_EVAL = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

parser = argparse.ArgumentParser()
parser.add_argument('--config', help='Path to TOML configuration file.')
parser.add_argument('--local_rank', type=int, default=-1,
                    help='local rank passed from distributed launcher')
parser.add_argument('--resume_from_checkpoint', nargs='?', const=True, default=None,
                    help='resume training from checkpoint. If no value is provided, resume from the most recent checkpoint. If a folder name is provided, resume from that specific folder.')
parser.add_argument('--reset_dataloader', action='store_true', help='Start dataloader from scratch when resuming from checkpoint, i.e. only load the optimizer states.')
parser.add_argument('--reset_optimizer', action='store_true')
parser.add_argument('--reset_optimizer_params', action='store_true')
parser.add_argument('--regenerate_cache', action='store_true', help='Force regenerate cache.')
parser.add_argument('--cache_only', action='store_true', help='Cache model inputs then exit.')
parser.add_argument('--trust_cache', action='store_true', help='Load from metadata cache files if they exist, without checking if any fingerprints have changed. Can make loading much faster for large datasets.')
parser.add_argument('--i_know_what_i_am_doing', action='store_true', help="Skip certain checks and overrides. You may end up using settings that won't work.")
parser.add_argument('--master_port', type=int, default=29500, help='Master port for distributed training')
parser.add_argument('--dump_dataset', type=Path, default=None, help='Decode cached latents and dump the dataset to this directory.')
parser.add_argument('--test_sample', action='store_true', help='Generate and write an image to example.png and then quit.')
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()


class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.state = defaultdict(dict)
        self.param_groups = []

    def step(self, closure=None):
        pass

    def zero_grad(self, set_to_none: bool = True):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass


# Monkeypatch this so it counts all layer parameters, not just trainable parameters.
# This helps it divide the layers between GPUs more evenly when training a LoRA.
def _count_all_layer_params(self):
    param_counts = [0] * len(self._layer_specs)
    for idx, layer in enumerate(self._layer_specs):
        if isinstance(layer, ds_pipe_module.LayerSpec):
            l = layer.build()
            param_counts[idx] = sum(p.numel() for p in l.parameters())
        elif isinstance(layer, nn.Module):
            param_counts[idx] = sum(p.numel() for p in layer.parameters())
    return param_counts
ds_pipe_module.PipelineModule._count_layer_params = _count_all_layer_params


def set_config_defaults(config):
    # Force the user to set this. If we made it a default of 1, it might use a lot of disk space.
    assert 'save_every_n_epochs' in config or 'save_every_n_steps' in config or 'save_every_n_examples' in config

    config.setdefault('pipeline_stages', 1)
    config.setdefault('activation_checkpointing', False)
    config.setdefault('reentrant_activation_checkpointing', False)
    if config['activation_checkpointing'] == 'unsloth':
        # unsloth's gradient checkpointing (Unsloth_Offloaded_Gradient_Checkpointer)
        # requires reentrant mode because it replays the forward pass during backward.
        config['reentrant_activation_checkpointing'] = True
    config.setdefault('warmup_steps', 0)
    if 'save_dtype' in config:
        config['save_dtype'] = DTYPE_MAP[config['save_dtype']]

    model_config = config['model']
    model_dtype_str = model_config['dtype']
    model_config['dtype'] = DTYPE_MAP[model_dtype_str]
    if transformer_dtype := model_config.get('transformer_dtype', None):
        model_config['transformer_dtype'] = DTYPE_MAP[transformer_dtype]
    if diffusion_model_dtype := model_config.get('diffusion_model_dtype', None):
        model_config['diffusion_model_dtype'] = DTYPE_MAP[diffusion_model_dtype]
    model_config.setdefault('guidance', 1.0)

    if 'adapter' in config:
        adapter_config = config['adapter']
        adapter_type = adapter_config['type']
        if 'alpha' in adapter_config:
            raise NotImplementedError(
                'This script forces alpha=rank to make the saved adapter format simpler and more predictable with downstream inference programs. Please remove alpha from the config.'
            )
        adapter_config['alpha'] = adapter_config['rank']
        adapter_config.setdefault('dtype', model_dtype_str)
        adapter_config['dtype'] = DTYPE_MAP[adapter_config['dtype']]

        # per-adapter defaults
        if adapter_config['type'] == 'lora':
            adapter_config.setdefault('dropout', 0.0)
        elif adapter_config['type'] == 'lokr':
            adapter_config.setdefault('decompose_factor', -1)
            adapter_config.setdefault('rank_dropout', 0.0)
        else:
            raise NotImplementedError(f'Adapter type {adapter_type} is not implemented')

    config.setdefault('logging_steps', 1)
    config.setdefault('eval_datasets', [])
    config.setdefault('eval_gradient_accumulation_steps', 1)
    config.setdefault('eval_every_n_steps', None)
    config.setdefault('eval_every_n_epochs', None)
    config.setdefault('eval_every_n_examples', None)
    config.setdefault('eval_before_first_step', True)
    config.setdefault('compile', False)
    config.setdefault('x_axis_examples', False)


def get_most_recent_run_dir(output_dir):
    return list(sorted(glob.glob(os.path.join(output_dir, '*'))))[-1]


def print_model_info(model):
    if not is_main_process():
        return
    print(model)
    for name, module in model.named_modules():
        print(f'{type(module)}: {name}')
        for pname, p in module.named_parameters(recurse=False):
            print(pname)
            print(p.dtype)
            print(p.device)
            print(p.requires_grad)
            print()


# Need to preload all micro batches since pulling from the dataloader does IPC between the
# first and last stage. Can't do that during the train or inference pipeline schedule execution
# because it conflicts with the send / recv steps.
def get_data_iterator_for_step(dataloader, engine, num_micro_batches=None):
    num_micro_batches = num_micro_batches or engine.micro_batches
    if not (engine.is_first_stage() or engine.is_last_stage()):
        return None
    dataloader_iter = iter(dataloader)
    items = [next(dataloader_iter) for _ in range(num_micro_batches)]
    return iter(items)


def evaluate_single(model_engine, eval_dataloader, eval_gradient_accumulation_steps, quantile, pbar=None):
    eval_dataloader.set_eval_quantile(quantile)
    total_loss = 0
    count = 0
    while True:
        model_engine.reset_activation_shape()
        iterator = get_data_iterator_for_step(eval_dataloader, model_engine, num_micro_batches=eval_gradient_accumulation_steps)
        loss = model_engine.eval_batch(iterator, num_micro_batches=eval_gradient_accumulation_steps).item()
        eval_dataloader.sync_epoch()
        if pbar:
            pbar.update(1)
        total_loss += loss
        count += 1
        if eval_dataloader.epoch == 2:
            break

    eval_dataloader.reset()
    return total_loss / count


def _evaluate(model_engine, eval_dataloaders, tb_writer, step, eval_gradient_accumulation_steps):
    pbar_total = 0
    for eval_dataloader in eval_dataloaders.values():
        pbar_total += len(eval_dataloader) * len(TIMESTEP_QUANTILES_FOR_EVAL) // eval_gradient_accumulation_steps
    if is_main_process():
        print('Running eval')
        pbar = tqdm(total=pbar_total)
    else:
        pbar = None

    start = time.time()
    for name, eval_dataloader in eval_dataloaders.items():
        losses = []
        for quantile in TIMESTEP_QUANTILES_FOR_EVAL:
            loss = evaluate_single(model_engine, eval_dataloader, eval_gradient_accumulation_steps, quantile, pbar=pbar)
            losses.append(loss)
            if is_main_process():
                tb_writer.add_scalar(f'{name}/loss_quantile_{quantile:.2f}', loss, step)
                if wandb_enable:
                    wandb.log({f'{name}/loss_quantile_{quantile:.2f}': loss, 'step': step})
        avg_loss = sum(losses) / len(losses)
        if is_main_process():
            tb_writer.add_scalar(f'{name}/loss', avg_loss, step)
            if wandb_enable:
                wandb.log({f'{name}/loss': avg_loss, 'step': step})

    duration = time.time() - start
    if is_main_process():
        tb_writer.add_scalar('eval/eval_time_sec', duration, step)
        if wandb_enable:
            wandb.log({'eval/eval_time_sec': duration, 'step': step})
        pbar.close()


def evaluate(model, model_engine, eval_dataloaders, tb_writer, step, eval_gradient_accumulation_steps, disable_block_swap):
    if len(eval_dataloaders) == 0:
        return
    empty_cuda_cache()
    model.prepare_block_swap_inference(disable_block_swap=disable_block_swap)
    try:
        with torch.no_grad(), isolate_rng():
            seed = get_rank()
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
            _evaluate(model_engine, eval_dataloaders, tb_writer, step, eval_gradient_accumulation_steps)
    finally:
        empty_cuda_cache()
        model.prepare_block_swap_training()


def distributed_init(args):
    """Initialize distributed training environment."""
    world_size = int(os.getenv('WORLD_SIZE', '1'))
    rank = int(os.getenv('RANK', '0'))
    local_rank = args.local_rank

    # Set environment variables for distributed training
    os.environ['MASTER_ADDR'] = os.getenv('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = str(args.master_port)

    return world_size, rank, local_rank


def get_prodigy_d(optimizer):
    d = 0
    for group in optimizer.param_groups:
        d += group['d']
    return d / len(optimizer.param_groups)


def _get_automagic_lrs(optimizer):
    lrs = []
    for group in optimizer.param_groups:
        for p in group['params']:
            state = optimizer.state[p]
            lr = optimizer._get_lr(group, state)
            lrs.append(lr)
    lrs = torch.stack(lrs)
    return lrs, lrs.mean()
def _remap_legacy_checkpoint(model_engine, pipeline_model, model):
    """Remap legacy checkpoint keys to the current model layout after DeepSpeed loads.

    DeepSpeed PipelineModule saves/loads checkpoints per-layer. When the checkpoint
    was saved with a different model structure (e.g. pre-QKV-fusion: separate
    q_proj/k_proj/v_proj vs fused qkv_proj), load_state_dict(strict=False) silently
    skips unmatched keys, leaving fused parameters at their initialization values.

    This reads each layer's checkpoint file from disk, applies the pipeline's
    remap_checkpoint_state_dict hook to convert legacy keys → current keys, and
    reloads the remapped state_dict into the layer. Only layers whose checkpoint
    contains legacy keys are reloaded.
    """
    if not hasattr(model_engine, '_curr_ckpt_path') or model_engine._curr_ckpt_path is None:
        return
    ckpt_dir = model_engine._curr_ckpt_path
    if not os.path.isdir(ckpt_dir):
        return

    remap_fn = getattr(model, 'remap_checkpoint_state_dict', None)
    if remap_fn is None:
        return

    grid = model_engine.grid
    mp_rank = 0 if grid is None else grid.get_slice_parallel_rank()
    mp_world_size = 1 if grid is None else grid.get_slice_parallel_world_size()

    remapped_count = 0
    for idx, layer in enumerate(pipeline_model.forward_funcs):
        if not hasattr(layer, 'load_state_dict'):
            continue
        # Reuse the same path resolution as PipelineModule.ckpt_layer_path_list
        layer_idx = idx + pipeline_model._local_start
        pattern = os.path.join(ckpt_dir, f'layer_{layer_idx:02d}-*model_states.pt')
        ckpt_files = sorted(glob.glob(pattern))
        if len(ckpt_files) == 0:
            continue
        # Pick this rank's shard (same logic as SDLoaderFactory for pipe-parallel)
        ckpt_file = ckpt_files[mp_rank % len(ckpt_files)]
        try:
            ckpt_sd = torch.load(ckpt_file, map_location='cpu', weights_only=True)
        except Exception:
            continue
        remapped_sd = remap_fn(ckpt_sd)
        if remapped_sd is ckpt_sd:
            # No remap happened (identity) — skip
            continue
        # Detect if any legacy keys were actually remapped (non-identity)
        if set(remapped_sd.keys()) == set(ckpt_sd.keys()):
            continue
        layer.load_state_dict(remapped_sd, strict=False)
        remapped_count += 1
        if is_main_process():
            print(f'  [checkpoint remap] layer {idx}: remapped {len(ckpt_sd)} -> {len(remapped_sd)} keys')

    if remapped_count > 0 and is_main_process():
        print(f'Checkpoint remap complete: {remapped_count} layers had legacy keys remapped to fused layout.')
        print('WARNING: optimizer state for remapped parameters may be stale (from pre-fusion training).')
        print('         Consider using --reset_optimizer if you observe training instability.')


if __name__ == '__main__':
    # With multiple GPUs / large batch sizes, the dataloader can trigger "too many open files" errors unless we do this.
    torch.multiprocessing.set_sharing_strategy('file_system')
    deepspeed.utils.set_log_level_from_string('info')
    apply_patches()

    with open(args.config) as f:
        # Inline TOML tables are not pickleable, which messes up the multiprocessing dataset stuff. This is a workaround.
        config = json.loads(json.dumps(toml.load(f)))

    set_config_defaults(config)
    common.AUTOCAST_DTYPE = config['model']['dtype']
    dataset_util.UNCOND_FRACTION = config.get('uncond_fraction', 0.0)
    if map_num_proc := config.get('map_num_proc', None):
        dataset_util.NUM_PROC = map_num_proc

    # Initialize distributed environment before deepspeed
    world_size, rank, local_rank = distributed_init(args)

    # Now initialize deepspeed
    deepspeed.init_distributed()

    # needed for broadcasting Queue in dataset.py
    torch.cuda.set_device(local_rank)

    resume_from_checkpoint = (
        args.resume_from_checkpoint if args.resume_from_checkpoint is not None
        else config.get('resume_from_checkpoint', False)
    )
    regenerate_cache = (
        args.regenerate_cache if args.regenerate_cache is not None
        else config.get('regenerate_cache', False)
    )

    model_type = config['model']['type']

    if model_type == 'flux':
        from models import flux
        model = flux.FluxPipeline(config)
    elif model_type == 'ltx-video':
        from models import ltx_video
        model = ltx_video.LTXVideoPipeline(config)
    elif model_type == 'hunyuan-video':
        from models import hunyuan_video
        model = hunyuan_video.HunyuanVideoPipeline(config)
    elif model_type == 'sdxl':
        from models import sdxl
        model = sdxl.SDXLPipeline(config)
    elif model_type == 'cosmos':
        from models import cosmos
        model = cosmos.CosmosPipeline(config)
    elif model_type == 'lumina_2':
        from models import lumina_2
        model = lumina_2.Lumina2Pipeline(config)
    elif model_type == 'wan':
        from models.wan import wan
        model = wan.WanPipeline(config)
    elif model_type == 'chroma':
        from models import chroma
        model = chroma.ChromaPipeline(config)
    elif model_type == 'hidream':
        from models import hidream
        model = hidream.HiDreamPipeline(config)
    elif model_type == 'sd3':
        from models import sd3
        model = sd3.SD3Pipeline(config)
    elif model_type == 'cosmos_predict2' or model_type == 'anima':
        from models import cosmos_predict2
        model = cosmos_predict2.CosmosPredict2Pipeline(config)
    elif model_type == 'omnigen2':
        from models import omnigen2
        model = omnigen2.OmniGen2Pipeline(config)
    elif model_type == 'qwen_image':
        from models import qwen_image
        model = qwen_image.QwenImagePipeline(config)
    elif model_type == 'hunyuan_image':
        from models import hunyuan_image
        model = hunyuan_image.HunyuanImagePipeline(config)
    elif model_type == 'auraflow':
        from models import auraflow
        model = auraflow.AuraFlowPipeline(config)
    elif model_type == 'z_image':
        from models import z_image
        model = z_image.ZImagePipeline(config)
    elif model_type == 'hunyuan_video_15':
        from models import hunyuan_video_15
        model = hunyuan_video_15.HunyuanVideo15Pipeline(config)
    elif model_type == 'flux2':
        from models import flux2
        model = flux2.Flux2Pipeline(config)
    elif model_type == 'ernie_image':
        from models import ernie_image
        model = ernie_image.ErnieImagePipeline(config)
    elif model_type == 'ltx2':
        from models import ltx2
        model = ltx2.LTX2Pipeline(config)
    elif model_type == 'ideogram4':
        from models import ideogram4
        model = ideogram4.Ideogram4Pipeline(config)
    else:
        raise NotImplementedError(f'Model type {model_type} is not implemented')

    # import sys, PIL
    # test_image = sys.argv[1]
    # with torch.no_grad():
    #     vae = model.get_vae().to('cuda')
    #     latents = dataset.encode_pil_to_latents(PIL.Image.open(test_image), vae)
    #     pil_image = dataset.decode_latents_to_pil(latents, vae)
    #     pil_image.save('test.jpg')
    # quit()

    with open(config['dataset']) as f:
        dataset_config = toml.load(f)

    micro_batch_size_per_gpu = config.get('micro_batch_size_per_gpu', 1)
    if isinstance(micro_batch_size_per_gpu, int):
        micro_batch_size_per_gpu = {None: micro_batch_size_per_gpu}
    elif isinstance(micro_batch_size_per_gpu, list):
        micro_batch_size_per_gpu = {x[0]: x[1] for x in micro_batch_size_per_gpu}

    eval_micro_batch_size_per_gpu = config.get('eval_micro_batch_size_per_gpu', micro_batch_size_per_gpu)
    if isinstance(eval_micro_batch_size_per_gpu, int):
        eval_micro_batch_size_per_gpu = {None: eval_micro_batch_size_per_gpu}
    elif isinstance(eval_micro_batch_size_per_gpu, list):
        eval_micro_batch_size_per_gpu = {x[0]: x[1] for x in eval_micro_batch_size_per_gpu}

    image_micro_batch_size_per_gpu = config.get('image_micro_batch_size_per_gpu', micro_batch_size_per_gpu)
    if isinstance(image_micro_batch_size_per_gpu, int):
        image_micro_batch_size_per_gpu = {None: image_micro_batch_size_per_gpu}
    elif isinstance(image_micro_batch_size_per_gpu, list):
        image_micro_batch_size_per_gpu = {x[0]: x[1] for x in image_micro_batch_size_per_gpu}

    eval_image_micro_batch_size_per_gpu = config.get('eval_image_micro_batch_size_per_gpu', eval_micro_batch_size_per_gpu)
    if isinstance(eval_image_micro_batch_size_per_gpu, int):
        eval_image_micro_batch_size_per_gpu = {None: eval_image_micro_batch_size_per_gpu}
    elif isinstance(eval_image_micro_batch_size_per_gpu, list):
        eval_image_micro_batch_size_per_gpu = {x[0]: x[1] for x in eval_image_micro_batch_size_per_gpu}

    default_micro_batch_size_per_gpu = list(micro_batch_size_per_gpu.values())[0]

    gradient_release = config['optimizer'].get('gradient_release', False)
    ds_config = {
        'train_micro_batch_size_per_gpu': default_micro_batch_size_per_gpu,
        'gradient_accumulation_steps': config.get('gradient_accumulation_steps', 1),
        # Can't do gradient clipping with gradient release, since there are no grads at the end of the step anymore.
        'gradient_clipping': 0. if gradient_release else config.get('gradient_clipping', 1.0),
        'steps_per_print': config.get('steps_per_print', 1),
    }
    caching_batch_size = config.get('caching_batch_size', 1)
    dataset_manager = dataset_util.DatasetManager(model, regenerate_cache=regenerate_cache, trust_cache=args.trust_cache, caching_batch_size=caching_batch_size, keep_models_loaded=args.test_sample)

    train_data = dataset_util.Dataset(dataset_config, model, skip_dataset_validation=args.i_know_what_i_am_doing)
    dataset_manager.register(train_data)

    eval_data_map = {}
    for i, eval_dataset in enumerate(config['eval_datasets']):
        if type(eval_dataset) == str:
            name = f'eval{i}'
            config_path = eval_dataset
        else:
            name = eval_dataset['name']
            config_path = eval_dataset['config']
        with open(config_path) as f:
            eval_dataset_config = toml.load(f)
        eval_data_map[name] = dataset_util.Dataset(eval_dataset_config, model, skip_dataset_validation=args.i_know_what_i_am_doing)
        dataset_manager.register(eval_data_map[name])

    # For testing

    # import imageio
    # from pathlib import Path
    # import torch.nn.functional as F
    # dataset_manager.cache(unload_models=False)
    # output_dir = Path('/home/anon/tmp')
    # train_data.post_init(
    #     0,
    #     1,
    #     1,
    #     1,
    # )
    # vae = model.vae
    # vae.model.to('cuda')
    # count = 1
    # for item in train_data:
    #     latents = item['latents'].to('cuda')
    #     h, w = latents.shape[-2:]
    #     mask = item['mask'].to('cuda')
    #     caption = item['caption'][0]
    #     mask = mask.unsqueeze(1)  # make mask (bs, 1, img_h, img_w)
    #     mask = F.interpolate(mask, size=(h, w), mode='nearest-exact')  # resize to latent spatial dimension
    #     mask = mask.unsqueeze(2)  # make mask same number of dims as target
    #     latents = latents * mask.to(latents.device)
    #     video = vae.model.decode(latents, vae.scale).float().clamp_(-1, 1).squeeze(0)
    #     video = torch.permute(video, (1, 2, 3, 0))
    #     video = (video + 1) / 2
    #     video = (video * 255).type(torch.uint8).cpu()
    #     imageio.v3.imwrite(output_dir / f'{count}.mp4', video, fps=16)
    #     with open(output_dir / f'{count}.txt', 'w') as f:
    #         f.write(caption)
    #     if count >= 10:
    #         break
    #     count += 1
    # quit()

    if args.dump_dataset:
        # only works for flux
        import torchvision
        dataset_manager.cache(unload_models=False)
        if is_main_process():
            with torch.no_grad():
                os.makedirs(args.dump_dataset, exist_ok=True)
                vae = model.vae.to('cuda')
                train_data.post_init(
                    0,
                    1,
                    1,
                    1,
                    1,
                )
                for i, item in enumerate(train_data):
                    latents = item['latents']
                    latents = latents / vae.config.scaling_factor
                    if hasattr(vae.config, 'shift_factor') and vae.config.shift_factor is not None:
                        latents = latents + vae.config.shift_factor
                    img = vae.decode(latents.to(vae.device, vae.dtype)).sample.to(torch.float32)
                    img = img.squeeze(0)
                    img = ((img + 1) / 2).clamp(0, 1)
                    pil_img = torchvision.transforms.functional.to_pil_image(img)
                    pil_img.save(args.dump_dataset / f'{i}.png')
                    if i >= 100:
                        break
        dist.barrier()
        quit()

    dataset_manager.cache()
    if args.cache_only:
        quit()

    if args.test_sample:
        model.prepare_sample_test('a golden retriever running through a grassy field', cfg=1)

    model.load_diffusion_model()

    if adapter_config := config.get('adapter', None):
        model.configure_adapter(adapter_config)
        is_adapter = True
        if init_from_existing := adapter_config.get('init_from_existing', None):
            model.load_adapter_weights(init_from_existing)
    else:
        is_adapter = False

    # Determine run_dir on rank 0 and broadcast it
    run_dir_container = [None]
    if is_main_process():
        if resume_from_checkpoint is True:
            run_dir_container[0] = get_most_recent_run_dir(config['output_dir'])
        elif isinstance(resume_from_checkpoint, str):
            run_dir_container[0] = os.path.join(config['output_dir'], resume_from_checkpoint)
        else:
            run_dir_container[0] = os.path.join(config['output_dir'], datetime.now(timezone.utc).strftime('%Y%m%d_%H-%M-%S'))

    torch.distributed.broadcast_object_list(run_dir_container, src=0, group=dist.get_world_group())
    run_dir = run_dir_container[0]

    os.makedirs(run_dir, exist_ok=True)
    if not resume_from_checkpoint and is_main_process():
        shutil.copy(args.config, run_dir)
        shutil.copy(config['dataset'], run_dir)
        for eval_dataset in config['eval_datasets']:
            shutil.copy(eval_dataset['config'], run_dir)
    dist.barrier()

    # WandB logging
    wandb_enable = config.get('monitoring', {}).get('enable_wandb', False)
    if wandb_enable and is_main_process():
        wandb_api_key     = config['monitoring']['wandb_api_key']
        wandb_tracker     = config['monitoring']['wandb_tracker_name']
        wandb_run_name    = config['monitoring']['wandb_run_name']
        logging_dir       = run_dir
        wandb.login(key=wandb_api_key)
        wandb.init(
            project=wandb_tracker,
            name=wandb_run_name,
            config=config,
            dir=logging_dir
        )

    # Block swapping
    if blocks_to_swap := config.get('blocks_to_swap', 0):
        assert config['pipeline_stages'] == 1, 'Block swapping only works with pipeline_stages=1'
        assert 'adapter' in config, 'Block swapping only works when training LoRA'
        # Don't automatically move to GPU, we'll do that ourselves.
        def to(self, *args, **kwargs):
            pass
        deepspeed.pipe.PipelineModule.to = to
        model.enable_block_swap(blocks_to_swap)

    layers = model.to_layers()
    additional_pipeline_module_kwargs = {}
    activation_checkpointing = config['activation_checkpointing']
    if activation_checkpointing:
        if activation_checkpointing == True:
            # TODO: block swapping doesn't work with Deepspeed non-reentrant checkpoint, but PyTorch native one is fine. Some
            # weights end up on CPU where they shouldn't. Why? Are we giving anything up by not using the Deepspeed implementation?
            #checkpoint_func = deepspeed.checkpointing.non_reentrant_checkpoint
            from functools import partial
            checkpoint_func = partial(torch.utils.checkpoint.checkpoint, use_reentrant=config['reentrant_activation_checkpointing'])
        elif activation_checkpointing == 'unsloth':
            checkpoint_func = unsloth_checkpoint
        else:
            raise NotImplementedError(f'activation_checkpointing={activation_checkpointing} is not implemented')
        additional_pipeline_module_kwargs.update({
            'activation_checkpoint_interval': 1,
            'checkpointable_layers': model.checkpointable_layers,
            'activation_checkpoint_func': checkpoint_func,
        })

    num_stages = config.get('pipeline_stages', 1)
    partition_method=config.get('partition_method', 'parameters')
    partition_split = config.get('partition_split',[len(layers) / num_stages])
    pipeline_model = ManualPipelineModule(
        layers=layers,
        num_stages=num_stages,
        partition_method=partition_method,
        manual_partition_split=partition_split,
        loss_fn=model.get_loss_fn(),
        **additional_pipeline_module_kwargs
    )
    model.pipeline_model = pipeline_model
    parameters_to_train = [p for p in pipeline_model.parameters() if p.requires_grad]

    if config['compile']:
        pipeline_model.compile(dynamic=True)

    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=pipeline_model,
        config=ds_config,
    )
    # Newer Deepspeed versions fail when pipeline_stages>1 because of a check on this field which defaults to False. But, pipeline
    # parallelism has always relied on "Torch-style" backward(), so I think this is an oversight by Deepspeed devs and it's safe
    # to force this to True to get it to work.
    model_engine._support_torch_style_backward = True
    global_batch_size = model_engine.train_micro_batch_size_per_gpu() * model_engine.gradient_accumulation_steps() * model_engine.grid.get_data_parallel_world_size()
    print(f'Global batch size = {global_batch_size}')

    if args.test_sample:
        import torchvision
        img = model.sample(w=512, h=512)
        img = img.squeeze(0).movedim(-1, 0)
        print(img.shape, img.min().item(), img.max().item())
        torchvision.utils.save_image(img, 'example.png')
        quit()

    if save_every_n_examples := config.pop('save_every_n_examples', None):
        config['save_every_n_steps'] = save_every_n_examples // global_batch_size
        print(f"Computed save_every_n_steps = {config['save_every_n_steps']}")
    if eval_every_n_examples := config.pop('eval_every_n_examples', None):
        config['eval_every_n_steps'] = eval_every_n_examples // global_batch_size
        print(f"Computed eval_every_n_steps = {config['eval_every_n_steps']}")

    def get_optimizer(model_parameters):
        if len(model_parameters) == 0:
            return DummyOptimizer()

        optim_config = config['optimizer']
        optim_type = optim_config['type']
        optim_type_lower = optim_type.lower()

        if beta2_half_life := optim_config.pop('beta2_half_life', None):
            betas = optim_config['betas']
            assert len(betas) == 2
            betas[1] = 0.5 ** (global_batch_size / beta2_half_life)
            print(f'Computed beta2 = {betas[1]}')
            optim_config['betas'] = betas

        args = []
        kwargs = {k: v for k, v in optim_config.items() if k not in ['type', 'gradient_release']}

        if optim_type_lower == 'adamw':
            # TODO: fix this. I'm getting "fatal error: cuda_runtime.h: No such file or directory"
            # when Deepspeed tries to build the fused Adam extension.
            # klass = deepspeed.ops.adam.FusedAdam
            klass = torch.optim.AdamW
            kwargs.setdefault('fused', True)
        elif optim_type_lower == 'adamw8bit':
            import bitsandbytes
            klass = bitsandbytes.optim.AdamW8bit
        elif optim_type_lower == 'adamw_optimi':
            import optimi
            klass = optimi.AdamW
        elif optim_type_lower == 'stableadamw':
            import optimi
            klass = optimi.StableAdamW
        elif optim_type_lower == 'sgd':
            klass = torch.optim.SGD
        elif optim_type_lower == 'adamw8bitkahan':
            from optimizers import adamw_8bit
            klass = adamw_8bit.AdamW8bitKahan
        elif optim_type_lower == 'offload':
            from torchao.prototype.low_bit_optim import CPUOffloadOptimizer
            klass = CPUOffloadOptimizer
            args.append(torch.optim.AdamW)
            kwargs['fused'] = True
        elif optim_type_lower == 'automagic':
            from optimizers import automagic
            klass = automagic.Automagic
        elif optim_type_lower == 'genericoptim':
            from optimizers import generic_optim
            klass = generic_optim.GenericOptim
        else:
            import pytorch_optimizer
            klass = getattr(pytorch_optimizer, optim_type)

        if optim_config.get('gradient_release', False):
            if config.get('pipeline_stages', 1) > 1:
                raise ValueError(
                    'gradient_release is incompatible with pipeline_stages > 1. '
                    'gradient_release uses per-parameter optimizers and modifies '
                    'parameter .data.add_() which bypasses DeepSpeed pipeline scheduling. '
                    'Set pipeline_stages=1 or disable gradient_release.'
                )
            # Prevent deepspeed from logging every single param group lr
            def _report_progress(self, step):
                lr = self.get_lr()
                mom = self.get_mom()
                deepspeed.utils.logging.log_dist(f"step={step}, skipped={self.skipped_steps}, lr={lr[0]}, mom={mom[0]}", ranks=[0])
            deepspeed.runtime.engine.DeepSpeedEngine._report_progress = _report_progress

            # Deepspeed executes all the code to reduce grads across data parallel ranks even if the DP world size is 1.
            # As part of this, any grads that are None are set to zeros. We're doing gradient release to save memory,
            # so we have to avoid this.
            def _exec_reduce_grads(self):
                assert self.mpu.get_data_parallel_world_size() == 1, 'When using gradient release, data parallel world size must be 1. Make sure pipeline_stages = num_gpus.'
                return
            deepspeed.runtime.pipe.engine.PipelineEngine._INSTRUCTION_MAP[deepspeed.runtime.pipe.schedule.ReduceGrads] = _exec_reduce_grads

            # When pipelining multiple forward and backward passes, normally updating the parameter in-place causes an error when calling
            # backward() on future micro-batches. But we can modify .data directly so the autograd engine doesn't detect in-place modifications.
            # TODO: this is unbelievably hacky and not mathematically sound, I'm just seeing if it works at all.
            def add_(self, *args, **kwargs):
                self.data.add_(*args, **kwargs)
            for p in model_parameters:
                p.add_ = add_.__get__(p)

            if 'foreach' in inspect.signature(klass).parameters:
                kwargs['foreach'] = False

            # We're doing an optimizer step for each micro-batch. Scale momentum and EMA betas so that the contribution
            # decays at the same rate it would if we were doing one step per batch like normal.
            # Reference: https://alexeytochin.github.io/posts/batch_size_vs_momentum/batch_size_vs_momentum.html
            gas = ds_config['gradient_accumulation_steps']
            if 'betas' in kwargs:
                for i in range(len(kwargs['betas'])):
                    kwargs['betas'][i] = kwargs['betas'][i] ** (1/gas)
            if 'momentum' in kwargs:
                kwargs['momentum'] = kwargs['momentum'] ** (1/gas)

            # We're doing an optimizer step for each micro-batch, so the effective
            # update frequency is GAS× higher than normal. Scale lr by 1/gas by
            # default so the per-step contribution matches one-step-per-batch.
            # Set gas_lr_scale=1.0 to keep the original (unscaled) lr.
            gas_lr_scale = optim_config.get('gas_lr_scale', 1.0 / gas)

            # Group params by lr so params sharing a learning rate share one optimizer
            # instance instead of one-per-param. register_post_accumulate_grad_hook fires
            # per-param, but torch optimizer.step() only updates params whose .grad is set,
            # so a shared instance steps correctly for a single param. This reduces
            # kernel launches from N_params to N_groups.
            optimizer_dict = {}
            grouped = {}  # lr_key -> (lr_value, [params])
            for pg in model.get_param_groups(model_parameters):
                if isinstance(pg, dict):
                    lr_key = pg['lr']
                else:
                    # bare param — use the top-level lr from kwargs
                    lr_key = kwargs.get('lr')
                    pg = {'params': [pg], 'lr': lr_key}
                grouped.setdefault(lr_key, [lr_key, []])[1].extend(pg['params'])

            # One optimizer instance per unique lr. Each param maps to its shared optimizer.
            shared_optimizers = []
            for lr_key, (lr_value, params) in grouped.items():
                if lr_value == 0:
                    for p in params:
                        p.requires_grad_(False)
                    continue
                param_kwargs = kwargs.copy()
                param_kwargs['lr'] = lr_value * gas_lr_scale
                opt = klass(params, **param_kwargs)
                shared_optimizers.append(opt)
                for p in params:
                    optimizer_dict[p] = opt

            def optimizer_hook(p):
                optimizer_dict[p].step()
                # Set only this param's grad to None — not zero_grad() on the shared
                # optimizer, which would wipe other params' not-yet-consumed grads.
                p.grad = None

            for p in model_parameters:
                p.register_post_accumulate_grad_hook(optimizer_hook)

            from optimizers import gradient_release
            return gradient_release.GradientReleaseOptimizerWrapper(shared_optimizers)
        elif optim_type_lower == 'genericoptim':
            kwargs['compile'] = config['compile']
            kwargs['mpu'] = pipeline_model.mpu()
            new_param_groups = []
            param_groups = model.get_param_groups(model_parameters)
            for pg in param_groups:
                params = pg.pop('params')
                params_2d = []
                params_other = []
                for p in params:
                    if p.ndim == 2:
                        params_2d.append(p)
                    else:
                        params_other.append(p)
                pg_2d = pg.copy()
                pg_2d['params'] = params_2d
                if kwargs.get('second_moment_type', None) == 'sn':
                    pg_2d['subset_size'] = 'heuristics'
                for key in ('rank', 'proj_type', 'update_proj_gap'):
                    if key in kwargs:
                        pg_2d[key] = kwargs.pop(key)
                new_param_groups.append(pg_2d)
                pg_other = pg
                pg_other['params'] = params_other
                new_param_groups.append(pg_other)
            param_groups = new_param_groups
        else:
            param_groups = model.get_param_groups(model_parameters)

        # split weight decay and no weight decay params
        new_param_groups = []
        for pg in param_groups:
            params_no_wd = []
            params_wd = []
            params = pg.pop('params')
            for p in params:
                if p.ndim == 1 or getattr(p, 'original_name', '').startswith('llm_adapter.embed'):
                    params_no_wd.append(p)
                else:
                    params_wd.append(p)
            pg_no_wd = pg.copy()
            pg['params'] = params_wd
            pg_no_wd['params'] = params_no_wd
            pg_no_wd['weight_decay'] = 0
            if optim_type_lower == 'genericoptim':
                # If we aren't using weight decay, don't use Muon either (handles LLM adapter embed properly)
                pg_no_wd['muon'] = False
                pg_no_wd['adamuon'] = False
                pg_no_wd['normuon'] = False
            if len(params_wd) > 0:
                new_param_groups.append(pg)
            if len(params_no_wd) > 0:
                new_param_groups.append(pg_no_wd)
        param_groups = new_param_groups

        return klass(param_groups, *args, **kwargs)

    model_engine._configure_optimizer(get_optimizer, parameters_to_train)
    optimizer = model_engine.optimizer

    model.model_engine = model_engine
    if model_engine.is_pipe_parallel:
         grid = model_engine.grid
         model_engine.first_last_stage_group = dist.new_group(ranks=[grid.pp_group[0], grid.pp_group[-1]])

    train_data.post_init(
        model_engine.grid.get_data_parallel_rank(),
        model_engine.grid.get_data_parallel_world_size(),
        micro_batch_size_per_gpu,
        model_engine.gradient_accumulation_steps(),
        image_micro_batch_size_per_gpu,
    )
    for eval_data in eval_data_map.values():
        eval_data.post_init(
            model_engine.grid.get_data_parallel_rank(),
            model_engine.grid.get_data_parallel_world_size(),
            eval_micro_batch_size_per_gpu,
            config['eval_gradient_accumulation_steps'],
            eval_image_micro_batch_size_per_gpu,
        )

    # Might be useful because we set things in fp16 / bf16 without explicitly enabling Deepspeed fp16 mode.
    # Unsure if really needed.
    communication_data_type = config['adapter']['dtype'] if 'adapter' in config else config['model']['dtype']
    model_engine.communication_data_type = communication_data_type

    train_dataloader = dataset_util.PipelineDataLoader(train_data, model_engine, model_engine.gradient_accumulation_steps(), model)
    steps_per_epoch = len(train_dataloader) // model_engine.gradient_accumulation_steps()

    scheduler_type = config.get('lr_scheduler', 'constant')
    if scheduler_type == 'constant':
        lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    elif scheduler_type == 'linear':
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=config['epochs'] * steps_per_epoch)
    elif scheduler_type == 'cosine':
        eta_min = config.get('lr_scheduler_eta_min', 0.0)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'] * steps_per_epoch, eta_min=eta_min)
    else:
        raise NotImplementedError(f'Unknown lr_scheduler: {scheduler_type}')
    if config['warmup_steps'] > 0:
        warmup_steps = config['warmup_steps']
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1/warmup_steps, total_iters=warmup_steps)
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, lr_scheduler], milestones=[warmup_steps])
    model_engine.lr_scheduler = lr_scheduler

    step = 1
    examples = global_batch_size
    # make sure to do this before calling model_engine.set_dataloader(), as that method creates an iterator
    # which starts creating dataloader internal state
    if resume_from_checkpoint:
        param_groups = optimizer.param_groups.copy()
        load_path, client_state = model_engine.load_checkpoint(
            run_dir,
            load_module_strict=False,
            load_lr_scheduler_states='force_constant_lr' not in config and not args.reset_optimizer and not args.reset_optimizer_params,
            load_optimizer_states=not args.reset_optimizer,
        )
        if args.reset_optimizer_params:
            optimizer.param_groups = param_groups
        dist.barrier()  # just so the print below doesn't get swamped
        assert load_path is not None
        if args.reset_dataloader:
            train_dataloader.epoch = client_state['custom_loader']['epoch']
        else:
            train_dataloader.load_state_dict(client_state['custom_loader'])
        step = client_state['step'] + 1
        if 'examples' in client_state:
            examples = client_state['examples'] + global_batch_size
        else:
            examples = step * global_batch_size
        del client_state
        if is_main_process():
            print(f'Resuming training from checkpoint. Resuming at epoch: {train_dataloader.epoch}, step: {step}')

        # Remap legacy checkpoint keys to the current model layout.
        # This handles the case where the checkpoint was saved with a different
        # model structure (e.g. before QKV/AdaLN fusion). DeepSpeed loads per-layer
        # checkpoint files with load_state_dict(strict=False), so legacy keys that
        # don't match the current fused parameter names are silently skipped,
        # leaving fused params at their initialization values (e.g.
        # adaln_modulation_2 stays zero-initialized → all modulation lost → black images).
        _remap_legacy_checkpoint(model_engine, pipeline_model, model)

    if 'force_constant_lr' in config:
        model_engine.lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
        for pg in optimizer.param_groups:
            pg['lr'] = config['force_constant_lr']

    eval_dataloaders = {
        name: dataset_util.PipelineDataLoader(eval_data, model_engine, config['eval_gradient_accumulation_steps'], model, num_dataloader_workers=0)
        for name, eval_data in eval_data_map.items()
    }

    epoch = train_dataloader.epoch
    tb_writer = SummaryWriter(log_dir=run_dir) if is_main_process() else None
    saver = utils.saver.Saver(args, config, is_adapter, run_dir, model, train_dataloader, model_engine, pipeline_model)

    disable_block_swap_for_eval = config.get('disable_block_swap_for_eval', False)
    if config['eval_before_first_step'] and not resume_from_checkpoint:
        evaluate(model, model_engine, eval_dataloaders, tb_writer, 0, config['eval_gradient_accumulation_steps'], disable_block_swap_for_eval)

    # TODO: this is state we need to save and resume when resuming from checkpoint. It only affects logging.
    epoch_loss = 0
    num_steps = 0
    empty_cuda_cache()

    # Cross-step data prefetch: overlap next step's data preparation with current step's GPU compute.
    # NOTE: Disabled when pipeline_stages > 1 because _broadcast_target() performs
    # dist.send/recv on the default NCCL communicator, which races with train_batch()
    # NCCL operations. NCCL does not guarantee thread safety on the same communicator
    # for concurrent point-to-point and collective operations — this can cause deadlocks
    # or silent data corruption.
    from concurrent.futures import ThreadPoolExecutor
    _prefetch_executor = None
    if (model_engine.is_first_stage() or model_engine.is_last_stage()) and not model_engine.is_pipe_parallel:
        _prefetch_executor = ThreadPoolExecutor(max_workers=1)
    _prefetch_future = None

    def _prefetch_next():
        if _prefetch_executor is None:
            return None
        return _prefetch_executor.submit(get_data_iterator_for_step, train_dataloader, model_engine)

    # Kick off the first prefetch
    _prefetch_future = _prefetch_next()

    # Progress bar
    total_steps = config.get('max_steps', config['epochs'] * steps_per_epoch)
    if is_main_process():
        pbar = tqdm(total=total_steps, initial=step, desc='Training',
                    bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}')
    else:
        pbar = None

    _profile = os.environ.get('DP_PROFILE', '') == '1'
    _step_start = _step_end = None
    if _profile:
        _step_start = torch.cuda.Event(enable_timing=True)
        _step_end = torch.cuda.Event(enable_timing=True)

    while True:
        if _profile:
            _step_start.record()

        model_engine.reset_activation_shape()
        # Wait for the prefetch to complete (should be ready by now since GPU was busy)
        if _prefetch_future is not None:
            iterator = _prefetch_future.result()
        else:
            iterator = get_data_iterator_for_step(train_dataloader, model_engine)
        # Start prefetching the next step's data while GPU computes
        _prefetch_future = _prefetch_next()
        loss = model_engine.train_batch(iterator).item()

        if _profile:
            _step_end.record()
            _total_ms = _step_start.elapsed_time(_step_end)
            if step <= 10 or step % 50 == 0:
                print(f'[PROF] total step {step}: {_total_ms:.0f}ms', flush=True)

        # One-shot detailed profiler on step 6 (steady state, gated by DP_PROFILE)
        if _profile and step == 6:
            print('[PROF] Starting detailed profiler on step 7...', flush=True)
            torch.cuda.synchronize()
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
            ) as prof:
                model_engine.reset_activation_shape()
                if _prefetch_future is not None:
                    iterator = _prefetch_future.result()
                else:
                    iterator = get_data_iterator_for_step(train_dataloader, model_engine)
                _prefetch_future = _prefetch_next()
                loss = model_engine.train_batch(iterator).item()
                torch.cuda.synchronize()

            print('[PROF] === Top 30 CUDA operators (by CUDA time) ===', flush=True)
            print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=30), flush=True)
            print('[PROF] === Top 15 CPU operators (by CPU time) ===', flush=True)
            print(prof.key_averages().table(sort_by='cpu_time_total', row_limit=15), flush=True)
            # Also save chrome trace for deeper analysis
            prof.export_chrome_trace('/tmp/dp_trace_step7.json')
            print('[PROF] Chrome trace saved to /tmp/dp_trace_step7.json', flush=True)
        epoch_loss += loss
        num_steps += 1
        train_dataloader.sync_epoch()

        if pbar is not None:
            postfix = {'loss': f'{loss:.4f}'}
            step_save = config.get('save_every_n_steps')
            epoch_save = config.get('save_every_n_epochs')
            if step_save:
                postfix['save'] = f'{step_save - (step % step_save)}/{step_save}st'
            elif epoch_save:
                steps_to_epoch = steps_per_epoch - (step % steps_per_epoch)
                postfix['save'] = f'{steps_to_epoch}s→ep'
            pbar.set_postfix(postfix)

        if step % 50 == 0:
            empty_cuda_cache()

        new_epoch, checkpointed, saved = saver.process_epoch(epoch, step, examples)
        finished_epoch = True if new_epoch != epoch else False

        x_axis = examples if config['x_axis_examples'] else step

        if is_main_process() and step % config['logging_steps'] == 0:
            tb_writer.add_scalar('train/loss', loss, x_axis)
            if hasattr(optimizer, '_grad_norm'):
                tb_writer.add_scalar('train/grad_norm', optimizer._grad_norm, x_axis)
            if wandb_enable:
                wandb.log({'train/loss': loss, 'step': x_axis})
                if hasattr(optimizer, '_grad_norm'):
                    wandb.log({'train/grad_norm': optimizer._grad_norm, 'step': x_axis})
            if optimizer.__class__.__name__ == 'Prodigy':
                prodigy_d = get_prodigy_d(optimizer)
                tb_writer.add_scalar('train/prodigy_d', prodigy_d, x_axis)
            if optimizer.__class__.__name__ in ('Automagic', 'GenericOptim'):
                lrs, avg_lr = _get_automagic_lrs(optimizer)
                if avg_lr > 0:
                    tb_writer.add_histogram('train/automagic_lrs', lrs, x_axis)
                    tb_writer.add_scalar('train/automagic_avg_lr', avg_lr, x_axis)

        if (config['eval_every_n_steps'] and step % config['eval_every_n_steps'] == 0) or (finished_epoch and config['eval_every_n_epochs'] and epoch % config['eval_every_n_epochs'] == 0):
            evaluate(model, model_engine, eval_dataloaders, tb_writer, x_axis, config['eval_gradient_accumulation_steps'], disable_block_swap_for_eval)

        if finished_epoch:
            if is_main_process():
                tb_writer.add_scalar('train/epoch_loss', epoch_loss/num_steps, epoch)
                if wandb_enable:
                    wandb.log({'train/epoch_loss': epoch_loss/num_steps, 'epoch': epoch})
            epoch_loss = 0
            num_steps = 0
            if new_epoch is None:
                final_model_name = f'epoch{epoch}'
                break
            epoch = new_epoch

        checkpointed, saved = saver.process_step(step, examples)
        if 'max_steps' in config and step >= config['max_steps']:
            final_model_name = f'step{step}'
            break
        step += 1
        examples += global_batch_size
        if pbar is not None:
            pbar.update(1)

    # Save final training state checkpoint and model, unless we just saved them.
    if not checkpointed:
        saver.save_checkpoint(step, examples)
    if not saved:
        saver.save_model(final_model_name)

    if is_main_process():
        if pbar is not None:
            pbar.close()
        print('TRAINING COMPLETE!')
