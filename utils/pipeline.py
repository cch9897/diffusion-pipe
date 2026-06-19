from torch import nn
import torch

from deepspeed.pipe import PipelineModule
from deepspeed.runtime.pipe import LayerSpec


# PipelineModule partition_method doesn't support uneven partitioning
# This allow for loading more layers into selected GPU
# For example if you have 2 gpus - one with 16GB and other with 24GB normal partitioning would throw OOM
# With this implementation you can set partition_split in config so that less layers is loaded onto 16GB GPU
class ManualPipelineModule(PipelineModule):
    def __init__(self, *args, manual_partition_split=None, **kwargs):
        self.manual_partition_split = manual_partition_split
        # Workaround PyTorch 2.12: Module.to() no longer supports meta→device.
        # DeepSpeed PipelineModule.__init__ calls self.to(device) on meta-param layers.
        # If ANY parameter is still meta (not loaded from checkpoint), Module.to()
        # raises NotImplementedError. The old workaround called to_empty(), which
        # replaces ALL parameters with uninitialized (zeroed) CUDA tensors — silently
        # destroying checkpoint values and producing all-zero gradients.
        # Fix: materialize only the meta tensors, preserving loaded parameters.
        _original_to = nn.Module.to
        def _to_empty_fallback(module, *to_args, **to_kwargs):
            try:
                return _original_to(module, *to_args, **to_kwargs)
            except NotImplementedError:
                device = to_kwargs.pop('device', None)
                if device is None and to_args:
                    arg0 = to_args[0]
                    if isinstance(arg0, (torch.device, str, int)):
                        device = arg0
                    elif hasattr(arg0, 'device'):  # e.g. a tensor
                        device = arg0.device

                # Count meta tensors and materialize them individually.
                n_meta = 0
                meta_names = []
                for name, p in module.named_parameters(recurse=True):
                    if p.is_meta:
                        n_meta += 1
                        if len(meta_names) < 10:
                            meta_names.append(name)

                if n_meta > 0:
                    print(f'[WARNING] {n_meta} parameters were not loaded from checkpoint '
                          f'(remained as meta tensors). They will be zero-initialized. '
                          f'Examples: {meta_names}', flush=True)

                    # Materialize meta tensors to real zero tensors on the target device,
                    # while moving non-meta tensors normally. Using _apply ensures all
                    # tensors (params + buffers) are handled.
                    def _convert(t):
                        if t.is_meta:
                            return torch.zeros(t.shape, dtype=t.dtype, device=device)
                        return t.to(device)
                    module._apply(_convert)
                    return module
                else:
                    # No meta tensors found — the NotImplementedError came from
                    # something else. Fall back to to_empty as before.
                    return module.to_empty(device=device, **to_kwargs)
        nn.Module.to = _to_empty_fallback
        try:
            super().__init__(*args, **kwargs)
        finally:
            nn.Module.to = _original_to

    def _partition_layers(self, method='uniform'):
        if method.lower() == 'manual' and self.manual_partition_split is not None:
            num_stages = self._topo.get_dim('pipe')
            stage_id = self._topo.get_coord(self.global_rank).pipe
            num_partitions = len(self.manual_partition_split)
            assert num_partitions == num_stages - 1, f'partition_split must be length {num_stages-1} (pipeline_stages-1), was actually {num_partitions}'

            total_layers = len(self._layer_specs)
            boundaries = [0] + self.manual_partition_split + [total_layers]
            self.parts = boundaries

            # Print some information on the partitioning.
            if self.global_rank == 0:
                for stage in range(num_stages):
                    start = self.parts[stage]
                    stop = self.parts[stage + 1]
                    print(f'stage={stage} layers={stop - start}')
                    for idx, layer in enumerate(self._layer_specs[start:stop]):
                        name = str(layer)
                        if isinstance(layer, LayerSpec):
                            name = layer.typename.__name__
                        if isinstance(layer, nn.Module):
                            name = layer.__class__.__name__
                        else:
                            try:
                                name = layer.__name__
                            except AttributeError:
                                pass
                        print(f'    {idx+start:2d}: {name}')
                if self.loss_fn:
                    try:
                        print(f'  loss: {self.loss_fn.__name__}')
                    except AttributeError:
                        print(f'  loss: {self.loss_fn.__class__.__name__}')

            self._set_bounds(start=self.parts[stage_id], stop=self.parts[stage_id+1])
        else:
            super()._partition_layers(method)
