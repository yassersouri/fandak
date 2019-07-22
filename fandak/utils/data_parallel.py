import torch
import torch.nn as nn

# noinspection PyProtectedMember
from torch.nn.parallel._functions import Scatter, Gather

from fandak.core.datasets import GeneralBatch
from fandak.core.models import GeneralForwardOut
from fandak.utils.torch import GeneralDataClass


def scatter_kwargs(inputs, kwargs, target_gpus, dim=0):
    inputs = scatter(inputs, target_gpus, dim) if inputs else []
    kwargs = scatter(kwargs, target_gpus, dim) if kwargs else []
    if len(inputs) < len(kwargs):
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    elif len(kwargs) < len(inputs):
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs


def scatter(inputs, target_gpus, dim=0):
    r"""
    Slices tensors into approximately equal chunks and
    distributes them across given GPUs. Duplicates
    references to objects that are not tensors.
    """

    def scatter_map(obj):
        if isinstance(obj, torch.Tensor):
            return Scatter.apply(target_gpus, None, dim, obj)
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(scatter_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            return list(map(list, zip(*map(scatter_map, obj))))
        if isinstance(obj, dict) and len(obj) > 0:
            return list(map(type(obj), zip(*map(scatter_map, obj.items()))))
        if isinstance(obj, GeneralDataClass):
            returned_values = {}
            all_attribute_names = obj.get_attribute_names()
            tensor_attribute_names = obj.get_tensor_attributes()
            non_tensor_attribute_names = list(
                set(all_attribute_names).difference(set(tensor_attribute_names))
            )
            for a in tensor_attribute_names:
                returned_values[a] = scatter_map(getattr(obj, a))
            num_objects_to_create = len(returned_values[tensor_attribute_names[0]])
            objects = []
            for i in range(num_objects_to_create):
                d = {a: returned_values[a][i] for a in tensor_attribute_names}
                d.update({a: getattr(obj, a) for a in non_tensor_attribute_names})
                x = obj.__class__(**d)
                objects.append(x)
            return objects
        return [obj for targets in target_gpus]

    # After scatter_map is called, a scatter_map cell will exist. This cell
    # has a reference to the actual function scatter_map, which has references
    # to a closure that has a reference to the scatter_map cell (because the
    # fn is recursive). To avoid this reference cycle, we set the function to
    # None, clearing the cell
    try:
        res = scatter_map(inputs)
    finally:
        scatter_map = None
    return res


def gather(outputs, target_device, dim=0):
    r"""
    Gathers tensors from different GPUs on a specified device
      (-1 means the CPU).
    """

    def gather_map(outputs):
        out = outputs[0]
        if isinstance(out, torch.Tensor):
            return Gather.apply(target_device, dim, *outputs)
        if out is None:
            return None
        if isinstance(out, dict):
            if not all((len(out) == len(d) for d in outputs)):
                raise ValueError("All dicts must have the same number of keys")
            return type(out)(((k, gather_map([d[k] for d in outputs])) for k in out))
        if isinstance(out, GeneralDataClass):
            values = {}
            all_attribute_names = out.get_attribute_names()
            tensor_attribute_names = out.get_tensor_attributes()
            non_tensor_attribute_names = list(
                set(all_attribute_names).difference(set(tensor_attribute_names))
            )
            for a in tensor_attribute_names:
                values[a] = gather_map([getattr(obj, a) for obj in outputs])
            for a in non_tensor_attribute_names:
                values[a] = getattr(out, a)

            return out.__class__(**values)
        return type(out)(map(gather_map, zip(*outputs)))

    # Recursive function calls like this create reference cycles.
    # Setting the function to None clears the refcycle.
    try:
        res = gather_map(outputs)
    finally:
        gather_map = None
    return res


class DataParallel(nn.DataParallel):
    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def gather(self, outputs, output_device):
        return gather(outputs, output_device, dim=self.dim)

    def loss(self, batch: GeneralBatch, forward_out: GeneralForwardOut):
        return self.module.loss(batch=batch, forward_out=forward_out)
