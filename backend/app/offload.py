"""Allocator policy for the isolated, single-job Krea2 worker."""
from contextlib import contextmanager
from functools import partial


def prefer_cudnn():
    from torch.nn.attention import sdpa_kernel,SDPBackend
    return sdpa_kernel([SDPBackend.CUDNN_ATTENTION,SDPBackend.FLASH_ATTENTION,
                        SDPBackend.EFFICIENT_ATTENTION,SDPBackend.MATH],set_priority=True)


@contextmanager
def reuse_allocator():
    import accelerate.hooks as hooks
    original=hooks.set_module_tensor_to_device
    # Accelerate otherwise empties the CUDA allocator after every materialized tensor.
    hooks.set_module_tensor_to_device=partial(original,clear_cache=False)
    try:
        yield
    finally:
        hooks.set_module_tensor_to_device=original

