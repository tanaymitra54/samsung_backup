"""
==============================================================================
FILE: pipeline/device_utils.py
ROLE: PyTorch CUDA & Compute Device Resolution Utility
BRANCH ADDITION (abhyuday): Added CUDA device index boundary validation (`idx >= count`)
to automatically fall back to the most-free GPU when an requested GPU index exceeds
available hardware, and made "no preference given" mean "pick the GPU with the most
free VRAM right now" instead of always defaulting to cuda:0 -- on a shared machine
cuda:0 is whichever card someone else's job happened to land on first.
==============================================================================
"""

import torch


def most_free_cuda_index() -> int | None:
    """Index of the CUDA device with the most free VRAM right now, or None if no CUDA."""
    if not torch.cuda.is_available():
        return None
    count = torch.cuda.device_count()
    if count == 1:
        return 0
    free_by_index = [(torch.cuda.mem_get_info(idx)[0], idx) for idx in range(count)]
    return max(free_by_index)[1]


def resolve_device(preferred: str | None = None) -> torch.device:
    if preferred:
        device = torch.device(preferred)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                return torch.device("cpu")
            count = torch.cuda.device_count()
            idx = device.index if device.index is not None else 0
            if idx >= count:
                idx = most_free_cuda_index()
                return torch.device(f"cuda:{idx}")
        return device
    idx = most_free_cuda_index()
    if idx is None:
        return torch.device("cpu")
    return torch.device(f"cuda:{idx}")


def hf_device_map_value(device: torch.device):
    if device.type != "cuda":
        return None
    return device.index or 0


def candidate_cuda_devices(preferred: str | None = None) -> list[torch.device]:
    if not torch.cuda.is_available():
        return []

    count = torch.cuda.device_count()
    preferred_device = resolve_device(preferred)
    ordered: list[int] = []

    if preferred_device.type == "cuda":
        preferred_index = preferred_device.index or 0
        if 0 <= preferred_index < count:
            ordered.append(preferred_index)

    free_by_index = []
    for idx in range(count):
        if idx in ordered:
            continue
        free_bytes, _ = torch.cuda.mem_get_info(idx)
        free_by_index.append((free_bytes, idx))

    free_by_index.sort(reverse=True)
    ordered.extend(idx for _, idx in free_by_index)
    return [torch.device(f"cuda:{idx}") for idx in ordered]
