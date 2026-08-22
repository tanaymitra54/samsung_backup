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

import os
import subprocess

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


def most_free_gpu_index_nvidia_smi() -> int | None:
    """Same idea as most_free_cuda_index(), but via `nvidia-smi` instead of torch.cuda.

    Safe to call before torch/CUDA has been initialized -- see
    mask_cuda_visible_devices() below for why that matters. Returns None if
    nvidia-smi is unavailable or reports no devices.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    free = [int(x) for x in out.strip().splitlines() if x.strip()]
    if not free:
        return None
    return max(range(len(free)), key=lambda i: free[i])


def mask_cuda_visible_devices(requested: str | None = None) -> None:
    """Restrict this process's CUDA visibility to a single GPU, before torch/CUDA exists.

    HuggingFace's Trainer/accelerate resolve their own training device
    independently of wherever a model was actually placed (via device_map or
    .to()) -- that resolution lands on cuda:0 regardless, so on a shared box
    where GPU 0 is saturated, Trainer.__init__ tries to move an
    already-placed model back onto the full card and OOMs even though the
    model loaded fine on the GPU we picked. Restricting CUDA_VISIBLE_DEVICES
    to one physical GPU sidesteps this: from the process's point of view
    there is only one GPU, "cuda:0", so every device-resolution path (ours
    and HF's) agrees on it.

    MUST be called before `import torch` (or anything that imports torch) --
    once a CUDA context exists the visible device list is fixed for the life
    of the process. Uses nvidia-smi rather than torch for the free-memory
    query for exactly that reason; this module itself is safe to import
    first since merely importing torch does not touch CUDA.

    `requested` mirrors a --device CLI value: "cuda:N" pins to physical GPU
    N, "cpu" leaves masking alone (nothing to restrict), None/anything else
    auto-picks the GPU with the most free VRAM. A no-op if CUDA_VISIBLE_DEVICES
    is already set (respects a scheduler/container that already restricted us).
    """
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        return
    if requested == "cpu":
        return
    if requested and requested.startswith("cuda:"):
        os.environ["CUDA_VISIBLE_DEVICES"] = requested.split(":", 1)[1]
        return
    idx = most_free_gpu_index_nvidia_smi()
    if idx is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)


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
