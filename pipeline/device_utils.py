import os

import torch


def resolve_device(preferred: str | None = None) -> torch.device:
    if not torch.cuda.is_available():
        if preferred and str(preferred).startswith("cuda"):
            raise RuntimeError("NVIDIA GPU is required, but CUDA is not available.")
        return torch.device("cpu")

    count = torch.cuda.device_count()
    if preferred:
        device = torch.device(preferred)
        if device.type != "cuda":
            return device
        index = device.index or 0
        if index < 0 or index >= count:
            index = 0
        return torch.device(f"cuda:{index}")
    return torch.device("cuda:0")


def hf_device_map_value(device: torch.device):
    """Pin Hugging Face loads to a single CUDA index (never span all visible GPUs)."""
    if device.type != "cuda":
        return None
    return device.index or 0


def resolve_torch_dtype(config: dict | None = None, fallback: str = "bfloat16"):
    """Pick torch dtype from hardware.dtype / model settings."""
    name = fallback
    if config:
        name = (
            (config.get("hardware") or {}).get("dtype")
            or (config.get("model") or {}).get("torch_dtype")
            or fallback
        )
    if not torch.cuda.is_available():
        return torch.float32
    return getattr(torch, str(name), torch.bfloat16)


def list_visible_gpu_ids() -> list[int]:
    if not torch.cuda.is_available():
        return []
    return list(range(torch.cuda.device_count()))


def mask_worker_gpu(gpu_id: int) -> str:
    """Restrict the current process to one physical GPU; return remapped device string."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    return "cuda:0"


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


def shard_indices(n: int, num_shards: int) -> list[list[int]]:
    """Split range(n) into num_shards contiguous (nearly equal) index lists."""
    if num_shards <= 0:
        return [[]]
    if n <= 0:
        return [[] for _ in range(num_shards)]
    shards: list[list[int]] = []
    base, rem = divmod(n, num_shards)
    start = 0
    for i in range(num_shards):
        size = base + (1 if i < rem else 0)
        shards.append(list(range(start, start + size)))
        start += size
    return shards
