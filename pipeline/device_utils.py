import torch


def resolve_device(preferred: str | None = None) -> torch.device:
    if preferred:
        device = torch.device(preferred)
        if device.type == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


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
