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
    return "cuda:" + str(device.index or 0)
