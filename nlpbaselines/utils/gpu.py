import torch


def select_device():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return device
