import torch

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        print(f"Using {device_count} {device_name} GPU")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device
