import torch

def OneHot(tsr, n, axis=1):
    """
    Returns a one-hot encoded tensor with bool data type.

    Inputs
    tsr: torch.Tensor containing class indices. <N, H, W, [D]>
    n: Number of classes
    axis: Which axis to concatenate channels

    Output
    torch.Tensor <N, C, H, W, [D]>
    """
    return torch.cat([(tsr == i).unsqueeze(axis) for i in range(n)], axis)
