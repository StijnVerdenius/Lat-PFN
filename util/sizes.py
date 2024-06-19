import torch


def randomly_divide_shape(total: int, choose: int, device: str = "cpu"):
    random_idxs = torch.randperm(total, device=device)
    heldout, main = random_idxs[:choose], random_idxs[choose:]
    return main, heldout



