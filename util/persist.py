import torch
import gc
import os


def load_model(path, model, device):
    state_dict = torch.load(path)["state_dict"]
    remove_prefix = "module"
    state_dict = {
        k[len(remove_prefix) :]: v
        for k, v in state_dict.items()
        if not ("mask" in k and "pfn" in k)
    }

    model.load_state_dict(state_dict)

    model.to(device)

    model.eval()

    return model


def flush_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
