from dataclasses import dataclass
from typing import Optional
import torch
import functools
import torch
from copy import deepcopy



class dotdict(dict):
    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        del self[name]


@dataclass
class ShapeConfig:

    n_sequence: int
    n_context: int

    n_features: Optional[int] = 1

    n_examples: Optional[int] = None
    n_heldout: Optional[int] = None

    n_prompt: Optional[int] = None
    n_history: Optional[int] = None

    def __post_init__(self):
        """
        Makes it possible to set n_examples or n_heldout and n_prompt or n_history
        """

        assert not (
            self.n_prompt is None and self.n_history is None
        ), "Either n_prompt or n_history must be set"

        if self.n_prompt is None:
            self.n_prompt = self.n_sequence - self.n_history

        if self.n_history is None:
            self.n_history = self.n_sequence - self.n_prompt

        assert (
            self.n_sequence == self.n_prompt + self.n_history
            and self.n_sequence > 0
            and self.n_history > 0
            and self.n_prompt > 0
        )

        assert not (
            self.n_examples is None and self.n_heldout is None
        ), "Either n_examples or n_heldout must be set"

        if self.n_examples is None:
            self.n_examples = self.n_context - self.n_heldout

        if self.n_heldout is None:
            self.n_heldout = self.n_context - self.n_examples

        assert (
            self.n_context == self.n_examples + self.n_heldout
            and self.n_context > 0
            and self.n_heldout > 0
            and self.n_examples > 0
        )

    def __dict__(self):
        return {
            "n_sequence": self.n_sequence,
            "n_context": self.n_context,
            "n_features": self.n_features,
            "n_examples": self.n_examples,
            "n_heldout": self.n_heldout,
            "n_prompt": self.n_prompt,
            "n_history": self.n_history,
        }

    @property
    def percent_heldout(self):
        return self.n_heldout / self.n_context

    @property
    def percent_examples(self):
        return self.n_examples / self.n_context

    @property
    def percent_prompt(self):
        return self.n_prompt / self.n_sequence

    @property
    def percent_history(self):
        return self.n_history / self.n_sequence


def dictify(to_dict):
    """
    Recursively converts a config to a dictionary
    """
    if isinstance(to_dict, (int, str, float, bool)):
        return to_dict
    elif isinstance(to_dict, (list, tuple)):
        return [dictify(x) for x in to_dict]
    elif isinstance(to_dict, (type(lambda x: x), functools.partial)):
        return to_dict.__name__ if hasattr(to_dict, "__name__") else str(to_dict)
    elif isinstance(to_dict, dict):
        return {k: dictify(v) for k, v in to_dict.items()}
    elif isinstance(to_dict, (type(None), torch.device)):
        return str(to_dict)
    elif isinstance(to_dict, ShapeConfig):
        return to_dict.__dict__()
    elif isinstance(to_dict, torch.Tensor):
        return to_dict.numpy().tolist()
    else:
        return dict(to_dict)


## Normalisations

unit_normalize = lambda t: (t - t.min()) / (t.max() - t.min())

normalize_T_to_abstract_2pi = lambda t: unit_normalize(t) * 2 * torch.pi
normalize_T_to_abstract_x = lambda t, x: unit_normalize(t) * x

normalize_V_mean_2std = lambda v: (v - v.mean(dim=1).unsqueeze(-1)) / (
    v.std(dim=1) * 2 + 1e-8
).unsqueeze(-1)

identity = lambda x: x

def V_normalize(V, mean=None, std=None):
    if mean is None:
        mean = V.mean(dim=1).unsqueeze(-1)
    if std is None:
        std = V.std(dim=1).unsqueeze(-1)
    return (V - mean) / (std * 2 + 1e-8), mean, std


def V_denormalize(V, mean, std):  # handy for comparing predictions in denomalized space
    
    return V * (std * 2 + 1e-8) + mean

def normalize(t, maximum, minimum):
    return unit_normalize(t) * (maximum - minimum) + minimum


def normalize_by_stepsize_and_max(t, step_size, dim, maximum):
    return (unit_normalize(t) - 1) * step_size * (t.shape[dim] - 1) + maximum

def config_deepcopy(x: dotdict):

    output = dotdict()

    for key, value in x.items():
        if isinstance(value, dotdict):
            output[key] = config_deepcopy(value)
        elif isinstance(value, ShapeConfig):
            output[key] = ShapeConfig(**value.__dict__())
        else:
            try:
                output[key] = deepcopy(value)
            except:
                output[key] = value
                print(f"Could not deepcopy {key} so using the original value.")

    return output
