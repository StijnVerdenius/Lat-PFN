import numpy as np
import torch
import typing as T

from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal

def double_sampling(
    hyperprior_min: T.Union[int, float],
    hyperprior_max: T.Union[int, float],
    num_samples: int,
    device: str = "cpu",
    discrete: bool = False,
):
    """
    Double sampling of prior parameters
    Args:
    hyperprior_min: Minimum value for the hyperprior range
    hyperprior_max: Maximum value for the hyperprior range
    num_samples: Number of samples to generate
    device: Device to use
    discrete: If the distribution is discrete
    return: Samples from the distribution
    """

    context_ranges = (
        Uniform(hyperprior_min, hyperprior_max).sample([4]).to(device).view(2, 2)
    )

    point_estimates_a = Uniform(
        context_ranges[0].min(), context_ranges[0].max() + 1e-4
    ).sample([num_samples // 2])

    point_estimates_b = Uniform(
        context_ranges[1].min(), context_ranges[1].max() + 1e-4
    ).sample([num_samples // 2])

    if discrete:
        point_estimates_a = torch.round(point_estimates_a)
        point_estimates_b = torch.round(point_estimates_b)

    return torch.cat([point_estimates_a, point_estimates_b])


def triple_sampling(
    hyperprior_min: T.Union[int, float],
    hyperprior_max: T.Union[int, float],
    fixed_variance: T.Union[int, float],
    num_samples: int,
    device: str = "cpu",
):
    """
    Triple sampling of prior parameters
    Args:
    hyperprior_min: Minimum value for the hyperprior range
    hyperprior_max: Maximum value for the hyperprior range
    fixed_variance: Fixed variance for the distribution
    num_samples: Number of samples to generate
    device: Device to use
    return: Samples from the distribution
    """

    context_range = Uniform(hyperprior_min, hyperprior_max).sample([2]).to(device)

    sub_context_means = Uniform(context_range.min(), context_range.max() + 1e-4).sample(
        [2]
    )

    point_estimates_a = Normal(sub_context_means[0], fixed_variance).sample(
        [num_samples // 2]
    )

    point_estimates_b = Normal(sub_context_means[1], fixed_variance).sample(
        [num_samples // 2]
    )

    return torch.cat([point_estimates_a, point_estimates_b])


def noise_scale_sampling(num_samples: int, device: str = "cpu"):
    rand = np.random.rand()
    # very low noise
    if rand <= 0.6:
        noise = Uniform(0, 0.1).sample([num_samples])
    # moderate noise
    elif rand <= 0.9:
        noise = Uniform(0.2, 0.4).sample([num_samples])
    # high noise
    else:
        noise = Uniform(0.6, 0.8).sample([num_samples])

    return noise.to(device)


