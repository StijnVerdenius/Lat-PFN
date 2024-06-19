from torch import Tensor
import torch.nn as nn
import torch

class CrossEntropyLoss(nn.Module):
    """
    Cross entropy loss with binning and label smoothing
    """

    def __init__(
        self,
        reduction="mean",
        n_bins=100,
        label_smoothing=0.1,
        range_min=-3.5,
        range_max=3.5,
        device="cpu",
        **kwargs,
    ):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(
            reduction=reduction, label_smoothing=label_smoothing
        )
        print("CrossEntropyLoss", "ignores", kwargs)
        assert n_bins > 2, f"n_bins must be greater than 2, got {n_bins}"
        self.n_bins = n_bins
        self.range_min = range_min
        self.range_max = range_max
        bin_area = (range_max - range_min) / (n_bins)
        bin_offset = bin_area / 2
        self.bin_centroids = torch.linspace(
            range_min + bin_offset, range_max - bin_offset, n_bins, device=device
        )
        self.bin_borders = torch.linspace(
            range_min + bin_area, range_max - bin_area, n_bins - 1, device=device
        )

    @property
    def n_outputs(self):
        return self.n_bins

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return self.loss(input.permute(0, -1, 1, 2), target)

    def prepare_target(self, target: Tensor) -> Tensor:
        with torch.no_grad():
            x = torch.bucketize(target, self.bin_borders).long()
        return x.squeeze()

    def prepare_output(self, output: Tensor) -> dict:
        return dict(input=output.squeeze())

    def output_to_mean(self, output: dict) -> Tensor:
        output = output["input"]
        probs = torch.nn.functional.softmax(output, dim=-1)
        mean = (probs * self.bin_centroids).sum(-1)
        return mean

    def output_to_std(self, output: dict) -> Tensor:
        mean = self.output_to_mean(output)
        output = output["input"]
        probs = torch.nn.functional.softmax(output, dim=-1)
        return torch.sqrt(
            (probs * (self.bin_centroids - mean.unsqueeze(-1)).pow(2)).sum(-1)
        )
