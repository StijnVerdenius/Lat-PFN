import torch.nn as nn
import torch

import mup
from model.mup_imp.mup_custom import (
    MupTransformerEncoderLayer,
    MupTransformerDecoderLayer,
    MupMultiheadAttention,
)


def expand_diagonal(original_diag: int, entries_per_block: int) -> torch.Tensor:
    """
    Expands a diagonal tensor based on the original diagonal size and the number of entries per block.

    Parameters:
    - original_diag: The size of the original diagonal.
    - entries_per_block: The number of times each entry in the diagonal is repeated in both dimensions.

    Returns:
    - An expanded torch.Tensor.
    """
    # Create an identity matrix based on the original diagonal size.
    identity_matrix = torch.eye(original_diag)

    # Expand each element in the identity matrix 'entries_per_block' times in both dimensions.
    expanded_matrix = identity_matrix.repeat_interleave(
        entries_per_block, dim=0
    ).repeat_interleave(entries_per_block, dim=1)

    return expanded_matrix


def diagonal_mask(original_diag: int, entries_per_block: int) -> torch.Tensor:
    return torch.eye(original_diag * entries_per_block)


def causal_expanded_mask(original_diag: int, entries_per_block: int) -> torch.Tensor:
    """
    Creates a causal mask for an expanded diagonal tensor.

    Parameters:
    - original_diag: The size of the original diagonal.
    - entries_per_block: The number of times each entry in the diagonal is repeated in both dimensions.

    Returns:
    - A causal mask for the expanded diagonal tensor.
    """
    # Create a diagonal mask based on the original diagonal size.
    mask = diagonal_mask(original_diag, entries_per_block)

    # Create a causal mask by setting the upper triangular part of the mask to -inf.
    causal_mask = torch.ones_like(mask) - torch.triu(torch.ones_like(mask), diagonal=1)

    # multiply with the expanded mask to get the causal mask for the expanded tensor
    causal_mask = expand_diagonal(original_diag, entries_per_block) * causal_mask

    return causal_mask


masking = dict(
    per_series_causal=causal_expanded_mask,
    per_series=expand_diagonal,
    independent=diagonal_mask,
)


class BasePFN(nn.Module):
    """
    Class implements a simple prior-data fitting network (PFN) that makes no assumptions about the data.
    The PFN is a transformer that takes as input a batch of example pairs (x, y), features to predict for x*
    and outputs a batch of predictions y^ for each x*.
    """

    def __init__(
        self,
        d_model=8,
        d_ff=32,
        nhead=4,
        num_layers=2,
        dropout=0.1,
        use_mup_parametrization: bool = True,
        masking_type="independent",
    ):
        super().__init__()

        encoder_layer, decoder_layer = (
            (
                MupTransformerEncoderLayer
                if use_mup_parametrization
                else nn.TransformerEncoderLayer
            ),
            (
                MupTransformerDecoderLayer
                if use_mup_parametrization
                else nn.TransformerDecoderLayer
            ),
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_ff,
                dropout=dropout,
                activation="relu",
                batch_first=True,
            ),
            num_layers=num_layers,
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_ff,
                dropout=dropout,
                activation="relu",
                batch_first=True,
            ),
            num_layers=num_layers,
        )

        self.masks = set()
        self.masking_type = masking_type

    def forward(self, D, x_star):
        """
        Forward pass of the PFN.

        Parameters:
        - D: The PFN-context
        - x_star: The features / prompts to predict for

        """

        # 1. encode D
        D = self.encoder(D)

        a, b, c, d = x_star.shape

        # 2. decode x_star
        y_hat = self.decoder(
            x_star.flatten(1, 2), D, tgt_mask=self.get_mask((b, c), x_star.device)
        )

        return y_hat.view(a, b, c, d)

    def get_mask(self, mask_dim, device):
        key = f"mask_{mask_dim}"
        if key not in self.masks:
            with torch.no_grad():
                mask = masking[self.masking_type](*mask_dim).float().to(device)
                mask.requires_grad = False
                mask = mask.masked_fill(mask == 0, float("-inf")).masked_fill(
                    mask == 1, float(0.0)
                )
                self.register_buffer(key, mask)
                self.masks.add(key)
                # todo: protect against memory leak
        return getattr(self, key)


class AttentionAVGPool(nn.Module):
    """
    AttentionAVGPool is a module that computes a weighted average of the input
    using a query vector. The query vector is learned during training.
    """

    def __init__(self, d_model=128, nhead=4, dropout=0.1, use_mup_parametrization=True):
        super().__init__()
        self.query = nn.Embedding(3, d_model)
        self.embedding_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        attn_clss = (
            MupMultiheadAttention if use_mup_parametrization else nn.MultiheadAttention
        )
        self.attn = attn_clss(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.output = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        self.d_model = d_model

    def reinitialization(self, use_mup_parametrization: bool):
        init = mup.init if use_mup_parametrization else torch.nn.init
        for module in [self.query, self.embedding_layer, self.output, self.attn]:
            if hasattr(module, "reinitialization"):
                module.reinitialization(use_mup_parametrization)
            else:
                for name, param in module.named_parameters():
                    if "weight" in name and param.dim() > 1:
                        init.kaiming_normal_(param)

    def forward(self, x):

        # save batch size
        bs = x.shape[0]

        # flatten batch id needed
        if len(x.shape) == 4:
            x = x.view(-1, x.shape[-2], x.shape[-1])

        # compute attention
        x = self.embedding_layer(x)
        query = self.query((torch.arange(3, device=x.device).long())).repeat(
            x.shape[0], 1, 1
        )

        # avg pool and head
        x, _ = self.attn(query, x, x, need_weights=True)
        x = self.output(x.reshape(-1, 3 * self.d_model))

        # reshape if needed
        if x.shape[0] != bs:
            x = x.view(bs, -1, x.shape[-1])
        return x
