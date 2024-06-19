##############################################################################################################################################
# REFERENCE
# This is an adapted implementation of MUP, first introduced in:

# G. Yang, E. J. Hu, I. Babuschkin, S. Sidor, X. Liu, D. Farhi, N. Ryder, J. Pachocki, W. Chen, and J. Gao.,
# Tensor programs v: Tuning large neural networks via zero-shot hyperparameter transfer. arXiv preprint
# arXiv:2203.03466, 2022.

# @article{yang2022tensor,
#   title={Tensor programs v: Tuning large neural networks via zero-shot hyperparameter transfer},
#   author={Yang, Greg and Hu, Edward J and Babuschkin, Igor and Sidor, Szymon and Liu, Xiaodong and Farhi, David and Ryder, Nick and Pachocki, Jakub and Chen, Weizhu and Gao, Jianfeng},
#   journal={arXiv preprint arXiv:2203.03466},
#   year={2022}
# }
##############################################################################################################################################




from mup import MuReadout
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Dropout, LayerNorm, Linear, Parameter
from torch.nn.init import constant_


class MuReadoutModified(MuReadout):

    # modified by LaT_PFN from MUReadout 

    def _rescale_parameters(self):
        """
        Coordinate check shows that we shouldn't rescale the weights at initialization.
        """
        if hasattr(self, "_has_rescaled_params") and self._has_rescaled_params:
            raise RuntimeError(
                "`_rescale_parameters` has been called once before already"
            )
        if self.bias is not None:
            self.bias.data *= self.width_mult()
        self.weight.data *= self.width_mult()
        self._has_rescaled_params = True


class MupMultiheadAttention(nn.Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces."""

    __annotations__ = {
        "bias_k": torch._jit_internal.Optional[torch.Tensor],
        "bias_v": torch._jit_internal.Optional[torch.Tensor],
    }
    __constants__ = [
        "q_proj_weight",
        "k_proj_weight",
        "v_proj_weight",
        "in_proj_weight",
    ]

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        attn_mult=1,
        encoder_var=1,
        standparam=False,
        batch_first=True,
    ):
        super().__init__()
        assert batch_first, "batch non-first not implemented"
        self.batch_first = batch_first
        self.embed_dim = embed_dim
        self.attn_mult = attn_mult
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.standparam = standparam

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))
            self.register_parameter("in_proj_weight", None)
        else:
            self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))
            self.register_parameter("q_proj_weight", None)
            self.register_parameter("k_proj_weight", None)
            self.register_parameter("v_proj_weight", None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter("in_proj_bias", None)
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.init_method = init_method_normal((encoder_var / embed_dim) ** 0.5)

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            self.init_method(self.in_proj_weight)
            # zero initializing query head
            constant_(self.in_proj_weight[: self.embed_dim], 0.0)
        else:
            # zero initializing query head
            constant_(self.q_proj_weight, 0.0)
            self.init_method(self.k_proj_weight)
            self.init_method(self.v_proj_weight)

        self.init_method(self.out_proj.weight)
        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.0)
            constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            constant_(self.bias_k, 0.0)
        if self.bias_v is not None:
            constant_(self.bias_v, 0.0)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if "_qkv_same_embed_dim" not in state:
            state["_qkv_same_embed_dim"] = True

        super().__setstate__(state)

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
    ):
        if not self._qkv_same_embed_dim:
            # swapping pytorch's attn with ours
            return mup_multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight,
                k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
                attn_mult=self.attn_mult,
                standparam=self.standparam,
            )
        else:
            return mup_multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                attn_mult=self.attn_mult,
                standparam=self.standparam,
            )


def mup_multi_head_attention_forward(
    query,
    key,
    value,
    embed_dim_to_check,
    num_heads,
    in_proj_weight,
    in_proj_bias,
    bias_k,
    bias_v,
    add_zero_attn,
    dropout_p,
    out_proj_weight,
    out_proj_bias,
    training=True,
    key_padding_mask=None,
    need_weights=True,
    attn_mask=None,
    use_separate_proj_weight=False,
    q_proj_weight=None,
    k_proj_weight=None,
    v_proj_weight=None,
    static_k=None,
    static_v=None,
    attn_mult=1,
    standparam=False,
):
    query, key, value = (  # added for batch_first
        torch.permute(query, (1, 0, 2)),
        torch.permute(key, (1, 0, 2)),
        torch.permute(value, (1, 0, 2)),
    )

    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    assert key.size() == value.size()

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    if standparam:
        scaling = float(head_dim) ** -0.5 * math.sqrt(attn_mult)
    else:
        scaling = float(head_dim) ** -1 * math.sqrt(attn_mult)

    if not use_separate_proj_weight:
        if torch.equal(query, key) and torch.equal(key, value):
            # self-attention
            q, k, v = F.linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)

        elif torch.equal(key, value):
            # encoder-decoder attention
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)

            if key is None:
                assert value is None
                k = None
                v = None
            else:
                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = in_proj_bias
                _start = embed_dim
                _end = None
                _w = in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                k, v = F.linear(key, _w, _b).chunk(2, dim=-1)

        else:
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = F.linear(key, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = F.linear(value, _w, _b)
    else:
        q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)  # type: ignore
        len1, len2 = q_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == query.size(-1)

        k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)  # type: ignore
        len1, len2 = k_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == key.size(-1)

        v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)  # type: ignore
        len1, len2 = v_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == value.size(-1)

        if in_proj_bias is not None:
            q = F.linear(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim])
            k = F.linear(
                key, k_proj_weight_non_opt, in_proj_bias[embed_dim : (embed_dim * 2)]
            )
            v = F.linear(value, v_proj_weight_non_opt, in_proj_bias[(embed_dim * 2) :])
        else:
            q = F.linear(query, q_proj_weight_non_opt, in_proj_bias)
            k = F.linear(key, k_proj_weight_non_opt, in_proj_bias)
            v = F.linear(value, v_proj_weight_non_opt, in_proj_bias)

    q = q * scaling

    if attn_mask is not None:
        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
            if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                raise RuntimeError("The size of the 2D attn_mask is not correct.")
        elif attn_mask.dim() == 3:
            if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
                raise RuntimeError("The size of the 3D attn_mask is not correct.")
        else:
            raise RuntimeError(
                "attn_mask's dimension {} is not supported".format(attn_mask.dim())
            )
        # attn_mask's dim is 3 now.

    if bias_k is not None and bias_v is not None:
        if static_k is None and static_v is None:
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))
        else:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
    else:
        assert bias_k is None
        assert bias_v is None

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    if static_k is not None:
        assert static_k.size(0) == bsz * num_heads
        assert static_k.size(2) == head_dim
        k = static_k

    if static_v is not None:
        assert static_v.size(0) == bsz * num_heads
        assert static_v.size(2) == head_dim
        v = static_v

    src_len = k.size(1)

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len

    if add_zero_attn:
        src_len += 1
        k = torch.cat(
            [
                k,
                torch.zeros(
                    (k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device
                ),
            ],
            dim=1,
        )
        v = torch.cat(
            [
                v,
                torch.zeros(
                    (v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device
                ),
            ],
            dim=1,
        )
        if attn_mask is not None:
            attn_mask = F.pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, 1))

    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

    if attn_mask is not None:
        attn_output_weights += attn_mask

    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float("-inf"),
        )
        attn_output_weights = attn_output_weights.view(
            bsz * num_heads, tgt_len, src_len
        )

    attn_output_weights = F.softmax(attn_output_weights, dim=-1)
    attn_output_weights = F.dropout(attn_output_weights, p=dropout_p, training=training)

    attn_output = torch.bmm(attn_output_weights, v)
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_output = torch.permute(attn_output, (1, 0, 2))  # added for batch_first

        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        attn_output = torch.permute(attn_output, (1, 0, 2))  # added for batch_first

        return attn_output, None


class MupTransformerEncoderLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.
    """

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        encoder_var=1,
        attn_mult=1,
        bias=True,
        nlayers=1,
        standparam=False,
        batch_first=True,
    ):
        super().__init__()
        self.batch_first = batch_first
        self.attn_mult = attn_mult
        self.self_attn = MupMultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            attn_mult=attn_mult,
            bias=bias,
            add_bias_kv=bias,
            encoder_var=encoder_var,
            standparam=standparam,
            batch_first=batch_first,
        )
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias)
        self.dropout = Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias)

        self.norm1 = LayerNorm(d_model, elementwise_affine=True)
        self.norm2 = LayerNorm(d_model, elementwise_affine=True)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.init_method = init_method_normal((encoder_var / d_model) ** 0.5)

        self.nlayers = nlayers
        self.d_model = d_model
        self.reset_parameters()

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super().__setstate__(state)

    def reset_parameters(self):
        self.init_method(self.linear1.weight)
        self.init_method(self.linear2.weight)
        if self.linear1.bias is not None:
            constant_(self.linear1.bias, 0.0)
        if self.linear2.bias is not None:
            constant_(self.linear2.bias, 0.0)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """

        src1 = self.self_attn(
            src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]

        src = src + self.dropout1(src1)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class MupTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        encoder_var=1,
        attn_mult=1,
        bias=True,
        standparam=False,
        batch_first=True,
    ):
        super().__init__()
        self.attn_mult = attn_mult
        self.self_attn = MupMultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            attn_mult=attn_mult,
            bias=bias,
            add_bias_kv=bias,
            encoder_var=encoder_var,
            standparam=standparam,
            batch_first=batch_first,
        )
        self.multihead_attn = MupMultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            attn_mult=attn_mult,
            bias=bias,
            add_bias_kv=bias,
            encoder_var=encoder_var,
            standparam=standparam,
            batch_first=batch_first,
        )
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias)

        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=True)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=True)
        self.norm3 = nn.LayerNorm(d_model, elementwise_affine=True)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.init_method = init_method_normal((encoder_var / d_model) ** 0.5)

        self.reset_parameters()

    def reset_parameters(self):
        self.init_method(self.linear1.weight)
        self.init_method(self.linear2.weight)
        if self.linear1.bias is not None:
            nn.init.constant_(self.linear1.bias, 0.0)
        if self.linear2.bias is not None:
            nn.init.constant_(self.linear2.bias, 0.0)

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        tgt_is_causal=False,
        memory_is_causal=False,
    ):
        # if tgt.size(1) != self.d_model:
        #     tgt = tgt.transpose(0, 1)
        tgt2 = self.self_attn(
            tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.multihead_attn(
            tgt,
            memory,
            memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


def init_method_normal(sigma):
    # Define the init_method_normal function, assuming it's a standard normal initializer
    """Init method based on N(0, sigma)."""

    def init_(tensor):
        return nn.init.normal_(tensor, mean=0.0, std=sigma)

    return init_


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))
