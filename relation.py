"""PyTorch implementation of Relation.

Relation shares a single sequence-level context vector with every token while
remaining linear in sequence length. The implementation follows Equations
(7)-(12) of Yamada, Baladram, and Lin (2022).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

__all__ = ["Relation"]
__version__ = "0.1.0"


class Relation(nn.Module):
    """Process sequence context without an ``N x N`` attention matrix.

    For an input ``x`` with shape ``(..., N, in_features)``, the layer computes

    ``G = W_G(x)``, ``H = W_H(x)``, and
    ``activation(W(G * mean(H, sequence)))``.

    Args:
        in_features: Size of each input token vector.
        out_features: Relation depth. If omitted, ``in_features`` is used.
        bias: Add bias to the ``G`` and ``H`` projections. The Relation output
            projection remains bias-free, matching the published code.
        activation: Output activation. The default is ``nn.ReLU()``. Pass an
            ``nn.Identity()`` to obtain the pre-activation output.

    The optional ``mask`` passed to :meth:`forward` must have shape ``(..., N)``.
    ``True`` marks a valid token. Masked tokens are excluded from the context
    mean. An entirely masked sequence produces a finite zero output regardless
    of the output activation.
    """

    __constants__ = ["in_features", "out_features"]

    def __init__(
        self,
        in_features: int,
        out_features: int | None = None,
        *,
        bias: bool = True,
        activation: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(in_features, int) or isinstance(in_features, bool):
            raise TypeError("in_features must be an integer")
        if in_features <= 0:
            raise ValueError("in_features must be positive")
        if out_features is None:
            out_features = in_features
        if not isinstance(out_features, int) or isinstance(out_features, bool):
            raise TypeError("out_features must be an integer or None")
        if out_features <= 0:
            raise ValueError("out_features must be positive")
        if activation is not None and not isinstance(activation, nn.Module):
            raise TypeError("activation must be a torch.nn.Module or None")

        self.in_features = in_features
        self.out_features = out_features
        self.g_projection = nn.Linear(in_features, out_features, bias=bias)
        self.h_projection = nn.Linear(in_features, out_features, bias=bias)
        self.relation_projection = nn.Linear(
            out_features, out_features, bias=False
        )
        self.activation = nn.ReLU() if activation is None else activation
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weights as specified in the paper's experiments."""

        for layer in (
            self.g_projection,
            self.h_projection,
            self.relation_projection,
        ):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    @staticmethod
    def _sequence_mean(h: Tensor, mask: Tensor | None) -> Tensor:
        if mask is None:
            return h.mean(dim=-2, keepdim=True)

        expected_shape = h.shape[:-1]
        if mask.shape != expected_shape:
            raise ValueError(
                "mask must match the input's leading and sequence dimensions; "
                f"expected {tuple(expected_shape)}, received {tuple(mask.shape)}"
            )

        valid = mask.to(device=h.device, dtype=torch.bool).unsqueeze(-1)
        masked_h = torch.where(valid, h, torch.zeros((), dtype=h.dtype, device=h.device))
        accumulation_dtype = (
            torch.float32 if h.dtype in (torch.float16, torch.bfloat16) else h.dtype
        )
        numerator = masked_h.sum(dim=-2, keepdim=True, dtype=accumulation_dtype)
        denominator = valid.sum(dim=-2, keepdim=True).to(dtype=accumulation_dtype)
        return (numerator / denominator.clamp_min(1)).to(dtype=h.dtype)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """Apply Relation to ``x``.

        Args:
            x: Tensor shaped ``(..., sequence_length, in_features)``.
            mask: Optional boolean or 0/1 tensor shaped ``(..., sequence_length)``.
                Nonzero values mark valid tokens.
        """

        if x.ndim < 2:
            raise ValueError("x must have at least sequence and feature dimensions")
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"expected input feature size {self.in_features}, received {x.shape[-1]}"
            )
        if x.shape[-2] == 0:
            raise ValueError("sequence length must be positive")

        g = self.g_projection(x)
        h = self.h_projection(x)
        context = self._sequence_mean(h, mask)
        output = self.activation(self.relation_projection(g * context))
        if mask is not None:
            # Apply after activation, since sigmoid(0) and softplus(0) are nonzero.
            has_valid = mask.to(device=output.device, dtype=torch.bool).any(
                dim=-1, keepdim=True
            )
            output = output.masked_fill(~has_valid.unsqueeze(-1), 0)
        return output

    def extra_repr(self) -> str:
        bias = self.g_projection.bias is not None
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={bias}, activation={self.activation.__class__.__name__}"
        )
