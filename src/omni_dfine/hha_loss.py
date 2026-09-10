"""
HHA Loss: Hierarchical Harmonic Alignment
==========================================

Implements Eq.(8) and Eq.(9) of the Omni-DFINE paper.

Geometric idea
--------------
Standard contrastive losses (InfoNCE) push *all* negatives apart with the
same margin, collapsing the hierarchical nature of visual semantics on the
hypersphere.  HHA instead assigns each negative a margin that decays
harmonically with its *ranked* similarity to the anchor:

    Sort the negative similarities s_k = z_i . z-_k  in descending order,
    s_(1) >= s_(2) >= ... >= s_(K).  Hard negatives (large s) receive the
    largest margin m_(1); easier negatives live in wider shells with
    smaller margins.

        m(j) = lambda_ref / H_j ,   H_j = sum_{n=1}^{j} 1/n    (Eq. 8)

        L_HHA = - log( exp(s+/eta)
                       / ( exp(s+/eta) + sum_j exp( (s_(j) + m(j)) / eta ) ) )
                                                            (Eq. 9)

    Defaults:  eta = 0.07 ,  lambda_ref = 0.3   (from the paper).

The harmonic denominator H_j grows slowly with rank, so margins decay
smoothly -- hard negatives are forced to be far from the anchor on the
hypersphere while easier negatives are allowed broader shells, preventing
the global manifold collapse that plain InfoNCE suffers from.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _harmonic_numbers(j_max: int, device, dtype) -> torch.Tensor:
    """Return H_1, H_2, ..., H_{j_max} as a 1-D tensor of length j_max.

    H_j = 1 + 1/2 + ... + 1/j.
    """
    n = torch.arange(1, j_max + 1, device=device, dtype=dtype)
    return torch.cumsum(1.0 / n, dim=0)


class HHALoss(nn.Module):
    """Hierarchical Harmonic Alignment loss.

    Parameters
    ----------
    eta : float, default 0.07
        Temperature ``eta`` in Eq.(9).
    lambda_ref : float, default 0.3
        Scaling factor ``lambda_ref`` controlling the initial margin width.
        The paper finds ``lambda_ref = 0.3`` optimal for Omni-DFINE-S.
    reduction : {'mean', 'sum', 'none'}, default 'mean'
        Reduction strategy across the batch dimension.
    """

    def __init__(
        self,
        eta: float = 0.07,
        lambda_ref: float = 0.3,
        reduction: str = "mean",
    ):
        super().__init__()
        assert reduction in {"mean", "sum", "none"}
        self.eta = float(eta)
        self.lambda_ref = float(lambda_ref)
        self.reduction = reduction

    @staticmethod
    def _l2_normalize(x: torch.Tensor) -> torch.Tensor:
        """L2-normalize along the last dimension."""
        return F.normalize(x, p=2, dim=-1, eps=1e-12)

    def forward(
        self,
        image_emb: torch.Tensor,
        pos_prompt_emb: torch.Tensor,
        neg_prompt_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the HHA loss for a batch of (anchor, positive, negatives).

        Parameters
        ----------
        image_emb : Tensor
            Anchor image embeddings, shape ``(B, D)``. They will be
            L2-normalized inside the loss.
        pos_prompt_emb : Tensor
            Positive prompt embeddings matched to each anchor, ``(B, D)``.
        neg_prompt_emb : Tensor
            Negative prompt embeddings, shape ``(B, K, D)`` -- K negatives
            per anchor. They will be L2-normalized inside the loss.

        Returns
        -------
        Tensor
            Scalar loss when ``reduction`` is 'mean' or 'sum'; otherwise a
            (B,) tensor of per-anchor losses.
        """
        assert image_emb.dim() == 2, "image_emb must be (B, D)"
        assert pos_prompt_emb.dim() == 2 and pos_prompt_emb.shape == image_emb.shape
        assert neg_prompt_emb.dim() == 3, "neg_prompt_emb must be (B, K, D)"
        assert neg_prompt_emb.shape[0] == image_emb.shape[0]
        assert neg_prompt_emb.shape[2] == image_emb.shape[1]
        B, K, _ = neg_prompt_emb.shape

        # L2-normalize embeddings (Eq.(9) is defined on the unit hypersphere).
        z_i = self._l2_normalize(image_emb)               # (B, D)
        z_p = self._l2_normalize(pos_prompt_emb)          # (B, D)
        z_n = self._l2_normalize(neg_prompt_emb)          # (B, K, D)

        # Positive similarity: s+ = z_i . z+_i -> (B,)
        s_pos = (z_i * z_p).sum(dim=-1)                   # (B,)

        # Negative similarities s_k = z_i . z-_k -> (B, K).
        s_neg = torch.bmm(z_n, z_i.unsqueeze(-1)).squeeze(-1)   # (B, K)

        # Sort negatives in *descending* order (hardest first).
        s_sorted, _ = torch.sort(s_neg, dim=1, descending=True)  # (B, K)

        # Harmonic margins:  m(j) = lambda_ref / H_j , j = 1..K.
        H = _harmonic_numbers(K, s_sorted.device, s_sorted.dtype)  # (K,)
        margins = self.lambda_ref / H                              # (K,)

        # Apply temperature and add margins to the sorted negatives only.
        # (Adding margins *before* dividing by eta would just rescale, but
        #  the paper writes the margin inside the exponent -- Eq.(9) -- so
        #  we follow that exactly.)
        logits_pos = s_pos / self.eta                       # (B,)
        logits_neg = (s_sorted + margins) / self.eta        # (B, K)

        # Numerator: exp(s+ / eta).  Denominator: exp(s+/eta) + sum_neg exp.
        # Use log-sum-exp form for numerical stability.
        log_denom = torch.logsumexp(
            torch.cat([logits_pos.unsqueeze(1), logits_neg], dim=1),
            dim=1,
        )                                                  # (B,)
        loss = -(logits_pos - log_denom)                    # (B,)

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss                                        # (B,)

    def extra_repr(self) -> str:
        return (
            f"eta={self.eta}, lambda_ref={self.lambda_ref}, "
            f"reduction={self.reduction}"
        )
