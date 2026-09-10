"""
LW-VTFE: Lightweight Orthogonal Vision-Text Feature Enhancer
=============================================================

Implements Eq.(4) – Eq.(7) of the Omni-DFINE paper.

Given a text feature  F_t  in R^{N x C}  and a visual feature  F_v  in R^{N x C}:

    1. MOD  (Manifold Orthogonal Decomposition)
       S   = L2-normalize(F_t * F_v, dim = -1)          # shared semantic basis
       R_t = F_t - <F_t, S>_c * S                       # text private residual
       R_v = F_v - <F_v, S>_c * S                       # visual private residual

       <F, S>_c denotes the *channel-wise* dot product that contracts the
       channel axis and returns an N-vector. The orthogonal projection
       subtracts the shared-basis component from each modality so that
       (R_t, S) and (R_v, S) live on disjoint subspaces.

    2. CAG  (Competitive Affinity Gating)
       Q = (F_t + F_v) @ W_q                            # lightweight query
       [alpha_s, alpha_t, alpha_v] =
           Softmax( <Q, S>_c / tau,  <Q, R_t>_c / tau,  <Q, R_v>_c / tau )

       F_fused = alpha_s * S + alpha_t * R_t + alpha_v * R_v

       `tau` is a learnable scalar temperature (init 0.1) that sharpens the
       gating distribution.  The three subspaces compete for each token,
       which prevents the entangled representation that ordinary
       concatenation/sum would produce.

Notes
-----
* The dot product <., .>_c is performed along the channel axis, returning
  an N-vector; the multiplication `a_n * X_nxC` therefore broadcasts over
  channels per token.
* No dense N x N affinity matrix is constructed -- the cost is O(N * C).
* Parameter count: only the C x C projection matrix W_q plus a scalar
  temperature are introduced, matching the paper's claim that fusion
  parameters drop by ~70% versus standard cross-attention.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LWVTFE(nn.Module):
    """Lightweight Orthogonal Vision-Text Feature Enhancer.

    Parameters
    ----------
    embed_dim : int
        Channel dimension ``C`` of the input features.
    tau_init : float, default 0.1
        Initial value of the learnable temperature used in the softmax
        gating. The paper uses ``0.1``.
    """

    def __init__(self, embed_dim: int, tau_init: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        # Lightweight query projection W_q in R^{C x C}.
        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        # Learnable temperature, registered as a parameter so it is updated
        # by the optimizer. We keep it positive via `abs` in `forward`.
        self.tau = nn.Parameter(torch.tensor(float(tau_init)))

        nn.init.xavier_uniform_(self.query_proj.weight)

    @staticmethod
    def _channel_dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """<a, b>_c -- contract channel axis, broadcast over tokens.

        Both inputs are (N, C). Output is (N,).
        """
        return (a * b).sum(dim=-1)

    def forward(self, f_t: torch.Tensor, f_v: torch.Tensor) -> torch.Tensor:
        """Fuse text and visual features.

        Parameters
        ----------
        f_t, f_v : Tensor
            Text and visual features, each of shape ``(N, C)`` (or
            ``(B, N, C)``).

        Returns
        -------
        Tensor
            Fused features with the same shape as the inputs.
        """
        single = False
        if f_t.dim() == 2:
            f_t = f_t.unsqueeze(0)            # (1, N, C)
            f_v = f_v.unsqueeze(0)
            single = True
        assert f_t.dim() == 3 and f_t.shape == f_v.shape, (
            "LW-VTFE expects matching (B, N, C) or (N, C) inputs"
        )

        # ------------------------------------------------------------------ #
        # MOD: shared basis + orthogonal residuals                            #
        # ------------------------------------------------------------------ #
        # Eq. (4):  S = Norm(F_t * F_v)  -- channel-wise L2 normalization
        s_raw = f_t * f_v
        s = F.normalize(s_raw, p=2, dim=-1)

        # Eq. (5):  R_t = F_t - <F_t, S>_c * S ; R_v likewise.
        # <., .>_c returns (B, N); expand last axis to broadcast with (B,N,C).
        dot_t = self._channel_dot(f_t, s).unsqueeze(-1)   # (B, N, 1)
        dot_v = self._channel_dot(f_v, s).unsqueeze(-1)
        r_t = f_t - dot_t * s
        r_v = f_v - dot_v * s

        # ------------------------------------------------------------------ #
        # CAG: competitive softmax gating                                     #
        # ------------------------------------------------------------------ #
        # Eq. (6):  Q = (F_t + F_v) @ W_q
        q = self.query_proj(f_t + f_v)                    # (B, N, C)

        # Positive temperature (paper uses scalar tau; we keep it learnable).
        tau = self.tau.abs().clamp(min=1e-3)

        # Channel dot products with each subspace -> (B, N) each.
        score_s = self._channel_dot(q, s) / tau
        score_t = self._channel_dot(q, r_t) / tau
        score_v = self._channel_dot(q, r_v) / tau

        # Stack and softmax over the three-subspace axis -> (B, N, 3).
        scores = torch.stack([score_s, score_t, score_v], dim=-1)
        alpha = F.softmax(scores, dim=-1)                  # (B, N, 3)

        alpha_s = alpha[..., 0:1]                          # (B, N, 1)
        alpha_t = alpha[..., 1:2]
        alpha_v = alpha[..., 2:3]

        # Eq. (7):  F_fused = alpha_s * S + alpha_t * R_t + alpha_v * R_v
        fused = alpha_s * s + alpha_t * r_t + alpha_v * r_v

        return fused.squeeze(0) if single else fused

    def extra_repr(self) -> str:
        return f"embed_dim={self.embed_dim}, tau_init={self.tau.item():.3f}"


# Convenience alias -- some codebases spell the module name with an extra "S".
LWSVTFE = LWVTFE
