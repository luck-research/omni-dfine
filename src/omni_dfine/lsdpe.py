"""
LSDPE: Lightweight Spectral-Decoupled Visual Prompt Encoder
============================================================

Implements Eq.(1) – Eq.(3) of the Omni-DFINE paper.

Pipeline
--------
For an input visual-prompt feature map  X_v in R^{C x H x W}:

    1. ASD  (Analytical Spectral Decoupling, parameter-free)
       X_freq_v = DCT2(X_v)               # 2-D Discrete Cosine Transform
       rho*_c   = min { rho |
                       sum_{r=0}^{rho} ||X_freq_v(c, r)||_2^2
                       >=  alpha * sum_{r=0}^{Rmax} ||X_freq_v(c, r)||_2^2 }
       M_low(c,h,w) = 1 if radial_freq(h,w) <= rho*_c  else 0
       X_freq_low = X_freq_v * M_low
       X_freq_high = X_freq_v * (1 - M_low)
       X_low  = IDCT2(X_freq_low)
       X_high = IDCT2(X_freq_high)

       The "radial frequency radius" r of cell (h, w) is the integer
       Euclidean distance from the DC component (top-left corner of the
       2-D DCT output): r = round(sqrt(h^2 + w^2)).

    2. CFM  (Cross-Frequency Modulation)
       X_hat_v = X_low  * sigmoid(GAP(X_high))
               + X_high * sigmoid(GAP(X_low ))

       GAP = Global Average Pooling over (H, W).  The sigmoid gates are
       broadcast channel-wise, so low-frequency structure is modulated
       by global high-frequency evidence and vice-versa.

Key design choices that keep this implementation "minimal":
    * Pure-PyTorch orthonormal DCT-II (no scipy / cv2 dependency).
    * Energy threshold `alpha` defaults to 0.9 (the paper's optimum).
    * The energy sum is computed per-channel over the radial frequency
      index, with the bins ordered 0..Rmax along the *flattened* spectrum.
    * Fully `torch.jit.script`-able when needed; runs on CPU and CUDA.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Pure-PyTorch 2-D DCT-II / IDCT-II                                            #
# --------------------------------------------------------------------------- #
def _build_dct_matrix(N: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return the orthonormal DCT-II transform matrix D in R^{N x N}.

    D @ x performs the 1-D DCT-II along the last dimension of x.
    D.T @ x performs the corresponding inverse DCT (D is orthonormal, so
    D^{-1} = D.T).
    """
    n = torch.arange(N, device=device, dtype=dtype).view(-1, 1)   # (N,1)
    k = torch.arange(N, device=device, dtype=dtype).view(1, -1)   # (1,N)
    m = math.sqrt(2.0 / N) * torch.cos(math.pi / N * (n + 0.5) * k)
    m[0, :] = 1.0 / math.sqrt(N)            # k=0 row uses 1/sqrt(N)
    return m


def _dct2(x: torch.Tensor) -> torch.Tensor:
    """2-D orthonormal DCT-II on tensors of shape (..., H, W)."""
    H, W = x.shape[-2], x.shape[-1]
    d_h = _build_dct_matrix(H, x.device, x.dtype)
    d_w = _build_dct_matrix(W, x.device, x.dtype)
    # Apply along H (left), then along W (right).
    y = d_h @ x          # contracts the H axis
    y = y @ d_w.t()      # contracts the W axis
    return y


def _idct2(x: torch.Tensor) -> torch.Tensor:
    """Inverse of `_dct2` (D is orthonormal so just transpose)."""
    H, W = x.shape[-2], x.shape[-1]
    d_h = _build_dct_matrix(H, x.device, x.dtype)
    d_w = _build_dct_matrix(W, x.device, x.dtype)
    y = d_h.t() @ x      # inverse along H
    y = y @ d_w          # inverse along W
    return y


# --------------------------------------------------------------------------- #
# LSDPE                                                                        #
# --------------------------------------------------------------------------- #
class LSDPE(nn.Module):
    """Lightweight Spectral-Decoupled Visual Prompt Encoder.

    Parameters
    ----------
    alpha : float, default 0.9
        Dominant energy fraction used to compute the analytical cutoff
        frequency rho*_c (Eq. 1 of the paper). The paper reports that
        ``alpha = 0.9`` yields the best noise/contour trade-off.
    """

    def __init__(self, alpha: float = 0.9):
        super().__init__()
        # `alpha` is registered as buffer so it follows `.to(device)` etc.
        self.register_buffer("alpha", torch.tensor(float(alpha)))

    # -- analytical spectral decoupling (Eq. 1, Eq. 2) ---------------------- #
    @staticmethod
    def _radial_index(H: int, W: int, device, dtype) -> torch.Tensor:
        """Integer radial frequency index r(h, w) = floor(sqrt(h^2 + w^2)).

        The DC component lives at (0, 0); for DCT-II the radial frequency
        increases away from the origin. The maximum radial index is
        Rmax = floor(sqrt((H-1)^2 + (W-1)^2)).
        """
        ys = torch.arange(H, device=device, dtype=dtype).view(H, 1)
        xs = torch.arange(W, device=device, dtype=dtype).view(1, W)
        return torch.floor(torch.sqrt(ys * ys + xs * xs)).to(torch.long)

    def _asdm(self, x_freq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ASD: return (X_low, X_high, mask_low) given a frequency tensor."""
        C, H, W = x_freq.shape
        r_idx = self._radial_index(H, W, x_freq.device, x_freq.dtype)
        Rmax = int(r_idx.max().item())

        # Energy per (channel, radial_bin) -- sum |F|^2 over all cells
        # whose radial index equals r.
        # E[c, r] = sum_{(h,w): r(h,w) == r} |x_freq[c, h, w]|^2
        power = (x_freq * x_freq).reshape(C, -1)            # (C, H*W)
        r_flat = r_idx.reshape(-1)                          # (H*W,)
        E = torch.zeros(C, Rmax + 1, dtype=x_freq.dtype, device=x_freq.device)
        E.index_put_(
            (torch.arange(C, device=x_freq.device).unsqueeze(1).expand(C, H * W),
             r_flat.unsqueeze(0).expand(C, H * W)),
            power,
            accumulate=True,
        )

        # Cumulative energy and total energy per channel.
        cumE = torch.cumsum(E, dim=1)                        # (C, Rmax+1)
        total = cumE[:, -1].clamp(min=1e-12).unsqueeze(1)    # (C, 1)
        # rho*_c = min r s.t. cumE[c, r] >= alpha * total[c]
        threshold = self.alpha * total                       # (C, 1)
        # Boolean mask: True where the cumulative energy has reached the
        # threshold. The first True column is the cutoff index per channel.
        reached = cumE >= threshold                          # (C, Rmax+1)
        # If, for a channel, the threshold is reached only at Rmax (or never
        # numerically), `argmax` returns 0; we always fall back to Rmax in
        # that case to avoid losing the high-frequency band entirely.
        first_hit = reached.float().argmax(dim=1)            # (C,)
        any_hit = reached.any(dim=1)                         # (C,)
        first_hit = torch.where(any_hit, first_hit, torch.full_like(first_hit, Rmax))

        # Build the per-channel binary low-pass mask M_low in (C, H, W).
        # mask[c, h, w] = 1  iff  r(h, w) <= first_hit[c]
        rho_star = first_hit.view(C, 1, 1).to(r_idx.dtype)  # (C, 1, 1)
        mask_low = (r_idx.unsqueeze(0) <= rho_star).to(x_freq.dtype)  # (C, H, W)

        x_freq_low = x_freq * mask_low
        x_freq_high = x_freq * (1.0 - mask_low)
        return x_freq_low, x_freq_high, mask_low

    # -- forward ------------------------------------------------------------- #
    def forward(self, x_v: torch.Tensor) -> torch.Tensor:
        """Run LSDPE on a visual-prompt feature map.

        Parameters
        ----------
        x_v : Tensor
            Visual prompt features, shape ``(B, C, H, W)`` or ``(C, H, W)``.

        Returns
        -------
        Tensor
            Refined visual-prompt features with the same shape as ``x_v``.
        """
        single = False
        if x_v.dim() == 3:
            x_v = x_v.unsqueeze(0)            # (1, C, H, W)
            single = True
        assert x_v.dim() == 4, "LSDPE expects (B, C, H, W) or (C, H, W)"
        B, C, H, W = x_v.shape

        # Reshape to (B*C, H, W) so the DCT acts on each sample/channel pair.
        x_flat = x_v.reshape(B * C, H, W)

        x_freq = _dct2(x_flat)                # Eq. (2) -- frequency domain

        x_freq_low, x_freq_high, _ = self._asdm(x_freq)   # ASD, Eq. (1) + Eq. (2)
        x_low = _idct2(x_freq_low)            # back to spatial domain
        x_high = _idct2(x_freq_high)

        # CFM -- Eq. (3)
        # GAP over the spatial dims gives a per-channel scalar gate.
        gap_low = x_low.mean(dim=(-2, -1), keepdim=True)     # (B*C, 1, 1)
        gap_high = x_high.mean(dim=(-2, -1), keepdim=True)
        x_hat = x_low * torch.sigmoid(gap_high) + x_high * torch.sigmoid(gap_low)

        x_hat = x_hat.reshape(B, C, H, W)
        return x_hat.squeeze(0) if single else x_hat

    def extra_repr(self) -> str:
        return f"alpha={self.alpha.item():.2f}"
