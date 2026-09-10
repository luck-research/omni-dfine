"""
Omni-DFINE: Real-time Unified Vision-Language Prompting for Open-Vocabulary Detection
Minimal re-implementation of the three core modules introduced in the paper:

1. LSDPE    : Lightweight Spectral-Decoupled Visual Prompt Encoder
              (Analytical Spectral Decoupling + Cross-Frequency Modulation)
2. LW-VTFE  : Lightweight Orthogonal Vision-Text Feature Enhancer
              (Manifold Orthogonal Decomposition + Competitive Affinity Gating)
3. HHA-Loss : Hierarchical Harmonic Alignment loss
              (concentric harmonic decaying margins on the hypersphere)

"""

from .lsdpe import LSDPE
from .lw_vtfe import LWSVTFE  # backwards-friendly alias
from .lw_vtfe import LWVTFE
from .hha_loss import HHALoss

__all__ = ["LSDPE", "LWVTFE", "LWSVTFE", "HHALoss"]
__version__ = "0.1.0"
