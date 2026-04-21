"""Clean-room re-implementation of Wang et al. 2025 Conditional Neural ODE (CNODE).

Re-implemented independently from the paper:
  Wang, X., Zhao, Y., Han, K., Luo, X., van Rooij, S., Stevens, J., He, L.,
  Zhan, L., Sun, Y., Wang, W., Yang, C. (2025).
  "Conditional Neural ODE for Longitudinal Parkinson's Disease Progression
  Forecasting." arXiv:2511.04789.

No source from any upstream repository was consulted. See CLEAN_ROOM_NOTES.md
for assumptions made when the paper under-specifies details.
"""
from wang_2025_cnode_ppmi.cnode import CNODE, CNODEConfig

__all__ = ["CNODE", "CNODEConfig"]
