"""
GIMIN training subpackage.

Provides the composite loss function, iterative graph refinement logic,
and the main training loop for the GIMIN framework.

Modules:
    losses: Composite loss combining reconstruction, distribution, and
        cross-modal consistency objectives.
    graph_refinement: Iterative graph rebuilding with blended features.
    trainer: Main training loop with masked-value self-supervision.
"""

from gimin.training.graph_refinement import IterativeGraphRefiner
from gimin.training.losses import GIMINLoss
from gimin.training.trainer import GIMINTrainer

__all__ = [
    "GIMINLoss",
    "IterativeGraphRefiner",
    "GIMINTrainer",
]
