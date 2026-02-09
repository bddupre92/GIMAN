"""
GIMIN evaluation subpackage.

Provides imputation quality metrics, baseline comparisons, masked-value
reconstruction experiments, downstream task evaluation, advanced baselines,
and publication-quality visualization.

Modules:
    metrics: Scalar and distributional metrics for imputation quality.
    baselines: Wrapper classes for standard imputation baselines (MICE,
        kNN, mean, median).
    advanced_baselines: SOTA deep/ML baselines (MissForest, GAIN, SoftImpute).
    masked_experiment: Controlled masked-value reconstruction benchmark.
    downstream: Evaluation of imputed data on downstream SAA classification.
    visualization: Publication-quality figure generation (IEEE style).
"""

from gimin.evaluation.advanced_baselines import (
    GAINBaseline,
    MissForestBaseline,
    SoftImputeBaseline,
)
from gimin.evaluation.baselines import (
    KNNBaseline,
    MeanBaseline,
    MedianBaseline,
    MICEBaseline,
)
from gimin.evaluation.downstream import DownstreamEvaluator
from gimin.evaluation.masked_experiment import MaskedValueExperiment
from gimin.evaluation.metrics import (
    calibration_metrics,
    expected_calibration_error,
    ks_test_per_feature,
    mae,
    nrmse,
    r_squared,
    rmse,
)
from gimin.evaluation.visualization import GIMINVisualizer

__all__ = [
    # Metrics
    "rmse",
    "mae",
    "r_squared",
    "nrmse",
    "ks_test_per_feature",
    "calibration_metrics",
    "expected_calibration_error",
    # Baselines
    "MICEBaseline",
    "KNNBaseline",
    "MeanBaseline",
    "MedianBaseline",
    # Advanced baselines
    "MissForestBaseline",
    "GAINBaseline",
    "SoftImputeBaseline",
    # Experiments
    "MaskedValueExperiment",
    "DownstreamEvaluator",
    # Visualization
    "GIMINVisualizer",
]
