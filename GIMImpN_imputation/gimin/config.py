"""
Configuration module for GIMIN.

Provides a dataclass-based configuration system with YAML serialization
support. All default hyperparameters, data paths, modality definitions,
and training settings are defined here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


@dataclass
class DataPaths:
    """Paths for data directories relative to project root."""

    raw_data_dir: str = "../../data/00_raw"
    processed_data_dir: str = "data/01_processed"
    interim_data_dir: str = "data/02_interim"
    output_dir: str = "outputs"
    checkpoint_dir: str = "outputs/checkpoints"
    log_dir: str = "outputs/logs"


@dataclass
class ModalityDefinition:
    """Definition of a single data modality."""

    name: str
    features: list[str]


@dataclass
class GraphParams:
    """Parameters for patient similarity graph construction."""

    k_neighbors: int = 15
    similarity_metric: str = "cosine"
    min_overlap: int = 3
    graph_refinement_iterations: int = 3
    alpha_start: float = 1.0
    alpha_end: float = 0.5


@dataclass
class ModelArchitecture:
    """Neural network architecture parameters."""

    embed_dim: int = 64
    num_gnn_layers: int = 3
    num_heads: int = 4
    mc_dropout_rate: float = 0.1


@dataclass
class TrainingParams:
    """Training hyperparameters."""

    lr: float = 1e-3
    weight_decay: float = 1e-5
    num_epochs: int = 200
    batch_mask_fraction: float = 0.2
    lambda_dist: float = 0.1
    lambda_cross: float = 0.05
    early_stopping_patience: int = 30


@dataclass
class EvaluationParams:
    """Evaluation configuration."""

    eval_mask_fractions: list[float] = field(
        default_factory=lambda: [0.1, 0.2, 0.3, 0.5]
    )
    eval_num_runs: int = 10
    mc_samples: int = 50


@dataclass
class IncrementalParams:
    """Parameters for incremental graph updates."""

    rebuild_interval: int = 100
    local_hop_count: int = 2


@dataclass
class GIMINConfig:
    """
    Master configuration for the GIMIN framework.

    Contains all hyperparameters, paths, modality definitions, and
    settings required to run the full pipeline. Supports YAML
    serialization for experiment reproducibility.
    """

    # Sub-configurations
    data_paths: DataPaths = field(default_factory=DataPaths)
    graph: GraphParams = field(default_factory=GraphParams)
    model: ModelArchitecture = field(default_factory=ModelArchitecture)
    training: TrainingParams = field(default_factory=TrainingParams)
    evaluation: EvaluationParams = field(default_factory=EvaluationParams)
    incremental: IncrementalParams = field(default_factory=IncrementalParams)

    # Global settings
    random_seed: int = 42

    # Modality definitions
    modalities: list[ModalityDefinition] = field(
        default_factory=lambda: [
            ModalityDefinition(
                name="genetic",
                features=["LRRK2", "GBA", "APOE_E4", "SNCA", "GENETIC_RISK_SCORE"],
            ),
            ModalityDefinition(
                name="motor_clinical",
                features=[
                    "NP3TOT",
                    "NP1RTOT",
                    "NHY",
                    "PIGD_SCORE",
                    "TREMOR_SCORE",
                    "MCATOT",
                ],
            ),
            ModalityDefinition(
                name="structural_imaging",
                features=[
                    "CAUDATE_L_VOL",
                    "CAUDATE_R_VOL",
                    "PUTAMEN_L_VOL",
                    "PUTAMEN_R_VOL",
                    "HIPPOCAMPUS_L_VOL",
                    "HIPPOCAMPUS_R_VOL",
                ],
            ),
            ModalityDefinition(
                name="spect_sbr",
                features=[
                    "CAUDATE_L_SBR",
                    "CAUDATE_R_SBR",
                    "PUTAMEN_L_SBR",
                    "PUTAMEN_R_SBR",
                    "CAUDATE_ASYMMETRY",
                    "PUTAMEN_ASYMMETRY",
                ],
            ),
            ModalityDefinition(
                name="csf_biomarkers",
                features=[
                    "ALPHA_SYNUCLEIN",
                    "TOTAL_TAU",
                    "ABETA42",
                    "PTAU181",
                ],
            ),
            ModalityDefinition(
                name="clinical_biomarkers",
                features=[
                    "UPSIT_TOTAL",
                    "RBD_TOTAL",
                    "SCOPA_AUT_TOTAL",
                    "ESS_TOTAL",
                ],
            ),
            ModalityDefinition(
                name="cortical_thickness",
                features=[
                    "ENTORHINAL_L_CTH",
                    "ENTORHINAL_R_CTH",
                    "CINGULATE_L_CTH",
                    "CINGULATE_R_CTH",
                    "PRECENTRAL_L_CTH",
                    "PRECENTRAL_R_CTH",
                ],
            ),
            ModalityDefinition(
                name="demographics",
                features=["SEX", "AGE_AT_VISIT"],
            ),
        ]
    )

    def to_yaml(self, path: str | None = None) -> str:
        """
        Serialize configuration to YAML format.

        Args:
            path: If provided, write YAML to this file path.

        Returns:
            YAML string representation of the configuration.
        """
        config_dict = asdict(self)
        yaml_str = yaml.dump(
            config_dict,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )
        if path is not None:
            filepath = Path(path)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.write_text(yaml_str, encoding="utf-8")
        return yaml_str

    @classmethod
    def from_yaml(cls, path: str) -> GIMINConfig:
        """
        Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            A GIMINConfig instance populated from the YAML file.
        """
        filepath = Path(path)
        with filepath.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        if raw is None:
            return cls()

        # Reconstruct nested dataclasses from the flat dict
        data_paths = DataPaths(**raw.get("data_paths", {}))
        graph = GraphParams(**raw.get("graph", {}))
        model = ModelArchitecture(**raw.get("model", {}))
        training = TrainingParams(**raw.get("training", {}))
        evaluation = EvaluationParams(**raw.get("evaluation", {}))
        incremental = IncrementalParams(**raw.get("incremental", {}))

        modalities_raw = raw.get("modalities", [])
        modalities = [ModalityDefinition(**m) for m in modalities_raw]

        return cls(
            data_paths=data_paths,
            graph=graph,
            model=model,
            training=training,
            evaluation=evaluation,
            incremental=incremental,
            random_seed=raw.get("random_seed", 42),
            modalities=modalities if modalities else cls().modalities,
        )

    @property
    def all_feature_names(self) -> list[str]:
        """Return a flat list of all feature names across modalities."""
        features = []
        for mod in self.modalities:
            features.extend(mod.features)
        return features

    @property
    def num_features(self) -> int:
        """Total number of features across all modalities."""
        return len(self.all_feature_names)

    @property
    def modality_name_to_features(self) -> dict[str, list[str]]:
        """Map from modality name to its feature list."""
        return {mod.name: mod.features for mod in self.modalities}

    @property
    def modality_dims(self) -> list[int]:
        """List of per-modality feature counts (e.g., [5, 6, 6, 6, 4, 4, 6, 2])."""
        return [len(mod.features) for mod in self.modalities]

    @property
    def feature_to_modality(self) -> dict[str, str]:
        """Map from feature name to its parent modality name."""
        mapping = {}
        for mod in self.modalities:
            for feat in mod.features:
                mapping[feat] = mod.name
        return mapping
