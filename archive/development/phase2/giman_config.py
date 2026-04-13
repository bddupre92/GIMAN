"""GIMAN Configuration: PATNO Standardization
=========================================

Configuration settings to ensure PATNO standardization throughout
the entire GIMAN pipeline, addressing the naming inconsistency
between imaging manifest (patient_id) and main GIMAN pipeline (PATNO).

This configuration enforces:
1. PATNO as the universal patient identifier
2. Standardized column mapping across all data sources
3. Consistent naming conventions for all pipeline components
"""

# Universal patient identifier
PATIENT_ID_COLUMN = "PATNO"

# Column mapping for different data sources
COLUMN_MAPPINGS = {
    # Imaging manifest uses patient_id -> standardize to PATNO
    "imaging_manifest": {
        "patient_id": "PATNO",
        "session": "EVENT_ID",
        "modality": "MODALITY",
    },
    # Clinical datasets already use PATNO (keep as-is)
    "clinical_datasets": {"PATNO": "PATNO", "EVENT_ID": "EVENT_ID"},
    # Spatiotemporal embeddings use numeric prefixes -> standardize to PATNO
    "spatiotemporal_embeddings": {
        "session_key_pattern": r"(\d+)_(\w+)",  # e.g., "1001_baseline"
        "patient_id_group": 1,  # First capture group
        "event_group": 2,  # Second capture group
        "patient_column": "PATNO",
        "event_column": "EVENT_ID",
    },
    # Ensure all outputs use PATNO
    "output_standardization": {
        "patient_column": "PATNO",
        "required_columns": ["PATNO", "EVENT_ID"],
    },
}

# PPMI-specific standardizations
PPMI_STANDARDIZATIONS = {
    # Standard PPMI event codes
    "event_mappings": {
        "BL": "baseline",
        "V01": "visit_01",
        "V02": "visit_02",
        "V03": "visit_03",
        "V04": "visit_04",
        "V05": "visit_05",
        "V06": "visit_06",
        "V07": "visit_07",
        "V08": "visit_08",
        "V09": "visit_09",
        "V10": "visit_10",
    },
    # Standard PPMI cohort definitions
    "cohort_mappings": {
        "Parkinson's Disease": "PD",
        "Healthy Control": "HC",
        "SWEDD": "SWEDD",
        "Prodromal": "PROD",
    },
    # Standard demographic encodings
    "demographic_mappings": {
        "GENDER": {1: "Male", 2: "Female"},
        "HANDED": {1: "Right", 2: "Left", 3: "Both"},
        "EDUCYRS_ranges": {"Low": (0, 12), "Medium": (13, 16), "High": (17, 25)},
    },
}

# Data validation rules
VALIDATION_RULES = {
    "required_columns": ["PATNO"],
    "patno_format": {"type": "integer", "min_value": 1000, "max_value": 99999},
    "event_id_format": {
        "allowed_values": [
            "BL",
            "V01",
            "V02",
            "V03",
            "V04",
            "V05",
            "V06",
            "V07",
            "V08",
            "V09",
            "V10",
        ],
        "alternative_formats": [
            "baseline",
            "visit_01",
            "visit_02",
            "visit_03",
            "visit_04",
        ],
    },
}

# Integration settings for Phase 3.1
GRAPH_INTEGRATION_CONFIG = {
    "patient_similarity": {
        "similarity_threshold": 0.3,
        "top_k_connections": 5,
        "feature_weights": {"clinical": 0.4, "spatiotemporal": 0.6},
    },
    "gat_architecture": {
        "hidden_dim": 128,
        "output_dim": 256,
        "num_heads": 4,
        "num_layers": 2,
        "dropout": 0.3,
    },
    "model_fusion": {
        "fusion_dim": 512,
        "classification_classes": 2,
        "regression_outputs": 1,
    },
}

# File path configurations
FILE_PATHS = {
    "base_directories": [
        "data/01_processed",
        "data/02_processed",
        "outputs",
        "archive/development/phase2",
    ],
    "clinical_datasets": [
        "giman_dataset_final_enhanced.csv",
        "giman_dataset_final_base.csv",
        "giman_dataset_enhanced.csv",
    ],
    "imaging_manifest": "imaging_manifest_expanded_cohort.csv",
    "spatiotemporal_embeddings": "spatiotemporal_embeddings.py",
    "output_directory": "graph_integration_output",
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(levelname)s - %(message)s",
    "validation_logs": True,
    "standardization_logs": True,
}
