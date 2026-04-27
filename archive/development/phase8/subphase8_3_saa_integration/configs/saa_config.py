"""
Phase 8.3: GIMAN-SAA Training Configuration

This module defines all training hyperparameters, model architecture settings,
and data processing parameters for the GIMAN-SAA model.

Author: GIMAN Research Team
Date: October 13, 2025
"""

from pathlib import Path


class SAAConfig:
    """Configuration for GIMAN-SAA training."""
    
    # ============================================================================
    # PROJECT PATHS
    # ============================================================================
    PROJECT_ROOT = Path("e:/My Drive/CSCI FALL 2025")
    
    # Data directories
    DATA_ROOT = PROJECT_ROOT / "data"
    SAA_DATA_DIR = DATA_ROOT / "04_saa"
    PHASE8_2_DATA = DATA_ROOT / "03_prodromal" / "final_training_dataset"
    
    # Output directories
    OUTPUT_DIR = PROJECT_ROOT / "outputs" / "phase8_3_saa"
    MODEL_DIR = OUTPUT_DIR / "models"
    RESULTS_DIR = OUTPUT_DIR / "results"
    FIGURES_DIR = OUTPUT_DIR / "figures"
    
    # Input files
    SAA_LABELS_FILE = SAA_DATA_DIR / "saa_raw_labels.csv"
    MULTIMODAL_FEATURES_FILE = PHASE8_2_DATA / "unified_longitudinal_early_pd.csv"
    
    # Output files
    SAA_TRAINING_DATA = SAA_DATA_DIR / "saa_training_data.csv"
    TRAIN_DATA_FILE = SAA_DATA_DIR / "train_data.pt"
    VAL_DATA_FILE = SAA_DATA_DIR / "val_data.pt"
    TEST_DATA_FILE = SAA_DATA_DIR / "test_data.pt"
    
    # ============================================================================
    # MODEL ARCHITECTURE
    # ============================================================================
    
    # Feature dimensions
    NUM_FEATURES = 49  # From Phase 8.2 multimodal features
    
    # GAT encoder settings
    HIDDEN_DIM = 256  # Increased from 128 for better model capacity
    NUM_GAT_LAYERS = 3
    NUM_HEADS = 4
    DROPOUT = 0.3
    USE_BATCH_NORM = True
    
    # ============================================================================
    # TRAINING HYPERPARAMETERS
    # ============================================================================
    
    # Batch and epoch settings
    BATCH_SIZE = 32
    MAX_EPOCHS = 100
    PATIENCE = 20  # Early stopping patience
    
    # Optimizer settings
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-5
    OPTIMIZER = "AdamW"  # or "Adam"
    
    # Learning rate scheduler
    USE_SCHEDULER = True
    SCHEDULER_TYPE = "ReduceLROnPlateau"
    SCHEDULER_PARAMS = {
        'mode': 'max',  # Maximize validation AUC
        'factor': 0.5,
        'patience': 10,
        'min_lr': 1e-6
    }
    
    # Loss function
    LOSS_FUNCTION = "focal"  # "focal" or "weighted_bce"
    POS_WEIGHT = 2.0  # Weight for SAA+ class (if using weighted_bce)
    FOCAL_ALPHA = 0.75  # Alpha for Focal Loss (weight for positive class)
    FOCAL_GAMMA = 2.0   # Gamma for Focal Loss (focusing parameter)
    
    # ============================================================================
    # DATA PROCESSING
    # ============================================================================
    
    # Train/val/test splits
    TRAIN_SPLIT = 0.70
    VAL_SPLIT = 0.15
    TEST_SPLIT = 0.15
    RANDOM_SEED = 42
    
    # Cross-validation
    USE_CV = True
    NUM_FOLDS = 5
    
    # Graph construction
    GRAPH_TYPE = "knn"  # k-nearest neighbors
    KNN_K = 10  # Number of neighbors
    SIMILARITY_METRIC = "cosine"  # or "euclidean"
    
    # Feature preprocessing
    NORMALIZE_FEATURES = True
    IMPUTE_MISSING = True
    IMPUTATION_STRATEGY = "knn"  # or "mean", "median"
    KNN_IMPUTER_NEIGHBORS = 5
    
    # ============================================================================
    # EVALUATION METRICS
    # ============================================================================
    
    PRIMARY_METRIC = "auc_roc"  # Primary metric for model selection
    
    METRICS = [
        "auc_roc",      # Area under ROC curve
        "auc_pr",       # Area under precision-recall curve
        "sensitivity",  # True positive rate
        "specificity",  # True negative rate
        "f1_score",     # Harmonic mean of precision and recall
        "accuracy",     # Overall accuracy
        "brier_score"   # Calibration metric
    ]
    
    # Thresholds for binary classification
    CLASSIFICATION_THRESHOLD = 0.5
    
    # Success criteria
    TARGET_AUC = 0.85
    TARGET_SENSITIVITY = 0.80
    TARGET_SPECIFICITY = 0.75
    
    # ============================================================================
    # FEATURE GROUPS
    # ============================================================================
    
    # Feature group definitions (from Phase 8.2)
    FEATURE_GROUPS = {
        'clinical_baseline': [
            'AGE_COMPUTED',
            'SEX',
            'UPDRS_PART_III_TOTAL',
            'MOCA_TOTAL'
        ],
        'clinical_expanded': [
            'UPDRS_PART_I',
            'UPDRS_PART_II',
            'SCHWAB_ENGLAND',
            'PIGD_SCORE',
            'TREMOR_SCORE'
        ],
        'genetic': [
            'LRRK2_CARRIER',
            'GBA_CARRIER',
            'APOE_E4_CARRIER',
            'SNCA_RISK_SCORE',
            'GENETIC_RISK_SCORE'
        ],
        'mri_structural': [
            'CAUDATE_LEFT_VOLUME',
            'CAUDATE_RIGHT_VOLUME',
            'PUTAMEN_LEFT_VOLUME',
            'PUTAMEN_RIGHT_VOLUME',
            'TOTAL_STRIATAL_VOLUME',
            'STRIATAL_ASYMMETRY_INDEX'
        ],
        'dat_spect': [
            'CAUDATE_LEFT_SBR',
            'CAUDATE_RIGHT_SBR',
            'PUTAMEN_LEFT_SBR',
            'PUTAMEN_RIGHT_SBR',
            'TOTAL_STRIATAL_SBR',
            'ASYMMETRY_INDEX_SBR'
        ],
        'csf_biomarkers': [
            'ALPHA_SYN_CSF',
            'PTAU_CSF',
            'TTAU_CSF',
            'ABETA_CSF'
        ],
        'clinical_biomarkers': [
            'UPSIT_TOTAL',
            'RBDSQ_TOTAL',
            'SCOPA_AUT_TOTAL',
            'ESS_TOTAL'
        ]
    }
    
    # ============================================================================
    # REPRODUCIBILITY
    # ============================================================================
    
    SEED = 42
    DETERMINISTIC = True  # Use deterministic algorithms
    BENCHMARK = False  # Disable CUDNN benchmark for reproducibility
    
    # ============================================================================
    # LOGGING AND CHECKPOINTING
    # ============================================================================
    
    # Logging
    LOG_INTERVAL = 10  # Log every N batches
    SAVE_PREDICTIONS = True
    SAVE_EMBEDDINGS = True
    
    # Checkpointing
    SAVE_BEST_MODEL = True
    SAVE_LAST_MODEL = True
    CHECKPOINT_METRIC = "val_auc_roc"
    CHECKPOINT_MODE = "max"  # Save when metric increases
    
    # ============================================================================
    # COMPUTATIONAL RESOURCES
    # ============================================================================
    
    # Device
    USE_GPU = True  # Use GPU if available
    GPU_ID = 0
    
    # Parallelization
    NUM_WORKERS = 4  # DataLoader workers
    
    # Memory management
    PIN_MEMORY = True
    
    # ============================================================================
    # VISUALIZATION
    # ============================================================================
    
    # Figure settings
    DPI = 300
    FIGURE_FORMAT = "png"  # or "pdf", "svg"
    
    # Plot types to generate
    GENERATE_PLOTS = [
        "roc_curve",
        "pr_curve",
        "confusion_matrix",
        "calibration_plot",
        "feature_importance",
        "attention_heatmap",
        "saa_distribution"
    ]
    
    @classmethod
    def create_directories(cls):
        """Create all necessary directories."""
        directories = [
            cls.SAA_DATA_DIR,
            cls.OUTPUT_DIR,
            cls.MODEL_DIR,
            cls.RESULTS_DIR,
            cls.FIGURES_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        print("Created all necessary directories:")
        for directory in directories:
            print(f"  ✓ {directory}")
    
    @classmethod
    def validate_paths(cls):
        """Validate that required input files exist."""
        required_files = []
        
        # Check if SAA labels exist (will be created by extract_saa_data.py)
        if cls.SAA_LABELS_FILE.exists():
            required_files.append(("SAA Labels", cls.SAA_LABELS_FILE, True))
        else:
            required_files.append(("SAA Labels", cls.SAA_LABELS_FILE, False))
        
        # Check if Phase 8.2 features exist
        if cls.MULTIMODAL_FEATURES_FILE.exists():
            required_files.append(("Multimodal Features", cls.MULTIMODAL_FEATURES_FILE, True))
        else:
            required_files.append(("Multimodal Features", cls.MULTIMODAL_FEATURES_FILE, False))
        
        print("\nPath Validation:")
        print("=" * 70)
        all_exist = True
        for name, path, exists in required_files:
            status = "✓" if exists else "✗"
            print(f"{status} {name}: {path}")
            if not exists:
                all_exist = False
        
        if not all_exist:
            print("\n⚠ WARNING: Some required files are missing!")
            print("Run the following scripts in order:")
            print("  1. extract_saa_data.py (to create SAA labels)")
            print("  2. align_saa_features.py (to merge with Phase 8.2 features)")
        
        return all_exist
    
    @classmethod
    def print_config(cls):
        """Print current configuration."""
        print("\n" + "=" * 70)
        print("GIMAN-SAA CONFIGURATION")
        print("=" * 70)
        
        print("\n📊 Model Architecture:")
        print(f"  Features: {cls.NUM_FEATURES}")
        print(f"  Hidden Dim: {cls.HIDDEN_DIM}")
        print(f"  GAT Layers: {cls.NUM_GAT_LAYERS}")
        print(f"  Attention Heads: {cls.NUM_HEADS}")
        print(f"  Dropout: {cls.DROPOUT}")
        
        print("\n🎯 Training:")
        print(f"  Batch Size: {cls.BATCH_SIZE}")
        print(f"  Max Epochs: {cls.MAX_EPOCHS}")
        print(f"  Learning Rate: {cls.LEARNING_RATE}")
        print(f"  Weight Decay: {cls.WEIGHT_DECAY}")
        print(f"  Early Stopping Patience: {cls.PATIENCE}")
        
        print("\n📈 Data Splits:")
        print(f"  Train: {cls.TRAIN_SPLIT*100:.0f}%")
        print(f"  Val: {cls.VAL_SPLIT*100:.0f}%")
        print(f"  Test: {cls.TEST_SPLIT*100:.0f}%")
        print(f"  Cross-Validation: {cls.NUM_FOLDS} folds" if cls.USE_CV else "  Cross-Validation: Disabled")
        
        print("\n🎯 Target Metrics:")
        print(f"  AUC-ROC: > {cls.TARGET_AUC}")
        print(f"  Sensitivity: > {cls.TARGET_SENSITIVITY}")
        print(f"  Specificity: > {cls.TARGET_SPECIFICITY}")
        
        print("\n" + "=" * 70)


# Create an instance for easy import
config = SAAConfig()


def main():
    """Test configuration setup."""
    print("Testing GIMAN-SAA Configuration")
    print("=" * 70)
    
    # Create directories
    SAAConfig.create_directories()
    
    # Validate paths
    SAAConfig.validate_paths()
    
    # Print configuration
    SAAConfig.print_config()


if __name__ == "__main__":
    main()
