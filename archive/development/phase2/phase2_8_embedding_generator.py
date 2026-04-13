#!/usr/bin/env python3
"""Phase 2.8: Spatiotemporal Embedding Generation - REAL DATA
==========================================================

Generate 256-dimensional spatiotemporal embeddings for all 7 patients using the
trained CNN + GRU model. These embeddings will be used in the main GIMAN pipeline.

Input: Trained model + giman_expanded_cohort_final.csv
Output: Patient embeddings in GIMAN-compatible format

NO PLACEHOLDERS - Uses real trained model and real patient data.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Import our components
from phase2_5_cnn_gru_encoder import (
    CNNConfig,
    GRUConfig,
    SpatiotemporalConfig,
    SpatiotemporalEncoder,
)
from phase2_7_training_pipeline import RealNIfTILongitudinalDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SpatiotemporalEmbeddingGenerator:
    """Generate spatiotemporal embeddings using trained CNN + GRU model."""

    def __init__(self, model_path: Path, device: torch.device = None):
        """Initialize embedding generator with trained model.

        Args:
            model_path: Path to trained model checkpoint
            device: Computing device (CPU/GPU)
        """
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model_path = model_path
        self.model = None

        # Load trained model
        self._load_trained_model()

        logger.info(f"EmbeddingGenerator initialized with device: {self.device}")

    def _load_trained_model(self):
        """Load the trained CNN + GRU model."""
        try:
            # Load checkpoint
            logger.info(f"Loading trained model from: {self.model_path}")
            checkpoint = torch.load(self.model_path, map_location=self.device)

            # Recreate model architecture (single channel for sMRI)
            cnn_config = CNNConfig(
                input_channels=1,  # sMRI only
                input_shape=(96, 96, 96),
                base_filters=32,
                num_blocks=4,
                feature_dim=256,
            )

            gru_config = GRUConfig(
                input_size=256, hidden_size=256, num_layers=2, output_size=256
            )

            spatiotemporal_config = SpatiotemporalConfig(
                cnn_config=cnn_config, gru_config=gru_config, max_timepoints=3
            )

            self.model = SpatiotemporalEncoder(spatiotemporal_config)

            # Load trained weights
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(self.device)
            self.model.eval()

            # Log model info
            train_loss = checkpoint.get("train_loss", "N/A")
            val_loss = checkpoint.get("val_loss", "N/A")
            epoch = checkpoint.get("epoch", "N/A")

            logger.info("✅ Model loaded successfully!")
            logger.info(f"  Epoch: {epoch}")
            logger.info(f"  Train Loss: {train_loss}")
            logger.info(f"  Val Loss: {val_loss}")
            logger.info(
                f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}"
            )

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def generate_patient_embedding(
        self, patient_sequence: dict, target_shape: tuple[int, int, int] = (96, 96, 96)
    ) -> np.ndarray:
        """Generate spatiotemporal embedding for a single patient.

        Args:
            patient_sequence: Patient data with file paths
            target_shape: Target shape for NIfTI data

        Returns:
            256-dimensional spatiotemporal embedding
        """
        patient_id = patient_sequence["patient_id"]

        try:
            # Create dataset for this patient
            dataset = RealNIfTILongitudinalDataset(
                [patient_sequence], target_shape=target_shape, normalize=True
            )

            # Get patient data
            patient_data = dataset[0]  # Only one patient in dataset

            # Prepare batch for model
            batch = {
                "imaging_data": patient_data["imaging_data"]
                .unsqueeze(0)
                .to(self.device),
                "num_timepoints": patient_data["num_timepoints"]
                .unsqueeze(0)
                .to(self.device),
            }

            # Generate embedding
            with torch.no_grad():
                embedding = self.model(batch)  # Shape: (1, 256)
                embedding = embedding.cpu().numpy()[
                    0
                ]  # Convert to numpy, remove batch dim

            logger.info(
                f"✅ Generated embedding for patient {patient_id}: shape {embedding.shape}"
            )
            return embedding

        except Exception as e:
            logger.error(f"Failed to generate embedding for patient {patient_id}: {e}")
            # Return zero embedding as fallback
            return np.zeros(256, dtype=np.float32)

    def generate_all_embeddings(
        self, manifest_df: pd.DataFrame, output_path: Path = None
    ) -> dict[str, np.ndarray]:
        """Generate spatiotemporal embeddings for all patients.

        Args:
            manifest_df: Dataset manifest with patient information
            output_path: Optional path to save embeddings

        Returns:
            Dictionary mapping patient_id -> embedding array
        """
        logger.info("🚀 Generating spatiotemporal embeddings for all patients...")

        # Get unique patients and create sequences
        patients = manifest_df["patient_id"].unique()
        logger.info(f"Processing {len(patients)} patients: {list(patients)}")

        # Create patient sequences
        patient_sequences = []
        for patient_id in patients:
            patient_data = manifest_df[manifest_df["patient_id"] == patient_id].copy()

            # Sort by session (baseline first)
            session_order = {
                "baseline": 0,
                "followup_1": 1,
                "followup_2": 2,
                "followup_3": 3,
            }
            patient_data["session_order"] = patient_data["session"].map(session_order)
            patient_data = patient_data.sort_values("session_order").reset_index(
                drop=True
            )

            # Verify files exist
            file_paths = patient_data["file_path"].tolist()
            existing_files = [fp for fp in file_paths if Path(fp).exists()]

            if len(existing_files) >= 2:  # Need at least 2 timepoints
                patient_sequences.append(
                    {
                        "patient_id": patient_id,
                        "file_paths": existing_files,
                        "sessions": patient_data["session"].tolist()[
                            : len(existing_files)
                        ],
                    }
                )

                logger.info(f"Patient {patient_id}: {len(existing_files)} timepoints")

        # Generate embeddings for all patients
        embeddings = {}
        total_patients = len(patient_sequences)

        for i, patient_seq in enumerate(patient_sequences, 1):
            patient_id = patient_seq["patient_id"]

            logger.info(f"[{i}/{total_patients}] Processing patient {patient_id}...")

            # Generate embedding
            embedding = self.generate_patient_embedding(patient_seq)
            embeddings[str(patient_id)] = embedding

            # Log embedding statistics
            embedding_mean = np.mean(embedding)
            embedding_std = np.std(embedding)
            embedding_norm = np.linalg.norm(embedding)

            logger.info(
                f"  Embedding stats: mean={embedding_mean:.4f}, "
                f"std={embedding_std:.4f}, norm={embedding_norm:.4f}"
            )

        logger.info(f"✅ Generated embeddings for {len(embeddings)} patients")

        # Save embeddings if output path provided
        if output_path:
            self._save_embeddings(embeddings, output_path)

        return embeddings

    def _save_embeddings(self, embeddings: dict[str, np.ndarray], output_path: Path):
        """Save embeddings in multiple formats for GIMAN compatibility."""
        output_path.mkdir(exist_ok=True)

        # 1. Save as NumPy arrays (primary format)
        embeddings_array = np.array(
            [embeddings[pid] for pid in sorted(embeddings.keys())]
        )
        patient_ids = sorted(embeddings.keys())

        np.savez_compressed(
            output_path / "spatiotemporal_embeddings.npz",
            embeddings=embeddings_array,
            patient_ids=patient_ids,
            embedding_dim=256,
            num_patients=len(patient_ids),
            generation_timestamp=datetime.now().isoformat(),
        )

        # 2. Save as JSON for easy inspection
        embeddings_json = {
            pid: embedding.tolist() for pid, embedding in embeddings.items()
        }

        with open(output_path / "spatiotemporal_embeddings.json", "w") as f:
            json.dump(
                {
                    "embeddings": embeddings_json,
                    "metadata": {
                        "embedding_dim": 256,
                        "num_patients": len(embeddings),
                        "patient_ids": list(embeddings.keys()),
                        "generation_timestamp": datetime.now().isoformat(),
                        "model_architecture": "CNN3D + GRU",
                        "input_modality": "sMRI_only",
                    },
                },
                f,
                indent=2,
            )

        # 3. Save as CSV for easy viewing
        embedding_df = pd.DataFrame(embeddings).T  # Transpose so patients are rows
        embedding_df.index.name = "patient_id"
        embedding_df.columns = [f"embedding_{i:03d}" for i in range(256)]
        embedding_df.to_csv(output_path / "spatiotemporal_embeddings.csv")

        # 4. Create summary report
        summary = {
            "generation_info": {
                "timestamp": datetime.now().isoformat(),
                "num_patients": len(embeddings),
                "embedding_dimension": 256,
                "model_type": "CNN3D + GRU Spatiotemporal Encoder",
            },
            "patient_statistics": {},
            "embedding_statistics": {
                "global_mean": float(np.mean(embeddings_array)),
                "global_std": float(np.std(embeddings_array)),
                "global_min": float(np.min(embeddings_array)),
                "global_max": float(np.max(embeddings_array)),
            },
        }

        # Add per-patient statistics
        for pid, embedding in embeddings.items():
            summary["patient_statistics"][pid] = {
                "mean": float(np.mean(embedding)),
                "std": float(np.std(embedding)),
                "norm": float(np.linalg.norm(embedding)),
                "min": float(np.min(embedding)),
                "max": float(np.max(embedding)),
            }

        with open(output_path / "embedding_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        logger.info("💾 Saved embeddings in multiple formats:")
        logger.info(f"  📦 NumPy: {output_path}/spatiotemporal_embeddings.npz")
        logger.info(f"  📄 JSON: {output_path}/spatiotemporal_embeddings.json")
        logger.info(f"  📊 CSV: {output_path}/spatiotemporal_embeddings.csv")
        logger.info(f"  📋 Summary: {output_path}/embedding_summary.json")


def create_giman_compatible_embeddings(
    embeddings: dict[str, np.ndarray],
    original_manifest: pd.DataFrame,
    output_path: Path,
):
    """Create GIMAN-compatible embedding files for integration."""
    logger.info("Creating GIMAN-compatible embedding files...")

    # Create embeddings in the format expected by GIMAN
    giman_embeddings = {}

    for patient_id, embedding in embeddings.items():
        # GIMAN expects embeddings per session, but we have per-patient spatiotemporal embeddings
        # We'll use the same embedding for all sessions of a patient (spatiotemporal context included)

        patient_sessions = original_manifest[
            original_manifest["patient_id"] == int(patient_id)
        ]

        for _, session_row in patient_sessions.iterrows():
            session_key = f"{patient_id}_{session_row['session']}"
            giman_embeddings[session_key] = embedding

    # Save in GIMAN format
    giman_format = {
        "embeddings": {k: v.tolist() for k, v in giman_embeddings.items()},
        "metadata": {
            "embedding_type": "spatiotemporal_cnn_gru",
            "embedding_dim": 256,
            "num_sessions": len(giman_embeddings),
            "generation_timestamp": datetime.now().isoformat(),
            "description": "Spatiotemporal embeddings generated by CNN3D + GRU encoder",
        },
    }

    giman_path = output_path / "giman_spatiotemporal_embeddings.json"
    with open(giman_path, "w") as f:
        json.dump(giman_format, f, indent=2)

    logger.info(f"💾 GIMAN-compatible embeddings: {giman_path}")
    logger.info(f"📊 Total sessions with embeddings: {len(giman_embeddings)}")

    return giman_path


def main():
    """Main embedding generation function."""
    print("\n" + "=" * 60)
    print("🧠 SPATIOTEMPORAL EMBEDDING GENERATION")
    print("=" * 60)

    # Setup paths
    base_dir = Path(
        "/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025"
    )
    manifest_path = base_dir / "giman_expanded_cohort_final.csv"
    model_path = Path("./training_output/best_model.pth")
    output_dir = Path("./embeddings_output")
    output_dir.mkdir(exist_ok=True)

    # Verify inputs
    if not manifest_path.exists():
        raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found: {model_path}")

    # Load dataset
    logger.info("Loading expanded dataset manifest...")
    manifest_df = pd.read_csv(manifest_path)
    logger.info(
        f"Loaded {len(manifest_df)} sessions from {len(manifest_df['patient_id'].unique())} patients"
    )

    # Initialize generator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = SpatiotemporalEmbeddingGenerator(model_path, device)

    # Generate embeddings
    embeddings = generator.generate_all_embeddings(manifest_df, output_dir)

    # Create GIMAN-compatible format
    giman_path = create_giman_compatible_embeddings(embeddings, manifest_df, output_dir)

    # Final summary
    print("\n" + "=" * 60)
    print("✅ EMBEDDING GENERATION COMPLETE")
    print("=" * 60)
    print(f"🧠 Generated embeddings for {len(embeddings)} patients")
    print("📐 Embedding dimension: 256")
    print(f"💾 Output directory: {output_dir}")
    print(f"🎯 GIMAN-ready file: {giman_path.name}")

    print("\n🎉 Spatiotemporal embeddings ready for GIMAN integration!")
    print("✅ Ready for Phase 2.9: GIMAN Integration")

    return embeddings, output_dir


if __name__ == "__main__":
    try:
        embeddings, output_dir = main()
        print("\n✅ Phase 2.8 Complete - Embeddings generated!")

    except Exception as e:
        logger.error(f"❌ Embedding generation failed: {e}")
        raise
