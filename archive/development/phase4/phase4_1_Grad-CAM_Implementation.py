#!/usr/bin/env python3
"""GIMAN Phase 4.1: Saliency Map Generation for Unified System

This script loads the trained Phase 4 Unified GIMAN System and uses Captum's
Integrated Gradients to generate attribution maps for the cognitive prediction
task. The resulting maps are saved as .npz files for each patient, which can
then be converted into NIfTI format for SPM analysis in MATLAB.

Author: GIMAN Development Team
Date: September 26, 2025
Phase: 4.1 - Saliency Map Generation
"""

import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path for module imports
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(project_root))

from archive.development.phase4.phase4_unified_giman_system import (
    RealDataPhase3Integration,
    UnifiedGIMANSystem,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ModelWrapperForCaptum(nn.Module):
    """A wrapper to make the UnifiedGIMANSystem compatible with Captum for the cognitive task."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, spatial_emb, genomic_emb, temporal_emb):
        """Forward pass that returns only the cognitive prediction, as required by Captum.
        We apply sigmoid to get probabilities.
        """
        _, cog_pred, _ = self.model(spatial_emb, genomic_emb, temporal_emb)
        return torch.sigmoid(cog_pred)


def generate_and_save_saliency_maps(
    model, data_loader, patient_ids, device, output_dir="saliency_maps/cognitive"
):
    """Generates and saves saliency maps for the cognitive task using Integrated Gradients."""
    model.eval()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving saliency maps to {output_path.resolve()}")

    # Wrap the model for Captum
    wrapped_model = ModelWrapperForCaptum(model)
    ig = IntegratedGradients(wrapped_model)

    patient_id_map = {i: pid for i, pid in enumerate(patient_ids)}

    with torch.no_grad():
        for i, (s_emb, g_emb, t_emb) in enumerate(data_loader):
            s_emb, g_emb, t_emb = s_emb.to(device), g_emb.to(device), t_emb.to(device)

            # Define baselines (zero tensors)
            baselines = (s_emb * 0, g_emb * 0, t_emb * 0)

            # Generate attributions using Integrated Gradients
            # The target is 0, as we have a single output from the wrapped model
            attributions = ig.attribute(
                (s_emb, g_emb, t_emb),
                baselines=baselines,
                target=0,
                n_steps=50,
                internal_batch_size=s_emb.shape[0],
            )

            # Save attributions for each patient in the batch
            for j in range(s_emb.shape[0]):
                global_idx = i * data_loader.batch_size + j
                patient_id = patient_id_map.get(global_idx)
                if patient_id is None:
                    continue

                # Detach and move to CPU before converting to numpy
                attr_spatial = attributions[0][j].cpu().detach().numpy()
                attr_genomic = attributions[1][j].cpu().detach().numpy()
                attr_temporal = attributions[2][j].cpu().detach().numpy()

                # Save as a compressed NumPy array
                np.savez_compressed(
                    output_path / f"patient_{patient_id}_attributions.npz",
                    spatial=attr_spatial,
                    genomic=attr_genomic,
                    temporal=attr_temporal,
                )

    logger.info(f"Saliency map generation complete. Files are in {output_path}")


def main():
    """Main function to generate saliency maps from the trained Phase 4 model."""
    logger.info("🎬 GIMAN Phase 4.1: Saliency Map Generation")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- 1. Load Data ---
    logger.info("📊 Loading and preparing data...")
    data_integrator = RealDataPhase3Integration()
    data_integrator.load_and_prepare_data()

    dataset = TensorDataset(
        torch.tensor(data_integrator.spatiotemporal_embeddings, dtype=torch.float32),
        torch.tensor(data_integrator.genomic_embeddings, dtype=torch.float32),
        torch.tensor(data_integrator.temporal_embeddings, dtype=torch.float32),
    )
    data_loader = DataLoader(dataset, batch_size=16, shuffle=False)

    # --- 2. Load Trained Model ---
    logger.info("🧠 Loading trained Unified GIMAN model...")
    model_path = "phase4_best_model.pth"
    if not Path(model_path).exists():
        logger.error(f"Model file not found: {model_path}. Please run training first.")
        return

    model = UnifiedGIMANSystem().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    logger.info("✅ Model loaded successfully.")

    # --- 3. Generate Saliency Maps ---
    logger.info("🔥 Generating saliency maps for the cognitive prediction task...")
    generate_and_save_saliency_maps(
        model, data_loader, data_integrator.patient_ids, device
    )

    # --- 4. Prepare for SPM ---
    logger.info(
        "✅ Process complete. Next step: Convert saliency maps to NIfTI for SPM."
    )
    logger.info(
        "The generated .npz files contain attribution scores for each input feature embedding."
    )
    logger.info(
        "A separate script will be needed to map these 1D attributions back to a 3D brain space for SPM."
    )


if __name__ == "__main__":
    main()
