# GIMAN Phase 8: Model Frontier Development

**Status:** Planning Phase  
**Timeline:** November 2025 - August 2026 (10 months)  
**Version:** 8.0.0

---

## Overview

Phase 8 represents the next-generation evolution of the Graph-Informed Multimodal Attention Network (GIMAN) framework. This phase transitions GIMAN from a research prototype to a production-ready, clinically deployable system spanning the full Parkinson's disease spectrum.

### Key Innovations
- 🔄 **Dual-Model Architecture**: Separate specialized models for progression and conversion
- ⏱️ **Dynamic Survival Analysis**: Time-to-event predictions replace static classifications
- 🧬 **SAA Integration**: Alpha-synuclein pathology prediction from non-invasive data
- 🌈 **Continuous Heterogeneity**: VAE latent space modeling of disease landscape
- 🎯 **Multi-Task Learning**: Unified architecture for multiple prediction tasks
- 🔍 **Enhanced Explainability**: XAI adapted for dynamic predictions
- ✅ **External Validation**: PDBP cohort validation for generalizability
- 🌐 **Open-Source Release**: Community-accessible framework

---

## Directory Structure

```
phase8/
├── README.md (this file)
├── PHASE8_STRATEGIC_ROADMAP.md (comprehensive plan)
│
├── subphase8_1_foundational/ (Nov 4-25)
│   ├── SUBPHASE_8_1_DETAILED_PLAN.md
│   ├── giman_progression.py
│   ├── giman_conversion.py
│   ├── prodromal_cohort_curation.py
│   ├── prodromal_inclusion_criteria.yaml
│   ├── prodromal_data_dictionary.md
│   └── results/
│       ├── prodromal_cohort_baseline.csv
│       ├── prodromal_cohort_longitudinal.csv
│       └── prodromal_cohort_characterization.md
│
├── subphase8_2_dynamic_endpoints/ (Nov 25 - Dec 23)
│   ├── SUBPHASE_8_2_DETAILED_PLAN.md
│   ├── disability_milestones.py
│   ├── phenoconversion_endpoints.py
│   ├── survival_data_engineering.py
│   ├── cox_baseline.py
│   ├── deepsurv_integration.py
│   └── results/
│       ├── milestones_data.csv
│       ├── survival_statistics.json
│       └── baseline_cox_performance.json
│
├── subphase8_3_saa_integration/ (Dec 23 - Jan 20)
│   ├── SUBPHASE_8_3_DETAILED_PLAN.md
│   ├── giman_saa.py
│   ├── saa_data_curation.py
│   ├── saa_proxy_validation.py
│   ├── saa_feature_importance.py
│   └── results/
│       ├── saa_predictions.csv
│       ├── saa_model_performance.json
│       └── saa_explainability_report.md
│
├── subphase8_4_vae_heterogeneity/ (Jan 20 - Feb 10)
│   ├── SUBPHASE_8_4_DETAILED_PLAN.md
│   ├── giman_vae_heterogeneity.py
│   ├── embedding_extraction.py
│   ├── latent_space_analysis.py
│   ├── continuous_subtyping.py
│   └── results/
│       ├── vae_model.pth
│       ├── latent_coordinates.csv
│       ├── latent_space_interpretation.md
│       └── interactive_latent_viz.html
│
├── subphase8_5_multitask_architecture/ (Feb 10 - Mar 10)
│   ├── SUBPHASE_8_5_DETAILED_PLAN.md
│   ├── giman_multitask.py
│   ├── survival_head.py
│   ├── multitask_loss.py
│   ├── shared_encoder.py
│   ├── task_balancing.py
│   └── results/
│       ├── multitask_model.pth
│       ├── task_performance_comparison.json
│       └── training_curves.png
│
├── subphase8_6_explainability/ (Mar 10 - Mar 31)
│   ├── SUBPHASE_8_6_DETAILED_PLAN.md
│   ├── shap_survival_analysis.py
│   ├── gnnexplainer_prodromal.py
│   ├── gradcam_saa_mapping.py
│   ├── temporal_feature_importance.py
│   ├── counterfactual_survival.py
│   └── results/
│       ├── shap_values_survival.csv
│       ├── subgraph_explanations.json
│       ├── saliency_maps/
│       └── explainability_consensus_report.md
│
├── subphase8_7_validation/ (Mar 31 - Apr 28)
│   ├── SUBPHASE_8_7_DETAILED_PLAN.md
│   ├── pdbp_data_integration.py
│   ├── external_validation_pipeline.py
│   ├── sota_benchmarking.py
│   ├── improvement_quantification.py
│   └── results/
│       ├── pdbp_validation_results.json
│       ├── benchmark_comparison_table.csv
│       ├── validation_report.pdf
│       └── calibration_plots/
│
├── subphase8_8_dissemination/ (Apr 28 - Jul 21)
│   ├── SUBPHASE_8_8_DETAILED_PLAN.md
│   ├── manuscript_templates/
│   │   ├── paper1_multitask_framework/
│   │   ├── paper2_saa_prediction/
│   │   └── paper3_continuous_heterogeneity/
│   ├── open_source_packaging/
│   │   ├── setup.py
│   │   ├── requirements.txt
│   │   ├── README.md
│   │   └── LICENSE
│   ├── tutorials/
│   │   ├── 01_quickstart.ipynb
│   │   ├── 02_custom_data.ipynb
│   │   └── 03_explainability.ipynb
│   └── results/
│       ├── manuscript_drafts/
│       ├── github_release/
│       └── workshop_materials/
│
├── configs/
│   ├── dual_model_config.yaml (master configuration)
│   ├── subphase_configs/
│   │   ├── config_8_1.yaml
│   │   ├── config_8_2.yaml
│   │   └── ... (one per subphase)
│   └── experiment_tracking.yaml
│
├── docs/
│   ├── API_REFERENCE.md
│   ├── ARCHITECTURE_GUIDE.md
│   ├── DATA_SPECIFICATIONS.md
│   ├── SURVIVAL_ANALYSIS_TUTORIAL.md
│   ├── MULTITASK_LEARNING_GUIDE.md
│   └── CONTRIBUTING.md
│
├── notebooks/
│   ├── exploratory_analysis/
│   ├── model_development/
│   └── results_visualization/
│
└── results/
    ├── subphase_summaries/
    ├── models/
    │   ├── giman_progression_v8.pth
    │   ├── giman_conversion_v8.pth
    │   ├── giman_multitask_v8.pth
    │   └── vae_heterogeneity_v8.pth
    ├── predictions/
    ├── explainability/
    └── validation/
```

---

## Subphase Summary

### 8.1: Foundational Re-scoping (Nov 4-25)
**Goal:** Establish dual-model framework and curate prodromal data  
**Key Outputs:** 
- `giman_progression.py` and `giman_conversion.py`
- Prodromal cohort (n≥150) with >85% data completeness
- Unified configuration system

**Status:** 🟡 Planning

---

### 8.2: Dynamic Endpoint Modeling (Nov 25 - Dec 23)
**Goal:** Implement survival analysis for time-to-event predictions  
**Key Outputs:**
- 25 disability milestone endpoints operationalized
- Phenoconversion endpoint defined
- Cox baseline models (C-index >0.70)

**Status:** 🟡 Planning

---

### 8.3: SAA Integration (Dec 23 - Jan 20)
**Goal:** Predict CSF SAA status from non-invasive biomarkers  
**Key Outputs:**
- `giman_saa.py` model (AUC >0.85)
- SAA prediction validation
- Feature importance for synucleinopathy

**Status:** 🟡 Planning

---

### 8.4: VAE Heterogeneity (Jan 20 - Feb 10)
**Goal:** Model continuous disease heterogeneity landscape  
**Key Outputs:**
- VAE trained on GIMAN embeddings
- 8-16 dimensional latent space
- Interpretation of latent axes (correlation with biology)

**Status:** 🟡 Planning

---

### 8.5: Multi-Task Architecture (Feb 10 - Mar 10)
**Goal:** Unified model for simultaneous multi-task prediction  
**Key Outputs:**
- `giman_multitask.py` with 4 prediction heads
- Composite loss function with task balancing
- Performance matching single-task models

**Status:** 🟡 Planning

---

### 8.6: Enhanced Explainability (Mar 10 - Mar 31)
**Goal:** Adapt XAI framework to dynamic survival predictions  
**Key Outputs:**
- SHAP for time-dependent hazards
- GNNExplainer for prodromal conversion
- Cross-method consensus >85%

**Status:** 🟡 Planning

---

### 8.7: External Validation (Mar 31 - Apr 28)
**Goal:** Validate on PDBP and benchmark against SOTA  
**Key Outputs:**
- PDBP validation (C-index >0.73)
- Benchmarking vs. RSF, DeepSurv, Cox
- External validation report

**Status:** 🟡 Planning

---

### 8.8: Dissemination (Apr 28 - Jul 21)
**Goal:** Publish findings and release open-source framework  
**Key Outputs:**
- 3 manuscripts (Nature MI, Movement Disorders, Brain)
- GitHub release with pretrained models
- Tutorials and documentation

**Status:** 🟡 Planning

---

## Quick Start (For Developers)

### Prerequisites
```bash
# Python 3.10+
# CUDA 11.8+ (for GPU acceleration)
# 16GB+ RAM (32GB recommended)
# 500GB storage for PPMI + PDBP data
```

### Installation
```bash
# Clone repository
cd "E:\My Drive\CSCI FALL 2025"

# Create virtual environment
python -m venv phase8_env
.\phase8_env\Scripts\activate  # Windows
# source phase8_env/bin/activate  # Unix/Mac

# Install dependencies
pip install -r archive/development/phase8/requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch_geometric; print(f'PyG: {torch_geometric.__version__}')"
```

### Configuration
```bash
# Edit master config
code archive/development/phase8/configs/dual_model_config.yaml

# Set data paths
export GIMAN_DATA_ROOT="E:/My Drive/CSCI FALL 2025/data"
export GIMAN_RESULTS_ROOT="E:/My Drive/CSCI FALL 2025/results/phase8"
```

### Running Subphase 8.1
```bash
cd archive/development/phase8/subphase8_1_foundational

# Step 1: Curate prodromal cohort
python prodromal_cohort_curation.py \
    --config ../configs/dual_model_config.yaml \
    --output results/prodromal_cohort_baseline.csv

# Step 2: Characterize cohort
python prodromal_cohort_characterization.py \
    --input results/prodromal_cohort_baseline.csv \
    --output results/prodromal_cohort_characterization.md

# Step 3: Validate dual models
python test_dual_models.py --config ../configs/config_8_1.yaml
```

---

## Key Dependencies

### Core Libraries
```python
# Deep Learning
torch==2.1.0
torch-geometric==2.4.0
torch-scatter==2.1.2
torch-sparse==0.6.18

# Survival Analysis
lifelines==0.27.8
pycox==0.2.3
scikit-survival==0.22.0

# Data & ML
pandas==2.1.3
numpy==1.24.3
scikit-learn==1.3.2

# Explainability
shap==0.44.0
captum==0.7.0

# Visualization
matplotlib==3.8.2
seaborn==0.13.0
plotly==5.18.0

# Utilities
pyyaml==6.0.1
tqdm==4.66.1
wandb==0.16.0  # Experiment tracking
```

---

## Performance Targets

### Model Performance
| Task | Metric | Target | Baseline (Phase 4-6) |
|------|--------|--------|----------------------|
| Progression | C-index | >0.78 | 0.75 (static classification) |
| Conversion | C-index | >0.79 | 0.74 (DeepSurv) |
| SAA Prediction | AUC | >0.85 | N/A (new task) |
| Diagnostic | Accuracy | >0.82 | 0.82 |

### External Validation (PDBP)
| Task | Metric | Target | Acceptable Degradation |
|------|--------|--------|------------------------|
| Progression | C-index | >0.73 | <5% drop from PPMI |
| Conversion | C-index | >0.74 | <5% drop from PPMI |

### Computational Performance
- **Training Time**: <6 hours for full multi-task model (4×A100 GPUs)
- **Inference Speed**: <100ms per patient (batch size 32)
- **Memory Usage**: <16GB GPU memory during training

---

## Success Metrics

### Scientific
- [ ] Multi-task model matches single-task performance
- [ ] External validation within 5% of PPMI performance
- [ ] SAA prediction AUC >0.85
- [ ] VAE latent space interpretable (≥3 axes correlate with biology)
- [ ] Explainability cross-method consensus >85%

### Dissemination
- [ ] ≥2 papers accepted in high-impact journals (IF >10)
- [ ] Open-source package: >500 GitHub stars, >1000 downloads
- [ ] Workshop: >50 attendees
- [ ] Community adoption: ≥3 independent groups using GIMAN

### Clinical Translation
- [ ] Clinician feedback: >80% find tool useful
- [ ] Clinical validation study initiated
- [ ] FDA pre-submission meeting completed
- [ ] Industry licensing interest

---

## Team & Roles

| Role | Responsibilities | Time Commitment |
|------|------------------|-----------------|
| **Lead Developer** | Architecture, coding, analysis | 100% FTE (10 months) |
| **Principal Investigator** | Scientific direction, supervision | 20% FTE |
| **Clinical Collaborator** | Endpoint definition, validation | 10% FTE |
| **Biostatistician** | Survival analysis guidance | 20% FTE |
| **Software Engineer** | Open-source packaging | 30% FTE (months 8-10) |

---

## Timeline Overview

```
Nov 2025     Dec 2025     Jan 2026     Feb 2026     Mar 2026     Apr-Jul 2026
    |            |            |            |            |               |
  [8.1]────────[8.2]────────[8.3]────────[8.4]────────[8.5]──────────[8.6-8.8]
    3w           4w           4w           3w           4w              12w
    
Foundation   Dynamic      SAA          VAE         Multi-Task    Valid + Publish
& Prodromal  Endpoints    Integration  Heterogen.  Architecture
```

**Total Duration:** 10 months (44 weeks)  
**Target Completion:** August 2026  
**Manuscript Submission:** June-July 2026

---

## Risk Management

### High-Priority Risks
1. **Prodromal cohort size insufficient** (n<120)
   - Mitigation: Relax inclusion criteria, external data augmentation
   
2. **Multi-task training unstable**
   - Mitigation: Task balancing algorithms, separate stage training

3. **External validation poor performance** (>15% degradation)
   - Mitigation: Domain adaptation, PDBP preprocessing alignment

### Medium-Priority Risks
4. **SAA data limited** (n<100)
   - Mitigation: Use published SAA cohorts, focus on available modalities

5. **PDBP access delayed**
   - Mitigation: Use PPMI holdout set as backup validation

---

## Related Resources

### Documentation
- [GIMAN Phases 1-7 Summary](../GIMAN_Complete_Progression_Summary.md)
- [Phase 4-6 Comprehensive Guide](../../manuscripts/overall_manuscript/README.md)
- [PPMI Data Dictionary](../../../Docs/data_dictionary.md)

### Code Repositories
- **Main GIMAN Repo**: `E:\My Drive\CSCI FALL 2025\src\`
- **Phase 4-6 Code**: `E:\My Drive\CSCI FALL 2025\models\`
- **Preprocessing**: `E:\My Drive\CSCI FALL 2025\scripts\preprocessing\`

### External Links
- [PPMI Website](https://www.ppmi-info.org/)
- [PDBP Website](https://pdbp.ninds.nih.gov/)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- [Lifelines (Survival Analysis)](https://lifelines.readthedocs.io/)

---

## Contributing

### For Internal Team
1. Create feature branch: `git checkout -b subphase8_X_feature_name`
2. Follow coding standards: PEP 8, Google-style docstrings
3. Write tests for new functionality
4. Submit pull request with detailed description
5. Request code review from Lead Developer

### For External Contributors (Post Open-Source Release)
- See `CONTRIBUTING.md` in GitHub repository
- Join discussions in GitHub Issues
- Submit bug reports with reproducible examples
- Propose enhancements via GitHub Discussions

---

## Citation

### Phase 8 Framework (Preprint - Available August 2026)
```bibtex
@article{giman_phase8_2026,
  title={GIMAN-MultiTask: A Unified Graph Neural Network Framework for 
         Parkinson's Disease Across the Clinical Spectrum},
  author={[Authors]},
  journal={Nature Machine Intelligence},
  year={2026},
  note={In preparation}
}
```

### Original GIMAN (Phases 1-6)
```bibtex
@article{giman_comprehensive_2025,
  title={Graph-Informed Multimodal Attention Networks for Interpretable 
         Precision Medicine in Parkinson's Disease},
  author={[Authors]},
  journal={Nature Machine Intelligence},
  year={2025},
  note={Submitted}
}
```

---

## Contact

**Project Lead:** [Your Name]  
**Email:** [your.email@institution.edu]  
**Lab:** [Lab/Department Name]  
**Institution:** [Your Institution]

**GitHub:** [Future Public Repo URL]  
**Documentation:** [Future Website URL]

---

## License

**Internal Development Phase:** Proprietary (Institution Copyright)  
**Post-Publication:** Open-source release planned (MIT or Apache 2.0)

---

## Acknowledgments

- **Funding:** [Grant information]
- **Data:** Parkinson's Progression Markers Initiative (PPMI)
- **Compute:** [HPC cluster or cloud provider]
- **Collaborators:** PPMI investigators, clinical partners

---

*Last Updated: October 6, 2025*  
*Version: 8.0.0*  
*Next Review: November 11, 2025*
