# GIMAN Phase 2 Development Roadmap
# =====================================
# CNN + GRU Spatiotemporal Encoder Implementation
# Updated: September 26, 2025

## 🎯 OBJECTIVE
Implement 3D CNN + GRU architecture for longitudinal neuroimaging analysis in GIMAN

## 📊 CURRENT STATUS

### ✅ COMPLETED
- **Dataset Expansion**: Successfully expanded from 2 to 7 patients (3.5x increase)
  - 14 longitudinal structural MRI sessions
  - 164MB of high-quality NIfTI data
  - Perfect conversion success rate (100%)
  - Files: `data/02_nifti_expanded/`

- **Architecture Implementation**: 
  - 3D CNN feature extractor (`phase2_5_cnn_gru_encoder.py`)
  - GRU temporal encoder (`phase2_5_cnn_gru_encoder.py`)
  - Spatiotemporal integration (`phase2_5_cnn_gru_encoder.py`)
  - Data loading pipeline (`phase2_4_nifti_data_loader.py`)

- **Integration Pipeline**:
  - CNN + GRU integration framework (`phase2_6_cnn_gru_integration.py`)
  - Single-modality adaptation (sMRI only)
  - Development environment organization

### 🔄 IN PROGRESS  
- **Real Data Integration**: Connecting NIfTI files with PyTorch pipeline
- **Preprocessing Pipeline**: Optimizing for structural MRI
- **Training Loop**: CNN + GRU model training

### ⏳ TODO
- **Model Training**: Train CNN + GRU on expanded cohort
- **Embedding Generation**: Create 256-dim spatiotemporal embeddings
- **GIMAN Integration**: Connect embeddings with existing GIMAN pipeline
- **Performance Validation**: Compare vs. baseline 2-patient model

## 🗂️ FILE ORGANIZATION

### Development Files (`archive/development/phase2/`)
```
phase2_1_spatiotemporal_imaging_encoder.py    # Original architecture
phase2_2_genomic_transformer_encoder.py       # Genomic component
phase2_3_longitudinal_cohort_definition.py    # Cohort identification
phase2_3_simplified_longitudinal_cohort.py    # Simplified cohort
phase2_4_nifti_data_loader.py                 # Data loading pipeline
phase2_5_cnn_gru_encoder.py                   # CNN + GRU implementation
phase2_6_cnn_gru_integration.py               # Integration pipeline (NEW)

# Data expansion scripts (moved from main directory)
comprehensive_ppmi3_analyzer.py               # PPMI 3 analysis
ppmi3_expansion_planner.py                    # Expansion planning
phase_1_conversion.py                         # Structural MRI conversion
phase_2_conversion.py                         # DAT-SPECT conversion (blocked)
phase_2_alt_conversion.py                     # Alternative DAT-SPECT
expansion_summary.py                          # Expansion summary
final_expansion_report.py                     # Final report
```

### Data Files
```
data/02_nifti_expanded/                       # Expanded NIfTI files (164MB)
giman_expanded_cohort_final.csv               # Final dataset manifest
archive/development/phase2/integration_output/ # Development outputs
```

## 🚀 EXECUTION PLAN

### Phase 2.7: Training Pipeline (NEXT)
**Goal**: Train CNN + GRU model on expanded dataset
**Files**: `phase2_7_training_pipeline.py`
**Tasks**:
- Implement real NIfTI data loading
- Create training/validation splits  
- Add model checkpointing
- Generate loss curves and metrics

### Phase 2.8: Embedding Generation  
**Goal**: Generate spatiotemporal embeddings for GIMAN
**Files**: `phase2_8_embedding_generator.py`
**Tasks**:
- Run trained model on full cohort
- Generate 256-dim embeddings per patient
- Save embeddings for GIMAN integration
- Validate embedding quality

### Phase 2.9: GIMAN Integration
**Goal**: Connect CNN + GRU embeddings with existing GIMAN
**Files**: `phase2_9_giman_integration.py`  
**Tasks**:
- Replace placeholder embeddings in GIMAN
- Update GIMAN input pipeline
- Test end-to-end functionality
- Performance comparison

## 🧠 TECHNICAL DETAILS

### Dataset Characteristics
- **Patients**: 7 (100232, 100677, 100712, 100960, 101021, 101178, 121109)
- **Sessions**: 14 total (2 per patient: baseline + follow-up)
- **Modality**: Structural MRI only (MPRAGE)
- **Resolution**: High-resolution 3D NIfTI
- **Size**: 164MB total

### Model Architecture
- **Input**: (batch_size, timepoints, 1, 96, 96, 96) - single modality
- **3D CNN**: 4-block ResNet architecture, 256-dim features
- **GRU**: 2-layer bidirectional, 256-dim hidden states  
- **Output**: (batch_size, 256) spatiotemporal embeddings
- **Parameters**: ~2M trainable parameters

### Preprocessing Pipeline
- **Intensity normalization**: Z-score on brain tissue
- **Spatial resampling**: 96³ voxels
- **Memory optimization**: Caching enabled
- **Quality checks**: File validation, missing data handling

## 📝 DEVELOPMENT NOTES

### Adaptations Made
1. **Single Modality**: Adapted from dual-modality (sMRI + DAT-SPECT) to sMRI-only
2. **Reduced Complexity**: Simplified preprocessing for development phase
3. **Memory Efficiency**: Smaller input size (96³ vs 128³) for faster iteration
4. **Flexible Architecture**: Configurable for future multimodal expansion

### Known Limitations
1. **DAT-SPECT Missing**: Technical issues prevented DAT-SPECT conversion
2. **Small Cohort**: 7 patients sufficient for proof-of-concept, may need more for production
3. **Single Modality**: Missing multimodal fusion benefits
4. **Preprocessing**: Simplified pipeline may need enhancement for production

### Future Enhancements
1. **DAT-SPECT Recovery**: Investigate alternative conversion tools
2. **Cohort Expansion**: Add more patients from PPMI dataset
3. **Advanced Preprocessing**: Add skull stripping, bias correction, registration
4. **Multi-site Data**: Incorporate additional imaging sites

## 🎯 SUCCESS METRICS
- [ ] Model trains successfully on expanded cohort
- [ ] Generates stable 256-dim embeddings
- [ ] Integrates seamlessly with existing GIMAN
- [ ] Shows improved performance vs. 2-patient baseline  
- [ ] Maintains reasonable training/inference time

---
**Status**: Ready for Phase 2.7 (Training Pipeline)  
**Priority**: High - Core architecture implementation  
**Estimated Time**: 1-2 development sessions  
**Dependencies**: None (all prerequisites complete)