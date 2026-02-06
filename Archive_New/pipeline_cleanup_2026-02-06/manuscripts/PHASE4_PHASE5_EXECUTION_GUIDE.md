# Phase 4 & Phase 5 Pipeline Execution Guide

## Overview

This guide provides instructions for executing the GIMAN Phase 4 (Progression Subtype Discovery) and Phase 5 (Prodromal Transition Prediction) research pipelines.

**Total Expected Runtime**: 7-11 hours  
**Total Expected Outputs**: ~90 files across both phases

---

## 📋 Prerequisites

### Data Requirements

1. **Phase 4**: Phase 1 cohort data
   - File: `giman_expanded_cohort_final.csv`
   - Expected: 2,046 patients with longitudinal data
   - Must include: PATNO, EVENT_ID, clinical assessments

2. **Phase 5**: PPMI prodromal cohort data
   - Directory or manifest CSV with prodromal participants
   - Must include: RBD, hyposmia, DAT SPECT, conversion events

### Python Environment

```bash
# Ensure you have the following packages installed:
pip install numpy pandas scikit-learn torch lifelines pycox shap matplotlib seaborn
```

---

## 🚀 Execution Options

### Option 1: Run Both Pipelines Sequentially (Recommended)

Use the master orchestrator to run both Phase 4 and Phase 5 automatically:

```bash
python execute_all_pipelines.py \
  --phase1-data "giman_expanded_cohort_final.csv" \
  --ppmi-data "[PATH_TO_PPMI_DATA]" \
  --output-dir "results/master_execution" \
  --random-seed 42
```

**Advantages**:
- Single command execution
- Automatic dependency management
- Unified results report
- Cross-phase analysis

**Expected Runtime**: 7-11 hours

---

### Option 2: Run Phase 4 Only

Execute only the progression subtype discovery pipeline:

```bash
python archive/development/phase4/execute_phase4_pipeline.py \
  --phase1-data "giman_expanded_cohort_final.csv" \
  --output-dir "results/phase4_execution" \
  --random-seed 42
```

**Expected Runtime**: 2-4 hours  
**Expected Outputs**: ~50 files

**Output Structure**:
```
results/phase4_execution/
├── phase4_master_results.json          # Master results JSON
├── PHASE4_EXECUTIVE_SUMMARY.md         # Executive summary
├── phase4_pipeline_YYYYMMDD_HHMMSS.log # Detailed log
├── task_4_1/                           # Longitudinal data prep
├── task_4_2/                           # Latent time alignment
├── task_4_3/                           # Trajectory clustering
├── task_4_4/                           # Subtype characterization
├── task_4_5/                           # Baseline prediction
└── task_4_6/                           # Trial enrichment
```

---

### Option 3: Run Phase 5 Only

Execute only the prodromal transition prediction pipeline:

```bash
python archive/development/phase5/execute_phase5_pipeline.py \
  --ppmi-data "[PATH_TO_PPMI_DATA]" \
  --output-dir "results/phase5_execution" \
  --random-seed 42
```

**Expected Runtime**: 3-5 hours  
**Expected Outputs**: ~40 files

**Output Structure**:
```
results/phase5_execution/
├── phase5_master_results.json          # Master results JSON
├── PHASE5_EXECUTIVE_SUMMARY.md         # Executive summary
├── phase5_pipeline_YYYYMMDD_HHMMSS.log # Detailed log
├── task_5_1/                           # Prodromal cohort
├── task_5_2/                           # Time-varying biomarkers
├── task_5_3/                           # Cox models
├── task_5_4/                           # DeepSurv
├── task_5_5/                           # Biomarker thresholds
└── task_5_6/                           # Risk stratification tool
```

---

## 🔄 Resume from Existing Results

If one phase has already been executed, you can skip it:

```bash
# Skip Phase 4, only run Phase 5
python execute_all_pipelines.py \
  --phase1-data "giman_expanded_cohort_final.csv" \
  --ppmi-data "[PATH_TO_PPMI_DATA]" \
  --skip-phase4 \
  --output-dir "results/master_execution"

# Skip Phase 5, only run Phase 4
python execute_all_pipelines.py \
  --phase1-data "giman_expanded_cohort_final.csv" \
  --ppmi-data "[PATH_TO_PPMI_DATA]" \
  --skip-phase5 \
  --output-dir "results/master_execution"
```

---

## 📊 Key Outputs

### Phase 4 Deliverables

1. **Subtype Labels**: Patient-level subtype assignments (Fast/Moderate/Slow)
2. **Clustering Metrics**: Silhouette score, Calinski-Harabasz index
3. **Baseline Prediction Model**: GNN classifier (AUC target: 0.75-0.80)
4. **Trial Enrichment Analysis**: Sample size reduction estimates (30-43%)
5. **Visualizations**: Trajectory plots, UMAP embeddings, feature importance

### Phase 5 Deliverables

1. **Cox Models**: Baseline and time-varying hazards models
2. **DeepSurv Model**: Neural survival network with SHAP importance
3. **Biomarker Thresholds**: Evidence-based cutpoints for risk stratification
4. **Risk Calculator**: Interactive clinical decision support tool
5. **Survival Curves**: Kaplan-Meier and predicted risk distributions

### Unified Deliverables (Master Pipeline)

1. **UNIFIED_EXECUTIVE_SUMMARY.md**: Cross-phase synthesis
2. **master_results.json**: Complete results from both phases
3. **Cross-Phase Analysis**: Subtype-specific conversion risk insights

---

## ⚡ Performance Tips

### For Long Execution Times

Run in a persistent session to avoid interruption:

```bash
# Using screen (Linux/Mac)
screen -S giman_pipeline
python execute_all_pipelines.py [OPTIONS]
# Detach: Ctrl+A, then D
# Reattach: screen -r giman_pipeline

# Using tmux (Linux/Mac)
tmux new -s giman_pipeline
python execute_all_pipelines.py [OPTIONS]
# Detach: Ctrl+B, then D
# Reattach: tmux attach -t giman_pipeline

# Using nohup (all platforms)
nohup python execute_all_pipelines.py [OPTIONS] > pipeline.out 2>&1 &
```

### Monitor Progress

All pipelines log to both console and file. Monitor with:

```bash
# Watch the log file
tail -f results/master_execution/master_pipeline_YYYYMMDD_HHMMSS.log

# Or for individual phases
tail -f results/phase4_execution/phase4_pipeline_YYYYMMDD_HHMMSS.log
tail -f results/phase5_execution/phase5_pipeline_YYYYMMDD_HHMMSS.log
```

---

## 🐛 Troubleshooting

### Issue: Import Errors

**Symptom**: `ModuleNotFoundError: No module named 'task_4_1_longitudinal_data_prep'`

**Solution**: The scripts automatically add archive directories to the Python path. If issues persist, run from the project root directory.

### Issue: Out of Memory

**Symptom**: Process killed or `MemoryError`

**Solution**: 
- Reduce batch sizes in neural network training
- Use a machine with more RAM (16GB+ recommended)
- Process data in chunks

### Issue: Data Not Found

**Symptom**: `FileNotFoundError: [Errno 2] No such file or directory`

**Solution**: 
- Verify data paths are correct
- Use absolute paths if relative paths fail
- Check that CSV files have expected columns (PATNO, EVENT_ID, etc.)

### Issue: Long Runtime

**Symptom**: Pipeline exceeds expected time

**Solution**:
- Check log file for stuck processes
- Verify hardware meets requirements (GPU recommended for Phase 5 DeepSurv)
- Consider running phases separately to identify bottleneck

---

## 📞 Support

For issues or questions:

1. Check the log files for detailed error messages
2. Review this guide for common issues
3. Contact the GIMAN Research Team

---

## ✅ Success Criteria

### Phase 4 Success

- ✅ All 6 tasks complete without errors
- ✅ Subtypes discovered (typically 2-3)
- ✅ Silhouette score > 0.3
- ✅ Baseline prediction AUC > 0.70
- ✅ PHASE4_EXECUTIVE_SUMMARY.md generated

### Phase 5 Success

- ✅ All 6 tasks complete without errors
- ✅ Prodromal cohort identified (converters + non-converters)
- ✅ Cox C-index > 0.65
- ✅ DeepSurv C-index > 0.70
- ✅ Risk calculator dashboard created
- ✅ PHASE5_EXECUTIVE_SUMMARY.md generated

### Master Pipeline Success

- ✅ Both phases complete
- ✅ UNIFIED_EXECUTIVE_SUMMARY.md generated
- ✅ Cross-phase analysis completed
- ✅ No critical errors in master log

---

## 📅 Timeline

| Phase | Task | Expected Time | Cumulative |
|-------|------|---------------|------------|
| Phase 4 | Task 4.1: Longitudinal Prep | 10-15 min | 0.25 hr |
| Phase 4 | Task 4.2: Latent Time | 20-30 min | 0.75 hr |
| Phase 4 | Task 4.3: Clustering | 30-45 min | 1.5 hr |
| Phase 4 | Task 4.4: Characterization | 15-20 min | 1.75 hr |
| Phase 4 | Task 4.5: Baseline Prediction | 40-60 min | 2.75 hr |
| Phase 4 | Task 4.6: Trial Enrichment | 30-45 min | 3.5 hr |
| Phase 5 | Task 5.1: Prodromal Cohort | 10-15 min | 3.75 hr |
| Phase 5 | Task 5.2: Time-Varying | 15-20 min | 4 hr |
| Phase 5 | Task 5.3: Cox Models | 30-40 min | 4.5 hr |
| Phase 5 | Task 5.4: DeepSurv | 90-120 min | 6.5 hr |
| Phase 5 | Task 5.5: Thresholds | 20-30 min | 7 hr |
| Phase 5 | Task 5.6: Risk Tool | 30-45 min | 7.5 hr |
| **Total** | **All Tasks** | **7-11 hours** | **7-11 hr** |

*Note: Times vary based on hardware (CPU vs GPU for neural networks)*

---

**Last Updated**: October 5, 2025  
**Version**: 1.0.0  
**Author**: GIMAN Research Team
