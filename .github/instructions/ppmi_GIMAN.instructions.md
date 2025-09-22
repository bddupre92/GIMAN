---
applyTo: '**'
---
## Project Context: GIMAN Preprocessing for PPMI Data

Our primary goal is to preprocess multimodal data from the Parkinson's Progression Markers Initiative (PPMI) to prepare it for a novel machine learning model called the Graph-Informed Multimodal Attention Network (GIMAN).

The core task involves cleaning, merging, and curating data from various sources into a single, analysis-ready master dataframe.

---

## Key Data Files & Identifiers

The project uses several key CSV files. When I mention them by name, please recognize their purpose:

* **`Demographics_18Sep2025.csv`**: Contains baseline patient info like sex and birth date.
* **`Participant_Status_18Sep2025.csv`**: Crucial for cohort definition (e.g., `COHORT_DEFINITION` column specifies 'Parkinson's Disease' or 'Healthy Control').
* **`MDS-UPDRS_Part_I_18Sep2025.csv`** & **`MDS-UPDRS_Part_III_18Sep2025.csv`**: Contain clinical assessment scores (non-motor and motor).
* **`FS7_APARC_CTH_18Sep2025.csv`**: Contains structural MRI (sMRI) features, specifically regional cortical thickness.
* **`Xing_Core_Lab_-_Quant_SBR_18Sep2025.csv`**: Contains DAT-SPECT imaging features, specifically Striatal Binding Ratios (SBRs).
* **`iu_genetic_consensus_20250515_18Sep2025.csv`**: Contains summarized genetic data (e.g., `LRRK2`, `GBA`, `APOE` status).

**The most important rule:** All dataframes must be merged using the following key columns:
* `PATNO`: The unique patient identifier.
* `EVENT_ID`: The visit identifier (e.g., `BL` for baseline, `V04` for visit 4). This is critical for longitudinal analysis.

---

## Core Libraries & Workflow

* **Primary Tool:** Use the **`pandas`** library for all data manipulation.
* **Numerical Operations:** Use **`numpy`**.
* **ML Preprocessing:** Use **`scikit-learn`** for tasks like scaling (`StandardScaler`) and imputation (`SimpleImputer`, `KNNImputer`).
* **Workflow:** The standard workflow we will follow is:
    1.  Load individual CSVs into pandas DataFrames.
    2.  Clean and preprocess each DataFrame individually.
    3.  Merge all DataFrames into a single `master_df` using `PATNO` and `EVENT_ID`.
    4.  Perform final cohort selection and feature engineering on the `master_df`.
    5.  Handle any remaining missing values.
    6.  Scale numerical features for the model.

---

## Coding Style & Rules

1.  **Clarity is Key:** Generate Python code that is readable and well-commented. Use clear and descriptive variable names (e.g., `df_demographics`, `merged_clinical_data`, `final_cohort_df`).
2.  **Functional Programming:** When appropriate, suggest breaking down complex preprocessing steps into smaller, reusable functions with clear inputs, outputs, and docstrings.
3.  **Pandas Best Practices:** Use efficient pandas methods. Avoid iterating over rows (`iterrows`). Prefer vectorized operations. Be mindful of the `SettingWithCopyWarning`.
4.  **Assume the Context:** When I ask a question like "how should I merge the clinical data?", assume I am referring to the specific PPMI files mentioned above and that the goal is to support the GIMAN model.