## **PRD: GIMAN Preprocessing \- Phase 1 Setup**

Document Version: 1.0  
Date: September 20, 2025  
Author: PPMI Research Gem

### **1\. Objective 🎯**

The objective of this phase is to establish a consistent, reproducible development environment and to load, merge, and consolidate all raw PPMI data sources into a single, unified pandas DataFrame. This **master\_df** will serve as the foundational dataset for all subsequent cleaning, feature engineering, and analysis steps required for the GIMAN model.

### **2\. User Profile 🧑‍🔬**

The primary user is a data scientist or ML researcher who needs a structured and efficient way to begin the data preprocessing workflow for the GIMAN project using VS Code and Python.

### **3\. Functional Requirements 📋**

#### **Phase 1: Environment & Project Setup (FR-ENV)**

* **FR-ENV-01: Create an Isolated Python Environment:**  
  * A dedicated virtual environment (e.g., using venv or conda) must be created to manage project-specific dependencies and ensure reproducibility.  
  * **Acceptance Criteria:** The virtual environment can be successfully activated and deactivated within the VS Code terminal.  
* **FR-ENV-02: Install Core Libraries:**  
  * The environment must have the following core Python libraries installed via pip: pandas, numpy, scikit-learn, matplotlib, and seaborn.  
  * **Acceptance Criteria:** Running pip list in the activated environment shows the correct versions of the installed libraries.  
* **FR-ENV-03: Establish Project Directory Structure:**  
  * A standardized folder structure must be created to organize project assets logically.  
    GIMAN\_PPMI\_Project/  
    ├── .vscode/  
    │   └── instructions.md  
    ├── .venv/  
    ├── data/  
    │   ├── raw/         \# All original CSVs go here  
    │   └── processed/   \# Processed data will be saved here  
    ├── notebooks/  
    │   └── 01\_environment\_and\_merge.ipynb  
    └── scripts/

  * **Acceptance Criteria:** The directory structure is created as specified.

#### **Phase 2: Data Loading & DataFrame Creation (FR-LOAD)**

* **FR-LOAD-01: Load all CSVs into Pandas:**  
  * A Jupyter Notebook or Python script must load each raw CSV file from the data/raw/ directory into a separate pandas DataFrame.  
  * **Acceptance Criteria:** Each CSV is successfully loaded without errors.  
* **FR-LOAD-02: Use Standardized DataFrame Naming:**  
  * DataFrames must be named according to a clear, descriptive convention.  
    * Demographics\_18Sep2025.csv \-\> **df\_demographics**  
    * Participant\_Status\_18Sep2025.csv \-\> **df\_status**  
    * MDS-UPDRS\_Part\_I\_18Sep2025.csv \-\> **df\_updrs1**  
    * MDS-UPDRS\_Part\_III\_18Sep2025.csv \-\> **df\_updrs3**  
    * iu\_genetic\_consensus\_20250515\_18Sep2025.csv \-\> **df\_genetics**  
    * FS7\_APARC\_CTH\_18Sep2025.csv \-\> **df\_smri**  
    * Xing\_Core\_Lab\_-\_Quant\_SBR\_18Sep2025.csv \-\> **df\_datscan**  
  * **Acceptance Criteria:** All DataFrames are created in memory with the specified names.

#### **Phase 3: Dataframe Merging Strategy (FR-MERGE)**

* **FR-MERGE-01: Create the Base Cohort DataFrame:**  
  * Create a base DataFrame, **df\_cohort**, by performing a **left merge** of df\_status onto df\_demographics using the PATNO column as the key. This ensures the base contains all demographic information for every participant listed in the status file.  
  * **Acceptance Criteria:** df\_cohort is created with columns from both source DataFrames.  
* **FR-MERGE-02: Merge Longitudinal Data:**  
  * Sequentially merge all longitudinal (time-varying) DataFrames into the df\_cohort DataFrame. All merges in this step must use both **PATNO** and **EVENT\_ID** as keys and be **left merges** to preserve all patient-visit records from the base cohort.  
    1. Merge **df\_updrs1** into df\_cohort.  
    2. Merge **df\_updrs3** into the result.  
    3. Merge **df\_smri** into the result.  
    4. Merge **df\_datscan** into the result.  
  * **Acceptance Criteria:** The df\_cohort DataFrame grows in columns after each merge, containing data from all longitudinal sources.  
* **FR-MERGE-03: Merge Static Data:**  
  * Merge the static (non-time-varying) genetic data, **df\_genetics**, into the df\_cohort. This merge will be a **left merge** using only **PATNO** as the key.  
  * **Acceptance Criteria:** Genetic information is successfully broadcast to all visit records for each corresponding patient.  
* **FR-MERGE-04: Create the Final Master DataFrame:**  
  * The final, fully merged DataFrame must be named **master\_df**.  
  * **Acceptance Criteria:** master\_df exists and contains the complete, unified dataset. An inspection of master\_df.info() shows a high column count and a mix of data types from all original files.

### **4\. Out of Scope for This Phase 🚫**

* Data cleaning (handling missing values, correcting data types).  
* Feature engineering (e.g., calculating total UPDRS scores, deriving age from birthdate).  
* Data visualization and Exploratory Data Analysis (EDA).  
* Model training and evaluation.
