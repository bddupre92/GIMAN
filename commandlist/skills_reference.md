# Skills Reference

Exported Wed Apr  8 13:17:31 PDT 2026

## adaptyv

---
name: adaptyv
author: "K-Dense, Inc."
description: "How to use the Adaptyv Bio Foundry API and Python SDK for protein experiment design, submission, and results retrieval. Use this skill whenever the user mentions Adaptyv, Foundry API, protein binding assays, protein screening experiments, BLI/SPR assays, thermostability assays, or wants to submit protein sequences for experimental characterization. Also trigger when code imports `adaptyv`, `adaptyv_sdk`, or `FoundryClient`, or references `foundry-api-public.adaptyvbio.com`."
---

# Adaptyv Bio Foundry API

Adaptyv Bio is a cloud lab that turns protein sequences into experimental data. Users submit amino acid sequences via API or UI; Adaptyv's automated lab runs assays (binding, thermostability, expression, fluorescence) and delivers results in ~21 days.

## Quick Start

**Base URL:** `https://foundry-api-public.adaptyvbio.com/api/v1`

**Authentication:** Bearer token in the `Authorization` header. Tokens are obtained from [foundry.adaptyvbio.com](https://foundry.adaptyvbio.com/) sidebar.

When writing code, always read the API key from the environment variable `ADAPTYV_API_KEY` or from a `.env` file — never hardcode tokens. Check for a `.env` file in the project root first; if one exists, use a library like `python-dotenv` to load it.

```bash
export FOUNDRY_API_TOKEN="abs0_..."

---

## aeon

---
name: aeon
description: This skill should be used for time series machine learning tasks including classification, regression, clustering, forecasting, anomaly detection, segmentation, and similarity search. Use when working with temporal data, sequential patterns, or time-indexed observations requiring specialized algorithms beyond standard ML approaches. Particularly suited for univariate and multivariate time series analysis with scikit-learn compatible APIs.
license: BSD-3-Clause license
metadata:
    skill-author: K-Dense Inc.
---

# Aeon Time Series Machine Learning

## Overview

Aeon is a scikit-learn compatible Python toolkit for time series machine learning. It provides state-of-the-art algorithms for classification, regression, clustering, forecasting, anomaly detection, segmentation, and similarity search.

## When to Use This Skill

Apply this skill when:
- Classifying or predicting from time series data
- Detecting anomalies or change points in temporal sequences
- Clustering similar time series patterns

---

## anndata

---
name: anndata
description: Data structure for annotated matrices in single-cell analysis. Use when working with .h5ad files or integrating with the scverse ecosystem. This is the data format skill—for analysis workflows use scanpy; for probabilistic models use scvi-tools; for population-scale queries use cellxgene-census.
license: BSD-3-Clause license
metadata:
    skill-author: K-Dense Inc.
---

# AnnData

## Overview

AnnData is a Python package for handling annotated data matrices, storing experimental measurements (X) alongside observation metadata (obs), variable metadata (var), and multi-dimensional annotations (obsm, varm, obsp, varp, uns). Originally designed for single-cell genomics through Scanpy, it now serves as a general-purpose framework for any annotated data requiring efficient storage, manipulation, and analysis.

## When to Use This Skill

Use this skill when:
- Creating, reading, or writing AnnData objects
- Working with h5ad, zarr, or other genomics data formats
- Performing single-cell RNA-seq analysis

---

## arboreto

---
name: arboreto
description: Infer gene regulatory networks (GRNs) from gene expression data using scalable algorithms (GRNBoost2, GENIE3). Use when analyzing transcriptomics data (bulk RNA-seq, single-cell RNA-seq) to identify transcription factor-target gene relationships and regulatory interactions. Supports distributed computation for large-scale datasets.
license: BSD-3-Clause license
metadata:
    skill-author: K-Dense Inc.
---

# Arboreto

## Overview

Arboreto is a computational library for inferring gene regulatory networks (GRNs) from gene expression data using parallelized algorithms that scale from single machines to multi-node clusters.

**Core capability**: Identify which transcription factors (TFs) regulate which target genes based on expression patterns across observations (cells, samples, conditions).

## Quick Start

Install arboreto:
```bash

---

## astropy

---
name: astropy
description: Comprehensive Python library for astronomy and astrophysics. This skill should be used when working with astronomical data including celestial coordinates, physical units, FITS files, cosmological calculations, time systems, tables, world coordinate systems (WCS), and astronomical data analysis. Use when tasks involve coordinate transformations, unit conversions, FITS file manipulation, cosmological distance calculations, time scale conversions, or astronomical data processing.
license: BSD-3-Clause license
metadata:
    skill-author: K-Dense Inc.
---

# Astropy

## Overview

Astropy is the core Python package for astronomy, providing essential functionality for astronomical research and data analysis. Use astropy for coordinate transformations, unit and quantity calculations, FITS file operations, cosmological calculations, precise time handling, tabular data manipulation, and astronomical image processing.

## When to Use This Skill

Use astropy when tasks involve:
- Converting between celestial coordinate systems (ICRS, Galactic, FK5, AltAz, etc.)
- Working with physical units and quantities (converting Jy to mJy, parsecs to km, etc.)
- Reading, writing, or manipulating FITS files (images or tables)

---

## benchling-integration

---
name: benchling-integration
description: Benchling R&D platform integration. Access registry (DNA, proteins), inventory, ELN entries, workflows via API, build Benchling Apps, query Data Warehouse, for lab data management automation.
license: Unknown
compatibility: Requires a Benchling account and API key
metadata:
    skill-author: K-Dense Inc.
---

# Benchling Integration

## Overview

Benchling is a cloud platform for life sciences R&D. Access registry entities (DNA, proteins), inventory, electronic lab notebooks, and workflows programmatically via Python SDK and REST API.

## When to Use This Skill

This skill should be used when:
- Working with Benchling's Python SDK or REST API
- Managing biological sequences (DNA, RNA, proteins) and registry entities

---

## bgpt-paper-search

---
name: bgpt-paper-search
description: Search scientific papers and retrieve structured experimental data extracted from full-text studies via the BGPT MCP server. Returns 25+ fields per paper including methods, results, sample sizes, quality scores, and conclusions. Use for literature reviews, evidence synthesis, and finding experimental details not available in abstracts alone.
allowed-tools: Bash
license: MIT
metadata:
    skill-author: BGPT
    website: https://bgpt.pro/mcp
    github: https://github.com/connerlambden/bgpt-mcp
---

# BGPT Paper Search

## Overview

BGPT is a remote MCP server that searches a curated database of scientific papers built from raw experimental data extracted from full-text studies. Unlike traditional literature databases that return titles and abstracts, BGPT returns structured data from the actual paper content — methods, quantitative results, sample sizes, quality assessments, and 25+ metadata fields per paper.

## When to Use This Skill

Use this skill when:

---

## biopython

---
name: biopython
description: Comprehensive molecular biology toolkit. Use for sequence manipulation, file parsing (FASTA/GenBank/PDB), phylogenetics, and programmatic NCBI/PubMed access (Bio.Entrez). Best for batch processing, custom bioinformatics pipelines, BLAST automation. For quick lookups use gget; for multi-service integration use bioservices.
license: Unknown
metadata:
    skill-author: K-Dense Inc.
---

# Biopython: Computational Molecular Biology in Python

## Overview

Biopython is a comprehensive set of freely available Python tools for biological computation. It provides functionality for sequence manipulation, file I/O, database access, structural bioinformatics, phylogenetics, and many other bioinformatics tasks. The current version is **Biopython 1.85** (released January 2025), which supports Python 3 and requires NumPy.

## When to Use This Skill

Use this skill when:

- Working with biological sequences (DNA, RNA, or protein)
- Reading, writing, or converting biological file formats (FASTA, GenBank, FASTQ, PDB, mmCIF, etc.)

---

## bioservices

---
name: bioservices
description: Unified Python interface to 40+ bioinformatics services. Use when querying multiple databases (UniProt, KEGG, ChEMBL, Reactome) in a single workflow with consistent API. Best for cross-database analysis, ID mapping across services. For quick single-database lookups use gget; for sequence/file manipulation use biopython.
license: GPLv3 license
metadata:
    skill-author: K-Dense Inc.
---

# BioServices

## Overview

BioServices is a Python package providing programmatic access to approximately 40 bioinformatics web services and databases. Retrieve biological data, perform cross-database queries, map identifiers, analyze sequences, and integrate multiple biological resources in Python workflows. The package handles both REST and SOAP/WSDL protocols transparently.

## When to Use This Skill

This skill should be used when:
- Retrieving protein sequences, annotations, or structures from UniProt, PDB, Pfam
- Analyzing metabolic pathways and gene functions via KEGG or Reactome
- Searching compound databases (ChEBI, ChEMBL, PubChem) for chemical information

---

## cellxgene-census

---
name: cellxgene-census
description: Query the CELLxGENE Census (61M+ cells) programmatically. Use when you need expression data across tissues, diseases, or cell types from the largest curated single-cell atlas. Best for population-scale queries, reference atlas comparisons. For analyzing your own data use scanpy or scvi-tools.
license: Unknown
metadata:
    skill-author: K-Dense Inc.
---

# CZ CELLxGENE Census

## Overview

The CZ CELLxGENE Census provides programmatic access to a comprehensive, versioned collection of standardized single-cell genomics data from CZ CELLxGENE Discover. This skill enables efficient querying and analysis of millions of cells across thousands of datasets.

The Census includes:
- **61+ million cells** from human and mouse
- **Standardized metadata** (cell types, tissues, diseases, donors)
- **Raw gene expression** matrices
- **Pre-calculated embeddings** and statistics
- **Integration with PyTorch, scanpy, and other analysis tools**

---

## cirq

---
name: cirq
description: Google quantum computing framework. Use when targeting Google Quantum AI hardware, designing noise-aware circuits, or running quantum characterization experiments. Best for Google hardware, noise modeling, and low-level circuit design. For IBM hardware use qiskit; for quantum ML with autodiff use pennylane; for physics simulations use qutip.
license: Apache-2.0 license
metadata:
    skill-author: K-Dense Inc.
---

# Cirq - Quantum Computing with Python

Cirq is Google Quantum AI's open-source framework for designing, simulating, and running quantum circuits on quantum computers and simulators.

## Installation

```bash
uv pip install cirq
```

For hardware integration:
```bash

---

## citation-management

---
name: citation-management
description: Comprehensive citation management for academic research. Search Google Scholar and PubMed for papers, extract accurate metadata, validate citations, and generate properly formatted BibTeX entries. This skill should be used when you need to find papers, verify citation information, convert DOIs to BibTeX, or ensure reference accuracy in scientific writing.
allowed-tools: Read Write Edit Bash
license: MIT License
metadata:
    skill-author: K-Dense Inc.
---

# Citation Management

## Overview

Manage citations systematically throughout the research and writing process. This skill provides tools and strategies for searching academic databases (Google Scholar, PubMed), extracting accurate metadata from multiple sources (CrossRef, PubMed, arXiv), validating citation information, and generating properly formatted BibTeX entries.

Critical for maintaining citation accuracy, avoiding reference errors, and ensuring reproducible research. Integrates seamlessly with the literature-review skill for comprehensive research workflows.

## When to Use This Skill

Use this skill when:

---

## clinical-decision-support

---
name: clinical-decision-support
description: Generate professional clinical decision support (CDS) documents for pharmaceutical and clinical research settings, including patient cohort analyses (biomarker-stratified with outcomes) and treatment recommendation reports (evidence-based guidelines with decision algorithms). Supports GRADE evidence grading, statistical analysis (hazard ratios, survival curves, waterfall plots), biomarker integration, and regulatory compliance. Outputs publication-ready LaTeX/PDF format optimized for drug development, clinical research, and evidence synthesis.
allowed-tools: Read Write Edit Bash
license: MIT License
metadata:
    skill-author: K-Dense Inc.
---

# Clinical Decision Support Documents

## Description

Generate professional clinical decision support (CDS) documents for pharmaceutical companies, clinical researchers, and medical decision-makers. This skill specializes in analytical, evidence-based documents that inform treatment strategies and drug development:

1. **Patient Cohort Analysis** - Biomarker-stratified group analyses with statistical outcome comparisons
2. **Treatment Recommendation Reports** - Evidence-based clinical guidelines with GRADE grading and decision algorithms

All documents are generated as publication-ready LaTeX/PDF files optimized for pharmaceutical research, regulatory submissions, and clinical guideline development.


---

## clinical-reports

---
name: clinical-reports
description: Write comprehensive clinical reports including case reports (CARE guidelines), diagnostic reports (radiology/pathology/lab), clinical trial reports (ICH-E3, SAE, CSR), and patient documentation (SOAP, H&P, discharge summaries). Full support with templates, regulatory compliance (HIPAA, FDA, ICH-GCP), and validation tools.
allowed-tools: Read Write Edit Bash
license: MIT License
metadata:
    skill-author: K-Dense Inc.
---

# Clinical Report Writing

## Overview

Clinical report writing is the process of documenting medical information with precision, accuracy, and compliance with regulatory standards. This skill covers four major categories of clinical reports: case reports for journal publication, diagnostic reports for clinical practice, clinical trial reports for regulatory submission, and patient documentation for medical records. Apply this skill for healthcare documentation, research dissemination, and regulatory compliance.

**Critical Principle: Clinical reports must be accurate, complete, objective, and compliant with applicable regulations (HIPAA, FDA, ICH-GCP).** Patient privacy and data integrity are paramount. All clinical documentation must support evidence-based decision-making and meet professional standards.

## When to Use This Skill

This skill should be used when:

---

## cobrapy

---
name: cobrapy
description: Constraint-based metabolic modeling (COBRA). FBA, FVA, gene knockouts, flux sampling, SBML models, for systems biology and metabolic engineering analysis.
license: GPL-2.0 license
metadata:
    skill-author: K-Dense Inc.
---

# COBRApy - Constraint-Based Reconstruction and Analysis

## Overview

COBRApy is a Python library for constraint-based reconstruction and analysis (COBRA) of metabolic models, essential for systems biology research. Work with genome-scale metabolic models, perform computational simulations of cellular metabolism, conduct metabolic engineering analyses, and predict phenotypic behaviors.

## Core Capabilities

COBRApy provides comprehensive tools organized into several key areas:

### 1. Model Management


---

## consciousness-council

---
name: consciousness-council
description: Run a multi-perspective Mind Council deliberation on any question, decision, or creative challenge. Use this skill whenever the user wants diverse viewpoints, needs help making a tough decision, asks for a council/panel/board discussion, wants to explore a problem from multiple angles, requests devil's advocate analysis, or says things like "what would different experts think about this", "help me think through this from all sides", "council mode", "mind council", or "deliberate on this". Also trigger when the user faces a dilemma, trade-off, or complex choice with no obvious answer.
allowed-tools: Read Write
license: MIT license
metadata:
  skill-author: AHK Strategies (ashrafkahoush-ux)
---

# Consciousness Council

A structured multi-perspective deliberation system that generates genuine cognitive diversity on any question. Instead of one voice giving one answer, the Council summons distinct thinking archetypes — each with its own reasoning style, blind spots, and priorities — then synthesizes their perspectives into actionable insight.

## Why This Exists

Single-perspective thinking has a ceiling. When you ask one mind for an answer, you get one frame. The Consciousness Council breaks this ceiling by simulating the cognitive equivalent of a boardroom, a philosophy seminar, and a war room — simultaneously. It's not roleplay. It's structured epistemic diversity.

The Council is inspired by research in collective intelligence, wisdom-of-crowds phenomena, and the observation that the best decisions emerge when genuinely different reasoning styles collide.

## How It Works

---

## dask

---
name: dask
description: Distributed computing for larger-than-RAM pandas/NumPy workflows. Use when you need to scale existing pandas/NumPy code beyond memory or across clusters. Best for parallel file processing, distributed ML, integration with existing pandas code. For out-of-core analytics on single machine use vaex; for in-memory speed use polars.
license: BSD-3-Clause license
metadata:
    skill-author: K-Dense Inc.
---

# Dask

## Overview

Dask is a Python library for parallel and distributed computing that enables three critical capabilities:
- **Larger-than-memory execution** on single machines for data exceeding available RAM
- **Parallel processing** for improved computational speed across multiple cores
- **Distributed computation** supporting terabyte-scale datasets across multiple machines

Dask scales from laptops (processing ~100 GiB) to clusters (processing ~100 TiB) while maintaining familiar Python APIs.

## When to Use This Skill

---

## database-lookup

---
name: database-lookup
description: Search 78 public scientific, biomedical, materials science, and economic databases via their REST APIs and return structured JSON results. Covers physics/astronomy (NASA, NIST, SDSS, SIMBAD, Exoplanet Archive), earth/environment (USGS, NOAA, EPA, OpenWeatherMap), chemistry/drugs (PubChem, ChEMBL, DrugBank, FDA, KEGG, DailyMed, ZINC, BindingDB), materials science (Materials Project, COD), biology/genomics (Reactome, BRENDA, UniProt, STRING, Ensembl, NCBI Gene, GEO, GTEx, PDB, AlphaFold, InterPro, ChEBI, BioGRID, Gene Ontology, QuickGO, NCBI Protein/Taxonomy, dbSNP, SRA, ENA, gnomAD, UCSC Genome, ENCODE, JASPAR, MouseMine, PRIDE, LINCS L1000, Human Protein Atlas, Human Cell Atlas, RummaGEO, Metabolomics Workbench, EMDB, Addgene), disease/clinical (COSMIC, Open Targets, ClinPGx, ClinicalTrials.gov, OMIM, ClinVar, GDC/TCGA, cBioPortal, DisGeNET, GWAS Catalog, Monarch, HPO), regulatory (FDA, USPTO, SEC EDGAR), economics/finance (FRED, BEA, BLS, Federal Reserve, World Bank, ECB, US Treasury, Alpha Vantage, Data Commons), and demographics (US Census, Eurostat, WHO). Use this skill whenever the user wants to look up compounds, drugs, proteins, genes, pathways, enzymes, gene expression, variants, clinical trials, patents, SEC filings, economic indicators, crystal structures, astronomical objects, earthquakes, weather, or any data from a public database API. Also trigger when the user mentions any database by name or asks about molecular properties, drug-target interactions, binding affinities, protein interactions, pathway membership, pharmacogenomics, economic time series, materials properties, commercially available compounds, virtual screening, compound purchasability, chemical libraries, building blocks, cancer genomics, somatic mutations, tumor mutation profiles, nucleotide sequences, genome assemblies, sequencing reads, ENA accessions, INSDC data, or wants to cross-reference entities across sources.
metadata:
  skill-author: K-Dense Inc.
---

# Database Lookup

You have access to 78 public databases through their REST APIs. Your job is to figure out which database(s) are relevant to the user's question, query them, and return the raw JSON results along with which databases you used.

## Core Workflow

1. **Understand the query** — What is the user looking for? A compound? A gene? A pathway? A patent? Expression data? An economic indicator? This determines which database(s) to hit.

2. **Select database(s)** — Use the database selection guide below. When in doubt, search multiple databases — it's better to cast a wide net than to miss relevant data.

3. **Read the reference file** — Each database has a reference file in `references/` with endpoint details, query formats, and example calls. Read the relevant file(s) before making API calls.

4. **Make the API call(s)** — See the **Making API Calls** section below for which HTTP fetch tool to use on your platform.

---

## datamol

---
name: datamol
description: Pythonic wrapper around RDKit with simplified interface and sensible defaults. Preferred for standard drug discovery including SMILES parsing, standardization, descriptors, fingerprints, clustering, 3D conformers, parallel processing. Returns native rdkit.Chem.Mol objects. For advanced control or custom parameters, use rdkit directly.
license: Apache-2.0 license
metadata:
    skill-author: K-Dense Inc.
---

# Datamol Cheminformatics Skill

## Overview

Datamol is a Python library that provides a lightweight, Pythonic abstraction layer over RDKit for molecular cheminformatics. Simplify complex molecular operations with sensible defaults, efficient parallelization, and modern I/O capabilities. All molecular objects are native `rdkit.Chem.Mol` instances, ensuring full compatibility with the RDKit ecosystem.

**Key capabilities**:
- Molecular format conversion (SMILES, SELFIES, InChI)
- Structure standardization and sanitization
- Molecular descriptors and fingerprints
- 3D conformer generation and analysis
- Clustering and diversity selection

---

## deepchem

---
name: deepchem
description: Molecular ML with diverse featurizers and pre-built datasets. Use for property prediction (ADMET, toxicity) with traditional ML or GNNs when you want extensive featurization options and MoleculeNet benchmarks. Best for quick experiments with pre-trained models, diverse molecular representations. For graph-first PyTorch workflows use torchdrug; for benchmark datasets use pytdc.
license: MIT license
metadata:
    skill-author: K-Dense Inc.
---

# DeepChem

## Overview

DeepChem is a comprehensive Python library for applying machine learning to chemistry, materials science, and biology. Enable molecular property prediction, drug discovery, materials design, and biomolecule analysis through specialized neural networks, molecular featurization methods, and pretrained models.

## When to Use This Skill

This skill should be used when:
- Loading and processing molecular data (SMILES strings, SDF files, protein sequences)
- Predicting molecular properties (solubility, toxicity, binding affinity, ADMET properties)
- Training models on chemical/biological datasets

---

## deeptools

---
name: deeptools
description: NGS analysis toolkit. BAM to bigWig conversion, QC (correlation, PCA, fingerprints), heatmaps/profiles (TSS, peaks), for ChIP-seq, RNA-seq, ATAC-seq visualization.
license: BSD license
metadata:
    skill-author: K-Dense Inc.
---

# deepTools: NGS Data Analysis Toolkit

## Overview

deepTools is a comprehensive suite of Python command-line tools designed for processing and analyzing high-throughput sequencing data. Use deepTools to perform quality control, normalize data, compare samples, and generate publication-quality visualizations for ChIP-seq, RNA-seq, ATAC-seq, MNase-seq, and other NGS experiments.

**Core capabilities:**
- Convert BAM alignments to normalized coverage tracks (bigWig/bedGraph)
- Quality control assessment (fingerprint, correlation, coverage)
- Sample comparison and correlation analysis
- Heatmap and profile plot generation around genomic features
- Enrichment analysis and peak region visualization

---

## depmap

---
name: depmap
description: Query the Cancer Dependency Map (DepMap) for cancer cell line gene dependency scores (CRISPR Chronos), drug sensitivity data, and gene effect profiles. Use for identifying cancer-specific vulnerabilities, synthetic lethal interactions, and validating oncology drug targets.
license: CC-BY-4.0
metadata:
    skill-author: Kuan-lin Huang
---

# DepMap — Cancer Dependency Map

## Overview

The Cancer Dependency Map (DepMap) project, run by the Broad Institute, systematically characterizes genetic dependencies across hundreds of cancer cell lines using genome-wide CRISPR knockout screens (DepMap CRISPR), RNA interference (RNAi), and compound sensitivity assays (PRISM). DepMap data is essential for:
- Identifying which genes are essential for specific cancer types
- Finding cancer-selective dependencies (therapeutic targets)
- Validating oncology drug targets
- Discovering synthetic lethal interactions

**Key resources:**
- DepMap Portal: https://depmap.org/portal/

---

## dhdna-profiler

---
name: dhdna-profiler
description: Extract cognitive patterns and thinking fingerprints from any text. Use this skill when the user wants to analyze how someone thinks, understand cognitive style, profile writing or speech patterns, compare thinking styles between people, asks "what's my thinking style", "analyze how this person reasons", "cognitive profile", "thinking pattern", "DHDNA", "digital DNA", or wants to understand the mind behind any text. Also trigger when the user provides text and wants deeper insight into the author's reasoning patterns, decision-making style, or cognitive signature.
allowed-tools: Read Write
license: MIT license
metadata:
  skill-author: AHK Strategies (ashrafkahoush-ux)
---

# DHDNA Profiler — Cognitive Pattern Extraction

A structured system for extracting the cognitive fingerprint of any text's author. Based on the Digital Human DNA (DHDNA) framework — the theory that every mind has a unique signature pattern expressed through how it reasons, decides, values, and communicates.

Published research: [DHDNA Pre-print (DOI: 10.5281/zenodo.18736629)](https://doi.org/10.5281/zenodo.18736629) | [IDNA Consolidation v2 (DOI: 10.5281/zenodo.18807387)](https://doi.org/10.5281/zenodo.18807387)

## Core Concept

Just as biological DNA encodes physical identity through base pairs, Digital Human DNA encodes cognitive identity through thinking patterns. Every person's combination of analytical depth, creative range, emotional processing, strategic thinking, and ethical reasoning creates a **unique cognitive signature** — as distinctive as a fingerprint.

The profiler doesn't judge thinking as "good" or "bad." It maps the topology of how a mind works.

---

## diffdock

---
name: diffdock
description: Diffusion-based molecular docking. Predict protein-ligand binding poses from PDB/SMILES, confidence scores, virtual screening, for structure-based drug design. Not for affinity prediction.
license: MIT license
metadata:
    skill-author: K-Dense Inc.
---

# DiffDock: Molecular Docking with Diffusion Models

## Overview

DiffDock is a diffusion-based deep learning tool for molecular docking that predicts 3D binding poses of small molecule ligands to protein targets. It represents the state-of-the-art in computational docking, crucial for structure-based drug discovery and chemical biology.

**Core Capabilities:**
- Predict ligand binding poses with high accuracy using deep learning
- Support protein structures (PDB files) or sequences (via ESMFold)
- Process single complexes or batch virtual screening campaigns
- Generate confidence scores to assess prediction reliability
- Handle diverse ligand inputs (SMILES, SDF, MOL2)

---

## dnanexus-integration

---
name: dnanexus-integration
description: DNAnexus cloud genomics platform. Build apps/applets, manage data (upload/download), dxpy Python SDK, run workflows, FASTQ/BAM/VCF, for genomics pipeline development and execution.
license: Unknown
compatibility: Requires a DNAnexus account
metadata:
    skill-author: K-Dense Inc.
---

# DNAnexus Integration

## Overview

DNAnexus is a cloud platform for biomedical data analysis and genomics. Build and deploy apps/applets, manage data objects, run workflows, and use the dxpy Python SDK for genomics pipeline development and execution.

## When to Use This Skill

This skill should be used when:
- Creating, building, or modifying DNAnexus apps/applets
- Uploading, downloading, searching, or organizing files and records

---

## docx

---
name: docx
description: "Use this skill whenever the user wants to create, read, edit, or manipulate Word documents (.docx files). Triggers include: any mention of 'Word doc', 'word document', '.docx', or requests to produce professional documents with formatting like tables of contents, headings, page numbers, or letterheads. Also use when extracting or reorganizing content from .docx files, inserting or replacing images in documents, performing find-and-replace in Word files, working with tracked changes or comments, or converting content into a polished Word document. If the user asks for a 'report', 'memo', 'letter', 'template', or similar deliverable as a Word or .docx file, use this skill. Do NOT use for PDFs, spreadsheets, Google Docs, or general coding tasks unrelated to document generation."
license: Proprietary. LICENSE.txt has complete terms
---

# DOCX creation, editing, and analysis

## Overview

A .docx file is a ZIP archive containing XML files.

## Quick Reference

| Task | Approach |
|------|----------|
| Read/analyze content | `pandoc` or unpack for raw XML |
| Create new document | Use `docx-js` - see Creating New Documents below |
| Edit existing document | Unpack → edit XML → repack - see Editing Existing Documents below |


---

## esm

---
name: esm
description: Comprehensive toolkit for protein language models including ESM3 (generative multimodal protein design across sequence, structure, and function) and ESM C (efficient protein embeddings and representations). Use this skill when working with protein sequences, structures, or function prediction; designing novel proteins; generating protein embeddings; performing inverse folding; or conducting protein engineering tasks. Supports both local model usage and cloud-based Forge API for scalable inference.
license: MIT license
metadata:
    skill-author: K-Dense Inc.
---

# ESM: Evolutionary Scale Modeling

## Overview

ESM provides state-of-the-art protein language models for understanding, generating, and designing proteins. This skill enables working with two model families: ESM3 for generative protein design across sequence, structure, and function, and ESM C for efficient protein representation learning and embeddings.

## Core Capabilities

### 1. Protein Sequence Generation with ESM3

Generate novel protein sequences with desired properties using multimodal generative modeling.


---

## etetoolkit

---
name: etetoolkit
description: Phylogenetic tree toolkit (ETE). Tree manipulation (Newick/NHX), evolutionary event detection, orthology/paralogy, NCBI taxonomy, visualization (PDF/SVG), for phylogenomics.
license: GPL-3.0 license
metadata:
    skill-author: K-Dense Inc.
---

# ETE Toolkit Skill

## Overview

ETE (Environment for Tree Exploration) is a toolkit for phylogenetic and hierarchical tree analysis. Manipulate trees, analyze evolutionary events, visualize results, and integrate with biological databases for phylogenomic research and clustering analysis.

## Core Capabilities

### 1. Tree Manipulation and Analysis

Load, manipulate, and analyze hierarchical tree structures with support for:


---

## exploratory-data-analysis

---
name: exploratory-data-analysis
description: Perform comprehensive exploratory data analysis on scientific data files across 200+ file formats. This skill should be used when analyzing any scientific data file to understand its structure, content, quality, and characteristics. Automatically detects file type and generates detailed markdown reports with format-specific analysis, quality metrics, and downstream analysis recommendations. Covers chemistry, bioinformatics, microscopy, spectroscopy, proteomics, metabolomics, and general scientific data formats.
license: MIT license
metadata:
    skill-author: K-Dense Inc.
---

# Exploratory Data Analysis

## Overview

Perform comprehensive exploratory data analysis (EDA) on scientific data files across multiple domains. This skill provides automated file type detection, format-specific analysis, data quality assessment, and generates detailed markdown reports suitable for documentation and downstream analysis planning.

**Key Capabilities:**
- Automatic detection and analysis of 200+ scientific file formats
- Comprehensive format-specific metadata extraction
- Data quality and integrity assessment
- Statistical summaries and distributions
- Visualization recommendations

---

## flowio

---
name: flowio
description: Parse FCS (Flow Cytometry Standard) files v2.0-3.1. Extract events as NumPy arrays, read metadata/channels, convert to CSV/DataFrame, for flow cytometry data preprocessing.
license: BSD-3-Clause license
metadata:
    skill-author: K-Dense Inc.
---

# FlowIO: Flow Cytometry Standard File Handler

## Overview

FlowIO is a lightweight Python library for reading and writing Flow Cytometry Standard (FCS) files. Parse FCS metadata, extract event data, and create new FCS files with minimal dependencies. The library supports FCS versions 2.0, 3.0, and 3.1, making it ideal for backend services, data pipelines, and basic cytometry file operations.

## When to Use This Skill

This skill should be used when:

- FCS files requiring parsing or metadata extraction
- Flow cytometry data needing conversion to NumPy arrays

---

## fluidsim

---
name: fluidsim
description: Framework for computational fluid dynamics simulations using Python. Use when running fluid dynamics simulations including Navier-Stokes equations (2D/3D), shallow water equations, stratified flows, or when analyzing turbulence, vortex dynamics, or geophysical flows. Provides pseudospectral methods with FFT, HPC support, and comprehensive output analysis.
license: CeCILL FREE SOFTWARE LICENSE AGREEMENT
metadata:
    skill-author: K-Dense Inc.
---

# FluidSim

## Overview

FluidSim is an object-oriented Python framework for high-performance computational fluid dynamics (CFD) simulations. It provides solvers for periodic-domain equations using pseudospectral methods with FFT, delivering performance comparable to Fortran/C++ while maintaining Python's ease of use.

**Key strengths**:
- Multiple solvers: 2D/3D Navier-Stokes, shallow water, stratified flows
- High performance: Pythran/Transonic compilation, MPI parallelization
- Complete workflow: Parameter configuration, simulation execution, output analysis
- Interactive analysis: Python-based post-processing and visualization


---

## generate-image

---
name: generate-image
description: Generate or edit images using AI models (FLUX, Nano Banana 2). Use for general-purpose image generation including photos, illustrations, artwork, visual assets, concept art, and any image that is not a technical diagram or schematic. For flowcharts, circuits, pathways, and technical diagrams, use the scientific-schematics skill instead.
license: MIT license
compatibility: Requires an OpenRouter API key
metadata:
    skill-author: K-Dense Inc.
---

# Generate Image

Generate and edit high-quality images using OpenRouter's image generation models including FLUX.2 Pro and Gemini 3.1 Flash Image Preview.

## When to Use This Skill

**Use generate-image for:**
- Photos and photorealistic images
- Artistic illustrations and artwork
- Concept art and visual concepts
- Visual assets for presentations or documents

---

## geniml

---
name: geniml
description: This skill should be used when working with genomic interval data (BED files) for machine learning tasks. Use for training region embeddings (Region2Vec, BEDspace), single-cell ATAC-seq analysis (scEmbed), building consensus peaks (universes), or any ML-based analysis of genomic regions. Applies to BED file collections, scATAC-seq data, chromatin accessibility datasets, and region-based genomic feature learning.
license: BSD-2-Clause license
metadata:
    skill-author: K-Dense Inc.
---

# Geniml: Genomic Interval Machine Learning

## Overview

Geniml is a Python package for building machine learning models on genomic interval data from BED files. It provides unsupervised methods for learning embeddings of genomic regions, single cells, and metadata labels, enabling similarity searches, clustering, and downstream ML tasks.

## Installation

Install geniml using uv:

```bash
uv uv pip install geniml

---

## geomaster

---
name: geomaster
description: Comprehensive geospatial science skill covering remote sensing, GIS, spatial analysis, machine learning for earth observation, and 30+ scientific domains. Supports satellite imagery processing (Sentinel, Landsat, MODIS, SAR, hyperspectral), vector and raster data operations, spatial statistics, point cloud processing, network analysis, cloud-native workflows (STAC, COG, Planetary Computer), and 8 programming languages (Python, R, Julia, JavaScript, C++, Java, Go, Rust) with 500+ code examples. Use for remote sensing workflows, GIS analysis, spatial ML, Earth observation data processing, terrain analysis, hydrological modeling, marine spatial analysis, atmospheric science, and any geospatial computation task.
license: MIT License
metadata:
    skill-author: K-Dense Inc.
---

# GeoMaster

Comprehensive geospatial science skill covering GIS, remote sensing, spatial analysis, and ML for Earth observation across 70+ topics with 500+ code examples in 8 programming languages.

## Installation

```bash
# Core Python stack (conda recommended)
conda install -c conda-forge gdal rasterio fiona shapely pyproj geopandas

# Remote sensing & ML
uv pip install rsgislib torchgeo earthengine-api

---

## geopandas

---
name: geopandas
description: Python library for working with geospatial vector data including shapefiles, GeoJSON, and GeoPackage files. Use when working with geographic data for spatial analysis, geometric operations, coordinate transformations, spatial joins, overlay operations, choropleth mapping, or any task involving reading/writing/analyzing vector geographic data. Supports PostGIS databases, interactive maps, and integration with matplotlib/folium/cartopy. Use for tasks like buffer analysis, spatial joins between datasets, dissolving boundaries, clipping data, calculating areas/distances, reprojecting coordinate systems, creating maps, or converting between spatial file formats.
license: BSD-3-Clause license
metadata:
    skill-author: K-Dense Inc.
---

# GeoPandas

GeoPandas extends pandas to enable spatial operations on geometric types. It combines the capabilities of pandas and shapely for geospatial data analysis.

## Installation

```bash
uv pip install geopandas
```

### Optional Dependencies


---

## get-available-resources

---
name: get-available-resources
description: This skill should be used at the start of any computationally intensive scientific task to detect and report available system resources (CPU cores, GPUs, memory, disk space). It creates a JSON file with resource information and strategic recommendations that inform computational approach decisions such as whether to use parallel processing (joblib, multiprocessing), out-of-core computing (Dask, Zarr), GPU acceleration (PyTorch, JAX), or memory-efficient strategies. Use this skill before running analyses, training models, processing large datasets, or any task where resource constraints matter.
license: MIT license
metadata:
    skill-author: K-Dense Inc.
---

# Get Available Resources

## Overview

Detect available computational resources and generate strategic recommendations for scientific computing tasks. This skill automatically identifies CPU capabilities, GPU availability (NVIDIA CUDA, AMD ROCm, Apple Silicon Metal), memory constraints, and disk space to help make informed decisions about computational approaches.

## When to Use This Skill

Use this skill proactively before any computationally intensive task:

- **Before data analysis**: Determine if datasets can be loaded into memory or require out-of-core processing
- **Before model training**: Check if GPU acceleration is available and which backend to use

---

## gget

---
name: gget
description: "Fast CLI/Python queries to 20+ bioinformatics databases. Use for quick lookups: gene info, BLAST searches, AlphaFold structures, enrichment analysis. Best for interactive exploration, simple queries. For batch processing or advanced BLAST use biopython; for multi-database Python workflows use bioservices."
license: BSD-2-Clause license
metadata:
    skill-author: K-Dense Inc.
---

# gget

## Overview

gget is a command-line bioinformatics tool and Python package providing unified access to 20+ genomic databases and analysis methods. Query gene information, sequence analysis, protein structures, expression data, and disease associations through a consistent interface. All gget modules work both as command-line tools and as Python functions.

**Important**: The databases queried by gget are continuously updated, which sometimes changes their structure. gget modules are tested automatically on a biweekly basis and updated to match new database structures when necessary.

## Installation

Install gget in a clean virtual environment to avoid conflicts:


---

## ginkgo-cloud-lab

---
name: ginkgo-cloud-lab
description: Submit and manage protocols on Ginkgo Bioworks Cloud Lab (cloud.ginkgo.bio), a web-based interface for autonomous lab execution on Reconfigurable Automation Carts (RACs). Use when the user wants to run cell-free protein expression (validation or optimization), generate fluorescent pixel art, or interact with Ginkgo Cloud Lab services. Covers protocol selection, input preparation, pricing, and ordering workflows.
---

# Ginkgo Cloud Lab

## Overview

Ginkgo Cloud Lab (https://cloud.ginkgo.bio) provides remote access to Ginkgo Bioworks' autonomous lab infrastructure. Protocols are executed on Reconfigurable Automation Carts (RACs) -- modular units with robotic arms, maglev sample transport, and industrial-grade software spanning 70+ instruments.

The platform also includes **EstiMate**, an AI agent that accepts human-language protocol descriptions and returns feasibility assessments and pricing for custom workflows beyond the listed protocols.

## Available Protocols

### 1. Cell Free Protein Expression Validation

Rapid go/no-go expression screening using reconstituted E. coli CFPS. Submit a FASTA sequence (up to 1800 bp) and receive expression confirmation, baseline titer (mg/L), and initial purity with virtual gel images.

- **Price:** $39/sample | **Turnaround:** 5-10 days | **Status:** Certified

---

## glycoengineering

---
name: glycoengineering
description: Analyze and engineer protein glycosylation. Scan sequences for N-glycosylation sequons (N-X-S/T), predict O-glycosylation hotspots, and access curated glycoengineering tools (NetOGlyc, GlycoShield, GlycoWorkbench). For glycoprotein engineering, therapeutic antibody optimization, and vaccine design.
license: Unknown
metadata:
    skill-author: Kuan-lin Huang
---

# Glycoengineering

## Overview

Glycosylation is the most common and complex post-translational modification (PTM) of proteins, affecting over 50% of all human proteins. Glycans regulate protein folding, stability, immune recognition, receptor interactions, and pharmacokinetics of therapeutic proteins. Glycoengineering involves rational modification of glycosylation patterns for improved therapeutic efficacy, stability, or immune evasion.

**Two major glycosylation types:**
- **N-glycosylation**: Attached to asparagine (N) in the sequon N-X-[S/T] where X ≠ Proline; occurs in the ER/Golgi
- **O-glycosylation**: Attached to serine (S) or threonine (T); no strict consensus motif; primarily GalNAc initiation

## When to Use This Skill


---

## gtars

---
name: gtars
description: High-performance toolkit for genomic interval analysis in Rust with Python bindings. Use when working with genomic regions, BED files, coverage tracks, overlap detection, tokenization for ML models, or fragment analysis in computational genomics and machine learning applications.
license: Unknown
metadata:
    skill-author: K-Dense Inc.
---

# Gtars: Genomic Tools and Algorithms in Rust

## Overview

Gtars is a high-performance Rust toolkit for manipulating, analyzing, and processing genomic interval data. It provides specialized tools for overlap detection, coverage analysis, tokenization for machine learning, and reference sequence management.

Use this skill when working with:
- Genomic interval files (BED format)
- Overlap detection between genomic regions
- Coverage track generation (WIG, BigWig)
- Genomic ML preprocessing and tokenization
- Fragment analysis in single-cell genomics

---

## histolab

---
name: histolab
description: Lightweight WSI tile extraction and preprocessing. Use for basic slide processing tissue detection, tile extraction, stain normalization for H&E images. Best for simple pipelines, dataset preparation, quick tile-based analysis. For advanced spatial proteomics, multiplexed imaging, or deep learning pipelines use pathml.
license: Apache-2.0 license
metadata:
    skill-author: K-Dense Inc.
---

# Histolab

## Overview

Histolab is a Python library for processing whole slide images (WSI) in digital pathology. It automates tissue detection, extracts informative tiles from gigapixel images, and prepares datasets for deep learning pipelines. The library handles multiple WSI formats, implements sophisticated tissue segmentation, and provides flexible tile extraction strategies.

## Installation

```bash
uv pip install histolab
```


---

## hypogenic

---
name: hypogenic
description: Automated LLM-driven hypothesis generation and testing on tabular datasets. Use when you want to systematically explore hypotheses about patterns in empirical data (e.g., deception detection, content analysis). Combines literature insights with data-driven hypothesis testing. For manual hypothesis formulation use hypothesis-generation; for creative ideation use scientific-brainstorming.
license: MIT license
metadata:
    skill-author: K-Dense Inc.
---

# Hypogenic

## Overview

Hypogenic provides automated hypothesis generation and testing using large language models to accelerate scientific discovery. The framework supports three approaches: HypoGeniC (data-driven hypothesis generation), HypoRefine (synergistic literature and data integration), and Union methods (mechanistic combination of literature and data-driven hypotheses).

## Quick Start

Get started with Hypogenic in minutes:

```bash
# Install the package

---

## hypothesis-generation

---
name: hypothesis-generation
description: Structured hypothesis formulation from observations. Use when you have experimental observations or data and need to formulate testable hypotheses with predictions, propose mechanisms, and design experiments to test them. Follows scientific method framework. For open-ended ideation use scientific-brainstorming; for automated LLM-driven hypothesis testing on datasets use hypogenic.
allowed-tools: Read Write Edit Bash
license: MIT license
metadata:
    skill-author: K-Dense Inc.
---

# Scientific Hypothesis Generation

## Overview

Hypothesis generation is a systematic process for developing testable explanations. Formulate evidence-based hypotheses from observations, design experiments, explore competing explanations, and develop predictions. Apply this skill for scientific inquiry across domains.

## When to Use This Skill

This skill should be used when:
- Developing hypotheses from observations or preliminary data
- Designing experiments to test scientific questions

---

## imaging-data-commons

---
name: imaging-data-commons
description: Query and download public cancer imaging data from NCI Imaging Data Commons using idc-index. Use for accessing large-scale radiology (CT, MR, PET) and pathology datasets for AI training or research. No authentication required. Query by metadata, visualize in browser, check licenses.
license: This skill is provided under the MIT License. IDC data itself has individual licensing (mostly CC-BY, some CC-NC) that must be respected when using the data.
metadata:
    version: 1.4.0
    skill-author: Andrey Fedorov, @fedorov
    idc-index: "0.11.10"
    idc-data-version: "v23"
    repository: https://github.com/ImagingDataCommons/idc-claude-skill
---

# Imaging Data Commons

## Overview

Use the `idc-index` Python package to query and download public cancer imaging data from the National Cancer Institute Imaging Data Commons (IDC). No authentication required for data access.

**Current IDC Data Version: v23** (always verify with `IDCClient().get_idc_version()`)


---

## infographics

---
name: infographics
description: "Create professional infographics using Nano Banana Pro AI with smart iterative refinement. Uses Gemini 3 Pro for quality review. Integrates research-lookup and web search for accurate data. Supports 10 infographic types, 8 industry styles, and colorblind-safe palettes."
allowed-tools: Read Write Edit Bash
---

# Infographics

## Overview

Infographics are visual representations of information, data, or knowledge designed to present complex content quickly and clearly. **This skill uses Nano Banana Pro AI for infographic generation with Gemini 3 Pro quality review and Perplexity Sonar for research.**

**How it works:**
- (Optional) **Research phase**: Gather accurate facts and statistics using Perplexity Sonar
- Describe your infographic in natural language
- Nano Banana Pro generates publication-quality infographics automatically
- **Gemini 3 Pro reviews quality** against document-type thresholds
- **Smart iteration**: Only regenerates if quality is below threshold
- Professional-ready output in minutes
- No design skills required

---

## iso-13485-certification

---
name: iso-13485-certification
description: Comprehensive toolkit for preparing ISO 13485 certification documentation for medical device Quality Management Systems. Use when users need help with ISO 13485 QMS documentation, including (1) conducting gap analysis of existing documentation, (2) creating Quality Manuals, (3) developing required procedures and work instructions, (4) preparing Medical Device Files, (5) understanding ISO 13485 requirements, or (6) identifying missing documentation for medical device certification. Also use when users mention medical device regulations, QMS certification, FDA QMSR, EU MDR, or need help with quality system documentation.
license: MIT license
metadata:
    skill-author: K-Dense Inc.
---

# ISO 13485 Certification Documentation Assistant

## Overview

This skill helps medical device manufacturers prepare comprehensive documentation for ISO 13485:2016 certification. It provides tools, templates, references, and guidance to create, review, and gap-analyze all required Quality Management System (QMS) documentation.

**What this skill provides:**
- Gap analysis of existing documentation
- Templates for all mandatory documents
- Comprehensive requirements guidance
- Step-by-step documentation creation
- Identification of missing documentation

---

## labarchive-integration

---
name: labarchive-integration
description: Electronic lab notebook API integration. Access notebooks, manage entries/attachments, backup notebooks, integrate with Protocols.io/Jupyter/REDCap, for programmatic ELN workflows.
license: Unknown
metadata:
    skill-author: K-Dense Inc.
---

# LabArchives Integration

## Overview

LabArchives is an electronic lab notebook platform for research documentation and data management. Access notebooks, manage entries and attachments, generate reports, and integrate with third-party tools programmatically via REST API.

## When to Use This Skill

This skill should be used when:
- Working with LabArchives REST API for notebook automation
- Backing up notebooks programmatically
- Creating or managing notebook entries and attachments

---

## lamindb

---
name: lamindb
description: This skill should be used when working with LaminDB, an open-source data framework for biology that makes data queryable, traceable, reproducible, and FAIR. Use when managing biological datasets (scRNA-seq, spatial, flow cytometry, etc.), tracking computational workflows, curating and validating data with biological ontologies, building data lakehouses, or ensuring data lineage and reproducibility in biological research. Covers data management, annotation, ontologies (genes, cell types, diseases, tissues), schema validation, integrations with workflow managers (Nextflow, Snakemake) and MLOps platforms (W&B, MLflow), and deployment strategies.
license: Apache-2.0 license
metadata:
    skill-author: K-Dense Inc.
---

# LaminDB

## Overview

LaminDB is an open-source data framework for biology designed to make data queryable, traceable, reproducible, and FAIR (Findable, Accessible, Interoperable, Reusable). It provides a unified platform that combines lakehouse architecture, lineage tracking, feature stores, biological ontologies, LIMS (Laboratory Information Management System), and ELN (Electronic Lab Notebook) capabilities through a single Python API.

**Core Value Proposition:**
- **Queryability**: Search and filter datasets by metadata, features, and ontology terms
- **Traceability**: Automatic lineage tracking from raw data through analysis to results
- **Reproducibility**: Version control for data, code, and environment
- **FAIR Compliance**: Standardized annotations using biological ontologies


---

## latchbio-integration

---
name: latchbio-integration
description: Latch platform for bioinformatics workflows. Build pipelines with Latch SDK, @workflow/@task decorators, deploy serverless workflows, LatchFile/LatchDir, Nextflow/Snakemake integration.
license: Unknown
metadata:
    skill-author: K-Dense Inc.
---

# LatchBio Integration

## Overview

Latch is a Python framework for building and deploying bioinformatics workflows as serverless pipelines. Built on Flyte, create workflows with @workflow/@task decorators, manage cloud data with LatchFile/LatchDir, configure resources, and integrate Nextflow/Snakemake pipelines.

## Core Capabilities

The Latch platform provides four main areas of functionality:

### 1. Workflow Creation and Deployment
- Define serverless workflows using Python decorators

---

## latex-posters

---
name: latex-posters
description: "Create professional research posters in LaTeX using beamerposter, tikzposter, or baposter. Support for conference presentations, academic posters, and scientific communication. Includes layout design, color schemes, multi-column formats, figure integration, and poster-specific best practices for visual communication."
allowed-tools: Read Write Edit Bash
---

# LaTeX Research Posters

## Overview

Research posters are a critical medium for scientific communication at conferences, symposia, and academic events. This skill provides comprehensive guidance for creating professional, visually appealing research posters using LaTeX packages. Generate publication-quality posters with proper layout, typography, color schemes, and visual hierarchy.

## When to Use This Skill

This skill should be used when:
- Creating research posters for conferences, symposia, or poster sessions
- Designing academic posters for university events or thesis defenses
- Preparing visual summaries of research for public engagement
- Converting scientific papers into poster format
- Creating template posters for research groups or departments

---

## literature-review

---
name: literature-review
description: Conduct comprehensive, systematic literature reviews using multiple academic databases (PubMed, arXiv, bioRxiv, Semantic Scholar, etc.). This skill should be used when conducting systematic literature reviews, meta-analyses, research synthesis, or comprehensive literature searches across biomedical, scientific, and technical domains. Creates professionally formatted markdown documents and PDFs with verified citations in multiple citation styles (APA, Nature, Vancouver, etc.).
allowed-tools: Read Write Edit Bash
license: MIT license
metadata:
    skill-author: K-Dense Inc.
---

# Literature Review

## Overview

Conduct systematic, comprehensive literature reviews following rigorous academic methodology. Search multiple literature databases, synthesize findings thematically, verify all citations for accuracy, and generate professional output documents in markdown and PDF formats.

This skill integrates with multiple scientific skills for database access (gget, bioservices, datacommons-client) and provides specialized tools for citation verification, result aggregation, and document generation.

## When to Use This Skill

Use this skill when:

---

## markdown-mermaid-writing

---
name: markdown-mermaid-writing
description: Comprehensive markdown and Mermaid diagram writing skill. Use when creating any scientific document, report, analysis, or visualization. Establishes text-based diagrams as the default documentation standard with full style guides (markdown + mermaid), 24 diagram type references, and 9 document templates.
allowed-tools: Read Write Edit Bash
license: Apache-2.0
metadata:
  skill-author: Clayton Young / Superior Byte Works, LLC (@borealBytes)
  skill-source: https://github.com/SuperiorByteWorks-LLC/agent-project
  skill-version: "1.0.0"
  skill-contributors:
    - name: Clayton Young
      org: Superior Byte Works, LLC / @borealBytes
      role: Author and originator
    - name: K-Dense Team
      org: K-Dense Inc.
      role: Integration target and community feedback
---

# Markdown and Mermaid Writing


---

## market-research-reports

---
name: market-research-reports
description: Generate comprehensive market research reports (50+ pages) in the style of top consulting firms (McKinsey, BCG, Gartner). Features professional LaTeX formatting, extensive visual generation with scientific-schematics and generate-image, deep integration with research-lookup for data gathering, and multi-framework strategic analysis including Porter Five Forces, PESTLE, SWOT, TAM/SAM/SOM, and BCG Matrix.
allowed-tools: Read Write Edit Bash
license: MIT license
metadata:
    skill-author: K-Dense Inc.
---

# Market Research Reports

## Overview

Market research reports are comprehensive strategic documents that analyze industries, markets, and competitive landscapes to inform business decisions, investment strategies, and strategic planning. This skill generates **professional-grade reports of 50+ pages** with extensive visual content, modeled after deliverables from top consulting firms like McKinsey, BCG, Bain, Gartner, and Forrester.

**Key Features:**
- **Comprehensive length**: Reports are designed to be 50+ pages with no token constraints
- **Visual-rich content**: 5-6 key diagrams generated at start (more added as needed during writing)
- **Data-driven analysis**: Deep integration with research-lookup for market data
- **Multi-framework approach**: Porter's Five Forces, PESTLE, SWOT, BCG Matrix, TAM/SAM/SOM

---

## markitdown

---
name: markitdown
description: Convert files and office documents to Markdown. Supports PDF, DOCX, PPTX, XLSX, images (with OCR), audio (with transcription), HTML, CSV, JSON, XML, ZIP, YouTube URLs, EPubs and more.
allowed-tools: Read Write Edit Bash
license: MIT license
metadata:
    skill-author: K-Dense Inc.
---

# MarkItDown - File to Markdown Conversion

## Overview

MarkItDown is a Python tool developed by Microsoft for converting various file formats to Markdown. It's particularly useful for converting documents into LLM-friendly text format, as Markdown is token-efficient and well-understood by modern language models.

**Key Benefits**:
- Convert documents to clean, structured Markdown
- Token-efficient format for LLM processing
- Supports 15+ file formats
- Optional AI-enhanced image descriptions

---

## matchms

---
name: matchms
description: Spectral similarity and compound identification for metabolomics. Use for comparing mass spectra, computing similarity scores (cosine, modified cosine), and identifying unknown compounds from spectral libraries. Best for metabolite identification, spectral matching, library searching. For full LC-MS/MS proteomics pipelines use pyopenms.
license: Apache-2.0 license
metadata:
    skill-author: K-Dense Inc.
---

# Matchms

## Overview

Matchms is an open-source Python library for mass spectrometry data processing and analysis. Import spectra from various formats, standardize metadata, filter peaks, calculate spectral similarities, and build reproducible analytical workflows.

## Core Capabilities

### 1. Importing and Exporting Mass Spectrometry Data

Load spectra from multiple file formats and export processed data:


---

## matlab

---
name: matlab
description: MATLAB and GNU Octave numerical computing for matrix operations, data analysis, visualization, and scientific computing. Use when writing MATLAB/Octave scripts for linear algebra, signal processing, image processing, differential equations, optimization, statistics, or creating scientific visualizations. Also use when the user needs help with MATLAB syntax, functions, or wants to convert between MATLAB and Python code. Scripts can be executed with MATLAB or the open-source GNU Octave interpreter.
license: For MATLAB (https://www.mathworks.com/pricing-licensing.html) and for Octave (GNU General Public License version 3)
compatibility: Requires either MATLAB or Octave to be installed for testing, but not required for just generating scripts.
metadata:
    skill-author: K-Dense Inc.
---

# MATLAB/Octave Scientific Computing

MATLAB is a numerical computing environment optimized for matrix operations and scientific computing. GNU Octave is a free, open-source alternative with high MATLAB compatibility.

## Quick Start

**Running MATLAB scripts:**
```bash
# MATLAB (commercial)
matlab -nodisplay -nosplash -r "run('script.m'); exit;"


---

## matplotlib

---
name: matplotlib
description: Low-level plotting library for full customization. Use when you need fine-grained control over every plot element, creating novel plot types, or integrating with specific scientific workflows. Export to PNG/PDF/SVG for publication. For quick statistical plots use seaborn; for interactive plots use plotly; for publication-ready multi-panel figures with journal styling, use scientific-visualization.
license: https://github.com/matplotlib/matplotlib/tree/main/LICENSE
metadata:
    skill-author: K-Dense Inc.
---

# Matplotlib

## Overview

Matplotlib is Python's foundational visualization library for creating static, animated, and interactive plots. This skill provides guidance on using matplotlib effectively, covering both the pyplot interface (MATLAB-style) and the object-oriented API (Figure/Axes), along with best practices for creating publication-quality visualizations.

## When to Use This Skill

This skill should be used when:
- Creating any type of plot or chart (line, scatter, bar, histogram, heatmap, contour, etc.)
- Generating scientific or statistical visualizations
- Customizing plot appearance (colors, styles, labels, legends)

---

## medchem

---
name: medchem
description: Medicinal chemistry filters. Apply drug-likeness rules (Lipinski, Veber), PAINS filters, structural alerts, complexity metrics, for compound prioritization and library filtering.
license: Apache-2.0 license
metadata:
    skill-author: K-Dense Inc.
---

# Medchem

## Overview

Medchem is a Python library for molecular filtering and prioritization in drug discovery workflows. Apply hundreds of well-established and novel molecular filters, structural alerts, and medicinal chemistry rules to efficiently triage and prioritize compound libraries at scale. Rules and filters are context-specific—use as guidelines combined with domain expertise.

## When to Use This Skill

This skill should be used when:
- Applying drug-likeness rules (Lipinski, Veber, etc.) to compound libraries
- Filtering molecules by structural alerts or PAINS patterns
- Prioritizing compounds for lead optimization

---

## modal

---
name: modal
description: Cloud computing platform for running Python on GPUs and serverless infrastructure. Use when deploying AI/ML models, running GPU-accelerated workloads, serving web endpoints, scheduling batch jobs, or scaling Python code to the cloud. Use this skill whenever the user mentions Modal, serverless GPU compute, deploying ML models to the cloud, serving inference endpoints, running batch processing in the cloud, or needs to scale Python workloads beyond their local machine. Also use when the user wants to run code on H100s, A100s, or other cloud GPUs, or needs to create a web API for a model.
license: Apache-2.0
metadata:
  skill-author: K-Dense Inc.
---

# Modal

## Overview

Modal is a cloud platform for running Python code serverlessly, with a focus on AI/ML workloads. Key capabilities:
- **GPU compute** on demand (T4, L4, A10, L40S, A100, H100, H200, B200)
- **Serverless functions** with autoscaling from zero to thousands of containers
- **Custom container images** built entirely in Python code
- **Persistent storage** via Volumes for model weights and datasets
- **Web endpoints** for serving models and APIs
- **Scheduled jobs** via cron or fixed intervals
- **Sub-second cold starts** for low-latency inference

---

## molecular-dynamics

---
name: molecular-dynamics
description: Run and analyze molecular dynamics simulations with OpenMM and MDAnalysis. Set up protein/small molecule systems, define force fields, run energy minimization and production MD, analyze trajectories (RMSD, RMSF, contact maps, free energy surfaces). For structural biology, drug binding, and biophysics.
license: MIT
metadata:
    skill-author: Kuan-lin Huang
---

# Molecular Dynamics

## Overview

Molecular dynamics (MD) simulation computationally models the time evolution of molecular systems by integrating Newton's equations of motion. This skill covers two complementary tools:

- **OpenMM** (https://openmm.org/): High-performance MD simulation engine with GPU support, Python API, and flexible force field support
- **MDAnalysis** (https://mdanalysis.org/): Python library for reading, writing, and analyzing MD trajectories from all major simulation packages

**Installation:**
```bash
conda install -c conda-forge openmm mdanalysis nglview

---

## molfeat

---
name: molfeat
description: Molecular featurization for ML (100+ featurizers). ECFP, MACCS, descriptors, pretrained models (ChemBERTa), convert SMILES to features, for QSAR and molecular ML.
license: Apache-2.0 license
metadata:
    skill-author: K-Dense Inc.
---

# Molfeat - Molecular Featurization Hub

## Overview

Molfeat is a comprehensive Python library for molecular featurization that unifies 100+ pre-trained embeddings and hand-crafted featurizers. Convert chemical structures (SMILES strings or RDKit molecules) into numerical representations for machine learning tasks including QSAR modeling, virtual screening, similarity searching, and deep learning applications. Features fast parallel processing, scikit-learn compatible transformers, and built-in caching.

## When to Use This Skill

This skill should be used when working with:
- **Molecular machine learning**: Building QSAR/QSPR models, property prediction
- **Virtual screening**: Ranking compound libraries for biological activity
- **Similarity searching**: Finding structurally similar molecules

---

## networkx

---
name: networkx
description: Comprehensive toolkit for creating, analyzing, and visualizing complex networks and graphs in Python. Use when working with network/graph data structures, analyzing relationships between entities, computing graph algorithms (shortest paths, centrality, clustering), detecting communities, generating synthetic networks, or visualizing network topologies. Applicable to social networks, biological networks, transportation systems, citation networks, and any domain involving pairwise relationships.
license: 3-clause BSD license
metadata:
    skill-author: K-Dense Inc.
---

# NetworkX

## Overview

NetworkX is a Python package for creating, manipulating, and analyzing complex networks and graphs. Use this skill when working with network or graph data structures, including social networks, biological networks, transportation systems, citation networks, knowledge graphs, or any system involving relationships between entities.

## When to Use This Skill

Invoke this skill when tasks involve:

- **Creating graphs**: Building network structures from data, adding nodes and edges with attributes
- **Graph analysis**: Computing centrality measures, finding shortest paths, detecting communities, measuring clustering

---

## neurokit2

---
name: neurokit2
description: Comprehensive biosignal processing toolkit for analyzing physiological data including ECG, EEG, EDA, RSP, PPG, EMG, and EOG signals. Use this skill when processing cardiovascular signals, brain activity, electrodermal responses, respiratory patterns, muscle activity, or eye movements. Applicable for heart rate variability analysis, event-related potentials, complexity measures, autonomic nervous system assessment, psychophysiology research, and multi-modal physiological signal integration.
license: MIT license
metadata:
    skill-author: K-Dense Inc.
---

# NeuroKit2

## Overview

NeuroKit2 is a comprehensive Python toolkit for processing and analyzing physiological signals (biosignals). Use this skill to process cardiovascular, neural, autonomic, respiratory, and muscular signals for psychophysiology research, clinical applications, and human-computer interaction studies.

## When to Use This Skill

Apply this skill when working with:
- **Cardiac signals**: ECG, PPG, heart rate variability (HRV), pulse analysis
- **Brain signals**: EEG frequency bands, microstates, complexity, source localization
- **Autonomic signals**: Electrodermal activity (EDA/GSR), skin conductance responses (SCR)

---

## neuropixels-analysis

---
name: neuropixels-analysis
description: Neuropixels neural recording analysis. Load SpikeGLX/OpenEphys data, preprocess, motion correction, Kilosort4 spike sorting, quality metrics, Allen/IBL curation, AI-assisted visual analysis, for Neuropixels 1.0/2.0 extracellular electrophysiology. Use when working with neural recordings, spike sorting, extracellular electrophysiology, or when the user mentions Neuropixels, SpikeGLX, Open Ephys, Kilosort, quality metrics, or unit curation.
license: MIT license
metadata:
    skill-author: K-Dense Inc.
---

# Neuropixels Data Analysis

## Overview

Comprehensive toolkit for analyzing Neuropixels high-density neural recordings using current best practices from SpikeInterface, Allen Institute, and International Brain Laboratory (IBL). Supports the full workflow from raw data to publication-ready curated units.

## When to Use This Skill

This skill should be used when:
- Working with Neuropixels recordings (.ap.bin, .lf.bin, .meta files)
- Loading data from SpikeGLX, Open Ephys, or NWB formats
- Preprocessing neural recordings (filtering, CAR, bad channel detection)

---

## omero-integration

---
name: omero-integration
description: Microscopy data management platform. Access images via Python, retrieve datasets, analyze pixels, manage ROIs/annotations, batch processing, for high-content screening and microscopy workflows.
license: Unknown
metadata:
    skill-author: K-Dense Inc.
---

# OMERO Integration

## Overview

OMERO is an open-source platform for managing, visualizing, and analyzing microscopy images and metadata. Access images via Python API, retrieve datasets, analyze pixels, manage ROIs and annotations, for high-content screening and microscopy workflows.

## When to Use This Skill

This skill should be used when:
- Working with OMERO Python API (omero-py) to access microscopy data
- Retrieving images, datasets, projects, or screening data programmatically
- Analyzing pixel data and creating derived images

---

## open-notebook

---
name: open-notebook
description: Self-hosted, open-source alternative to Google NotebookLM for AI-powered research and document analysis. Use when organizing research materials into notebooks, ingesting diverse content sources (PDFs, videos, audio, web pages, Office documents), generating AI-powered notes and summaries, creating multi-speaker podcasts from research, chatting with documents using context-aware AI, searching across materials with full-text and vector search, or running custom content transformations. Supports 16+ AI providers including OpenAI, Anthropic, Google, Ollama, Groq, and Mistral with complete data privacy through self-hosting.
license: MIT
metadata:
    skill-author: K-Dense Inc.
---

# Open Notebook

## Overview

Open Notebook is an open-source, self-hosted alternative to Google's NotebookLM that enables researchers to organize materials, generate AI-powered insights, create podcasts, and have context-aware conversations with their documents — all while maintaining complete data privacy.

Unlike Google's Notebook LM, which has no publicly available API outside of the Enterprise version, Open Notebook provides a comprehensive REST API, supports 16+ AI providers, and runs entirely on your own infrastructure.

**Key advantages over NotebookLM:**
- Full REST API for programmatic access and automation
- Choice of 16+ AI providers (not locked to Google models)
- Multi-speaker podcast generation with 1-4 customizable speakers (vs. 2-speaker limit)

---

## opentrons-integration

---
name: opentrons-integration
description: Official Opentrons Protocol API for OT-2 and Flex robots. Use when writing protocols specifically for Opentrons hardware with full access to Protocol API v2 features. Best for production Opentrons protocols, official API compatibility. For multi-vendor automation or broader equipment control use pylabrobot.
license: Unknown
metadata:
    skill-author: K-Dense Inc.
---

# Opentrons Integration

## Overview

Opentrons is a Python-based lab automation platform for Flex and OT-2 robots. Write Protocol API v2 protocols for liquid handling, control hardware modules (heater-shaker, thermocycler), manage labware, for automated pipetting workflows.

## When to Use This Skill

This skill should be used when:
- Writing Opentrons Protocol API v2 protocols in Python
- Automating liquid handling workflows on Flex or OT-2 robots
- Controlling hardware modules (temperature, magnetic, heater-shaker, thermocycler)

---

## optimize-for-gpu

---
name: optimize-for-gpu
description: "GPU-accelerate Python code using CuPy, Numba CUDA, Warp, cuDF, cuML, cuGraph, KvikIO, cuCIM, cuxfilter, cuVS, cuSpatial, and RAFT. Use whenever the user mentions GPU/CUDA/NVIDIA acceleration, or wants to speed up NumPy, pandas, scikit-learn, scikit-image, NetworkX, GeoPandas, or Faiss workloads. Covers physics simulation, differentiable rendering, mesh ray casting, particle systems (DEM/SPH/fluids), vector/similarity search, GPUDirect Storage file IO, interactive dashboards, geospatial analysis, medical imaging, and sparse eigensolvers. Also use when you see CPU-bound Python code (loops, large arrays, ML pipelines, graph analytics, image processing) that would benefit from GPU acceleration, even if not explicitly requested."
metadata:
  author: K-Dense, Inc.
---

# GPU Optimization for Python with NVIDIA

You are an expert GPU optimization engineer. Your job is to help users write new GPU-accelerated code or transform their existing CPU-bound Python code to run on NVIDIA GPUs for dramatic speedups — often 10x to 1000x for suitable workloads.

## When This Skill Applies

- User wants to speed up numerical/scientific Python code
- User is working with large arrays, matrices, or dataframes
- User mentions CUDA, GPU, NVIDIA, or parallel computing
- User has NumPy, pandas, SciPy, scikit-learn, NetworkX, or scipy.sparse.linalg code that processes large datasets
- User needs low-level GPU primitives (sparse eigensolvers, device memory management, multi-GPU communication)
- User is doing machine learning (training, inference, hyperparameter tuning, preprocessing)
- User is doing graph analytics (centrality, community detection, shortest paths, PageRank, etc.)

---

## paper-lookup

---
name: paper-lookup
description: Search 10 academic paper databases via their REST APIs to find research papers, preprints, and scholarly articles. Covers biomedical literature (PubMed, PMC full text), preprint servers (bioRxiv, medRxiv, arXiv), multidisciplinary indexes (OpenAlex, Crossref, Semantic Scholar), open access aggregators (CORE, Unpaywall). Use this skill whenever the user wants to search for research papers, find citations, look up articles by DOI or PMID, retrieve abstracts or full text, check open access availability, find preprints, explore citation graphs, search by author or keyword, or access any scholarly literature database. Also trigger when the user mentions PubMed, PMC, bioRxiv, medRxiv, arXiv, OpenAlex, Crossref, Semantic Scholar, CORE, Unpaywall, or asks about paper metadata, citation counts, journal articles, manuscript lookups, literature reviews, or systematic searches. Even if the user just says "find papers on X" or "what's been published about Y" or "look up this DOI", this skill should activate.
metadata:
  skill-author: K-Dense Inc.
---

# Paper Lookup

You have access to 10 academic paper databases through their REST APIs. Your job is to figure out which database(s) best serve the user's query, call them, and return the results.

## Core Workflow

1. **Understand the query** -- What is the user looking for? A specific paper by DOI? Papers on a topic? An author's publications? Open access PDFs? Full text? This determines which database(s) to hit.

2. **Select database(s)** -- Use the database selection guide below. Many queries benefit from hitting multiple databases -- for example, searching PubMed for papers and then checking Unpaywall for open access copies.

3. **Read the reference file** -- Each database has a reference file in `references/` with endpoint details, query formats, and example calls. Read the relevant file(s) before making API calls.

4. **Make the API call(s)** -- See the **Making API Calls** section below for which HTTP fetch tool to use on your platform.

---

## parallel-web

---
name: parallel-web
description: Search the web, extract URL content, and run deep research using the Parallel Chat API and Extract API. Use for ALL web searches, research queries, and general information gathering. Provides synthesized summaries with citations.
allowed-tools: Read Write Edit Bash
license: MIT license
compatibility: PARALLEL_API_KEY required
metadata:
    skill-author: K-Dense Inc.
---

# Parallel Web Systems API

## Overview

This skill provides access to **Parallel Web Systems** APIs for web search, deep research, and content extraction. It is the **primary tool for all web-related operations** in the scientific writer workflow.

**Primary interface:** Parallel Chat API (OpenAI-compatible) for search and research.
**Secondary interface:** Extract API for URL verification and special cases only.

**API Documentation:** https://docs.parallel.ai

---

## pathml

---
name: pathml
description: Full-featured computational pathology toolkit. Use for advanced WSI analysis including multiplexed immunofluorescence (CODEX, Vectra), nucleus segmentation, tissue graph construction, and ML model training on pathology data. Supports 160+ slide formats. For simple tile extraction from H&E slides, histolab may be simpler.
license: GPL-2.0 license
metadata:
    skill-author: K-Dense Inc.
---

# PathML

## Overview

PathML is a comprehensive Python toolkit for computational pathology workflows, designed to facilitate machine learning and image analysis for whole-slide pathology images. The framework provides modular, composable tools for loading diverse slide formats, preprocessing images, constructing spatial graphs, training deep learning models, and analyzing multiparametric imaging data from technologies like CODEX and multiplex immunofluorescence.

## When to Use This Skill

Apply this skill for:
- Loading and processing whole-slide images (WSI) in various proprietary formats
- Preprocessing H&E stained tissue images with stain normalization
- Nucleus detection, segmentation, and classification workflows

---

## pdf

---
name: pdf
description: Use this skill whenever the user wants to do anything with PDF files. This includes reading or extracting text/tables from PDFs, combining or merging multiple PDFs into one, splitting PDFs apart, rotating pages, adding watermarks, creating new PDFs, filling PDF forms, encrypting/decrypting PDFs, extracting images, and OCR on scanned PDFs to make them searchable. If the user mentions a .pdf file or asks to produce one, use this skill.
license: Proprietary. LICENSE.txt has complete terms
---

# PDF Processing Guide

## Overview

This guide covers essential PDF processing operations using Python libraries and command-line tools. For advanced features, JavaScript libraries, and detailed examples, see REFERENCE.md. If you need to fill out a PDF form, read FORMS.md and follow its instructions.

## Quick Start

```python
from pypdf import PdfReader, PdfWriter

# Read a PDF
reader = PdfReader("document.pdf")
print(f"Pages: {len(reader.pages)}")

---

## peer-review

---
name: peer-review
description: Structured manuscript/grant review with checklist-based evaluation. Use when writing formal peer reviews with specific criteria methodology assessment, statistical validity, reporting standards compliance (CONSORT/STROBE), and constructive feedback. Best for actual review writing, manuscript revision. For evaluating claims/evidence quality use scientific-critical-thinking; for quantitative scoring frameworks use scholar-evaluation.
allowed-tools: Read Write Edit Bash
license: MIT license
metadata:
    skill-author: K-Dense Inc.
---

# Scientific Critical Evaluation and Peer Review

## Overview

Peer review is a systematic process for evaluating scientific manuscripts. Assess methodology, statistics, design, reproducibility, ethics, and reporting standards. Apply this skill for manuscript and grant review across disciplines with constructive, rigorous evaluation.

## When to Use This Skill

This skill should be used when:
- Conducting peer review of scientific manuscripts for journals
- Evaluating grant proposals and research applications

---

## pennylane

---
name: pennylane
description: Hardware-agnostic quantum ML framework with automatic differentiation. Use when training quantum circuits via gradients, building hybrid quantum-classical models, or needing device portability across IBM/Google/Rigetti/IonQ. Best for variational algorithms (VQE, QAOA), quantum neural networks, and integration with PyTorch/JAX/TensorFlow. For hardware-specific optimizations use qiskit (IBM) or cirq (Google); for open quantum systems use qutip.
license: Apache-2.0 license
metadata:
    skill-author: K-Dense Inc.
---

# PennyLane

## Overview

PennyLane is a quantum computing library that enables training quantum computers like neural networks. It provides automatic differentiation of quantum circuits, device-independent programming, and seamless integration with classical machine learning frameworks.

## Installation

Install using uv:

```bash
uv pip install pennylane

---

## perplexity-search

---
name: perplexity-search
description: Perform AI-powered web searches with real-time information using Perplexity models via LiteLLM and OpenRouter. This skill should be used when conducting web searches for current information, finding recent scientific literature, getting grounded answers with source citations, or accessing information beyond the model knowledge cutoff. Provides access to multiple Perplexity models including Sonar Pro, Sonar Pro Search (advanced agentic search), and Sonar Reasoning Pro through a single OpenRouter API key.
license: MIT license
compatibility: An OpenRouter API key is required to use Perplexity search
metadata:
    skill-author: K-Dense Inc.
---

# Perplexity Search

## Overview

Perform AI-powered web searches using Perplexity models through LiteLLM and OpenRouter. Perplexity provides real-time, web-grounded answers with source citations, making it ideal for finding current information, recent scientific literature, and facts beyond the model's training data cutoff.

This skill provides access to all Perplexity models through OpenRouter, requiring only a single API key (no separate Perplexity account needed).

## When to Use This Skill

Use this skill when:

---

## phylogenetics

---
name: phylogenetics
description: Build and analyze phylogenetic trees using MAFFT (multiple alignment), IQ-TREE 2 (maximum likelihood), and FastTree (fast NJ/ML). Visualize with ETE3 or FigTree. For evolutionary analysis, microbial genomics, viral phylodynamics, protein family analysis, and molecular clock studies.
license: Unknown
metadata:
    skill-author: Kuan-lin Huang
---

# Phylogenetics

## Overview

Phylogenetic analysis reconstructs the evolutionary history of biological sequences (genes, proteins, genomes) by inferring the branching pattern of descent. This skill covers the standard pipeline:

1. **MAFFT** — Multiple sequence alignment
2. **IQ-TREE 2** — Maximum likelihood tree inference with model selection
3. **FastTree** — Fast approximate maximum likelihood (for large datasets)
4. **ETE3** — Python library for tree manipulation and visualization

**Installation:**

---

## polars-bio

---
name: polars-bio
description: High-performance genomic interval operations and bioinformatics file I/O on Polars DataFrames. Overlap, nearest, merge, coverage, complement, subtract for BED/VCF/BAM/GFF intervals. Streaming, cloud-native, faster bioframe alternative.
license: https://github.com/biodatageeks/polars-bio/blob/main/LICENSE
metadata:
    skill-author: K-Dense Inc.
---

# polars-bio

## Overview

polars-bio is a high-performance Python library for genomic interval operations and bioinformatics file I/O, built on Polars, Apache Arrow, and Apache DataFusion. It provides a familiar DataFrame-centric API for interval arithmetic (overlap, nearest, merge, coverage, complement, subtract) and reading/writing common bioinformatics formats (BED, VCF, BAM, CRAM, GFF/GTF, FASTA, FASTQ).

Key value propositions:
- **6-38x faster** than bioframe on real-world genomic benchmarks
- **Streaming/out-of-core** support for large genomes via DataFusion
- **Cloud-native** file I/O (S3, GCS, Azure) with predicate pushdown
- **Two API styles**: functional (`pb.overlap(df1, df2)`) and method-chaining (`df1.lazy().pb.overlap(df2)`)
- **SQL interface** for genomic data via DataFusion SQL engine

---

## polars

---
name: polars
description: Fast in-memory DataFrame library for datasets that fit in RAM. Use when pandas is too slow but data still fits in memory. Lazy evaluation, parallel execution, Apache Arrow backend. Best for 1-100GB datasets, ETL pipelines, faster pandas replacement. For larger-than-RAM data use dask or vaex.
license: https://github.com/pola-rs/polars/blob/main/LICENSE
metadata:
    skill-author: K-Dense Inc.
---

# Polars

## Overview

Polars is a lightning-fast DataFrame library for Python and Rust built on Apache Arrow. Work with Polars' expression-based API, lazy evaluation framework, and high-performance data manipulation capabilities for efficient data processing, pandas migration, and data pipeline optimization.

## Quick Start

### Installation and Basic Usage

Install Polars:
```python

---

## pptx-posters

---
name: pptx-posters
description: Create research posters using HTML/CSS that can be exported to PDF or PPTX. Use this skill ONLY when the user explicitly requests PowerPoint/PPTX poster format. For standard research posters, use latex-posters instead. This skill provides modern web-based poster design with responsive layouts and easy visual integration.
allowed-tools: Read Write Edit Bash
license: MIT license
metadata:
    skill-author: K-Dense Inc.
---

# PPTX Research Posters (HTML-Based)

## Overview

**⚠️ USE THIS SKILL ONLY WHEN USER EXPLICITLY REQUESTS PPTX/POWERPOINT POSTER FORMAT.**

For standard research posters, use the **latex-posters** skill instead, which provides better typographic control and is the default for academic conferences.

This skill creates research posters using HTML/CSS, which can then be exported to PDF or converted to PowerPoint format. The web-based approach offers:
- Modern, responsive layouts
- Easy integration of AI-generated visuals

---

## pptx

---
name: pptx
description: "Use this skill any time a .pptx file is involved in any way — as input, output, or both. This includes: creating slide decks, pitch decks, or presentations; reading, parsing, or extracting text from any .pptx file (even if the extracted content will be used elsewhere, like in an email or summary); editing, modifying, or updating existing presentations; combining or splitting slide files; working with templates, layouts, speaker notes, or comments. Trigger whenever the user mentions \"deck,\" \"slides,\" \"presentation,\" or references a .pptx filename, regardless of what they plan to do with the content afterward. If a .pptx file needs to be opened, created, or touched, use this skill."
license: Proprietary. LICENSE.txt has complete terms
---

# PPTX Skill

## Quick Reference

| Task | Guide |
|------|-------|
| Read/analyze content | `python -m markitdown presentation.pptx` |
| Edit or create from template | Read [editing.md](editing.md) |
| Create from scratch | Read [pptxgenjs.md](pptxgenjs.md) |

---

## Reading Content


---

## primekg

---
name: primekg
description: Query the Precision Medicine Knowledge Graph (PrimeKG) for multiscale biological data including genes, drugs, diseases, phenotypes, and more.
license: Unknown
metadata:
    skill-author: K-Dense Inc. (PrimeKG original from Harvard MIMS)
---

# PrimeKG Knowledge Graph Skill

## Overview

PrimeKG is a precision medicine knowledge graph that integrates over 20 primary databases and high-quality scientific literature into a single resource. It contains over 100,000 nodes and 4 million edges across 29 relationship types, including drug-target, disease-gene, and phenotype-disease associations.

**Key capabilities:**
- Search for nodes (genes, proteins, drugs, diseases, phenotypes)
- Retrieve direct neighbors (associated entities and clinical evidence)
- Analyze local disease context (related genes, drugs, phenotypes)
- Identify drug-disease paths (potential repurposing opportunities)


---

## protocolsio-integration

---
name: protocolsio-integration
description: Integration with protocols.io API for managing scientific protocols. This skill should be used when working with protocols.io to search, create, update, or publish protocols; manage protocol steps and materials; handle discussions and comments; organize workspaces; upload and manage files; or integrate protocols.io functionality into workflows. Applicable for protocol discovery, collaborative protocol development, experiment tracking, lab protocol management, and scientific documentation.
license: Unknown
metadata:
    skill-author: K-Dense Inc.
---

# Protocols.io Integration

## Overview

Protocols.io is a comprehensive platform for developing, sharing, and managing scientific protocols. This skill provides complete integration with the protocols.io API v3, enabling programmatic access to protocols, workspaces, discussions, file management, and collaboration features.

## When to Use This Skill

Use this skill when working with protocols.io in any of the following scenarios:

- **Protocol Discovery**: Searching for existing protocols by keywords, DOI, or category
- **Protocol Management**: Creating, updating, or publishing scientific protocols

---

## pufferlib

---
name: pufferlib
description: High-performance reinforcement learning framework optimized for speed and scale. Use when you need fast parallel training, vectorized environments, multi-agent systems, or integration with game environments (Atari, Procgen, NetHack). Achieves 2-10x speedups over standard implementations. For quick prototyping or standard algorithm implementations with extensive documentation, use stable-baselines3 instead.
license: MIT license
metadata:
    skill-author: K-Dense Inc.
---

# PufferLib - High-Performance Reinforcement Learning

## Overview

PufferLib is a high-performance reinforcement learning library designed for fast parallel environment simulation and training. It achieves training at millions of steps per second through optimized vectorization, native multi-agent support, and efficient PPO implementation (PuffeRL). The library provides the Ocean suite of 20+ environments and seamless integration with Gymnasium, PettingZoo, and specialized RL frameworks.

## When to Use This Skill

Use this skill when:
- **Training RL agents** with PPO on any environment (single or multi-agent)
- **Creating custom environments** using the PufferEnv API
- **Optimizing performance** for parallel environment simulation (vectorization)

---

## pydeseq2

---
name: pydeseq2
description: Differential gene expression analysis (Python DESeq2). Identify DE genes from bulk RNA-seq counts, Wald tests, FDR correction, volcano/MA plots, for RNA-seq analysis.
license: MIT license
metadata:
    skill-author: K-Dense Inc.
---

# PyDESeq2

## Overview

PyDESeq2 is a Python implementation of DESeq2 for differential expression analysis with bulk RNA-seq data. Design and execute complete workflows from data loading through result interpretation, including single-factor and multi-factor designs, Wald tests with multiple testing correction, optional apeGLM shrinkage, and integration with pandas and AnnData.

## When to Use This Skill

This skill should be used when:
- Analyzing bulk RNA-seq count data for differential expression
- Comparing gene expression between experimental conditions (e.g., treated vs control)
- Performing multi-factor designs accounting for batch effects or covariates

---

## pydicom

---
name: pydicom
description: Python library for working with DICOM (Digital Imaging and Communications in Medicine) files. Use this skill when reading, writing, or modifying medical imaging data in DICOM format, extracting pixel data from medical images (CT, MRI, X-ray, ultrasound), anonymizing DICOM files, working with DICOM metadata and tags, converting DICOM images to other formats, handling compressed DICOM data, or processing medical imaging datasets. Applies to tasks involving medical image analysis, PACS systems, radiology workflows, and healthcare imaging applications.
license: https://github.com/pydicom/pydicom/blob/main/LICENSE
metadata:
    skill-author: K-Dense Inc.
---

# Pydicom

## Overview

Pydicom is a pure Python package for working with DICOM files, the standard format for medical imaging data. This skill provides guidance on reading, writing, and manipulating DICOM files, including working with pixel data, metadata, and various compression formats.

## When to Use This Skill

Use this skill when working with:
- Medical imaging files (CT, MRI, X-ray, ultrasound, PET, etc.)
- DICOM datasets requiring metadata extraction or modification
- Pixel data extraction and image processing from medical scans

---

## pyhealth

---
name: pyhealth
description: Comprehensive healthcare AI toolkit for developing, testing, and deploying machine learning models with clinical data. This skill should be used when working with electronic health records (EHR), clinical prediction tasks (mortality, readmission, drug recommendation), medical coding systems (ICD, NDC, ATC), physiological signals (EEG, ECG), healthcare datasets (MIMIC-III/IV, eICU, OMOP), or implementing deep learning models for healthcare applications (RETAIN, SafeDrug, Transformer, GNN).
license: MIT license
metadata:
    skill-author: K-Dense Inc.
---

# PyHealth: Healthcare AI Toolkit

## Overview

PyHealth is a comprehensive Python library for healthcare AI that provides specialized tools, models, and datasets for clinical machine learning. Use this skill when developing healthcare prediction models, processing clinical data, working with medical coding systems, or deploying AI solutions in healthcare settings.

## When to Use This Skill

Invoke this skill when:

- **Working with healthcare datasets**: MIMIC-III, MIMIC-IV, eICU, OMOP, sleep EEG data, medical images
- **Clinical prediction tasks**: Mortality prediction, hospital readmission, length of stay, drug recommendation

---

## pylabrobot

---
name: pylabrobot
description: Vendor-agnostic lab automation framework. Use when controlling multiple equipment types (Hamilton, Tecan, Opentrons, plate readers, pumps) or needing unified programming across different vendors. Best for complex workflows, multi-vendor setups, simulation. For Opentrons-only protocols with official API, opentrons-integration may be simpler.
license: MIT license
metadata:
    skill-author: K-Dense Inc.
---

# PyLabRobot

## Overview

PyLabRobot is a hardware-agnostic, pure Python Software Development Kit for automated and autonomous laboratories. Use this skill to control liquid handling robots, plate readers, pumps, heater shakers, incubators, centrifuges, and other laboratory automation equipment through a unified Python interface that works across platforms (Windows, macOS, Linux).

## When to Use This Skill

Use this skill when:
- Programming liquid handling robots (Hamilton STAR/STARlet, Opentrons OT-2, Tecan EVO)
- Automating laboratory workflows involving pipetting, sample preparation, or analytical measurements
- Managing deck layouts and laboratory resources (plates, tips, containers, troughs)

---

## pymatgen

---
name: pymatgen
description: Materials science toolkit. Crystal structures (CIF, POSCAR), phase diagrams, band structure, DOS, Materials Project integration, format conversion, for computational materials science.
license: MIT license
metadata:
    skill-author: K-Dense Inc.
---

# Pymatgen - Python Materials Genomics

## Overview

Pymatgen is a comprehensive Python library for materials analysis that powers the Materials Project. Create, analyze, and manipulate crystal structures and molecules, compute phase diagrams and thermodynamic properties, analyze electronic structure (band structures, DOS), generate surfaces and interfaces, and access Materials Project's database of computed materials. Supports 100+ file formats from various computational codes.

## When to Use This Skill

This skill should be used when:
- Working with crystal structures or molecular systems in materials science
- Converting between structure file formats (CIF, POSCAR, XYZ, etc.)
- Analyzing symmetry, space groups, or coordination environments

---

## pymc

---
name: pymc
description: Bayesian modeling with PyMC. Build hierarchical models, MCMC (NUTS), variational inference, LOO/WAIC comparison, posterior checks, for probabilistic programming and inference.
license: Apache License, Version 2.0
metadata:
    skill-author: K-Dense Inc.
---

# PyMC Bayesian Modeling

## Overview

PyMC is a Python library for Bayesian modeling and probabilistic programming. Build, fit, validate, and compare Bayesian models using PyMC's modern API (version 5.x+), including hierarchical models, MCMC sampling (NUTS), variational inference, and model comparison (LOO, WAIC).

## When to Use This Skill

This skill should be used when:
- Building Bayesian models (linear/logistic regression, hierarchical models, time series, etc.)
- Performing MCMC sampling or variational inference
- Conducting prior/posterior predictive checks

---

## pymoo

---
name: pymoo
description: Multi-objective optimization framework. NSGA-II, NSGA-III, MOEA/D, Pareto fronts, constraint handling, benchmarks (ZDT, DTLZ), for engineering design and optimization problems.
license: Apache-2.0 license
metadata:
    skill-author: K-Dense Inc.
---

# Pymoo - Multi-Objective Optimization in Python

## Overview

Pymoo is a comprehensive Python framework for optimization with emphasis on multi-objective problems. Solve single and multi-objective optimization using state-of-the-art algorithms (NSGA-II/III, MOEA/D), benchmark problems (ZDT, DTLZ), customizable genetic operators, and multi-criteria decision making methods. Excels at finding trade-off solutions (Pareto fronts) for problems with conflicting objectives.

## When to Use This Skill

This skill should be used when:
- Solving optimization problems with one or multiple objectives
- Finding Pareto-optimal solutions and analyzing trade-offs
- Implementing evolutionary algorithms (GA, DE, PSO, NSGA-II/III)

---

## pyopenms

---
name: pyopenms
description: Complete mass spectrometry analysis platform. Use for proteomics workflows feature detection, peptide identification, protein quantification, and complex LC-MS/MS pipelines. Supports extensive file formats and algorithms. Best for proteomics, comprehensive MS data processing. For simple spectral comparison and metabolite ID use matchms.
license: 3 clause BSD license
metadata:
    skill-author: K-Dense Inc.
---

# PyOpenMS

## Overview

PyOpenMS provides Python bindings to the OpenMS library for computational mass spectrometry, enabling analysis of proteomics and metabolomics data. Use for handling mass spectrometry file formats, processing spectral data, detecting features, identifying peptides/proteins, and performing quantitative analysis.

## Installation

Install using uv:

```bash
uv uv pip install pyopenms

---

## pysam

---
name: pysam
description: Genomic file toolkit. Read/write SAM/BAM/CRAM alignments, VCF/BCF variants, FASTA/FASTQ sequences, extract regions, calculate coverage, for NGS data processing pipelines.
license: MIT license
metadata:
    skill-author: K-Dense Inc.
---

# Pysam

## Overview

Pysam is a Python module for reading, manipulating, and writing genomic datasets. Read/write SAM/BAM/CRAM alignment files, VCF/BCF variant files, and FASTA/FASTQ sequences with a Pythonic interface to htslib. Query tabix-indexed files, perform pileup analysis for coverage, and execute samtools/bcftools commands.

## When to Use This Skill

This skill should be used when:
- Working with sequencing alignment files (BAM/CRAM)
- Analyzing genetic variants (VCF/BCF)
- Extracting reference sequences or gene regions

---

## pytdc

---
name: pytdc
description: Therapeutics Data Commons. AI-ready drug discovery datasets (ADME, toxicity, DTI), benchmarks, scaffold splits, molecular oracles, for therapeutic ML and pharmacological prediction.
license: MIT license
metadata:
    skill-author: K-Dense Inc.
---

# PyTDC (Therapeutics Data Commons)

## Overview

PyTDC is an open-science platform providing AI-ready datasets and benchmarks for drug discovery and development. Access curated datasets spanning the entire therapeutics pipeline with standardized evaluation metrics and meaningful data splits, organized into three categories: single-instance prediction (molecular/protein properties), multi-instance prediction (drug-target interactions, DDI), and generation (molecule generation, retrosynthesis).

## When to Use This Skill

This skill should be used when:
- Working with drug discovery or therapeutic ML datasets
- Benchmarking machine learning models on standardized pharmaceutical tasks
- Predicting molecular properties (ADME, toxicity, bioactivity)

---

## pytorch-lightning

---
name: pytorch-lightning
description: Deep learning framework (PyTorch Lightning). Organize PyTorch code into LightningModules, configure Trainers for multi-GPU/TPU, implement data pipelines, callbacks, logging (W&B, TensorBoard), distributed training (DDP, FSDP, DeepSpeed), for scalable neural network training.
license: Apache-2.0 license
metadata:
    skill-author: K-Dense Inc.
---

# PyTorch Lightning

## Overview

PyTorch Lightning is a deep learning framework that organizes PyTorch code to eliminate boilerplate while maintaining full flexibility. Automate training workflows, multi-device orchestration, and implement best practices for neural network training and scaling across multiple GPUs/TPUs.

## When to Use This Skill

This skill should be used when:
- Building, training, or deploying neural networks using PyTorch Lightning
- Organizing PyTorch code into LightningModules
- Configuring Trainers for multi-GPU/TPU training

---

## pyzotero

---
name: pyzotero
description: Interact with Zotero reference management libraries using the pyzotero Python client. Retrieve, create, update, and delete items, collections, tags, and attachments via the Zotero Web API v3. Use this skill when working with Zotero libraries programmatically, managing bibliographic references, exporting citations, searching library contents, uploading PDF attachments, or building research automation workflows that integrate with Zotero.
allowed-tools: Read Write Edit Bash
license: MIT License
metadata:
    skill-author: K-Dense Inc.
---

# Pyzotero

Pyzotero is a Python wrapper for the [Zotero API v3](https://www.zotero.org/support/dev/web_api/v3/start). Use it to programmatically manage Zotero libraries: read items and collections, create and update references, upload attachments, manage tags, and export citations.

## Authentication Setup

**Required credentials** — get from https://www.zotero.org/settings/keys:
- **User ID**: shown as "Your userID for use in API calls"
- **API Key**: create at https://www.zotero.org/settings/keys/new
- **Library ID**: for group libraries, the integer after `/groups/` in the group URL


---

## qiskit

---
name: qiskit
description: IBM quantum computing framework. Use when targeting IBM Quantum hardware, working with Qiskit Runtime for production workloads, or needing IBM optimization tools. Best for IBM hardware execution, quantum error mitigation, and enterprise quantum computing. For Google hardware use cirq; for gradient-based quantum ML use pennylane; for open quantum system simulations use qutip.
license: Apache-2.0 license
metadata:
    skill-author: K-Dense Inc.
---

# Qiskit

## Overview

Qiskit is the world's most popular open-source quantum computing framework with 13M+ downloads. Build quantum circuits, optimize for hardware, execute on simulators or real quantum computers, and analyze results. Supports IBM Quantum (100+ qubit systems), IonQ, Amazon Braket, and other providers.

**Key Features:**
- 83x faster transpilation than competitors
- 29% fewer two-qubit gates in optimized circuits
- Backend-agnostic execution (local simulators or cloud hardware)
- Comprehensive algorithm libraries for optimization, chemistry, and ML


---

## qutip

---
name: qutip
description: Quantum physics simulation library for open quantum systems. Use when studying master equations, Lindblad dynamics, decoherence, quantum optics, or cavity QED. Best for physics research, open system dynamics, and educational simulations. NOT for circuit-based quantum computing—use qiskit, cirq, or pennylane for quantum algorithms and hardware execution.
license: BSD-3-Clause license
metadata:
    skill-author: K-Dense Inc.
---

# QuTiP: Quantum Toolbox in Python

## Overview

QuTiP provides comprehensive tools for simulating and analyzing quantum mechanical systems. It handles both closed (unitary) and open (dissipative) quantum systems with multiple solvers optimized for different scenarios.

## Installation

```bash
uv pip install qutip
```


---

## rdkit

---
name: rdkit
description: Cheminformatics toolkit for fine-grained molecular control. SMILES/SDF parsing, descriptors (MW, LogP, TPSA), fingerprints, substructure search, 2D/3D generation, similarity, reactions. For standard workflows with simpler interface, use datamol (wrapper around RDKit). Use rdkit for advanced control, custom sanitization, specialized algorithms.
license: BSD-3-Clause license
metadata:
    skill-author: K-Dense Inc.
---

# RDKit Cheminformatics Toolkit

## Overview

RDKit is a comprehensive cheminformatics library providing Python APIs for molecular analysis and manipulation. This skill provides guidance for reading/writing molecular structures, calculating descriptors, fingerprinting, substructure searching, chemical reactions, 2D/3D coordinate generation, and molecular visualization. Use this skill for drug discovery, computational chemistry, and cheminformatics research tasks.

## Core Capabilities

### 1. Molecular I/O and Creation

**Reading Molecules:**


---

## research-grants

---
name: research-grants
description: Write competitive research proposals for NSF, NIH, DOE, DARPA, and Taiwan NSTC. Agency-specific formatting, review criteria, budget preparation, broader impacts, significance statements, innovation narratives, and compliance with submission requirements.
allowed-tools: Read Write Edit Bash
license: MIT license
metadata:
    skill-author: K-Dense Inc.
---

# Research Grant Writing

## Overview

Research grant writing is the process of developing competitive funding proposals for federal agencies and foundations. Master agency-specific requirements, review criteria, narrative structure, budget preparation, and compliance for NSF (National Science Foundation), NIH (National Institutes of Health), DOE (Department of Energy), DARPA (Defense Advanced Research Projects Agency), and Taiwan's NSTC (National Science and Technology Council) submissions.

**Critical Principle: Grants are persuasive documents that must simultaneously demonstrate scientific rigor, innovation, feasibility, and broader impact.** Each agency has distinct priorities, review criteria, formatting requirements, and strategic goals that must be addressed.

## When to Use This Skill

This skill should be used when:

---

## research-lookup

---
name: research-lookup
description: Look up current research information using the Parallel Chat API (primary) or Perplexity sonar-pro-search (academic paper searches). Automatically routes queries to the best backend. Use for finding papers, gathering research data, and verifying scientific information.
allowed-tools: Read Write Edit Bash
license: MIT license
compatibility: PARALLEL_API_KEY and OPENROUTER_API_KEY required
metadata:
    skill-author: K-Dense Inc.
---

# Research Information Lookup

## Overview

This skill provides real-time research information lookup with **intelligent backend routing**:

- **Parallel Chat API** (`core` model): Default backend for all general research queries. Provides comprehensive, multi-source research reports with inline citations via the OpenAI-compatible Chat API at `https://api.parallel.ai`.
- **Perplexity sonar-pro-search** (via OpenRouter): Used only for academic-specific paper searches where scholarly database access is critical.

The skill automatically detects query type and routes to the optimal backend.

---

## rowan

---
name: rowan
description: Rowan is a cloud-native molecular modeling and medicinal-chemistry workflow platform with a Python API. Use for pKa and macropKa prediction, conformer and tautomer ensembles, docking and analogue docking, protein-ligand cofolding, MSA generation, molecular dynamics, permeability, descriptor workflows, and related small-molecule or protein modeling tasks. Ideal for programmatic batch screening, multi-step chemistry pipelines, and workflows that would otherwise require maintaining local HPC/GPU infrastructure.
license: Proprietary (API key required)
compatibility: Python 3.12+, API key required
metadata:
  skill-author: Rowan Science
  trigger-keywords: ["pKa prediction", "molecular docking", "conformer search", "chemistry workflow", "drug discovery", "SMILES", "protein structure", "batch molecular modeling", "cloud chemistry"]
---

# Rowan: Cloud-Native Molecular-Modeling and Drug-Design Workflows

## Overview

Rowan is a cloud-native workflow platform for molecular simulation, medicinal chemistry, and structure-based design. Its Python API exposes a unified interface for small-molecule modeling, property prediction, docking, molecular dynamics, and AI structure workflows.

Use Rowan when you want to run medicinal-chemistry or molecular-design workflows programmatically without maintaining local HPC infrastructure, GPU provisioning, or a collection of separate modeling tools. Rowan handles all infrastructure, result management, and computation scaling.

## When to use Rowan


---

## scanpy

---
name: scanpy
description: Standard single-cell RNA-seq analysis pipeline. Use for QC, normalization, dimensionality reduction (PCA/UMAP/t-SNE), clustering, differential expression, and visualization. Best for exploratory scRNA-seq analysis with established workflows. For deep learning models use scvi-tools; for data format questions use anndata.
license: SD-3-Clause license
metadata:
    skill-author: K-Dense Inc.
---

# Scanpy: Single-Cell Analysis

## Overview

Scanpy is a scalable Python toolkit for analyzing single-cell RNA-seq data, built on AnnData. Apply this skill for complete single-cell workflows including quality control, normalization, dimensionality reduction, clustering, marker gene identification, visualization, and trajectory analysis.

## When to Use This Skill

This skill should be used when:
- Analyzing single-cell RNA-seq data (.h5ad, 10X, CSV formats)
- Performing quality control on scRNA-seq datasets
- Creating UMAP, t-SNE, or PCA visualizations

---

## scholar-evaluation

---
name: scholar-evaluation
description: Systematically evaluate scholarly work using the ScholarEval framework, providing structured assessment across research quality dimensions including problem formulation, methodology, analysis, and writing with quantitative scoring and actionable feedback.
license: MIT license
metadata:
    skill-author: K-Dense Inc.
---

# Scholar Evaluation

## Overview

Apply the ScholarEval framework to systematically evaluate scholarly and research work. This skill provides structured evaluation methodology based on peer-reviewed research assessment criteria, enabling comprehensive analysis of academic papers, research proposals, literature reviews, and scholarly writing across multiple quality dimensions.

## When to Use This Skill

Use this skill when:
- Evaluating research papers for quality and rigor
- Assessing literature review comprehensiveness and quality
- Reviewing research methodology design

---

## scientific-brainstorming

---
name: scientific-brainstorming
description: Creative research ideation and exploration. Use for open-ended brainstorming sessions, exploring interdisciplinary connections, challenging assumptions, or identifying research gaps. Best for early-stage research planning when you do not have specific observations yet. For formulating testable hypotheses from data use hypothesis-generation.
license: MIT license
metadata:
    skill-author: K-Dense Inc.
---

# Scientific Brainstorming

## Overview

Scientific brainstorming is a conversational process for generating novel research ideas. Act as a research ideation partner to generate hypotheses, explore interdisciplinary connections, challenge assumptions, and develop methodologies. Apply this skill for creative scientific problem-solving.

## When to Use This Skill

This skill should be used when:
- Generating novel research ideas or directions
- Exploring interdisciplinary connections and analogies
- Challenging assumptions in existing research frameworks

---

## scientific-critical-thinking

---
name: scientific-critical-thinking
description: Evaluate scientific claims and evidence quality. Use for assessing experimental design validity, identifying biases and confounders, applying evidence grading frameworks (GRADE, Cochrane Risk of Bias), or teaching critical analysis. Best for understanding evidence quality, identifying flaws. For formal peer review writing use peer-review.
allowed-tools: Read Write Edit Bash
license: MIT license
metadata:
    skill-author: K-Dense Inc.
---

# Scientific Critical Thinking

## Overview

Critical thinking is a systematic process for evaluating scientific rigor. Assess methodology, experimental design, statistical validity, biases, confounding, and evidence quality using GRADE and Cochrane ROB frameworks. Apply this skill for critical analysis of scientific claims.

## When to Use This Skill

This skill should be used when:
- Evaluating research methodology and experimental design
- Assessing statistical validity and evidence quality

---

## scientific-schematics

---
name: scientific-schematics
description: Create publication-quality scientific diagrams using Nano Banana 2 AI with smart iterative refinement. Uses Gemini 3.1 Pro Preview for quality review. Only regenerates if quality is below threshold for your document type. Specialized in neural network architectures, system diagrams, flowcharts, biological pathways, and complex scientific visualizations.
allowed-tools: Read Write Edit Bash
license: MIT license
metadata:
    skill-author: K-Dense Inc.
---

# Scientific Schematics and Diagrams

## Overview

Scientific schematics and diagrams transform complex concepts into clear visual representations for publication. **This skill uses Nano Banana 2 AI for diagram generation with Gemini 3.1 Pro Preview quality review.**

**How it works:**
- Describe your diagram in natural language
- Nano Banana 2 generates publication-quality images automatically
- **Gemini 3.1 Pro Preview reviews quality** against document-type thresholds
- **Smart iteration**: Only regenerates if quality is below threshold

---

## scientific-slides

---
name: scientific-slides
description: Build slide decks and presentations for research talks. Use this for making PowerPoint slides, conference presentations, seminar talks, research presentations, thesis defense slides, or any scientific talk. Provides slide structure, design templates, timing guidance, and visual validation. Works with PowerPoint and LaTeX Beamer.
allowed-tools: Read Write Edit Bash
license: MIT license
metadata:
    skill-author: K-Dense Inc.
---

# Scientific Slides

## Overview

Scientific presentations are a critical medium for communicating research, sharing findings, and engaging with academic and professional audiences. This skill provides comprehensive guidance for creating effective scientific presentations, from structure and content development to visual design and delivery preparation.

**Key Focus**: Oral presentations for conferences, seminars, defenses, and professional talks.

**CRITICAL DESIGN PHILOSOPHY**: Scientific presentations should be VISUALLY ENGAGING and RESEARCH-BACKED. Avoid dry, text-heavy slides at all costs. Great scientific presentations combine:
- **Compelling visuals**: High-quality figures, images, diagrams (not just bullet points)
- **Research context**: Proper citations from research-lookup establishing credibility

---

## scientific-visualization

---
name: scientific-visualization
description: Meta-skill for publication-ready figures. Use when creating journal submission figures requiring multi-panel layouts, significance annotations, error bars, colorblind-safe palettes, and specific journal formatting (Nature, Science, Cell). Orchestrates matplotlib/seaborn/plotly with publication styles. For quick exploration use seaborn or plotly directly.
license: MIT license
metadata:
    skill-author: K-Dense Inc.
---

# Scientific Visualization

## Overview

Scientific visualization transforms data into clear, accurate figures for publication. Create journal-ready plots with multi-panel layouts, error bars, significance markers, and colorblind-safe palettes. Export as PDF/EPS/TIFF using matplotlib, seaborn, and plotly for manuscripts.

## When to Use This Skill

This skill should be used when:
- Creating plots or visualizations for scientific manuscripts
- Preparing figures for journal submission (Nature, Science, Cell, PLOS, etc.)
- Ensuring figures are colorblind-friendly and accessible

---

## scientific-writing

---
name: scientific-writing
description: Core skill for the deep research and writing tool. Write scientific manuscripts in full paragraphs (never bullet points). Use two-stage process with (1) section outlines with key points using research-lookup then (2) convert to flowing prose. IMRAD structure, citations (APA/AMA/Vancouver), figures/tables, reporting guidelines (CONSORT/STROBE/PRISMA), for research papers and journal submissions.
allowed-tools: Read Write Edit Bash
license: MIT license
metadata:
    skill-author: K-Dense Inc.
---

# Scientific Writing

## Overview

**This is the core skill for the deep research and writing tool**—combining AI-driven deep research with well-formatted written outputs. Every document produced is backed by comprehensive literature search and verified citations through the research-lookup skill.

Scientific writing is a process for communicating research with precision and clarity. Write manuscripts using IMRAD structure, citations (APA/AMA/Vancouver), figures/tables, and reporting guidelines (CONSORT/STROBE/PRISMA). Apply this skill for research papers and journal submissions.

**Critical Principle: Always write in full paragraphs with flowing prose. Never submit bullet points in the final manuscript.** Use a two-stage process: first create section outlines with key points using research-lookup, then convert those outlines into complete paragraphs.

## When to Use This Skill

---

## scikit-bio

---
name: scikit-bio
description: Biological data toolkit. Sequence analysis, alignments, phylogenetic trees, diversity metrics (alpha/beta, UniFrac), ordination (PCoA), PERMANOVA, FASTA/Newick I/O, for microbiome analysis.
license: BSD-3-Clause license
metadata:
    skill-author: K-Dense Inc.
---

# scikit-bio

## Overview

scikit-bio is a comprehensive Python library for working with biological data. Apply this skill for bioinformatics analyses spanning sequence manipulation, alignment, phylogenetics, microbial ecology, and multivariate statistics.

## When to Use This Skill

This skill should be used when the user:
- Works with biological sequences (DNA, RNA, protein)
- Needs to read/write biological file formats (FASTA, FASTQ, GenBank, Newick, BIOM, etc.)
- Performs sequence alignments or searches for motifs

---

## scikit-learn

---
name: scikit-learn
description: Machine learning in Python with scikit-learn. Use when working with supervised learning (classification, regression), unsupervised learning (clustering, dimensionality reduction), model evaluation, hyperparameter tuning, preprocessing, or building ML pipelines. Provides comprehensive reference documentation for algorithms, preprocessing techniques, pipelines, and best practices.
license: BSD-3-Clause license
metadata:
    skill-author: K-Dense Inc.
---

# Scikit-learn

## Overview

This skill provides comprehensive guidance for machine learning tasks using scikit-learn, the industry-standard Python library for classical machine learning. Use this skill for classification, regression, clustering, dimensionality reduction, preprocessing, model evaluation, and building production-ready ML pipelines.

## Installation

```bash
# Install scikit-learn using uv
uv uv pip install scikit-learn


---

## scikit-survival

---
name: scikit-survival
description: Comprehensive toolkit for survival analysis and time-to-event modeling in Python using scikit-survival. Use this skill when working with censored survival data, performing time-to-event analysis, fitting Cox models, Random Survival Forests, Gradient Boosting models, or Survival SVMs, evaluating survival predictions with concordance index or Brier score, handling competing risks, or implementing any survival analysis workflow with the scikit-survival library.
license: GPL-3.0 license
metadata:
    skill-author: K-Dense Inc.
---

# scikit-survival: Survival Analysis in Python

## Overview

scikit-survival is a Python library for survival analysis built on top of scikit-learn. It provides specialized tools for time-to-event analysis, handling the unique challenge of censored data where some observations are only partially known.

Survival analysis aims to establish connections between covariates and the time of an event, accounting for censored records (particularly right-censored data from studies where participants don't experience events during observation periods).

## When to Use This Skill

Use this skill when:
- Performing survival analysis or time-to-event modeling

---

## scvelo

---
name: scvelo
description: RNA velocity analysis with scVelo. Estimate cell state transitions from unspliced/spliced mRNA dynamics, infer trajectory directions, compute latent time, and identify driver genes in single-cell RNA-seq data. Complements Scanpy/scVI-tools for trajectory inference.
license: BSD-3-Clause
metadata:
    skill-author: Kuan-lin Huang
---

# scVelo — RNA Velocity Analysis

## Overview

scVelo is the leading Python package for RNA velocity analysis in single-cell RNA-seq data. It infers cell state transitions by modeling the kinetics of mRNA splicing — using the ratio of unspliced (pre-mRNA) to spliced (mature mRNA) abundances to determine whether a gene is being upregulated or downregulated in each cell. This allows reconstruction of developmental trajectories and identification of cell fate decisions without requiring time-course data.

**Installation:** `pip install scvelo`

**Key resources:**
- Documentation: https://scvelo.readthedocs.io/
- GitHub: https://github.com/theislab/scvelo
- Paper: Bergen et al. (2020) Nature Biotechnology. PMID: 32747759

---

## scvi-tools

---
name: scvi-tools
description: Deep generative models for single-cell omics. Use when you need probabilistic batch correction (scVI), transfer learning, differential expression with uncertainty, or multi-modal integration (TOTALVI, MultiVI). Best for advanced modeling, batch effects, multimodal data. For standard analysis pipelines use scanpy.
license: BSD-3-Clause license
metadata:
    skill-author: K-Dense Inc.
---

# scvi-tools

## Overview

scvi-tools is a comprehensive Python framework for probabilistic models in single-cell genomics. Built on PyTorch and PyTorch Lightning, it provides deep generative models using variational inference for analyzing diverse single-cell data modalities.

## When to Use This Skill

Use this skill when:
- Analyzing single-cell RNA-seq data (dimensionality reduction, batch correction, integration)
- Working with single-cell ATAC-seq or chromatin accessibility data
- Integrating multimodal data (CITE-seq, multiome, paired/unpaired datasets)

---

## seaborn

---
name: seaborn
description: Statistical visualization with pandas integration. Use for quick exploration of distributions, relationships, and categorical comparisons with attractive defaults. Best for box plots, violin plots, pair plots, heatmaps. Built on matplotlib. For interactive plots use plotly; for publication styling use scientific-visualization.
license: BSD-3-Clause license
metadata:
    skill-author: K-Dense Inc.
---

# Seaborn Statistical Visualization

## Overview

Seaborn is a Python visualization library for creating publication-quality statistical graphics. Use this skill for dataset-oriented plotting, multivariate analysis, automatic statistical estimation, and complex multi-panel figures with minimal code.

## Design Philosophy

Seaborn follows these core principles:

1. **Dataset-oriented**: Work directly with DataFrames and named variables rather than abstract coordinates
2. **Semantic mapping**: Automatically translate data values into visual properties (colors, sizes, styles)

---

## shap

---
name: shap
description: Model interpretability and explainability using SHAP (SHapley Additive exPlanations). Use this skill when explaining machine learning model predictions, computing feature importance, generating SHAP plots (waterfall, beeswarm, bar, scatter, force, heatmap), debugging models, analyzing model bias or fairness, comparing models, or implementing explainable AI. Works with tree-based models (XGBoost, LightGBM, Random Forest), deep learning (TensorFlow, PyTorch), linear models, and any black-box model.
license: MIT license
metadata:
    skill-author: K-Dense Inc.
---

# SHAP (SHapley Additive exPlanations)

## Overview

SHAP is a unified approach to explain machine learning model outputs using Shapley values from cooperative game theory. This skill provides comprehensive guidance for:

- Computing SHAP values for any model type
- Creating visualizations to understand feature importance
- Debugging and validating model behavior
- Analyzing fairness and bias
- Implementing explainable AI in production


---

## simpy

---
name: simpy
description: Process-based discrete-event simulation framework in Python. Use this skill when building simulations of systems with processes, queues, resources, and time-based events such as manufacturing systems, service operations, network traffic, logistics, or any system where entities interact with shared resources over time.
license: MIT license
metadata:
    skill-author: K-Dense Inc.
---

# SimPy - Discrete-Event Simulation

## Overview

SimPy is a process-based discrete-event simulation framework based on standard Python. Use SimPy to model systems where entities (customers, vehicles, packets, etc.) interact with each other and compete for shared resources (servers, machines, bandwidth, etc.) over time.

**Core capabilities:**
- Process modeling using Python generator functions
- Shared resource management (servers, containers, stores)
- Event-driven scheduling and synchronization
- Real-time simulations synchronized with wall-clock time
- Comprehensive monitoring and data collection

---

## stable-baselines3

---
name: stable-baselines3
description: Production-ready reinforcement learning algorithms (PPO, SAC, DQN, TD3, DDPG, A2C) with scikit-learn-like API. Use for standard RL experiments, quick prototyping, and well-documented algorithm implementations. Best for single-agent RL with Gymnasium environments. For high-performance parallel training, multi-agent systems, or custom vectorized environments, use pufferlib instead.
license: MIT license
metadata:
    skill-author: K-Dense Inc.
---

# Stable Baselines3

## Overview

Stable Baselines3 (SB3) is a PyTorch-based library providing reliable implementations of reinforcement learning algorithms. This skill provides comprehensive guidance for training RL agents, creating custom environments, implementing callbacks, and optimizing training workflows using SB3's unified API.

## Core Capabilities

### 1. Training RL Agents

**Basic Training Pattern:**


---

## statistical-analysis

---
name: statistical-analysis
description: Guided statistical analysis with test selection and reporting. Use when you need help choosing appropriate tests for your data, assumption checking, power analysis, and APA-formatted results. Best for academic research reporting, test selection guidance. For implementing specific models programmatically use statsmodels.
license: MIT license
metadata:
    skill-author: K-Dense Inc.
---

# Statistical Analysis

## Overview

Statistical analysis is a systematic process for testing hypotheses and quantifying relationships. Conduct hypothesis tests (t-test, ANOVA, chi-square), regression, correlation, and Bayesian analyses with assumption checks and APA reporting. Apply this skill for academic research.

## When to Use This Skill

This skill should be used when:
- Conducting statistical hypothesis tests (t-tests, ANOVA, chi-square)
- Performing regression or correlation analyses
- Running Bayesian statistical analyses

---

## statsmodels

---
name: statsmodels
description: Statistical models library for Python. Use when you need specific model classes (OLS, GLM, mixed models, ARIMA) with detailed diagnostics, residuals, and inference. Best for econometrics, time series, rigorous inference with coefficient tables. For guided statistical test selection with APA reporting use statistical-analysis.
license: BSD-3-Clause license
metadata:
    skill-author: K-Dense Inc.
---

# Statsmodels: Statistical Modeling and Econometrics

## Overview

Statsmodels is Python's premier library for statistical modeling, providing tools for estimation, inference, and diagnostics across a wide range of statistical methods. Apply this skill for rigorous statistical analysis, from simple linear regression to complex time series models and econometric analyses.

## When to Use This Skill

This skill should be used when:
- Fitting regression models (OLS, WLS, GLS, quantile regression)
- Performing generalized linear modeling (logistic, Poisson, Gamma, etc.)
- Analyzing discrete outcomes (binary, multinomial, count, ordinal)

---

## sympy

---
name: sympy
description: Use this skill when working with symbolic mathematics in Python. This skill should be used for symbolic computation tasks including solving equations algebraically, performing calculus operations (derivatives, integrals, limits), manipulating algebraic expressions, working with matrices symbolically, physics calculations, number theory problems, geometry computations, and generating executable code from mathematical expressions. Apply this skill when the user needs exact symbolic results rather than numerical approximations, or when working with mathematical formulas that contain variables and parameters.
license: https://github.com/sympy/sympy/blob/master/LICENSE
metadata:
    skill-author: K-Dense Inc.
---

# SymPy - Symbolic Mathematics in Python

## Overview

SymPy is a Python library for symbolic mathematics that enables exact computation using mathematical symbols rather than numerical approximations. This skill provides comprehensive guidance for performing symbolic algebra, calculus, linear algebra, equation solving, physics calculations, and code generation using SymPy.

## When to Use This Skill

Use this skill when:
- Solving equations symbolically (algebraic, differential, systems of equations)
- Performing calculus operations (derivatives, integrals, limits, series)
- Manipulating and simplifying algebraic expressions

---

## tiledbvcf

---
name: tiledbvcf
description: Efficient storage and retrieval of genomic variant data using TileDB. Scalable VCF/BCF ingestion, incremental sample addition, compressed storage, parallel queries, and export capabilities for population genomics.
license: MIT license
metadata:
    skill-author: Jeremy Leipzig
---

# TileDB-VCF

## Overview

TileDB-VCF is a high-performance C++ library with Python and CLI interfaces for efficient storage and retrieval of genomic variant-call data. Built on TileDB's sparse array technology, it enables scalable ingestion of VCF/BCF files, incremental sample addition without expensive merging operations, and efficient parallel queries of variant data stored locally or in the cloud.

## When to Use This Skill

This skill should be used when:
- Learning TileDB-VCF concepts and workflows
- Prototyping genomics analyses and pipelines
- Working with small-to-medium datasets (< 1000 samples)

---

## timesfm-forecasting

---
name: timesfm-forecasting
description: Zero-shot time series forecasting with Google's TimesFM foundation model. Use for any univariate time series (sales, sensors, energy, vitals, weather) without training a custom model. Supports CSV/DataFrame/array inputs with point forecasts and prediction intervals. Includes a preflight system checker script to verify RAM/GPU before first use.
allowed-tools: Read Write Edit Bash
license: Apache-2.0 license
metadata:
  skill-author: Clayton Young / Superior Byte Works, LLC (@borealBytes)
  skill-version: "1.0.0"
---

# TimesFM Forecasting

## Overview

TimesFM (Time Series Foundation Model) is a pretrained decoder-only foundation model
developed by Google Research for time-series forecasting. It works **zero-shot** — feed it
any univariate time series and it returns point forecasts with calibrated quantile
prediction intervals, no training required.

This skill wraps TimesFM for safe, agent-friendly local inference. It includes a

---

## torch-geometric

---
name: torch-geometric
description: "Guide for building Graph Neural Networks with PyTorch Geometric (PyG). Use this skill whenever the user asks about graph neural networks, GNNs, node classification, link prediction, graph classification, message passing networks, heterogeneous graphs, neighbor sampling, or any task involving torch_geometric / PyG. Also trigger when you see imports from torch_geometric, or the user mentions graph convolutions (GCN, GAT, GraphSAGE, GIN), graph data structures, or working with relational/network data. Even if the user just says 'graph learning' or 'geometric deep learning', use this skill."
---

# PyTorch Geometric (PyG)

PyG is the standard library for Graph Neural Networks built on PyTorch. It provides data structures for graphs, 60+ GNN layer implementations, scalable mini-batch training, and support for heterogeneous graphs.

Install: `uv add torch_geometric` (or `uv pip install torch_geometric`; requires PyTorch). Optional: `pyg-lib`, `torch-scatter`, `torch-sparse`, `torch-cluster` for accelerated ops.

## Core Concepts

### Graph Data: `Data` and `HeteroData`

A graph lives in a `Data` object. The key attributes:

```python
from torch_geometric.data import Data


---

## torchdrug

---
name: torchdrug
description: PyTorch-native graph neural networks for molecules and proteins. Use when building custom GNN architectures for drug discovery, protein modeling, or knowledge graph reasoning. Best for custom model development, protein property prediction, retrosynthesis. For pre-trained models and diverse featurizers use deepchem; for benchmark datasets use pytdc.
license: Apache-2.0 license
metadata:
    skill-author: K-Dense Inc.
---

# TorchDrug

## Overview

TorchDrug is a comprehensive PyTorch-based machine learning toolbox for drug discovery and molecular science. Apply graph neural networks, pre-trained models, and task definitions to molecules, proteins, and biological knowledge graphs, including molecular property prediction, protein modeling, knowledge graph reasoning, molecular generation, retrosynthesis planning, with 40+ curated datasets and 20+ model architectures.

## When to Use This Skill

This skill should be used when working with:

**Data Types:**
- SMILES strings or molecular structures

---

## transformers

---
name: transformers
description: This skill should be used when working with pre-trained transformer models for natural language processing, computer vision, audio, or multimodal tasks. Use for text generation, classification, question answering, translation, summarization, image classification, object detection, speech recognition, and fine-tuning models on custom datasets.
license: Apache-2.0 license
compatibility: Some features require an Huggingface token
metadata:
    skill-author: K-Dense Inc.
---

# Transformers

## Overview

The Hugging Face Transformers library provides access to thousands of pre-trained models for tasks across NLP, computer vision, audio, and multimodal domains. Use this skill to load models, perform inference, and fine-tune on custom data.

## Installation

Install transformers and core dependencies:

```bash

---

## treatment-plans

---
name: treatment-plans
description: Generate concise (3-4 page), focused medical treatment plans in LaTeX/PDF format for all clinical specialties. Supports general medical treatment, rehabilitation therapy, mental health care, chronic disease management, perioperative care, and pain management. Includes SMART goal frameworks, evidence-based interventions with minimal text citations, regulatory compliance (HIPAA), and professional formatting. Prioritizes brevity and clinical actionability.
allowed-tools: Read Write Edit Bash
license: MIT license
metadata:
    skill-author: K-Dense Inc.
---

# Treatment Plan Writing

## Overview

Treatment plan writing is the systematic documentation of clinical care strategies designed to address patient health conditions through evidence-based interventions, measurable goals, and structured follow-up. This skill provides comprehensive LaTeX templates and validation tools for creating **concise, focused** treatment plans (3-4 pages standard) across all medical specialties with full regulatory compliance.

**Critical Principles:**
1. **CONCISE & ACTIONABLE**: Treatment plans default to 3-4 pages maximum, focusing only on clinically essential information that impacts care decisions
2. **Patient-Centered**: Plans must be evidence-based, measurable, and compliant with healthcare regulations (HIPAA, documentation standards)
3. **Minimal Citations**: Use brief in-text citations only when needed to support clinical recommendations; avoid extensive bibliographies


---

## umap-learn

---
name: umap-learn
description: UMAP dimensionality reduction. Fast nonlinear manifold learning for 2D/3D visualization, clustering preprocessing (HDBSCAN), supervised/parametric UMAP, for high-dimensional data.
license: BSD-3-Clause license
metadata:
    skill-author: K-Dense Inc.
---

# UMAP-Learn

## Overview

UMAP (Uniform Manifold Approximation and Projection) is a dimensionality reduction technique for visualization and general non-linear dimensionality reduction. Apply this skill for fast, scalable embeddings that preserve local and global structure, supervised learning, and clustering preprocessing.

## Quick Start

### Installation

```bash
uv pip install umap-learn

---

## usfiscaldata

---
name: usfiscaldata
description: Query the U.S. Treasury Fiscal Data API for federal financial data including national debt, government spending, revenue, interest rates, exchange rates, and savings bonds. Access 54 datasets and 182 data tables with no API key required. Use when working with U.S. federal fiscal data, national debt tracking (Debt to the Penny), Daily Treasury Statements, Monthly Treasury Statements, Treasury securities auctions, interest rates on Treasury securities, foreign exchange rates, savings bonds, or any U.S. government financial statistics.
license: MIT
metadata:
    skill-author: K-Dense Inc.
---

# U.S. Treasury Fiscal Data API

Free, open REST API from the U.S. Department of the Treasury for federal financial data. No API key or registration required.

**Base URL:** `https://api.fiscaldata.treasury.gov/services/api/fiscal_service`

## Quick Start

```python
import requests
import pandas as pd


---

## vaex

---
name: vaex
description: Use this skill for processing and analyzing large tabular datasets (billions of rows) that exceed available RAM. Vaex excels at out-of-core DataFrame operations, lazy evaluation, fast aggregations, efficient visualization of big data, and machine learning on large datasets. Apply when users need to work with large CSV/HDF5/Arrow/Parquet files, perform fast statistics on massive datasets, create visualizations of big data, or build ML pipelines that do not fit in memory.
license: MIT license
metadata:
    skill-author: K-Dense Inc.
---

# Vaex

## Overview

Vaex is a high-performance Python library designed for lazy, out-of-core DataFrames to process and visualize tabular datasets that are too large to fit into RAM. Vaex can process over a billion rows per second, enabling interactive data exploration and analysis on datasets with billions of rows.

## When to Use This Skill

Use Vaex when:
- Processing tabular datasets larger than available RAM (gigabytes to terabytes)
- Performing fast statistical aggregations on massive datasets
- Creating visualizations and heatmaps of large datasets

---

## venue-templates

---
name: venue-templates
description: Access comprehensive LaTeX templates, formatting requirements, and submission guidelines for major scientific publication venues (Nature, Science, PLOS, IEEE, ACM), academic conferences (NeurIPS, ICML, CVPR, CHI), research posters, and grant proposals (NSF, NIH, DOE, DARPA). This skill should be used when preparing manuscripts for journal submission, conference papers, research posters, or grant proposals and need venue-specific formatting requirements and templates.
allowed-tools: Read Write Edit Bash
license: MIT license
metadata:
    skill-author: K-Dense Inc.
---

# Venue Templates

## Overview

Access comprehensive LaTeX templates, formatting requirements, and submission guidelines for major scientific publication venues, academic conferences, research posters, and grant proposals. This skill provides ready-to-use templates and detailed specifications for successful academic submissions across disciplines.

Use this skill when preparing manuscripts for journal submission, conference papers, research posters, or grant proposals and need venue-specific formatting requirements and templates.

## When to Use This Skill

This skill should be used when:

---

## what-if-oracle

---
name: what-if-oracle
description: Run structured What-If scenario analysis with multi-branch possibility exploration. Use this skill when the user asks speculative questions like "what if...", "what would happen if...", "what are the possibilities", "explore scenarios", "scenario analysis", "possibility space", "what could go wrong", "best case / worst case", "risk analysis", "contingency planning", "strategic options", or any question about uncertain futures. Also trigger when the user faces a fork-in-the-road decision, wants to stress-test an idea, or needs to think through consequences before committing.
allowed-tools: Read Write
license: MIT license
metadata:
  skill-author: AHK Strategies (ashrafkahoush-ux)
---

# What-If Oracle — Possibility Space Explorer

A structured system for exploring uncertain futures through rigorous multi-branch scenario analysis. Instead of one prediction, the Oracle maps the full **possibility space** — branching timelines where each path has its own logic, probability, and consequences.

Based on the What-If Statement paradigm: the idea that speculative questions ("What if X?") are not idle daydreaming but a **fundamental computing operation** — the mind's way of simulating futures before committing resources to one.

Published research: [The What-If Statement (DOI: 10.5281/zenodo.18736841)](https://doi.org/10.5281/zenodo.18736841) | [IDNA Consolidation v2 (DOI: 10.5281/zenodo.18807387)](https://doi.org/10.5281/zenodo.18807387)

## Core Principle: 0·IF·1

Every scenario analysis has three elements:

---

## xlsx

---
name: xlsx
description: "Use this skill any time a spreadsheet file is the primary input or output. This means any task where the user wants to: open, read, edit, or fix an existing .xlsx, .xlsm, .csv, or .tsv file (e.g., adding columns, computing formulas, formatting, charting, cleaning messy data); create a new spreadsheet from scratch or from other data sources; or convert between tabular file formats. Trigger especially when the user references a spreadsheet file by name or path — even casually (like \"the xlsx in my downloads\") — and wants something done to it or produced from it. Also trigger for cleaning or restructuring messy tabular data files (malformed rows, misplaced headers, junk data) into proper spreadsheets. The deliverable must be a spreadsheet file. Do NOT trigger when the primary deliverable is a Word document, HTML report, standalone Python script, database pipeline, or Google Sheets API integration, even if tabular data is involved."
license: Proprietary. LICENSE.txt has complete terms
---

# Requirements for Outputs

## All Excel files

### Professional Font
- Use a consistent, professional font (e.g., Arial, Times New Roman) for all deliverables unless otherwise instructed by the user

### Zero Formula Errors
- Every Excel model MUST be delivered with ZERO formula errors (#REF!, #DIV/0!, #VALUE!, #N/A, #NAME?)

### Preserve Existing Templates (when updating templates)
- Study and EXACTLY match existing format, style, and conventions when modifying files
- Never impose standardized formatting on files with established patterns
- Existing template conventions ALWAYS override these guidelines

---

## zarr-python

---
name: zarr-python
description: Chunked N-D arrays for cloud storage. Compressed arrays, parallel I/O, S3/GCS integration, NumPy/Dask/Xarray compatible, for large-scale scientific computing pipelines.
license: MIT license
metadata:
    skill-author: K-Dense Inc.
---

# Zarr Python

## Overview

Zarr is a Python library for storing large N-dimensional arrays with chunking and compression. Apply this skill for efficient parallel I/O, cloud-native workflows, and seamless integration with NumPy, Dask, and Xarray.

## Quick Start

### Installation

```bash
uv pip install zarr

---

