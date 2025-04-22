# Artifacts for Master Thesis: A Model-Agnostic Approach for AI-Powered Code Reviews

This repository contains the code, processed data, configuration files, prompts, reports, and key visualizations supporting the Master Thesis titled:

**"A Model-Agnostic Approach for AI-Powered Code Reviews: Integrating User-Specific Nudges and Customizable Features"**

*   **Author:** Luca Mariotto
*   **Institution:** Hasso Plattner Institute, University of Potsdam
*   **Submission Date:** April 31, 2025

**Archived Version (Zenodo):**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX) **<-- REPLACE XXXXXXX with your actual Zenodo DOI number once published!**

This DOI links to the specific version of these artifacts archived on Zenodo corresponding to the submitted thesis.

## Overview

This research investigates the potential of Large Language Models (LLMs) to enhance Pull Request (PR) descriptions. The study evaluates the perceived quality of LLM-generated PR variations compared to human baselines and assesses the model agnosticism of the generation process. The artifacts in this repository allow for the replication of the data analysis presented in the thesis.

## Repository Structure
llm-pr-review-thesis-artifacts/
├── README.md # This file
├── LICENSE # License information (MIT)
├── requirements.txt # Python dependencies <--- YOU NEED TO CREATE THIS!
├── .gitignore # Files ignored by Git
│
├── scripts/ # Python scripts for fetching, scoring, analysis, etc. & JS/HTML for survey
├── prompts/ # Text files containing LLM prompt templates
│
├── data/
│ ├── config/ # Configuration files (e.g., Latin Square)
│ ├── baseline_pr_stimuli/ # Baseline JSONs (O, D, ID, IO) for the 6 PRs (ChatGPT-4o)
│ ├── model_agnosticism_llm_outputs/ # Generated JSONs from other LLMs for model agnosticism tests
│ ├── analysis_outputs/ # Processed data tables (CSV, XLSX) from analyses
│ └── # Folder containing essential raw-like data (e.g., anonymized responses, raw metrics/similarity)
│
├── reports/ # Generated text/markdown reports summarizing analyses
│
└── visualizations/ # Key PNG plots referenced in the thesis text


## Running the Analysis

The primary analyses (survey results, model agnosticism metrics) were performed using Python scripts located in the `scripts/` directory.

1.  **Prerequisites:**
    *   Python 3.9+ recommended.
    *   Install required libraries:
        ```bash
        pip install -r requirements.txt
        ```
        *Note: You need to generate `requirements.txt` based on the libraries used in your scripts (e.g., pandas, numpy, scipy, matplotlib, seaborn, nltk, transformers, sentence-transformers, torch, etc.). Run `pip freeze > requirements.txt` in your activated project environment.*

2.  **Key Scripts:**
    *   `survey_analysis_7.py`: Processes the (anonymized) survey data (`data/analysis_outputs/combined_processed_responses_v7_anonymized.csv`) and the setup config (`data/config/SETUP.csv`) to generate the hypothesis tests, demographic breakdowns, correlation analyses, and summary reports found in `reports/` and `data/analysis_outputs/`. Also generates plots in `visualizations/`.
    *   Scripts like `pr_analyzer9.py` / `pr_comparer.py` / `analyze_agnosticism.py` (referenced as S4, S5, S6 in the thesis appendix) were used to generate the raw metrics (`pr_quality_analysis.csv`) and similarity comparisons (`enhanced_pr_comparison.csv`) for the model agnosticism study, which are then summarized in `data/analysis_outputs/`. Re-running these might require API keys or specific model access setup not included here.
    *   Other scripts in `scripts/` were used for earlier phases like PR fetching, scoring, and candidate selection.

## Data Availability Statement

This repository contains scripts, configuration files, prompts, and processed/aggregated data outputs necessary to replicate the main analyses presented in the thesis.

**Included:**

*   Analysis scripts (`scripts/`)
*   Configuration files (`data/config/`)
*   LLM prompt templates (`prompts/`)
*   Anonymized, aggregated survey results (`data/analysis_outputs/`)
*   Anonymized combined processed survey responses (`data/analysis_outputs/combined_processed_responses_v7_anonymized.csv`)
*   Baseline PR stimuli JSONs (`data/baseline_pr_stimuli/`)
*   Other LLM output JSONs (`data/model_agnosticism_llm_outputs/`)
*   Raw metrics and similarity scores for model agnosticism (`data/pr_quality_analysis.csv`, `data/enhanced_pr_comparison.csv`)
*   Generated reports (`reports/`)
*   Key visualizations (`visualizations/`)

**Excluded:**

*   **Raw individual-level survey responses:** These are not shared publicly to protect participant confidentiality as outlined in the study information sheet and consent process. The provided anonymized, combined dataset (`combined_processed_responses_v7_anonymized.csv`) and aggregated results (`data/analysis_outputs/`) support the findings while maintaining privacy.
*   **Raw Qualitative Remarks:** The raw text of qualitative remarks is not shared due to the risk of potential re-identification when linked with demographic data. Key themes and illustrative, anonymized quotes derived from this data are presented within the thesis text itself.
*   **Expert Ranking Data:** Files containing expert names used for PR validation are not shared.

## License

The content of this repository is licensed under the MIT License. See the `LICENSE` file for details.

## Contact

For questions regarding the thesis or these artifacts, please contact: Luca Mariotto ([https://www.linkedin.com/in/luca-mariotto/])