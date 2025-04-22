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

-   `/README.md`: This file
-   `/LICENSE`: License information (MIT License assumed)
-   `/requirements.txt`: Python dependencies
-   `/.gitignore`: Files ignored by Git
-   `scripts/`: Python scripts for fetching, scoring, analysis, etc. & JS/HTML for survey load balancer
    -   `analyze_agnosticism.py`
    -   `pr_analyzer9.py` # Or pr_quality_analyzer.py (use final name)
    -   `pr_comparer.py`
    -   `pr_fetcher_4.py`
    -   `pr_link_creator.py`
    -   `pr_scorer.py`
    -   `survey_analysis_7.py`
    -   `top_prs_getter.py`
    -   `index.html`
-   `prompts/`: Text files containing LLM prompt templates
    -   `competencies_list.txt`
    -   `degradation_prompt_template.txt`
    -   `improvement_degraded_prompt_template.txt`
    -   `improvement_original_prompt_template.txt`
    -   `json_to_timeline_prompt.txt`
-   `data/`: Contains configuration, stimuli, raw outputs, and processed analysis results
    -   `config/`
        -   `SETUP.csv`: Latin Square setup
    -   `baseline_pr_stimuli/`: Baseline JSONs (O, D, ID, IO) for the 6 PRs (ChatGPT-4o)
        -   `1/`
            -   `original_pr.json`
            -   `degraded_pr.json`
            -   `improved_original_pr.json`
            -   `improved_degraded.json`
        -   `...` (Folders 2 through 6) ...
    -   `model_agnosticism_llm_outputs/`: Generated JSONs from other LLMs for model agnosticism tests
        -   `1/`: PR Number 1
            -   `claude_3.7_sonnet/`
                -   `degraded_pr.json`
                -   `improved_degraded_pr.json`
                -   `improved_original_pr.json`
            -   `deepseek_deepthink_r1/`
            -   `gemini_2.0_flash_thinking/`
            -   `grok_3_thinking/`
            -   `qwen2.5-max_thinking/`
                -   `...` (3 json files per LLM) ...
        -   `...` (Folders 2 through 6 with LLM subfolders) ...
    -   `analysis_outputs/`: Processed data tables (CSV, XLSX) from analyses
        -   `demographic_summary.xlsx`
        -   `hypothesis_alignment_by_demographic_group.csv`
        -   `overall_alignment_descriptive_analysis.csv`
        -   `preference_correlations_kruskal_wallis.csv`
        -   `preference_correlations_spearman.csv`
        -   `specific_hypothesis_tests_formatted.csv`
        -   `specific_hypothesis_tests_raw.csv`
        -   # Model Agnosticism Summaries:
        -   `metrics_summary_by_llm.csv`
        -   `metrics_summary_by_version.csv`
        -   `similarity_summary_by_llm.csv`
        -   `similarity_summary_by_version.csv`
        -   # Repo Scoring Summaries:
        -   `cluster_characteristics.csv`
        -   `competency_correlation_matrix.csv`
        -   `overall_score_descriptives.csv`
        -   `repo_kruskal_wallis_results.csv`
        -   `repo_pr_counts.csv`
        -   `repository_mean_std_summary_filtered.csv`
        -   `repository_summary_with_clusters_pca.csv`
    -   `# Essential Raw(ish) Data for Repro:`
        -   `enhanced_pr_comparison.csv`: Raw similarity scores needed by S6
        -   `pr_quality_analysis.csv`: Raw metrics needed by S6
        -   `preprocessed_scored_prs.csv`: #<- Needed by S4 
-   `reports/`: Generated text/markdown reports summarizing analyses
    -   `meta_analysis_report.md`
    -   `meta_results_summary_report_v7.md`
    -   `report.txt`
-   `visualizations/`: Key PNG plots referenced in the thesis text
    -   `# Survey Plots:`
    -   `demographic_experience_distribution.png`
    -   `demographic_role_distribution.png`
    -   `overall_alignment_comparison_percentage.png`
    -   `# Model Agnosticism Plots:`
    -   `variability_by_llm_overall_word_count.png`
    -   `variability_by_version_body_readability_flesch_reading_ease.png`
    -   `similarity_by_llm_sbert_cosine.png`
    -   `similarity_by_llm_bleu.png`
    -   `# Repo Scoring / PR Selection Plots:`
    -   `repo_pr_counts_barplot_log_scale.png`
    -   `overall_score_histograms_bundled.png`
    -   `repo_score_distributions_bundled.png`
    -   `competency_correlation_heatmap.png`
    -   `kmeans_clusters_pca_binned_size.png`
    -   `# Approach Diagrams:`
    -   `approach_visualizations/model_agnosticism_flowchart.png`
    -   `approach_visualizations/pr_selection_funnel_matplotlib.png`
    -   `approach_visualizations/pr_variation_flowchart.png`
    -   `approach_visualizations/survey_load_balancer_flowchart.png`
    -   `# Appendix Diagrams:`
    -   `appendix/images/prototype_workflow_diagram.png`


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