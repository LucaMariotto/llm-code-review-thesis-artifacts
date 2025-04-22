import pandas as pd
import numpy as np
import os
import logging
import re

# --- Configuration ---
QUALITY_ANALYSIS_FILE = 'pr_quality_analysis.csv'
COMPARISON_FILE = 'enhanced_pr_comparison.csv'
OUTPUT_DIR = 'model_agnostic_analysis_summary' # Directory to save summary files

# Key metrics from pr_quality_analysis.csv to summarize variability
METRICS_TO_ANALYZE_VARIABILITY = [
    'overall_word_count',
    'body_readability_flesch_reading_ease',
    'body_readability_flesch_kincaid_grade',
    'overall_sentiment_compound'
]

# Key metrics from enhanced_pr_comparison.csv to summarize similarity
SIMILARITY_METRICS_TO_ANALYZE = [
    'bleu',
    'jaccard',
    'edit_distance', # Note: Lower is *more* similar here
    'sbert_cosine',
    'bertscore_f1',
    'body_rougeL',   # Using ROUGE-L F1 for body as representative
    'tfidf_cosine'
]

# --- Setup ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Helper Functions ---
def load_csv_safe(filepath, index_col=None):
    """Safely loads a CSV file."""
    logger.info(f"Attempting to load: {filepath}")
    try:
        if not os.path.exists(filepath):
            logger.error(f"File not found: {filepath}")
            return None
        df = pd.read_csv(filepath)
        # Basic check for empty file content besides header
        if df.empty:
             logger.warning(f"Loaded CSV is empty (may only contain header): {filepath}")
             # Return empty DataFrame structure if possible, or None
             try:
                 # Try reading again just to get columns if possible
                 header_df = pd.read_csv(filepath, nrows=0)
                 return pd.DataFrame(columns=header_df.columns)
             except Exception:
                 return None

        logger.info(f"Successfully loaded: {os.path.basename(filepath)} ({len(df)} rows)")
        return df
    except pd.errors.EmptyDataError:
         logger.error(f"EmptyDataError: No data or columns found in {filepath}")
         return None
    except Exception as e:
        logger.error(f"Could not load file {filepath}: {e}")
        return None

def analyze_metric_variability(quality_df, metrics, output_dir):
    """Calculates and saves descriptive stats for metrics grouped by version and type."""
    logger.info("--- Analyzing Metric Variability Across Models ---")
    if quality_df is None or quality_df.empty:
        logger.warning("Quality analysis dataframe is empty. Skipping variability analysis.")
        return

    # Ensure required columns exist
    required_cols = ['version', 'type'] + metrics
    missing_cols = [col for col in required_cols if col not in quality_df.columns]
    if missing_cols:
        logger.error(f"Missing required columns in {QUALITY_ANALYSIS_FILE}: {missing_cols}. Skipping variability analysis.")
        return

    valid_metrics = [m for m in metrics if m in quality_df.columns]
    if not valid_metrics:
        logger.error(f"None of the specified metrics_to_analyze exist in {QUALITY_ANALYSIS_FILE}. Skipping.")
        return

    # 1. Group by Version (O, D, ID, IO) - Across all models
    logger.info("Calculating stats grouped by version type...")
    try:
        # Use agg for specific stats
        agg_funcs = ['mean', 'std', 'min', 'max', 'count']
        summary_by_version = quality_df.groupby('version')[valid_metrics].agg(agg_funcs)
        # Optional: Reformat multi-index columns for better readability
        summary_by_version.columns = ['_'.join(col).strip() for col in summary_by_version.columns.values]

        output_path_version = os.path.join(output_dir, 'metrics_summary_by_version.csv')
        summary_by_version.to_csv(output_path_version, float_format='%.3f')
        logger.info(f"Summary by version type saved to: {output_path_version}")
        # Log a snippet
        logger.info("Snippet of Summary by Version:\n" + summary_by_version.head().to_string(float_format='%.2f'))

    except Exception as e:
        logger.error(f"Error calculating summary by version: {e}", exc_info=True)

    # 2. Group by Type (LLM/baseline) - Across all versions/PRs
    logger.info("Calculating stats grouped by LLM type...")
    try:
        summary_by_llm = quality_df.groupby('type')[valid_metrics].agg(agg_funcs)
        summary_by_llm.columns = ['_'.join(col).strip() for col in summary_by_llm.columns.values]

        output_path_llm = os.path.join(output_dir, 'metrics_summary_by_llm.csv')
        summary_by_llm.to_csv(output_path_llm, float_format='%.3f')
        logger.info(f"Summary by LLM type saved to: {output_path_llm}")
        # Log a snippet
        logger.info("Snippet of Summary by LLM:\n" + summary_by_llm.head().to_string(float_format='%.2f'))

    except Exception as e:
        logger.error(f"Error calculating summary by LLM type: {e}", exc_info=True)


def extract_llm_info(version_string):
    """Extracts PR num, LLM name, and version type from 'other_llm_version' string."""
    # Expected format: PRNUM_LLMNAME_VERSIONTYPE (e.g., 1_claude 3.7 sonnet_degraded)
    # Handle potential variations if needed
    parts = str(version_string).split('_')
    if len(parts) < 3:
        logger.warning(f"Unexpected format in 'other_llm_version': {version_string}. Returning Nones.")
        return None, None, None
    pr_num = parts[0]
    version_type = parts[-1]
    llm_name = '_'.join(parts[1:-1]) # Join middle parts for multi-word names
    # Basic validation
    if not pr_num.isdigit(): pr_num = None
    if version_type not in ['degraded', 'improved_degraded', 'improved_original']: version_type = None

    return pr_num, llm_name, version_type


def analyze_similarity(comparison_df, metrics, output_dir):
    """Calculates and saves average similarity scores grouped by LLM and version type."""
    logger.info("--- Analyzing Similarity Between Baseline and Other Models ---")
    if comparison_df is None or comparison_df.empty:
        logger.warning("Comparison dataframe is empty. Skipping similarity analysis.")
        return

    # Ensure required columns exist
    required_cols = ['other_llm_version'] + metrics
    missing_cols = [col for col in required_cols if col not in comparison_df.columns]
    if missing_cols:
        logger.error(f"Missing required columns in {COMPARISON_FILE}: {missing_cols}. Skipping similarity analysis.")
        return

    valid_metrics = [m for m in metrics if m in comparison_df.columns]
    if not valid_metrics:
        logger.error(f"None of the specified similarity_metrics exist in {COMPARISON_FILE}. Skipping.")
        return

    # Extract LLM name and version type
    logger.info("Extracting LLM info from 'other_llm_version' column...")
    extracted_info = comparison_df['other_llm_version'].apply(extract_llm_info)
    comparison_df[['parsed_pr_num', 'llm', 'version_type']] = pd.DataFrame(extracted_info.tolist(), index=comparison_df.index)

    # Drop rows where parsing failed
    initial_rows = len(comparison_df)
    comparison_df.dropna(subset=['llm', 'version_type'], inplace=True)
    dropped_rows = initial_rows - len(comparison_df)
    if dropped_rows > 0:
        logger.warning(f"Dropped {dropped_rows} rows due to parsing errors in 'other_llm_version'.")

    if comparison_df.empty:
        logger.error("No valid rows remaining after parsing LLM info. Skipping similarity analysis.")
        return

    # 1. Group by LLM (compare each LLM to baseline)
    logger.info("Calculating average similarity grouped by LLM...")
    try:
        summary_by_llm = comparison_df.groupby('llm')[valid_metrics].agg(['mean', 'std', 'count'])
        # Optional: Flatten multi-index for easier CSV reading
        summary_by_llm.columns = ['_'.join(col).strip() for col in summary_by_llm.columns.values]

        output_path_llm = os.path.join(output_dir, 'similarity_summary_by_llm.csv')
        summary_by_llm.to_csv(output_path_llm, float_format='%.4f')
        logger.info(f"Average similarity by LLM saved to: {output_path_llm}")
        logger.info("Snippet of Summary by LLM:\n" + summary_by_llm.head().to_string(float_format='%.3f'))
    except Exception as e:
        logger.error(f"Error calculating similarity summary by LLM: {e}", exc_info=True)

    # 2. Group by Version Type (compare D vs D, ID vs ID, IO vs IO across LLMs)
    logger.info("Calculating average similarity grouped by version type...")
    try:
        summary_by_version = comparison_df.groupby('version_type')[valid_metrics].agg(['mean', 'std', 'count'])
        summary_by_version.columns = ['_'.join(col).strip() for col in summary_by_version.columns.values]

        output_path_version = os.path.join(output_dir, 'similarity_summary_by_version.csv')
        summary_by_version.to_csv(output_path_version, float_format='%.4f')
        logger.info(f"Average similarity by version type saved to: {output_path_version}")
        logger.info("Snippet of Summary by Version Type:\n" + summary_by_version.head().to_string(float_format='%.3f'))
    except Exception as e:
        logger.error(f"Error calculating similarity summary by version type: {e}", exc_info=True)


# --- Main Execution ---
if __name__ == "__main__":
    logger.info("====== Starting Model Agnosticism Analysis Script ======")

    # Load data
    quality_df = load_csv_safe(QUALITY_ANALYSIS_FILE)
    comparison_df = load_csv_safe(COMPARISON_FILE)

    # Perform analyses
    analyze_metric_variability(quality_df, METRICS_TO_ANALYZE_VARIABILITY, OUTPUT_DIR)
    analyze_similarity(comparison_df, SIMILARITY_METRICS_TO_ANALYZE, OUTPUT_DIR)

    logger.info(f"====== Analysis Complete. Summary files saved in '{OUTPUT_DIR}' ======")