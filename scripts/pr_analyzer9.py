"""
Holistic PR Analysis Script for Master Thesis (v5.0 - Integrated Meta-Report)
- Loads scored PR data from CSV.
- Performs comprehensive EDA across repositories and time.
- Analyzes competency correlations (repo-level and PR-level).
- Conducts statistical tests (Kruskal-Wallis) for group differences.
- Applies PCA and Clustering to identify repository archetypes.
- Generates detailed plots and summary tables in a PDF report and PNGs.
- Generates a Markdown meta-report summarizing key findings.
- Organizes outputs into subdirectories.
"""

import pandas as pd
import numpy as np
import glob
import os
import warnings
from datetime import datetime
from scipy import stats
# from scipy.cluster import hierarchy # Optional for dendrograms
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib # Import top-level matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker # For y-axis formatting
import seaborn as sns
import plotly.express as px
from matplotlib.backends.backend_pdf import PdfPages
import logging
from adjustText import adjust_text # For better plot labels
import re # Added for potential URL parsing in meta-report

# --- Configuration ---
INPUT_SCORED_FILE = "analysis_results/scored_repos/*_analysis.csv" # Use glob pattern
OUTPUT_DIR = "./thesis_holistic_analysis_v7/" # Output directory
META_REPORT_FILENAME = "meta_analysis_report.md" # Name for the Markdown summary

PLOT_STYLE = "seaborn-v0_8-whitegrid"
START_DATE = "2022-01-01" # Analysis start date
END_DATE = datetime.today().strftime('%Y-%m-%d') # Analysis end date
SIGNIFICANCE_LEVEL = 0.05
CLUSTER_COUNT = 3 # Adjust based on elbow method or domain knowledge
MIN_PRS_PER_REPO_FOR_STATS = 10 # Min PRs to include repo in comparisons
PCA_SIZE_BINS = 4 # Number of bins for sizing PCA/Cluster plots (e.g., 4: S, M, L, XL)
PCA_LABEL_COUNT = 15 # Number of largest repos to label on PCA plot

# Paths to optional manual ranking files (place them relative to script or use absolute paths)
# Set to None if a file doesn't exist
RENE_RANKING_FILE = "(Rene) Ranking_PRs - Sheet1.csv"
CHRIS_RANKING_FILE = "(Chris) Ranking_PRs - Sheet1.csv"

# --- Plot Subdirectory Configuration ---
PLOT_SUBDIR_DESC = "plots/descriptive"
PLOT_SUBDIR_COMP_BUNDLE = "plots/comparison/bundled"
PLOT_SUBDIR_COMP_SINGLE = "plots/comparison/individual_scores"
PLOT_SUBDIR_CORR = "plots/correlation"
PLOT_SUBDIR_PCA = "plots/pca_cluster"
PLOT_SUBDIR_TS_BUNDLE = "plots/timeseries/bundled_decomposition"
PLOT_SUBDIR_TS_SINGLE = "plots/timeseries/individual_components"

# --- Setup ---
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib') # Ignore minor style warnings
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 120)

os.makedirs(OUTPUT_DIR, exist_ok=True)
# Create plot subdirectories
for subdir in [PLOT_SUBDIR_DESC, PLOT_SUBDIR_COMP_BUNDLE, PLOT_SUBDIR_COMP_SINGLE,
               PLOT_SUBDIR_CORR, PLOT_SUBDIR_PCA, PLOT_SUBDIR_TS_BUNDLE, PLOT_SUBDIR_TS_SINGLE]:
    os.makedirs(os.path.join(OUTPUT_DIR, subdir), exist_ok=True)


# Configure logging
log_file = os.path.join(OUTPUT_DIR, 'holistic_analysis_log_v5.log')
logging.basicConfig(
    level=logging.INFO, # Set to DEBUG for more details
    format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s', # Added funcName
    handlers=[
        logging.FileHandler(log_file, mode='w'), # Overwrite log file each run
        logging.StreamHandler()
    ]
)
logging.info("="*30 + " Starting Holistic PR Analysis Script (v5.0 - Integrated Meta-Report) " + "="*30)
logging.info(f"Using input file/pattern: {INPUT_SCORED_FILE}")
logging.info(f"Output directory: {OUTPUT_DIR}")
logging.info(f"Plot subdirectories created under {OUTPUT_DIR}/plots/")

# Apply plot style cautiously
try:
    plt.style.use(PLOT_STYLE)
    logging.info(f"Plot style set to: {PLOT_STYLE}")
except OSError:
    logging.warning(f"Plot style '{PLOT_STYLE}' not found. Using default.")
    PLOT_STYLE = 'default'
    plt.style.use(PLOT_STYLE)

# --- Helper Functions ---

def save_plot(fig, filename_base, pdf_pages=None, subdir=None, is_single_component=False):
    """
    Saves plot to PNG in the specified subdirectory and optionally to PDF.
    is_single_component: Flag for individual time series components (don't add to main PDF).
    """
    logger = logging.getLogger(__name__) # Use logger defined globally
    try:
        # Construct path with subdirectory
        if subdir:
            plot_dir = os.path.join(OUTPUT_DIR, subdir)
        else:
            plot_dir = OUTPUT_DIR
        os.makedirs(plot_dir, exist_ok=True)
        filepath_png = os.path.join(plot_dir, f"{filename_base}.png")

        # Check if figure has axes AND if any of those axes contain plottable data
        has_content = False
        if fig and fig.axes:
             for ax in fig.axes:
                 # Simplified check: if there are lines, collections, patches, images, or texts, assume content
                 if ax.lines or ax.collections or ax.patches or ax.images or ax.texts:
                    # More robust check for empty scatter plots (PathCollection)
                     is_empty_scatter = False
                     if ax.collections:
                         try:
                             # Check specifically for PathCollection used by scatter plots
                             if isinstance(ax.collections[0], matplotlib.collections.PathCollection):
                                 offsets = ax.collections[0].get_offsets()
                                 # Check if offsets array is empty or masked array with no data
                                 if isinstance(offsets, np.ma.MaskedArray):
                                     if offsets.count() == 0: is_empty_scatter = True
                                 elif not isinstance(offsets, np.ndarray) or offsets.size == 0:
                                     is_empty_scatter = True # Treat non-numpy array or empty array as empty
                         except (IndexError, AttributeError):
                              pass # Ignore errors if collection is empty or malformed

                     if not is_empty_scatter:
                        has_content = True
                        break
        if not has_content:
             logger.warning(f"Skipping save for {filename_base} in {subdir}: Figure detected as empty or without plottable data.")
             if plt.fignum_exists(fig.number): # Check before closing
                plt.close(fig)
             return

        fig.savefig(filepath_png, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot: {filepath_png}")

        # Add to PDF if requested and not excluded
        if pdf_pages and not is_single_component:
            pdf_pages.savefig(fig, bbox_inches='tight')
            logger.debug(f"Added {filename_base} from {subdir} to PDF")

        if plt.fignum_exists(fig.number): # Check before closing
             plt.close(fig)

    except Exception as e:
        logger.error(f"Error saving plot {filename_base} in {subdir}: {e}", exc_info=True)
        # Attempt to close figure even if saving failed
        if fig and plt.fignum_exists(fig.number):
             plt.close(fig)


def run_kruskal_wallis(df, group_col, value_col):
    """
    Runs Kruskal-Wallis test if conditions are met.
    Checks for >= 2 groups, non-empty groups, and variance within groups.
    """
    logger = logging.getLogger(__name__)
    groups_data = []
    group_names = []
    valid_groups_count = 0

    if value_col not in df.columns:
        logger.error(f"Value column '{value_col}' not found in dataframe for Kruskal-Wallis.")
        return np.nan, np.nan
    if not df[value_col].notna().any():
        logger.debug(f"No valid (non-NaN) values for '{value_col}' before grouping.")
        return np.nan, np.nan


    for name, group in df.groupby(group_col):
        group_scores = group[value_col].dropna()
        if not group_scores.empty:
            unique_scores = group_scores.nunique()
            if unique_scores > 1: # Check for variance here
                groups_data.append(group_scores.values)
                group_names.append(name)
                valid_groups_count += 1
                logger.debug(f"  KW Group Check - Group: {name}, Score: {value_col}, Valid Count: {len(group_scores)}, Variance: Yes")
            else:
                logger.debug(f"  KW Group Check - Group: {name}, Score: {value_col}, Valid Count: {len(group_scores)}, Variance: No (constant value)")
        else:
            logger.debug(f"  KW Group Check - Group: {name}, Score: {value_col}, Valid Count: 0 (empty or all NaN)")

    if valid_groups_count < 2:
        logger.warning(f"Skipping Kruskal-Wallis for '{value_col}' by '{group_col}': Found only {valid_groups_count} group(s) with internal variance. Need at least 2.")
        return np.nan, np.nan

    logger.debug(f"Running Kruskal-Wallis for '{value_col}' with {len(groups_data)} groups having variance: {group_names}")
    try:
        # Ensure data is float for stats calculation
        groups_float = [g.astype(float) for g in groups_data]
        stat, p_val = stats.kruskal(*groups_float)
        if pd.isna(stat) or pd.isna(p_val):
             logger.warning(f"Kruskal-Wallis for '{value_col}' by '{group_col}' returned NaN from scipy. Stat={stat}, P-value={p_val}")
             return np.nan, np.nan
        logger.debug(f"  KW Result for '{value_col}': Stat={stat:.4g}, P-value={p_val:.4g}")
        return stat, p_val
    except ValueError as e:
        logger.warning(f"Kruskal-Wallis ValueError for '{value_col}' by '{group_col}': {e}.")
        return np.nan, np.nan
    except Exception as e:
        logger.error(f"Unexpected error in Kruskal-Wallis calculation for '{value_col}' by '{group_col}': {e}", exc_info=True)
        return np.nan, np.nan


def load_and_preprocess_data(file_pattern_or_path):
    """Load and preprocess scored PR data from CSV file(s)."""
    logger = logging.getLogger(__name__)
    logger.info("Loading and preprocessing scored PR data...")
    loaded_files = []
    try:
        # Handle potential glob pattern
        if "*" in file_pattern_or_path or "?" in file_pattern_or_path: # More robust glob check
            files = glob.glob(file_pattern_or_path)
            if not files:
                raise FileNotFoundError(f"No files found matching pattern: {file_pattern_or_path}")
            logger.info(f"Found {len(files)} files matching pattern.")
            df_list = []
            for f in files:
                try:
                    repo_name_from_file = os.path.basename(f).replace('_analysis.csv', '').replace('_scored.csv', '')
                    df_temp = pd.read_csv(f, low_memory=False)
                    if 'repo' not in df_temp.columns:
                         df_temp['repo'] = repo_name_from_file
                         logger.debug(f"Added 'repo' column with value '{repo_name_from_file}' from filename {f}")
                    elif df_temp['repo'].isnull().any():
                         logger.warning(f"File {f} has 'repo' column but contains NaN values. Filling with name from file: '{repo_name_from_file}'")
                         df_temp['repo'].fillna(repo_name_from_file, inplace=True)
                    df_list.append(df_temp)
                    loaded_files.append(f)
                except pd.errors.EmptyDataError:
                    logger.warning(f"Skipping empty file: {f}")
                except Exception as e:
                    logger.warning(f"Could not read or process file {f}: {e}")
            if not df_list:
                 raise ValueError("No data loaded from any file in the pattern.")
            df = pd.concat(df_list, ignore_index=True)
        else: # Single file path
             # ... (single file handling logic remains the same) ...
            if not os.path.exists(file_pattern_or_path):
                 raise FileNotFoundError(f"Input file not found: {file_pattern_or_path}")
            df = pd.read_csv(file_pattern_or_path, low_memory=False)
            loaded_files.append(file_pattern_or_path)
            if 'repo' not in df.columns:
                 logger.warning(f"Single input file {file_pattern_or_path} lacks 'repo' column. Using placeholder 'unknown_repo'.")
                 df['repo'] = 'unknown_repo'

        logger.info(f"Initial data loaded from {len(loaded_files)} file(s): {df.shape[0]} rows, {df.shape[1]} columns")
        if df.empty:
             raise ValueError("Loaded dataframe is empty.")

        # --- Data Cleaning & Standardizing ---
        logger.debug(f"Original columns: {df.columns.tolist()}")
        df.columns = df.columns.str.strip().str.lower().str.replace('[^a-z0-9_]', '_', regex=True)
        logger.debug(f"Standardized columns: {df.columns.tolist()}")

        # --- Renaming columns (ensure idempotency) ---
        rename_map = {
            'pr_number': 'pr_id',
            'comments_count': 'comments',
            'repository': 'repo',
        }
        actual_renames = {k: v for k, v in rename_map.items() if k in df.columns and v not in df.columns}
        if actual_renames:
            df = df.rename(columns=actual_renames)
            logger.info(f"Renamed columns: {actual_renames}")

        # --- Essential Column Checks ---
        essential_cols = ['repo', 'created_at']
        if 'pr_id' not in df.columns: # Check for the renamed column
            essential_cols.append('pr_number') # Add original if renamed wasn't present
        else:
            essential_cols.append('pr_id')

        missing_essential = [col for col in essential_cols if col not in df.columns]
        if missing_essential:
             raise ValueError(f"Essential columns missing after standardization/rename: {missing_essential}. Cannot proceed.")

        # --- Score Column Identification ---
        score_cols_raw = sorted([c for c in df.columns if c.startswith('score_')])
        if not score_cols_raw:
             logger.warning("No columns starting with 'score_' found. Checking for 'rating_'...")
             score_cols_raw = sorted([c for c in df.columns if c.startswith('rating_')])
             if not score_cols_raw:
                  raise ValueError("No columns starting with 'score_' or 'rating_' found in the input data.")
             logger.info(f"Using potential score columns starting with 'rating_': {score_cols_raw}")
        else:
             logger.info(f"Using potential score columns starting with 'score_': {score_cols_raw}")

        # --- Type Conversions & Filtering ---
        # ... (created_at conversion and date filtering remain the same) ...
        logger.info("Converting 'created_at' to datetime (UTC)...")
        if 'created_at' not in df.columns:
             raise ValueError("Essential column 'created_at' missing.")
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce', utc=True)
        initial_rows = df.shape[0]
        df = df.dropna(subset=['created_at'])
        dropped_rows = initial_rows - df.shape[0]
        if dropped_rows > 0:
            logger.warning(f"Dropped {dropped_rows} rows due to invalid 'created_at' dates.")

        try:
            start_dt = pd.Timestamp(START_DATE, tz="UTC")
            end_dt = pd.Timestamp(END_DATE, tz="UTC")
            logger.info(f"Filtering data between {start_dt} and {end_dt}")
        except Exception as date_e:
            logger.error(f"Invalid START_DATE ('{START_DATE}') or END_DATE ('{END_DATE}'): {date_e}")
            raise ValueError("Invalid date format in configuration.") from date_e

        initial_rows = df.shape[0]
        df = df[(df['created_at'] >= start_dt) & (df['created_at'] <= end_dt)]
        rows_after_filter = df.shape[0]
        logger.info(f"Rows after applying date filter ({START_DATE} to {END_DATE}): {rows_after_filter} (Removed {initial_rows - rows_after_filter})")

        if df.empty:
            raise ValueError(f"No data remaining after date filtering ({START_DATE} to {END_DATE}). Check date range and input data.")

        # ... (score conversion remains the same) ...
        score_cols = []
        logger.info("Converting score columns to numeric...")
        for col in score_cols_raw:
             if col in df.columns:
                 original_non_na = df[col].notna().sum()
                 try:
                     df[col] = pd.to_numeric(df[col], errors='coerce')
                     converted_non_na = df[col].notna().sum()
                     nan_created = original_non_na - converted_non_na
                     if nan_created > 0:
                         logger.warning(f"Column '{col}': {nan_created} values failed numeric conversion (became NaN).")
                     if df[col].notna().any():
                        score_cols.append(col)
                        logger.debug(f"Successfully converted '{col}' to numeric. Valid values: {converted_non_na}")
                     else:
                        logger.warning(f"Column '{col}' has no valid numeric data after conversion. Excluding from score_cols.")
                 except Exception as conv_e:
                      logger.error(f"Error converting column '{col}' to numeric: {conv_e}. Skipping this column.")

        if not score_cols:
            raise ValueError("No valid score columns available for analysis.")


        # ... (boolean/categorical conversions remain the same) ...
        logger.info("Converting boolean/categorical columns...")
        merged_col_candidates = ['is_merged', 'merged']
        target_merged_col = 'is_merged'
        merged_col_found = False
        for col in merged_col_candidates:
             if col in df.columns:
                 logger.debug(f"Attempting conversion of '{col}' to boolean '{target_merged_col}'.")
                 try:
                     map_dict = {'true': True, 'false': False, 'True': True, 'False': False,
                                 '1': True, '0': False, 1: True, 0: False, 1.0: True, 0.0: False,
                                 'yes': True, 'no': False, 'y': True, 'n': False}
                     bool_series = df[col].copy()
                     if pd.api.types.is_string_dtype(bool_series) or pd.api.types.is_object_dtype(bool_series):
                         bool_series = bool_series.astype(str).str.strip().str.lower()
                     # Use map, but explicitly handle bools and numbers first
                     bool_map = {
                         True: True, False: False, 1: True, 0: False, 1.0: True, 0.0: False,
                         'true': True, 'false': False, 'yes': True, 'no': False, '1': True, '0': False, 'y': True, 'n': False
                     }
                     # Apply mapping intelligently
                     converted_series = bool_series.map(bool_map)
                     # For values not in the map, try coercing to numeric then map 1.0/0.0
                     numeric_coerced = pd.to_numeric(bool_series[converted_series.isna()], errors='coerce')
                     converted_series.update(numeric_coerced.map({1.0: True, 0.0: False}))

                     # Final conversion to nullable boolean
                     df[target_merged_col] = converted_series.astype('boolean')

                     original_non_na_bool = df[col].notna().sum()
                     final_non_na_bool = df[target_merged_col].notna().sum()
                     nan_introduced = original_non_na_bool - final_non_na_bool
                     if nan_introduced > 0:
                         logger.warning(f"Column '{col}' -> '{target_merged_col}': {nan_introduced} values could not be converted to True/False and became NaN.")

                     if col != target_merged_col and target_merged_col in df.columns:
                         df.drop(columns=[col], inplace=True, errors='ignore') # Use errors='ignore'
                         logger.debug(f"Dropped original column '{col}' after conversion to '{target_merged_col}'.")
                     logger.info(f"Converted column '{col}' to nullable boolean '{target_merged_col}'.")
                     merged_col_found = True
                     break
                 except Exception as e:
                      logger.warning(f"Could not robustly convert column '{col}' to boolean '{target_merged_col}': {e}.")
        if not merged_col_found:
             logger.warning(f"Could not find or convert a 'merged'/'is_merged' column to boolean '{target_merged_col}'.")

        if 'state' in df.columns:
             try:
                 df['state'] = df['state'].astype('category')
                 logger.info(f"Converted 'state' column to category type. Categories: {df['state'].cat.categories.tolist()}")
             except Exception as e:
                  logger.warning(f"Could not convert 'state' column to category: {e}")
        if 'repo' in df.columns:
             try:
                 df['repo'] = df['repo'].astype('category')
                 logger.info(f"Converted 'repo' column to category type. Number of categories: {df['repo'].nunique()}")
             except Exception as e:
                 logger.error(f"Could not convert 'repo' column to category: {e}")


        nan_counts = df[score_cols].isnull().sum()
        if nan_counts.sum() > 0:
            logger.warning(f"NaN values present in final score columns:\n{nan_counts[nan_counts > 0].to_string()}")
        else:
             logger.info("No NaN values found in the final score columns.")

        # --- Feature Engineering ---
        # ... (date feature engineering remains the same) ...
        logger.info("Engineering date features (year, month, quarter, year_month)...")
        try:
            df['year'] = df['created_at'].dt.year
            df['month'] = df['created_at'].dt.month
            df['quarter'] = df['created_at'].dt.to_period('Q').astype(str)
            df['year_month'] = df['created_at'].dt.to_period('M').astype(str)
        except Exception as e:
             logger.error(f"Failed to engineer date features: {e}")

        logger.info("Preprocessing complete.")
        # ... (logging final shape etc. remains the same) ...
        final_shape = df.shape
        logger.info(f"Final dataset shape: {final_shape}")
        try:
            date_min = df['created_at'].min()
            date_max = df['created_at'].max()
            logger.info(f"Date range in final data: {date_min} to {date_max}")
        except Exception as e:
             logger.warning(f"Could not determine final date range: {e}")
        logger.info(f"Repositories in final data: {df['repo'].nunique()}")
        logger.info(f"Final valid score columns for analysis: {score_cols}")


        # --- Save the fully preprocessed data --- THIS IS THE KEY FILE
        cleaned_output_path = os.path.join(OUTPUT_DIR, 'preprocessed_scored_prs.csv')
        try:
            df.to_csv(cleaned_output_path, index=False, date_format='%Y-%m-%d %H:%M:%S%z') # Ensure consistent date format
            logger.info(f"Preprocessed data saved to: {cleaned_output_path}")
        except Exception as e:
             logger.error(f"Failed to save preprocessed data to CSV: {e}")

        return df, score_cols

    except FileNotFoundError as fnf_err:
        logging.error(f"Input file/pattern error: {fnf_err}", exc_info=True)
        return pd.DataFrame(), []
    except ValueError as ve:
         logging.error(f"Data validation or processing error: {ve}", exc_info=True)
         return pd.DataFrame(), []
    except Exception as e:
        logging.error(f"Unexpected error during data loading/preprocessing: {e}", exc_info=True)
        return pd.DataFrame(), []


# --- Log Formatter Class ---
class LogFormatterCommaNoSci(mticker.LogFormatterSciNotation): # Inherit from SciNotation for base logic
    """Log formatter that uses commas and avoids scientific notation where possible."""
    def __call__(self, x, pos=None):
        try:
            # Check if number is large enough or small enough to likely trigger sci notation
            if x >= 1e4 or x <= 1e-3: # Adjust thresholds as needed
                 # Use standard formatting with commas for large numbers
                 if x >= 1:
                     return format(int(x), ',')
                 else:
                     # For small numbers, default LogFormatter is usually okay, but try fixed point
                     return f"{x:.2g}" # Use general format with limited precision
            # For numbers in the 'middle' range, format with commas
            elif x == float(int(x)): # Handle integers
                 return format(int(x), ',')
            else: # Handle floats
                 return f"{x:,.1f}" # Adjust precision (e.g., .2f)
        except (ValueError, TypeError):
            # Fallback to default LogFormatter if conversion fails
            return super().__call__(x, pos)


# --- Analysis Functions ---

def perform_descriptive_analysis(df, score_cols, pdf_pages):
    """Calculate and visualize descriptive statistics."""
    logger = logging.getLogger(__name__)
    logger.info("Performing Descriptive Analysis...")
    if df.empty or not score_cols:
        logger.warning("Skipping descriptive analysis: No data or score columns.")
        return

    # 1. Overall Score Distributions (based on df passed in, which is the full df_main)
    logger.info("Generating overall score distributions (bundled and individual)...")
    try:
        desc_stats = df[score_cols].describe().transpose()
        desc_stats_path = os.path.join(OUTPUT_DIR, 'overall_score_descriptives.csv')
        desc_stats.to_csv(desc_stats_path)
        logger.info(f"Overall descriptive statistics saved to {desc_stats_path}")
    except Exception as e:
        logger.error(f"Failed to calculate/save descriptive stats: {e}")

    # --- Define Font Sizes for Bundled Histogram Plot ---
    hist_suptitle_fontsize = 24
    hist_subplot_title_fontsize = 22
    hist_axis_label_fontsize = 18
    hist_tick_label_fontsize = 16

    # Bundled Histogram Plot (Larger Fonts)
    try:
        num_scores = len(score_cols)
        ncols = 2
        nrows = (num_scores + ncols - 1) // ncols
        # Adjust figsize slightly if needed for larger fonts
        fig_bundle, axes_bundle = plt.subplots(nrows, ncols, figsize=(ncols * 8, nrows * 5.5), squeeze=False)
        axes_bundle = axes_bundle.flatten()
        plotted_count_bundle = 0

        for i, col in enumerate(score_cols):
            if i >= len(axes_bundle): break
            ax = axes_bundle[i]
            plot_title = f"Distribution of {col.replace('score_', '').replace('_', ' ').title()}"
            if df[col].notna().any():
                try:
                    # Use sns.histplot which automatically handles density/count on y-axis
                    sns.histplot(df[col].dropna(), kde=True, ax=ax, bins=30)
                    ax.set_title(plot_title, fontsize=hist_subplot_title_fontsize)
                    ax.set_xlabel("Score", fontsize=hist_axis_label_fontsize)
                    # Get the default ylabel set by histplot and apply fontsize
                    current_ylabel = ax.get_ylabel()
                    ax.set_ylabel(current_ylabel, fontsize=hist_axis_label_fontsize)
                    # Set tick label sizes
                    ax.tick_params(axis='both', which='major', labelsize=hist_tick_label_fontsize)
                    plotted_count_bundle += 1
                except Exception as hist_e:
                    logger.error(f"Error plotting histogram for {col}: {hist_e}")
                    ax.set_title(f"{plot_title} (Plot Error)", fontsize=hist_subplot_title_fontsize)
                    ax.text(0.5, 0.5, 'Plot Error', ha='center', va='center', transform=ax.transAxes)
                    # Still set label/tick sizes for consistency on error plots
                    ax.set_xlabel("Score", fontsize=hist_axis_label_fontsize)
                    ax.set_ylabel("Frequency", fontsize=hist_axis_label_fontsize) # Default ylabel
                    ax.tick_params(axis='both', which='major', labelsize=hist_tick_label_fontsize)

                # Individual Plot (Keep existing logic, font sizes not modified here unless desired)
                try:
                    fig_single, ax_single = plt.subplots(figsize=(10, 6))
                    sns.histplot(df[col].dropna(), kde=True, ax=ax_single, bins=30)
                    # You could apply larger fonts here too if needed, following the pattern above
                    ax_single.set_title(plot_title) # Default font size unless specified
                    ax_single.set_xlabel("Score")
                    plt.tight_layout()
                    save_plot(fig_single, f"overall_score_histogram_{col}",
                              subdir=PLOT_SUBDIR_DESC, is_single_component=True)
                except Exception as e_single:
                    logger.error(f"Failed to generate single histogram for {col}: {e_single}")
                    if 'fig_single' in locals() and plt.fignum_exists(fig_single.number): plt.close(fig_single)
            else:
                logger.warning(f"No valid data to plot for score: {col}")
                ax.set_title(f"{plot_title} (No Data)", fontsize=hist_subplot_title_fontsize)
                ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
                ax.set_xlabel("Score", fontsize=hist_axis_label_fontsize)
                ax.set_ylabel("Frequency", fontsize=hist_axis_label_fontsize) # Default ylabel
                ax.tick_params(axis='both', which='major', labelsize=hist_tick_label_fontsize)

        for j in range(plotted_count_bundle, len(axes_bundle)):
            axes_bundle[j].set_visible(False)

        if plotted_count_bundle > 0:
            fig_bundle.suptitle("Overall Score Distributions", fontsize=hist_suptitle_fontsize, y=1.03) # Adjust y if needed
            # Use tight_layout with padding for suptitle
            plt.tight_layout(rect=[0, 0, 1, 0.97]) # Adjust top padding (0.97) if suptitle overlaps
            save_plot(fig_bundle, "overall_score_histograms_bundled", pdf_pages=pdf_pages,
                      subdir=PLOT_SUBDIR_DESC)
        else:
            logger.warning("No scores had data to plot in the bundled histogram.")
            if 'fig_bundle' in locals() and plt.fignum_exists(fig_bundle.number): plt.close(fig_bundle)

    except Exception as e:
        logger.error(f"Failed to generate score histograms (bundling): {e}")
        if 'fig_bundle' in locals() and plt.fignum_exists(fig_bundle.number): plt.close(fig_bundle)


    # 2. Temporal Distribution of PRs
    # ... (remains the same, uses df directly) ...
    logger.info("Analyzing PR activity over time...")
    try:
        fig_ts, ax_ts = plt.subplots(figsize=(16, 6))
        # Use pr_id if available, otherwise fallback (needs to exist check)
        id_col = 'pr_id' if 'pr_id' in df.columns else ('pr_number' if 'pr_number' in df.columns else None)
        if id_col and pd.api.types.is_datetime64_any_dtype(df['created_at']):
            df_time_indexed = df.set_index('created_at')
            pr_counts = df_time_indexed.resample('M')[id_col].count()
            if not pr_counts.empty and pr_counts.sum() > 0:
                pr_counts.plot(ax=ax_ts)
                ax_ts.set_title("Pull Request Activity Over Time (Monthly)")
                ax_ts.set_ylabel("Number of PRs")
                ax_ts.set_xlabel("Date")
                ax_ts.set_ylim(bottom=0)
                ax_ts.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: format(int(x), ',')))
                save_plot(fig_ts, "pr_activity_monthly", pdf_pages=pdf_pages, subdir=PLOT_SUBDIR_DESC)
            else:
                 logger.warning("No PR counts found after resampling monthly.")
                 ax_ts.set_title("Pull Request Activity Over Time (Monthly) - No Data")
                 ax_ts.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax_ts.transAxes)
                 save_plot(fig_ts, "pr_activity_monthly_nodata", pdf_pages=pdf_pages, subdir=PLOT_SUBDIR_DESC)
        elif not id_col:
            logger.error("Neither 'pr_id' nor 'pr_number' column found. Skipping temporal plot.")
            plt.close(fig_ts)
        else: # 'created_at' is not datetime
            logger.error("'created_at' column not datetime. Skipping temporal plot.")
            plt.close(fig_ts)
    except Exception as e:
        logger.error(f"Failed to generate PR activity plot: {e}")
        if 'fig_ts' in locals() and plt.fignum_exists(fig_ts.number): plt.close(fig_ts)

    # 3. PR Count per Repository
    # ... (previous code for calculating repo_counts and saving CSV) ...
    logger.info("Analyzing PR counts per repository...")
    try:
        if 'repo' not in df.columns:
            logger.error("Missing 'repo' column. Cannot analyze PR counts per repository.")
            return

        repo_counts = df['repo'].value_counts() # Sorted descending
        repo_counts_path = os.path.join(OUTPUT_DIR, 'repo_pr_counts.csv')
        repo_counts.to_csv(repo_counts_path, header=True)
        logger.info(f"Repo PR counts saved to {repo_counts_path}")

        # Filter for repositories with at least one PR
        repo_counts_positive = repo_counts[repo_counts > 0]

        if not repo_counts_positive.empty:
             num_repos_to_plot = len(repo_counts_positive) # Plot all positive counts
             logger.info(f"Found {num_repos_to_plot} repositories with > 0 PRs.")
             logger.info(f"Repository counts range (positive only): {repo_counts_positive.min()} to {repo_counts_positive.max()}")

             # --- Set Fixed Font Sizes ---
             fixed_xtick_fontsize = 16 # Font size for the repo names (tick labels)
             axis_label_fontsize = 18   # Font size for "Repository" and "Number of PRs"
             title_fontsize = 20        # Font size for the plot title
             logger.info(f"Plotting {num_repos_to_plot} repositories. X-tick font: {fixed_xtick_fontsize}, Axis label font: {axis_label_fontsize}, Title font: {title_fontsize}.")
             if num_repos_to_plot > 50: # Add warning if potentially crowded
                logger.warning(f"Plotting many ({num_repos_to_plot}) repositories. Labels may overlap significantly.")

             # --- Linear Scale Plot (All Positive Counts, Explicit Order, All Labels, Larger Fonts, Adjust Left Margin) ---
             try:
                 # Dynamic width
                 fig_lin, ax_lin = plt.subplots(figsize=(max(14, num_repos_to_plot*0.25), 8))
                 sns.barplot(x=repo_counts_positive.index,
                             y=repo_counts_positive.values,
                             order=repo_counts_positive.index, # Explicit order
                             palette="viridis",
                             ax=ax_lin)

                 # Set tick labels with specific font size
                 ax_lin.set_xticklabels(ax_lin.get_xticklabels(), rotation=90, fontsize=fixed_xtick_fontsize)

                 # Set axis labels with specific font size
                 ax_lin.set_xlabel("Repository", fontsize=axis_label_fontsize)
                 ax_lin.set_ylabel("Number of PRs Analyzed", fontsize=axis_label_fontsize)

                 # Set Title with specific font size
                 ax_lin.set_title(
                     f"PR Count per Repository (All {num_repos_to_plot} Repos with > 0 PRs - Linear Scale, Sorted by Count)",
                     fontsize=title_fontsize
                 )

                 ax_lin.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: format(int(x), ',')))
                 # Increase font size for y-axis tick labels too
                 ax_lin.tick_params(axis='y', labelsize=fixed_xtick_fontsize)

                 # Apply tight_layout first to handle general spacing
                 plt.tight_layout()
                 # --- Manually adjust left margin to give more space for Y axis ---
                 fig_lin.subplots_adjust(left=0.15) # Increase left margin (try values like 0.15, 0.18, etc.)

                 save_plot(fig_lin, "repo_pr_counts_barplot_linear", pdf_pages=pdf_pages, subdir=PLOT_SUBDIR_DESC)
             except Exception as e_lin:
                 logger.error(f"Failed to generate linear scale repo count plot: {e_lin}")
                 if 'fig_lin' in locals() and plt.fignum_exists(fig_lin.number): plt.close(fig_lin)

             # --- Logarithmic Scale Plot (All Positive Counts, Explicit Order, All Labels, Larger Fonts, Adjust Left Margin) ---
             try:
                 # Dynamic width
                 fig_log, ax_log = plt.subplots(figsize=(max(14, num_repos_to_plot*0.25), 8))
                 logger.debug(f"Plotting log scale bar chart for all {num_repos_to_plot} repos with > 0 counts (sorted).")
                 sns.barplot(x=repo_counts_positive.index,
                             y=repo_counts_positive.values,
                             order=repo_counts_positive.index, # Explicit order
                             palette="viridis",
                             ax=ax_log)
                 ax_log.set_yscale('log')

                 # Set tick labels with specific font size
                 ax_log.set_xticklabels(ax_log.get_xticklabels(), rotation=90, fontsize=fixed_xtick_fontsize)

                 # Set axis labels with specific font size
                 ax_log.set_xlabel("Repository", fontsize=axis_label_fontsize)
                 ax_log.set_ylabel("Number of PRs Analyzed (Log Scale)", fontsize=axis_label_fontsize)

                 # Set Title with specific font size
                 ax_log.set_title(
                     f"PR Count per Repository (All {num_repos_to_plot} Repos with > 0 PRs - Log Scale, Sorted by Count)",
                     fontsize=title_fontsize
                 )

                 ax_log.yaxis.set_major_formatter(LogFormatterCommaNoSci())
                 # Increase font size for y-axis tick labels too
                 ax_log.tick_params(axis='y', labelsize=fixed_xtick_fontsize)

                 # Apply tight_layout first
                 plt.tight_layout()
                 # --- Manually adjust left margin to give more space for Y axis ---
                 fig_log.subplots_adjust(left=0.15) # Increase left margin (use same value or adjust if needed)

                 save_plot(fig_log, "repo_pr_counts_barplot_log_scale", pdf_pages=pdf_pages, subdir=PLOT_SUBDIR_DESC)
             except Exception as e_log:
                 logger.error(f"Failed to generate log scale repo count plot: {e_log}", exc_info=True)
                 if 'fig_log' in locals() and plt.fignum_exists(fig_log.number): plt.close(fig_log)
        else:
             logger.warning("No repositories found with > 0 PRs. Skipping count plots.")
    except Exception as e:
        logger.error(f"Failed during PR count per repository analysis: {e}")

# --- Repository Comparison ---
# ... (perform_repository_comparison remains the same as it filters df internally) ...
def perform_repository_comparison(df, score_cols, pdf_pages):
    """Compare scores across different repositories using boxplots and Kruskal-Wallis."""
    logger = logging.getLogger(__name__)
    logger.info("Performing Repository Comparison Analysis...")
    repo_summary_filtered = pd.DataFrame() # Initialize return dataframe

    if df.empty or not score_cols:
        logger.warning("Skipping repository comparison: No data or score columns.")
        return repo_summary_filtered
    if 'repo' not in df.columns:
        logger.error("Skipping repository comparison: Missing 'repo' column.")
        return repo_summary_filtered

    repo_counts = df['repo'].value_counts()
    repos_to_analyze = repo_counts[repo_counts >= MIN_PRS_PER_REPO_FOR_STATS].index
    df_filtered = df[df['repo'].isin(repos_to_analyze)].copy()

    # Calculate summary stats for ALL repositories (saved for reference)
    try:
        if not df.empty:
            # Use agg for efficiency
            agg_funcs = ['mean', 'std']
            repo_summary_stats = df.groupby('repo')[score_cols].agg(agg_funcs)
            # Flatten MultiIndex columns
            repo_summary_stats.columns = ['_'.join(col).strip() for col in repo_summary_stats.columns.values]
            # Rename columns for clarity
            rename_map_stats = {f'{score}_mean': f'Mean_{score}' for score in score_cols}
            rename_map_stats.update({f'{score}_std': f'StdDev_{score}' for score in score_cols})
            repo_summary_stats = repo_summary_stats.rename(columns=rename_map_stats)

            repo_counts_all = df['repo'].value_counts().rename('pr_count')
            repo_summary_all = pd.DataFrame(index=repo_summary_stats.index)
            repo_summary_all['pr_count'] = repo_counts_all.reindex(repo_summary_stats.index)
            repo_summary_all = repo_summary_all.join(repo_summary_stats)

            repo_summary_all_path = os.path.join(OUTPUT_DIR, 'repository_mean_std_summary_all.csv')
            repo_summary_all.to_csv(repo_summary_all_path)
            logger.info(f"Summary for ALL repositories saved to {repo_summary_all_path}")
        else:
            logger.warning("Input dataframe 'df' was empty, cannot calculate overall repo summary.")
    except Exception as e_sum_all:
        logger.error(f"Failed to calculate or save summary for ALL repositories: {e_sum_all}")

    # Proceed with FILTERED data
    if not df_filtered.empty:
        # Ensure 'repo' column is category and remove unused categories
        if pd.api.types.is_categorical_dtype(df_filtered['repo']):
            df_filtered['repo'] = df_filtered['repo'].cat.remove_unused_categories()
        else:
             df_filtered['repo'] = df_filtered['repo'].astype('category')

        num_filtered_repos = df_filtered['repo'].nunique()
        logger.info(f"Analyzing {num_filtered_repos} repositories with >= {MIN_PRS_PER_REPO_FOR_STATS} PRs.")

        # Calculate summary stats for FILTERED repos (this is the primary summary used later)
        try:
            agg_funcs_filtered = ['mean', 'std']
            repo_summary_stats_filtered = df_filtered.groupby('repo')[score_cols].agg(agg_funcs_filtered)
            repo_summary_stats_filtered.columns = ['_'.join(col).strip() for col in repo_summary_stats_filtered.columns.values]
            rename_map_stats_filtered = {f'{score}_mean': f'Mean_{score}' for score in score_cols}
            rename_map_stats_filtered.update({f'{score}_std': f'StdDev_{score}' for score in score_cols})
            repo_summary_stats_filtered = repo_summary_stats_filtered.rename(columns=rename_map_stats_filtered)

            repo_counts_filtered = df_filtered['repo'].value_counts().rename('pr_count')
            repo_summary_filtered = pd.DataFrame(index=repo_summary_stats_filtered.index)
            repo_summary_filtered['pr_count'] = repo_counts_filtered.reindex(repo_summary_stats_filtered.index)
            repo_summary_filtered = repo_summary_filtered.join(repo_summary_stats_filtered)

            repo_summary_filtered_path = os.path.join(OUTPUT_DIR, 'repository_mean_std_summary_filtered.csv')
            repo_summary_filtered.to_csv(repo_summary_filtered_path)
            logger.info(f"Summary for {len(repo_summary_filtered)} FILTERED repositories saved to {repo_summary_filtered_path}")
        except Exception as e_sum_filt:
            logger.error(f"Failed to calculate or save summary for FILTERED repositories: {e_sum_filt}")
            logger.critical("Cannot proceed reliably with repository comparison/clustering due to error in filtered summary creation.")
            return pd.DataFrame() # Return empty

        if num_filtered_repos < 2:
            logger.warning(f"Only {num_filtered_repos} repository meets the min PR count. Skipping comparison plots & Kruskal-Wallis.")
            return repo_summary_filtered

        # --- Boxplots ---
        logger.info("Generating score distributions by repository (bundled and individual)...")
        try:
            num_scores = len(score_cols)
            ncols = 2
            nrows = (num_scores + ncols - 1) // ncols
            # Adjust figsize slightly if needed for larger fonts
            fig_bundle, axes_bundle = plt.subplots(nrows, ncols, figsize=(ncols * 11, nrows * 8), sharey=False, squeeze=False)
            axes_bundle = axes_bundle.flatten()
            plotted_count_bundle = 0

            # --- Define Font Sizes ---
            tick_fontsize = 16        # For repo names and score values
            axis_label_fontsize = 18  # For "Repository" / "Score"
            plot_title_fontsize = 26  # For individual subplot titles
            super_title_fontsize = 22 # For main figure title

            # Pre-calculate orderings to avoid repeated groupby
            median_orders = {}
            for score in score_cols:
                if df_filtered[score].notna().any():
                    try:
                         median_orders[score] = df_filtered.groupby('repo')[score].median().sort_values(ascending=False).index
                    except Exception as order_e:
                        logger.error(f"Failed to get median order for {score}: {order_e}. Ordering will be alphabetical.")
                        median_orders[score] = sorted(df_filtered['repo'].unique()) # Fallback to alphabetical

            for i, score in enumerate(score_cols):
                if i >= len(axes_bundle): break
                ax = axes_bundle[i]
                plot_title_text = f"{score.replace('score_', '').replace('_', ' ').title()} by Repository"
                order = median_orders.get(score)
                num_repos_in_order = len(order) if order is not None else 0

                if df_filtered[score].notna().any() and order is not None and num_repos_in_order > 0:
                     # Bundled Plot
                     try:
                         sns.boxplot(data=df_filtered, x='repo', y=score, order=order, ax=ax, palette="coolwarm", showfliers=False)
                         ax.set_title(plot_title_text, fontsize=plot_title_fontsize) # Set title font size

                         # Determine label visibility and set X labels/ticks
                         LABEL_VISIBILITY_THRESHOLD_COMP = 60 # Threshold for comparison plot
                         show_labels = num_repos_in_order <= LABEL_VISIBILITY_THRESHOLD_COMP
                         if show_labels:
                             # Use defined tick_fontsize
                             ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=tick_fontsize)
                             ax.set_xlabel("Repository", fontsize=axis_label_fontsize) # Set x-label font size
                         else:
                             ax.set_xticks([]) # Hide ticks
                             ax.set_xlabel(f"{num_repos_in_order} Repos (Labels Hidden)", fontsize=axis_label_fontsize) # Set x-label font size

                         ax.set_ylabel("Score", fontsize=axis_label_fontsize) # Set y-label font size
                         # Set y-axis tick label font size
                         ax.tick_params(axis='y', labelsize=tick_fontsize)

                         plotted_count_bundle += 1
                     except Exception as e_bundle_box:
                         logger.error(f"Failed bundled boxplot for {score}: {e_bundle_box}")
                         ax.set_title(f"{plot_title_text} (Plot Error)", fontsize=plot_title_fontsize) # Show error with correct font size

                     # Individual Plot (Keep font sizes consistent if desired, or leave as default)
                     # You might want to apply similar fontsize settings here if needed.
                     # For brevity, I'm leaving the individual plot font sizes as default for now.
                     try:
                         fig_single, ax_single = plt.subplots(figsize=(max(12, num_repos_in_order*0.25), 7))
                         sns.boxplot(data=df_filtered, x='repo', y=score, order=order, ax=ax_single, palette="coolwarm", showfliers=False)
                         ax_single.set_title(plot_title_text) # Default font size here
                         xtick_fontsize_single = max(6, min(10, 300 // num_repos_in_order))
                         ax_single.set_xticklabels(ax_single.get_xticklabels(), rotation=90, fontsize=xtick_fontsize_single)
                         ax_single.set_xlabel("Repository") # Default font size here
                         ax_single.set_ylabel("Score") # Default font size here
                         plt.tight_layout()
                         save_plot(fig_single, f"repo_score_distribution_{score}",
                                   subdir=PLOT_SUBDIR_COMP_SINGLE, is_single_component=True)
                     except Exception as e_single_box:
                         logger.error(f"Failed single boxplot for {score}: {e_single_box}")
                         if 'fig_single' in locals() and plt.fignum_exists(fig_single.number): plt.close(fig_single)
                elif order is None:
                     logger.error(f"Could not determine order for {score}. Skipping plots.")
                     ax.set_title(f"{plot_title_text} (Order Error)", fontsize=plot_title_fontsize) # Apply font size
                     ax.text(0.5, 0.5, 'Order Error', ha='center', va='center', transform=ax.transAxes)
                else:
                     logger.warning(f"No valid data for score '{score}' in filtered repositories.")
                     ax.set_title(f"{plot_title_text} (No Data)", fontsize=plot_title_fontsize) # Apply font size
                     ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)

            for j in range(plotted_count_bundle, len(axes_bundle)):
                axes_bundle[j].set_visible(False)

            if plotted_count_bundle > 0:
                # Set super title font size
                fig_bundle.suptitle("Score Distributions by Repository (Filtered)", fontsize=super_title_fontsize, y=1.02)
                # Adjust tight_layout rect to accommodate larger super title if necessary
                plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjusted top slightly
                save_plot(fig_bundle, "repo_score_distributions_bundled", pdf_pages=pdf_pages,
                          subdir=PLOT_SUBDIR_COMP_BUNDLE)
            else:
                logger.warning("No scores had data for bundled repository distribution plot.")
                if 'fig_bundle' in locals() and plt.fignum_exists(fig_bundle.number): plt.close(fig_bundle)

        except Exception as e:
            logger.error(f"Failed generating repository score plots (outer loop): {e}", exc_info=True)
            if 'fig_bundle' in locals() and plt.fignum_exists(fig_bundle.number): plt.close(fig_bundle)

        # --- Kruskal Wallis ---
        # ... (KW logic remains the same) ...
        logger.info("Running Kruskal-Wallis tests for repository differences...")
        kw_results = []
        for score in score_cols:
            stat, p_val = run_kruskal_wallis(df_filtered, 'repo', score) # Use filtered df
            kw_results.append({
                'Competency': score,
                'Kruskal_Statistic': stat,
                'P_Value': p_val
            })
        kw_df = pd.DataFrame(kw_results).set_index('Competency')
        valid_pvals = kw_df['P_Value'].dropna()
        if not valid_pvals.empty:
            try:
                reject, pvals_corrected, _, _ = multipletests(valid_pvals, alpha=SIGNIFICANCE_LEVEL, method='fdr_bh')
                kw_df.loc[valid_pvals.index, 'P_Value_Corrected'] = pvals_corrected
                kw_df.loc[valid_pvals.index, 'Significant_After_Correction'] = reject
                kw_df['Significant_After_Correction'] = kw_df['Significant_After_Correction'].fillna(False)
            except Exception as multi_e:
                logger.error(f"Failed multiple testing correction: {multi_e}")
                kw_df['P_Value_Corrected'] = np.nan
                kw_df['Significant_After_Correction'] = False
        else:
            logger.info("No valid P-values for multiple testing correction.")
            kw_df['P_Value_Corrected'] = np.nan
            kw_df['Significant_After_Correction'] = False

        kw_results_path = os.path.join(OUTPUT_DIR, 'repo_kruskal_wallis_results.csv')
        kw_df.to_csv(kw_results_path)
        logger.info(f"Kruskal-Wallis results saved to {kw_results_path}")
        logger.info(f"\nKruskal-Wallis Summary (Filtered Repos):\n{kw_df.to_string(float_format='%.4g')}")

        return repo_summary_filtered # Return the successful filtered summary
    else:
        logger.warning(f"No repositories meet min PR count ({MIN_PRS_PER_REPO_FOR_STATS}). Comparison plots & Kruskal-Wallis skipped.")
        return pd.DataFrame()


# perform_correlation_analysis remains the same, operates on repo_summary
def perform_correlation_analysis(repo_summary, pdf_pages):
    """Analyze correlations between mean competency scores across repositories."""
    logger = logging.getLogger(__name__)
    logger.info("Performing Correlation Analysis between Repo Mean Competencies...")
    if repo_summary is None or repo_summary.empty or len(repo_summary) < 2:
        logger.warning(f"Skipping repo-level correlation: Insufficient repo data (need >= 2).")
        return

    mean_score_cols = [col for col in repo_summary.columns if col.startswith('Mean_score_')]
    if len(mean_score_cols) < 2 :
         logger.warning(f"Skipping repo-level correlation: Need >= 2 mean score columns.")
         return

    repo_means = repo_summary[mean_score_cols].copy()
    repo_means.columns = [c.replace('Mean_', '') for c in mean_score_cols] # Keep score_ prefix

    initial_repos = len(repo_means)
    repo_means.dropna(axis=0, how='any', inplace=True)
    if len(repo_means) < initial_repos:
        logger.warning(f"Dropped {initial_repos - len(repo_means)} repos with NaN means before repo-level correlation.")

    if repo_means.shape[0] < 2 or repo_means.shape[1] < 2:
         logger.warning(f"Skipping repo-level correlation: Insufficient data after NaNs (Repos: {repo_means.shape[0]}, Scores: {repo_means.shape[1]})")
         return

    logger.info(f"Calculating repo-level correlations on {repo_means.shape[0]} repos.")
    fig = None
    try:
        corr_matrix = repo_means.corr(method='spearman') # Use original score names
        corr_matrix_path = os.path.join(OUTPUT_DIR, 'competency_correlation_matrix.csv')
        corr_matrix.to_csv(corr_matrix_path)
        logger.info(f"Repo-level competency correlation matrix saved to {corr_matrix_path}")

        # Plotting
        fig, ax = plt.subplots(figsize=(max(8, corr_matrix.shape[1]*0.9), max(6, corr_matrix.shape[1]*0.8)))
        # Use title-case names for plot
        plot_corr_matrix = corr_matrix.copy()
        plot_corr_matrix.columns = [c.replace('score_', '').replace('_', ' ').title() for c in plot_corr_matrix.columns]
        plot_corr_matrix.index = [c.replace('score_', '').replace('_', ' ').title() for c in plot_corr_matrix.index]
        sns.heatmap(plot_corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, center=0, ax=ax, vmin=-1, vmax=1, annot_kws={"size": 9})
        ax.set_title("Spearman Correlation Between Mean Competency Scores Across Repositories")
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()
        save_plot(fig, "competency_correlation_heatmap", pdf_pages=pdf_pages, subdir=PLOT_SUBDIR_CORR)

    except Exception as e:
        logger.error(f"Failed repo-level correlation analysis: {e}", exc_info=True)
        if fig is not None and plt.fignum_exists(fig.number):
             plt.close(fig)


# perform_pca_and_clustering remains the same, operates on repo_summary
def perform_pca_and_clustering(repo_summary, pdf_pages):
    """Perform PCA and Clustering on repository mean scores, using binned size."""
    logger = logging.getLogger(__name__)
    logger.info("Performing PCA and Clustering Analysis on Repository Means...")
    min_repos_for_pca = max(3, CLUSTER_COUNT, PCA_SIZE_BINS)
    repo_summary_pca = pd.DataFrame() # Initialize return df

    if repo_summary is None or repo_summary.empty:
        logger.warning("Skipping PCA/Clustering: repo_summary is empty.")
        return repo_summary_pca # Return empty dataframe

    if len(repo_summary) < min_repos_for_pca:
        logger.warning(f"Skipping PCA/Clustering: Insufficient repo data ({len(repo_summary)} < {min_repos_for_pca}).")
        return repo_summary_pca

    mean_score_cols = [col for col in repo_summary.columns if col.startswith('Mean_score_')]
    if len(mean_score_cols) < 2:
         logger.warning(f"Skipping PCA/Clustering: Need >= 2 mean score columns.")
         return repo_summary_pca

    if 'pr_count' not in repo_summary.columns:
         logger.error("Skipping PCA/Clustering: 'pr_count' column missing.")
         return repo_summary_pca

    analysis_cols = mean_score_cols + ['pr_count']
    repo_analysis_data = repo_summary[analysis_cols].copy()
    initial_rows = len(repo_analysis_data)
    # Drop NaNs in mean scores or invalid pr_count
    repo_analysis_data.dropna(subset=mean_score_cols, inplace=True)
    repo_analysis_data.dropna(subset=['pr_count'], inplace=True)
    repo_analysis_data = repo_analysis_data[repo_analysis_data['pr_count'] > 0]
    rows_after_na = len(repo_analysis_data)
    if rows_after_na < initial_rows:
        logger.warning(f"Dropped {initial_rows - rows_after_na} repos with NaN means or invalid PR counts before PCA.")
    if rows_after_na < min_repos_for_pca:
        logger.warning(f"Skipping PCA/Clustering: Insufficient repos ({rows_after_na} < {min_repos_for_pca}) after dropping NaNs.")
        return repo_summary_pca

    # --- Bin PR Counts for Size ---
    bin_labels = []
    try:
        pr_counts = repo_analysis_data['pr_count']
        # Use qcut, duplicates='drop'. Ensure integer labels if labels=False
        repo_analysis_data['pr_size_category_int'], bin_edges = pd.qcut(pr_counts, q=PCA_SIZE_BINS, labels=False, retbins=True, duplicates='drop')
        num_actual_bins = repo_analysis_data['pr_size_category_int'].nunique()

        descriptive_labels = []
        for i in range(len(bin_edges) - 1):
             lower = int(np.floor(bin_edges[i]))
             upper = int(np.ceil(bin_edges[i+1]))
             # Adjust lower bound for the first bin to match actual min
             if i == 0: lower = int(pr_counts.min())
             descriptive_labels.append(f'{lower:,} - {upper:,} PRs')

        # Check if number of labels matches number of unique bins created by qcut
        if len(descriptive_labels) >= num_actual_bins:
             # Create mapping from integer bin (0, 1, ...) to descriptive label
             # Need to map based on sorted unique integer bins present in the data
             unique_bins_sorted = sorted(repo_analysis_data['pr_size_category_int'].unique())
             # Ensure we have enough descriptive labels for the unique bins found
             labels_to_use = descriptive_labels[:num_actual_bins]
             bin_mapping = {int_bin: label for int_bin, label in zip(unique_bins_sorted, labels_to_use)}
             repo_analysis_data['pr_size_category'] = repo_analysis_data['pr_size_category_int'].map(bin_mapping)
             bin_labels = [bin_mapping[b] for b in unique_bins_sorted] # Get labels in correct order
             logger.info(f"Binned PR counts into {num_actual_bins} size categories: {bin_labels}")
        else: # Fallback if qcut dropped bins significantly or label generation mismatch
             repo_analysis_data['pr_size_category'] = repo_analysis_data['pr_size_category_int'].astype(str) # Use integer bin number as string label
             bin_labels = sorted(repo_analysis_data['pr_size_category'].unique())
             logger.warning(f"Could not create descriptive labels matching {num_actual_bins} bins. Using bin numbers: {bin_labels}")

        repo_analysis_data['pr_size_category'] = repo_analysis_data['pr_size_category'].astype('category')
        # Order categories based on bin_labels
        repo_analysis_data['pr_size_category'] = repo_analysis_data['pr_size_category'].cat.set_categories(bin_labels, ordered=True)
        size_categories = repo_analysis_data['pr_size_category']

    except ValueError as ve:
         # Handle case where PCA_SIZE_BINS is too high for unique counts
         logger.warning(f"Failed to bin PR counts (likely too few unique counts for {PCA_SIZE_BINS} bins): {ve}. Using single category.")
         repo_analysis_data['pr_size_category'] = 'All Repos'
         repo_analysis_data['pr_size_category'] = repo_analysis_data['pr_size_category'].astype('category')
         size_categories = repo_analysis_data['pr_size_category']
         bin_labels = ['All Repos']
    except Exception as bin_e:
        logger.error(f"Failed to bin PR counts for PCA sizing: {bin_e}. Proceeding without size encoding.")
        repo_analysis_data['pr_size_category'] = 'DefaultSize'
        repo_analysis_data['pr_size_category'] = repo_analysis_data['pr_size_category'].astype('category')
        size_categories = repo_analysis_data['pr_size_category']
        bin_labels = ['DefaultSize']

    # Drop the temporary integer column
    if 'pr_size_category_int' in repo_analysis_data.columns:
         repo_analysis_data = repo_analysis_data.drop(columns=['pr_size_category_int'])


    # Separate means for scaling
    repo_means = repo_analysis_data[mean_score_cols]
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(repo_means)

    # --- PCA ---
    pca = None
    # Start building repo_summary_pca from the filtered/binned data index
    repo_summary_pca = repo_summary.loc[repo_analysis_data.index].copy()
    repo_summary_pca['pr_size_category'] = size_categories # Add the category column

    logger.info("Running PCA on scaled mean competency scores...")
    logger.info("PC1 and PC2 represent the two dimensions capturing the largest variance in the combined competency scores across repositories.")

    # ... (rest of PCA logic including plotting remains largely the same, ensures repo_summary_pca is updated) ...
    try:
        n_components = min(2, scaled_data.shape[0], scaled_data.shape[1])
        if n_components < 2:
            logger.warning(f"PCA needs >= 2 components, data allows {n_components}. Skipping PCA plot.")
            repo_summary_pca[['PC1', 'PC2']] = np.nan
        else:
            pca = PCA(n_components=n_components)
            principal_components = pca.fit_transform(scaled_data)
            pc_cols = [f'PC{i+1}' for i in range(n_components)]
            # Create PCA DataFrame with the index of repo_analysis_data (which matches repo_means)
            pca_df = pd.DataFrame(data=principal_components, columns=pc_cols, index=repo_analysis_data.index)
            # Join PCA results back to the summary DataFrame
            repo_summary_pca = repo_summary_pca.join(pca_df) # Join based on index
            explained_var_str = ", ".join([f"PC{i+1}={ratio:.2%}" for i, ratio in enumerate(pca.explained_variance_ratio_)])
            logger.info(f"PCA Explained Variance Ratio: {explained_var_str}")

            # Plot PCA (Binned Size)
            fig_pca, ax_pca = plt.subplots(figsize=(14, 10))
            # Reset index for plotting, ensure PCA columns exist
            plot_data_pca = repo_summary_pca.reset_index()
            if 'PC1' not in plot_data_pca.columns or 'PC2' not in plot_data_pca.columns:
                 logger.error("PCA columns not found in dataframe for plotting.")
                 plt.close(fig_pca)
            else:
                try:
                    scatter = sns.scatterplot(
                        data=plot_data_pca.dropna(subset=['PC1', 'PC2']),
                        x='PC1', y='PC2',
                        size='pr_size_category', hue='pr_size_category',
                        palette='viridis_r', sizes=(50, 500),
                        size_order=bin_labels, hue_order=bin_labels,
                        alpha=0.7, legend='full', ax=ax_pca
                    )
                    # Check if explained_variance_ratio_ exists and has 2 elements
                    xlabel = "Principal Component 1"
                    ylabel = "Principal Component 2"
                    if hasattr(pca, 'explained_variance_ratio_') and len(pca.explained_variance_ratio_) >= 2:
                         xlabel += f" ({pca.explained_variance_ratio_[0]:.1%})"
                         ylabel += f" ({pca.explained_variance_ratio_[1]:.1%})"
                    ax_pca.set_xlabel(xlabel)
                    ax_pca.set_ylabel(ylabel)
                    ax_pca.set_title("Repository Quality Profiles (PCA) - Size/Color indicates PR Count Category")

                    # Improve legend
                    handles, labels = ax_pca.get_legend_handles_labels()
                    unique_hl = {}
                    for h, l in zip(handles, labels):
                         if l not in unique_hl and l in bin_labels: unique_hl[l] = h
                    ordered_handles = [unique_hl[label] for label in bin_labels if label in unique_hl]
                    ordered_labels = [label for label in bin_labels if label in unique_hl]
                    if ordered_handles:
                        ax_pca.legend(ordered_handles, ordered_labels, title="PR Count Category", loc='best', frameon=True)
                    else: ax_pca.legend(title="PR Count Category", loc='best', frameon=True)

                    # Add labels
                    texts = []
                    try:
                         # Ensure 'pr_count' exists before using nlargest
                         if 'pr_count' in repo_summary_pca.columns:
                             label_indices = repo_summary_pca['pr_count'].nlargest(PCA_LABEL_COUNT).index
                             label_subset = repo_summary_pca.loc[label_indices]
                             for repo_name_label, row in label_subset.iterrows():
                                 if pd.notna(row['PC1']) and pd.notna(row['PC2']):
                                     texts.append(ax_pca.text(row['PC1'], row['PC2'], repo_name_label, fontsize=8))
                         else:
                             logger.warning("'pr_count' column not found for labeling PCA plot.")

                         if texts:
                             try:
                                 from adjustText import adjust_text
                                 adjust_text(texts, arrowprops=dict(arrowstyle="-", color='gray', lw=0.5), ax=ax_pca)
                                 logger.info(f"Added labels for largest {len(texts)} repositories using adjustText.")
                             except ImportError: logger.warning("adjustText not found. Install for better PCA labels.")
                             except Exception as adj_e: logger.warning(f"adjustText failed: {adj_e}.")
                         # else: logger.info("No labels added to PCA plot.") # Already logged if pr_count missing
                    except KeyError as ke:
                        logger.error(f"KeyError during PCA labeling (likely missing column like 'pr_count'): {ke}")
                    except Exception as label_e:
                        logger.error(f"Error adding labels to PCA plot: {label_e}")

                    save_plot(fig_pca, "pca_repo_projection_binned_size", pdf_pages=pdf_pages, subdir=PLOT_SUBDIR_PCA)
                except Exception as e_pca_plot:
                     logger.error(f"Failed to generate PCA scatter plot: {e_pca_plot}", exc_info=True)
                     if plt.fignum_exists(fig_pca.number): plt.close(fig_pca)

    except Exception as e:
        logger.error(f"PCA analysis failed: {e}", exc_info=True)
        if 'fig_pca' in locals() and plt.fignum_exists(fig_pca.number): plt.close(fig_pca)
        if 'PC1' not in repo_summary_pca.columns: repo_summary_pca['PC1'] = np.nan
        if 'PC2' not in repo_summary_pca.columns: repo_summary_pca['PC2'] = np.nan


    # --- K-Means Clustering ---
    kmeans = None
    if 'Cluster' not in repo_summary_pca.columns: repo_summary_pca['Cluster'] = np.nan # Initialize column
    logger.info("Performing K-Means clustering...")
    try:
        num_samples_for_kmeans = scaled_data.shape[0]
        if num_samples_for_kmeans < CLUSTER_COUNT:
            logger.warning(f"Skipping K-Means: Samples ({num_samples_for_kmeans}) < Clusters ({CLUSTER_COUNT}).")
        else:
            kmeans = KMeans(n_clusters=CLUSTER_COUNT, random_state=42, n_init=10) # Use n_init='auto' in newer sklearn
            clusters = kmeans.fit_predict(scaled_data)
            # Assign clusters back using the index from scaled_data (which is repo_analysis_data.index)
            repo_summary_pca.loc[repo_analysis_data.index, 'Cluster'] = clusters
            # Convert to category *after* assigning all values
            repo_summary_pca['Cluster'] = repo_summary_pca['Cluster'].astype('category')

            # --- Plot Clusters on PCA (with Increased Font Sizes) ---
            # Ensure PCA components and Cluster columns exist and are not all NaN
            if (pca is not None and
                'PC1' in repo_summary_pca.columns and 'PC2' in repo_summary_pca.columns and
                repo_summary_pca[['PC1', 'PC2']].notna().all(axis=1).any() and
                'Cluster' in repo_summary_pca.columns and repo_summary_pca['Cluster'].notna().any()):

                # --- Define Font Sizes for Cluster Plot ---
                cluster_plot_title_fontsize = 26
                cluster_plot_axis_label_fontsize = 20
                cluster_plot_legend_title_fontsize = 18
                cluster_plot_legend_item_fontsize = 16
                cluster_plot_tick_fontsize = 16 # For axis ticks if needed

                fig_clus, ax_clus = plt.subplots(figsize=(15, 11)) # Slightly larger figure
                plot_data_cluster = repo_summary_pca.reset_index()
                try:
                    # Ensure Cluster is categorical before plotting
                    plot_data_cluster['Cluster'] = plot_data_cluster['Cluster'].astype('category')
                    plot_data_cluster['pr_size_category'] = plot_data_cluster['pr_size_category'].astype('category')
                    # Ensure size_order and hue_order use categories present in the data
                    valid_size_order = [cat for cat in bin_labels if cat in plot_data_cluster['pr_size_category'].cat.categories]

                    sns.scatterplot(
                        data=plot_data_cluster.dropna(subset=['PC1', 'PC2', 'Cluster']), # Drop NaNs essential for plot
                        x='PC1', y='PC2',
                        hue='Cluster', size='pr_size_category',
                        palette='tab10', sizes=(60, 600), # Slightly larger markers
                        size_order=valid_size_order, # Use valid categories
                        alpha=0.8, ax=ax_clus, legend='full'
                    )
                    # Check explained_variance_ratio_ exists
                    xlabel = "Principal Component 1"
                    ylabel = "Principal Component 2"
                    if hasattr(pca, 'explained_variance_ratio_') and len(pca.explained_variance_ratio_) >= 2:
                         xlabel += f" ({pca.explained_variance_ratio_[0]:.1%})"
                         ylabel += f" ({pca.explained_variance_ratio_[1]:.1%})"

                    # Apply Font Sizes
                    ax_clus.set_xlabel(xlabel, fontsize=cluster_plot_axis_label_fontsize)
                    ax_clus.set_ylabel(ylabel, fontsize=cluster_plot_axis_label_fontsize)
                    ax_clus.set_title(
                        f"Repository Clusters (k={CLUSTER_COUNT}) in PCA Space - Size indicates PR Count Category",
                        fontsize=cluster_plot_title_fontsize
                    )
                    # Apply tick font size
                    ax_clus.tick_params(axis='both', which='major', labelsize=cluster_plot_tick_fontsize)

                    # Improve Legend with Font Sizes
                    handles, labels = ax_clus.get_legend_handles_labels()
                    # Ensure Cluster categories are strings for comparison
                    cluster_categories = plot_data_cluster['Cluster'].cat.categories.astype(str).tolist()
                    size_categories_plot = valid_size_order # Use the validated order

                    cluster_handles = [h for h, l in zip(handles, labels) if str(l) in cluster_categories]
                    cluster_labels_leg = [l for l in labels if str(l) in cluster_categories]
                    size_handles = [h for h, l in zip(handles, labels) if l in size_categories_plot]
                    size_labels_leg = [l for l in labels if l in size_categories_plot]

                    if ax_clus.get_legend() is not None: ax_clus.get_legend().remove()
                    leg1 = None
                    if cluster_handles:
                        leg1 = ax_clus.legend(cluster_handles, cluster_labels_leg,
                                             title=f'Cluster (k={CLUSTER_COUNT})',
                                             loc='upper left', frameon=True,
                                             title_fontsize=cluster_plot_legend_title_fontsize, # Legend title size
                                             fontsize=cluster_plot_legend_item_fontsize) # Legend item size
                        ax_clus.add_artist(leg1)

                    if size_handles:
                        # Reorder handles/labels to match original bin_labels order
                        ordered_size_handles = []
                        ordered_size_labels = []
                        current_legend_map = dict(zip(size_labels_leg, size_handles))
                        for bin_l in bin_labels: # Use the master bin_labels list for order
                             if bin_l in current_legend_map:
                                 ordered_size_handles.append(current_legend_map[bin_l])
                                 ordered_size_labels.append(bin_l)
                        # Only add legend if there's something to show
                        if ordered_size_handles:
                             leg2 = ax_clus.legend(ordered_size_handles, ordered_size_labels,
                                                  title='PR Count Category (Size)',
                                                  loc='lower left', frameon=True,
                                                  title_fontsize=cluster_plot_legend_title_fontsize, # Legend title size
                                                  fontsize=cluster_plot_legend_item_fontsize) # Legend item size
                             # If leg1 exists, ensure it's still displayed
                             if leg1 is not None:
                                 ax_clus.add_artist(leg1)

                    # Adjust layout after adding legends
                    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust top for title

                    save_plot(fig_clus, "kmeans_clusters_pca_binned_size", pdf_pages=pdf_pages, subdir=PLOT_SUBDIR_PCA)
                except Exception as e_clus_plot:
                    logger.error(f"Failed K-Means cluster plot: {e_clus_plot}", exc_info=True)
                    if plt.fignum_exists(fig_clus.number): plt.close(fig_clus)
            else:
                logger.warning("Skipping PCA cluster plot: PCA results incomplete or Cluster data missing.")

            # Analyze Cluster Characteristics
            # ... (remains the same, uses repo_summary_pca) ...
            # (Cluster characteristics calculation code)

    except Exception as e:
        logger.error(f"K-Means clustering failed: {e}", exc_info=True)
        if 'fig_clus' in locals() and plt.fignum_exists(fig_clus.number): plt.close(fig_clus)
        if 'Cluster' not in repo_summary_pca.columns: repo_summary_pca['Cluster'] = np.nan


    # --- Save Final Summary ---
    try:
        # Ensure essential columns exist before saving, even if NaN
        for col in ['Cluster', 'PC1', 'PC2', 'pr_size_category']:
             if col not in repo_summary_pca.columns:
                  repo_summary_pca[col] = np.nan
        summary_path = os.path.join(OUTPUT_DIR, 'repository_summary_with_clusters_pca.csv')
        repo_summary_pca.to_csv(summary_path)
        logger.info(f"Final repository summary saved to {summary_path}")
    except Exception as e_save_summary:
         logger.error(f"Failed to save final repository summary: {e_save_summary}")

    # --- Interactive Plot (Plotly) ---
    # ... (remains the same, uses repo_summary_pca) ...
    logger.info("Generating interactive PCA/cluster plot (Plotly)...")
    try:
        # Ensure repo_summary_pca is not empty and has necessary columns
        if not repo_summary_pca.empty and all(col in repo_summary_pca.columns for col in ['PC1', 'PC2', 'Cluster', 'pr_count', 'pr_size_category']):
             plot_data_interactive = repo_summary_pca.reset_index() # Use reset_index to get 'repo' column
             required_interactive_cols = ['repo', 'PC1', 'PC2', 'Cluster', 'pr_count', 'pr_size_category']
             # Check for NaN issues specifically in PC1/PC2
             pca_nan_issue = plot_data_interactive[['PC1', 'PC2']].isna().all(axis=1).any()

             if not pca_nan_issue:
                 hover_data_dict = {
                     'repo': False, # Displayed by hover_name
                     'PC1':':.2f', 'PC2':':.2f',
                     'Cluster': True,
                     'pr_count': True,
                     'pr_size_category': True
                 }
                 # Add mean scores to hover data if they exist
                 mean_score_cols_present_interactive = [col for col in repo_summary.columns if col.startswith('Mean_score_') and col in plot_data_interactive.columns]
                 for col_name in mean_score_cols_present_interactive:
                      hover_data_dict[col_name] = ':.3f'

                 # Ensure Cluster is treated as categorical for color mapping
                 plot_data_interactive['Cluster'] = plot_data_interactive['Cluster'].astype('category')
                 plot_data_interactive['pr_size_category'] = plot_data_interactive['pr_size_category'].astype('category')

                 # Handle potential categorical Cluster mapping error if cluster IDs are not numeric-like
                 try:
                     cluster_categories = plot_data_interactive['Cluster'].cat.categories
                     color_map = {c: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
                                  for i, c in enumerate(cluster_categories)}
                 except Exception: # Fallback if categories are weird
                     color_map = None
                     logger.warning("Could not generate discrete color map for clusters, using default.")


                 fig_interactive = px.scatter(
                     plot_data_interactive.dropna(subset=['PC1', 'PC2', 'Cluster']), # Ensure no NaNs in core plot vars
                     x='PC1', y='PC2',
                     color='Cluster',
                     size='pr_count',
                     symbol='pr_size_category',
                     color_discrete_map=color_map, # Use generated map or None
                     symbol_sequence=['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up'],
                     hover_name='repo',
                     hover_data=hover_data_dict,
                     title=f"Interactive Repository Clusters (k={CLUSTER_COUNT}) - Size(PRs) / Symbol(PR Category)",
                     labels={'Cluster': f'Cluster (k={CLUSTER_COUNT})', 'pr_size_category': 'PR Size Category'}
                 )
                 fig_interactive.update_traces(marker=dict(opacity=0.7, line=dict(width=0.5, color='DarkSlateGrey')))
                 fig_interactive.update_layout(legend_title_text=f'Cluster / Size Category (Symbol)')

                 interactive_plot_filename = "interactive_repo_clusters_sized_symbols.html"
                 interactive_plot_filepath = os.path.join(OUTPUT_DIR, PLOT_SUBDIR_PCA, interactive_plot_filename)
                 fig_interactive.write_html(interactive_plot_filepath)
                 logger.info(f"Interactive cluster plot saved to: {interactive_plot_filepath}")
             else:
                 logger.warning(f"Skipping interactive plot: PCA results have critical NaN values.")
        else:
             logger.warning(f"Skipping interactive plot: Missing required columns in repo_summary_pca or it's empty.")

    except Exception as e:
        logger.error(f"Failed interactive plot generation: {e}", exc_info=True)


    return repo_summary_pca # Return the dataframe with PCA/Cluster info


# perform_timeseries_analysis remains the same, operates on df_main
def perform_timeseries_analysis(df, score_cols, pdf_pages):
    """Perform time series decomposition and plot components individually."""
    logger = logging.getLogger(__name__)
    logger.info("Generating Time Series Decomposition Plots (Bundled & Individual Components)...")

    if not pd.api.types.is_datetime64_any_dtype(df['created_at']):
        logger.error("Cannot perform time series: 'created_at' not datetime.")
        return
    if not score_cols:
        logger.warning("No score columns provided for time series analysis.")
        return

    df_ts_ready = df.set_index('created_at').copy()
    min_periods_for_decomp = 2 * 12 # For yearly seasonality (period=12)

    for score in score_cols:
        fig_ts_bundle = None # For the combined plot
        try:
            if score in df_ts_ready.columns and df_ts_ready[score].notna().any():
                ts_data = df_ts_ready.resample('M')[score].mean()
                ts_data_clean = ts_data.dropna()

                if ts_data_clean.shape[0] >= min_periods_for_decomp:
                    logger.debug(f"Attempting decomposition for {score} ({ts_data_clean.shape[0]} points).")
                    decomposition = seasonal_decompose(ts_data_clean, model='additive', period=12)

                    # Plot Bundled Decomposition
                    try:
                        fig_ts_bundle = decomposition.plot()
                        fig_ts_bundle.set_size_inches(12, 8)
                        title_bundle = f"Time Series Decomposition for {score.replace('score_', '').replace('_', ' ').title()}"
                        fig_ts_bundle.suptitle(title_bundle, y=1.01)
                        plt.tight_layout(rect=[0, 0, 1, 1])
                        save_plot(fig_ts_bundle, f"timeseries_decomp_bundled_{score}", pdf_pages=pdf_pages,
                                  subdir=PLOT_SUBDIR_TS_BUNDLE)
                    except Exception as e_bundle_ts:
                         logger.error(f"Failed plotting bundled TS decomp for {score}: {e_bundle_ts}")
                         if fig_ts_bundle and plt.fignum_exists(fig_ts_bundle.number): plt.close(fig_ts_bundle)

                    # Plot Individual Components
                    logger.debug(f"Generating individual component plots for {score}...")
                    for comp_name in ['observed', 'trend', 'seasonal', 'resid']:
                        fig_comp = None # Init inside loop
                        try:
                            component_data = getattr(decomposition, comp_name)
                            if component_data is not None and component_data.notna().any():
                                fig_comp, ax_comp = plt.subplots(figsize=(12, 3))
                                component_data.plot(ax=ax_comp)
                                ax_comp.set_title(f"{comp_name.capitalize()} Component")
                                ax_comp.set_ylabel("Score" if comp_name == 'observed' else "Component Value")
                                if comp_name == 'observed':
                                     ax_comp.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f"{x:,.2f}"))
                                plt.tight_layout()
                                save_plot(fig_comp, f"timeseries_comp_{score}_{comp_name}",
                                          subdir=PLOT_SUBDIR_TS_SINGLE, is_single_component=True)
                            else:
                                 logger.debug(f"Component '{comp_name}' empty or None for {score}. Skipping individual plot.")
                                 if fig_comp and plt.fignum_exists(fig_comp.number): plt.close(fig_comp)
                        except Exception as e_comp_ts:
                            logger.error(f"Failed plotting individual TS comp {comp_name} for {score}: {e_comp_ts}")
                            if fig_comp and plt.fignum_exists(fig_comp.number): plt.close(fig_comp)

                else:
                    logger.warning(f"Skipping decomposition for {score}: insufficient data ({ts_data_clean.shape[0]} < {min_periods_for_decomp}).")
                    if fig_ts_bundle and plt.fignum_exists(fig_ts_bundle.number): plt.close(fig_ts_bundle)
            else:
                logger.warning(f"Skipping decomposition for {score}: No valid data.")
        except Exception as e:
            logger.error(f"Error during time series analysis for {score}: {e}", exc_info=True)
            if fig_ts_bundle and plt.fignum_exists(fig_ts_bundle.number): plt.close(fig_ts_bundle)


# --- Meta-Report Generation Helper Functions ---

def read_csv_safe(filepath, index_col=None):
    """Safely reads a CSV file for the meta-report."""
    logger = logging.getLogger(__name__)
    try:
        if not os.path.exists(filepath):
            logger.warning(f"[MetaReport] File not found: {filepath}")
            return None
        df = pd.read_csv(filepath, index_col=index_col)
        logger.info(f"[MetaReport] Successfully read: {os.path.basename(filepath)}")
        return df
    except Exception as e:
        logger.error(f"[MetaReport] Could not read file {filepath}: {e}")
        return None

def format_corr(value):
    """Formats correlation values for the meta-report."""
    if pd.isna(value): return "N/A"
    strength = ""
    if abs(value) >= 0.7: strength = " (Strong)"
    elif abs(value) >= 0.4: strength = " (Moderate)"
    elif abs(value) >= 0.1: strength = " (Weak)"
    else: strength = " (Very Weak)"
    return f"{value:.3f}{strength}"

def get_correlation_summary(corr_matrix, n=3, level="Repo-Level"):
    """Gets the top N positive/negative correlations and formats for report."""
    if corr_matrix is None or corr_matrix.empty:
        return f"### {level} Correlations\n\nCorrelation matrix not available or empty.\n"

    # Ensure index/columns match for stacking, handle potential duplicate names if any
    matrix_copy = corr_matrix.copy()
    matrix_copy.index.name = 'Score1'
    matrix_copy.columns.name = 'Score2'

    # Create a mask for the upper triangle (excluding the diagonal)
    mask = np.triu(np.ones(matrix_copy.shape), k=1).astype(bool)
    corrs = matrix_copy.where(mask).stack().sort_values(ascending=False)

    if corrs.empty:
        return f"### {level} Correlations\n\nNo off-diagonal correlations found.\n"

    strongest_pos = corrs.head(n)
    strongest_neg = corrs[corrs < 0].tail(n).sort_values() # Get most negative

    report_lines = [f"### {level} Correlations"]
    report_lines.append(f"\n**Top {n} Positive Correlations:**")
    if not strongest_pos.empty:
        for (idx1, idx2), val in strongest_pos.items():
            report_lines.append(f"- {idx1.replace('score_', '')} & {idx2.replace('score_', '')}: {format_corr(val)}")
    else: report_lines.append("- *None significant or found.*")

    report_lines.append(f"\n**Top {n} Negative Correlations:**")
    if not strongest_neg.empty:
        for (idx1, idx2), val in strongest_neg.items():
            report_lines.append(f"- {idx1.replace('score_', '')} & {idx2.replace('score_', '')}: {format_corr(val)}")
    else: report_lines.append("- *None found.*")

    return "\n".join(report_lines)


def describe_cluster(cluster_id, cluster_data, all_scores_desc):
    """Generates a Markdown description for a single cluster."""
    if cluster_data is None or cluster_data.empty:
        return f"### Cluster {cluster_id}: Data Unavailable\n\n---\n"

    lines = [f"### Cluster {cluster_id}"]
    try:
        size = cluster_data.get('ClusterSize')
        mean_prs = cluster_data.get('Mean_PR_Count')
        median_prs = cluster_data.get('Median_PR_Count')

        lines.append(f"- **Size:** {int(size):,} repositories" if pd.notna(size) else "- **Size:** N/A")
        lines.append(f"- **PR Volume:** Mean ~{mean_prs:,.0f}, Median ~{median_prs:,.0f} PRs per repo" if pd.notna(mean_prs) and pd.notna(median_prs) else "- **PR Volume:** N/A")

        lines.append("- **Dominant PR Size Categories:**")
        size_cats = {k.replace('SizeCatPerc_', ''): v for k, v in cluster_data.items() if k.startswith('SizeCatPerc_')}
        if size_cats:
            sorted_cats = sorted(size_cats.items(), key=lambda item: item[1], reverse=True)
            for cat, perc in sorted_cats[:3]: # Show top 3
                if perc > 0.01: lines.append(f"  - {cat}: {perc:.1%}")
            if not any(perc > 0.01 for _, perc in sorted_cats): lines.append("  - *Distribution spread across categories.*")
        else: lines.append("  - *Size category data unavailable.*")


        lines.append("- **Competency Profile (Mean Scores vs Overall):**")
        score_cols_cluster = [col for col in cluster_data.index if col.startswith('score_')]
        comparisons = []
        for score in score_cols_cluster:
            mean_val = cluster_data.get(score)
            overall_mean = all_scores_desc.loc[score, 'mean'] if all_scores_desc is not None and score in all_scores_desc.index else None
            comparison = " (vs Overall: N/A)"
            comparison_cat = "neutral" # For highlighting later
            if pd.notna(mean_val) and pd.notna(overall_mean):
                diff_ratio = mean_val / overall_mean if overall_mean != 0 else np.inf
                if diff_ratio > 1.20: comparison, comparison_cat = " (**Higher**)", "high"
                elif diff_ratio < 0.80: comparison, comparison_cat = " (_Lower_)", "low"
                else: comparison = " (Avg)"
            comparisons.append((score.replace('score_', '').replace('_', ' ').title(), f"{mean_val:.3f}{comparison}", comparison_cat))

        # Highlight distinctive scores
        high_scores = [f"**{name}**" for name, text, cat in comparisons if cat == "high"]
        low_scores = [f"_{name}_" for name, text, cat in comparisons if cat == "low"]
        if high_scores or low_scores:
            lines.append(f"  - **Distinctive:** {', '.join(high_scores)}{' | ' if high_scores and low_scores else ''}{', '.join(low_scores)}")

        # List all scores for reference
        for name, text, _ in comparisons:
             lines.append(f"  - {name}: {text}")

    except Exception as e:
        lines.append(f"- *Error generating cluster description: {e}*")

    lines.append("\n---\n") # Separator
    return "\n".join(lines)


# --- Meta-Report Generation Helper Functions ---
# (read_csv_safe, format_corr, get_correlation_summary, describe_cluster functions remain the same as in the previous version)
# Make sure these helper functions are included before generate_meta_report

def read_csv_safe(filepath, index_col=None):
    """Safely reads a CSV file for the meta-report."""
    logger = logging.getLogger(__name__)
    try:
        if not os.path.exists(filepath):
            logger.warning(f"[MetaReport] File not found: {filepath}")
            return None
        # Try reading with utf-8-sig to handle potential BOM
        try:
            df = pd.read_csv(filepath, index_col=index_col, encoding='utf-8-sig')
        except UnicodeDecodeError:
            logger.warning(f"[MetaReport] utf-8-sig failed for {filepath}, trying default encoding.")
            df = pd.read_csv(filepath, index_col=index_col) # Fallback to default
        logger.info(f"[MetaReport] Successfully read: {os.path.basename(filepath)}")
        return df
    except Exception as e:
        logger.error(f"[MetaReport] Could not read file {filepath}: {e}")
        return None

def format_corr(value):
    """Formats correlation values for the meta-report."""
    if pd.isna(value): return "N/A"
    strength = ""
    if abs(value) >= 0.7: strength = " (Strong)"
    elif abs(value) >= 0.4: strength = " (Moderate)"
    elif abs(value) >= 0.1: strength = " (Weak)"
    else: strength = " (Very Weak)"
    return f"{value:.3f}{strength}"

def get_correlation_summary(corr_matrix, n=3, level="Repo-Level"):
    """Gets the top N positive/negative correlations and formats for report."""
    if corr_matrix is None or corr_matrix.empty:
        return f"### {level} Correlations\n\nCorrelation matrix not available or empty.\n"

    # Ensure index/columns match for stacking, handle potential duplicate names if any
    matrix_copy = corr_matrix.copy()
    matrix_copy.index.name = 'Score1'
    matrix_copy.columns.name = 'Score2'

    # Create a mask for the upper triangle (excluding the diagonal)
    mask = np.triu(np.ones(matrix_copy.shape), k=1).astype(bool)
    corrs = matrix_copy.where(mask).stack().sort_values(ascending=False)

    if corrs.empty:
        return f"### {level} Correlations\n\nNo off-diagonal correlations found.\n"

    strongest_pos = corrs.head(n)
    # Get most negative - handle edge case where there are no negative correlations
    neg_corrs = corrs[corrs < 0]
    strongest_neg = neg_corrs.tail(n).sort_values() if not neg_corrs.empty else pd.Series(dtype=float)


    report_lines = [f"### {level} Correlations"]
    report_lines.append(f"\n**Top {n} Positive Correlations:**")
    if not strongest_pos.empty:
        for (idx1, idx2), val in strongest_pos.items():
            # Clean names for display
            name1 = idx1.replace('score_', '').replace('_', ' ').title()
            name2 = idx2.replace('score_', '').replace('_', ' ').title()
            report_lines.append(f"- {name1} & {name2}: {format_corr(val)}")
    else: report_lines.append("- *None significant or found.*")

    report_lines.append(f"\n**Top {n} Negative Correlations:**")
    if not strongest_neg.empty:
        for (idx1, idx2), val in strongest_neg.items():
            name1 = idx1.replace('score_', '').replace('_', ' ').title()
            name2 = idx2.replace('score_', '').replace('_', ' ').title()
            report_lines.append(f"- {name1} & {name2}: {format_corr(val)}")
    else: report_lines.append("- *None found.*")

    return "\n".join(report_lines)


def describe_cluster(cluster_id, cluster_data, all_scores_desc):
    """Generates a Markdown description for a single cluster."""
    if cluster_data is None or cluster_data.empty:
        return f"### Cluster {cluster_id}: Data Unavailable\n\n---\n"

    lines = [f"### Cluster {cluster_id}"]
    try:
        size = cluster_data.get('ClusterSize')
        mean_prs = cluster_data.get('Mean_PR_Count')
        median_prs = cluster_data.get('Median_PR_Count')

        lines.append(f"- **Size:** {int(size):,} repositories" if pd.notna(size) else "- **Size:** N/A")
        lines.append(f"- **PR Volume:** Mean ~{mean_prs:,.0f}, Median ~{median_prs:,.0f} PRs per repo" if pd.notna(mean_prs) and pd.notna(median_prs) else "- **PR Volume:** N/A")

        lines.append("- **Dominant PR Size Categories:**")
        size_cats = {k.replace('SizeCatPerc_', ''): v for k, v in cluster_data.items() if k.startswith('SizeCatPerc_')}
        if size_cats:
            sorted_cats = sorted(size_cats.items(), key=lambda item: item[1], reverse=True)
            shown_cats = 0
            for cat, perc in sorted_cats: # Show top categories making up bulk
                if perc > 0.05: # Threshold for showing
                    lines.append(f"  - {cat}: {perc:.1%}")
                    shown_cats += 1
            if shown_cats == 0: lines.append("  - *Distribution spread or data unavailable.*")
        else: lines.append("  - *Size category data unavailable.*")


        lines.append("- **Competency Profile (Mean Scores vs Overall):**")
        # Use original score names for comparison lookup
        score_cols_cluster = [col for col in cluster_data.index if col.startswith('score_')]
        comparisons = []
        if all_scores_desc is not None:
             for score in score_cols_cluster:
                 mean_val = cluster_data.get(score)
                 # Ensure score exists in descriptives_df index
                 if score in all_scores_desc.index:
                     overall_mean = all_scores_desc.loc[score, 'mean']
                     comparison = " (vs Overall: N/A)"
                     comparison_cat = "neutral" # For highlighting later
                     if pd.notna(mean_val) and pd.notna(overall_mean):
                         diff_ratio = mean_val / overall_mean if overall_mean != 0 else np.inf
                         # Adjusted thresholds for highlighting
                         if diff_ratio > 1.15: comparison, comparison_cat = " (**Higher**)", "high"
                         elif diff_ratio < 0.85: comparison, comparison_cat = " (_Lower_)", "low"
                         else: comparison = " (Avg)"
                 else:
                     overall_mean = None
                     comparison = " (vs Overall: N/A)"
                     comparison_cat = "neutral"

                 # Add to list for sorting/display
                 comparisons.append({
                     'name': score.replace('score_', '').replace('_', ' ').title(),
                     'value_text': f"{mean_val:.3f}{comparison}" if pd.notna(mean_val) else "N/A",
                     'category': comparison_cat,
                     'raw_value': mean_val if pd.notna(mean_val) else -np.inf # For potential sorting
                 })
        else:
            # Fallback if overall descriptives are missing
            for score in score_cols_cluster:
                 mean_val = cluster_data.get(score)
                 comparisons.append({
                     'name': score.replace('score_', '').replace('_', ' ').title(),
                     'value_text': f"{mean_val:.3f}" if pd.notna(mean_val) else "N/A",
                     'category': 'neutral',
                     'raw_value': mean_val if pd.notna(mean_val) else -np.inf
                 })

        # Highlight distinctive scores first
        high_scores = [f"**{comp['name']}**" for comp in comparisons if comp['category'] == "high"]
        low_scores = [f"_{comp['name']}_" for comp in comparisons if comp['category'] == "low"]
        if high_scores or low_scores:
            distinctive_str = " | ".join(filter(None, [", ".join(high_scores), ", ".join(low_scores)]))
            lines.append(f"  - **Distinctive:** {distinctive_str}")

        # List all scores for reference
        for comp in comparisons:
             lines.append(f"  - {comp['name']}: {comp['value_text']}")

    except Exception as e:
        lines.append(f"- *Error generating cluster description detail: {e}*")

    lines.append("\n---\n") # Separator
    return "\n".join(lines)


# --- Meta-Report Generation Function (Corrected) ---

def generate_meta_report(output_dir, score_cols, config):
    """Generates the Markdown meta-report file."""
    logger = logging.getLogger(__name__)
    logger.info("--- Starting Meta-Report Generation ---")
    report_content = []

    # --- Read Data Files ---
    logger.info("Reading analysis output files for meta-report...")
    cluster_chars_df = read_csv_safe(os.path.join(output_dir, 'cluster_characteristics.csv'), index_col='ClusterID')
    descriptives_df = read_csv_safe(os.path.join(output_dir, 'overall_score_descriptives.csv'), index_col=0)
    repo_corr_matrix_df = read_csv_safe(os.path.join(output_dir, 'competency_correlation_matrix.csv'), index_col=0)
    repo_counts_df = read_csv_safe(os.path.join(output_dir, 'repo_pr_counts.csv'))
    kruskal_df = read_csv_safe(os.path.join(output_dir, 'repo_kruskal_wallis_results.csv'), index_col='Competency')
    repo_summary_filtered_df = read_csv_safe(os.path.join(output_dir, 'repository_mean_std_summary_filtered.csv'))
    repo_summary_pca_df = read_csv_safe(os.path.join(output_dir, 'repository_summary_with_clusters_pca.csv'))
    preprocessed_df = read_csv_safe(os.path.join(output_dir, 'preprocessed_scored_prs.csv')) # Read the big file
    # Manual rankings (use paths from config)
    rene_ranking_df = read_csv_safe(config.get('RENE_RANKING_FILE')) if config.get('RENE_RANKING_FILE') else None
    chris_ranking_df = read_csv_safe(config.get('CHRIS_RANKING_FILE')) if config.get('CHRIS_RANKING_FILE') else None


    # --- Basic Counts and Config ---
    num_total_repos = len(repo_counts_df) if repo_counts_df is not None else "N/A"
    num_total_prs = int(descriptives_df['count'].iloc[0]) if descriptives_df is not None and not descriptives_df.empty else (len(preprocessed_df) if preprocessed_df is not None else "N/A")
    num_analyzed_repos = len(repo_summary_filtered_df) if repo_summary_filtered_df is not None else "N/A"
    num_clusters = len(cluster_chars_df) if cluster_chars_df is not None else "N/A"
    # Ensure score_cols are consistent
    if descriptives_df is not None:
        score_names = list(descriptives_df.index)
    elif preprocessed_df is not None:
        score_names = [col for col in preprocessed_df.columns if col.startswith('score_')]
    else:
        score_names = score_cols # Fallback to list passed in

    # --- Start Generating Markdown ---
    report_content.append(f"# Holistic PR Analysis Meta-Report")
    report_content.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_content.append(f"Based on analysis output from: `{output_dir}`")
    report_content.append("\n---\n")

    # --- Introduction ---
    report_content.append("## 1. Introduction & Scope")
    report_content.append(f"- **Total Repositories Scanned:** {num_total_repos:,}" if isinstance(num_total_repos, int) else f"- **Total Repositories Scanned:** {num_total_repos}")
    report_content.append(f"- **Total PRs Analyzed (within timeframe):** {num_total_prs:,}" if isinstance(num_total_prs, int) else f"- **Total PRs Analyzed (within timeframe):** {num_total_prs}")
    report_content.append(f"- **Analysis Timeframe:** {config.get('START_DATE', 'N/A')} to {config.get('END_DATE', 'N/A')}")
    report_content.append(f"- **Repositories Included in Statistical Comparisons (>= {config.get('MIN_PRS_PER_REPO_FOR_STATS', 'N/A')} PRs):** {num_analyzed_repos}")
    report_content.append(f"- **Identified Repository Clusters:** {num_clusters}")
    report_content.append(f"- **Competency Scores Analyzed:** {', '.join(s.replace('score_', '').replace('_', ' ').title() for s in score_names)}")
    report_content.append("\n---\n")

    # --- Overall Descriptives ---
    report_content.append("## 2. Overall Competency Score Distributions")
    if descriptives_df is not None:
        # ... (Descriptive summary generation remains the same) ...
        report_content.append("The following table summarizes the distribution of each competency score across *all* analyzed PRs within the timeframe.")
        desc_display = descriptives_df[['mean', 'std', 'min', '25%', '50%', '75%', 'max']].copy()
        desc_display.columns = ['Mean', 'Std Dev', 'Min', '25% (Q1)', 'Median (Q2)', '75% (Q3)', 'Max']
        desc_display.index.name = "Competency"
        desc_display.index = desc_display.index.str.replace('score_', '').str.replace('_', ' ').str.title()
        try:
            report_content.append(desc_display.to_markdown(floatfmt=".3f"))
        except ImportError:
            report_content.append("```\n" + desc_display.to_string(float_format="%.3f") + "\n```")
            report_content.append("*Note: Install 'tabulate' (`pip install tabulate`) for better table formatting.*")

        # Highlights
        mean_scores = descriptives_df['mean'].sort_values()
        if not mean_scores.empty:
            highest_mean_score = mean_scores.index[-1].replace('score_', '').replace('_', ' ').title()
            lowest_mean_score = mean_scores.index[0].replace('score_', '').replace('_', ' ').title()
            report_content.append("\n**Highlights:**")
            report_content.append(f"- Highest average score: **{highest_mean_score}** (`{mean_scores.iloc[-1]:.3f}`).")
            report_content.append(f"- Lowest average score: **{lowest_mean_score}** (`{mean_scores.iloc[0]:.3f}`).")
        std_devs = descriptives_df['std'].sort_values()
        if not std_devs.empty:
             most_variable_score = std_devs.index[-1].replace('score_', '').replace('_', ' ').title()
             least_variable_score = std_devs.index[0].replace('score_', '').replace('_', ' ').title()
             report_content.append(f"- Most score variability: **{most_variable_score}** (Std Dev: `{std_devs.iloc[-1]:.3f}`).")
             report_content.append(f"- Least score variability: **{least_variable_score}** (Std Dev: `{std_devs.iloc[0]:.3f}`).")
    else:
        report_content.append("Overall descriptive statistics data (`overall_score_descriptives.csv`) not found.")
    report_content.append("\n*Refer to plots in `plots/descriptive/` for visual distributions.*")
    report_content.append("\n---\n")

    # --- Repository Comparison (Kruskal-Wallis) ---
    report_content.append("## 3. Repository Differences (Kruskal-Wallis)")
    if kruskal_df is not None:
        # ... (Kruskal-Wallis summary generation remains the same) ...
        report_content.append(f"Kruskal-Wallis tests were performed to check for significant differences in median competency scores across the {num_analyzed_repos} repositories meeting the minimum PR threshold.")
        report_content.append(f"Significance level (alpha) = {config.get('SIGNIFICANCE_LEVEL', 0.05)}, with FDR Benjamini-Hochberg correction.")

        significant_diffs = kruskal_df[kruskal_df['Significant_After_Correction'] == True]
        non_significant_diffs = kruskal_df[kruskal_df['Significant_After_Correction'] == False]

        if not significant_diffs.empty:
            report_content.append("\n**Significant differences *between repositories* found for:**")
            for score in significant_diffs.index:
                p_corr = significant_diffs.loc[score, 'P_Value_Corrected']
                report_content.append(f"- **{score.replace('score_', '').replace('_', ' ').title()}** (p_corr = {p_corr:.3g})")
        else:
            report_content.append("\n**No significant differences** were found between repositories for any competency score after correction.")

        if not non_significant_diffs.empty and significant_diffs.empty: # Only show this if *none* were significant
             report_content.append("\nCompetencies tested (no significant difference found):")
             report_content.append(f"- {', '.join(non_significant_diffs.index.str.replace('score_', '').str.replace('_', ' ').str.title())}")
        elif not non_significant_diffs.empty: # List non-significant ones if some *were* significant
             report_content.append("\n*No significant differences found for remaining competencies.*")

    else:
        report_content.append("Kruskal-Wallis results (`repo_kruskal_wallis_results.csv`) not found.")
    report_content.append("\n*Refer to boxplots in `plots/comparison/` for visual comparisons.*")
    report_content.append("\n---\n")

    # --- Correlations ---
    report_content.append("## 4. Competency Correlations")
    pr_corr_matrix = None # Initialize in case PR level fails

    # 4.1 Repo-Level Correlation (using competency_correlation_matrix.csv)
    if repo_corr_matrix_df is not None:
        report_content.append(get_correlation_summary(repo_corr_matrix_df, n=3, level="Repository-Level (Based on Mean Scores)"))
    else:
        report_content.append("### Repository-Level Correlations\n\nRepo-level correlation matrix (`competency_correlation_matrix.csv`) not found.")

    # 4.2 PR-Level Correlation (using preprocessed_scored_prs.csv)
    report_content.append("\n") # Add space
    if preprocessed_df is not None and not preprocessed_df.empty and score_names:
        logger.info("[MetaReport] Calculating PR-level correlations...")
        try:
            pr_scores_df = preprocessed_df[score_names].copy()
            pr_scores_df.dropna(inplace=True)
            if len(pr_scores_df) >= 2:
                pr_corr_matrix = pr_scores_df.corr(method='spearman') # Assign to variable
                report_content.append(get_correlation_summary(pr_corr_matrix, n=3, level="PR-Level (Across All Individual PRs)"))
                pr_corr_path = os.path.join(output_dir, 'pr_level_competency_correlation_matrix.csv')
                pr_corr_matrix.to_csv(pr_corr_path)
                logger.info(f"[MetaReport] Saved PR-level correlation matrix to {pr_corr_path}")
            else:
                report_content.append("### PR-Level Correlations\n\nInsufficient data after dropping NaNs for PR-level correlation.")
        except Exception as pr_corr_e:
            logger.error(f"[MetaReport] Failed to calculate PR-level correlations: {pr_corr_e}")
            report_content.append(f"### PR-Level Correlations\n\nError calculating PR-level correlations: {pr_corr_e}")
    else:
        report_content.append("### PR-Level Correlations\n\nPreprocessed data (`preprocessed_scored_prs.csv`) not found or empty, skipping PR-level correlation.")

    report_content.append("\n**Interpretation Note:** Comparing repository-level and PR-level correlations can be insightful. Repo-level correlations show relationships between the *average* tendencies of repositories, while PR-level correlations reflect relationships within individual units of work across the entire dataset.")
    report_content.append("\n*Refer to heatmaps in `plots/correlation/` for repo-level visualization.*")
    report_content.append("\n---\n")

    # --- PCA and Clustering ---
    report_content.append("## 5. Repository Archetypes (PCA & Clustering)")
    if cluster_chars_df is not None and repo_summary_pca_df is not None:
        report_content.append(f"Principal Component Analysis (PCA) and K-Means clustering (k={num_clusters}) were used to identify potential repository archetypes based on mean competency scores.")
        report_content.append("Refer to the analysis log (`holistic_analysis_log_v5.log`) or PCA plots for explained variance ratios.")
        report_content.append("\n**Cluster Characteristics:**")
        for cluster_id in cluster_chars_df.index:
            cluster_data = cluster_chars_df.loc[cluster_id]
            report_content.append(describe_cluster(cluster_id, cluster_data, descriptives_df)) # Pass overall descriptives
    else:
        report_content.append("Cluster characteristics (`cluster_characteristics.csv`) or PCA summary data (`repository_summary_with_clusters_pca.csv`) not found.")
    report_content.append("\n*Refer to PCA/Cluster plots (static and interactive) in `plots/pca_cluster/` for visualization.*")
    report_content.append("\n---\n")

    # --- Manual Rankings ---
    # ... (Manual ranking section remains the same) ...
    report_content.append("## 6. Manual Ranking Files")
    found_manual = False
    if rene_ranking_df is not None:
        report_content.append(f"- Found Rene's ranking file (`{config.get('RENE_RANKING_FILE')}`) with {len(rene_ranking_df)} entries.")
        found_manual = True
    else:
        report_content.append(f"- Rene's ranking file ('{config.get('RENE_RANKING_FILE')}') not found or failed to load.")

    if chris_ranking_df is not None:
        report_content.append(f"- Found Chris's ranking file (`{config.get('CHRIS_RANKING_FILE')}`) with {len(chris_ranking_df)} entries.")
        found_manual = True
    else:
        report_content.append(f"- Chris's ranking file ('{config.get('CHRIS_RANKING_FILE')}') not found or failed to load.")

    if found_manual:
        report_content.append("\n*Note: Direct comparison between manual ranks and automated scores requires matching PR identifiers and potentially standardizing manual scores. This was not performed in this automated report.*")
    report_content.append("\n---\n")


    # --- Time Series ---
    report_content.append("## 7. Time Series Analysis")
    report_content.append("Time series decomposition (trend, seasonality, residuals) was performed on the monthly average scores for each competency.")
    report_content.append("This helps identify underlying patterns or shifts in PR quality aspects over the analyzed timeframe.")
    report_content.append("\n*Refer to individual component plots in `plots/timeseries/` for details.*")
    report_content.append("\n---\n")

    # --- Conclusion (Corrected Summary Point Generation) ---
    report_content.append("## 8. Key Takeaways & Conclusion")
    report_content.append("This automated analysis provides a multi-faceted view of PR quality characteristics across the studied repositories.")
    report_content.append("\n**Summary Points (Automated Extraction):**")
    try:
        # Kruskal-Wallis Summary
        if kruskal_df is not None and not kruskal_df.empty:
            significant_diffs_conclusion = kruskal_df[kruskal_df['Significant_After_Correction'] == True]
            if not significant_diffs_conclusion.empty:
                 sig_scores_conclusion = significant_diffs_conclusion.index.str.replace('score_', '').str.replace('_', ' ').str.title().tolist()
                 report_content.append(f"- Significant differences **between repositories** were observed for: {', '.join(sig_scores_conclusion)}.")
            else:
                 report_content.append("- No significant differences in median scores were detected between repositories after statistical correction.")
        else:
            report_content.append("- Kruskal-Wallis results for repository differences were unavailable.")

        # Clustering Summary
        if cluster_chars_df is not None:
             report_content.append(f"- Clustering identified **{len(cluster_chars_df)} distinct repository archetypes** based on their average competency profiles and PR volume (see Section 5 for details).")
        else:
            report_content.append("- Clustering results were unavailable.")

        # Correlation Summary (Safely extract examples)
        def extract_example_corr(summary_string):
            """Helper to safely extract the first correlation example line."""
            if not summary_string or not isinstance(summary_string, str): return "N/A"
            lines = summary_string.splitlines()
            # Find the first line starting with '-' under 'Positive' or 'Negative'
            for i, line in enumerate(lines):
                if line.strip().startswith("-") and i > 0 and "Top" in lines[i-1]:
                    return line.strip()[2:] # Return text after '- '
            return "N/A (No specific example found)"

        # Assign the single string results
        corr_summary_repo_str = get_correlation_summary(repo_corr_matrix_df, n=1, level="")
        corr_summary_pr_str = get_correlation_summary(pr_corr_matrix, n=1, level="") # Use matrix calculated earlier

        example_repo_corr = extract_example_corr(corr_summary_repo_str)
        example_pr_corr = extract_example_corr(corr_summary_pr_str)

        if example_repo_corr != "N/A":
             report_content.append(f"- Repo-level analysis suggests potential relationships, e.g., {example_repo_corr}.")
        else:
            report_content.append("- Repo-level correlation data was unavailable or showed no notable correlations.")
        if example_pr_corr != "N/A":
             report_content.append(f"- PR-level analysis showed different or similar patterns, e.g., {example_pr_corr}.")
        else:
             report_content.append("- PR-level correlation data was unavailable or showed no notable correlations.")

        # Descriptive Summary
        if descriptives_df is not None:
             mean_scores = descriptives_df['mean'].sort_values()
             if not mean_scores.empty:
                 highest_mean_score = mean_scores.index[-1].replace('score_', '').replace('_', ' ').title()
                 lowest_mean_score = mean_scores.index[0].replace('score_', '').replace('_', ' ').title()
                 report_content.append(f"- Overall, **{highest_mean_score}** received the highest average score, while **{lowest_mean_score}** received the lowest.")
        else:
            report_content.append("- Overall score descriptive statistics were unavailable.")

    except Exception as summary_e:
        logger.error(f"Error generating summary points for meta-report: {summary_e}", exc_info=True) # Log traceback
        report_content.append("- *Error generating automated summary points.*")


    report_content.append("\nFurther investigation could delve into the qualitative differences between clusters, the impact of specific repository practices, or the validation of automated scores against manual assessments.")
    report_content.append("\n---\n")


    # --- Write Report to File ---
    report_filepath = os.path.join(output_dir, META_REPORT_FILENAME)
    try:
        with open(report_filepath, 'w', encoding='utf-8') as f:
            f.write("\n".join(report_content))
        logger.info(f"Successfully generated meta-report: {report_filepath}")
    except Exception as e:
        logger.error(f"Failed to write meta-report file: {e}")

    logger.info("--- Meta-Report Generation Complete ---")


# --- Main Execution (Ensure generate_meta_report is called at the end) ---
if __name__ == "__main__":
    logger = logging.getLogger(__name__) # Get root logger
    df_main, score_cols_main = load_and_preprocess_data(INPUT_SCORED_FILE)

    if not df_main.empty and score_cols_main:
        pdf_report_path = os.path.join(OUTPUT_DIR, 'full_analysis_report_v5.pdf') # Keep PDF name consistent for now
        repo_summary_with_pca = pd.DataFrame() # Initialize to ensure it exists
        try:
            # --- Run Analysis ---
            logger.info("--- Starting Core Analysis ---")
            with PdfPages(pdf_report_path) as pdf:
                logger.info("Generating PDF Plots...")
                perform_descriptive_analysis(df_main, score_cols_main, pdf)
                repo_summary_main = perform_repository_comparison(df_main, score_cols_main, pdf)
                perform_correlation_analysis(repo_summary_main, pdf)
                # PCA/Clustering uses the filtered repo summary, save its result
                repo_summary_with_pca = perform_pca_and_clustering(repo_summary_main, pdf)
                perform_timeseries_analysis(df_main, score_cols_main, pdf)

            logger.info(f"Analysis plots/PDF Report saved to: {pdf_report_path}")
            logger.info("--- Core Analysis Complete ---")

            # --- Generate Meta Report ---
            # Create a config dict to pass relevant settings
            meta_report_config = {
                'START_DATE': START_DATE,
                'END_DATE': END_DATE,
                'MIN_PRS_PER_REPO_FOR_STATS': MIN_PRS_PER_REPO_FOR_STATS,
                'SIGNIFICANCE_LEVEL': SIGNIFICANCE_LEVEL,
                'RENE_RANKING_FILE': RENE_RANKING_FILE,
                'CHRIS_RANKING_FILE': CHRIS_RANKING_FILE,
                # Add other config items if needed by meta-report
            }
            # Pass necessary dataframes and config to the meta-report function
            generate_meta_report(OUTPUT_DIR, score_cols_main, meta_report_config)

        except Exception as analysis_err:
            logger.error(f"An error occurred during the main analysis pipeline: {analysis_err}", exc_info=True)

    else:
        logger.error("Analysis halted: Data loading failed, dataframe empty, or no valid score columns.")

    logging.info("="*30 + " Holistic Analysis V5.0 Complete " + "="*30)