import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, spearmanr, kruskal, binomtest # Added binomtest
from pathlib import Path
import csv
from collections import defaultdict
import warnings
from typing import Union, Tuple, Dict, List, Optional, Any
import math
import logging
import datetime # To timestamp the report
import re # For robust hypothesis number extraction

# --- Logging Setup ---
# Set level=logging.DEBUG to see detailed logs for troubleshooting plot generation
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
BASE_DIR = Path("SURVEY")
SETUP_PATH = BASE_DIR / "setup" / "SETUP.csv"
RESULTS_DIR = BASE_DIR / "results"
# New directory for V7 outputs
ANALYSIS_DIR = BASE_DIR / "analysis_enhanced_v7" # Updated version
VISUAL_DIR = ANALYSIS_DIR / "visualizations"
OUTPUT_DIR = ANALYSIS_DIR / "output_data"
META_REPORT_FILE = ANALYSIS_DIR / "meta_results_summary_report_v7.md" # Updated version

# Create directories if they don't exist
ANALYSIS_DIR.mkdir(exist_ok=True)
VISUAL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# --- Constants ---
COMPARISON_TYPES = [
    'Original vs Degraded', 'Original vs Improved_Degraded', 'Degraded vs Improved_Degraded',
    'Improved_Degraded vs Improved_Original', 'Original vs Improved_Original', 'Degraded vs Improved_Original'
]
VERSION_MAP = {
    'Original vs Degraded': ('Original', 'Degraded'),                             # O vs D
    'Original vs Improved_Degraded': ('Original', 'Improved_Degraded'),           # O vs ID
    'Degraded vs Improved_Degraded': ('Degraded', 'Improved_Degraded'),           # D vs ID
    'Improved_Degraded vs Improved_Original': ('Improved_Degraded', 'Improved_Original'), # ID vs IO
    'Original vs Improved_Original': ('Original', 'Improved_Original'),           # O vs IO
    'Degraded vs Improved_Original': ('Degraded', 'Improved_Original')            # D vs IO
}
# --- Specific Hypotheses for V7 ---
# Format: (Comparison Type, Expected Winner, Expected Loser, Hypothesis Label)
# Note: Winner/Loser mapping depends on VERSION_MAP (Left/Right)
SPECIFIC_HYPOTHESES = [
    # Original Set
    ('Original vs Degraded', 'Original', 'Degraded', 'H1: O > D'),
    ('Original vs Improved_Original', 'Improved_Original', 'Original', 'H2: IO > O'),
    ('Degraded vs Improved_Degraded', 'Improved_Degraded', 'Degraded', 'H3: ID > D'),
    ('Improved_Degraded vs Improved_Original', 'Improved_Original', 'Improved_Degraded', 'H4: IO > ID'),
    ('Original vs Improved_Degraded', 'Improved_Degraded', 'Original', 'H5: ID > O'), # Test Recovery
    ('Degraded vs Improved_Original', 'Improved_Original', 'Degraded', 'H6: IO > D'),  # Sanity check

    # --- Extended Set (Testing Opposite Direction) ---
    # If H1 is ns, is D > O significant? (Checks if degradation *unexpectedly* improved)
    ('Original vs Degraded', 'Degraded', 'Original', 'H7: D > O'),
    # If H2 is ns, is O > IO significant? (Checks if improvement *unexpectedly* degraded)
    ('Original vs Improved_Original', 'Original', 'Improved_Original', 'H8: O > IO'),
    # If H3 is ns, is D > ID significant? (Checks if improvement *unexpectedly* degraded)
    ('Degraded vs Improved_Degraded', 'Degraded', 'Improved_Degraded', 'H9: D > ID'),
     # If H4 is ns, is ID > IO significant? (Checks if improving degraded was *unexpectedly* better)
    ('Improved_Degraded vs Improved_Original', 'Improved_Degraded', 'Improved_Original', 'H10: ID > IO'),
    # If H5 is ns, is O > ID significant? (Checks if recovery *unexpectedly* failed)
    ('Original vs Improved_Degraded', 'Original', 'Improved_Degraded', 'H11: O > ID'),
    # If H6 is ns, is D > IO significant? (Checks if baseline check *unexpectedly* failed)
    ('Degraded vs Improved_Original', 'Degraded', 'Improved_Original', 'H12: D > IO')
]


KEY_IMPROVEMENT_COMPARISONS = ['Original vs Improved_Original', 'Degraded vs Improved_Degraded']
KEY_DEGRADATION_COMPARISON = 'Original vs Degraded'
KEY_BEST_IMPROVEMENT_COMPARISON = 'Improved_Degraded vs Improved_Original'
ALL_RELEVANT_COMPARISONS = list(VERSION_MAP.keys())

DEMOGRAPHIC_COLS_MAP = {
    "How many years did you spend professionally working as a software engineer?": "experience",
    "What is your current main professional activity?": "role",
    "How frequently do you read or write source code?": "code_frequency",
    "How frequently do you use version control tools (Git, GitHub, GitLab, SVN, etc.)?": "vcs_frequency",
    "How frequently do you take part in a code review (author/reviewer/etc.)?": "review_frequency"
}
SATISFACTION_PREFIX_SE = "How satisfied are you with the current capabilities of ChatGPT (and similar) for the following software engineering tasks?"
SATISFACTION_PREFIX_OTHER = "How satisfied are you with the current capabilities of ChatGPT (and similar) for other tasks?"
SE_SATISFACTION_TASKS = [
    "Code summerization/explanation", "Generate code for a given requirement", "Find bugs in existing code",
    "Improve existing code", "Generate tests for existing code", "Improve technical specifications",
    "Improve business specifications"
]
OTHER_SATISFACTION_TASKS = [
    "Write texts", "Summarize content", "Look things up (e.g. search engine replacement)",
    "Learn about something", "Proof read, error correction, etc.", "Create content (emails, presentations, etc.)"
]
SATISFACTION_COL_NAMES_SE = {task: f"{SATISFACTION_PREFIX_SE} [{task}]" for task in SE_SATISFACTION_TASKS}
SATISFACTION_COL_NAMES_OTHER = {task: f"{SATISFACTION_PREFIX_OTHER} [{task}]" for task in OTHER_SATISFACTION_TASKS}
DEMOGRAPHIC_CLEAN_NAMES = {
    "experience": "Years of Professional Experience", "role": "Primary Professional Role",
    "code_frequency": "Code Read/Write Frequency", "vcs_frequency": "Version Control Usage Frequency",
    "review_frequency": "Code Review Participation Frequency",
    "satisfaction_improve_code_category": "Satisfaction: Improving Code (Category)",
    "overall_se_satisfaction_score": "Overall LLM Satisfaction (SE Tasks Avg Score)",
    "overall_se_satisfaction_category": "Overall LLM Satisfaction (SE Tasks Category)",
    "overall_other_satisfaction_score": "Overall LLM Satisfaction (Other Tasks Avg Score)",
    "overall_other_satisfaction_category": "Overall LLM Satisfaction (Other Tasks Category)",
    "experience_numeric": "Experience (Numeric Proxy)"
}

EXPERIENCE_TO_NUMERIC = {'<5 years': 2.5, '>=5 years': 7.5}

# Significance level threshold
ALPHA = 0.05

# --- Constants for Demographic Grouping ---
# Define mapping for frequency to numeric/ordered category might be useful
FREQ_ORDER = ["Never", "A few times a year", "Monthly", "Weekly", "Daily"]
FREQ_MAP_SIMPLIFIED = {
    "Daily": "High", "Weekly": "High",
    "Monthly": "Low", "A few times a year": "Low", "Never": "Low",
    "Infrequent": "Low", # Group infrequent here
    "Infrequent, not strictly required for job": "Low",
    "Unknown/NA": "Unknown" # Keep Unknown separate initially
}

# Mapping for simplified roles
ROLE_GROUP_MAP = {
    'Software Engineer/Developer': 'Developer/Engineer',
    'Professional - Software Engineer, Programmer, Developer': 'Developer/Engineer', # Add raw values if cleaning fails
    'Professional': 'Developer/Engineer', # Map generic 'Professional' likely dev
    'Student': 'Student',
    'Student - PhD program': 'Student', # Group PhD students with students
    'Student - Master program': 'Student', # Group Master students with students
    'part time masters student, part time developer (50/50)': 'Developer/Engineer', # Prioritize Dev role
    'PostDoc': 'Other Academic/Research',
    'Principal': 'Other Tech Roles',
    'Scrum Master': 'Other Tech Roles',
    'Scrum Master former Software Engineer': 'Other Tech Roles', # Map Scrum Master
    'PO/PM': 'Other Tech Roles',
    'Data Engineer': 'Developer/Engineer', # Map Data Engineer
    # Add any other specific roles encountered and map them
}
DEFAULT_ROLE_GROUP = 'Other/Unknown'

# --- Mappings based on PRIMARY Hypotheses (H1-H6) ---
# Mapping from Comparison Type to Primary Hypothesis Label (H1-H6)
COMPARISON_TO_HYPOTHESIS_MAP = {item[0]: item[3] for item in SPECIFIC_HYPOTHESES if item[3].startswith('H') and 1 <= int(re.search(r'H(\d+)', item[3]).group(1)) <= 6}

# Mapping from Comparison Type to Expected Winner (for H1-H6)
COMPARISON_TO_WINNER_MAP = {item[0]: item[1] for item in SPECIFIC_HYPOTHESES if item[3].startswith('H') and 1 <= int(re.search(r'H(\d+)', item[3]).group(1)) <= 6}

# Mapping from Comparison Type to Expected Loser (for H1-H6)
COMPARISON_TO_LOSER_MAP = {item[0]: item[2] for item in SPECIFIC_HYPOTHESES if item[3].startswith('H') and 1 <= int(re.search(r'H(\d+)', item[3]).group(1)) <= 6}


# --- Helper Functions ---
def load_setup(setup_path: Path) -> dict:
    """Loads the Latin Square setup configuration, cleaning column names and validating."""
    logging.info(f"Loading setup configuration from: {setup_path}")
    setup: Dict[int, Dict[str, int]] = {}
    expected_comparison_types = set(COMPARISON_TYPES)

    try:
        with open(setup_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            csv_headers = reader.fieldnames
            if not csv_headers:
                 logging.error(f"Could not read headers from {setup_path}")
                 return {}
            logging.debug(f"Setup CSV Headers: {csv_headers}")

            for row in reader:
                try:
                    survey_num = int(row['Survey'])
                except (ValueError, KeyError, TypeError):
                    logging.warning(f"Skipping row in setup file due to missing or invalid 'Survey' column: {row}")
                    continue

                setup[survey_num] = {}
                missing_comparisons_for_survey = set(expected_comparison_types) # Track missing for this survey

                for raw_col_name in csv_headers:
                    if raw_col_name == 'Survey': continue

                    # Clean column name (handle variations in spacing/prefix)
                    cleaned_col_name = raw_col_name.strip()
                    # More robust cleaning for different possible prefixes/suffixes
                    if cleaned_col_name.startswith('PR (') and cleaned_col_name.endswith(')'):
                        cleaned_col_name = cleaned_col_name[4:-1].strip()
                    elif cleaned_col_name.startswith(' PR (') and cleaned_col_name.endswith(')'):
                         cleaned_col_name = cleaned_col_name[5:-1].strip()
                    # Add other potential variations if needed

                    # Check if the cleaned name is one of the expected comparisons
                    if cleaned_col_name in expected_comparison_types:
                        missing_comparisons_for_survey.discard(cleaned_col_name) # Found it
                        try:
                            pr_assignment_str = row[raw_col_name].strip()
                            if pr_assignment_str: # Check if not empty
                                pr_assignment = int(pr_assignment_str)
                                setup[survey_num][cleaned_col_name] = pr_assignment
                            else:
                                logging.warning(f"Empty PR assignment for survey {survey_num}, comparison '{cleaned_col_name}'. Assigning NaN.")
                                setup[survey_num][cleaned_col_name] = np.nan
                            # logging.debug(f"Survey {survey_num}: Mapping '{cleaned_col_name}' -> PR {pr_assignment}")
                        except (ValueError, KeyError, TypeError):
                            logging.warning(f"Could not parse PR assignment '{row.get(raw_col_name)}' for survey {survey_num}, comparison '{cleaned_col_name}'. Assigning NaN.")
                            setup[survey_num][cleaned_col_name] = np.nan

                if missing_comparisons_for_survey:
                    logging.warning(f"Survey {survey_num} in setup is missing assignments for: {', '.join(missing_comparisons_for_survey)}")

    except FileNotFoundError:
        logging.error(f"Setup file not found at {setup_path}")
        return {}
    except Exception as e:
        logging.error(f"Error reading setup file {setup_path}: {e}")
        return {}

    # Validation after loading all surveys
    all_surveys_valid = True
    all_expected_comps_found_somewhere = set()
    for survey, assignments in setup.items():
        assigned_comps = set(assignments.keys())
        all_expected_comps_found_somewhere.update(assigned_comps)
        nan_assignments = [k for k, v in assignments.items() if pd.isna(v)]
        if nan_assignments:
            logging.warning(f"Survey {survey} has invalid (NaN) PR assignments for: {', '.join(nan_assignments)}")
            all_surveys_valid = False

    overall_missing_comps = expected_comparison_types - all_expected_comps_found_somewhere
    if overall_missing_comps:
        logging.error(f"Critical: The following comparison types defined in COMPARISON_TYPES were not found as valid columns in {setup_path}: {', '.join(overall_missing_comps)}")
        all_surveys_valid = False

    if not setup:
        logging.error("Setup configuration could not be loaded.")
        all_surveys_valid = False
    elif not all_surveys_valid:
        logging.warning("\n !!! Potential errors found in setup file processing. Analysis may be incorrect or incomplete. Review warnings above. !!!\n")
    else:
        logging.info("Setup configuration loaded successfully.")
    return setup

def sanitize_task_name(task_string: str) -> str:
    """Replaces invalid characters with underscores, strips ends, and lowercases."""
    if not isinstance(task_string, str): # Added type check for robustness
        return ""
    # Replace spaces, slashes, commas, dots, parentheses with underscores
    sanitized = re.sub(r'[\s/.,()]+', '_', task_string)
    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')
    return sanitized.lower()

def parse_satisfaction(response: Union[str, float, None]) -> Tuple[Optional[int], str]:
    """Parses satisfaction score (1-5 or None) and category from various input formats."""
    if pd.isna(response):
        return None, "Unknown/NA"
    response_str = str(response).strip().lower()
    if not response_str:
        return None, "Unknown/NA"

    score: Optional[int] = None
    # Handle string formats first
    if "5" in response_str and ("very satisfied" in response_str or "satisfied" in response_str): score = 5 # Broader match
    elif "4" in response_str: score = 4
    elif "3" in response_str: score = 3
    elif "2" in response_str: score = 2
    elif "1" in response_str and ("very unsatisfied" in response_str or "unsatisfied" in response_str): score = 1 # Broader match
    # Handle numeric formats if string parsing failed
    elif score is None:
        try:
            num_response = float(response_str)
            if 1 <= num_response <= 5:
                 score = int(round(num_response)) # Allow floats like '5.0'
            else: score = None # Number out of range
        except ValueError:
            score = None # Cannot parse as number

    category = "Unknown/NA"
    if score is not None:
        if score >= 4: category = "High (4-5)"
        elif score == 3: category = "Neutral (3)"
        else: category = "Low (1-2)"
    elif "don't use" in response_str or "dont use" in response_str: # Added "dont use" variation
        category = "Don't Use"
        score = None # Ensure score is None if they don't use it

    return score, category

def calculate_overall_satisfaction(row: pd.Series, satisfaction_score_cols: List[str]) -> Tuple[Optional[float], str]:
    """Calculates average satisfaction score and category across multiple relevant task scores."""
    scores: List[float] = []
    num_dont_use = 0
    num_tasks_considered = 0

    for col in satisfaction_score_cols:
         # Construct category column name more robustly
         base_col_name = col.replace('_score', '')
         cat_col = f"{base_col_name}_category"
         # cat_col = col.replace('_score', '_category') # Old way

         if col in row.index:
             num_tasks_considered += 1
             score_val = row.get(col)
             # Check if category column actually exists
             category_val = row.get(cat_col) if cat_col in row.index else None

             if pd.notna(score_val):
                 try:
                     scores.append(float(score_val))
                 except (ValueError, TypeError):
                     logging.debug(f"Could not convert score '{score_val}' to float for column '{col}'. Skipping.") # Debug level
             elif category_val == "Don't Use":
                 num_dont_use += 1
             # Optional: Handle "Unknown/NA" category? Currently treated as missing score.

    if num_tasks_considered == 0: return None, "Unknown/NA"

    # Determine category based on scores AND 'Don't Use' counts
    if not scores: # No numeric scores were recorded
        if num_dont_use == num_tasks_considered:
            return None, "All Don't Use"
        else:
            # This case means some tasks had responses other than numeric or "Don't Use" (e.g., only "Unknown/NA")
            return None, "Unknown/NA" # Or perhaps a more specific category like "No Scores Provided"

    # Calculate average score if scores exist
    avg_score = np.mean(scores)
    category = "Unknown/NA"
    if pd.notna(avg_score):
        if avg_score >= 4.0: category = "Overall High (>=4.0)"
        elif avg_score >= 3.0: category = "Overall Neutral (>=3.0 & <4.0)"
        else: category = "Overall Low (<3.0)"
    # else: avg_score is NaN (shouldn't happen if scores list is not empty), category remains "Unknown/NA"

    return avg_score, category


# --- Main Data Processing Function ---
def process_survey_files(results_dir: Path, setup: dict) -> pd.DataFrame:
    """Processes all survey result CSV files, merges with setup, calculates overall satisfaction."""
    logging.info(f"Starting survey file processing from directory: {results_dir}")
    all_responses: List[Dict[str, Any]] = []
    setup_lookup: Dict[int, Dict[int, str]] = defaultdict(dict)
    for survey_num, assignments in setup.items():
        for comp_type, pr_idx in assignments.items():
             # Ensure pr_idx is a valid integer before using as key
             if pd.notna(pr_idx) and isinstance(pr_idx, (int, float)) and not pd.isna(comp_type):
                 try:
                     pr_idx_int = int(pr_idx)
                     setup_lookup[survey_num][pr_idx_int] = str(comp_type)
                 except (ValueError, TypeError):
                      logging.warning(f"Could not convert PR index '{pr_idx}' to int for survey {survey_num}, comp '{comp_type}'. Skipping.")


    all_sat_col_names_map = {**SATISFACTION_COL_NAMES_SE, **SATISFACTION_COL_NAMES_OTHER}

    # Determine max survey number dynamically or set a reasonable upper limit
    max_survey_num = max(setup.keys()) if setup else 6 # Use max from setup if possible
    logging.info(f"Looking for survey files up to {max_survey_num}.csv")

    for survey_num in range(1, max_survey_num + 1): # Iterate based on setup or default
        file_path = results_dir / f"{survey_num}.csv"
        if not file_path.exists():
            # Only warn if this survey number was actually in the setup
            if survey_num in setup:
                 logging.warning(f"Survey file {file_path} not found (but expected from setup). Skipping.")
            else:
                 logging.info(f"Survey file {file_path} not found. Skipping (not in setup).")
            continue

        logging.info(f"Processing file: {file_path}")
        try:
            # Use 'utf-8-sig' to handle potential BOM (Byte Order Mark)
            df_raw = pd.read_csv(file_path, dtype=str, keep_default_na=False, na_values=[''], encoding='utf-8-sig')
            if df_raw.empty:
                 logging.warning(f"Survey file {file_path} is empty. Skipping.")
                 continue
            # Strip whitespace from column names upon reading
            df_raw.columns = [str(col).strip() for col in df_raw.columns]
            header = df_raw.columns.tolist()

            # --- Dynamically Find Columns --- #
            ts_col = header[0] if header else None
            consent_col = None
            if len(header) > 1:
                col2_header = header[1] # Get header of 2nd column (already stripped)
                col2_header_lower = col2_header.lower() # Lowercase for comparison

                # 1. Check for keywords first (case-insensitive)
                if "consent" in col2_header_lower or "agree" in col2_header_lower:
                    consent_col = header[1]
                    logging.info(f"Identified consent column by keyword: '{header[1]}'")
                # 2. NEW CHECK: See if pandas named it 'Unnamed: X' (case-insensitive check)
                elif col2_header_lower.startswith("unnamed:"):
                    consent_col = header[1]
                    logging.info(f"Assuming second column ('{header[1]}') is Consent as it matches 'Unnamed:' pattern.")
                # 3. Check if it's an empty string (might still occur in some edge cases)
                elif col2_header == "":
                    consent_col = header[1]
                    logging.info(f"Assuming second column ('{header[1]}') is Consent as its header is empty.")
                # 4. Add a fallback warning if none of the above match
                else:
                    logging.warning(f"Could not identify consent column. Column 2 header ('{header[1]}') did not match keywords ('consent'/'agree'), 'Unnamed:' pattern, or empty string.")

            # --- Find Preference and Remark columns robustly ---
            pr_pref_cols: Dict[str, int] = {} # Map name to index
            pr_remark_cols: Dict[str, str] = {} # Map pref col name to remark col name
            pref_keyword = 'Which PR timeline was better?'
            remark_keyword = 'If you have additional remarks'
            potential_pref_indices = [i for i, h in enumerate(header) if pref_keyword in h]
            logging.debug(f"Found {len(potential_pref_indices)} potential preference columns in {file_path}.")

            pr_col_counter_in_header = 0
            for idx in potential_pref_indices:
                pref_col_name = header[idx]
                pr_col_counter_in_header += 1 # Count occurrence in header
                pr_pref_cols[pref_col_name] = idx # Store column name and index
                logging.debug(f"  Mapping Preference Column #{pr_col_counter_in_header}: '{pref_col_name}' (Index {idx})")

                # Look for corresponding remark column immediately after
                remark_col_name_found = None
                if idx + 1 < len(header) and remark_keyword in header[idx+1]:
                    remark_col_name_found = header[idx+1]
                    pr_remark_cols[pref_col_name] = remark_col_name_found
                    logging.debug(f"    Found corresponding Remark Column: '{remark_col_name_found}' (Index {idx+1})")
                else:
                     logging.debug(f"    No corresponding Remark Column found immediately after index {idx}.")
                     # Optional: Search more broadly if remarks aren't guaranteed to be adjacent
                     # (See previous version for commented-out broader search logic if needed)

            # Find Demographic and Satisfaction columns
            demo_cols_found = {short: full for full, short in DEMOGRAPHIC_COLS_MAP.items() if full in header}
            found_sat_cols_se = {task: full for task, full in SATISFACTION_COL_NAMES_SE.items() if full in header}
            found_sat_cols_other = {task: full for task, full in SATISFACTION_COL_NAMES_OTHER.items() if full in header}
            combined_found_sat_cols = {**found_sat_cols_se, **found_sat_cols_other}

            # --- Validation ---
            if not ts_col:
                logging.warning(f"Timestamp column (expected first column) not found in {file_path}. Skipping file.")
                continue # Skip this file
            if not consent_col:
                # This log message is now more informative because the logic above explains *why* it failed
                logging.error(f"Consent column could not be reliably identified in {file_path}. Skipping file.")
                continue # Skip this file
            if not pr_pref_cols:
                logging.warning(f"No PR preference columns found containing '{pref_keyword}' in {file_path}. Skipping file.")
                continue # Skip this file
            # Optional: Check number of preference columns found vs expected
            if len(pr_pref_cols) != len(COMPARISON_TYPES):
                 logging.warning(f"Found {len(pr_pref_cols)} preference columns, but expected {len(COMPARISON_TYPES)} based on COMPARISON_TYPES constant in {file_path}. Proceeding, but check setup/data.")
            # --- END OF Validation ---


            # --- Process Each Row (Participant Response) ---
            for row_idx, row in df_raw.iterrows():
                participant_id = f"S{survey_num}_P{row_idx}"
                timestamp = row.get(ts_col)

                # Check consent based on identified column
                consent_response = str(row.get(consent_col, '')).strip().lower()
                # More robust check for consent (e.g., handle variations like "Yes")
                if not ("yes" in consent_response or "agree" in consent_response):
                    logging.debug(f"Participant {participant_id} skipped due to non-consent response: '{consent_response}'")
                    continue

                demographics: Dict[str, str] = {}
                for short_name, full_question in demo_cols_found.items():
                    value = str(row.get(full_question, "Unknown/NA")).strip()
                    # Clean specific roles like 'Student - Master program' -> 'Student'
                    if short_name == 'role' and ' - ' in value:
                         value = value.split(' - ')[0].strip() # Take the part before ' - '
                    # Clean roles like 'Professional - Software Engineer...' -> 'Software Engineer'
                    elif short_name == 'role' and 'Professional - ' in value:
                         value = value.split(' - ')[1].split(',')[0].strip() # Take first role after ' - '
                    elif short_name == 'role':
                         # Handle cases like "part time masters student, part time developer (50/50)" - maybe map to 'Developer' or 'Student'?
                         # For simplicity, let's prioritize 'Developer' or 'Engineer' if present
                         value_lower = value.lower() # Avoid multiple lower() calls
                         if 'developer' in value_lower or 'engineer' in value_lower:
                             # Check if it also contains 'student' to handle the 50/50 case better
                             if 'student' in value_lower:
                                 value = 'part time masters student, part time developer (50/50)' # Keep original for mapping
                             else:
                                 value = 'Software Engineer/Developer' # Consolidate pure devs
                         elif 'professional' in value_lower and 'software' not in value_lower:
                             value = 'Professional' # Capture generic 'Professional'
                         elif 'student' in value_lower:
                              value = 'Student'
                         elif 'postdoc' in value_lower:
                              value = 'PostDoc'
                         elif 'principal' in value_lower:
                               value = 'Principal'
                         elif 'scrum master' in value_lower:
                               # Check if it includes former role info
                               if 'former' in value_lower:
                                    value = 'Scrum Master former Software Engineer'
                               else:
                                    value = 'Scrum Master'
                         elif 'po/pm' in value_lower:
                               value = 'PO/PM'
                         elif 'data engineer' in value_lower: # Added explicit check
                               value = 'Data Engineer'
                         # Add other specific role mappings if needed
                    demographics[short_name] = value if value else "Unknown/NA"
                    logging.debug(f"  {participant_id}: Demographic '{short_name}' raw='{row.get(full_question)}', processed='{value}'")


                satisfactions: Dict[str, Any] = {}
                row_parsed_scores: Dict[str, Optional[int]] = {}
                row_parsed_categories: Dict[str, str] = {}
                for task, full_question in combined_found_sat_cols.items():
                     response = row.get(full_question)
                     score, category = parse_satisfaction(response)
                     # Sanitize task name for column naming
                     sanitized_task = sanitize_task_name(task) # Use helper
                     base_col_name = f"satisfaction_{sanitized_task}"
                     score_col, cat_col = f"{base_col_name}_score", f"{base_col_name}_category"
                     satisfactions[score_col], satisfactions[cat_col] = score, category
                     row_parsed_scores[score_col], row_parsed_categories[cat_col] = score, category
                     logging.debug(f"  {participant_id}: Satisfaction '{task}' raw='{response}', score={score}, category='{category}'")

                # Get Improve Existing Code satisfaction category separately if needed
                improve_code_sanitized = sanitize_task_name("Improve existing code")
                satisfactions['satisfaction_improve_code_category'] = satisfactions.get(f'satisfaction_{improve_code_sanitized}_category', 'Unknown/NA')


                # Calculate overall satisfaction using the parsed data for this row
                temp_series = pd.Series({**row_parsed_scores, **row_parsed_categories}) # Use data just parsed

                # --- Use the helper function here ---
                se_score_cols_for_overall = [
                    f"satisfaction_{sanitize_task_name(task)}_score"
                    for task in SE_SATISFACTION_TASKS if task in found_sat_cols_se
                ]
                overall_se_score, overall_se_cat = calculate_overall_satisfaction(temp_series, se_score_cols_for_overall)
                satisfactions["overall_se_satisfaction_score"], satisfactions["overall_se_satisfaction_category"] = overall_se_score, overall_se_cat
                logging.debug(f"  {participant_id}: Overall SE Sat Score={overall_se_score}, Cat='{overall_se_cat}'")

                # --- And use the helper function here ---
                other_score_cols_for_overall = [
                    f"satisfaction_{sanitize_task_name(task)}_score"
                    for task in OTHER_SATISFACTION_TASKS if task in found_sat_cols_other
                ]
                overall_other_score, overall_other_cat = calculate_overall_satisfaction(temp_series, other_score_cols_for_overall)
                satisfactions["overall_other_satisfaction_score"], satisfactions["overall_other_satisfaction_category"] = overall_other_score, overall_other_cat
                logging.debug(f"  {participant_id}: Overall Other Sat Score={overall_other_score}, Cat='{overall_other_cat}'")


                # Process Preferences based on discovered columns and their order
                # Sort the preference columns by their original index in the CSV header
                sorted_pref_cols = sorted(pr_pref_cols.items(), key=lambda item: item[1])
                logging.debug(f"  {participant_id}: Sorted preference columns by index: {[(name, idx) for name, idx in sorted_pref_cols]}")


                pr_col_counter = 0 # Use a counter based on the sorted order
                for pref_col_name, pref_col_idx in sorted_pref_cols:
                    pr_col_counter += 1 # This should correspond to the PR number (1-6) within the survey
                    pr_assignment_num = pr_col_counter # This is the intended PR number (1-6)
                    logging.debug(f"    Processing PR Assignment #{pr_assignment_num} (Column '{pref_col_name}', Index {pref_col_idx})")


                    # Lookup comparison type using survey number and PR number (assignment)
                    comparison_type = setup_lookup.get(survey_num, {}).get(pr_assignment_num)

                    if not comparison_type:
                         logging.warning(f"Participant {participant_id}: No comparison type found in setup for Survey {survey_num}, PR assignment {pr_assignment_num} (Col '{pref_col_name}'). Skipping this preference.")
                         continue
                    if comparison_type not in VERSION_MAP:
                         logging.warning(f"Participant {participant_id}: Found comparison type '{comparison_type}' (Survey {survey_num}, PR {pr_assignment_num}), but it's not in VERSION_MAP. Skipping.")
                         continue

                    version_left, version_right = VERSION_MAP[comparison_type]
                    preference_raw = str(row.get(pref_col_name, "")).strip().upper()
                    # Retrieve remark using the pref_col_name as the key into pr_remark_cols map
                    remark_col_name = pr_remark_cols.get(pref_col_name) # Get corresponding remark column name
                    remark = str(row.get(remark_col_name, "")).strip() if remark_col_name else ""
                    logging.debug(f"      Pref Raw: '{preference_raw}', Remark Col: '{remark_col_name}', Remark: '{remark[:50]}...'")


                    # Determine preference
                    preference_direction, preferred_version = "Unknown/NA", "Unknown/NA"
                    if not preference_raw:
                         preference_direction, preferred_version = "No Answer", "No Answer"
                    elif "LEFT PR" in preference_raw:
                         preference_direction, preferred_version = "Left", version_left
                    elif "RIGHT PR" in preference_raw:
                         preference_direction, preferred_version = "Right", version_right
                    elif "NEITHER PR" in preference_raw:
                         preference_direction, preferred_version = "Neither", "Neither"
                    else:
                         # Log unexpected preference answers if needed
                         logging.debug(f"Participant {participant_id}, Comp '{comparison_type}': Unexpected preference text '{preference_raw}'")

                    logging.debug(f"      Mapped Pref: Dir='{preference_direction}', Ver='{preferred_version}'")

                    all_responses.append({
                        "participant_id": participant_id, "survey_num": survey_num, "timestamp": timestamp,
                        "pr_assignment_num": pr_assignment_num, # The PR number within the survey (1-6)
                        "comparison_type": comparison_type,     # e.g., 'Original vs Degraded'
                        "version_left": version_left,
                        "version_right": version_right,
                        "preference_direction": preference_direction, # Left, Right, Neither
                        "preferred_version": preferred_version,     # Original, Degraded, Neither, etc.
                        "remark": remark,
                        **demographics,
                        **satisfactions
                    })
        except pd.errors.EmptyDataError: logging.warning(f"Survey file {file_path} is empty after read. Skipping.")
        except KeyError as e:
            logging.error(f"KeyError processing file {file_path}: Column '{e}' not found. Check CSV headers and mappings (DEMOGRAPHIC_COLS_MAP, SATISFACTION_COL_NAMES_*).", exc_info=True)
        except Exception as e:
            logging.error(f"General error processing file {file_path}: {e}", exc_info=True)

    if not all_responses:
        logging.error("No responses were processed successfully from any survey file.")
        return pd.DataFrame()
    logging.info(f"Successfully processed {len(all_responses)} individual comparison responses.")
    # Convert to DataFrame before returning
    df_final = pd.DataFrame(all_responses)

    # --- Add Calculated Columns Needed Later ---
    # Ensure numeric experience column exists for correlation analysis
    if 'experience' in df_final.columns:
        df_final['experience_numeric'] = df_final['experience'].map(EXPERIENCE_TO_NUMERIC)
        logging.info("Created 'experience_numeric' column.")
    else:
         logging.warning("Cannot create 'experience_numeric'. 'experience' column might be missing.")

    # Save intermediate df for debugging if needed
    # df_final.to_csv(OUTPUT_DIR / "DEBUG_processed_data_before_analysis.csv", index=False)
    # logging.debug("Saved intermediate processed data for debugging.")

    return df_final


# --- Analysis Functions ---

# --- MODIFIED FUNCTION: Analyze Overall Alignment (Descriptive) ---
def analyze_overall_alignment_descriptive(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """
    Analyzes overall response alignment with primary hypotheses (H1-H6) for each comparison type (Descriptive).
    Categorizes responses as Aligned (Success), Contradicted (Failure), Neutral, or Other.
    """
    logging.info("--- Starting Overall Alignment Analysis (Descriptive) ---")
    results = []
    valid_comparison_types = df['comparison_type'].unique()

    for comp_type in COMPARISON_TYPES: # Iterate in defined order
        if comp_type not in valid_comparison_types: continue

        subset = df[df['comparison_type'] == comp_type].copy()
        total = len(subset)
        if total == 0: continue

        # Look up the primary hypothesis details (Winner/Loser based on H1-H6)
        primary_hypothesis = COMPARISON_TO_HYPOTHESIS_MAP.get(comp_type)
        expected_winner = COMPARISON_TO_WINNER_MAP.get(comp_type)
        expected_loser = COMPARISON_TO_LOSER_MAP.get(comp_type)

        if not primary_hypothesis or not expected_winner or not expected_loser:
             logging.warning(f"Skipping alignment analysis for '{comp_type}': No primary hypothesis (H1-H6) found.")
             continue

        # Count occurrences of each preference type
        counts = subset['preferred_version'].value_counts()

        # Determine counts based on alignment with the primary hypothesis
        count_aligned = counts.get(expected_winner, 0)
        count_contradicted = counts.get(expected_loser, 0)
        count_neutral = counts.get('Neither', 0)
        # Combine 'Unknown/NA' and 'No Answer' into 'Other'
        count_other = counts.get('Unknown/NA', 0) + counts.get('No Answer', 0)
        # Add any other unexpected values to 'Other' as well (shouldn't happen with good data)
        count_other += total - (count_aligned + count_contradicted + count_neutral + count_other)


        results.append({
            "Comparison Type": comp_type,
            "Primary Hypothesis": primary_hypothesis,
            "Expected Winner": expected_winner,
            "Expected Loser": expected_loser,
            "Total Responses": total,
            "Count Aligned (Success)": count_aligned,
            "Count Contradicted (Failure)": count_contradicted,
            "Count Neutral": count_neutral,
            "Count Other/Unknown": count_other,
            "% Aligned (of Total)": (count_aligned / total * 100) if total > 0 else 0,
            "% Contradicted (of Total)": (count_contradicted / total * 100) if total > 0 else 0,
            "% Neutral (of Total)": (count_neutral / total * 100) if total > 0 else 0,
            "% Other/Unknown (of Total)": (count_other / total * 100) if total > 0 else 0,
        })

    if not results:
        logging.warning("No overall alignment descriptive results generated.")
        return pd.DataFrame()

    results_df = pd.DataFrame(results)
    results_df_formatted = results_df.copy() # Format for display/CSV

    # Format numeric columns for better readability in CSV
    for col in results_df_formatted.columns:
        if '%' in col and pd.api.types.is_numeric_dtype(results_df_formatted[col]):
            results_df_formatted[col] = results_df_formatted[col].map('{:.1f}%'.format)

    output_path = output_dir / "overall_alignment_descriptive_analysis.csv" # New filename
    try:
        results_df_formatted.to_csv(output_path, index=False)
        logging.info(f"Overall alignment descriptive analysis saved to {output_path}")
    except Exception as e:
        logging.error(f"Failed to save overall alignment analysis to {output_path}: {e}")

    return results_df # Return numeric df for plotting/further use


# --- MODIFIED FUNCTION: Plot Overall Alignment (Descriptive) ---
def plot_overall_alignment_descriptive(results_df: pd.DataFrame, visual_dir: Path):
    """Plots overall alignment percentages and counts based on primary hypotheses."""
    logging.info("--- Generating Overall Alignment Plots (Descriptive) ---")
    if results_df.empty:
        logging.warning("Skipping overall alignment plots: Input DataFrame is empty.")
        return

    plot_data_pct = []
    plot_data_count = []

    # Use 'Comparison Type' as index for easier lookup
    try:
         results_df_indexed = results_df.set_index('Comparison Type')
    except KeyError:
         logging.error("Plotting error: 'Comparison Type' column not found in results dataframe.")
         return

    # Prepare data for plotting
    for comp_type in COMPARISON_TYPES: # Iterate in defined order
        if comp_type not in results_df_indexed.index: continue

        row = results_df_indexed.loc[comp_type]
        primary_hyp = row.get("Primary Hypothesis", "") # Get hypothesis label

        # Get counts and percentages using the new column names
        count_aligned = row.get("Count Aligned (Success)", 0)
        count_contradicted = row.get("Count Contradicted (Failure)", 0)
        count_neutral = row.get("Count Neutral", 0)
        count_other = row.get("Count Other/Unknown", 0)

        pct_aligned = row.get("% Aligned (of Total)", 0.0)
        pct_contradicted = row.get("% Contradicted (of Total)", 0.0)
        pct_neutral = row.get("% Neutral (of Total)", 0.0)
        pct_other = row.get("% Other/Unknown (of Total)", 0.0)

        # Data for Count Plot
        if all(isinstance(c, (int, float, np.number)) for c in [count_aligned, count_contradicted, count_neutral, count_other]):
             plot_data_count.extend([
                 {"Comparison": comp_type, "Hypothesis": primary_hyp, "Alignment Status": "Aligned (Success)", "Count": count_aligned},
                 {"Comparison": comp_type, "Hypothesis": primary_hyp, "Alignment Status": "Contradicted (Failure)", "Count": count_contradicted},
                 {"Comparison": comp_type, "Hypothesis": primary_hyp, "Alignment Status": "Neutral", "Count": count_neutral},
                 {"Comparison": comp_type, "Hypothesis": primary_hyp, "Alignment Status": "Other/Unknown", "Count": count_other}
             ])
        # Data for Percentage Plot
        if all(isinstance(p, (int, float, np.number)) for p in [pct_aligned, pct_contradicted, pct_neutral, pct_other]):
             plot_data_pct.extend([
                 {"Comparison": comp_type, "Hypothesis": primary_hyp, "Alignment Status": "Aligned (Success)", "Percentage": pct_aligned},
                 {"Comparison": comp_type, "Hypothesis": primary_hyp, "Alignment Status": "Contradicted (Failure)", "Percentage": pct_contradicted},
                 {"Comparison": comp_type, "Hypothesis": primary_hyp, "Alignment Status": "Neutral", "Percentage": pct_neutral},
                 {"Comparison": comp_type, "Hypothesis": primary_hyp, "Alignment Status": "Other/Unknown", "Percentage": pct_other}
             ])


    if not plot_data_pct or not plot_data_count:
        logging.warning("Skipping overall alignment plots: No plottable data extracted.")
        return

    plot_df_pct = pd.DataFrame(plot_data_pct)
    plot_df_count = pd.DataFrame(plot_data_count)

    if plot_df_pct.empty or plot_df_count.empty:
        logging.warning("Skipping overall alignment plots: Empty DataFrames after processing.")
        return

    # Define a consistent color palette and order for Alignment Status
    alignment_order = ['Aligned (Success)', 'Contradicted (Failure)', 'Neutral', 'Other/Unknown']
    # Choose a palette suitable for categorical data (e.g., 'viridis', 'magma', 'plasma', 'Set2')
    palette = sns.color_palette("viridis", n_colors=len(alignment_order))
    color_map = dict(zip(alignment_order, palette))

    # Create combined Comparison + Hypothesis labels for x-axis ticks
    comp_hyp_labels = {row['Comparison']: f"{row['Comparison']}\n({row['Hypothesis']})"
                      for _, row in plot_df_pct.drop_duplicates(subset=['Comparison']).iterrows()}
    ordered_tick_labels = [comp_hyp_labels.get(comp, comp) for comp in COMPARISON_TYPES if comp in comp_hyp_labels]


    try: # Plot 1: Percentages
        plt.figure(figsize=(14, 9)) # Increased height for longer x-labels
        ax_pct = sns.barplot(data=plot_df_pct, x="Comparison", y="Percentage", hue="Alignment Status",
                             palette=color_map, order=COMPARISON_TYPES, hue_order=alignment_order) # Use defined order

        plt.title('Overall Alignment with Primary Hypotheses (Percentages)', fontsize=16)
        plt.xlabel('Comparison Type (Primary Hypothesis)', fontsize=12) # Updated label
        plt.ylabel('Percentage of Total Responses (%)', fontsize=12) # Updated label
        ax_pct.set_xticklabels(ordered_tick_labels, rotation=45, ha='right', fontsize=10) # Set combined labels
        plt.yticks(fontsize=10)
        plt.legend(title='Alignment Status', bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.ylim(0, 100)
        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for legend

        # No significance markers here anymore

        plot_path_pct = visual_dir / "overall_alignment_comparison_percentage.png" # New filename
        plt.savefig(plot_path_pct, dpi=300, bbox_inches='tight')
        logging.info(f"Overall alignment plot (percentage) saved to {plot_path_pct}")
        plt.close()

    except Exception as e:
        logging.error(f"Failed to generate percentage alignment plot: {e}", exc_info=True)
        plt.close() # Ensure plot is closed on error

    try: # Plot 2: Counts
        plt.figure(figsize=(14, 9)) # Increased height
        ax_count = sns.barplot(data=plot_df_count, x="Comparison", y="Count", hue="Alignment Status",
                               palette=color_map, order=COMPARISON_TYPES, hue_order=alignment_order) # Use defined order

        plt.title('Overall Alignment with Primary Hypotheses (Counts)', fontsize=16)
        plt.xlabel('Comparison Type (Primary Hypothesis)', fontsize=12) # Updated label
        plt.ylabel('Number of Responses', fontsize=12)
        ax_count.set_xticklabels(ordered_tick_labels, rotation=45, ha='right', fontsize=10) # Set combined labels
        plt.yticks(fontsize=10)

        # Adjust y-limit dynamically
        max_y_count = plot_df_count['Count'].max()
        plt.ylim(0, max(10, max_y_count * 1.15) if pd.notna(max_y_count) else 10)

        plt.legend(title='Alignment Status', bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for legend

        plot_path_count = visual_dir / "overall_alignment_comparison_count.png" # New filename
        plt.savefig(plot_path_count, dpi=300, bbox_inches='tight')
        logging.info(f"Overall alignment plot (count) saved to {plot_path_count}")
        plt.close()

    except Exception as e:
        logging.error(f"Failed to generate count alignment plot: {e}", exc_info=True)
        plt.close() # Ensure plot is closed on error


def analyze_demographics(df: pd.DataFrame, output_dir: Path, visual_dir: Path):
    """Analyzes and visualizes demographic distributions, including overall satisfaction categories."""
    logging.info("--- Starting Demographic Analysis ---")
    # Define columns to analyze, including the calculated overall satisfaction
    demo_cols_to_analyze = list(DEMOGRAPHIC_COLS_MAP.values()) + [
        'overall_se_satisfaction_category', 'overall_other_satisfaction_category'
    ]
    # Ensure participant_id exists for uniqueness check
    if 'participant_id' not in df.columns:
        logging.error("'participant_id' column not found. Cannot analyze unique participants.")
        return

    # Get unique participants - crucial for demographics
    unique_participants_df = df.drop_duplicates(subset=['participant_id'])
    num_participants = len(unique_participants_df)
    logging.info(f"Analyzing demographics for {num_participants} unique participants.")

    if num_participants == 0:
        logging.warning("No unique participants found. Skipping demographic analysis.")
        return

    summary_dfs: Dict[str, pd.DataFrame] = {} # To store summary dataframes for Excel export

    for col in demo_cols_to_analyze:
        if col not in unique_participants_df.columns:
            logging.warning(f"Demographic column '{col}' not found in unique participant data. Skipping.")
            continue

        # Calculate counts and percentages, handling potential NaNs gracefully
        counts = unique_participants_df[col].fillna("Unknown/NA").value_counts()
        percentages = (counts / num_participants * 100)
        summary = pd.DataFrame({'Count': counts, 'Percentage': percentages})
        summary_dfs[col] = summary # Store for Excel

        # Get a cleaner name for logging/plotting
        clean_col_name = DEMOGRAPHIC_CLEAN_NAMES.get(col, col.replace("_", " ").title())
        logging.info(f"\nDistribution for {clean_col_name}:")
        try:
            # Print summary table to console
            print(summary.round(1).to_string())
        except Exception as e:
            logging.error(f"Error printing summary for {clean_col_name}: {e}")

        # --- Plotting ---
        try:
            plt.figure(figsize=(10, 7)) # Adjust figure size as needed

            # Determine a sensible order for categories in the plot
            plot_order = counts.index # Default order
            if col == 'experience':
                # Specific order for experience levels
                cat_order = ['<5 years', '>=5 years', 'Unknown/NA']
                plot_order = [c for c in cat_order if c in counts.index] + \
                             [c for c in sorted(counts.index) if c not in cat_order]
            elif 'satisfaction_category' in col:
                # Define a logical order for satisfaction categories
                cat_order = [
                    "Low (1-2)", "Overall Low (<3.0)",
                    "Neutral (3)", "Overall Neutral (>=3.0 & <4.0)",
                    "High (4-5)", "Overall High (>=4.0)",
                    "Don't Use", "All Don't Use", "Unknown/NA"
                ]
                # Apply order if categories exist in data
                plot_order = [cat for cat in cat_order if cat in counts.index]
                # Add any unexpected categories at the end, sorted alphabetically
                plot_order.extend(sorted([cat for cat in counts.index if cat not in plot_order]))
            elif 'frequency' in col:
                 # Define a logical order for frequency categories
                 cat_order = ["Never", "Infrequent", "A few times a year", "Monthly", "Weekly", "Daily", "Unknown/NA"]
                 # Apply order if categories exist
                 plot_order = [cat for cat in cat_order if cat in counts.index]
                 plot_order.extend(sorted([cat for cat in counts.index if cat not in plot_order]))
            elif col == 'role': # Use the simplified roles defined in ROLE_GROUP_MAP keys/values
                 # Use the *values* from ROLE_GROUP_MAP and DEFAULT_ROLE_GROUP for order
                 role_values_ordered = ['Student', 'Developer/Engineer', 'Other Academic/Research', 'Other Tech Roles', DEFAULT_ROLE_GROUP]
                 # Use the *original* role names found in the data (counts.index)
                 # but order them based on their mapping to the simplified groups
                 # Create a temporary mapping from original role to simplified group for sorting
                 role_to_group_temp = {role: ROLE_GROUP_MAP.get(role, DEFAULT_ROLE_GROUP) for role in counts.index}
                 # Sort the original roles (counts.index) based on the order of their simplified group
                 plot_order = sorted(counts.index, key=lambda r: role_values_ordered.index(role_to_group_temp[r]) if role_to_group_temp[r] in role_values_ordered else 99)

            else:
                # Default: sort alphabetically, but maybe put Unknown/NA last
                known_cats = sorted([c for c in counts.index if c != "Unknown/NA"])
                plot_order = known_cats + (["Unknown/NA"] if "Unknown/NA" in counts.index else [])


            # Create the bar plot - Use hue mapped to x to allow palette application per bar
            sns.barplot(x=percentages.index, y=percentages.values, order=plot_order,
                        hue=percentages.index, hue_order=plot_order, # Map hue to x, ensure order matches
                        palette="viridis", legend=False) # Use a palette, disable redundant legend

            plt.title(f'Distribution of Participants by {clean_col_name}', fontsize=14)
            plt.xlabel(clean_col_name, fontsize=12)
            plt.ylabel('Percentage of Participants (%)', fontsize=12)
            plt.xticks(rotation=45, ha='right', fontsize=10) # Rotate labels for readability
            plt.yticks(fontsize=10)

            # Adjust y-limit dynamically
            max_pct = percentages.max()
            plt.ylim(0, max(100, max_pct + 10) if pd.notna(max_pct) else 100) # Ensure scale up to 100% or slightly above max

            plt.tight_layout() # Adjust layout to prevent labels overlapping

            # Save the plot
            plot_filename = f"demographic_{col}_distribution.png"
            plot_path = visual_dir / plot_filename
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logging.info(f"Demographic plot saved to {plot_path}")
            plt.close() # Close the plot figure

        except Exception as e:
            logging.error(f"Failed to generate plot for demographic '{col}': {e}", exc_info=True)
            plt.close() # Ensure plot is closed on error

    # --- Save Summary Data to Excel ---
    if summary_dfs: # Check if there's anything to save
        excel_path = output_dir / "demographic_summary.xlsx"
        try:
            with pd.ExcelWriter(excel_path) as writer:
                for col, df_summary in summary_dfs.items():
                    # Create a sheet name, sanitize it for Excel limitations (e.g., length, invalid chars)
                    clean_sheet_name_base = DEMOGRAPHIC_CLEAN_NAMES.get(col, col)
                    # Remove characters invalid for Excel sheet names
                    invalid_chars = r'[\\*?:/\[\]]' # Corrected regex pattern for invalid Excel chars
                    sanitized_sheet_name = re.sub(invalid_chars, '', clean_sheet_name_base) # Use re.sub to remove them
                    # Truncate sheet name to Excel's limit (usually 31 chars)
                    sanitized_sheet_name = sanitized_sheet_name[:31]

                    # Write the summary dataframe to a sheet
                    df_summary.round(1).to_excel(writer, sheet_name=sanitized_sheet_name)
            logging.info(f"Demographic summaries saved to Excel file: {excel_path}")
        except Exception as e:
            logging.error(f"Error writing demographic summary to Excel file {excel_path}: {e}")
    else:
        logging.warning("No demographic summaries were generated to save.")


# --- V7 Function: Analyze Specific Pairwise Hypotheses (Includes Confidence Interval) ---
def analyze_pairwise_hypotheses(df: pd.DataFrame, hypotheses: List[Tuple[str, str, str, str]], output_dir: Path) -> pd.DataFrame:
    """
    Analyzes specific pairwise directional hypotheses using binomial tests, including confidence intervals.

    Args:
        df: DataFrame containing all processed responses.
        hypotheses: List of tuples, each: (comparison_type, winner_version, loser_version, hypothesis_label)
        output_dir: Path to save the results CSV.

    Returns:
        DataFrame summarizing the hypothesis test results.
    """
    logging.info("--- Starting Specific Pairwise Hypothesis Testing (with CIs) ---") # Updated log
    results = []

    required_cols = ['comparison_type', 'preferred_version', 'version_left', 'version_right']
    if not all(col in df.columns for col in required_cols):
        logging.error(f"Missing one or more required columns for hypothesis testing: {required_cols}. Skipping.")
        return pd.DataFrame()

    for comp_type, expected_winner, expected_loser, hypothesis_label in hypotheses:
        logging.info(f"Testing Hypothesis: {hypothesis_label} ({expected_winner} > {expected_loser} for {comp_type})")

        subset = df[df['comparison_type'] == comp_type].copy()
        total_responses = len(subset)

        # Default values for results dictionary
        result_data = {
            "Hypothesis": hypothesis_label, "Comparison": comp_type,
            "Expected Winner": expected_winner, "Expected Loser": expected_loser,
            "N Total": total_responses, "N Winner": 0, "N Loser": 0, "N Neither": 0, "N Other": 0,
            "% Winner (of W+L)": np.nan, "% Loser (of W+L)": np.nan, "% Neither (of Total)": np.nan,
            "P-value (Binomial)": np.nan, "Significance": "N/A",
            "Conf. Interval Low (%)": np.nan, "Conf. Interval High (%)": np.nan # New CI fields
        }

        if total_responses == 0:
            logging.warning(f"  No data found for comparison type '{comp_type}'. Skipping hypothesis '{hypothesis_label}'.")
            result_data["Significance"] = "N/A (No Data)"
            results.append(result_data)
            continue

        counts = subset['preferred_version'].value_counts()
        n_winner = counts.get(expected_winner, 0)
        n_loser = counts.get(expected_loser, 0)
        n_neither = counts.get('Neither', 0)
        # Recalculate n_other to include Unknown/NA, No Answer etc.
        n_other = total_responses - (n_winner + n_loser + n_neither) # Correct calculation

        # Update counts in results
        result_data.update({
            "N Winner": n_winner, "N Loser": n_loser,
            "N Neither": n_neither, "N Other": n_other
        })

        n_comparative = n_winner + n_loser
        p_value = np.nan
        ci_low, ci_high = np.nan, np.nan
        test_result_str = "N/A"

        if n_comparative < 5:
             test_result_str = f"N/A (Low Count: {n_comparative})"
             logging.warning(f"  Skipping binomial test for '{hypothesis_label}': Insufficient comparative responses ({n_comparative} < 5).")
        else:
            try:
                k_int = int(n_winner)
                n_int = int(n_comparative)
                if k_int < 0 or n_int < 0: raise ValueError("Counts cannot be negative")
                if k_int > n_int: raise ValueError("Winner count cannot exceed total comparative count")

                # Perform test (one-sided for p-value, two-sided for CI by default)
                # Note: We calculate the one-sided p-value for 'greater'
                # The confidence interval from binomtest is typically two-sided by default.
                binom_result_greater = binomtest(k=k_int, n=n_int, p=0.5, alternative='greater')
                p_value = binom_result_greater.pvalue

                # Get the default (two-sided) confidence interval for the proportion
                # Confidence level is 1 - ALPHA (e.g., 0.95 for ALPHA=0.05)
                ci = binom_result_greater.proportion_ci(confidence_level=1-ALPHA, method='exact') # Using 'exact' (Clopper-Pearson)
                ci_low = ci.low * 100  # Convert to percentage
                ci_high = ci.high * 100 # Convert to percentage

                if pd.isna(p_value): test_result_str = "Binom Test Error (NaN p-value)"
                elif p_value < 0.001: test_result_str = f"p < 0.001 (***)"
                elif p_value < 0.01: test_result_str = f"p = {p_value:.3f} (**)"
                elif p_value < ALPHA: test_result_str = f"p = {p_value:.3f} (*)"
                else: test_result_str = f"p = {p_value:.3f} (ns)"

            except ValueError as e:
                 test_result_str = f"Binom Test Error: {e}"
                 logging.error(f"  Binomial test failed for '{hypothesis_label}'. Input: k={n_winner}, n={n_comparative}. Error: {e}")
                 p_value, ci_low, ci_high = np.nan, np.nan, np.nan # Ensure reset on error
            except Exception as e:
                 test_result_str = f"Binom Test Error: Unknown"
                 logging.error(f"  An unexpected error occurred during binomial test for '{hypothesis_label}'. Error: {e}", exc_info=True)
                 p_value, ci_low, ci_high = np.nan, np.nan, np.nan # Ensure reset on error

        pct_winner_of_wl = (n_winner / n_comparative * 100) if n_comparative > 0 else np.nan
        pct_loser_of_wl = (n_loser / n_comparative * 100) if n_comparative > 0 else np.nan
        pct_neither_of_total = (n_neither / total_responses * 100) if total_responses > 0 else np.nan

        # Update final results data
        result_data.update({
            "% Winner (of W+L)": pct_winner_of_wl,
            "% Loser (of W+L)": pct_loser_of_wl,
            "% Neither (of Total)": pct_neither_of_total,
            "P-value (Binomial)": p_value,
            "Significance": test_result_str,
            "Conf. Interval Low (%)": ci_low,
            "Conf. Interval High (%)": ci_high
        })
        results.append(result_data)
        logging.info(f"  Result: N_Win={n_winner}, N_Los={n_loser}, N_Nei={n_neither}, N_Oth={n_other} -> {test_result_str} (CI: [{ci_low:.1f}%, {ci_high:.1f}%])") # Log CI


    if not results:
        logging.warning("No hypothesis test results generated.")
        return pd.DataFrame()

    results_df = pd.DataFrame(results)

    # --- Save raw results (Including CI) ---
    output_path_raw = output_dir / "specific_hypothesis_tests_raw.csv"
    try:
        # Save with more precision for CI maybe? 3f should be fine for percentages.
        results_df.to_csv(output_path_raw, index=False, float_format="%.3f")
        logging.info(f"Specific hypothesis test results (raw with CI) saved to {output_path_raw}")
    except Exception as e:
        logging.error(f"Failed to save hypothesis test results (raw) to {output_path_raw}: {e}")

    # --- Create formatted version ---
    results_df_formatted = results_df.copy()
    # Format p-value
    results_df_formatted['P-value (Formatted)'] = results_df_formatted['P-value (Binomial)'].apply(
        lambda p: '< 0.001' if pd.notna(p) and p < 0.001 else f'{p:.3f}' if pd.notna(p) else 'N/A'
    )
    # Format CI
    results_df_formatted['Conf. Interval (%)'] = results_df_formatted.apply(
        lambda row: f"[{row['Conf. Interval Low (%)']:.1f}, {row['Conf. Interval High (%)']:.1f}]"
        if pd.notna(row['Conf. Interval Low (%)']) and pd.notna(row['Conf. Interval High (%)'])
        else "N/A", axis=1
    )
    # Format percentages
    pct_cols = [col for col in results_df_formatted.columns if '%' in col and 'Conf.' not in col] # Exclude raw CI cols
    for col in pct_cols:
         results_df_formatted[col] = results_df_formatted[col].map('{:.1f}%'.format, na_action='ignore')


    # Select and reorder columns for formatted output
    formatted_cols = [
        "Hypothesis", "Comparison", "Expected Winner", "Expected Loser",
        "N Winner", "N Loser", "N Neither", "N Other", "N Total", # Added N Other
        "% Winner (of W+L)", # Formatted percentage
        "Conf. Interval (%)", # Formatted CI
        "P-value (Formatted)", # Formatted p-value
        "Significance"
    ]
    # Ensure all columns exist before selecting
    formatted_cols_present = [col for col in formatted_cols if col in results_df_formatted.columns]
    results_df_formatted_final = results_df_formatted[formatted_cols_present]


    output_path_formatted = output_dir / "specific_hypothesis_tests_formatted.csv"
    try:
        results_df_formatted_final.to_csv(output_path_formatted, index=False)
        logging.info(f"Specific hypothesis test results (formatted with CI) saved to {output_path_formatted}")
    except Exception as e:
        logging.error(f"Failed to save hypothesis test results (formatted) to {output_path_formatted}: {e}")

    return results_df # Return the raw numeric df (including raw CI cols)


def analyze_preference_by_demographic(df: pd.DataFrame, demographic_col: str, output_dir: Path, visual_dir: Path):
    """Analyzes preference broken down by demographic, generates plots and Excel."""
    # Check if the demographic column exists in the DataFrame
    if demographic_col not in df.columns:
        logging.warning(f"Demographic column '{demographic_col}' not found in DataFrame. Skipping breakdown.")
        return

    clean_demographic_name = DEMOGRAPHIC_CLEAN_NAMES.get(demographic_col, demographic_col.replace("_", " ").title())
    logging.info(f"\n--- Starting Preference Analysis by {clean_demographic_name} ---")

    # Filter data: Ensure comparison type is valid, demographic value is not NA/Unknown
    # Also filter out 'No Answer' or 'Unknown/NA' preferences for cleaner breakdown analysis? Optional.
    df_filtered = df[
        df['comparison_type'].isin(ALL_RELEVANT_COMPARISONS) &
        df[demographic_col].notna() &
        (df[demographic_col] != "Unknown/NA") # Exclude 'Unknown/NA' demographic category
        # Optional: & (~df['preferred_version'].isin(["Unknown/NA", "No Answer"])) # Exclude non-preferences
    ].copy() # Use .copy() to avoid SettingWithCopyWarning

    if df_filtered.empty:
        logging.warning(f"No valid data found for breakdown by '{clean_demographic_name}' after filtering. Skipping.")
        return

    all_stats_list = [] # To collect stats DataFrames for each comparison type
    plot_order = None # To store the desired order for the demographic categories in plots

    # --- Data Preparation Loop (for each comparison type) ---
    for comp_type in ALL_RELEVANT_COMPARISONS: # Iterate through all defined comparisons
        # Get the subset of data for this comparison type
        subset = df_filtered[df_filtered['comparison_type'] == comp_type].copy() # Use .copy()
        if subset.empty:
             continue # Skip if no data for this comparison

        # Get Left and Right version names
        version_left, version_right = VERSION_MAP[comp_type]

        # Group by the demographic column and the preferred version, then count occurrences
        # Use .unstack() to turn preferred versions into columns, fill missing with 0
        grouped = subset.groupby([demographic_col, 'preferred_version']).size().unstack(fill_value=0)

        # Ensure all expected preference columns exist (Left, Right, Neither, Unknown/NA, No Answer)
        expected_prefs = [version_left, version_right, 'Neither', 'Unknown/NA', 'No Answer']
        for pref in expected_prefs:
            if pref not in grouped.columns:
                grouped[pref] = 0 # Add missing columns with 0 count

        # Calculate total responses for each demographic group
        grouped['Total Responses'] = grouped[expected_prefs].sum(axis=1) # Sum counts across preference columns

        # Calculate percentages, handling potential division by zero
        with np.errstate(divide='ignore', invalid='ignore'): # Suppress warnings for 0/0 cases
            grouped[f"% Prefer '{version_left}'"] = (grouped[version_left] / grouped['Total Responses'] * 100).fillna(0)
            grouped[f"% Prefer '{version_right}'"] = (grouped[version_right] / grouped['Total Responses'] * 100).fillna(0)
            grouped["% Prefer 'Neither'"] = (grouped['Neither'] / grouped['Total Responses'] * 100).fillna(0)
            # Combine Unknown and No Answer for percentage calculation
            grouped["% Unknown/No Answer"] = ((grouped['Unknown/NA'] + grouped['No Answer']) / grouped['Total Responses'] * 100).fillna(0)

        # Rename count columns for clarity
        grouped = grouped.rename(columns={
            version_left: f"Count Prefer '{version_left}'",
            version_right: f"Count Prefer '{version_right}'",
            'Neither': "Count Prefer 'Neither'",
            'Unknown/NA': "Count Unknown/NA", # Keep separate for raw counts if needed
            'No Answer': "Count No Answer"
        })

        # Add comparison type and reset index to make demographic_col a regular column
        grouped['comparison_type'] = comp_type
        grouped = grouped.reset_index()
        all_stats_list.append(grouped)

        # --- Determine Plot Order (only once) ---
        if plot_order is None and not grouped.empty:
            unique_cats = grouped[demographic_col].unique()
            # Apply specific ordering logic based on the demographic column type
            if demographic_col == 'experience':
                 cat_order = ['<5 years', '>=5 years'] # Define desired order
                 plot_order = [c for c in cat_order if c in unique_cats] + \
                              [c for c in sorted(unique_cats) if c not in cat_order] # Add others alphabetically
            elif 'satisfaction_category' in demographic_col:
                 cat_order = [
                    "Low (1-2)", "Overall Low (<3.0)", "Neutral (3)",
                    "Overall Neutral (>=3.0 & <4.0)", "High (4-5)", "Overall High (>=4.0)",
                    "Don't Use", "All Don't Use" # Note: Unknown/NA was filtered out earlier
                 ]
                 plot_order = [cat for cat in cat_order if cat in unique_cats]
                 plot_order.extend(sorted([cat for cat in unique_cats if cat not in plot_order]))
            elif 'frequency' in demographic_col:
                 cat_order = ["Never", "Infrequent", "A few times a year", "Monthly", "Weekly", "Daily"]
                 plot_order = [cat for cat in cat_order if cat in unique_cats]
                 plot_order.extend(sorted([cat for cat in unique_cats if cat not in plot_order]))
            elif demographic_col == 'role':
                 # Use the *values* from ROLE_GROUP_MAP and DEFAULT_ROLE_GROUP for order
                 role_values_ordered = ['Student', 'Developer/Engineer', 'Other Academic/Research', 'Other Tech Roles', DEFAULT_ROLE_GROUP]
                 # Use the *original* role names found in the unique_cats
                 # but order them based on their mapping to the simplified groups
                 role_to_group_temp = {role: ROLE_GROUP_MAP.get(role, DEFAULT_ROLE_GROUP) for role in unique_cats}
                 # Sort the original roles (unique_cats) based on the order of their simplified group
                 plot_order = sorted(unique_cats, key=lambda r: role_values_ordered.index(role_to_group_temp[r]) if role_to_group_temp[r] in role_values_ordered else 99)

            else: # Default for other categoricals
                 plot_order = sorted(unique_cats)

    # --- Combine and Save Stats ---
    if not all_stats_list:
        logging.warning(f"No statistics generated for breakdown by '{clean_demographic_name}'. Skipping save and plots.")
        return

    combined_stats_df = pd.concat(all_stats_list, ignore_index=True)

    # Save combined stats to an Excel file with sheets for each comparison type
    excel_path = output_dir / f"preference_by_{demographic_col}.xlsx"
    try:
        with pd.ExcelWriter(excel_path) as writer:
            for comp_type in ALL_RELEVANT_COMPARISONS:
                 # Filter data for the sheet
                 df_sheet = combined_stats_df[combined_stats_df['comparison_type'] == comp_type].copy() # .copy() is good practice

                 if not df_sheet.empty:
                     # Define and select columns for the Excel sheet output
                     count_cols = [col for col in df_sheet.columns if 'Count Prefer' in col or col in ['Count Unknown/NA', 'Count No Answer']]
                     percent_cols = [col for col in df_sheet.columns if '%' in col]
                     output_cols = [demographic_col, 'Total Responses'] + count_cols + percent_cols
                     # Ensure selected columns actually exist
                     output_cols_present = [c for c in output_cols if c in df_sheet.columns]

                     # Apply categorical sorting if plot_order was determined
                     if plot_order:
                         try:
                              # Set the demographic column as a categorical type with the determined order
                              df_sheet[demographic_col] = pd.Categorical(df_sheet[demographic_col], categories=plot_order, ordered=True)
                              # Sort the DataFrame based on this categorical column
                              df_sheet = df_sheet.sort_values(demographic_col)
                         except KeyError:
                              logging.warning(f"KeyError: Could not apply category order for {demographic_col} in comp {comp_type} (maybe empty sheet or unexpected data?). Using default sort.")
                         except ValueError as e:
                              logging.warning(f"ValueError: Could not apply category order for {demographic_col} in comp {comp_type}: {e}. Using default sort.")
                         except Exception as e:
                              logging.warning(f"Unexpected error applying category order for {demographic_col} in comp {comp_type}: {e}. Using default sort.")


                     # Sanitize sheet name for Excel
                     sheet_name_base = comp_type.replace(" vs ", "_vs_")
                     invalid_chars = r'[\\*?:/\[\]]' # Corrected regex pattern
                     sanitized_sheet_name = re.sub(invalid_chars, '', sheet_name_base)[:31] # Use re.sub

                     # Write the selected and sorted data to the sheet
                     df_sheet[output_cols_present].to_excel(writer, sheet_name=sanitized_sheet_name, index=False, float_format="%.1f")

        logging.info(f"Preference breakdown by {clean_demographic_name} saved to {excel_path}")
    except Exception as e:
        logging.error(f"Error writing preference breakdown to Excel for {clean_demographic_name}: {e}", exc_info=True)


    # --- Plotting Loop (for each comparison type) ---
    for comp_type in ALL_RELEVANT_COMPARISONS:
        # Filter data for the current plot
        plot_data = combined_stats_df[combined_stats_df['comparison_type'] == comp_type]
        if plot_data.empty:
            continue # Skip if no data

        version_left, version_right = VERSION_MAP[comp_type]
        # Define percentage column names dynamically
        pct_col_right = f"% Prefer '{version_right}'"
        pct_col_left = f"% Prefer '{version_left}'"
        pct_col_neither = "% Prefer 'Neither'"

        # --- Plot 1: Percentage Preferring Right Version ---
        if pct_col_right in plot_data.columns:
            try:
                plt.figure(figsize=(10, 6))
                # Use hue mapped to x for palette application
                sns.barplot(data=plot_data, x=demographic_col, y=pct_col_right,
                            order=plot_order,        # Use determined category order for x-axis
                            hue=demographic_col,     # Map hue to x variable
                            hue_order=plot_order,    # Ensure hue order matches x order
                            palette="mako",          # Apply color palette
                            legend=False)            # Disable redundant legend

                plt.title(f"Preference for '{version_right}' in '{comp_type}'\nby {clean_demographic_name}", fontsize=14)
                plt.xlabel(clean_demographic_name, fontsize=12)
                plt.ylabel(f"% Prefer '{version_right}'", fontsize=12)
                plt.xticks(rotation=45, ha='right', fontsize=10)
                plt.yticks(fontsize=10)
                plt.ylim(0, 100) # Percentage scale
                plt.tight_layout()

                # Save the plot
                plot_filename = f"pref_pct_RIGHT_{comp_type.replace(' ', '_').replace('/', '')}_by_{demographic_col}.png"
                plot_path = visual_dir / plot_filename
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close() # Close figure
            except Exception as e:
                logging.error(f"Failed to generate plot '% Prefer Right' for {comp_type} by {demographic_col}: {e}", exc_info=True)
                plt.close()

        # --- Plot 2: Stacked Bar Chart (Left vs Neither vs Right) ---
        required_cols_stack = [pct_col_left, pct_col_right, pct_col_neither]
        if not all(col in plot_data.columns for col in required_cols_stack):
             logging.warning(f"Skipping stacked plot for {comp_type} by {demographic_col}: Missing one of {required_cols_stack}")
             continue # Skip if necessary columns for stacking aren't present

        try:
            # Prepare data for stacking: Set demographic as index, select percentage columns
            # Use .copy() when setting index or manipulating slices if needed
            stack_data = plot_data.set_index(demographic_col)[required_cols_stack].copy()

            # Apply sorting based on plot_order if available
            if plot_order:
                 try:
                      stack_data = stack_data.reindex(plot_order).dropna(how='all') # Reindex and remove rows that might become all NaN
                 except Exception as e: # Catch broader errors like non-unique index if duplicates exist
                      logging.warning(f"Could not reindex stacked data for {demographic_col} in comp {comp_type}: {e}. Using default index order.")

            if stack_data.empty: continue # Skip if no data after potential reindexing/filtering

            # Rename columns for a cleaner legend
            stack_data = stack_data.rename(columns={
                pct_col_left: f"'{version_left}' (Left)",
                pct_col_right: f"'{version_right}' (Right)",
                pct_col_neither: "'Neither'"
            })
            # Define stack order (e.g., Left, Neither, Right)
            stack_col_order = [f"'{version_left}' (Left)", "'Neither'", f"'{version_right}' (Right)"]
            # Ensure stack order columns exist
            stack_col_order = [col for col in stack_col_order if col in stack_data.columns]

            # Create the stacked bar plot
            plt.figure(figsize=(12, 7)) # Create a new figure explicitly
            ax = stack_data[stack_col_order].plot(
                kind='bar',
                stacked=True,
                figsize=(12, 7), # Can set size here or on plt.figure
                width=0.8,
                color=sns.color_palette("viridis", n_colors=len(stack_col_order)), # Color by stack segment
                ax=plt.gca() # Use the current axes
            )

            plt.title(f"Preference Distribution: '{comp_type}'\nby {clean_demographic_name}", fontsize=16)
            plt.xlabel(clean_demographic_name, fontsize=12)
            plt.ylabel('Percentage of Analyzed Responses (%)', fontsize=12)
            plt.xticks(rotation=45, ha='right', fontsize=10)
            plt.yticks(np.arange(0, 101, 10), fontsize=10) # Ticks from 0 to 100
            plt.ylim(0, 100) # Ensure y-axis is 0-100
            plt.legend(title='Preference', bbox_to_anchor=(1.02, 1), loc='upper left') # Place legend outside
            plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for legend

            # Save the plot
            plot_filename = f"pref_dist_{comp_type.replace(' ', '_').replace('/', '')}_by_{demographic_col}_stacked.png"
            plot_path = visual_dir / plot_filename
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close() # Close figure
        except Exception as e:
            logging.error(f"Failed to generate stacked distribution plot for {comp_type} by {demographic_col}: {e}", exc_info=True)
            plt.close()


def analyze_correlations(df: pd.DataFrame, output_dir: Path, visual_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calculates correlations (Spearman) and Kruskal-Wallis, generates plots. (Exploratory)"""
    logging.info("--- Starting Preference Correlation Analysis (Exploratory) ---")
    corr_results, kruskal_results, plot_data_list = [], [], []

    # Check if necessary 'experience_numeric' column exists
    if 'experience_numeric' not in df.columns:
         logging.warning("Column 'experience_numeric' not found. Correlation analysis involving experience will be skipped.")
         # Optionally, attempt to create it here if 'experience' exists, but it should be created in process_survey_files
         if 'experience' in df.columns:
              logging.info("Attempting to create 'experience_numeric' within analyze_correlations...")
              df = df.copy() # Ensure we work on a copy if modifying
              df['experience_numeric'] = df['experience'].map(EXPERIENCE_TO_NUMERIC)
         else:
              logging.error("'experience' column also missing. Cannot perform experience correlation.")


    # Focus correlations on key comparisons where differences are expected or interesting
    comparisons_for_corr = KEY_IMPROVEMENT_COMPARISONS + [KEY_BEST_IMPROVEMENT_COMPARISON, KEY_DEGRADATION_COMPARISON]
    logging.info(f"Analyzing correlations for specific comparisons: {comparisons_for_corr}")


    # Define variables for analysis
    analysis_vars = {
        'SE Satisfaction Score': 'overall_se_satisfaction_score',
        'Other Satisfaction Score': 'overall_other_satisfaction_score',
        'Experience (Numeric)': 'experience_numeric'
        }
    # Filter for variables actually present in the DataFrame
    valid_analysis_vars = {name: col for name, col in analysis_vars.items() if col in df.columns}

    if not valid_analysis_vars:
        logging.warning("No valid numeric columns (Satisfaction Scores, Experience) found for correlation analysis. Skipping.")
        return pd.DataFrame(), pd.DataFrame() # Return empty dataframes

    logging.info(f"Variables available for correlation: {list(valid_analysis_vars.keys())}")

    for comp_type in comparisons_for_corr:
        if comp_type not in df['comparison_type'].unique():
             logging.debug(f"Skipping correlation for {comp_type}: No data found.")
             continue

        subset = df[df['comparison_type'] == comp_type].copy()
        version_left, version_right = VERSION_MAP[comp_type]

        # Map preference to a numeric scale (-1 for Left, 0 for Neither, 1 for Right)
        def map_preference_numeric(p_version: str) -> Optional[int]:
            if p_version == version_right: return 1   # Prefer Right (often the 'improved' or hypothesized 'winner')
            if p_version == version_left: return -1  # Prefer Left (often the 'original' or hypothesized 'loser')
            if p_version == 'Neither': return 0
            return np.nan # Map Unknown/No Answer to NaN

        # Apply mapping using .loc to avoid warnings
        subset['preference_numeric'] = subset['preferred_version'].apply(map_preference_numeric)

        # Filter out rows where preference mapping resulted in NaN
        subset_valid_pref = subset.dropna(subset=['preference_numeric']).copy()

        if subset_valid_pref.empty:
             logging.debug(f"Skipping correlation for {comp_type}: No valid numeric preferences after mapping.")
             continue

        logging.info(f"\n--- Correlations & Tests for Comparison: {comp_type} ---")
        logging.info(f"(Score: 1='{version_right}', -1='{version_left}', 0='Neither')")

        for var_name, var_col in valid_analysis_vars.items():
            # Check if the analysis variable column exists in the current subset
            if var_col not in subset_valid_pref.columns:
                logging.warning(f"  Variable column '{var_col}' for '{var_name}' not found in subset for {comp_type}. Skipping.")
                continue

            # Prepare data: drop rows where either preference_numeric or the analysis variable is NaN
            analysis_data = subset_valid_pref[['preference_numeric', var_col]].dropna()

            if len(analysis_data) < 5: # Need a minimum number of data points
                logging.warning(f"  Skipping {var_name}: Insufficient valid data points ({len(analysis_data)} < 5).")
                continue

            pref_numeric_vals = analysis_data['preference_numeric']
            analysis_var_vals = analysis_data[var_col]

            # 1. Spearman Correlation (Rank correlation - suitable for ordinal/non-normal data)
            try:
                spearman_corr, spearman_p = spearmanr(pref_numeric_vals, analysis_var_vals)
                corr_sig = "" # Significance marker string
                if pd.isna(spearman_p):
                     corr_sig = "(p=NaN)"
                elif spearman_p < 0.001: corr_sig = "(p < 0.001 ***)"
                elif spearman_p < 0.01: corr_sig = f"(p={spearman_p:.3f} **)"
                elif spearman_p < ALPHA: corr_sig = f"(p={spearman_p:.3f} *)" # Use ALPHA
                else: corr_sig = f"(p={spearman_p:.3f} ns)" # Not significant

                logging.info(f"  - Spearman ({var_name}): rho = {spearman_corr:.3f} {corr_sig}")
                corr_results.append({
                    'Comparison': comp_type, 'Variable': var_name, 'Variable Column': var_col,
                    'Method': 'Spearman', 'Correlation (rho)': spearman_corr, 'P-value': spearman_p,
                    'N': len(analysis_data)
                    })
            except Exception as e:
                logging.error(f"  - Spearman calculation failed for {var_name}: {e}", exc_info=True)

            # 2. Kruskal-Wallis Test (Non-parametric ANOVA alternative)
            # Compare the distribution of the analysis variable across the preference groups (-1, 0, 1)
            groups_for_test, group_names = [], []
            min_group_size = 3 # Minimum observations per group for the test

            # Extract data for each preference group
            group_left = analysis_data[analysis_data['preference_numeric'] == -1][var_col]
            group_neither = analysis_data[analysis_data['preference_numeric'] == 0][var_col]
            group_right = analysis_data[analysis_data['preference_numeric'] == 1][var_col]

            # Include groups in the test only if they meet the minimum size requirement
            if len(group_left) >= min_group_size:
                 groups_for_test.append(group_left); group_names.append(f"'{version_left}' (N={len(group_left)})")
            if len(group_neither) >= min_group_size:
                 groups_for_test.append(group_neither); group_names.append(f"'Neither' (N={len(group_neither)})")
            if len(group_right) >= min_group_size:
                 groups_for_test.append(group_right); group_names.append(f"'{version_right}' (N={len(group_right)})")

            # Perform Kruskal-Wallis test if at least 2 valid groups exist
            if len(groups_for_test) >= 2:
                try:
                    # Check for constant data within groups (Kruskal requires variance)
                    if any(g.nunique() <= 1 for g in groups_for_test if len(g)>0):
                        kw_h, kw_p = np.nan, np.nan
                        kw_sig_str = "Skipped (constant data)"
                        logging.warning(f"  - Kruskal ({var_name}): Skipped due to constant data in one or more groups.")
                    else:
                        kw_h, kw_p = kruskal(*groups_for_test) # Unpack groups into arguments
                        kw_sig_str = "" # Significance marker string
                        if pd.isna(kw_p):
                              kw_sig_str = "H=NaN, p=NaN"
                        elif kw_p < 0.001:
                              kw_sig_str = f"H={kw_h:.2f}, p < 0.001 ***"
                        elif kw_p < 0.01:
                              kw_sig_str = f"H={kw_h:.2f}, p={kw_p:.3f} **"
                        elif kw_p < ALPHA: # Use ALPHA
                              kw_sig_str = f"H={kw_h:.2f}, p={kw_p:.3f} *"
                        else:
                              kw_sig_str = f"H={kw_h:.2f}, p={kw_p:.3f} ns" # Not significant

                    logging.info(f"  - Kruskal-Wallis ({var_name}): {kw_sig_str}")
                    # Extract just the significance part for the results table if not skipped
                    kw_sig = kw_sig_str.split(',')[-1].strip().split(' ')[-1] if kw_sig_str != "Skipped (constant data)" and "=" in kw_sig_str else kw_sig_str

                    if kw_sig_str != "Skipped (constant data)": # Log group details unless skipped
                         logging.debug(f"    Groups Compared: {group_names}")
                         kruskal_results.append({
                              'Comparison': comp_type, 'Variable': var_name, 'Variable Column': var_col,
                              'H-statistic': kw_h, 'P-value': kw_p, 'Significance': kw_sig,
                              'Groups Compared': len(groups_for_test), 'Group Details': "; ".join(group_names)
                              })
                except Exception as e:
                    logging.error(f"  - Kruskal-Wallis test failed for {var_name}: {e}", exc_info=True)
                    # Optionally add a failed result row
                    kruskal_results.append({
                         'Comparison': comp_type, 'Variable': var_name, 'Variable Column': var_col,
                         'H-statistic': np.nan, 'P-value': np.nan, 'Significance': 'Test Error',
                         'Groups Compared': len(groups_for_test), 'Group Details': "; ".join(group_names)
                     })
            else:
                logging.warning(f"  - Kruskal ({var_name}): Skipped. Need >= 2 groups with N >= {min_group_size}.")


            # --- Prepare data for Box Plots ---
            # Use the subset with valid numeric preferences, drop NaNs for the current analysis variable
            plot_subset = subset_valid_pref.dropna(subset=[var_col]).copy()
            if plot_subset.empty: continue

            # Map numeric preference back to a category for plotting labels
            def map_pref_category(p_num: float) -> str:
                if p_num == 1: return f"Prefer '{version_right}'"
                if p_num == -1: return f"Prefer '{version_left}'"
                if p_num == 0: return "Prefer 'Neither'"
                return "Other" # Should not happen if NaNs were dropped

            plot_subset['preference_category'] = plot_subset['preference_numeric'].apply(map_pref_category)

            # Define the desired order for preference categories on the plot's x-axis
            pref_category_order = [f"Prefer '{version_left}'", "Prefer 'Neither'", f"Prefer '{version_right}'"]
            # Ensure only categories present in the data are included in the order
            pref_category_order = [cat for cat in pref_category_order if cat in plot_subset['preference_category'].unique()]

            # Store plot information
            plot_data_list.append({
                'comparison': comp_type,
                'variable_name': var_name,
                'variable_col': var_col,
                'data': plot_subset,
                'pref_categories_order': pref_category_order
            })

    # --- Generate Consolidated Box Plots (One figure per analysis variable) ---
    logging.info("\n--- Generating Correlation Box Plots ---")
    if not plot_data_list:
        logging.warning("No data available for generating correlation box plots.")
    else:
        # Group plot data by the analysis variable (e.g., all plots for 'SE Satisfaction Score' together)
        plots_by_variable: Dict[str, List[Dict]] = defaultdict(list)
        for p_info in plot_data_list:
            plots_by_variable[p_info['variable_name']].append(p_info)

        # Create a separate figure for each variable
        for var_name, plot_infos in plots_by_variable.items():
            num_plots = len(plot_infos)
            if num_plots == 0: continue

            # Determine subplot layout (e.g., 2 columns)
            ncols_box = min(num_plots, 2)
            nrows_box = math.ceil(num_plots / ncols_box)

            fig, axes = plt.subplots(nrows=nrows_box, ncols=ncols_box,
                                     figsize=(ncols_box * 7, nrows_box * 5.5), # Adjust size as needed
                                     sharey=True, # Share y-axis for easier comparison
                                     squeeze=False) # Always return 2D array for axes

            fig.suptitle(f"Distribution of {var_name}\nby Preference Group across Comparisons", fontsize=16, y=1.02) # Main title
            axes_flat = axes.flatten() # Flatten axes array for easy iteration
            plot_success_count = 0

            # Iterate through the plot information for this variable
            for i, plot_info in enumerate(plot_infos):
                if i >= len(axes_flat): break # Safety break if more plots than subplots

                ax = axes_flat[i] # Current subplot axes
                comp_title = plot_info['comparison']
                data_to_plot = plot_info['data']
                var_col_to_plot = plot_info['variable_col']
                category_order = plot_info['pref_categories_order']

                # Check if data is valid for plotting
                if data_to_plot.empty or var_col_to_plot not in data_to_plot.columns or 'preference_category' not in data_to_plot.columns:
                    ax.set_title(f"{comp_title}\n(No data)", fontsize=11, color='grey')
                    ax.set_visible(False) # Hide empty subplots
                    continue

                try:
                    # Create boxplot - Use hue mapped to x for palette
                    sns.boxplot(data=data_to_plot, x='preference_category', y=var_col_to_plot,
                                order=category_order,        # Use defined category order for x-axis
                                hue='preference_category',   # Map hue to x variable
                                hue_order=category_order,    # Ensure hue order matches x order
                                palette="viridis",           # Apply color palette
                                showfliers=False,            # Hide outliers for cleaner look
                                legend=False,                # Disable redundant legend
                                ax=ax)                       # Plot on the current axes

                    # Overlay stripplot for individual data points
                    sns.stripplot(data=data_to_plot, x='preference_category', y=var_col_to_plot,
                                  order=category_order, ax=ax,
                                  color=".3", size=3.5, alpha=0.6, jitter=0.15) # Jitter points horizontally

                    # Set titles and labels
                    ax.set_title(comp_title, fontsize=13)
                    ax.set_xlabel("Preference Group", fontsize=11)
                    ax.set_ylabel(var_name if i % ncols_box == 0 else "", fontsize=11) # Y-label only on left column
                    ax.tick_params(axis='x', rotation=10, labelsize=10) # Rotate x-labels slightly
                    ax.tick_params(axis='y', labelsize=10)

                    def get_sig_marker_report_local(p_value: Optional[float]) -> str:
                        """Gets significance marker based on p-value and ALPHA."""
                        if pd.isna(p_value): return ""
                        if p_value < 0.001: return "***"
                        if p_value < 0.01: return "**"
                        if p_value < ALPHA: return "*" # Use ALPHA constant
                        return "(ns)" # non-significant

                    # Annotate with Kruskal-Wallis result if available and significant (optional)
                    kw_result = next((item for item in kruskal_results if item['Comparison'] == comp_title and item['Variable'] == var_name), None)
                    if kw_result and pd.notna(kw_result.get('P-value')) and kw_result.get('P-value') < ALPHA:
                         p_val_kw = kw_result['P-value']
                         p_text = f"p={p_val_kw:.3f}"
                         if p_val_kw < 0.001: p_text = "p<0.001"

                         sig_marker_kw = get_sig_marker_report_local(p_val_kw).replace('(ns)','') # Get *, **, ***
                         # Position annotation below the plot area
                         ax.annotate(f'K-W test: {p_text}{sig_marker_kw}', xy=(0.5, -0.25), xycoords='axes fraction',
                                     ha='center', va='center', fontsize=9, color='darkred')

                    plot_success_count += 1
                except Exception as plot_err:
                    logging.error(f"Error creating box plot for {comp_title} - {var_name}: {plot_err}", exc_info=True)
                    ax.set_title(f"{comp_title}\n(Plot Error)", fontsize=11, color='red') # Indicate error on plot

            # Hide any unused subplots
            for j in range(i + 1, len(axes_flat)):
                axes_flat[j].set_visible(False)

            # Save the figure if any plots were successful
            if plot_success_count > 0:
                 plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout to accommodate title
                 # Use the actual variable column name for the filename
                 safe_var_col_name = valid_analysis_vars.get(var_name, f"unknown_var_{var_name}").replace("_", "-")
                 plot_filename = f"corr_boxplot_{safe_var_col_name}.png"
                 plot_path = visual_dir / plot_filename
                 try:
                     plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                     logging.info(f"Correlation Boxplot saved: {plot_path}")
                 except Exception as save_err:
                     logging.error(f"Failed to save correlation boxplot {plot_path}: {save_err}", exc_info=True)
            else:
                 logging.warning(f"No successful boxplots generated for '{var_name}'. Figure not saved.")

            plt.close(fig) # Close the figure after saving/processing

    # --- Save Correlation and Test Results to CSV ---
    corr_df, kruskal_df = pd.DataFrame(), pd.DataFrame() # Initialize empty dataframes

    if corr_results:
        corr_df = pd.DataFrame(corr_results)
        corr_output_path = output_dir / "preference_correlations_spearman.csv"
        try:
            corr_df.to_csv(corr_output_path, index=False, float_format="%.3f")
            logging.info(f"Spearman correlation results saved to {corr_output_path}")
        except Exception as e:
            logging.error(f"Failed to save Spearman results: {e}")

    if kruskal_results:
        kruskal_df = pd.DataFrame(kruskal_results)
        kruskal_output_path = output_dir / "preference_correlations_kruskal_wallis.csv"
        try:
            kruskal_df.to_csv(kruskal_output_path, index=False, float_format="%.3f")
            logging.info(f"Kruskal-Wallis test results saved to {kruskal_output_path}")
        except Exception as e:
            logging.error(f"Failed to save Kruskal-Wallis results: {e}")

    return corr_df, kruskal_df # Return the results DataFrames


def extract_qualitative_remarks(df: pd.DataFrame, output_dir: Path):
    """Extracts non-empty remarks with context, including demographic groups if available."""
    logging.info("--- Extracting Qualitative Remarks ---")
    # Check if 'remark' column exists
    if 'remark' not in df.columns:
        logging.warning("Column 'remark' not found in DataFrame. Skipping qualitative extraction.")
        return

    # Filter for rows where 'remark' is not NaN and is not an empty or whitespace-only string
    remarks_df = df[df['remark'].notna() & (df['remark'].str.strip() != '')].copy()

    if remarks_df.empty:
        logging.info("No non-empty qualitative remarks found.")
        return

    # --- Recalculate/Ensure Group Columns for Context ---
    # It's safer to recalculate groups here based on the original df data
    # to ensure they are present for the context, even if analysis failed earlier.
    df_context = df.copy() # Work on a copy for adding context cols

    # a) Experience Group
    if 'experience' in df_context.columns:
        df_context['experience_group'] = df_context['experience'].fillna('Unknown')

    # b) Role Group
    if 'role' in df_context.columns:
        # Ensure ROLE_GROUP_MAP covers all unique roles found, or map others to default
        # Log unmapped roles before applying map
        unique_roles_in_data = df_context['role'].unique()
        unmapped = [r for r in unique_roles_in_data if r not in ROLE_GROUP_MAP]
        if unmapped:
            logging.warning(f"Unmapped roles found during remark extraction, will be assigned to '{DEFAULT_ROLE_GROUP}': {list(unmapped)}")
        df_context['role_group'] = df_context['role'].map(ROLE_GROUP_MAP).fillna(DEFAULT_ROLE_GROUP)


    # c) Code/Review Engagement Group
    if 'code_frequency' in df_context.columns and 'review_frequency' in df_context.columns:
        df_context['code_freq_simple'] = df_context['code_frequency'].map(FREQ_MAP_SIMPLIFIED).fillna('Unknown')
        df_context['review_freq_simple'] = df_context['review_frequency'].map(FREQ_MAP_SIMPLIFIED).fillna('Unknown')
        def assign_engagement_context(row):
            code_f = row['code_freq_simple']
            rev_f = row['review_freq_simple']
            if code_f == 'Unknown' or rev_f == 'Unknown': return 'Unknown'
            if code_f == 'High' and rev_f == 'High': return 'High Engagement'
            if code_f == 'Low' and rev_f == 'Low': return 'Low Engagement'
            return 'Medium Engagement'
        df_context['code_engagement_group'] = df_context.apply(assign_engagement_context, axis=1)

    # d) SE LLM Satisfaction Group
    if 'overall_se_satisfaction_category' in df_context.columns and 'overall_se_satisfaction_score' in df_context.columns:
        def assign_satisfaction_group_context(row):
            cat = row.get('overall_se_satisfaction_category', 'Unknown/NA')
            score = row.get('overall_se_satisfaction_score')
            if cat in ['All Don\'t Use', 'Unknown/NA']: return 'Non-User/Unknown'
            if pd.isna(score):
                 if cat == 'Overall High (>=4.0)': return 'High Sat'
                 if cat in ['Overall Neutral (>=3.0 & <4.0)', 'Overall Low (<3.0)']: return 'Med/Low Sat'
                 return 'Non-User/Unknown'
            if score >= 4.0: return 'High Sat'
            elif score < 4.0: return 'Med/Low Sat'
            else: return 'Non-User/Unknown'
        df_context['se_satisfaction_group'] = df_context.apply(assign_satisfaction_group_context, axis=1)

    # Define context columns to include with the remarks
    group_cols_to_include = [
        'experience_group', 'role_group', 'code_engagement_group', 'se_satisfaction_group'
    ]
    existing_group_cols = [col for col in group_cols_to_include if col in df_context.columns]

    base_context_cols = [
        'participant_id', 'survey_num', 'pr_assignment_num', 'comparison_type',
        'version_left', 'version_right',
        'preferred_version', 'preference_direction',
        'experience', 'role', # Original demographics
        'overall_se_satisfaction_category', # Key satisfaction category
        # Add other original demographics if desired
        'remark' # The remark itself
    ]

    # Combine base context, existing groups, and remark (ensuring remark is last)
    final_context_cols = [col for col in base_context_cols if col in df_context.columns and col != 'remark']
    final_context_cols.extend([col for col in existing_group_cols if col not in final_context_cols]) # Add group cols
    if 'remark' in df_context.columns:
        final_context_cols.append('remark')

    # Select only the desired context columns from the filtered remarks dataframe
    # Merge the context (including groups) based on index or participant_id+comparison?
    # Easiest is to just select the columns from the enriched df_context, filtered by index of remarks_df
    remarks_final_df = df_context.loc[remarks_df.index, final_context_cols].copy()


    output_path = output_dir / "qualitative_remarks_context.csv"
    try:
        # Save to CSV with UTF-8 encoding
        remarks_final_df.to_csv(output_path, index=False, encoding='utf-8-sig') # Use utf-8-sig for better Excel compat
        logging.info(f"Qualitative remarks ({len(remarks_final_df)}) extracted with context to {output_path}")
    except Exception as e:
        logging.error(f"Failed to save qualitative remarks to {output_path}: {e}")


# --- V7 Function: Analyze Hypothesis Alignment by Demographic Group (Corrected) ---
def analyze_hypothesis_alignment_by_demographic_group(df: pd.DataFrame, output_dir: Path) -> Optional[pd.DataFrame]:
    """
    Analyzes how different demographic groups align with core hypotheses (H1-H6).

    Args:
        df: DataFrame containing all processed responses, including demographics.
        output_dir: Path to save the results CSV.

    Returns:
        DataFrame summarizing alignment percentages per group and hypothesis, or None if error.
    """
    logging.info("--- Starting Hypothesis Alignment Analysis by Demographic Group ---")

    # Define required columns (original demographics + participant ID + comparison info)
    required_cols = [
        'participant_id', 'comparison_type', 'preferred_version',
        'experience', 'role', 'code_frequency', 'review_frequency',
        'overall_se_satisfaction_score', 'overall_se_satisfaction_category'
    ]
    # Check if all required columns for grouping exist
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logging.error(f"Missing one or more required columns for demographic group analysis: {missing_cols}. Skipping.")
        return None

    # --- 1. Prepare Data & Create Groups ---
    df_processed = df.copy()
    unique_participant_ids = df_processed['participant_id'].unique()
    df_unique_participants = df_processed[df_processed['participant_id'].isin(unique_participant_ids)].drop_duplicates(subset=['participant_id']).copy()

    # a) Experience Group
    df_unique_participants['experience_group'] = df_unique_participants['experience'].fillna('Unknown')
    # b) Role Group
    # Log unmapped roles before applying map
    unique_roles_in_data = df_unique_participants['role'].unique()
    unmapped = [r for r in unique_roles_in_data if r not in ROLE_GROUP_MAP]
    if unmapped:
         logging.warning(f"Unmapped roles found, assigned to '{DEFAULT_ROLE_GROUP}': {list(unmapped)}")
    df_unique_participants['role_group'] = df_unique_participants['role'].map(ROLE_GROUP_MAP).fillna(DEFAULT_ROLE_GROUP)

    # c) Code/Review Engagement Group
    df_unique_participants['code_freq_simple'] = df_unique_participants['code_frequency'].map(FREQ_MAP_SIMPLIFIED).fillna('Unknown')
    df_unique_participants['review_freq_simple'] = df_unique_participants['review_frequency'].map(FREQ_MAP_SIMPLIFIED).fillna('Unknown')
    def assign_engagement(row):
        code_f = row['code_freq_simple']
        rev_f = row['review_freq_simple']
        if code_f == 'Unknown' or rev_f == 'Unknown': return 'Unknown'
        if code_f == 'High' and rev_f == 'High': return 'High Engagement'
        if code_f == 'Low' and rev_f == 'Low': return 'Low Engagement'
        return 'Medium Engagement'
    df_unique_participants['code_engagement_group'] = df_unique_participants.apply(assign_engagement, axis=1)
    # d) SE LLM Satisfaction Group
    def assign_satisfaction_group(row):
        cat = row.get('overall_se_satisfaction_category', 'Unknown/NA')
        score = row.get('overall_se_satisfaction_score')
        if cat in ['All Don\'t Use', 'Unknown/NA']: return 'Non-User/Unknown'
        if pd.isna(score):
             if cat == 'Overall High (>=4.0)': return 'High Sat'
             if cat in ['Overall Neutral (>=3.0 & <4.0)', 'Overall Low (<3.0)']: return 'Med/Low Sat'
             return 'Non-User/Unknown'
        if score >= 4.0: return 'High Sat'
        elif score < 4.0: return 'Med/Low Sat'
        else: return 'Non-User/Unknown'
    df_unique_participants['se_satisfaction_group'] = df_unique_participants.apply(assign_satisfaction_group, axis=1)

    group_cols = ['participant_id', 'experience_group', 'role_group', 'code_engagement_group', 'se_satisfaction_group']
    df_participant_groups = df_unique_participants[group_cols]
    df_merged = pd.merge(df_processed, df_participant_groups, on='participant_id', how='left')

    # --- 2. Calculate Alignment Status per Response ---
    def get_alignment_status(row):
        comp_type = row['comparison_type']
        preferred = row['preferred_version']
        expected_winner = COMPARISON_TO_WINNER_MAP.get(comp_type)
        expected_loser = COMPARISON_TO_LOSER_MAP.get(comp_type)
        if not expected_winner or not expected_loser: return 'N/A'
        if preferred == expected_winner: return 'Aligned'
        elif preferred == expected_loser: return 'Contradicted'
        elif preferred == 'Neither': return 'Neutral'
        elif preferred in ['Unknown/NA', 'No Answer']: return 'Other' # Capture these explicitly
        else: return 'Other' # Catch-all for unexpected values
    df_merged['alignment_status'] = df_merged.apply(get_alignment_status, axis=1)
    df_merged['hypothesis_label'] = df_merged['comparison_type'].map(COMPARISON_TO_HYPOTHESIS_MAP)

    df_analysis = df_merged.dropna(subset=['hypothesis_label']) # Only keep rows relevant to H1-H6
    df_analysis = df_analysis[df_analysis['alignment_status'] != 'N/A'] # Should be redundant now
    if df_analysis.empty:
         logging.warning("No data remaining after calculating alignment status for H1-H6. Cannot perform group analysis.")
         return None
    # Debug: Check labels after filtering
    logging.debug(f"Unique hypothesis labels in df_analysis (used for grouping): {df_analysis['hypothesis_label'].unique()}")

    # --- 3. Aggregate Alignment by Group and Hypothesis ---
    results_list = []
    demographic_groupings_cols = ['experience_group', 'role_group', 'code_engagement_group', 'se_satisfaction_group']

    for demo_group_col in demographic_groupings_cols:
        if demo_group_col not in df_analysis.columns:
             logging.warning(f"Demographic group column '{demo_group_col}' not found in merged data. Skipping this grouping.")
             continue
        try:
             # Group by the actual category values within the column, plus hypothesis and status
             grouped = df_analysis.groupby([demo_group_col, 'hypothesis_label', 'alignment_status']).size().unstack(fill_value=0)
        except Exception as e:
             logging.error(f"Error grouping data for '{demo_group_col}': {e}. Skipping this grouping.")
             continue

        # Ensure all expected alignment columns exist after unstacking
        for status_col in ['Aligned', 'Contradicted', 'Neutral', 'Other']:
            if status_col not in grouped.columns:
                grouped[status_col] = 0

        grouped['N_Aligned_Contradicted'] = grouped['Aligned'] + grouped['Contradicted']
        # Use all non-'Other' columns for the denominator (Total Valid for Alignment % calc)
        grouped['N_Total_Valid'] = grouped['Aligned'] + grouped['Contradicted'] + grouped['Neutral']

        with np.errstate(divide='ignore', invalid='ignore'):
            # Calculate % Aligned based on Aligned + Contradicted (excluding Neither)
            grouped['Percent_Aligned_of_AC'] = (grouped['Aligned'] / grouped['N_Aligned_Contradicted'] * 100)
            grouped['Percent_Aligned_of_AC'] = grouped['Percent_Aligned_of_AC'].replace([np.inf, -np.inf], np.nan).fillna(np.nan) # Ensure NaNs for 0/0

        grouped = grouped.reset_index()

        if demo_group_col in grouped.columns:
             # Rename the specific demographic column (e.g., 'experience_group') to 'Demographic_Category'
             grouped = grouped.rename(columns={demo_group_col: 'Demographic_Category'})
        else:
             logging.warning(f"Column '{demo_group_col}' not found after grouping/resetting index. Skipping rename for this group.")
             continue # Skip appending if rename failed

        grouped['Demographic_Grouping_Type'] = demo_group_col
        results_list.append(grouped)

    if not results_list:
        logging.warning("No results generated from demographic group alignment analysis.")
        return None

    # Combine results from all groupings
    final_results_df = pd.concat(results_list, ignore_index=True, sort=False) # Assign the result of concat

    # Define simplified output columns using the generic category name
    output_cols_ordered = [
        'Demographic_Grouping_Type', # e.g., 'experience_group'
        'Demographic_Category',     # e.g., '<5 years', '>=5 years'
        'hypothesis_label',
        'Aligned', 'Contradicted', 'Neutral', 'Other',
        'N_Aligned_Contradicted', 'N_Total_Valid', 'Percent_Aligned_of_AC'
    ]

    # Select only columns that actually exist in the final dataframe
    final_output_cols_present = [col for col in output_cols_ordered if col in final_results_df.columns]

    # Ensure the core columns needed for plotting are present before selection
    essential_plot_cols = ['Demographic_Grouping_Type', 'Demographic_Category', 'hypothesis_label', 'Percent_Aligned_of_AC']
    if not all(col in final_results_df.columns for col in essential_plot_cols):
         logging.error(f"Essential columns for plotting are missing from concatenated results. Columns: {final_results_df.columns}")
         return None

    # Reorder columns using the present list
    final_results_df_ordered = final_results_df[final_output_cols_present].copy() # Use copy to avoid SettingWithCopyWarning

    # Save to CSV
    output_path = output_dir / "hypothesis_alignment_by_demographic_group.csv"
    try:
        # Format percentages before saving
        final_results_df_tosave = final_results_df_ordered.copy()
        if 'Percent_Aligned_of_AC' in final_results_df_tosave.columns:
             final_results_df_tosave['Percent_Aligned_of_AC'] = final_results_df_tosave['Percent_Aligned_of_AC'].map('{:.1f}'.format, na_action='ignore')

        final_results_df_tosave.to_csv(output_path, index=False, encoding='utf-8-sig') # Use utf-8-sig
        logging.info(f"Hypothesis alignment by demographic group saved to {output_path}")
    except Exception as e:
        logging.error(f"Failed to save demographic group alignment analysis: {e}")

    # Debug: Check final labels before returning
    logging.debug(f"Unique hypothesis labels in FINAL results_df for plotting: {final_results_df_ordered['hypothesis_label'].unique()}")

    # Return the dataframe with numeric percentages for plotting
    return final_results_df_ordered


# --- V7 Function: Plot Alignment by Demographic Group (Corrected V3) ---
def plot_alignment_by_demographic_group(results_df: Optional[pd.DataFrame], visual_dir: Path):
    """
    Generates bar plots showing hypothesis alignment percentage for different demographic groups.

    Args:
        results_df: DataFrame generated by analyze_hypothesis_alignment_by_demographic_group
                   (expected cols: Demographic_Grouping_Type, Demographic_Category, hypothesis_label, Percent_Aligned_of_AC).
        visual_dir: Path to save the plots.
    """
    logging.info("--- Generating Plots for Alignment by Demographic Group ---")

    if results_df is None or results_df.empty:
        logging.warning("No data provided for demographic group alignment plotting. Skipping.")
        return
    required_plot_cols = ['Demographic_Grouping_Type', 'Demographic_Category', 'hypothesis_label', 'Percent_Aligned_of_AC']
    if not all(col in results_df.columns for col in required_plot_cols):
        logging.error(f"Missing required columns for plotting ({required_plot_cols}). Found: {results_df.columns}. Skipping.")
        return

    try:
        # Ensure column is numeric *before* plotting loop
        results_df['Percent_Aligned_of_AC'] = pd.to_numeric(results_df['Percent_Aligned_of_AC'], errors='coerce')
        logging.debug("Successfully converted Percent_Aligned_of_AC to numeric.")
    except Exception as e:
        logging.error(f"Could not convert 'Percent_Aligned_of_AC' to numeric: {e}. Skipping plots.")
        return

    # --- Define target hypothesis orders directly and robustly ---
    target_hypotheses_h1_h6 = []
    try:
        # Use the global map which should be populated during script setup
        target_hypotheses_h1_h6 = list(COMPARISON_TO_HYPOTHESIS_MAP.values())
        def get_hyp_num_direct(h_label):
             match = re.search(r'H(\d+)', str(h_label))
             return int(match.group(1)) if match else 99
        target_hypotheses_h1_h6.sort(key=get_hyp_num_direct)
        logging.debug(f"Defined target_hypotheses_h1_h6 from map: {target_hypotheses_h1_h6}")
        if not target_hypotheses_h1_h6 or len(target_hypotheses_h1_h6) != 6:
             raise ValueError("Map incomplete or empty") # Trigger fallback
    except Exception as e:
        logging.warning(f"Failed to define target hypotheses from map ({e}). Using manual list.")
        target_hypotheses_h1_h6 = [
             'H1: O > D', 'H2: IO > O', 'H3: ID > D',
             'H4: IO > ID', 'H5: ID > O', 'H6: IO > D'
        ]
    target_hypotheses_h2_h6 = [h for h in target_hypotheses_h1_h6 if not str(h).startswith('H1:')]
    logging.debug(f"Final target_hypotheses_h1_h6: {target_hypotheses_h1_h6}")
    logging.debug(f"Final target_hypotheses_h2_h6: {target_hypotheses_h2_h6}")


    grouping_types = results_df['Demographic_Grouping_Type'].unique()

    for demo_grouping_type in grouping_types:
        logging.debug(f"\nProcessing plotting for Grouping Type: {demo_grouping_type}")

        # 1. Filter data for the current grouping type
        plot_data = results_df[results_df['Demographic_Grouping_Type'] == demo_grouping_type].copy()
        logging.debug(f"  1. Shape after filtering type: {plot_data.shape}")
        if plot_data.empty: logging.debug(f"  Skipping {demo_grouping_type}: empty after type filter."); continue

        # 2. Filter out 'Unknown'/'Non-User' categories
        categories_before_filter = plot_data['Demographic_Category'].unique()
        filter_out_cats = ['Unknown', 'Non-User/Unknown', DEFAULT_ROLE_GROUP]
        plot_data = plot_data[~plot_data['Demographic_Category'].isin(filter_out_cats)].copy()
        logging.debug(f"  2. Shape after filtering out cats {filter_out_cats}: {plot_data.shape}")
        logging.debug(f"     Categories AFTER filtering unknowns: {plot_data['Demographic_Category'].unique()}")
        if plot_data.empty: logging.debug(f"  Skipping {demo_grouping_type}: empty after unknown cat filter."); continue

        # 3. Filter rows where percentage is NaN (using the already converted numeric column)
        rows_before_nan_drop = len(plot_data)
        nan_pct_rows = plot_data[plot_data['Percent_Aligned_of_AC'].isna()]
        plot_data = plot_data.dropna(subset=['Percent_Aligned_of_AC']).copy()
        logging.debug(f"  3. Shape after dropping NaN percentages: {plot_data.shape}")
        if not nan_pct_rows.empty:
             logging.debug(f"     Example rows dropped for NaN Pct:\n{nan_pct_rows[['Demographic_Category', 'hypothesis_label', 'N_Aligned_Contradicted']].head().to_string()}")

        if plot_data.empty:
            logging.info(f"No plottable data for demographic grouping '{demo_grouping_type}' after filtering unknowns/NaNs. Skipping plot.")
            continue

        # --- Determine Category Order ---
        present_categories = plot_data['Demographic_Category'].unique()
        category_order = []
        if demo_grouping_type == 'experience_group':
            exp_order = ['<5 years', '>=5 years']
            category_order = [c for c in exp_order if c in present_categories] + [c for c in sorted(present_categories) if c not in exp_order]
        elif demo_grouping_type == 'code_engagement_group':
            eng_order = ['Low Engagement', 'Medium Engagement', 'High Engagement']
            category_order = [c for c in eng_order if c in present_categories] + [c for c in sorted(present_categories) if c not in eng_order]
        elif demo_grouping_type == 'se_satisfaction_group':
             sat_order = ['Med/Low Sat', 'High Sat']
             category_order = [c for c in sat_order if c in present_categories] + [c for c in sorted(present_categories) if c not in sat_order]
        elif demo_grouping_type == 'role_group':
             role_order = ['Student', 'Developer/Engineer', 'Other Academic/Research', 'Other Tech Roles']
             category_order = [c for c in role_order if c in present_categories] + [c for c in sorted(present_categories) if c not in role_order]
        else:
             category_order = sorted(present_categories)
        if not category_order and present_categories.size > 0:
             logging.warning(f"Could not determine category order for grouping '{demo_grouping_type}'. Using default alphabetical.")
             category_order = sorted(present_categories)
        elif not present_categories.size > 0:
             logging.debug(f"  No categories left for {demo_grouping_type} to determine order.")
             continue
        logging.debug(f"  Determined category_order: {category_order}")

        # --- Check labels present in filtered data against target list ---
        present_labels_in_plot_data = plot_data['hypothesis_label'].unique()
        logging.debug(f"  Labels present in FINAL plot_data for {demo_grouping_type}: {present_labels_in_plot_data}")

        # Determine the actual order for the plot based on TARGET labels found in the data
        hyp_order_plot1 = [h for h in target_hypotheses_h1_h6 if h in present_labels_in_plot_data]
        logging.debug(f"  Resulting hyp_order_plot1 for {demo_grouping_type}: {hyp_order_plot1}")

        # --- Plot 1: Hypotheses H1-H6 ---
        if not hyp_order_plot1:
             logging.info(f"No data for H1-H6 for grouping '{demo_grouping_type}' after filtering. Skipping H1-H6 plot.")
             # No need to continue to Plot 2 check if H1-H6 is empty
             continue
        else:
             logging.info(f"Proceeding to generate H1-H6 plot for {demo_grouping_type}...")
             try:
                 plt.figure(figsize=(12, 7))
                 ax = sns.barplot(data=plot_data, x='hypothesis_label', y='Percent_Aligned_of_AC',
                                  hue='Demographic_Category', order=hyp_order_plot1, hue_order=category_order,
                                  palette='viridis', errorbar=None)
                 plot_title_suffix = demo_grouping_type.replace("_"," ").title()
                 plt.title(f'Hypothesis Alignment by {plot_title_suffix} (H1-H6)', fontsize=16)
                 plt.xlabel('Hypothesis (Expected Winner > Loser)', fontsize=12)
                 plt.ylabel('% Aligned (of Aligned + Contradicted Responses)', fontsize=12)
                 plt.xticks(rotation=0, ha='center', fontsize=10)
                 plt.yticks(np.arange(0, 101, 10), fontsize=10)
                 plt.ylim(0, 100)
                 plt.axhline(50, color='grey', linestyle='--', linewidth=0.8)
                 plt.legend(title=plot_title_suffix, bbox_to_anchor=(1.02, 1), loc='upper left')
                 plt.tight_layout(rect=[0, 0, 0.85, 1])
                 plot_filename = f"alignment_by_{demo_grouping_type}_H1-H6.png"
                 plot_path = visual_dir / plot_filename
                 plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                 logging.info(f"Saved alignment plot: {plot_path}")
                 plt.close()
             except Exception as e:
                  logging.error(f"Error generating H1-H6 plot for {demo_grouping_type}: {e}", exc_info=True)
                  plt.close()

        # --- Plot 2: Hypotheses H2-H6 ---
        hyp_order_plot2 = [h for h in target_hypotheses_h2_h6 if h in present_labels_in_plot_data]
        logging.debug(f"  Resulting hyp_order_plot2 for {demo_grouping_type}: {hyp_order_plot2}")

        if not hyp_order_plot2:
             logging.info(f"No data for H2-H6 for grouping '{demo_grouping_type}' after filtering. Skipping H2-H6 plot.")
             continue
        else:
             logging.info(f"Proceeding to generate H2-H6 plot for {demo_grouping_type}...")
             try:
                 plt.figure(figsize=(10, 7))
                 ax_h2 = sns.barplot(data=plot_data, x='hypothesis_label', y='Percent_Aligned_of_AC',
                                     hue='Demographic_Category', order=hyp_order_plot2, hue_order=category_order,
                                     palette='viridis', errorbar=None)
                 plot_title_suffix = demo_grouping_type.replace("_"," ").title()
                 plt.title(f'Hypothesis Alignment by {plot_title_suffix} (H2-H6 Only)', fontsize=16)
                 plt.xlabel('Hypothesis (Expected Winner > Loser)', fontsize=12)
                 plt.ylabel('% Aligned (of Aligned + Contradicted Responses)', fontsize=12)
                 plt.xticks(rotation=0, ha='center', fontsize=10)
                 plt.yticks(np.arange(0, 101, 10), fontsize=10)
                 plt.ylim(0, 100)
                 plt.axhline(50, color='grey', linestyle='--', linewidth=0.8)
                 plt.legend(title=plot_title_suffix, bbox_to_anchor=(1.02, 1), loc='upper left')
                 plt.tight_layout(rect=[0, 0, 0.83, 1])
                 plot_filename_h2 = f"alignment_by_{demo_grouping_type}_H2-H6.png"
                 plot_path_h2 = visual_dir / plot_filename_h2
                 plt.savefig(plot_path_h2, dpi=300, bbox_inches='tight')
                 logging.info(f"Saved alignment plot: {plot_path_h2}")
                 plt.close()
             except Exception as e:
                  logging.error(f"Error generating H2-H6 plot for {demo_grouping_type}: {e}", exc_info=True)
                  plt.close()


# --- UPDATED V7 Function: Generate Meta Results Report ---
def generate_meta_report(
    meta_report_path: Path,
    df_responses: pd.DataFrame,
    df_overall_alignment: pd.DataFrame, # <-- CHANGED ARGUMENT NAME
    df_hypothesis_tests: pd.DataFrame,
    df_spearman: pd.DataFrame,
    df_kruskal: pd.DataFrame,
    df_alignment_results: Optional[pd.DataFrame] # <-- Demographic alignment results
):
    """Generates a consolidated Markdown report summarizing key findings from V7 analysis."""
    logging.info(f"--- Generating V7 Meta Results Summary Report ---")
    report_content = []
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    num_participants = df_responses['participant_id'].nunique()
    num_total_responses = len(df_responses)

    # --- Helper Functions for Formatting (Defined *within* generate_meta_report) ---
    def format_p_report(p_value: Optional[float]) -> str:
        """Formats p-value for the report."""
        if pd.isna(p_value): return "N/A"
        if p_value < 0.001: return "< 0.001"
        return f"{p_value:.3f}"

    def get_sig_marker_report(p_value: Optional[float]) -> str:
        """Gets significance marker based on p-value and ALPHA."""
        if pd.isna(p_value): return ""
        if p_value < 0.001: return "***"
        if p_value < 0.01: return "**"
        if p_value < ALPHA: return "*" # Use ALPHA constant
        return "(ns)" # non-significant
    # --- End of Helper Function Definitions ---

    # --- Report Header ---
    report_content.append(f"# Survey Analysis Meta Results Summary (V7 - Extended Hypotheses)") # Updated Title
    report_content.append(f"Generated: {now}")
    report_content.append(f"Analysis Version: v7")
    report_content.append(f"Significance Level (Alpha): {ALPHA}")
    report_content.append(f"\nBased on **{num_participants} unique participants** and **{num_total_responses} individual comparison responses**.")
    report_content.append(f"\n*Note: This report summarizes key findings. For full details and exploratory analyses, consult the CSV files and plots in the '{OUTPUT_DIR.name}' and '{VISUAL_DIR.name}' subdirectories.*")

    # --- Section 1: Key Hypothesis Test Results ---
    report_content.append(f"\n## 1. Specific Hypothesis Test Results (Binomial Test)") # Renamed Section
    report_content.append(f"This section tests specific directional hypotheses about preference (e.g., 'Is IO preferred over O?').")
    report_content.append(f"The test compares the proportion preferring the 'Expected Winner' vs. the 'Expected Loser' among participants who chose one or the other (N Winner + N Loser).")
    report_content.append(f"P-values test if the preference for the Winner is significantly greater than 50% (one-sided test, alpha = {ALPHA}).")
    report_content.append(f"An extended set (H7-H12) tests the *opposite* direction to help interpret non-significant results in H1-H6.\n")


    if df_hypothesis_tests.empty:
        report_content.append(f"*No hypothesis test results available.*")
    else:
        # Include N Other in the table header
        report_content.append(f"| Hypothesis | Comparison                         | N Winner | N Loser | N Neither | N Other | % Winner (of W+L) | P-value | Sig. |")
        report_content.append(f"| :--------- | :--------------------------------- | :------- | :------ | :-------- | :------ | :---------------- | :------ | :--- |")

        # Ensure data is sorted logically if needed (e.g., by hypothesis number)
        try:
            # Use regex to robustly extract number after 'H'
            df_hypothesis_tests['Hyp_Num'] = df_hypothesis_tests['Hypothesis'].str.extract(r'H(\d+)').astype(int)
            df_hypothesis_tests = df_hypothesis_tests.sort_values('Hyp_Num').drop(columns=['Hyp_Num'])
        except Exception as e: # Catch potential errors during extraction/conversion
             logging.warning(f"Could not extract hypothesis number for sorting report table: {e}")

        # Iterate through the hypothesis results dataframe
        for _, row in df_hypothesis_tests.iterrows():
            # Safely get values using .get() with defaults
            hyp_label = row.get('Hypothesis', 'N/A')
            comp_type = row.get('Comparison', 'N/A')
            n_winner = row.get('N Winner', 0)
            n_loser = row.get('N Loser', 0)
            n_neither = row.get('N Neither', 0)
            n_other = row.get('N Other', 0) # Get N Other
            pct_winner = row.get('% Winner (of W+L)', np.nan)
            p_val = row.get('P-value (Binomial)') # Keep as float for marker function

            # Format for table
            pct_winner_str = f"{pct_winner:.1f}%" if pd.notna(pct_winner) else "N/A"
            p_val_str = format_p_report(p_val) # Use helper
            sig_marker = get_sig_marker_report(p_val) # Use helper

            # Handle potential pipes in names for Markdown table
            hyp_label_md = str(hyp_label).replace('|', '\\|')
            comp_type_md = str(comp_type).replace('|', '\\|')

            # Ensure counts are integers for display
            n_winner_int = int(n_winner) if pd.notna(n_winner) else 0
            n_loser_int = int(n_loser) if pd.notna(n_loser) else 0
            n_neither_int = int(n_neither) if pd.notna(n_neither) else 0
            n_other_int = int(n_other) if pd.notna(n_other) else 0 # Format N Other

            # Add N Other to the table row
            report_content.append(f"| **{hyp_label_md}** | {comp_type_md} | {n_winner_int} | {n_loser_int} | {n_neither_int} | {n_other_int} | {pct_winner_str} | {p_val_str} | **{sig_marker}** |")

        # --- Enhanced Interpretation Section ---
        report_content.append(f"\n**Interpretation Guidance:**")
        report_content.append(f"- **Significant Result (*, **, ***):** There is sufficient statistical evidence (at alpha={ALPHA}) to support the stated directional hypothesis (e.g., H1: O > D).")
        report_content.append(f"- **Non-Significant Result (ns):** There is *not* sufficient statistical evidence (at alpha={ALPHA}) to conclude the stated directional hypothesis is true. This *does not* prove the opposite is true or that there is absolutely no difference (absence of evidence is not evidence of absence).") # Clarified wording
        report_content.append(f"  - Look at the **% Winner**: If it's clearly above 50% (e.g., >60%) and p-value is low (e.g., < 0.15), consider this a **potential trend** favoring the winner, requiring discussion and support from qualitative data.")
        report_content.append(f"  - Check the **opposing hypothesis** (e.g., H7 for H1): If *both* H1 (A>B) and H7 (B>A) are non-significant, this strengthens the conclusion that **no statistically significant difference in preference was detected** between A and B.") # Explicit wording
        report_content.append(f"- **Low Counts:** Tests with low 'N Winner + N Loser' may lack statistical power; interpret non-significant results with extra caution, as a real difference might exist but be undetectable.")

        report_content.append(f"\n**Interpretation of Specific Hypothesis Pairs:**")
        # Create dictionary from the results DF for easy lookup
        results_dict = {}
        if 'Hypothesis' in df_hypothesis_tests.columns:
             results_dict = df_hypothesis_tests.set_index('Hypothesis').to_dict('index')
        else:
             logging.error("Cannot create results dictionary: 'Hypothesis' column missing.")


        # --- interpret_pair function (defined within generate_meta_report) ---
        def interpret_pair(label_main, label_opposite):
            """Helper to interpret a hypothesis and its opposite."""
            res_main = results_dict.get(label_main)
            res_opp = results_dict.get(label_opposite)
            interpretation = f"- **({label_main} vs {label_opposite}):** " # Show the pair being interpreted
            found_main = False
            found_opp = False
            sig_main = '(ns)' # Default to non-significant if missing
            sig_opp = '(ns)'
            p_main = 1.0
            p_opp = 1.0
            pct_main = np.nan # Default percentage to NaN
            pct_opp = np.nan

            interp_main = f"{label_main} (Results Missing)" # Default text
            interp_opp = f"{label_opposite} (Results Missing)" # Default text


            if res_main:
                 found_main = True
                 p_main = res_main.get('P-value (Binomial)')
                 sig_main = get_sig_marker_report(p_main) # Uses helper defined above
                 pct_main = res_main.get('% Winner (of W+L)') # Get as float/NaN
                 # Format string carefully, handling potential NaN for percentage
                 pct_main_str = f"{pct_main:.1f}%" if pd.notna(pct_main) else "N/A %"
                 interp_main = f"{label_main} ({pct_main_str} pref, p={format_p_report(p_main)} {sig_main})"


            if res_opp:
                 found_opp = True
                 p_opp = res_opp.get('P-value (Binomial)')
                 sig_opp = get_sig_marker_report(p_opp) # Uses helper defined above
                 pct_opp = res_opp.get('% Winner (of W+L)') # Get as float/NaN
                 # Format string carefully, handling potential NaN for percentage
                 pct_opp_str = f"{pct_opp:.1f}%" if pd.notna(pct_opp) else "N/A %"
                 interp_opp = f"{label_opposite} ({pct_opp_str} pref, p={format_p_report(p_opp)} {sig_opp})"


            if not found_main and not found_opp: return interpretation + "Results Missing for both hypotheses."

            # Combine individual results display
            interpretation += f"Test results: [{interp_main}]; [{interp_opp}]. "

            # Add summary conclusion based on significance of the pair
            if found_main and sig_main not in ['(ns)', '']:
                 # Main hypothesis is significant
                 interpretation += f"**Conclusion: Evidence supports {label_main}.**"
            elif found_opp and sig_opp not in ['(ns)', '']:
                 # Opposite hypothesis is unexpectedly significant
                 interpretation += f"**Conclusion: Evidence unexpectedly supports {label_opposite}.** (Investigate qualitative data for reasons!)"
            elif found_main and found_opp and sig_main == '(ns)' and sig_opp == '(ns)':
                 # *** BOTH non-significant: Conclude no detected difference, but check trends ***
                 interpretation += "**Conclusion: No statistically significant difference in preference detected between the two versions.**"
                 # Add trend information if present
                 p_main_val = p_main if pd.notna(p_main) else 1.0
                 p_opp_val = p_opp if pd.notna(p_opp) else 1.0
                 pct_main_val = pct_main if pd.notna(pct_main) else 50.0 # Use 50 if NaN for trend check
                 pct_opp_val = pct_opp if pd.notna(pct_opp) else 50.0 # Use 50 if NaN for trend check

                 # Check for trend in primary direction (e.g., >60% pref AND p < 0.15)
                 if pct_main_val >= 60 and p_main_val < 0.15:
                      interpretation += f" (However, note the trend favoring {label_main.split(':')[0]} with {pct_main_val:.1f}% preference, p={format_p_report(p_main)}. Qualitative insights needed)."
                 # Check for trend in opposite direction (less likely if primary trend exists, but check anyway)
                 elif pct_opp_val >= 60 and p_opp_val < 0.15:
                      interpretation += f" (However, note the trend favoring {label_opposite.split(':')[0]} with {pct_opp_val:.1f}% preference, p={format_p_report(p_opp)}. Qualitative insights needed)."
                 else:
                     # If no strong trend, just stick with no detected difference.
                     pass
            else: # Handles cases where one result might be missing, or other odd combinations
                 interpretation += "**Conclusion: Mixed results or incomplete data.** (Refer to individual p-values)." # Default fallback

            return interpretation
        # --- End of interpret_pair function ---

        # Interpret the pairs using the modified helper function
        report_content.append(interpret_pair("H1: O > D", "H7: D > O")) # Degradation
        report_content.append(interpret_pair("H2: IO > O", "H8: O > IO")) # Improvement Original
        report_content.append(interpret_pair("H3: ID > D", "H9: D > ID")) # Improvement Degraded
        report_content.append(interpret_pair("H4: IO > ID", "H10: ID > IO")) # Optimal Path
        report_content.append(interpret_pair("H5: ID > O", "H11: O > ID")) # Recovery
        report_content.append(interpret_pair("H6: IO > D", "H12: D > IO")) # Baseline Check

    report_content.append(f"\n*See `specific_hypothesis_tests_*.csv` for detailed counts and statistics.*")


    # --- Section 2: Overall Alignment Distribution (Descriptive) ---
    # UPDATED SECTION 2
    report_content.append(f"\n## 2. Overall Alignment with Primary Hypotheses (Descriptive Overview)")
    report_content.append(f"This shows the overall percentage breakdown of responses based on whether they **aligned** with the primary hypothesis (H1-H6), **contradicted** it, were **neutral** ('Neither'), or **other/unknown**.\n")
    report_content.append(f"| Comparison (Hypothesis)              | % Aligned (Success) (N) | % Contradicted (Failure) (N) | % Neutral (N) | % Other/Unknown (N) |")
    report_content.append(f"| :----------------------------------- | :---------------------- | :--------------------------- | :------------ | :------------------ |")

    if df_overall_alignment.empty:
        report_content.append(f"| *No overall alignment descriptive data available* | - | - | - | - |")
    else:
        # Ensure index is set for easy lookup
        if 'Comparison Type' in df_overall_alignment.columns:
             df_overall_alignment_idx = df_overall_alignment.set_index('Comparison Type')
        else:
             df_overall_alignment_idx = df_overall_alignment # Assume already indexed

        for comp_type in COMPARISON_TYPES: # Iterate in defined order
            if comp_type not in df_overall_alignment_idx.index: continue

            row = df_overall_alignment_idx.loc[comp_type]
            primary_hyp = row.get("Primary Hypothesis", "")

            # Get counts and percentages safely using the exact column names from analyze_overall_alignment_descriptive
            n_align = row.get("Count Aligned (Success)", 0)
            n_contra = row.get("Count Contradicted (Failure)", 0)
            n_neut = row.get("Count Neutral", 0)
            n_other = row.get("Count Other/Unknown", 0)

            p_align = row.get("% Aligned (of Total)", 0.0)
            p_contra = row.get("% Contradicted (of Total)", 0.0)
            p_neut = row.get("% Neutral (of Total)", 0.0)
            p_other = row.get("% Other/Unknown (of Total)", 0.0)

            # Format for table
            p_align_str = f"{p_align:.1f}% ({int(n_align)})" if pd.notna(p_align) else "N/A"
            p_contra_str = f"{p_contra:.1f}% ({int(n_contra)})" if pd.notna(p_contra) else "N/A"
            p_neut_str = f"{p_neut:.1f}% ({int(n_neut)})" if pd.notna(p_neut) else "N/A"
            p_other_str = f"{p_other:.1f}% ({int(n_other)})" if pd.notna(p_other) else "N/A"

            # Handle potential pipes in names
            comp_type_md = str(comp_type).replace('|', '\\|')
            hyp_md = str(primary_hyp).replace('|', '\\|') if primary_hyp else "N/A"

            report_content.append(f"| **{comp_type_md}**<br>({hyp_md}) | {p_align_str} | {p_contra_str} | {p_neut_str} | {p_other_str} |")

    report_content.append(f"\n*This provides a high-level view. For inferential tests on preference direction, see Section 1.*")
    report_content.append(f"*See `overall_alignment_descriptive_analysis.csv` and `overall_alignment_comparison_*.png` plots for details.*") # Updated filenames


    # --- Section 3: Influence of Experience & LLM Satisfaction (Exploratory) ---
    # (This section remains the same - keep emphasizing 'exploratory')
    report_content.append(f"\n## 3. Influence of Experience & LLM Satisfaction (Exploratory Analysis)")
    report_content.append(f"(Focus on significant results, p < {ALPHA}, from Spearman and Kruskal-Wallis tests)")
    report_content.append(f"*Caution: These are exploratory correlations. Interpret significant findings carefully, considering the number of tests performed.*")

    significant_findings_found_corr = False
    # Spearman Correlations
    if not df_spearman.empty:
         report_content.append(f"\n**Spearman Correlations (Preference Score vs. Variable):**")
         if 'P-value' in df_spearman.columns:
             # Filter for significant results (p < ALPHA)
             sig_spearman = df_spearman[df_spearman['P-value'] < ALPHA].copy() # Use .copy()
             if not sig_spearman.empty:
                 significant_findings_found_corr = True
                 for _, row in sig_spearman.iterrows():
                     comp = row.get('Comparison', '?')
                     var = row.get('Variable', '?')
                     rho = row.get('Correlation (rho)', 0)
                     p_val = row.get('P-value')
                     # Simple interpretation of direction and strength
                     direction = "positively" if rho > 0 else "negatively"
                     strength = "weakly" if abs(rho) < 0.3 else "moderately" if abs(rho) < 0.6 else "strongly"
                     # Format output
                     comp_md = str(comp).replace('|', '\\|')
                     var_md = str(var).replace('|', '\\|')
                     p_val_fmt = format_p_report(p_val)
                     sig_marker = get_sig_marker_report(p_val)
                     report_content.append(f"- Preference in **{comp_md}** was significantly ({strength} {direction}) correlated with **{var_md}** (rho = {rho:.3f}, p {p_val_fmt} {sig_marker}).")
             else:
                  report_content.append(f"- *No significant Spearman correlations found at p < {ALPHA}.*")
         else:
              report_content.append(f"- *'P-value' column missing in Spearman results data.*")
         report_content.append(f"  *(See `preference_correlations_spearman.csv` for full results)*")
    else: report_content.append(f"\n*Spearman correlation data not available.*")

    # Kruskal-Wallis Tests
    if not df_kruskal.empty:
         report_content.append(f"\n**Kruskal-Wallis Tests (Variable Distribution across Preference Groups):**")
         if 'P-value' in df_kruskal.columns:
             # Filter for significant results
             sig_kruskal = df_kruskal[df_kruskal['P-value'] < ALPHA].copy() # Use .copy()
             if not sig_kruskal.empty:
                  significant_findings_found_corr = True
                  # sig_kruskal = sig_kruskal.sort_values('P-value') # Optional sort
                  for _, row in sig_kruskal.iterrows():
                      comp = row.get('Comparison', '?')
                      var = row.get('Variable', '?')
                      h_stat = row.get('H-statistic', np.nan)
                      p_val = row.get('P-value')
                      # Format output
                      comp_md = str(comp).replace('|', '\\|')
                      var_md = str(var).replace('|', '\\|')
                      h_stat_str = f"{h_stat:.2f}" if pd.notna(h_stat) else "N/A"
                      p_val_fmt = format_p_report(p_val)
                      sig_marker = get_sig_marker_report(p_val)
                      report_content.append(f"- The distribution of **{var_md}** significantly differed across preference groups (Left/Neither/Right) for the **{comp_md}** comparison (H = {h_stat_str}, p {p_val_fmt} {sig_marker}).")
             else:
                  report_content.append(f"- *No significant differences found in variable distributions across preference groups using Kruskal-Wallis (p < {ALPHA}).*")
         else:
             report_content.append(f"- *'P-value' column missing in Kruskal-Wallis results data.*")
         report_content.append(f"  *(See `preference_correlations_kruskal_wallis.csv` and associated boxplots for details)*")
    else: report_content.append(f"\n*Kruskal-Wallis test data not available.*")

    if not significant_findings_found_corr:
         report_content.append(f"\n*Overall, the exploratory analysis did not reveal strong significant associations between Experience/LLM Satisfaction and preferences in the tested comparisons (at p < {ALPHA}).*")


    # --- Section 4: Qualitative Insights Prompt ---
    # (This section remains largely the same, emphasizing the next steps)
    report_content.append(f"\n## 4. Qualitative Insights (Essential Next Step)")
    report_content.append(f"While the quantitative analysis shows *what* preferences exist (or don't significantly), the qualitative remarks explain *why*. This is crucial for understanding the nuances, especially for non-significant results or observed trends.") # Added emphasis
    report_content.append(f"**Action Required:** Perform thematic analysis on the extracted remarks in `output_data/qualitative_remarks_context.csv`.")
    report_content.append(f"\n**Key questions to investigate in the remarks:**")
    report_content.append(f"- What specific aspects made users prefer the Original, Degraded, ID, or IO versions?")
    report_content.append(f"- **Clarity & Conciseness:** Was the language easy to understand? Was there too much/too little detail?")
    report_content.append(f"- **Tone:** Was the tone perceived positively (friendly, professional) or negatively (robotic, verbose, informal)?")
    report_content.append(f"- **Structure:** Did elements like summaries, checklists, bullet points help or hinder understanding?")
    report_content.append(f"- **Accuracy & Usefulness:** Did participants comment on the perceived correctness or helpfulness of the generated content?")
    report_content.append(f"- **Specific Examples:** Are there recurring comments about specific phrases, sections, or generated elements (e.g., 'word noise', 'human touch', 'good structure')?")
    report_content.append(f"- **Reasons for 'Neither':** What trade-offs or lack of difference led participants to choose 'Neither'? This is very important if significance wasn't found.")
    report_content.append(f"- **Reasons for Trends (e.g., p<0.15):** For comparisons showing trends (like H2: IO > O), what do the comments say about *why* people leaned that way?") # Added specific prompt
    report_content.append(f"- **Explaining Lack of Difference:** For pairs where *neither* direction was significant (e.g., H1/H7), what reasons do comments give for similarity or ambivalence?") # Added prompt for this case
    report_content.append(f"- **Contradictions:** Why did some participants contradict the expected hypothesis (e.g., prefer Degraded over Original, or prefer O over IO)?") # Edited prompt
    report_content.append(f"- **Demographic Patterns:** Do comments suggest different priorities based on experience, role, or LLM satisfaction? (Refer to `alignment_by_*` plots for quantitative hints).") # Added pointer


    # --- Write Report to File ---
    try:
        with open(meta_report_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(report_content))
        logging.info(f"V7 (Extended Hypotheses) Meta results summary report saved to: {meta_report_path}") # Updated log
    except Exception as e:
        logging.error(f"Failed to write V7 (Extended Hypotheses) meta results report to {meta_report_path}: {e}")


# --- Main Execution Logic ---
def main():
    """Main function to orchestrate the V7 survey analysis workflow."""
    logging.info("=================================================")
    logging.info("Starting Enhanced Survey Analysis V7...")
    logging.info(f"Analysis output directory: {ANALYSIS_DIR}")
    logging.info("=================================================")

    # --- Step 1: Load Setup Configuration ---
    setup_config = load_setup(SETUP_PATH)
    if not setup_config:
        logging.error("Exiting: Setup configuration loading failed.")
        return

    # --- Step 2: Process Survey Result Files ---
    # This now returns df_final which includes 'experience_numeric'
    df_responses = process_survey_files(RESULTS_DIR, setup_config)
    if df_responses.empty:
        logging.error("Exiting: No data loaded from survey result files.")
        return

    logging.info(f"\nProcessed {len(df_responses)} comparison responses from {df_responses['participant_id'].nunique()} unique participants.")
    processed_data_path = OUTPUT_DIR / "combined_processed_responses_v7.csv"
    try:
        # Ensure numeric columns are saved correctly
        float_cols = df_responses.select_dtypes(include=['float']).columns
        # Use a general float format for CSV saving
        df_responses.to_csv(processed_data_path, index=False, encoding='utf-8-sig', float_format='%.3f') # Use utf-8-sig
        logging.info(f"Combined processed responses saved to {processed_data_path}")
    except Exception as e:
        logging.error(f"Failed to save combined processed responses: {e}")

    # --- Step 3: Analyze Overall Alignment (Descriptive - Success/Failure/Neutral) --- ## MODIFIED STEP
    df_overall_alignment_summary = analyze_overall_alignment_descriptive(df_responses, OUTPUT_DIR)
    if not df_overall_alignment_summary.empty:
        plot_overall_alignment_descriptive(df_overall_alignment_summary, VISUAL_DIR)
    else:
        logging.warning("Skipping overall alignment plotting due to missing descriptive results.")

    # --- Step 4: Analyze Specific Pairwise Hypotheses (Inferential) ---
    df_hypothesis_results = analyze_pairwise_hypotheses(df_responses, SPECIFIC_HYPOTHESES, OUTPUT_DIR)
    # Note: No specific plot for this table, results go into the meta report.

    # --- Step 5: Analyze Demographics ---
    analyze_demographics(df_responses, OUTPUT_DIR, VISUAL_DIR)

    # --- Step 6: Analyze Preferences by Demographics (Exploratory Breakdowns - Left/Right/Neither) ---
    logging.info("\n--- Generating Preference Breakdowns by Demographics (Exploratory) ---")
    demo_vars_for_breakdown = list(DEMOGRAPHIC_COLS_MAP.values()) + [
        'overall_se_satisfaction_category', 'overall_other_satisfaction_category'
        ]
    for demo_var in demo_vars_for_breakdown:
        analyze_preference_by_demographic(df_responses, demo_var, OUTPUT_DIR, VISUAL_DIR) # Keep this as is for L/R/N breakdown

    # --- Step 7: Analyze Correlations (Exploratory) ---
    # Pass df_responses which should now contain 'experience_numeric' if possible
    df_spearman_results, df_kruskal_results = analyze_correlations(df_responses, OUTPUT_DIR, VISUAL_DIR)

    # --- Step 8 (New): Analyze Hypothesis Alignment by Demographic Group ---
    # This function needs the full df_responses to access demographics
    df_alignment_by_group = analyze_hypothesis_alignment_by_demographic_group(df_responses, OUTPUT_DIR)

    # --- Step 9 (New): Plot Alignment by Demographic Group ---
    plot_alignment_by_demographic_group(df_alignment_by_group, VISUAL_DIR)

    # --- Step 10 (Old Step 8): Extract Qualitative Remarks ---
    extract_qualitative_remarks(df_responses, OUTPUT_DIR) # Pass df_responses

    # --- Step 11 (Old Step 9): Generate Meta Results Report (V7 - Extended Hypotheses) ---
    # Pass the NEW alignment summary dataframe instead of the old overall prefs df
    generate_meta_report(
        META_REPORT_FILE,
        df_responses, # Base dataframe for counts
        df_overall_alignment_summary if not df_overall_alignment_summary.empty else pd.DataFrame(), # <-- PASS NEW ALIGNMENT SUMMARY
        df_hypothesis_results if not df_hypothesis_results.empty else pd.DataFrame(),
        df_spearman_results if not df_spearman_results.empty else pd.DataFrame(),
        df_kruskal_results if not df_kruskal_results.empty else pd.DataFrame(),
        df_alignment_by_group # <-- PASS Demographic alignment results
    )

    # --- Final Message ---
    logging.info("\n=================================================")
    logging.info(f"Analysis V7 complete. Outputs are in the '{ANALYSIS_DIR.name}' directory.")
    logging.info(f"-> **Key Findings:** Check '{META_REPORT_FILE.name}' for hypothesis tests and summary.")
    logging.info(f"-> **Overall Alignment:** Check 'overall_alignment_descriptive_analysis.csv' and plots 'overall_alignment_comparison_*.png'.") # Added pointer
    logging.info(f"-> **Demographic Alignment:** Check 'hypothesis_alignment_by_demographic_group.csv' and associated plots ('alignment_by_*').") # Added pointer
    logging.info(f"-> **Detailed Data:** Other CSV files are in '{OUTPUT_DIR.name}'.")
    logging.info(f"-> **Visualizations:** Other plots are in '{VISUAL_DIR.name}'.")
    logging.info("-> **CRITICAL NEXT STEP:** Perform thorough thematic analysis on 'qualitative_remarks_context.csv' to understand the 'why' behind the results.")
    logging.info("=================================================")

if __name__ == "__main__":
    # Optional: Setup dummy files/dirs for testing if needed
    # (No changes needed here, keep dummy file creation logic)
    try:
        if not BASE_DIR.exists(): BASE_DIR.mkdir(exist_ok=True) # Added exist_ok=True
        if not (BASE_DIR / "setup").exists(): (BASE_DIR / "setup").mkdir(exist_ok=True)
        if not RESULTS_DIR.exists(): RESULTS_DIR.mkdir(exist_ok=True)
        # ANALYSIS_DIR, VISUAL_DIR, OUTPUT_DIR created at top

        # Create dummy SETUP.csv if it doesn't exist (using provided example)
        if not SETUP_PATH.exists():
             logging.warning(f"Setup file {SETUP_PATH} not found. Creating dummy.")
             # Use header exactly matching user's file
             dummy_setup_content = """Survey,PR (Original vs Degraded),PR (Original vs Improved_Degraded),PR (Degraded vs Improved_Degraded), PR (Improved_Degraded vs Improved_Original),PR (Original vs Improved_Original),PR (Degraded vs Improved_Original)
1,1,2,3,4,5,6
2,2,3,4,5,6,1
3,3,4,5,6,1,2
4,4,5,6,1,2,3
5,5,6,1,2,3,4
6,6,1,2,3,4,5
"""
             with open(SETUP_PATH, 'w', encoding='utf-8') as f: f.write(dummy_setup_content)

        # Create dummy result CSVs if they don't exist (using provided example structure)
        for i in range(1, 7): # Assuming surveys 1-6
            result_file = RESULTS_DIR / f"{i}.csv"
            if not result_file.exists():
                 logging.warning(f"Result file {result_file} not found. Creating dummy.")
                 # Use a header that includes most expected columns based on V7 script
                 dummy_header = [
                     "Timestamp", "Unnamed: 1", # Use "Unnamed: 1" to match pandas default
                     "How many years did you spend professionally working as a software engineer?",
                     "What is your current main professional activity?",
                     "How frequently do you read or write source code?",
                     "How frequently do you use version control tools (Git, GitHub, GitLab, SVN, etc.)?",
                     "How frequently do you take part in a code review (author/reviewer/etc.)?",
                     # Add some satisfaction columns
                     f"{SATISFACTION_PREFIX_SE} [Improve existing code]",
                     f"{SATISFACTION_PREFIX_OTHER} [Summarize content]",
                     # Add preference/remark pairs (6 pairs expected)
                     "Which PR timeline was better?", "If you have additional remarks...", # Pair 1
                     "Which PR timeline was better?", "If you have additional remarks...", # Pair 2
                     "Which PR timeline was better?", "If you have additional remarks...", # Pair 3
                     "Which PR timeline was better?", "If you have additional remarks...", # Pair 4
                     "Which PR timeline was better?", "If you have additional remarks...", # Pair 5
                     "Which PR timeline was better?", "If you have additional remarks...", # Pair 6
                 ]
                 # Create dummy row data (ensure correct number of fields)
                 # Example: Make Preferences align with H1-H6 for first participant
                 dummy_row_aligned = [
                     f"2025/03/{10+i} 10:00:00", "Yes, I agree", # Timestamp, Consent
                     "<5 years", "Professional - Software Engineer", "Daily", "Daily", "Weekly", # Demographics
                     "4", "5 (very satisfied)", # Satisfaction
                     # Preferences (Aligned with H1-H6 for Survey 1, PRs 1-6) & Remarks
                     # H1: O>D (Left='O'), H2: IO>O (Right='IO'), H3: ID>D (Right='ID'), H4: IO>ID (Right='IO'), H5: ID>O (Right='ID'), H6: IO>D (Right='IO')
                     # Assuming Survey 1 assignments: PR1=OvsD, PR2=OvsID, PR3=DvsID, PR4=IDvsIO, PR5=OvsIO, PR6=DvsIO
                     "The LEFT PR timeline was better.", "Aligned H1", # PR1 (O vs D) -> Prefer O (Left) -> Aligns H1
                     "The RIGHT PR timeline was better.", "Aligned H5", # PR2 (O vs ID) -> Prefer ID (Right) -> Aligns H5
                     "The RIGHT PR timeline was better.", "Aligned H3", # PR3 (D vs ID) -> Prefer ID (Right) -> Aligns H3
                     "The RIGHT PR timeline was better.", "Aligned H4", # PR4 (ID vs IO) -> Prefer IO (Right) -> Aligns H4
                     "The RIGHT PR timeline was better.", "Aligned H2", # PR5 (O vs IO) -> Prefer IO (Right) -> Aligns H2
                     "The RIGHT PR timeline was better.", "Aligned H6", # PR6 (D vs IO) -> Prefer IO (Right) -> Aligns H6
                 ]
                 # Example: Make Preferences contradict H1-H6 for second participant
                 dummy_row_contradicted = [
                     f"2025/03/{10+i} 10:01:00", "Yes, I agree", # Timestamp, Consent
                     ">=5 years", "Student", "Weekly", "Monthly", "Never", # Demographics
                     "2", "3", # Satisfaction
                     # Preferences (Contradict H1-H6 for Survey 1, PRs 1-6) & Remarks
                     "The RIGHT PR timeline was better.", "Contradict H1", # PR1 (O vs D) -> Prefer D (Right) -> Contradicts H1
                     "The LEFT PR timeline was better.", "Contradict H5",  # PR2 (O vs ID) -> Prefer O (Left) -> Contradicts H5
                     "The LEFT PR timeline was better.", "Contradict H3",  # PR3 (D vs ID) -> Prefer D (Left) -> Contradicts H3
                     "The LEFT PR timeline was better.", "Contradict H4",  # PR4 (ID vs IO) -> Prefer ID (Left) -> Contradicts H4
                     "The LEFT PR timeline was better.", "Contradict H2",  # PR5 (O vs IO) -> Prefer O (Left) -> Contradicts H2
                     "The LEFT PR timeline was better.", "Contradict H6",  # PR6 (D vs IO) -> Prefer D (Left) -> Contradicts H6
                 ]
                  # Example: Make Preferences neutral for third participant
                 dummy_row_neutral = [
                     f"2025/03/{10+i} 10:02:00", "Yes, I agree", # Timestamp, Consent
                     "<5 years", "Other Tech Roles", "Monthly", "Daily", "A few times a year", # Demographics
                     "3", "Don't Use", # Satisfaction
                     # Preferences (Neutral H1-H6 for Survey 1, PRs 1-6) & Remarks
                     "NEITHER PR timeline was significantly better than the other.", "Neutral H1", # PR1 (O vs D) -> Neither
                     "NEITHER PR timeline was significantly better than the other.", "Neutral H5", # PR2 (O vs ID) -> Neither
                     "NEITHER PR timeline was significantly better than the other.", "Neutral H3", # PR3 (D vs ID) -> Neither
                     "NEITHER PR timeline was significantly better than the other.", "Neutral H4", # PR4 (ID vs IO) -> Neither
                     "NEITHER PR timeline was significantly better than the other.", "Neutral H2", # PR5 (O vs IO) -> Neither
                     "NEITHER PR timeline was significantly better than the other.", "Neutral H6", # PR6 (D vs IO) -> Neither
                 ]
                 # Ensure header and row have same number of fields
                 if len(dummy_header) != len(dummy_row_aligned):
                      logging.error(f"Dummy header length ({len(dummy_header)}) != Dummy row length ({len(dummy_row_aligned)}) for {result_file}. Skipping dummy creation.")
                      continue

                 # Write dummy file
                 with open(result_file, 'w', newline='', encoding='utf-8') as f:
                     writer = csv.writer(f)
                     writer.writerow(dummy_header)
                     # Write the varied rows
                     writer.writerow(dummy_row_aligned)
                     writer.writerow(dummy_row_contradicted)
                     writer.writerow(dummy_row_neutral)


    except Exception as dir_err:
        logging.error(f"Error creating dummy directories/files: {dir_err}")

    # Run the main analysis
    main()