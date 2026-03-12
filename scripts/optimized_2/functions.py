import functools
import shutil
import time
import statistics
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from scipy.stats import rankdata
from scipy.stats import mannwhitneyu
from scipy.stats import norm
import statsmodels.formula.api as smf
import multiprocessing as mp
import argparse
import glob
import os
import logging
import math

# ......................................................
#
#   CONSTANTS ----
#
# ......................................................
N_ROUND_NORM = 3
N_ROUND_TEST = 1
# For ZIW only
CSTE_VARIANCE = 2 * math.sqrt(2 / math.pi)


# ......................................................
#
#   FUNCTIONS ----
#
# ......................................................


# ......................................................
#   TOP-N SELECTION via np.argpartition (O(n) vs O(n log n))
#   Carries a parallel `scores` array to avoid re-extracting
#   scores from 200k+ tuples each chunk call.
# ......................................................

def keep_top_n(current_top, current_scores, new_results, new_scores_arr, n, reverse=True):
    """
    Merge current_top with new_results and return the top-n entries
    using np.argpartition (O(n)) instead of a full sort (O(n log n)).

    Parameters
    ----------
    current_top    : list of tuples accumulated so far
    current_scores : np.ndarray of float64 scores for current_top
    new_results    : list of tuples from the latest chunk
    new_scores_arr : np.ndarray of float64 scores for new_results
    n              : maximum number of entries to keep
    reverse        : True  → keep largest values (t-stat, variance, …)
                     False → keep smallest values (pitest p-value)

    Returns
    -------
    (top_list, top_scores) : filtered list and corresponding scores array
    """
    combined = current_top + new_results
    if len(current_scores) == 0:
        scores = new_scores_arr
    else:
        scores = np.concatenate([current_scores, new_scores_arr])

    if len(combined) <= n:
        return combined, scores

    if reverse:
        idx = np.argpartition(scores, -n)[-n:]
    else:
        idx = np.argpartition(scores, n)[:n]

    new_top = [combined[i] for i in idx]
    new_top_scores = scores[idx]
    return new_top, new_top_scores


# Estimate total number of tags (lines)

def estimate_total_lines(file_path, sample_size=100):
    """ Estimate the total number of lines in a file using a sample of size sample_size """
    with open(file_path, 'r') as f:
        # Read a sample of sample_size lines
        sample_lines = [next(f) for _ in range(sample_size)]

    # Calculate the average line size in the sample
    average_line_size = sum(len(line) for line in sample_lines) / sample_size

    # Get the total file size in bytes
    file_size = os.path.getsize(file_path)

    # Estimate the total number of lines
    estimated_total_lines = int(file_size / average_line_size)

    return estimated_total_lines


# ......................................................
#   VECTORIZED TEST FUNCTIONS
#   Each function operates on the full chunk at once.
#   Returns raw numpy arrays instead of Python tuple lists.
# ......................................................

def perform_ttest_vectorized(grp_a_vals, grp_b_vals):
    """
    Vectorized t-test across all rows in a chunk.
    Returns (abs_t_stats, log2fc, p_values, valid_idx) as numpy arrays.
    """
    log_a = np.log1p(grp_a_vals)
    log_b = np.log1p(grp_b_vals)

    t_stats, p_values = ttest_ind(log_a, log_b, axis=1)

    log2fc = np.log2(grp_a_vals.mean(axis=1) + 1) - np.log2(grp_b_vals.mean(axis=1) + 1)

    valid = ~np.isnan(t_stats)
    valid_idx = np.where(valid)[0]

    abs_t = np.abs(np.round(t_stats[valid], N_ROUND_TEST))
    fc = np.round(log2fc[valid], N_ROUND_TEST)
    pv = p_values[valid]

    return abs_t, fc, pv, valid_idx


def perform_wilcoxon_vectorized(grp_a_vals, grp_b_vals):
    """
    Vectorized Mann-Whitney U test across all rows in a chunk.
    Returns (abs_stats, log2fc, p_values, valid_idx) as numpy arrays.
    """
    result = mannwhitneyu(grp_a_vals, grp_b_vals, axis=1)
    stats = result.statistic
    p_values = result.pvalue

    log2fc = np.log2(grp_a_vals.mean(axis=1) + 1) - np.log2(grp_b_vals.mean(axis=1) + 1)

    valid = ~np.isnan(stats)
    valid_idx = np.where(valid)[0]

    abs_s = np.abs(np.round(stats[valid], N_ROUND_TEST))
    fc = np.round(log2fc[valid], N_ROUND_TEST)
    pv = p_values[valid]

    return abs_s, fc, pv, valid_idx


def perform_pitest_vectorized(grp_a_vals, grp_b_vals):
    """
    Vectorized pi-test: pi = |log2FC * -log10(p_ttest)|
    Returns (pi_values, log2fc, valid_idx) as numpy arrays.
    """
    log_a = np.log1p(grp_a_vals)
    log_b = np.log1p(grp_b_vals)

    t_stats, p_values = ttest_ind(log_a, log_b, axis=1)

    log2fc = np.log2(grp_a_vals.mean(axis=1) + 1) - np.log2(grp_b_vals.mean(axis=1) + 1)

    with np.errstate(divide='ignore', invalid='ignore'):
        pi_values = np.abs(log2fc * (-np.log10(p_values)))

    valid = (~np.isnan(t_stats)) & (log2fc != 0) & (~np.isnan(pi_values))
    valid_idx = np.where(valid)[0]

    pi = np.round(pi_values[valid], N_ROUND_TEST)
    fc = np.round(log2fc[valid], N_ROUND_TEST)

    return pi, fc, valid_idx


def perform_variance_vectorized(chunk_vals):
    """
    Vectorized variance and CV over all rows in a chunk.
    Returns (variances, cvs, valid_idx) as numpy arrays.
    """
    means = chunk_vals.mean(axis=1)
    stds  = chunk_vals.std(axis=1)
    vars_ = chunk_vals.var(axis=1)

    with np.errstate(divide='ignore', invalid='ignore'):
        cvs = np.where(means != 0, stds / means, np.nan)

    valid = (~np.isnan(cvs)) & (means != 0)
    valid_idx = np.where(valid)[0]

    return vars_[valid], cvs[valid], valid_idx

# ......................................................
#   UNCHANGED: per-row functions kept for ZIW and ANOVA
#   which cannot be trivially vectorized.
# ......................................................

# Perform anova test with covariable

def perform_anova(tag, grp_a_data, grp_b_data, covariates_df):

    covariate_columns = list(covariates_df.columns)

    # 1. Extract expression values (log-transformed)
    grp_a_values = np.log(grp_a_data.loc[tag].values + 1)
    grp_b_values = np.log(grp_b_data.loc[tag].values + 1)

    # 2. Build sample order matching the expression data
    sample_order = list(grp_a_data.columns) + list(grp_b_data.columns)

    # 3. Align covariates safely
    cov_aligned = covariates_df.loc[sample_order, covariate_columns].reset_index(drop=True)

    # 4. Build analysis DataFrame
    df = pd.DataFrame({
        "y": np.concatenate([grp_a_values, grp_b_values]),
        "group": ["A"] * len(grp_a_values) + ["B"] * len(grp_b_values),
    })

    df[covariate_columns] = cov_aligned

    # 5. Fit regression
    formula = "y ~ group + " + " + ".join(covariate_columns)
    model = smf.ols(formula, data=df).fit()

    # Find the correct coefficient name for the group comparison
    group_coef = "group[T.B]" if "group[T.B]" in model.pvalues else "group"

    # 6. Log2FC
    log2fold_change = np.log2(np.mean(grp_a_data.loc[tag].values + 1) / np.mean(grp_b_data.loc[tag].values + 1))

    p_value = model.pvalues[group_coef]

    return str(tag), np.round(model.tvalues[group_coef], N_ROUND_TEST), np.round(log2fold_change, N_ROUND_TEST), p_value


# Calculate variance of the modified Wilcoxon rank sum statistic

def calculate_variance_ziw(data_dict: dict, prop_mean: float, n: int) -> float:
    # Calculate variance of the modified Wilcoxon rank sum statistic
    # @prop_mean: a float, average proportion of zeros in both groups
    prop_mean_complement = 1 - prop_mean

    # Variance in the first group
    v1 = prop_mean * prop_mean_complement
    var1 =  data_dict["product_n_a_n_b"] * prop_mean * v1 * (n * prop_mean + 3 * prop_mean_complement / 2) + \
    data_dict["sum_square_n_a_n_b"] * v1**2 * (5/4) + CSTE_VARIANCE * prop_mean * (n * v1)**(1.5) \
    * data_dict["sqrt_product_n_a_n_b"]
    var1 = var1 / 4

    # Variance in the second group
    var2 = data_dict["product_n_a_n_b"] * prop_mean**2 * (n * prop_mean + 1)/12

    # Variance of the modified Wilcoxon rank sum statistic
    variance = var1 + var2

    return variance

# Perform ZIW

def perform_ziw(tag, grp_a, grp_b, data_dict):
    grp_a_counts = grp_a.loc[tag].values
    grp_b_counts = grp_b.loc[tag].values

    n_non_zero_grp_a = np.count_nonzero(grp_a_counts)
    n_non_zero_grp_b = np.count_nonzero(grp_b_counts)

    prop_grp_a: float = n_non_zero_grp_a / data_dict["n_obs_grp_a"]
    prop_grp_b: float = n_non_zero_grp_b / data_dict["n_obs_grp_b"]

    prop_max: float = max(prop_grp_a, prop_grp_b)
    prop_mean: float = np.mean([prop_grp_a, prop_grp_b])

    n_truncated_grp_a: int = round(prop_max * data_dict["n_obs_grp_a"])
    n_truncated_grp_b: int = round(prop_max * data_dict["n_obs_grp_b"])
    indices_seq: list = range(n_truncated_grp_a)
    n_ziw: float = (n_truncated_grp_a + n_truncated_grp_b + 1) / 2

    non_zero_array = grp_a_counts[np.nonzero(grp_a_counts)]
    zero_array = np.repeat([0], n_truncated_grp_a - n_non_zero_grp_a)
    truncated_counts = np.concatenate((zero_array, non_zero_array))

    non_zero_array = grp_b_counts[np.nonzero(grp_b_counts)]
    zero_array = np.repeat([0], n_truncated_grp_b - n_non_zero_grp_b)
    truncated_counts = np.concatenate((truncated_counts, zero_array, non_zero_array), dtype=float)

    n_trun = n_truncated_grp_a + n_truncated_grp_b + 1
    ranks = n_trun - rankdata(truncated_counts, method="average")
    r: float = ranks[indices_seq].sum()

    s: float = r - n_truncated_grp_a * n_trun / 2

    variance: float = calculate_variance_ziw(data_dict, prop_mean, data_dict["n_tot"])

    if variance == 0:
        w: float = 0
    else:
        w: float = s / math.sqrt(variance)

    p_value = 2 * norm.sf(abs(w))

    log2fold_change = np.log2(np.mean(grp_a.loc[tag].values + 1) / np.mean(grp_b.loc[tag].values + 1))

    return tag, np.round(w, N_ROUND_TEST), np.round(log2fold_change, N_ROUND_TEST), p_value


def calculate_pre_statistics_for_ziw(data_dict: dict) -> dict:
    data_dict["n_obs_grp_a"] = sum(1 for sample in data_dict.values() if sample == "A")
    data_dict["n_obs_grp_b"] = sum(1 for sample in data_dict.values() if sample == "B")
    data_dict["n_tot"] = data_dict["n_obs_grp_a"] + data_dict["n_obs_grp_b"]
    data_dict["n_ziw"] = (data_dict["n_tot"] + 1) / 2
    data_dict["product_n_a_n_b"] = data_dict["n_obs_grp_a"] * data_dict["n_obs_grp_b"]
    data_dict["sum_square_n_a_n_b"] = data_dict["n_obs_grp_a"]**2 + data_dict["n_obs_grp_b"]**2
    data_dict["sqrt_product_n_a_n_b"] = math.sqrt(data_dict["n_obs_grp_a"] * data_dict["n_obs_grp_b"])

    return data_dict


def create_data_dict(condition_file, test_type):
    data_dict = {}
    conditions = {}

    with open(condition_file, 'r') as file:
        for line in file:
            row = line.strip().split()
            sample_id = row[0]
            condition = row[1]
            conditions.setdefault(condition, []).append(sample_id)

    assigned_condition = 'A'
    for condition, samples in sorted(conditions.items()):
        for sample_id in samples:
            data_dict[sample_id] = assigned_condition
        assigned_condition = 'B' if assigned_condition == 'A' else 'A'

    if test_type == "ziw":
        data_dict = calculate_pre_statistics_for_ziw(data_dict)

    return data_dict


# Normalize function — broadcast multiply instead of column-by-column loop
def normalize(chunk, kmer_nb_dict, header_row, norm_factor):
    """
    kmer_nb_dict: pre-built dict {sample_id: total_kmer_count}.
    Reads NO files from disk — caller must build the dict once and pass it in.
    Uses broadcast multiply for all columns at once.
    """
    chunk.columns = header_row
    if all(chunk.iloc[0] == header_row):
        chunk = chunk.iloc[1:]

    # Build factors array aligned to chunk columns
    factors = np.array([
        norm_factor / kmer_nb_dict[col] if col in kmer_nb_dict else 1.0
        for col in chunk.columns
    ], dtype=np.float64)

    # Check which columns need normalization
    needs_norm = np.array([col in kmer_nb_dict for col in chunk.columns])
    if needs_norm.any():
        vals = chunk.values.copy()
        # Broadcast multiply: (n_rows, n_cols) * (n_cols,)
        normed = vals * factors
        # Only round columns that were normalized
        normed[:, needs_norm] = np.round(normed[:, needs_norm], N_ROUND_NORM)
        # Preserve NaN handling
        null_mask = pd.isnull(chunk)
        result = pd.DataFrame(normed, index=chunk.index, columns=chunk.columns)
        result[null_mask] = chunk[null_mask]
        return result

    return chunk


# ......................................................
#   HELPER: format raw values for output
# ......................................................

def _format_tag_values(row_values):
    """Format a row of numeric values as a space-separated string."""
    parts = []
    for x in row_values:
        if isinstance(x, (int, float, np.number)) or str(x).replace('.', '', 1).isdigit():
            parts.append(f"{float(x):.2f}".rstrip('0').rstrip('.'))
        else:
            parts.append(str(x))
    return ' '.join(parts)


def _format_tag_values_batch(chunk_all_values, valid_idx):
    """Format multiple rows at once, returning a list of formatted strings."""
    results = []
    for k in valid_idx:
        row = chunk_all_values[k]
        parts = []
        for x in row:
            if isinstance(x, (int, float, np.number)) or str(x).replace('.', '', 1).isdigit():
                parts.append(f"{float(x):.2f}".rstrip('0').rstrip('.'))
            else:
                parts.append(str(x))
        results.append(' '.join(parts))
    return results


# Work function for the pool of processes
def work_for_parallel_processes(label_dict, data_chunk, cpm_normalization, header, test_type, covariates_df, norm_factor_c):

    if cpm_normalization:
        normalized_chunk = normalize(data_chunk, cpm_normalization, header, norm_factor_c)
    else:
        normalized_chunk = data_chunk
        normalized_chunk.columns = header

    grp_a_samples = [s for s, c in label_dict.items() if c == 'A']
    grp_b_samples = [s for s, c in label_dict.items() if c == 'B']

    grp_a_data = normalized_chunk[grp_a_samples]
    grp_b_data = normalized_chunk[grp_b_samples]

    # Extract numpy arrays once — avoids repeated .iloc[] / .loc[] calls
    grp_a_vals = grp_a_data.values.astype(np.float64)
    grp_b_vals = grp_b_data.values.astype(np.float64)
    chunk_all_values = normalized_chunk.values  # extracted once for tag formatting

    results = []

    # --------------------------------------------------
    # VECTORIZED PATHS — return raw arrays, zip in one pass
    # --------------------------------------------------

    if test_type == 'ttest':
        abs_t, fc, pv, valid_idx = perform_ttest_vectorized(grp_a_vals, grp_b_vals)
        tag_strs = _format_tag_values_batch(chunk_all_values, valid_idx)
        results = list(zip(abs_t.tolist(), tag_strs, fc.tolist(), pv.tolist()))

    elif test_type == 'wilcoxon':
        abs_s, fc, pv, valid_idx = perform_wilcoxon_vectorized(grp_a_vals, grp_b_vals)
        tag_strs = _format_tag_values_batch(chunk_all_values, valid_idx)
        results = list(zip(abs_s.tolist(), tag_strs, fc.tolist(), pv.tolist()))

    elif test_type == 'pitest':
        pi, fc, valid_idx = perform_pitest_vectorized(grp_a_vals, grp_b_vals)
        tag_strs = _format_tag_values_batch(chunk_all_values, valid_idx)
        results = list(zip(tag_strs, pi.tolist(), fc.tolist()))

    elif test_type == 'variance':
        chunk_vals = chunk_all_values.astype(np.float64)
        vars_, cvs, valid_idx = perform_variance_vectorized(chunk_vals)
        tag_strs = _format_tag_values_batch(chunk_all_values, valid_idx)
        results = list(zip(vars_.tolist(), cvs.tolist(), tag_strs))

    elif test_type == 'anova':
        for tag in data_chunk.index:
            result = perform_anova(tag, grp_a_data, grp_b_data, covariates_df)
            if result is not None:
                tag_values = _format_tag_values(data_chunk.loc[tag].values)
                results.append((abs(result[1]), tag_values, result[2], result[3]))

    # --------------------------------------------------
    # PER-ROW PATH (ZIW)
    # --------------------------------------------------

    elif test_type == "ziw":
        for tag in data_chunk.index:
            result = perform_ziw(tag, grp_a_data, grp_b_data, label_dict)
            tag_values = _format_tag_values(normalized_chunk.loc[tag].values)
            results.append((abs(result[1]), tag_values, result[2], result[3]))

    logging.info(f"Chunk processed: {len(data_chunk)} rows")
    return results
