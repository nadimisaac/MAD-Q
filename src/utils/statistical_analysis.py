"""Statistical analysis utilities for experiment results."""

from typing import List, Tuple, Optional
import numpy as np
from scipy import stats


def compute_confidence_interval(
    data: List[float],
    confidence: float = 0.95,
    method: str = 'bootstrap',
    n_bootstrap: int = 10000
) -> Tuple[float, float, float]:
    """
    Compute confidence interval for data.

    Args:
        data: List of data points
        confidence: Confidence level (default: 0.95)
        method: 'bootstrap' or 't-test'
        n_bootstrap: Number of bootstrap samples

    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    data_array = np.array(data)
    mean = np.mean(data_array)

    if method == 'bootstrap':
        # Bootstrap confidence interval
        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(data_array, size=len(data_array), replace=True)
            bootstrap_means.append(np.mean(sample))

        alpha = 1 - confidence
        lower = np.percentile(bootstrap_means, (alpha / 2) * 100)
        upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)

    elif method == 't-test':
        # T-distribution confidence interval
        sem = stats.sem(data_array)
        margin = sem * stats.t.ppf((1 + confidence) / 2, len(data_array) - 1)
        lower = mean - margin
        upper = mean + margin

    else:
        raise ValueError(f"Unknown method: {method}")

    return mean, lower, upper


def compute_win_rate_ci(
    wins: int,
    total: int,
    confidence: float = 0.95,
    method: str = 'wilson'
) -> Tuple[float, float, float]:
    """
    Compute confidence interval for win rate (binomial proportion).

    Args:
        wins: Number of wins
        total: Total number of games
        confidence: Confidence level
        method: 'wilson' (default) or 'normal'

    Returns:
        Tuple of (win_rate, lower_bound, upper_bound)
    """
    if total == 0:
        return 0.0, 0.0, 0.0

    p = wins / total

    if method == 'wilson':
        # Wilson score interval (better for small samples)
        z = stats.norm.ppf((1 + confidence) / 2)
        denominator = 1 + z**2 / total
        center = (p + z**2 / (2 * total)) / denominator
        margin = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator

        lower = max(0, center - margin)
        upper = min(1, center + margin)

    elif method == 'normal':
        # Normal approximation
        z = stats.norm.ppf((1 + confidence) / 2)
        margin = z * np.sqrt(p * (1 - p) / total)
        lower = max(0, p - margin)
        upper = min(1, p + margin)

    else:
        raise ValueError(f"Unknown method: {method}")

    return p, lower, upper


def compare_win_rates(
    wins1: int,
    total1: int,
    wins2: int,
    total2: int
) -> Tuple[float, str]:
    """
    Compare two win rates using chi-square test.

    Args:
        wins1: Wins for group 1
        total1: Total games for group 1
        wins2: Wins for group 2
        total2: Total games for group 2

    Returns:
        Tuple of (p_value, interpretation)
    """
    # Create contingency table
    losses1 = total1 - wins1
    losses2 = total2 - wins2

    contingency_table = np.array([
        [wins1, losses1],
        [wins2, losses2]
    ])

    # Chi-square test
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

    # Interpretation
    if p_value < 0.001:
        interp = "highly significant (p < 0.001)"
    elif p_value < 0.01:
        interp = f"significant (p = {p_value:.3f})"
    elif p_value < 0.05:
        interp = f"marginally significant (p = {p_value:.3f})"
    else:
        interp = f"not significant (p = {p_value:.2f})"

    return p_value, interp


def effect_size_cohens_d(group1: List[float], group2: List[float]) -> float:
    """
    Compute Cohen's d effect size for comparing two groups.

    Args:
        group1: Data for group 1
        group2: Data for group 2

    Returns:
        Cohen's d (standardized mean difference)
    """
    arr1 = np.array(group1)
    arr2 = np.array(group2)

    mean1 = np.mean(arr1)
    mean2 = np.mean(arr2)

    # Pooled standard deviation
    n1, n2 = len(arr1), len(arr2)
    var1 = np.var(arr1, ddof=1)
    var2 = np.var(arr2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    cohens_d = (mean1 - mean2) / pooled_std
    return cohens_d


def interpret_cohens_d(d: float) -> str:
    """
    Interpret Cohen's d effect size.

    Args:
        d: Cohen's d value

    Returns:
        Interpretation string
    """
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def paired_t_test(
    group1: List[float],
    group2: List[float],
    alternative: str = 'two-sided'
) -> Tuple[float, float]:
    """
    Perform paired t-test.

    Args:
        group1: Data for group 1
        group2: Data for group 2
        alternative: 'two-sided', 'less', or 'greater'

    Returns:
        Tuple of (t_statistic, p_value)
    """
    arr1 = np.array(group1)
    arr2 = np.array(group2)

    if len(arr1) != len(arr2):
        raise ValueError("Groups must have same length for paired t-test")

    t_stat, p_value = stats.ttest_rel(arr1, arr2, alternative=alternative)
    return t_stat, p_value


def independent_t_test(
    group1: List[float],
    group2: List[float],
    alternative: str = 'two-sided',
    equal_var: bool = True
) -> Tuple[float, float]:
    """
    Perform independent samples t-test.

    Args:
        group1: Data for group 1
        group2: Data for group 2
        alternative: 'two-sided', 'less', or 'greater'
        equal_var: Assume equal variance (default: True)

    Returns:
        Tuple of (t_statistic, p_value)
    """
    arr1 = np.array(group1)
    arr2 = np.array(group2)

    t_stat, p_value = stats.ttest_ind(arr1, arr2, alternative=alternative, equal_var=equal_var)
    return t_stat, p_value


def bonferroni_correction(p_values: List[float]) -> List[float]:
    """
    Apply Bonferroni correction for multiple comparisons.

    Args:
        p_values: List of p-values

    Returns:
        List of corrected p-values
    """
    n = len(p_values)
    corrected = [min(p * n, 1.0) for p in p_values]
    return corrected


def compute_summary_stats(data: List[float]) -> dict:
    """
    Compute comprehensive summary statistics.

    Args:
        data: List of data points

    Returns:
        Dictionary of statistics
    """
    arr = np.array(data)

    return {
        'n': len(arr),
        'mean': np.mean(arr),
        'median': np.median(arr),
        'std': np.std(arr, ddof=1),
        'min': np.min(arr),
        'max': np.max(arr),
        'q25': np.percentile(arr, 25),
        'q75': np.percentile(arr, 75),
        'iqr': np.percentile(arr, 75) - np.percentile(arr, 25)
    }
