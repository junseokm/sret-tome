"""
Descriptive statistics and confidence interval utilities for numeric samples.

This module provides small, reusable helpers to compute common summary
statistics (count, mean, median, std_dev, min, max) and optional confidence
intervals for the mean. Implementations are provided using both Python’s
standard library and NumPy, with confidence intervals available via:
    - a Normal-approximation (z-based) method, and
    - a Student’s t-distribution method (SciPy) for small-sample correctness.

These functions are intended for lightweight benchmarking, experiment
summaries, and performance reporting pipelines where consistent, CSV-friendly
statistical outputs are needed.

Copyright (c) 2024-2026, Uraz Odyurt

This source code is licenced under the BSD-style licence found in the
LICENCE file in the root directory of this source tree.
"""

__author__ = "Uraz Odyurt"
__copyright__ = "Copyright 2024-2026"
__credits__ = ["N/A"]
__license__ = "BSD-3-Clause"
__version__ = "1.0.0"
__maintainer__ = "N/A"
__email__ = "N/A"
__status__ = "Prototype"

# =============================================================================
import math
import statistics
import numpy as np
from scipy.stats import t
from typing import Iterable, Any, Optional


# =============================================================================
# FUNCTION: Compute stats
# =============================================================================
def compute_stats(
    values: Iterable[float],
    confidence_level: Optional[float] = 0.95
)-> dict[str, Any]:
    """
    Computes descriptive statistics and an optional confidence interval for a sequence of
    numeric values using Python's standard library.

    Statistics computed:
        - count
        - mean
        - median
        - standard deviation (sample, ddof=1)
        - minimum
        - maximum
        - ci_low, ci_high, confidence_level (if confidence_level is not None/0)

    If the input contains fewer than two values, the standard deviation is
    reported as 0.0 and the confidence interval is not computed.

    :param values: Iterable of numeric values. Must contain at least one element.
    :param confidence_level: Confidence level for CI (default 0.95).
                             Set to None or 0 to skip CI computation.
    :return: Dictionary containing computed statistics, with optional CI fields.
    :raises ValueError: If values is empty.
    """

    values_list = list(values)
    count = len(values_list)
    if count == 0:
        raise ValueError("Cannot compute statistics on empty data")

    mean_val = statistics.mean(values_list)
    median_val = statistics.median(values_list)
    min_val = min(values_list)
    max_val = max(values_list)

    std_dev = statistics.stdev(values_list) if count > 1 else 0.0

    stats: dict[str, Any] = {
        "count": count,
        "mean": mean_val,
        "median": median_val,
        "std_dev": std_dev,
        "min": min_val,
        "max": max_val
    }

    # Compute CI only if explicitly requested
    if confidence_level and count > 1:
        ci_low, ci_high = compute_ci(values_list, confidence_level)
        stats.update({
            "ci_low": ci_low,
            "ci_high": ci_high,
            "confidence_level": confidence_level
        })

    return stats


# =============================================================================
# FUNCTION: Compute stats (NumPy version)
# =============================================================================
def compute_stats_np(
    values: Iterable[float],
    confidence_level: Optional[float] = 0.95
) -> dict[str, Any]:
    """
    Computes descriptive statistics and an optional confidence interval for a sequence of
    numeric values using NumPy.

    Statistics computed:
        - count
        - mean
        - median
        - standard deviation (sample, ddof=1)
        - minimum
        - maximum
        - ci_low, ci_high, confidence_level (if confidence_level is not None/0)

    If the input contains fewer than two values, the standard deviation is
    reported as 0.0 and the confidence interval is not computed.

    :param values: Iterable of numeric values. Must contain at least one element.
    :param confidence_level: Confidence level for CI (default 0.95).
                             Set to None or 0 to skip CI computation.
    :return: Dictionary containing computed statistics, with optional CI fields.
    :raises ValueError: If values is empty.
    """

    array = np.asarray(values, dtype=float)
    count = array.size
    if count == 0:
        raise ValueError("Cannot compute statistics on empty data")

    mean_val = float(array.mean())
    median_val = float(np.median(array))
    min_val = float(array.min())
    max_val = float(array.max())

    std_dev = float(array.std(ddof=1)) if count > 1 else 0.0

    stats: dict[str, Any] = {
        "count": count,
        "mean": mean_val,
        "median": median_val,
        "std_dev": std_dev,
        "min": min_val,
        "max": max_val
    }

    # Compute CI only if requested
    if confidence_level and count > 1:
        ci_low, ci_high = compute_ci(array, confidence_level)
        stats.update({
            "ci_low": ci_low,
            "ci_high": ci_high,
            "confidence_level": confidence_level
        })

    return stats


# =============================================================================
# FUNCTION: Compute CI
# =============================================================================
def compute_ci(
    values: Iterable[float],
    confidence_level: float = 0.95
) -> tuple[Optional[float], Optional[float]]:
    """
    Computes a confidence interval for the mean of a sequence of numeric values
    using a Normal approximation (z-based).

    The interval is defined as:
        mean ± z_(α/2) * (std_dev / sqrt(n))

    If fewer than two values are provided, the confidence interval cannot be
    computed and (None, None) is returned.

    :param values: Iterable of numeric values. Must contain at least two elements.
    :param confidence_level: Confidence level for the interval (e.g., 0.95 for 95% CI).
    :return: Tuple (lower_bound, upper_bound) representing the confidence interval,
        or (None, None) if the interval cannot be computed.
    """

    values_list = list(values)
    n = len(values_list)
    if n < 2:
        return None, None

    mean_val = statistics.mean(values_list)
    std_dev = statistics.stdev(values_list)

    # Two-sided alpha
    alpha = 1.0 - confidence_level

    # Approximate t critical value using normal distribution
    # (acceptable for n >= ~10; conservative for smaller n)
    z = statistics.NormalDist().inv_cdf(1.0 - alpha / 2.0)

    margin = z * (std_dev / math.sqrt(n))

    lower_bound = mean_val - margin
    upper_bound = mean_val + margin

    return lower_bound, upper_bound


# =============================================================================
# FUNCTION: Compute CI from mean and standard deviation
# =============================================================================
def compute_ci_from_stats(
    mean_val: float,
    std_dev: float,
    n: int,
    confidence_level: float = 0.95
) -> tuple[Optional[float], Optional[float]]:
    """
    Computes a confidence interval for the mean using a Normal approximation (z-based).

    :param mean_val: Sample mean.
    :param std_dev: Sample standard deviation.
    :param n: Number of samples.
    :param confidence_level: Confidence level for the interval (e.g., 0.95 for 95% CI).
    :return: Tuple (lower_bound, upper_bound) representing the confidence interval,
        or (None, None) if the interval cannot be computed.
    """

    if n < 2:
        return None, None

    alpha = 1.0 - confidence_level

    # Approximate t critical value using normal distribution
    # (acceptable for n >= ~10; conservative for smaller n)
    z = statistics.NormalDist().inv_cdf(1.0 - alpha / 2.0)

    margin = z * (std_dev / math.sqrt(n))

    lower_bound = mean_val - margin
    upper_bound = mean_val + margin

    return lower_bound, upper_bound


# =============================================================================
# FUNCTION: Compute CI from mean and standard deviation (SciPy version)
# =============================================================================
def compute_ci_from_stats_sp(
    mean_val: float,
    std_dev: float,
    n: int,
    confidence_level: float = 0.95
) -> tuple[Optional[float], Optional[float]]:
    """
    Computes a confidence interval for the mean using Student's t-distribution.

    :param mean_val: Sample mean.
    :param std_dev: Sample standard deviation.
    :param n: Number of samples.
    :param confidence_level: Confidence level for the interval (e.g., 0.95 for 95% CI).
    :return: Tuple (lower_bound, upper_bound) representing the confidence interval,
        or (None, None) if the interval cannot be computed.
    """

    if n < 2:
        return None, None

    alpha = 1.0 - confidence_level
    t_crit = t.ppf(1.0 - alpha / 2.0, df=n - 1)

    margin = t_crit * (std_dev / math.sqrt(n))

    lower_bound = mean_val - margin
    upper_bound = mean_val + margin

    return lower_bound, upper_bound
