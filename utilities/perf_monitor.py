"""
High-resolution performance measurement and aggregation utilities for
instrumented execution profiling.

This module provides tools to capture precise wall-clock time, CPU usage,
and memory (RSS) metrics for Python processes, record per-operation
performance deltas, and aggregate results for analysis and reporting.
It supports both lightweight dictionary-based aggregation and pandas-based
statistical processing with confidence interval computation.

Key capabilities include:
    - Nanosecond-resolution performance snapshots using monotonic and CPU timers.
    - Recording structured performance metrics to persistent CSV storage.
    - Aggregating duration, CPU time, and memory usage statistics per
      (event_id, operation_id) grouping.
    - Optional cold-start trimming for more stable benchmarking.
    - Generation of summary statistics (mean, median, std_dev, min, max, count).
    - Optional pandas-based aggregation with confidence interval estimation.

Designed for integration into data pipelines, simulation environments,
batch processing systems, and performance benchmarking workflows where
repeatable, structured measurement and analysis of execution behaviour
is required.

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
import os
import sys
import resource
import pandas as pd
from time import perf_counter_ns, process_time_ns
from typing import NamedTuple, Any
from collections import defaultdict

from utilities import toolbox as tb, mathbox as mb
from logistics import params_runtime


# =============================================================================
# CLASS: Performance snapshot
# =============================================================================
class PerformanceSnapshot(NamedTuple):
    """
    Snapshot of performance-related metrics captured at a single point in time.
    All values are expressed in nanoseconds unless stated otherwise.
    """

    perf_time_ns: int
    cpu_time_ns: int
    rss_bytes: int | None = None


# =============================================================================
# FUNCTION: Capture performance snapshot
# =============================================================================
def capture_performance() -> PerformanceSnapshot:
    """
    Captures a snapshot of high-resolution performance metrics for the current process.

    :return: A PerformanceSnapshot containing:
        - perf_time_ns: Monotonic, high-resolution wall-clock time in nanoseconds.
        - cpu_time_ns: CPU time consumed by the current process in nanoseconds
          (user + system time, excluding sleep).
        - rss_bytes: Resident Set Size of the process in bytes.
    """

    perf_time_ns = perf_counter_ns()
    cpu_time_ns = process_time_ns()
    rss_bytes=get_rss_bytes()

    snapshot = PerformanceSnapshot(
        perf_time_ns=perf_time_ns,
        cpu_time_ns=cpu_time_ns,
        rss_bytes=rss_bytes
    )

    return snapshot


# =============================================================================
# FUNCTION: Get resident set size
# =============================================================================
def get_rss_bytes() -> int:
    """
    Returns the resident set size (RSS) of the current process in bytes.

    On different platforms, the underlying value returned by the operating system
    may be expressed in different units:
        - kilobytes on Linux
        - bytes on macOS

    This function normalises the value to bytes.

    :return: Resident set size of the current process in bytes.
    """

    usage = resource.getrusage(resource.RUSAGE_SELF)

    # 'ru_maxrss' is:
    #   - kilobytes on Linux
    #   - bytes on macOS
    rss: int = usage.ru_maxrss

    if sys.platform == "darwin":  # macOS
        return rss
    else:  # Linux and most others
        return rss * 1024


# =============================================================================
# FUNCTION: Performance stats recorder
# =============================================================================
def record_performance_stats(
    start_snapshot: PerformanceSnapshot,
    end_snapshot: PerformanceSnapshot,
    num_cores = os.cpu_count(),
    category_label = None,
    operation_label = None,
    operation_id = None,
    event_id = None,
) -> None:
    """
    Computes performance deltas from two performance snapshots and persists the results
    to a CSV file.

    Both wall-clock duration and CPU time are recorded in nanoseconds, as well as
    derived values in microseconds and milliseconds. No truncation or rounding is
    applied; all derived values are lossless floating-point representations.

    :param start_snapshot: PerformanceSnapshot captured at the start of the operation.
    :param end_snapshot: PerformanceSnapshot captured at the end of the operation.
    :param num_cores: Number of CPU cores used for normalising CPU time. Must be >= 1.
                      If None, the system CPU count is used.
    :param category_label: High-level category of the operation.
    :param operation_label: Human-readable description of the operation.
    :param operation_id: Identifier for the operation being timed.
    :param event_id: Identifier for a higher-level event associated with this operation.
    """

    params = params_runtime.get_params()
    runtime = params_runtime.get_runtime()

    if num_cores is None:
        num_cores = os.cpu_count() or 1

    num_cores = max(1, num_cores)

    delta_perf_ns = end_snapshot.perf_time_ns - start_snapshot.perf_time_ns
    delta_cpu_ns = end_snapshot.cpu_time_ns - start_snapshot.cpu_time_ns

    # RSS is not a delta by default — snapshot at end is usually what you want
    rss_bytes = end_snapshot.rss_bytes

    row_dict = {
        "execution_id": runtime.execution_id,
        "category": (category_label or "N/A").upper(),
        "operation": operation_label or "N/A",
        "operation_id": operation_id,
        "event_id": event_id,

        # canonical timing
        "duration_nano_sec": delta_perf_ns,
        "cpu_time_nano_sec": delta_cpu_ns,

        # CPU normalisation assumes ideal parallel scaling and is provided for heuristic
        # comparison only.
        # CPU time per assumed core, not OS-normalised, not parallelism-aware
        "cpu_time_per_core_nano_sec": delta_cpu_ns / num_cores,

        # memory
        "rss_bytes": rss_bytes,

        # derived units (lossless)
        "duration_micro_sec": delta_perf_ns / 1_000,
        "cpu_time_micro_sec": delta_cpu_ns / 1_000,
        "duration_milli_sec": delta_perf_ns / 1_000_000,
        "cpu_time_milli_sec": delta_cpu_ns / 1_000_000,
    }

    tb.save_csv(
        rows_list=[row_dict],
        filename="perf_stats",
        save_path=params.logistics.anchor_path / "performance",
        file_mode="a",
    )


# =============================================================================
# FUNCTION: Process stats
# =============================================================================
def process_performance_stats(
    performance_data: list[dict[str, Any]],
    drop_first_n: int = 0,
) -> list[dict[str, Any]]:
    """
    Aggregate recorded performance metrics by (event_id, operation_id) and return
    flattened rows suitable for CSV writing.

    Grouping:
        - event_id
        - operation_id

    Notes:
        - execution_id is assumed to be the same for all rows. It is included in every
          output row, but it is not used as a grouping key.
        - event_id is normalised to an integer when present; if empty/missing it is kept
          as an empty string "".
        - operation_id is normalised to an integer.
        - duration_nano_sec and cpu_time_nano_sec are normalised to float.
        - rss_bytes is optional; empty values are stored as None and aggregated only if present.

    The resulting structure is flattened so each output dictionary represents a single
    CSV row with scalar values only. Statistic keys are prefixed to avoid collisions:
        - duration_<stat_name>
        - cpu_time_<stat_name>
        - rss_<stat_name>

    :param performance_data: List of dictionaries containing recorded performance metrics.
        Each dictionary is expected to contain at least:
            - "execution_id" (str): execution identifier (same for all rows).
            - "event_id" (str|int|float|None): event identifier; may be empty.
            - "operation_id" (str|int|float): operation identifier.
            - "operation" (str): operation name/label.
            - "duration_nano_sec" (str|int|float): duration in nanoseconds.
            - "cpu_time_nano_sec" (str|int|float): CPU time in nanoseconds.
        Optional:
            - "rss_bytes" (str|int|float|None): resident set size in bytes.
    :param drop_first_n: Number of first samples per (event_id, operation_id) group
        to drop before computing statistics. Useful for ignoring cold-start effects.
        Default = 0 (no samples dropped).
    :return: List of dictionaries, where each dictionary represents one aggregated row
        suitable for CSV writing. Each row contains:
            - execution_id
            - event_id
            - operation_id
            - operation
            - num_samples
            - duration_<stat> values from mb.compute_stats()
            - cpu_time_<stat> values from mb.compute_stats()
            - rss_<stat> values from mb.compute_stats() (only if any rss was present)
            - rss_present (bool indicating whether memory stats were available)
    """

    operation_data = defaultdict(lambda: defaultdict(list))

    execution_id = str(performance_data[0]["execution_id"]) if performance_data else ""

    # store operation string once per operation_id
    operation_names: dict[int, str] = {}

    for entry in performance_data:
        # event_id: int if present, else ""
        event_raw = entry.get("event_id", "")
        if event_raw in ("", None):
            event_id: Any = ""
        else:
            event_id = int(float(event_raw))

        # operation_id + operation name
        operation_id = int(float(entry["operation_id"]))
        operation_names[operation_id] = str(entry.get("operation", ""))

        # required numeric metrics
        duration = float(entry["duration_nano_sec"])
        cpu_time = float(entry["cpu_time_nano_sec"])

        # optional rss
        rss_raw = entry.get("rss_bytes")

        if rss_raw in ("", None):
            rss: float | None = None
        else:
            assert isinstance(rss_raw, (int, float, str))
            rss = float(rss_raw)

        operation_data[event_id][operation_id].append((duration, cpu_time, rss))

    rows_list: list[dict[str, Any]] = []

    for event_id, operations in operation_data.items():
        for operation_id, samples in operations.items():

            # ==================================================
            # Drop first N samples (cold start trimming)
            # ==================================================
            if drop_first_n > 0:
                if len(samples) <= drop_first_n:
                    # nothing meaningful left → skip this group entirely
                    tb.log_message(
                        5,
                        f"Skipping stats for (event ID, operation ID): "
                        f"({event_id or 'N/A'}, {operation_id})\n"
                        f"Not enough samples available for cold-start filtering."
                    )
                    continue
                samples = samples[drop_first_n:]

            durations = [d for d, _, _ in samples]
            cpu_times = [c for _, c, _ in samples]
            rss_values = [m for _, _, m in samples if m is not None]

            duration_stats = mb.compute_stats(durations)
            cpu_time_stats = mb.compute_stats(cpu_times)
            memory_stats = mb.compute_stats(rss_values) if rss_values else None

            row: dict[str, Any] = {
                "execution_id": execution_id,
                "event_id": event_id,
                "operation_id": operation_id,
                "operation": operation_names.get(operation_id, ""),
                "num_samples": len(samples),
            }

            # duration stats
            for k, v in (duration_stats or {}).items():
                row[f"duration_{k}"] = v

            # cpu stats
            for k, v in (cpu_time_stats or {}).items():
                row[f"cpu_time_{k}"] = v

            # memory stats
            if memory_stats:
                for k, v in memory_stats.items():
                    row[f"rss_{k}"] = v
                row["rss_present"] = True
            else:
                row["rss_present"] = False

            rows_list.append(row)

    return rows_list


# =============================================================================
# FUNCTION: Process stats (Pandas version)
# =============================================================================
def process_performance_stats_df(
    performance_data: list[dict[str, Any]]
) -> pd.DataFrame:
    """
    Processes timing and memory statistics grouped by (event_id, operation_id) using
    pandas aggregation, returning a DataFrame of flattened, CSV-friendly columns.

    Grouping:
        - event_id
        - operation_id

    Notes:
        - execution_id is assumed to be the same for all rows. It is included in the
          output but is not used as a grouping key.
        - event_id is normalised to an integer when present; if empty/missing it is kept
          as an empty string "".
        - operation_id is normalised to an integer.
        - operation (string) is included in output for readability (taken from the rows
          for the given operation_id).
        - rss_bytes is optional; if not present, it is excluded from aggregation.
        - Confidence intervals are computed from mean/std_dev/count via mb.compute_ci_from_stats_sp().

    Output columns are flattened as:
        <metric>_<stat>   e.g. duration_nano_sec_mean
    with std renamed to std_dev.

    :param performance_data: List of dictionaries containing recorded performance metrics.
        Each dictionary is expected to contain at least:
            - "execution_id" (str): execution identifier (same for all rows).
            - "event_id" (str|int|float|None): event identifier; may be empty.
            - "operation_id" (str|int|float): operation identifier.
            - "operation" (str): operation name/label.
            - "duration_nano_sec" (str|int|float): duration in nanoseconds.
            - "cpu_time_nano_sec" (str|int|float): CPU time in nanoseconds.
        Optional:
            - "rss_bytes" (str|int|float|None): resident set size in bytes.

    :return: pandas DataFrame with one row per (event_id, operation_id) containing:
        - execution_id
        - event_id
        - operation_id
        - operation
        - <metric>_<stat> columns for each aggregated metric
        - <metric>_ci_low and <metric>_ci_high confidence interval bounds
    :raises ValueError: If required columns are missing from the input data.
    """

    # ==================================================
    # DataFrame construction & validation
    # ==================================================
    perf_df = pd.DataFrame(performance_data)

    required_columns = {
        "execution_id",
        "event_id",
        "operation_id",
        "operation",
        "duration_nano_sec",
        "cpu_time_nano_sec",
    }

    missing = required_columns - set(perf_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # ==================================================
    # Type normalisation (match non-pandas behavior)
    # ==================================================
    # execution_id: keep as string
    perf_df["execution_id"] = perf_df["execution_id"].astype(str)

    # event_id: int if present, else ""
    # (handles "", None, NaN -> "")
    event_series = perf_df["event_id"]
    event_is_empty = event_series.isna() | (event_series.astype(str) == "")
    perf_df.loc[event_is_empty, "event_id"] = ""
    perf_df.loc[~event_is_empty, "event_id"] = (
        pd.to_numeric(perf_df.loc[~event_is_empty, "event_id"], errors="coerce").astype("Int64")
    )
    # convert Int64 to python objects so we can have both ints and "" in the column
    perf_df["event_id"] = perf_df["event_id"].astype(object)
    perf_df.loc[perf_df["event_id"].apply(lambda x: pd.isna(x)), "event_id"] = ""

    # operation_id: integer
    perf_df["operation_id"] = pd.to_numeric(perf_df["operation_id"], errors="coerce").astype("Int64")

    # required numeric metrics
    perf_df["duration_nano_sec"] = pd.to_numeric(perf_df["duration_nano_sec"], errors="coerce")
    perf_df["cpu_time_nano_sec"] = pd.to_numeric(perf_df["cpu_time_nano_sec"], errors="coerce")

    # optional rss
    has_rss = "rss_bytes" in perf_df.columns
    if has_rss:
        perf_df["rss_bytes"] = pd.to_numeric(perf_df["rss_bytes"], errors="coerce")

    # operation label: keep as string
    perf_df["operation"] = perf_df["operation"].astype(str)

    # execution_id: take first value (same assumption as non-pandas)
    execution_id = perf_df["execution_id"].iloc[0]

    # ==================================================
    # Pandas aggregation
    # ==================================================
    metrics: list[str] = ["duration_nano_sec", "cpu_time_nano_sec"]
    if has_rss:
        metrics.append("rss_bytes")

    # NOTE: pandas uses "std" not "std_dev". If you want the output column to be *_std_dev,
    # compute with "std" and rename after flattening.
    agg_map = {metric: ["count", "mean", "median", "std", "min", "max"] for metric in metrics}

    grouped_stats = (
        perf_df
        .groupby(["event_id", "operation_id"], dropna=False)
        .agg(agg_map)
    )

    # Flatten columns: (metric, stat) -> f"{metric}_{stat}"
    grouped_stats.columns = [f"{metric}_{stat}" for metric, stat in grouped_stats.columns]

    # Rename std -> std_dev for consistency with your naming convention
    grouped_stats.rename(
        columns={f"{m}_std": f"{m}_std_dev" for m in metrics},
        inplace=True,
    )

    grouped_stats.reset_index(inplace=True)

    # ==================================================
    # Add execution_id + operation name (match non-pandas output)
    # ==================================================
    grouped_stats.insert(0, "execution_id", execution_id)

    # Map operation_id -> operation name (assumes stable mapping)
    op_map = (
        perf_df[["operation_id", "operation"]]
        .dropna(subset=["operation_id"])
        .drop_duplicates(subset=["operation_id"])
        .set_index("operation_id")["operation"]
        .to_dict()
    )
    grouped_stats["operation"] = grouped_stats["operation_id"].map(op_map).fillna("")

    # Keep operation right after operation_id (as requested)
    cols = grouped_stats.columns.tolist()
    if "operation" in cols:
        cols.remove("operation")
        op_id_idx = cols.index("operation_id")
        cols.insert(op_id_idx + 1, "operation")
        grouped_stats = grouped_stats[cols]

    # ==================================================
    # Confidence interval computation (SciPy-based helper)
    # ==================================================
    for metric in metrics:
        ci_low_col = f"{metric}_ci_low"
        ci_high_col = f"{metric}_ci_high"

        ci_lows: list[float | None] = []
        ci_highs: list[float | None] = []

        for _, row in grouped_stats.iterrows():
            mean_val = row[f"{metric}_mean"]
            std_dev = row[f"{metric}_std_dev"]
            count = int(row[f"{metric}_count"])

            ci_low, ci_high = mb.compute_ci_from_stats_sp(
                mean_val=mean_val,
                std_dev=std_dev,
                n=count,
            )

            ci_lows.append(float("nan") if ci_low is None else ci_low)
            ci_highs.append(float("nan") if ci_high is None else ci_high)

        grouped_stats[ci_low_col] = ci_lows
        grouped_stats[ci_high_col] = ci_highs

    return grouped_stats
