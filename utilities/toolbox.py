"""
General-purpose utility functions shared across projects.

This module groups frequently used helpers for:
    - Structured logging with action-based categories, optional terminal/file output,
      and a multiprocessing-safe logging pipeline (queue + listener process).
    - Common filesystem and configuration path utilities.
    - Lightweight persistence helpers for writing text logs, JSON, and CSV files
      (including execution_id-based filenames).
    - CSV loading and basic numeric coercion utilities for data ingestion.

The functions here are intended to be small, dependency-light building blocks
that are reused across pipelines, batch jobs, and multiprocessing workflows.

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
import sys
import csv
import math
import json
import hashlib
import multiprocessing
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Any, Iterable

from logistics import params_runtime


# =============================================================================
# FUNCTION: Message logger
# =============================================================================
def log_message(
    action_id: int,
    message: str
) -> None:
    """
    Logs a formatted message based on the provided action identifier.

    Output behaviour is controlled by runtime configuration:
        - params.flags.log_to_terminal
        - params.flags.log_to_file
        - params.flags.debug_flag

    Log files are written under '<anchor_path>/log' and include the current
    runtime execution_id in the filename.

    :param action_id: Identifier determining the log message category.
        Supported values:
            1  - Done
            2  - Debug
            3  - In progress
            4  - Finished
            5  - Warning
            6  - Error
            11 - Completion progress
            13 - Major error (terminates execution)
    :param message: The message content to log. Can be multi-line.
    """

    params = params_runtime.get_params()

    # Map action IDs to labels and fatality
    action_map = {
        1: ("Done", False),
        2: ("Debug", False),
        3: ("In progress", False),
        4: ("Finished", False),
        5: ("Warning", False),
        6: ("Error", False),
        11: ("Completion progress", False),
        13: ("Major error", True),
    }

    label, is_fatal = action_map.get(
        action_id,
        (f"Unknown action_id ({action_id})", False)
    )

    timestamp = get_current_timestamp()

    # Indent message lines
    formatted_message = "    " + message.replace("\n", "\n    ")

    writable = (
        f"{label}: [{timestamp}]\n"
        f"{formatted_message}\n\n"
    )

    # Terminal output
    if params.flags.log_to_terminal:
        print(writable)

    # File output
    if params.flags.log_to_file:
        log_filename = "debug" if params.flags.debug_flag else "log"
        log_path = params.logistics.anchor_path / "log"
        write_to_file(writable, log_filename, log_path)

    # Fatal error handling
    if is_fatal:
        raise SystemExit(1)


# =============================================================================
# FUNCTION: Message logger (multiprocessing)
# =============================================================================
def log_message_multiproc(
    action_id: int,
    message: str,
    log_queue: multiprocessing.Queue
) -> None:
    """
    Formats a log message, using the same style as log_message(), and sends it
    to a multiprocessing-safe queue for the listener process to handle.

    The listener process is responsible for printing to terminal and/or writing
    to file according to runtime configuration.

    If action_id == 13, this raises SystemExit(1) after enqueueing the message
    (terminates the current process only).

    :param action_id: Identifier determining the log message category.
        Supported values:
            1  - Done
            2  - Debug
            3  - In progress
            4  - Finished
            5  - Warning
            6  - Error
            11 - Completion progress
            13 - Major error (terminates the current process)
    :param message: The message content to log. Can be multi-line; each line
        will be indented in the output.
    :param log_queue: Multiprocessing Queue used to pass formatted log messages
        from worker processes to the dedicated listener process.
    """

    action_map = {
        1: ("Done", False),
        2: ("Debug", False),
        3: ("In progress", False),
        4: ("Finished", False),
        5: ("Warning", False),
        6: ("Error", False),
        11: ("Completion progress", False),
        13: ("Major error", True),
    }

    label, is_fatal = action_map.get(
        action_id,
        (f"Unknown action_id ({action_id})", False)
    )

    timestamp = get_current_timestamp()

    # Indent message lines
    formatted_message = "    " + message.replace("\n", "\n    ")

    writable = (
        f"{label}: [{timestamp}]\n"
        f"{formatted_message}\n\n"
    )

    log_queue.put(writable)

    # Fatal error handling
    if is_fatal:
        raise SystemExit(1)


# =============================================================================
# FUNCTION: Start log listener process
# =============================================================================
def start_logging_process() -> tuple[multiprocessing.Queue, multiprocessing.Process]:
    """
    Creates a multiprocessing Queue and starts a dedicated listener process
    that consumes queued log messages and performs terminal/file output.

    Output behaviour is controlled by runtime configuration:
        - params.flags.log_to_terminal
        - params.flags.log_to_file
        - params.flags.debug_flag

    File output is written under '<anchor_path>/log'.

    :return: A tuple (log_queue, listener):
        - log_queue: The multiprocessing Queue used to send log messages to the listener.
        - listener: The multiprocessing Process running log_listener().
    """

    log_queue = multiprocessing.Queue()
    listener = multiprocessing.Process(
        target=log_listener,
        args=(log_queue,),
        name="log-listener"
    )
    listener.start()

    return log_queue, listener


# =============================================================================
# FUNCTION: Stop log listener process
# =============================================================================
def stop_logging_process(
    log_queue: multiprocessing.Queue,
    listener: multiprocessing.Process,
    join_timeout: float = 5.0
) -> None:
    """
    Signals the log listener process to terminate and performs clean-up.

    This function sends a sentinel (None) to the queue, waits for the listener
    to exit, and terminates it forcefully if it does not stop within the timeout.
    The queue is closed and its background thread is joined.

    :param log_queue: The multiprocessing Queue previously returned by start_logging_process().
    :param listener: The listener Process previously returned by start_logging_process().
    :param join_timeout: Maximum number of seconds to wait for the listener to exit
        after sending the sentinel. If exceeded, the listener is terminated.
    """

    # Send a shutdown signal to the log listener
    try:
        log_queue.put(None)  # sentinel
    finally:
        listener.join(timeout=join_timeout)
        if listener.is_alive():
            listener.terminate()
            listener.join()

        # Optional but nice clean-up
        try:
            # Clean up the queue explicitly
            log_queue.close()
            log_queue.join_thread()
        except (OSError, ValueError, AssertionError):
            # Don't let clean-up errors crash shutdown
            # Queue may already be closed or process state gone — safe to ignore
            pass


# =============================================================================
# FUNCTION: Log listener (runs in separate process)
# =============================================================================
def log_listener(log_queue: multiprocessing.Queue) -> None:
    """
    Dedicated listener loop that consumes formatted log messages from a queue
    and outputs them according to runtime configuration.

    Terminal output:
        - If params.flags.log_to_terminal is True, prints each record.

    File output:
        - If params.flags.log_to_file is True, writes using write_to_file().
        - Uses filename "debug" if params.flags.debug_flag else "log".
        - Uses '<anchor_path>/log' as the save directory.

    The listener exits when it receives a sentinel value of None.

    :param log_queue: Multiprocessing Queue from which to read formatted log message strings.
        A value of None, signals shutdown.
    """

    params = params_runtime.get_params()
    log_path = params.logistics.anchor_path / "log"
    filename = "debug" if params.flags.debug_flag else "log"

    while True:
        try:
            record = log_queue.get()
            if record is None:
                break

            if params.flags.log_to_terminal:
                print(record, end="")

            if params.flags.log_to_file:
                write_to_file(record, filename, log_path)

        except Exception as e:
            print(f"Logging error: {e}", file=sys.stderr)


# =============================================================================
# FUNCTION: Formatted timestamp
# =============================================================================
def get_current_timestamp() -> str:
    """
    Returns a formatted timestamp suitable for log messages.

    :return: Timestamp in HH:MM:SS.mmm format
    """

    return datetime.now().time().isoformat(timespec="milliseconds")


# =============================================================================
# FUNCTION: Project config path
# =============================================================================
def get_default_config_path() -> Path:
    """
    Return the default configuration directory within the project structure.

    The configuration directory is defined relative to this module's location.
    The project root is taken as three directory levels above this file, and
    the configuration directory is expected at:

        <project_root>/config

    :return: Absolute path to the default configuration directory.
    """

    # Get this script's path (toolbox.py)
    current_path = Path(__file__).resolve()
    # Navigate 3 levels up to reach the project root
    project_root = current_path.parents[2]
    config_path = project_root / "config"

    return config_path


# ==============================================================================
# FUNCTION: Text file writer
# =============================================================================
def write_to_file(
    writable: str,
    filename: str,
    save_path: Path,
    file_mode: str = "a",
    include_execution_id: bool = True
) -> None:
    """
    Writes content to the specified file.

    By default, the output filename includes the current execution identifier in the form:
    `{filename}_{execution_id}.txt`. If `include_execution_id` is False, the filename is
    `{filename}.txt`.

    :param writable: The content to write to the file.
    :param filename: Base name of the text file (without extension).
    :param save_path: Directory path where the text file will be saved. Can be a `Path` or
        path-like object.
    :param file_mode: File mode to use. Defaults to 'a' (append).
        Supported values:
            'a' - append
            'w' - overwrite
    :param include_execution_id: If True, appends the current execution identifier. If False, the
        file will be saved without the execution identifier. Defaults to True.
    :raises ValueError: If file_mode is not 'a' or 'w'.
    :raises RuntimeError: If the file cannot be written.
    """

    if file_mode not in ("a", "w"):
        raise ValueError("Invalid file_mode. Use 'a' (append) or 'w' (write).")

    # Ensure save_path is a Path and exists
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    execution_id = params_runtime.get_runtime().execution_id

    if include_execution_id:
        text_file = save_path / f"{filename}_{execution_id}.txt"
    else:
        text_file = save_path / f"{filename}.txt"

    try:
        with text_file.open(mode=file_mode, encoding="utf-8") as file:
            file.write(writable)

    except OSError as exc:
        raise RuntimeError(
            f"Failed to write to file '{text_file}'"
        ) from exc


# =============================================================================
# FUNCTION: JSON file reader
# =============================================================================
def read_from_json(
    filename: str,
    load_path: Path
) -> list[Any]:
    """
    Reads data from a JSON file written by `write_to_json`.

    The writer always stores data as a JSON list. However, this reader
    also supports a single JSON object for robustness.

    :param filename: Base name of the JSON file (without '.json').
    :param load_path: Directory where the JSON file is located.
    :return: List of objects contained in the file.
    :raises FileNotFoundError: If the JSON file does not exist.
    :raises RuntimeError: If the file cannot be read.
    :raises TypeError: If the JSON structure is not supported.
    """

    load_path = Path(load_path)
    json_file = load_path / f"{filename}.json"

    if not json_file.exists():
        raise FileNotFoundError(f"JSON file not found: {json_file}")

    try:
        with json_file.open("r", encoding="utf-8") as f:
            payload = json.load(f)

    except OSError as exc:
        raise RuntimeError(
            f"Failed to read JSON file '{json_file}'"
        ) from exc

    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        # Support legacy / manual JSON files
        return [payload]

    raise TypeError(f"Unsupported JSON structure in '{json_file}'")


# =============================================================================
# FUNCTION: JSON file writer
# =============================================================================
def write_to_json(
    data: Iterable[Any],
    filename: str,
    save_path: Path,
    file_mode: str = "a",
    indent: int = 2,
    include_execution_id: bool = True
) -> None:
    """
    Writes data to a JSON file.

    Output filename:
        `{filename}_{execution_id}.json`

    If file_mode='a':
        - File is treated as a JSON list
        - Existing content is loaded and extended

    If file_mode='w':
        - File is overwritten with new data

    :param data: Iterable of JSON-serialisable items to write.
    :param filename: Base filename (no extension).
    :param save_path: Directory to save JSON file.
    :param file_mode: File mode to use. Defaults to 'a' (append).
        Supported values:
            'a' - append
            'w' - overwrite
    :param indent: JSON indentation level.
    :param include_execution_id: If True, appends the current execution identifier. If False, the
        file will be saved without the execution identifier. Defaults to True.
    :raises ValueError: If file_mode is not 'a' or 'w'.
    :raises RuntimeError: If the file cannot be written or append mode is used on a non-list file.
    :raises TypeError: If data contains non-serialisable values.
    """

    if file_mode not in ("a", "w"):
        raise ValueError("Invalid file_mode. Use 'a' or 'w'.")

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    execution_id = params_runtime.get_runtime().execution_id

    if include_execution_id:
        json_file = save_path / f"{filename}_{execution_id}.json"
    else:
        json_file = save_path / f"{filename}.json"

    try:
        if file_mode == "a" and json_file.exists():
            with json_file.open("r", encoding="utf-8") as f:
                existing = json.load(f)
            if not isinstance(existing, list):
                raise RuntimeError("JSON file must contain a list to append.")
            existing.extend(data)
            output = existing
        else:
            output = list(data)

        with json_file.open("w", encoding="utf-8") as f:
            json.dump(output, f, indent=indent)

    except OSError as exc:
        raise RuntimeError(
            f"Failed to write JSON file '{json_file}'"
        ) from exc


# =============================================================================
# FUNCTION: CSV file reader
# =============================================================================
def load_csv(
    filename: str,
    load_path: Path,
    delimiter: str = ",",
    coerce_numbers: bool = False
) -> list[dict[str, Any]]:
    """
    Loads a CSV file and returns a list of dictionaries (rows).

    :param filename: Base name of the CSV file (without '.csv').
    :param load_path: Directory where the CSV file is located.
    :param delimiter: CSV delimiter (default ';').
    :param coerce_numbers: If True, attempts to convert values to numeric;
        non-convertible values become math.nan.
    :return: List of row dictionaries.
    :raises FileNotFoundError: If the CSV file does not exist.
    :raises ValueError: If the CSV file contains no header row.
    """

    load_path = Path(load_path)
    csv_file = load_path / f"{filename}.csv"

    if not csv_file.exists():
        raise FileNotFoundError(f"File not found: {csv_file}")

    with csv_file.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if not reader.fieldnames:
            raise ValueError(f"No headers found in {csv_file}")

        rows: list[dict[str, Any]] = []
        for row in reader:
            if coerce_numbers:
                row = {k: coerce_numeric([v])[0] for k, v in row.items()}
            rows.append(row)

    return rows


# =============================================================================
# FUNCTION: Convert values to float
# =============================================================================
def coerce_numeric(
    values: list[Any]
) -> list[float]:
    """
    Converts values to floats, ignoring non-numeric entries.
    Non-convertible values become math.nan.

    :param values: List of values to convert.
    :return: List of floats, with non-convertible entries replaced by math.nan.
    """

    numeric_values = []
    for v in values:
        try:
            numeric_values.append(float(v))
        except (ValueError, TypeError):
            numeric_values.append(math.nan)
    return numeric_values


# =============================================================================
# FUNCTION: CSV file writer
# =============================================================================
def save_csv(
    rows_list,
    filename: str,
    save_path: Path,
    delimiter: str = ",",
    file_mode: str = "a",
    include_execution_id: bool = True
) -> None:
    """
    Saves a list of dictionary rows into a CSV file. Each dictionary represents a row, with keys
    used as column headers. Supports appending to an existing file or overwriting it.

    By default, the output filename includes the current execution identifier in the form:
    `{filename}_{execution_id}.csv`. If include_execution_id is False, the filename is `{filename}.csv`.

    :param rows_list: A list of dictionaries, where each dictionary represents a row to be saved in
        the CSV file. All dictionaries must share the same keys.
    :param filename: Base name of the CSV file (without extension).
    :param save_path: Directory path where the CSV file will be saved. Can be a `Path` or
        path-like object.
    :param delimiter: CSV delimiter (default ';').
    :param file_mode: File mode to use. Defaults to 'a' (append).
        Supported values:
            'a' - append
            'w' - overwrite
    :param include_execution_id: If True, appends the current execution identifier. If False, the
        file will be saved without the execution identifier. Defaults to True.
    :raises ValueError: If file_mode is invalid or rows do not all share the same keys.
    """

    if file_mode not in ("a", "w"):
        raise ValueError("Invalid file_mode. Supported modes are 'w' (write) and 'a' (append).")

    if not rows_list:
        # Nothing to write; silently return to avoid creating empty files
        return

    # Ensure "save_path" is a Path and exists
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    execution_id = params_runtime.get_runtime().execution_id

    if include_execution_id:
        csv_file = save_path / f"{filename}_{execution_id}.csv"
    else:
        csv_file = save_path / f"{filename}.csv"

    # Validate consistent keys across all rows
    first_row_keys = set(rows_list[0].keys())
    for row in rows_list:
        if set(row.keys()) != first_row_keys:
            raise ValueError("All rows must have the same keys to be written to CSV.")

    # Preserve original dictionary insertion order
    headers = list(rows_list[0].keys())

    # Determine whether to write header
    include_header = file_mode == 'w' or not csv_file.exists()

    with open(csv_file, file_mode, newline='', encoding='utf-8') as file:
        # noinspection PyTypeChecker
        writer = csv.DictWriter(
            file,
            fieldnames=headers,
            delimiter=delimiter,
            quoting=csv.QUOTE_NONNUMERIC,
        )

        if include_header:
            writer.writeheader()

        writer.writerows(rows_list)


# =============================================================================
# FUNCTION: DataFrame CSV writer
# =============================================================================
def save_dataframe_csv(
    df: pd.DataFrame,
    filename: str,
    save_path: Path,
    delimiter: str = ",",
    file_mode: str = "a",
    include_execution_id: bool = True,
) -> None:
    """
    Saves a pandas DataFrame into a CSV file.

    The output filename follows:
        `{filename}_{execution_id}.csv`

    If file_mode='a':
        - Data is appended to the file
        - Header is written only if file does not exist

    If file_mode='w':
        - File is overwritten
        - Header is always written

    :param df:
        DataFrame to write.
    :param filename:
        Base filename (without extension).
    :param save_path:
        Directory where the CSV file will be saved.
    :param delimiter: CSV delimiter (default ';')
    :param file_mode:
        File mode to use. Defaults to 'a'.
        Supported values:
            'a' - append
            'w' - overwrite
    :param include_execution_id:
        If True, appends execution identifier to filename.
        If False, saves as `{filename}.csv`.
    :raises ValueError:
        If file_mode is invalid.
    """

    if file_mode not in ("a", "w"):
        raise ValueError("Invalid file_mode. Use 'a' or 'w'.")

    if df.empty:
        # Avoid creating empty files
        return

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    execution_id = params_runtime.get_runtime().execution_id

    if include_execution_id:
        csv_file = save_path / f"{filename}_{execution_id}.csv"
    else:
        csv_file = save_path / f"{filename}.csv"

    include_header = file_mode == "w" or not csv_file.exists()

    df.to_csv(
        csv_file,
        mode=file_mode,
        index=False,
        header=include_header,
        sep=delimiter,
        quoting=1,  # csv.QUOTE_NONNUMERIC
        float_format="%.15f",
        encoding="utf-8",
    )


# =============================================================================
# FUNCTION: Derive random seed (8-bit)
# =============================================================================
def seed8(name: str) -> int:
    """
    Derive a deterministic 8-bit seed (0–255) from the global master seed and a name.

    The same master seed and name will always produce the same derived seed.
    Different names will produce independent seeds.

    :param name: Unique name for the component requiring a seed
                 (e.g., "shuffle_train", "model_init", "augment").
    :return: 8-bit integer seed in range 0–255.
    """

    master_seed = params_runtime.get_params().constants.master_seed
    s = f"{master_seed}_{name}".encode()
    digest = hashlib.sha256(s).digest()

    return digest[0]  # first byte → 0–255


# =============================================================================
# FUNCTION: Derive random seed (16-bit)
# =============================================================================
def seed16(name: str) -> int:
    """
    Derive a deterministic 16-bit seed (0–65535) from the global master seed and a name.

    Provides a larger seed space than seed8 while remaining compact and human-readable.
    The same master seed and name will always produce the same derived seed.

    :param name: Unique name for the component requiring a seed
                 (e.g., "shuffle_train", "model_init", "augment").
    :return: 16-bit integer seed in range 0–65535.
    """

    master_seed = params_runtime.get_params().constants.master_seed
    s = f"{master_seed}_{name}".encode()
    digest = hashlib.sha256(s).digest()

    return int.from_bytes(digest[:2], "big")  # first 2 bytes → 0–65535
