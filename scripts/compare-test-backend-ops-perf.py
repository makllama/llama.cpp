#!/usr/bin/env python3

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Tuple, Union

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


def parse_benchmark_line(line: str) -> Tuple[Union[str, None], Union[float, None]]:
    """
    Parses a single line of benchmark output.

    Example line:
    MUL_MAT(...): 744 runs - 1660.11 us/run - 134.48 MFLOP/run - 81.01 GFLOPS

    Returns a tuple of (key, gflops) or (None, None) if parsing fails.
    """
    line = line.strip()
    if ":" not in line:
        return None, None

    key, data_part = line.split(":", 1)
    key = key.strip()

    # Remove ANSI color codes from the data part
    data_part = re.sub(r"\x1b\[[0-9;]*m", "", data_part)

    # Find the last number and unit in the data part
    match = re.search(r"([\d\.]+)\s+(GFLOPS|TFLOPS|MFLOPS)\s*$", data_part.strip())
    if not match:
        return None, None

    value_str, unit = match.groups()
    value = float(value_str)

    # Normalize everything to GFLOPS
    if unit == "TFLOPS":
        gflops = value * 1000
    elif unit == "MFLOPS":
        gflops = value / 1000
    elif unit == "GFLOPS":
        gflops = value
    else:
        assert False

    return key, gflops


def extract_commit_id(filepath: Path) -> str:
    """Extract commit ID from filename like test-backend-ops-perf-abc1234.log"""
    filename = filepath.name
    # Pattern: test-backend-ops-perf-<commit_id>.log
    match = re.match(r"test-backend-ops-perf-([^.]+)\.log", filename)
    if match:
        return match.group(1)
    return ""


def load_results(filepath: Path) -> dict:
    """Loads all benchmark results from a file into a dictionary."""
    results = {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                key, gflops = parse_benchmark_line(line)
                if key:
                    results[key] = gflops
    except FileNotFoundError:
        logger.error(f"Error: File not found at {filepath}")
        sys.exit(1)
    return results


def format_change(change: float) -> str:
    """Formats the percentage change."""
    if change > 0.1:
        return f"+{change:.2f}%"
    elif change < -0.1:
        return f"{change:.2f}%"
    else:
        return " ~0.00%"


def main():
    """Main function to compare benchmark files."""
    parser = argparse.ArgumentParser(
        description="Compare two benchmark result files and generate a report.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    help_b = "Path to the baseline benchmark results file."
    parser.add_argument(
        "-b", "--baseline", dest="baseline", type=Path, required=True, help=help_b
    )
    help_c = "Path to the benchmark results file to compare against the baseline."
    parser.add_argument(
        "-c", "--compare", dest="compare", type=Path, required=True, help=help_c
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default="comparison_backend_ops_perf.txt",
        help="Path to the output report file (default: comparison_backend_ops_perf.txt).",
    )
    args = parser.parse_args()

    logger.info(f"Loading baseline results from: {args.baseline}")
    baseline_results = load_results(args.baseline)
    logger.info(f"Loading compare results from: {args.compare}")
    compare_results = load_results(args.compare)

    if not baseline_results or not compare_results:
        logger.error("Could not load results from one or both files. Exiting.")
        return

    # Extract commit IDs from filenames
    baseline_commit = extract_commit_id(args.baseline)
    compare_commit = extract_commit_id(args.compare)

    all_keys = sorted(list(set(baseline_results.keys()) | set(compare_results.keys())))

    comparisons = []

    for key in all_keys:
        baseline_val = baseline_results.get(key)
        compare_val = compare_results.get(key)

        entry = {
            "key": key,
            "baseline": baseline_val,
            "compare": compare_val,
            "change": 0,
        }

        if baseline_val is not None and compare_val is not None:
            entry["change"] = ((compare_val - baseline_val) / baseline_val) * 100

        comparisons.append(entry)

    # --- Generate Report ---
    with open(args.output, "w", encoding="utf-8") as f:

        # Create header with commit IDs extracted from filenames
        baseline_header = "Baseline GFLOPS"
        compare_header = "Compare GFLOPS"

        if baseline_commit:
            baseline_header = f"Baseline ({baseline_commit}) GFLOPS"
        if compare_commit:
            compare_header = f"Compare ({compare_commit}) GFLOPS"

        key_width = max(len(k) for k in all_keys) + 2
        header = f"{'Test Configuration':<{key_width}} {baseline_header:>25} {compare_header:>25} {'Change (%)':>15}"
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")

        for item in comparisons:
            baseline_str = (
                f"{item['baseline']:.2f}" if item["baseline"] is not None else "N/A"
            )
            compare_str = (
                f"{item['compare']:.2f}" if item["compare"] is not None else "N/A"
            )
            change_str = format_change(item["change"])
            f.write(
                f"{item['key']:<{key_width}} {baseline_str:>25} {compare_str:>25} {change_str:>15}\n"
            )

    logger.info(f"Comparison report successfully generated at: {args.output}")


if __name__ == "__main__":
    main()
