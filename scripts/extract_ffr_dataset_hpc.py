#!/usr/bin/env python3
"""Extract FFR values from HPC simulation results and create a dataset.

Scans directories with severity/slope/resistance parameters, extracts FFR values from
ffr.txt files, and saves results to CSV.

Folder format: 2026-04-07T21.29.48_severity0p41_slope0p1005_resistance17p90
where 'p' indicates the decimal point.

Usage:
    python scripts/extract_ffr_dataset_hpc.py --results-dir <path> --output ffr_dataset.csv
"""

import argparse
import os
import re
import sys
from pathlib import Path

import pandas as pd


def parse_decimal_param(param_str):
    """Parse decimal parameter where 'p' indicates the decimal point.

    Args:
        param_str: String like '0p41', '0p1005', '17p90'

    Returns:
        Float value
    """
    if "p" in param_str:
        parts = param_str.split("p")
        return float(f"{parts[0]}.{parts[1]}")
    return float(param_str)


def extract_params(folder_name):
    """Extract severity, slope, and resistance from folder name.

    Args:
        folder_name: String like '2026-04-07T21.29.48_severity0p41_slope0p1005_resistance17p90'

    Returns:
        (severity, slope, resistance) tuple as floats, or (None, None, None) if not found
    """
    severity_match = re.search(r"severity([0-9p]+)", folder_name)
    slope_match = re.search(r"slope([0-9p]+)", folder_name)
    resistance_match = re.search(r"resistance([0-9p]+)", folder_name)

    if severity_match and slope_match and resistance_match:
        severity = parse_decimal_param(severity_match.group(1))
        slope = parse_decimal_param(slope_match.group(1))
        resistance = parse_decimal_param(resistance_match.group(1))
        return severity, slope, resistance

    return None, None, None


def extract_ffr(ffr_path):
    """Extract FFR value from ffr.txt file.

    Args:
        ffr_path: Path to ffr.txt file

    Returns:
        FFR value as float, or None if not found
    """
    with open(ffr_path, "r") as f:
        content = f.read()
        ffr_match = re.search(r"FFR = .*?: ([\d.]+)", content)
        if ffr_match:
            return float(ffr_match.group(1))
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Extract FFR values from HPC simulation results and create dataset"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Path to directory containing severity/slope/resistance result folders",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ffr_dataset.csv",
        help="Output CSV filename (default: ffr_dataset.csv)",
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Directory not found: {results_dir}")
        sys.exit(1)

    data = []

    for folder in sorted(os.listdir(results_dir)):
        folder_path = results_dir / folder
        if (
            folder_path.is_dir()
            and "severity" in folder
            and "slope" in folder
            and "resistance" in folder
        ):
            severity, slope, resistance = extract_params(folder)
            ffr_file = folder_path / "ffr.txt"

            if (
                severity is not None
                and slope is not None
                and resistance is not None
                and ffr_file.exists()
            ):
                ffr = extract_ffr(ffr_file)
                if ffr is not None:
                    data.append(
                        {
                            "severity": severity,
                            "slope": slope,
                            "resistance": resistance,
                            "ffr": ffr,
                        }
                    )
                    print(
                        f"Extracted: severity={severity:.2f}, slope={slope:.4f}, "
                        f"resistance={resistance:.2f}, ffr={ffr:.6f}"
                    )
                else:
                    print(f"Warning: Could not extract FFR from {ffr_file}")
            else:
                print(f"Warning: Missing ffr.txt in {folder}")

    if not data:
        print("Error: No valid data extracted")
        sys.exit(1)

    df = pd.DataFrame(data)
    df = df.sort_values(["severity", "slope", "resistance"]).reset_index(drop=True)

    df.to_csv(args.output, index=False)
    print(f"\nDataset saved to {args.output}")
    print(f"Total rows: {len(df)}")
    print(df)


if __name__ == "__main__":
    main()
