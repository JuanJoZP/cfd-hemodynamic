#!/usr/bin/env python3
"""Extract FFR values from simulation results and create a dataset.

Scans directories with severity/slope parameters, extracts FFR values from
ffr.txt files, and saves results to CSV.

Usage:
    python scripts/extract_ffr_dataset.py --results-dir <path> --output ffr_dataset.csv
"""

import argparse
import os
import re
import sys
from pathlib import Path

import pandas as pd


def extract_severity_slope(folder_name):
    """Extract severity and slope from folder name.

    Args:
        folder_name: String like '2026-04-06T11.46.46_severity025_slope005'

    Returns:
        (severity, slope) tuple as floats, or (None, None) if not found
    """
    severity_match = re.search(r"severity(\d+)", folder_name)
    slope_match = re.search(r"slope(\d+)", folder_name)

    if severity_match and slope_match:
        severity_str = severity_match.group(1)
        slope_str = slope_match.group(1)

        severity = (
            int(severity_str[0]) + int(severity_str[1:]) / (10 ** len(severity_str[1:]))
            if len(severity_str) > 1
            else int(severity_str)
        )
        slope = (
            int(slope_str[0]) + int(slope_str[1:]) / (10 ** len(slope_str[1:]))
            if len(slope_str) > 1
            else int(slope_str)
        )

        return severity, slope
    return None, None


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
        description="Extract FFR values from simulation results and create dataset"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        required=True,
        help="Path to directory containing severity/slope result folders",
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
        if folder_path.is_dir() and "severity" in folder and "slope" in folder:
            severity, slope = extract_severity_slope(folder)
            ffr_file = folder_path / "ffr.txt"

            if severity is not None and slope is not None and ffr_file.exists():
                ffr = extract_ffr(ffr_file)
                if ffr is not None:
                    data.append({"severity": severity, "slope": slope, "ffr": ffr})
                    print(
                        f"Extracted: severity={severity:.2f}, slope={slope:.3f}, ffr={ffr:.6f}"
                    )
                else:
                    print(f"Warning: Could not extract FFR from {ffr_file}")
            else:
                print(f"Warning: Missing ffr.txt in {folder}")

    if not data:
        print("Error: No valid data extracted")
        sys.exit(1)

    df = pd.DataFrame(data)
    df = df.sort_values(["severity", "slope"]).reset_index(drop=True)

    df.to_csv(args.output, index=False)
    print(f"\nDataset saved to {args.output}")
    print(f"Total rows: {len(df)}")
    print(df)


if __name__ == "__main__":
    main()
