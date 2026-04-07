#!/usr/bin/env python3
"""Apply multivariate linear regression to FFR dataset.

Predictors: severity, slope
Dependent variable: ffr

Usage:
    python scripts/ffr_regression.py --input ffr_dataset.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description="Multivariate linear regression on FFR dataset"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="ffr_dataset.csv",
        help="Input CSV file (default: ffr_dataset.csv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV for predictions (optional)",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default="ffr_regression_plot.png",
        help="Output plot filename (default: ffr_regression_plot.png)",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        sys.exit(1)

    df = pd.read_csv(input_path)

    if not all(col in df.columns for col in ["severity", "slope", "ffr"]):
        print("Error: CSV must contain 'severity', 'slope', and 'ffr' columns")
        sys.exit(1)

    X = df[["severity", "slope"]].values
    y = df["ffr"].values

    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)
    errors = y_pred - y
    mae = mean_absolute_error(y, y_pred)

    print(f"{'=' * 60}")
    print("Linear Regression Results")
    print(f"{'=' * 60}")
    print(f"\nModel equation:")
    print(f"  FFR = {model.intercept_:.6f}")
    print(f"       + {model.coef_[0]:.6f} * severity")
    print(f"       + {model.coef_[1]:.6f} * slope")
    print(f"\nCoefficients:")
    print(f"  Intercept: {model.intercept_:.6f}")
    print(f"  Severity:  {model.coef_[0]:.6f}")
    print(f"  Slope:     {model.coef_[1]:.6f}")
    print(f"\nMetrics:")
    print(f"  R²:          {r2_score(y, y_pred):.6f}")
    print(f"  RMSE:        {np.sqrt(mean_squared_error(y, y_pred)):.6f}")
    print(f"  MAE:         {mae:.6f}")
    print(f"  Mean Error:  {np.mean(errors):.6f}")
    print(f"\nPredictions vs Actual:")
    print(
        f"{'Severity':>10} {'Slope':>10} {'Actual':>12} {'Predicted':>12} {'Error':>12}"
    )
    print(f"{'-' * 10} {'-' * 10} {'-' * 12} {'-' * 12} {'-' * 12}")
    for i in range(len(df)):
        print(
            f"{df['severity'].iloc[i]:>10.3f} {df['slope'].iloc[i]:>10.3f} "
            f"{y[i]:>12.6f} {y_pred[i]:>12.6f} {y_pred[i] - y[i]:>12.6f}"
        )
    print(f"{'=' * 60}")

    severities = sorted(df["severity"].unique())
    fig, axes = plt.subplots(
        1, len(severities), figsize=(5 * len(severities), 4), sharey=True
    )
    if len(severities) == 1:
        axes = [axes]

    for ax, sev in zip(axes, severities):
        mask = df["severity"] == sev
        slope_vals = df.loc[mask, "slope"].values
        ffr_actual = df.loc[mask, "ffr"].values
        ffr_predicted = y_pred[mask]

        sort_idx = np.argsort(slope_vals)
        slope_vals = slope_vals[sort_idx]
        ffr_actual = ffr_actual[sort_idx]
        ffr_predicted = ffr_predicted[sort_idx]

        ax.scatter(slope_vals, ffr_actual, color="blue", label="Actual", s=60, zorder=3)
        ax.plot(
            slope_vals,
            ffr_predicted,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label="Predicted",
            zorder=2,
        )

        ax.set_xlabel("Slope")
        ax.set_ylabel("FFR")
        ax.set_title(f"Severity = {sev}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.plot, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {args.plot}")

    if args.output:
        df_out = df.copy()
        df_out["ffr_predicted"] = y_pred
        df_out["error"] = y_pred - y
        df_out.to_csv(args.output, index=False)
        print(f"\nPredictions saved to: {args.output}")


if __name__ == "__main__":
    main()
