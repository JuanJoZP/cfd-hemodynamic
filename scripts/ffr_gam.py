#!/usr/bin/env python3
"""Apply GAM (Generalized Additive Model) to FFR dataset.

Predictors: severity, slope
Dependent variable: ffr

Usage:
    python scripts/ffr_gam.py --input ffr_dataset.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from pygam import LinearGAM, s, l, te
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="GAM model for FFR prediction")
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
        default="ffr_gam_plot.png",
        help="Output plot filename (default: ffr_gam_plot.png)",
    )
    parser.add_argument(
        "--plot-latent",
        type=str,
        default="ffr_gam_latent_plot.png",
        help="Output plot for latent variable (default: ffr_gam_latent_plot.png)",
    )
    parser.add_argument(
        "--n-splines",
        type=int,
        default=10,
        help="Number of splines for smooth terms (default: 10)",
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

    df["reduccion_area"] = (
        4.437 * df["severity"].values ** 2 * (1.0 / df["slope"].values)
    )

    X = df[["severity", "slope"]].values
    y = df["ffr"].values

    gam = LinearGAM(
        s(0, n_splines=args.n_splines)
        + s(1, n_splines=args.n_splines)
        + te(0, 1, n_splines=args.n_splines)
    ).fit(X, y)

    y_pred = gam.predict(X)
    errors = y_pred - y
    mae = mean_absolute_error(y, y_pred)

    print(f"{'=' * 60}")
    print("GAM (Generalized Additive Model) Results")
    print(f"{'=' * 60}")
    print(f"\nModel specification:")
    print(f"  FFR ~ s(severity) + s(slope) + te(severity, slope)")
    print(f"  Number of splines: {args.n_splines}")
    print(f"\nModel summary:")
    print(gam.summary())
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

    fig2, ax2 = plt.subplots(figsize=(8, 5))

    reduccion_vals = df["reduccion_area"].values
    sort_idx = np.argsort(reduccion_vals)
    reduccion_sorted = reduccion_vals[sort_idx]
    ffr_actual_sorted = y[sort_idx]
    ffr_predicted_sorted = y_pred[sort_idx]

    ax2.scatter(
        reduccion_sorted,
        ffr_actual_sorted,
        color="blue",
        label="Actual",
        s=60,
        zorder=3,
    )
    ax2.plot(
        reduccion_sorted,
        ffr_predicted_sorted,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="Predicted",
        zorder=2,
    )

    ax2.set_xlabel("Reduccion Area (4.437 × severity² × 1/slope)")
    ax2.set_ylabel("FFR")
    ax2.set_title("FFR vs Reduccion Area (GAM Model)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.plot_latent, dpi=150, bbox_inches="tight")
    print(f"Latent variable plot saved to: {args.plot_latent}")

    if args.output:
        df_out = df.copy()
        df_out["ffr_predicted"] = y_pred
        df_out["error"] = y_pred - y
        df_out.to_csv(args.output, index=False)
        print(f"\nPredictions saved to: {args.output}")


if __name__ == "__main__":
    main()
