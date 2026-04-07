#!/usr/bin/env python3
"""Compare multiple GAM models with different interaction terms for FFR prediction.

Models:
  1. te(0, 1) - tensor product
  2. s(0, by=1)
  3. s(1, by=0)
  4. s(area, by=0)  - where area = reduccion_area
  5. s(area, by=1)
  6. GLM: te(0, 1) + l(2) with binomial distribution and logit link
  7. LinearGAM: te(0, 1) + l(2) (same terms, no binomial/logit)

Usage:
    python scripts/ffr_gam_compare.py --input ffr_dataset.csv
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pygam import GAM, LinearGAM, l, s, te
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_latent_variable(severity, slope):
    """Compute relative area reduction = ΔA_abs / A_total_canal.
    A_total_canal = 138 * (1.57 + 1.2) = 382.26
    ΔA_abs = 2 * severity^2 * 1.4896^2 / slope
    """
    return (2.0 * severity**2 * 2.218804) / (slope * 382.26)


def main():
    parser = argparse.ArgumentParser(
        description="Compare multiple GAM models for FFR prediction"
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
        default="ffr_gam_compare_plot.png",
        help="Output plot filename (default: ffr_gam_compare_plot.png)",
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

    df["reduccion_area"] = compute_latent_variable(
        df["severity"].values, df["slope"].values
    )

    X_full = df[["severity", "slope"]].values
    X_area = df[["reduccion_area", "severity", "slope"]].values
    y = df["ffr"].values

    models = {}

    print("Fitting models...")

    # Model 1: te(0, 1) - tensor product only
    gam1 = LinearGAM(te(0, 1, n_splines=args.n_splines)).fit(X_full, y)
    y_pred1 = gam1.predict(X_full)
    models["te(0, 1)"] = {
        "y_pred": y_pred1,
        "r2": r2_score(y, y_pred1),
        "rmse": np.sqrt(mean_squared_error(y, y_pred1)),
        "mae": mean_absolute_error(y, y_pred1),
    }

    # Model 2: s(0, by=1) - severity(0) by slope(1)
    gam2 = LinearGAM(s(0, by=1, n_splines=args.n_splines)).fit(X_full, y)
    y_pred2 = gam2.predict(X_full)
    models["s(0, by=1)"] = {
        "y_pred": y_pred2,
        "r2": r2_score(y, y_pred2),
        "rmse": np.sqrt(mean_squared_error(y, y_pred2)),
        "mae": mean_absolute_error(y, y_pred2),
    }

    # Model 3: s(1, by=0) - slope(1) by severity(0)
    gam3 = LinearGAM(s(1, by=0, n_splines=args.n_splines)).fit(X_full, y)
    y_pred3 = gam3.predict(X_full)
    models["s(1, by=0)"] = {
        "y_pred": y_pred3,
        "r2": r2_score(y, y_pred3),
        "rmse": np.sqrt(mean_squared_error(y, y_pred3)),
        "mae": mean_absolute_error(y, y_pred3),
    }

    # Model 4: s(0, by=1) using area as var 0 - reduccion_area(0) by severity(1)
    # X: [area, severity, slope] -> s(0, by=1)
    gam4 = LinearGAM(s(0, by=1, n_splines=args.n_splines)).fit(X_area, y)
    y_pred4 = gam4.predict(X_area)
    models["s(area, by=0)"] = {
        "y_pred": y_pred4,
        "r2": r2_score(y, y_pred4),
        "rmse": np.sqrt(mean_squared_error(y, y_pred4)),
        "mae": mean_absolute_error(y, y_pred4),
    }

    # Model 5: s(0, by=2) using area as var 0 - reduccion_area(0) by slope(2)
    # X: [area, severity, slope] -> s(0, by=2)
    gam5 = LinearGAM(s(0, by=2, n_splines=args.n_splines)).fit(X_area, y)
    y_pred5 = gam5.predict(X_area)
    models["s(area, by=1)"] = {
        "y_pred": y_pred5,
        "r2": r2_score(y, y_pred5),
        "rmse": np.sqrt(mean_squared_error(y, y_pred5)),
        "mae": mean_absolute_error(y, y_pred5),
    }

    # Model 6: GAM with te(0, 1) + l(2), binomial distribution, logit link
    # X: [severity, slope, reduccion_area]
    gam6 = GAM(
        te(0, 1, n_splines=[5, 4]) + l(2), distribution="binomial", link="logit"
    ).fit(X_area, y)
    y_pred6 = gam6.predict(X_area)
    models["GAM: te(0,1) + l(2) [binomial/logit]"] = {
        "y_pred": y_pred6,
        "r2": r2_score(y, y_pred6),
        "rmse": np.sqrt(mean_squared_error(y, y_pred6)),
        "mae": mean_absolute_error(y, y_pred6),
    }

    # Model 7: LinearGAM with te(0, 1) + l(2) (same terms, no binomial/logit)
    # X: [severity, slope, reduccion_area]
    gam7 = LinearGAM(te(0, 1, n_splines=[5, 4]) + l(2)).fit(X_area, y)
    y_pred7 = gam7.predict(X_area)
    models["LinearGAM: te(0,1) + l(2)"] = {
        "y_pred": y_pred7,
        "r2": r2_score(y, y_pred7),
        "rmse": np.sqrt(mean_squared_error(y, y_pred7)),
        "mae": mean_absolute_error(y, y_pred7),
    }

    print(f"\n{'=' * 80}")
    print("GAM Model Comparison Results")
    print(f"{'=' * 80}")
    print(f"\n{'Model':<30} {'R²':>10} {'RMSE':>10} {'MAE':>10}")
    print(f"{'-' * 30} {'-' * 10} {'-' * 10} {'-' * 10}")

    best_r2 = 0
    best_model_name = None

    for name, res in models.items():
        print(f"{name:<30} {res['r2']:>10.6f} {res['rmse']:>10.6f} {res['mae']:>10.6f}")
        if res["r2"] > best_r2:
            best_r2 = res["r2"]
            best_model_name = name

    print(f"{'=' * 80}")
    print(f"\nBest model: {best_model_name} (R² = {best_r2:.6f})")

    print(f"\n{'=' * 80}")
    print("Detailed Predictions vs Actual")
    print(f"{'=' * 80}")

    header = f"{'Severity':>8} {'Slope':>8} {'Area':>10} {'Actual':>10}"
    for name in models.keys():
        header += f" {name:>15}"
    print(header)

    print(f"{'-' * 8} {'-' * 8} {'-' * 10} {'-' * 10}", end="")
    for name in models.keys():
        print(f" {'-' * 15}", end="")
    print()

    for i in range(len(df)):
        row = f"{df['severity'].iloc[i]:>8.3f} {df['slope'].iloc[i]:>8.3f} {df['reduccion_area'].iloc[i]:>10.6f} {y[i]:>10.6f}"
        for name in models.keys():
            row += f" {models[name]['y_pred'][i]:>15.6f}"
        print(row)

    fig, axes = plt.subplots(4, 2, figsize=(12, 20))
    axes = axes.flatten()

    plot_configs = [
        ("te(0, 1)", df["reduccion_area"].values, "Reduccion Area"),
        ("s(0, by=1)", df["severity"].values, "Severity"),
        ("s(1, by=0)", df["slope"].values, "Slope"),
        ("s(area, by=0)", df["reduccion_area"].values, "Reduccion Area"),
        ("s(area, by=1)", df["reduccion_area"].values, "Reduccion Area"),
        (
            "GAM: te(0,1) + l(2) [binomial/logit]",
            df["reduccion_area"].values,
            "Reduccion Area",
        ),
        ("LinearGAM: te(0,1) + l(2)", df["reduccion_area"].values, "Reduccion Area"),
    ]

    for idx, (name, x_vals, xlabel) in enumerate(plot_configs):
        ax = axes[idx]
        y_pred = models[name]["y_pred"]

        sort_idx = np.argsort(x_vals)
        x_sorted = x_vals[sort_idx]
        y_actual_sorted = y[sort_idx]
        y_pred_sorted = y_pred[sort_idx]

        ax.scatter(
            x_sorted, y_actual_sorted, color="blue", label="Actual", s=60, zorder=3
        )
        ax.plot(
            x_sorted,
            y_pred_sorted,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label="Predicted",
            zorder=2,
        )

        ax.set_xlabel(xlabel)
        ax.set_ylabel("FFR")
        ax.set_title(f"{name}\nR² = {models[name]['r2']:.4f}", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[7].axis("off")

    plt.tight_layout()
    plt.savefig(args.plot, dpi=150, bbox_inches="tight")
    print(f"\nComparison plot saved to: {args.plot}")

    if args.output:
        df_out = df.copy()
        df_out["best_model"] = best_model_name
        df_out["best_r2"] = best_r2
        for name in models.keys():
            col_name = (
                name.replace("(", "_")
                .replace(")", "")
                .replace("+", "plus")
                .replace(" ", "_")
                .replace(",", "")
            )
            df_out[f"pred_{col_name}"] = models[name]["y_pred"]
        df_out.to_csv(args.output, index=False)
        print(f"Predictions saved to: {args.output}")


if __name__ == "__main__":
    main()
