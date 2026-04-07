#!/usr/bin/env python3
"""Compare multiple SEM models for FFR prediction.

Models:
  1. reduccion_area + severity
  2. reduccion_area
  3. reduccion_area + slope
  4. severity * slope
  5. severity / slope
  6. severity + slope

Usage:
    python scripts/ffr_sem.py --input ffr_dataset.csv
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_latent_variable(severity, slope):
    """Compute relative area reduction = ΔA_abs / A_total_canal.
    A_total_canal = 138 * (1.57 + 1.2) = 382.26
    ΔA_abs = 2 * severity^2 * 1.4896^2 / slope
    """
    return (2.0 * severity**2 * 2.218804) / (slope * 382.26)


def fit_model_and_compute_stats(X, y):
    """Fit linear regression and compute t-statistics for coefficients."""
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    n = len(y)
    p = X.shape[1]
    mse = mean_squared_error(y, y_pred)
    X_with_intercept = np.column_stack([np.ones(n), X])
    cov_matrix = np.linalg.inv(X_with_intercept.T @ X_with_intercept) * mse
    std_errors = np.sqrt(np.diag(cov_matrix))
    intercept_se = std_errors[0]
    coef_se = std_errors[1:]

    t_values = np.concatenate(
        [[model.intercept_ / intercept_se], model.coef_ / coef_se]
    )
    dof = n - p - 1
    p_values = np.array([2 * (1 - stats.t.cdf(abs(t), dof)) for t in t_values])

    return {
        "model": model,
        "y_pred": y_pred,
        "r2": r2_score(y, y_pred),
        "rmse": np.sqrt(mse),
        "mae": mean_absolute_error(y, y_pred),
        "mean_error": np.mean(y_pred - y),
        "coef_se": coef_se,
        "intercept_se": intercept_se,
        "t_values": t_values,
        "p_values": p_values,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare multiple SEM models for FFR prediction"
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
        default="ffr_sem_compare_plot.png",
        help="Output plot filename (default: ffr_sem_compare_plot.png)",
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
    df["severity_slope"] = df["severity"].values * df["slope"].values
    df["severity_div_slope"] = df["severity"].values / df["slope"].values

    y = df["ffr"].values

    models = {}

    # Model 1: reduccion_area + severity
    X1 = df[["reduccion_area", "severity"]].values
    models["reduccion_area + severity"] = fit_model_and_compute_stats(X1, y)

    # Model 2: reduccion_area only
    X2 = df[["reduccion_area"]].values
    models["reduccion_area"] = fit_model_and_compute_stats(X2, y)

    # Model 3: reduccion_area + slope
    X3 = df[["reduccion_area", "slope"]].values
    models["reduccion_area + slope"] = fit_model_and_compute_stats(X3, y)

    # Model 4: severity * slope
    X4 = df[["severity_slope"]].values
    models["severity * slope"] = fit_model_and_compute_stats(X4, y)

    # Model 5: severity / slope
    X5 = df[["severity_div_slope"]].values
    models["severity / slope"] = fit_model_and_compute_stats(X5, y)

    # Model 6: severity + slope
    X6 = df[["severity", "slope"]].values
    models["severity + slope"] = fit_model_and_compute_stats(X6, y)

    print(f"{'=' * 100}")
    print("SEM Model Comparison Results")
    print(f"{'=' * 100}")
    print(f"\n{'Model':<35} {'R²':>10} {'RMSE':>10} {'MAE':>10} {'Mean Err':>10}")
    print(f"{'-' * 35} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}")

    best_r2 = 0
    best_model_name = None

    for name, res in models.items():
        print(
            f"{name:<35} {res['r2']:>10.6f} {res['rmse']:>10.6f} {res['mae']:>10.6f} {res['mean_error']:>10.6f}"
        )
        if res["r2"] > best_r2:
            best_r2 = res["r2"]
            best_model_name = name

    print(f"{'=' * 100}")
    print(f"\nBest model: {best_model_name} (R² = {best_r2:.6f})")

    # Detailed coefficients and t-tests for each model
    for name, res in models.items():
        print(f"\n{'=' * 100}")
        print(f"Model: {name}")
        print(f"{'=' * 100}")

        model = res["model"]
        y_pred = res["y_pred"]

        print(f"\nRegression model:")
        print(f"  FFR = {model.intercept_:.6f}", end="")
        for i, coef in enumerate(model.coef_):
            if coef >= 0:
                print(
                    f"\n       + {coef:.6f} * {name.split(' + ')[i] if ' + ' in name else name}",
                    end="",
                )
            else:
                print(
                    f"\n       - {abs(coef):.6f} * {name.split(' + ')[i] if ' + ' in name else name}",
                    end="",
                )
        print()

        print(f"\nCoefficients with t-tests:")
        print(
            f"  Intercept: {model.intercept_:.6f} (SE={res['intercept_se']:.6f}, t={res['t_values'][0]:.3f}, p={res['p_values'][0]:.4f})"
        )

        predictors = name.split(" + ") if " + " in name else [name]
        for i, coef in enumerate(model.coef_):
            pred_name = predictors[i] if i < len(predictors) else f"X{i}"
            print(
                f"  {pred_name:<20}: {coef:.6f} (SE={res['coef_se'][i]:.6f}, t={res['t_values'][i + 1]:.3f}, p={res['p_values'][i + 1]:.4f})"
            )

        print(f"\nSignificance (p < 0.05):")
        all_names = ["Intercept"] + (predictors if " + " in name else [name])
        for i, (n, p_val) in enumerate(zip(all_names, res["p_values"])):
            sig = "Significant" if p_val < 0.05 else "Not significant"
            print(f"  {n:<20}: {sig} (p={p_val:.4f})")

    # Predictions table
    print(f"\n{'=' * 100}")
    print("Predictions Comparison")
    print(f"{'=' * 100}")

    header = f"{'Severity':>8} {'Slope':>8} {'Actual':>10}"
    for name in models.keys():
        short_name = name[:20]
        header += f" {short_name:>22}"
    print(header)

    print(f"{'-' * 8} {'-' * 8} {'-' * 10}", end="")
    for name in models.keys():
        print(f" {'-' * 22}", end="")
    print()

    for i in range(len(df)):
        row = (
            f"{df['severity'].iloc[i]:>8.3f} {df['slope'].iloc[i]:>8.3f} {y[i]:>10.6f}"
        )
        for name in models.keys():
            short_name = name[:20]
            row += f" {models[name]['y_pred'][i]:>22.6f}"
        print(row)

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    plot_vars = [
        ("reduccion_area + severity", df["reduccion_area"].values, "Reduccion Area"),
        ("reduccion_area", df["reduccion_area"].values, "Reduccion Area"),
        ("reduccion_area + slope", df["reduccion_area"].values, "Reduccion Area"),
        ("severity * slope", df["severity_slope"].values, "Severity * Slope"),
        ("severity / slope", df["severity_div_slope"].values, "Severity / Slope"),
        ("severity + slope", df["severity"].values, "Severity"),
    ]

    for idx, (name, x_vals, xlabel) in enumerate(plot_vars):
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

    plt.tight_layout()
    plt.savefig(args.plot, dpi=150, bbox_inches="tight")
    print(f"\nComparison plot saved to: {args.plot}")

    if args.output:
        df_out = df.copy()
        df_out["best_model"] = best_model_name
        df_out["best_r2"] = best_r2
        for name, res in models.items():
            col_name = (
                name.replace(" ", "_")
                .replace("+", "plus")
                .replace("*", "times")
                .replace("/", "div")
            )
            df_out[f"pred_{col_name}"] = res["y_pred"]
        df_out.to_csv(args.output, index=False)
        print(f"Predictions saved to: {args.output}")


if __name__ == "__main__":
    main()
