#!/usr/bin/env python3
"""Compare additive vs multiplicative FFR models combining area reduction and microvascular resistance.

Models:
  1. Additive: FFR = β₀ + β₁*ΔA_rel + β₂*severity + β₃*resistance
  2. Multiplicative: FFR = (β₀ + β₁*ΔA_rel + β₂*severity) * (γ₀ + γ₁*resistance)

Both models are FITTED from data (not using fixed coefficients).

Usage:
    python scripts/ffr_combine_models.py --input ffr_dataset.csv
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_latent_variable(severity, slope):
    """Compute relative area reduction = ΔA_abs / A_total_canal.
    A_total_canal = 138 * (1.57 + 1.2) = 382.26
    ΔA_abs = 2 * severity^2 * 1.4896^2 / slope
    """
    return (2.0 * severity**2 * 2.218804) / (slope * 382.26)


def multiplicative_model(params, delta_A, severity, resistance, ffr_actual):
    """Multiplicative model: FFR = FFR_area * FFR_micro.
    FFR_area = β₀ + β₁*ΔA_rel + β₂*severity
    FFR_micro = γ₀ + γ₁*resistance
    """
    beta0, beta1, beta2, gamma0, gamma1 = params
    ffr_area = beta0 + beta1 * delta_A + beta2 * severity
    ffr_micro = gamma0 + gamma1 * resistance
    ffr_pred = ffr_area * ffr_micro
    return np.sum((ffr_pred - ffr_actual) ** 2)


def main():
    parser = argparse.ArgumentParser(
        description="Compare additive vs multiplicative FFR models"
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
        default="ffr_combine_models_plot.png",
        help="Output plot filename (default: ffr_combine_models_plot.png)",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        sys.exit(1)

    df = pd.read_csv(input_path)

    required_cols = ["severity", "slope", "resistance", "ffr"]
    if not all(col in df.columns for col in required_cols):
        print(f"Error: CSV must contain {required_cols} columns")
        sys.exit(1)

    severity = df["severity"].values
    slope = df["slope"].values
    resistance = df["resistance"].values
    ffr_actual = df["ffr"].values
    delta_A = compute_latent_variable(severity, slope)

    df["delta_A"] = delta_A

    results = {}

    # Model 1: Area reduction only (baseline)
    X_area = np.column_stack([delta_A, severity])
    model_area = LinearRegression()
    model_area.fit(X_area, ffr_actual)
    ffr_pred_area = model_area.predict(X_area)
    results["Area Reduction Only"] = {
        "y_pred": ffr_pred_area,
        "r2": r2_score(ffr_actual, ffr_pred_area),
        "rmse": np.sqrt(mean_squared_error(ffr_actual, ffr_pred_area)),
        "mae": mean_absolute_error(ffr_actual, ffr_pred_area),
        "mean_error": np.mean(ffr_pred_area - ffr_actual),
        "model": model_area,
    }

    # Model 2: Additive: FFR = β₀ + β₁*ΔA_rel + β₂*severity + β₃*resistance
    X_additive = np.column_stack([delta_A, severity, resistance])
    model_additive = LinearRegression()
    model_additive.fit(X_additive, ffr_actual)
    ffr_pred_additive = model_additive.predict(X_additive)
    results["Additive"] = {
        "y_pred": ffr_pred_additive,
        "r2": r2_score(ffr_actual, ffr_pred_additive),
        "rmse": np.sqrt(mean_squared_error(ffr_actual, ffr_pred_additive)),
        "mae": mean_absolute_error(ffr_actual, ffr_pred_additive),
        "mean_error": np.mean(ffr_pred_additive - ffr_actual),
        "model": model_additive,
    }

    # Model 3: Multiplicative: FFR = FFR_area * FFR_micro
    # FFR_area = β₀ + β₁*ΔA_rel + β₂*severity
    # FFR_micro = γ₀ + γ₁*resistance
    initial_params = [1.0, -0.5, -0.2, 1.0, 0.0]
    result_opt = minimize(
        multiplicative_model,
        initial_params,
        args=(delta_A, severity, resistance, ffr_actual),
        method="L-BFGS-B",
    )
    beta0, beta1, beta2, gamma0, gamma1 = result_opt.x
    ffr_area_mult = beta0 + beta1 * delta_A + beta2 * severity
    ffr_micro = gamma0 + gamma1 * resistance
    ffr_pred_mult = ffr_area_mult * ffr_micro
    results["Multiplicative"] = {
        "y_pred": ffr_pred_mult,
        "r2": r2_score(ffr_actual, ffr_pred_mult),
        "rmse": np.sqrt(mean_squared_error(ffr_actual, ffr_pred_mult)),
        "mae": mean_absolute_error(ffr_actual, ffr_pred_mult),
        "mean_error": np.mean(ffr_pred_mult - ffr_actual),
        "params": (beta0, beta1, beta2, gamma0, gamma1),
    }

    # Model 4: Resistance only (baseline)
    X_resistance = resistance.reshape(-1, 1)
    model_resistance = LinearRegression()
    model_resistance.fit(X_resistance, ffr_actual)
    ffr_pred_resistance = model_resistance.predict(X_resistance)
    results["Resistance Only"] = {
        "y_pred": ffr_pred_resistance,
        "r2": r2_score(ffr_actual, ffr_pred_resistance),
        "rmse": np.sqrt(mean_squared_error(ffr_actual, ffr_pred_resistance)),
        "mae": mean_absolute_error(ffr_actual, ffr_pred_resistance),
        "mean_error": np.mean(ffr_pred_resistance - ffr_actual),
        "model": model_resistance,
    }

    print(f"{'=' * 100}")
    print("FFR Model Comparison: Area Reduction + Microvascular Resistance")
    print(f"{'=' * 100}")
    print(f"\n{'Model':<30} {'R²':>10} {'RMSE':>10} {'MAE':>10} {'Mean Err':>10}")
    print(f"{'-' * 30} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}")

    best_r2 = 0
    best_model = None
    for name, res in results.items():
        print(
            f"{name:<30} {res['r2']:>10.6f} {res['rmse']:>10.6f} {res['mae']:>10.6f} {res['mean_error']:>10.6f}"
        )
        if res["r2"] > best_r2:
            best_r2 = res["r2"]
            best_model = name

    print(f"\n{'=' * 100}")
    print(f"Best model: {best_model} (R² = {best_r2:.6f})")

    print(f"\n{'=' * 100}")
    print("Model 1: Area Reduction Only")
    print(f"{'=' * 100}")
    print(
        f"  FFR = {model_area.intercept_:.6f} + {model_area.coef_[0]:.6f} * ΔA_rel + {model_area.coef_[1]:.6f} * severity"
    )

    print(f"\n{'=' * 100}")
    print("Model 2: Additive (ΔA_rel + severity + resistance)")
    print(f"{'=' * 100}")
    print(f"  FFR = {model_additive.intercept_:.6f}")
    print(f"       + {model_additive.coef_[0]:.6f} * ΔA_rel")
    print(f"       + {model_additive.coef_[1]:.6f} * severity")
    print(f"       + {model_additive.coef_[2]:.6f} * resistance")

    print(f"\n{'=' * 100}")
    print("Model 3: Multiplicative (FFR_area * FFR_micro)")
    print(f"{'=' * 100}")
    print(f"  FFR_area = {beta0:.6f} + {beta1:.6f} * ΔA_rel + {beta2:.6f} * severity")
    print(f"  FFR_micro = {gamma0:.6f} + {gamma1:.6f} * resistance")
    print(f"  FFR = FFR_area * FFR_micro")

    print(f"\n{'=' * 100}")
    print("Model 4: Resistance Only")
    print(f"{'=' * 100}")
    print(
        f"  FFR = {model_resistance.intercept_:.6f} + {model_resistance.coef_[0]:.6f} * resistance"
    )

    print(f"\n{'=' * 100}")
    print("Predictions Comparison")
    print(f"{'=' * 100}")
    header = f"{'Sev':>6} {'Slope':>8} {'Res':>8} {'Actual':>10}"
    for name in [
        "Area Reduction Only",
        "Additive",
        "Multiplicative",
        "Resistance Only",
    ]:
        short_name = name.split()[0]
        header += f" {short_name:>12}"
    print(header)
    print(f"{'-' * 6} {'-' * 8} {'-' * 8} {'-' * 10}", end="")
    for _ in range(4):
        print(f" {'-' * 12}", end="")
    print()

    for i in range(len(df)):
        row = f"{severity[i]:>6.3f} {slope[i]:>8.4f} {resistance[i]:>8.2f} {ffr_actual[i]:>10.6f}"
        for name in [
            "Area Reduction Only",
            "Additive",
            "Multiplicative",
            "Resistance Only",
        ]:
            row += f" {results[name]['y_pred'][i]:>12.6f}"
        print(row)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    sort_idx = np.argsort(ffr_actual)
    axes[0, 0].scatter(
        ffr_actual, ffr_pred_area, color="blue", label="Area Reduction", s=60, alpha=0.7
    )
    axes[0, 0].scatter(
        ffr_actual, ffr_pred_additive, color="green", label="Additive", s=60, alpha=0.7
    )
    axes[0, 0].scatter(
        ffr_actual, ffr_pred_mult, color="red", label="Multiplicative", s=60, alpha=0.7
    )
    min_val = min(
        ffr_actual.min(),
        ffr_pred_area.min(),
        ffr_pred_additive.min(),
        ffr_pred_mult.min(),
    )
    max_val = max(
        ffr_actual.max(),
        ffr_pred_area.max(),
        ffr_pred_additive.max(),
        ffr_pred_mult.max(),
    )
    axes[0, 0].plot(
        [min_val, max_val],
        [min_val, max_val],
        "k--",
        alpha=0.5,
        label="Perfect prediction",
    )
    axes[0, 0].set_xlabel("Actual FFR")
    axes[0, 0].set_ylabel("Predicted FFR")
    axes[0, 0].set_title("Predicted vs Actual")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    res_idx = np.argsort(resistance)
    axes[0, 1].scatter(
        resistance[res_idx],
        ffr_actual[res_idx],
        color="black",
        label="Actual",
        s=60,
        alpha=0.7,
    )
    axes[0, 1].plot(
        resistance[res_idx],
        ffr_pred_additive[res_idx],
        "g-",
        linewidth=2,
        label="Additive",
    )
    axes[0, 1].plot(
        resistance[res_idx],
        ffr_pred_mult[res_idx],
        "r--",
        linewidth=2,
        label="Multiplicative",
    )
    axes[0, 1].set_xlabel("Resistance")
    axes[0, 1].set_ylabel("FFR")
    axes[0, 1].set_title("FFR vs Resistance")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    errors_area = ffr_pred_area - ffr_actual
    errors_add = ffr_pred_additive - ffr_actual
    errors_mult = ffr_pred_mult - ffr_actual
    axes[1, 0].hist(
        errors_area,
        bins=15,
        alpha=0.5,
        label=f"Area Only (MAE={results['Area Reduction Only']['mae']:.4f})",
    )
    axes[1, 0].hist(
        errors_add,
        bins=15,
        alpha=0.5,
        label=f"Additive (MAE={results['Additive']['mae']:.4f})",
    )
    axes[1, 0].hist(
        errors_mult,
        bins=15,
        alpha=0.5,
        label=f"Multiplicative (MAE={results['Multiplicative']['mae']:.4f})",
    )
    axes[1, 0].axvline(x=0, color="k", linestyle="--", alpha=0.5)
    axes[1, 0].set_xlabel("Prediction Error")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].set_title("Error Distribution")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    models_comparison = [
        "Area Reduction Only",
        "Additive",
        "Multiplicative",
        "Resistance Only",
    ]
    r2_values = [results[m]["r2"] for m in models_comparison]
    mae_values = [results[m]["mae"] for m in models_comparison]
    x_pos = np.arange(len(models_comparison))
    width = 0.35
    axes[1, 1].bar(x_pos - width / 2, r2_values, width, label="R²", color="steelblue")
    axes[1, 1].bar(x_pos + width / 2, mae_values, width, label="MAE", color="coral")
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(["Area", "Additive", "Mult", "Resist"], fontsize=9)
    axes[1, 1].set_ylabel("Metric Value")
    axes[1, 1].set_title("Model Comparison")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(args.plot, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {args.plot}")

    if args.output:
        df_out = df.copy()
        df_out["ffr_pred_area"] = ffr_pred_area
        df_out["ffr_pred_additive"] = ffr_pred_additive
        df_out["ffr_pred_multiplicative"] = ffr_pred_mult
        df_out["ffr_pred_resistance"] = ffr_pred_resistance
        df_out.to_csv(args.output, index=False)
        print(f"Predictions saved to: {args.output}")


if __name__ == "__main__":
    main()
