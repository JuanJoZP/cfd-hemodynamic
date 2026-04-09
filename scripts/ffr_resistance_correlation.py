#!/usr/bin/env python3
"""Test correlation between FFR and resistance.

Performs multiple statistical tests:
  - Pearson correlation
  - Spearman correlation
  - Partial correlation controlling for severity and slope
  - Visualization

Usage:
    python scripts/ffr_resistance_correlation.py --input ffr_dataset.csv
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression


def compute_latent_variable(severity, slope):
    """Compute relative area reduction = ΔA_abs / A_total_canal."""
    return (2.0 * severity**2 * 2.218804) / (slope * 382.26)


def partial_correlation(x, y, covariates):
    """Compute partial correlation between x and y controlling for covariates.

    Uses the regression residual method.
    """
    # Regress x on covariates
    model_x = LinearRegression()
    model_x.fit(covariates, x)
    residual_x = x - model_x.predict(covariates)

    # Regress y on covariates
    model_y = LinearRegression()
    model_y.fit(covariates, y)
    residual_y = y - model_y.predict(covariates)

    # Correlation of residuals
    corr, p_value = stats.pearsonr(residual_x, residual_y)
    return corr, p_value


def main():
    parser = argparse.ArgumentParser(
        description="Test correlation between FFR and resistance"
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
        help="Output CSV for analysis (optional)",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default="ffr_resistance_correlation.png",
        help="Output plot filename",
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
    ffr = df["ffr"].values
    delta_A = compute_latent_variable(severity, slope)

    print(f"{'=' * 80}")
    print("Correlation Analysis: FFR vs Resistance")
    print(f"{'=' * 80}")
    print(f"\nDataset: {len(df)} observations")
    print(f"\nDescriptive Statistics:")
    print(f"{'Variable':<15} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
    print(f"{'-' * 15} {'-' * 12} {'-' * 12} {'-' * 12} {'-' * 12}")
    print(
        f"{'FFR':<15} {np.mean(ffr):>12.6f} {np.std(ffr):>12.6f} {np.min(ffr):>12.6f} {np.max(ffr):>12.6f}"
    )
    print(
        f"{'Resistance':<15} {np.mean(resistance):>12.6f} {np.std(resistance):>12.6f} {np.min(resistance):>12.6f} {np.max(resistance):>12.6f}"
    )
    print(
        f"{'Severity':<15} {np.mean(severity):>12.6f} {np.std(severity):>12.6f} {np.min(severity):>12.6f} {np.max(severity):>12.6f}"
    )
    print(
        f"{'Slope':<15} {np.mean(slope):>12.6f} {np.std(slope):>12.6f} {np.min(slope):>12.6f} {np.max(slope):>12.6f}"
    )

    # Pearson correlation
    pearson_r, pearson_p = stats.pearsonr(resistance, ffr)
    print(f"\n{'=' * 80}")
    print("1. Pearson Correlation (linear relationship)")
    print(f"{'=' * 80}")
    print(f"   r = {pearson_r:.6f}")
    print(f"   p-value = {pearson_p:.6f}")
    if pearson_p < 0.05:
        print(f"   Conclusion: SIGNIFICANT (p < 0.05)")
    else:
        print(f"   Conclusion: NOT SIGNIFICANT (p >= 0.05)")

    # Spearman correlation
    spearman_r, spearman_p = stats.spearmanr(resistance, ffr)
    print(f"\n{'=' * 80}")
    print("2. Spearman Correlation (monotonic relationship)")
    print(f"{'=' * 80}")
    print(f"   ρ = {spearman_r:.6f}")
    print(f"   p-value = {spearman_p:.6f}")
    if spearman_p < 0.05:
        print(f"   Conclusion: SIGNIFICANT (p < 0.05)")
    else:
        print(f"   Conclusion: NOT SIGNIFICANT (p >= 0.05)")

    # Partial correlation controlling for severity and slope
    covariates = np.column_stack([severity, slope])
    partial_r, partial_p = partial_correlation(resistance, ffr, covariates)
    print(f"\n{'=' * 80}")
    print("3. Partial Correlation (controlling for severity and slope)")
    print(f"{'=' * 80}")
    print(f"   r_partial = {partial_r:.6f}")
    print(f"   p-value = {partial_p:.6f}")
    if partial_p < 0.05:
        print(f"   Conclusion: SIGNIFICANT (p < 0.05)")
    else:
        print(f"   Conclusion: NOT SIGNIFICANT (p >= 0.05)")

    # Partial correlation controlling for delta_A
    covariates_delta = delta_A.reshape(-1, 1)
    partial_r_delta, partial_p_delta = partial_correlation(
        resistance, ffr, covariates_delta
    )
    print(f"\n{'=' * 80}")
    print("4. Partial Correlation (controlling for ΔA_rel)")
    print(f"{'=' * 80}")
    print(f"   r_partial = {partial_r_delta:.6f}")
    print(f"   p-value = {partial_p_delta:.6f}")
    if partial_p_delta < 0.05:
        print(f"   Conclusion: SIGNIFICANT (p < 0.05)")
    else:
        print(f"   Conclusion: NOT SIGNIFICANT (p >= 0.05)")

    # Regression: FFR ~ resistance alone
    model_res = LinearRegression()
    model_res.fit(resistance.reshape(-1, 1), ffr)
    ffr_pred_res = model_res.predict(resistance.reshape(-1, 1))
    r2_resistance_only = 1 - np.sum((ffr - ffr_pred_res) ** 2) / np.sum(
        (ffr - np.mean(ffr)) ** 2
    )

    print(f"\n{'=' * 80}")
    print("5. Regression: FFR ~ Resistance")
    print(f"{'=' * 80}")
    print(
        f"   FFR = {model_res.intercept_:.6f} + {model_res.coef_[0]:.6f} * resistance"
    )
    print(f"   R² = {r2_resistance_only:.6f}")
    print(
        f"   Interpretation: Resistance alone explains {r2_resistance_only * 100:.2f}% of FFR variance"
    )

    # Regression: FFR ~ resistance + severity + slope (full model)
    X_full = np.column_stack([resistance, severity, slope])
    model_full = LinearRegression()
    model_full.fit(X_full, ffr)
    ffr_pred_full = model_full.predict(X_full)
    r2_full = 1 - np.sum((ffr - ffr_pred_full) ** 2) / np.sum((ffr - np.mean(ffr)) ** 2)

    # Regression: FFR ~ severity + slope (without resistance)
    X_nores = np.column_stack([severity, slope])
    model_nores = LinearRegression()
    model_nores.fit(X_nores, ffr)
    ffr_pred_nores = model_nores.predict(X_nores)
    r2_nores = 1 - np.sum((ffr - ffr_pred_nores) ** 2) / np.sum(
        (ffr - np.mean(ffr)) ** 2
    )

    # Incremental R² from adding resistance
    delta_r2 = r2_full - r2_nores

    # F-test for incremental R²
    n = len(ffr)
    k_nores = 2
    k_full = 3
    f_stat = (delta_r2 / (k_full - k_nores)) / ((1 - r2_full) / (n - k_full - 1))
    f_pvalue = 1 - stats.f.cdf(f_stat, k_full - k_nores, n - k_full - 1)

    print(f"\n{'=' * 80}")
    print("6. Incremental R² Analysis")
    print(f"{'=' * 80}")
    print(f"   R² (severity + slope)              = {r2_nores:.6f}")
    print(f"   R² (severity + slope + resistance) = {r2_full:.6f}")
    print(f"   ΔR² from adding resistance         = {delta_r2:.6f}")
    print(f"   F-statistic = {f_stat:.6f}")
    print(f"   p-value = {f_pvalue:.6f}")
    if f_pvalue < 0.05:
        print(
            f"   Conclusion: Adding resistance SIGNIFICANTLY improves model (p < 0.05)"
        )
    else:
        print(
            f"   Conclusion: Adding resistance does NOT significantly improve model (p >= 0.05)"
        )

    # Correlation matrix
    print(f"\n{'=' * 80}")
    print("7. Full Correlation Matrix")
    print(f"{'=' * 80}")
    variables = {
        "FFR": ffr,
        "Resistance": resistance,
        "Severity": severity,
        "Slope": slope,
        "ΔA_rel": delta_A,
    }
    names = list(variables.keys())
    print(f"\n{'':>12}", end="")
    for name in names:
        print(f"{name:>12}", end="")
    print()
    for i, name1 in enumerate(names):
        print(f"{name1:>12}", end="")
        for j, name2 in enumerate(names):
            r, _ = stats.pearsonr(variables[name1], variables[name2])
            print(f"{r:>12.4f}", end="")
        print()

    # Data table
    print(f"\n{'=' * 80}")
    print("8. Data Summary")
    print(f"{'=' * 80}")
    print(f"{'Severity':>8} {'Slope':>8} {'Resistance':>12} {'FFR':>10}")
    print(f"{'-' * 8} {'-' * 8} {'-' * 12} {'-' * 10}")
    for i in range(len(df)):
        print(
            f"{severity[i]:>8.3f} {slope[i]:>8.4f} {resistance[i]:>12.4f} {ffr[i]:>10.6f}"
        )

    # Plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Scatter: FFR vs Resistance
    axes[0, 0].scatter(resistance, ffr, color="steelblue", s=60, alpha=0.7)
    z = np.polyfit(resistance, ffr, 1)
    p = np.poly1d(z)
    x_line = np.linspace(resistance.min(), resistance.max(), 100)
    axes[0, 0].plot(x_line, p(x_line), "r--", linewidth=2, label=f"r={pearson_r:.4f}")
    axes[0, 0].set_xlabel("Resistance")
    axes[0, 0].set_ylabel("FFR")
    axes[0, 0].set_title(
        f"FFR vs Resistance\nPearson r={pearson_r:.4f}, p={pearson_p:.4f}"
    )
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Scatter: FFR vs Resistance colored by severity
    scatter = axes[0, 1].scatter(
        resistance, ffr, c=severity, cmap="viridis", s=60, alpha=0.7
    )
    plt.colorbar(scatter, ax=axes[0, 1], label="Severity")
    axes[0, 1].set_xlabel("Resistance")
    axes[0, 1].set_ylabel("FFR")
    axes[0, 1].set_title("FFR vs Resistance (colored by Severity)")
    axes[0, 1].grid(True, alpha=0.3)

    # Residual plot: FFR residuals after removing severity+slope vs resistance
    axes[1, 0].scatter(resistance, ffr - ffr_pred_nores, color="coral", s=60, alpha=0.7)
    axes[1, 0].axhline(y=0, color="k", linestyle="--", alpha=0.5)
    z_res = np.polyfit(resistance, ffr - ffr_pred_nores, 1)
    p_res = np.poly1d(z_res)
    axes[1, 0].plot(x_line, p_res(x_line), "b--", linewidth=2)
    axes[1, 0].set_xlabel("Resistance")
    axes[1, 0].set_ylabel("FFR Residual (after severity+slope)")
    axes[1, 0].set_title(
        f"Residuals vs Resistance\nPartial r={partial_r:.4f}, p={partial_p:.4f}"
    )
    axes[1, 0].grid(True, alpha=0.3)

    # Correlation heatmap
    corr_matrix = np.zeros((len(names), len(names)))
    for i, name1 in enumerate(names):
        for j, name2 in enumerate(names):
            corr_matrix[i, j], _ = stats.pearsonr(variables[name1], variables[name2])

    im = axes[1, 1].imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[1, 1].set_xticks(range(len(names)))
    axes[1, 1].set_yticks(range(len(names)))
    axes[1, 1].set_xticklabels(names, rotation=45, ha="right")
    axes[1, 1].set_yticklabels(names)
    for i in range(len(names)):
        for j in range(len(names)):
            axes[1, 1].text(
                j, i, f"{corr_matrix[i, j]:.2f}", ha="center", va="center", fontsize=9
            )
    axes[1, 1].set_title("Correlation Matrix")
    plt.colorbar(im, ax=axes[1, 1], label="Correlation")

    plt.tight_layout()
    plt.savefig(args.plot, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {args.plot}")

    if args.output:
        df_out = df.copy()
        df_out["delta_A"] = delta_A
        df_out["ffr_pred_resistance"] = ffr_pred_res
        df_out["ffr_pred_noresistance"] = ffr_pred_nores
        df_out["ffr_pred_full"] = ffr_pred_full
        df_out["residual_noresistance"] = ffr - ffr_pred_nores
        df_out.to_csv(args.output, index=False)
        print(f"Analysis saved to: {args.output}")

    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(
        f"Pearson correlation (FFR, Resistance):   r = {pearson_r:.4f}, p = {pearson_p:.4f}"
    )
    print(
        f"Spearman correlation (FFR, Resistance): ρ = {spearman_r:.4f}, p = {spearman_p:.4f}"
    )
    print(
        f"Partial correlation (controlling for severity+slope): r = {partial_r:.4f}, p = {partial_p:.4f}"
    )
    print(f"R² from resistance alone: {r2_resistance_only:.4f}")
    print(f"ΔR² from adding resistance: {delta_r2:.4f} (p = {f_pvalue:.4f})")
    if f_pvalue >= 0.05 and pearson_p >= 0.05:
        print(
            "\n>>> CONCLUSION: NO SIGNIFICANT CORRELATION between FFR and Resistance <<<"
        )


if __name__ == "__main__":
    main()
