import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# SEM Model parameters (main.tex eq. 1017)
B0 = 1.0238
B_ETA = -0.2004
B_LATENT = -0.008917

# Domains defined
ETA_RANGE = (0.10, 0.85)
ALPHA_RANGE = (0.01, 0.40)
R_RANGE = (13.79, 20.19)
N_SAMPLES = 15


def generate_stratified_sampling():
    # 1. Create 15 target FFR values uniformly distributed across the range
    ffr_targets = np.linspace(0.22, 0.98, N_SAMPLES)

    final_samples = []

    for f_target in ffr_targets:
        found = False
        attempts = 0
        while not found and attempts < 2000:
            # Pick a random severity in the range
            eta = np.random.uniform(*ETA_RANGE)

            # Solve for alpha: FFR = B0 + B_ETA*eta + B_LATENT*(eta^2/alpha)
            num = B_LATENT * (eta**2)
            den = f_target - B0 - B_ETA * eta

            if abs(den) > 1e-7:
                alpha = num / den
                if ALPHA_RANGE[0] <= alpha <= ALPHA_RANGE[1]:
                    final_samples.append(
                        {
                            "severity_eta": eta,
                            "slope_alpha": alpha,
                            "expected_ffr": f_target,
                        }
                    )
                    found = True
            attempts += 1

        if not found:
            print(
                f"Warning: Could not find exact match for FFR={f_target:.2f} within [0.01, 0.4]."
            )

    df = pd.DataFrame(final_samples)

    # 2. Assign Resistance (LHS property)
    rs = np.linspace(*R_RANGE, len(df))
    np.random.shuffle(rs)
    df["resistance_R"] = rs

    return df


if __name__ == "__main__":
    np.random.seed(42)
    samples = generate_stratified_sampling()

    # Generate CSV
    output_csv = "scripts/sampling_plan.csv"
    samples.to_csv(output_csv, index=False)
    print(f"Sampling plan generated successfully in {output_csv}")

    # --- Generate Simulation Commands ---
    print("\n--- Generating Simulation Commands ---")
    bash_commands = []
    for idx, row in samples.iterrows():
        # Format strings for the name (replace . with p)
        s_val = f"{row['severity_eta']:.2f}".replace(".", "p")
        a_val = f"{row['slope_alpha']:.4f}".replace(".", "p")
        r_val = f"{row['resistance_R']:.2f}".replace(".", "p")

        run_name = f"severity{s_val}_slope{a_val}_resistance{r_val}"

        cmd = (
            f"python main.py simulate "
            f"--simulation stenosis_with_tree_2d_pressure "
            f"--solver stabilized_schur_pressure_backflow "
            f"--R_resistance {row['resistance_R']:.4f} "
            f"--T 1 --dt 0.0001 --beta_backflow 0.6 --hpc --cores 4 "
            f"--severity {row['severity_eta']:.4f} "
            f"--slope {row['slope_alpha']:.6f} "
            f"--x_position_stenosis 69 "
            f"--name {run_name}"
        )
        bash_commands.append(cmd)

    # Save to bash script
    output_bash = "scripts/run_simulations.sh"
    with open(output_bash, "w") as f:
        f.write("#!/bin/bash\n\n")
        f.write("\n".join(bash_commands))
        f.write("\n")

    print(f"Commands saved to: {output_bash}")

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    sc = ax1.scatter(
        samples["severity_eta"],
        samples["slope_alpha"],
        c=samples["expected_ffr"],
        cmap="RdYlGn",
        s=100,
        edgecolors="k",
    )
    plt.colorbar(sc, ax=ax1, label="Expected FFR")
    ax1.set_xlabel("Severity (eta)")
    ax1.set_ylabel("Slope (alpha)")
    ax1.set_title("Final Stratified Sampling Plan")
    ax1.grid(True, alpha=0.3)

    ax2.hist(
        samples["expected_ffr"],
        bins=10,
        color="lightgreen",
        edgecolor="black",
        alpha=0.7,
    )
    ax2.set_xlabel("Expected FFR")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Expected FFR Distribution (Uniform)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("scripts/sampling_verification.png")
    print("Verification plots updated in: scripts/sampling_verification.png")
