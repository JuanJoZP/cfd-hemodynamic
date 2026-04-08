import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

# Configuración estética similar a las imágenes de referencia
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "grid.linestyle": "-",
        "grid.alpha": 0.8,
        "axes.linewidth": 1.0,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.titleweight": "normal",
        "axes.labelweight": "normal",
    }
)


def main():
    parser = argparse.ArgumentParser(
        description="Generar plots de convergencia a partir de CSV."
    )
    parser.add_argument(
        "--csv_file",
        default="convergence_data.csv",
        help="Archivo CSV con los datos extraídos.",
    )
    parser.add_argument(
        "--output_dir", default="figures", help="Directorio para guardar los plots."
    )
    args = parser.parse_args()

    if not os.path.exists(args.csv_file):
        print(f"Error: No se encontró el archivo {args.csv_file}")
        return

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    df = pd.read_csv(args.csv_file)

    # Limpiar datos nulos y ordenar por DOFs
    df = df.dropna(subset=["dofs"]).sort_values(by="dofs")

    # Colores exactos de las imágenes de referencia (Matplotlib tab10)
    colors = {
        "velocity": "#1f77b4",  # Azul
        "pressure": "#ff7f0e",  # Naranja
        "drag": "#2ca02c",  # Verde
        "lift": "#d62728",  # Rojo
    }

    # Helper para plots consistentes
    def plot_convergence(x, y, ylabel, title, filename, color):
        plt.figure(figsize=(6, 4.5))
        plt.plot(
            x,
            y,
            marker="o",
            linestyle="-",
            linewidth=1.5,
            color=color,
            markersize=6,
            markerfacecolor=color,
            markeredgecolor=color,
        )

        plt.xlabel("Grados de Libertad (DOFs)")
        plt.ylabel(ylabel)
        # plt.title(title, pad=10) # Títulos eliminados a petición del usuario

        # Escala lineal como en las imágenes
        plt.xscale("linear")
        plt.yscale("linear")

        plt.grid(True)
        plt.tight_layout()

        save_path = os.path.join(args.output_dir, filename)
        plt.savefig(save_path)
        print(f"[OK] Plot guardado: {save_path}")
        plt.close()

    # 1. Norma L2 Velocidad
    if "l2_velocity" in df.columns:
        plot_convergence(
            df["dofs"],
            df["l2_velocity"],
            "Norma L2 de la velocidad",
            "Norma L2 de la velocidad vs grados de libertad",
            "norma_velocidad_dfg1.png",
            colors["velocity"],
        )

    # 2. Norma L2 Presión
    if "l2_pressure" in df.columns:
        plot_convergence(
            df["dofs"],
            df["l2_pressure"],
            "Norma L2 de la presión",
            "Norma L2 de la presión vs grados de libertad",
            "norma_presion_dfg1.png",
            colors["pressure"],
        )

    # 3. Coeficiente de Arrastre (Cd)
    if "cd" in df.columns:
        plot_convergence(
            df["dofs"],
            df["cd"],
            "Coeficiente de arrastre $C_D$",
            "Coeficiente de arrastre $C_D$ vs grados de libertad",
            "dfg1_drag_convergence.png",
            colors["drag"],
        )

    # 4. Coeficiente de Sustentación (Cl)
    if "cl" in df.columns:
        plot_convergence(
            df["dofs"],
            df["cl"],
            "Coeficiente de sustentación $C_L$",
            "Coeficiente de sustentación $C_L$ vs grados de libertad",
            "dfg1_lift_convergence.png",
            colors["lift"],
        )


if __name__ == "__main__":
    main()
