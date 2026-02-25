#!/usr/bin/env python3
"""
Converts a gmsh .msh file to .vtu (VTK XML) readable by ParaView.

Usage:
    python msh_to_paraview.py path/to/mesh.msh
    python msh_to_paraview.py path/to/mesh.msh --out path/to/output.vtu
"""
import argparse
import sys
from pathlib import Path


def convert(msh_path: Path, out_path: Path):
    try:
        import meshio
    except ImportError:
        sys.exit("[ERROR] meshio not found. Install it with: pip install meshio")

    mesh = meshio.read(str(msh_path))

    # meshio has a bug converting mixed cell_sets to vtu when indices are out of bound.
    # Move physical group tags to cell_data manually (one int per cell per block),
    # then clear cell_sets to avoid the crash.
    if mesh.cell_sets:
        tag_data = []
        for block in mesh.cells:
            n = len(block.data)
            tag_data.append([-1] * n)

        for set_name, set_blocks in mesh.cell_sets.items():
            for block_idx, cell_indices in enumerate(set_blocks):
                if cell_indices is None or block_idx >= len(tag_data):
                    continue
                for ci in cell_indices:
                    if 0 <= ci < len(tag_data[block_idx]):
                        tag_data[block_idx][ci] = hash(set_name) % 1000

        import numpy as np

        mesh.cell_data["gmsh:physical"] = [
            np.array(d, dtype=np.int32) for d in tag_data
        ]
        mesh.cell_sets = {}

    meshio.write(str(out_path), mesh)
    print(f"[OK] Written: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert gmsh .msh → ParaView-compatible .vtu"
    )
    parser.add_argument("msh", help="Input .msh file")
    parser.add_argument(
        "--out", help="Output file (default: same path, .vtu extension)"
    )
    args = parser.parse_args()

    msh_path = Path(args.msh)
    if not msh_path.exists():
        sys.exit(f"[ERROR] File not found: {msh_path}")

    out_path = Path(args.out) if args.out else msh_path.with_suffix(".vtu")
    convert(msh_path, out_path)


if __name__ == "__main__":
    main()
