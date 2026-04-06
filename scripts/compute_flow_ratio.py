#!/usr/bin/env python3
"""Compute inlet/outlet flow ratio from saved simulation results.

Uses ParaView Python API to read .bp files and compute flow integrals.
Inlet/outlet identification is done geometrically since boundary tags
are not preserved in the output files.

Usage:
    pvpython scripts/compute_flow_ratio.py --result-dir <path>
"""

import argparse
import json
import os
import sys
from pathlib import Path

try:
    from paraview import servermanager
    from paraview.simple import (
        Delete,
        GetActiveSource,
        IntegrateVariables,
        OpenDataFile,
        PointDatatoCellData,
        Render,
        RenderView,
        SaveScreenshot,
        Threshold,
        UpdatePipeline,
        cellDatatoPointData,
    )
    from paraview.vtk.numpy_interface import dataset_adapter as dsa
    from paraview.vtk.numpy_interface.dataset_adapter import numpy_support
except ImportError:
    print("Error: This script must be run with pvpython")
    print("Usage: pvpython scripts/compute_flow_ratio.py --result-dir <path>")
    sys.exit(1)

import numpy as np


def load_simulation_params(result_dir):
    """Load simulation parameters from the result directory."""
    params_file = Path(result_dir) / "simulation_params.txt"
    params = {}

    if params_file.exists():
        with open(params_file, "r") as f:
            for line in f:
                line = line.strip()
                if "=" in line:
                    key, value = line.split("=", 1)
                    try:
                        params[key.strip()] = float(value.strip())
                    except ValueError:
                        params[key.strip()] = value.strip()

    return params


def get_point_data_arrays(reader):
    """Get available point data arrays."""
    reader.UpdatePipeline()
    data = servermanager.Fetch(reader)
    point_data = data.GetPointData()
    arrays = []
    for i in range(point_data.GetNumberOfArrays()):
        arrays.append(point_data.GetArrayName(i))
    return arrays


def compute_boundary_flow(reader, target_x, tolerance, use_negative_normal=False):
    """Compute flow through boundary at given x coordinate.

    Uses threshold to extract cells near x=target_x, then integrates
    velocity over those cells to get flow rate.

    For 2D: Q = ∫ u_x * dy = average(u_x) * (ymax - ymin)

    Returns: (flow_rate, boundary_length, avg_u_x, avg_u_y, n_points)
    """
    threshold = Threshold(Input=reader)
    threshold.Scalars = ["POINTS", "x"]
    threshold.LowerThreshold = target_x - tolerance
    threshold.UpperThreshold = target_x + tolerance
    threshold.UpdatePipeline()

    data = servermanager.Fetch(threshold)

    bounds = data.GetBounds()
    ymin, ymax = bounds[2], bounds[3]
    boundary_length = ymax - ymin

    point_data = data.GetPointData()

    vel_array = None
    for name in ["v", "velocity", "u"]:
        vel_array = point_data.GetArray(name)
        if vel_array is not None:
            break

    if vel_array is None:
        print("Available point data arrays:")
        for i in range(point_data.GetNumberOfArrays()):
            print(f"  - {point_data.GetArrayName(i)}")
        raise ValueError(
            "Could not find velocity array (expected 'v', 'velocity', or 'u')"
        )

    num_points = data.GetNumberOfPoints()

    if num_points == 0:
        return 0.0, 0.0, 0.0, 0.0, 0

    u_x_sum = 0.0
    u_y_sum = 0.0

    for i in range(num_points):
        vel = vel_array.GetTuple(i)
        u_x_sum += vel[0]
        u_y_sum += vel[1] if len(vel) > 1 else 0.0

    avg_u_x = u_x_sum / num_points
    avg_u_y = u_y_sum / num_points

    flow = avg_u_x * boundary_length

    if use_negative_normal:
        flow = -flow

    Delete(threshold)

    return flow, boundary_length, avg_u_x, avg_u_y, num_points


def compute_flow_integral_direct(
    reader, target_x, tolerance, use_negative_normal=False
):
    """Alternative: Use integration filter to compute flow."""

    threshold = Threshold(Input=reader)
    threshold.Scalars = ["POINTS", "x"]
    threshold.LowerThreshold = target_x - tolerance
    threshold.UpperThreshold = target_x + tolerance

    integrate = IntegrateVariables(Input=threshold)
    integrate.UpdatePipeline()

    data = servermanager.Fetch(integrate)

    bounds = data.GetBounds()
    ymin, ymax = bounds[2], bounds[3]
    boundary_length = ymax - ymin

    num_points = data.GetNumberOfPoints()

    point_data = data.GetPointData()
    vel_array = None
    for name in ["v", "velocity", "u"]:
        vel_array = point_data.GetArray(name)
        if vel_array is not None:
            break

    if vel_array is None or num_points == 0:
        return 0.0, boundary_length, 0.0, 0.0, 0

    u_x_sum = 0.0
    u_y_sum = 0.0
    for i in range(num_points):
        vel = vel_array.GetTuple(i)
        u_x_sum += vel[0]
        u_y_sum += vel[1] if len(vel) > 1 else 0.0

    avg_u_x = u_x_sum / num_points if num_points > 0 else 0.0
    avg_u_y = u_y_sum / num_points if num_points > 0 else 0.0

    flow = avg_u_x * boundary_length
    if use_negative_normal:
        flow = -flow

    Delete(integrate)
    Delete(threshold)

    return flow, boundary_length, avg_u_x, avg_u_y, num_points


def compute_flow_on_boundary(result_dir, scenario_type, time_step=None):
    """Main function to compute flow on inlet and outlet boundaries."""

    v_file = Path(result_dir) / "v.bp"
    if not v_file.exists():
        print(f"Error: Velocity file not found: {v_file}")
        print(f"Available files in {result_dir}:")
        for f in Path(result_dir).iterdir():
            print(f"  {f.name}")
        return None

    params = load_simulation_params(result_dir)

    L = float(params.get("L", 138.0))
    R_in = float(params.get("R_in", 1.57))
    R_out = float(params.get("R_out", 1.2))

    print(f"Geometry parameters: L={L}, R_in={R_in}, R_out={R_out}")

    print(f"Loading velocity from: {v_file}")
    reader = OpenDataFile(str(v_file))

    if time_step is not None:
        reader.TimeStep = time_step

    UpdatePipeline()

    arrays = get_point_data_arrays(reader)
    print(f"Available point arrays: {arrays}")

    data = servermanager.Fetch(reader)
    bounds = data.GetBounds()
    xmin, xmax = bounds[0], bounds[1]
    ymin, ymax = bounds[2], bounds[3]

    print(f"Mesh bounds: x=[{xmin:.3f}, {xmax:.3f}], y=[{ymin:.3f}, {ymax:.3f}]")

    tolerance = (xmax - xmin) * 0.01

    print(f"\n{'=' * 60}")
    print(f"--- Inlet (x = {xmin:.3f}) ---")
    inlet_flow, inlet_length, inlet_u_x, inlet_u_y, n_inlet = compute_boundary_flow(
        reader, xmin, tolerance, use_negative_normal=True
    )
    print(f"  Boundary length: {inlet_length:.3f} mm")
    print(f"  Points found: {n_inlet}")
    print(f"  Avg velocity: ({inlet_u_x:.3f}, {inlet_u_y:.3f}) mm/s")
    print(f"  Flow rate (Q_in = -∫ u_x dy): {inlet_flow:.6f} mm²/s")

    print(f"\n{'=' * 60}")
    print(f"--- Outlet (x = {xmax:.3f}) ---")
    outlet_flow, outlet_length, outlet_u_x, outlet_u_y, n_outlet = (
        compute_boundary_flow(reader, xmax, tolerance, use_negative_normal=False)
    )
    print(f"  Boundary length: {outlet_length:.3f} mm")
    print(f"  Points found: {n_outlet}")
    print(f"  Avg velocity: ({outlet_u_x:.3f}, {outlet_u_y:.3f}) mm/s")
    print(f"  Flow rate (Q_out = ∫ u_x dy): {outlet_flow:.6f} mm²/s")

    ratio = inlet_flow / outlet_flow if abs(outlet_flow) > 1e-12 else float("inf")
    conservation_error = abs(inlet_flow - outlet_flow)

    print(f"\n{'=' * 60}")
    print(f"RESULTS:")
    print(f"  Inlet flow:  {inlet_flow:.6f} mm²/s")
    print(f"  Outlet flow: {outlet_flow:.6f} mm²/s")
    print(f"  Flow ratio (inlet/outlet): {ratio:.6f}")
    if ratio > 0:
        print(f"  Reciprocal (outlet/inlet): {1 / ratio:.6f}")
    print(
        f"  Conservation error: {conservation_error:.6f} mm²/s ({100 * conservation_error / max(abs(inlet_flow), 1e-12):.2f}%)"
    )
    print(f"{'=' * 60}")

    result = {
        "scenario": scenario_type,
        "result_dir": str(result_dir),
        "L": L,
        "R_in": R_in,
        "R_out": R_out,
        "inlet_flow_mm2_s": inlet_flow,
        "outlet_flow_mm2_s": outlet_flow,
        "flow_ratio_inlet_over_outlet": ratio,
        "flow_ratio_outlet_over_inlet": 1 / ratio if ratio > 0 else None,
        "inlet_length_mm": inlet_length,
        "outlet_length_mm": outlet_length,
        "inlet_avg_velocity_mm_s": [inlet_u_x, inlet_u_y],
        "outlet_avg_velocity_mm_s": [outlet_u_x, outlet_u_y],
        "mesh_bounds": {
            "xmin": xmin,
            "xmax": xmax,
            "ymin": ymin,
            "ymax": ymax,
        },
        "n_inlet_points": n_inlet,
        "n_outlet_points": n_outlet,
    }

    Delete(reader)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Compute inlet/outlet flow ratio from simulation results"
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        required=True,
        help="Path to the simulation result directory containing v.bp file",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        choices=["stenosis", "stenosis_with_tree_2d"],
        default="stenosis",
        help="Scenario type (for reference)",
    )
    parser.add_argument(
        "--time-step",
        type=int,
        default=None,
        help="Time step to analyze (default: last)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="OutputJSON file for results",
    )

    args = parser.parse_args()

    result = compute_flow_on_boundary(
        args.result_dir,
        args.scenario,
        args.time_step,
    )

    if result is None:
        sys.exit(1)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2, default=float)
        print(f"\nResults saved to: {args.output}")

    return result


if __name__ == "__main__":
    main()
