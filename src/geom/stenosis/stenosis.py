# -*- coding: utf-8 -*-
import argparse
import os
import sys
import traceback

import cadquery as cq
import gmsh
import numpy as np

# Tags for physical groups
INLET_TAG = 1
OUTLET_TAG = 2
WALL_TAG = 3
FLUID_TAG = 4


def parse_args():
    parser = argparse.ArgumentParser(description="Generate stenosis geometry and mesh.")

    parser.add_argument(
        "--start",
        nargs=3,
        type=float,
        required=True,
        metavar=("X", "Y", "Z"),
        help="Start point coordinates",
    )
    parser.add_argument(
        "--end",
        nargs=3,
        type=float,
        required=True,
        metavar=("X", "Y", "Z"),
        help="End point coordinates",
    )
    parser.add_argument(
        "--radius-in",
        dest="radius_in",
        type=float,
        required=True,
        help="Inlet vessel radius (at start)",
    )
    parser.add_argument(
        "--radius-out",
        dest="radius_out",
        type=float,
        required=True,
        help="Outlet vessel radius (at end), must be <= radius_in",
    )
    parser.add_argument(
        "--severity",
        type=float,
        required=True,
        help="Stenosis severity η ∈ [0,1]. η=0: healthy vessel, η→1: extreme constriction. R_min = (1-η)*R",
    )
    parser.add_argument(
        "--slope",
        dest="pendiente",
        type=float,
        required=True,
        help="Slope parameter (PENDIENTE)",
    )
    parser.add_argument(
        "--output_dir",
        dest="output",
        type=str,
        required=True,
    )

    return parser.parse_args()


def generate_stenosis_geometry(
    start, end, radius_in, radius_out, min_radius, slope, position=0.5
):
    """
    Generate a stenosis geometry with progressive tapering.

    Args:
        start: Start point coordinates (x, y, z)
        end: End point coordinates (x, y, z)
        radius_in: Radius at the inlet (start)
        radius_out: Radius at the outlet (end), must be <= radius_in
        min_radius: Minimum radius at the stenosis center
        slope: Slope parameter controlling stenosis steepness
        position: Relative position of stenosis center [0, 1]. 0=proximal (start), 1=distal (end).
    """
    if slope >= 0.85:
        raise ValueError(
            """Valores tan altos de slope generan geometrias dificiles de mallar,
            si gmsh logra mallarla por lo general es una malla de baja calidad"""
        )

    if radius_out > radius_in:
        raise ValueError(
            "radius_out ({}) must be <= radius_in ({})".format(radius_out, radius_in)
        )

    if not (0.0 <= position <= 1.0):
        raise ValueError("Position must be in [0, 1], got {}".format(position))

    start_v = np.array(start)
    end_v = np.array(end)
    vector = end_v - start_v
    length = np.linalg.norm(vector)

    if length == 0:
        raise ValueError("Start and End points are identical.")

    # local 2D coordinates (axial, radial)

    r_base_start = radius_in
    r_base_mid = (radius_in + radius_out) / 2
    r_base_end = radius_out

    p_start = (0, r_base_start)
    p_end = (length, r_base_end)

    # Calculate mid point based on position
    mid_x = length * position
    p_mid = (mid_x, min_radius)

    # control points follow the line from (0, r_in) to (L, r_out)
    # slope to determine how far the CPs are from center
    # Radius at mid_x if it were a straight taper
    r_taper_at_mid = radius_in + (radius_out - radius_in) * position

    height_at_mid = r_taper_at_mid - min_radius

    if height_at_mid < 0:
        raise ValueError(
            "El min_radius es muy grande, debe ser menor o igual a {r_taper_at_mid} at position {position}".format(
                r_taper_at_mid=r_taper_at_mid, position=position
            )
        )

    dist_x = height_at_mid / slope if slope != 0 else length / 4

    # Force a minimum width for the stenosis region if dist_x is too small
    # This prevents points from collapsing onto each other
    min_dist = length * 0.05
    if dist_x < min_dist:
        print(
            "[WARN] Calculated dist_x ({}) too small. Enforcing min_dist ({})".format(
                dist_x, min_dist
            )
        )
        dist_x = min_dist

    # Check boundaries and clamp
    max_dist_avail = min(mid_x, length - mid_x)

    if dist_x >= max_dist_avail:
        print(
            "[WARN] Stenosis region too wide for position {}. Clamping width.".format(
                position
            )
        )
        dist_x = max_dist_avail * 0.95

    cp1_x = mid_x - dist_x
    cp2_x = mid_x + dist_x

    # Calculate radius at CP x-positions based on linear taper from inlet to outlet
    cp1_r = radius_in + (radius_out - radius_in) * (cp1_x / length)
    cp2_r = radius_in + (radius_out - radius_in) * (cp2_x / length)

    cp1 = (float(cp1_x), float(cp1_r))
    cp2 = (float(cp2_x), float(cp2_r))
    p_mid = (float(mid_x), float(min_radius))

    # Validation: Ensure points are strictly ordered in x
    if not (p_start[0] < cp1[0] < p_mid[0] < cp2[0] < p_end[0]):
        raise ValueError(
            "Invalid points order for spline: Start={}, CP1={}, Mid={}, CP2={}, End={}".format(
                p_start, cp1, p_mid, cp2, p_end
            )
        )

    points_2d = [(float(p[0]), float(p[1])) for p in [p_start, cp1, p_mid, cp2, p_end]]
    print("[INFO] Profile Points (Local 2D): {}".format(points_2d))
    print(
        "[INFO] Tapering from radius_in={} to radius_out={}".format(
            radius_in, radius_out
        )
    )
    print("[INFO] Stenosis position: {} (x={:.4f})".format(position, mid_x))

    # construct the profile in XY plane, then revolve around X axis
    # straight lines at extremes, spline only for stenosis

    v_start = cq.Vector(float(p_start[0]), float(p_start[1]), 0.0)
    v_cp1 = cq.Vector(float(cp1[0]), float(cp1[1]), 0.0)
    v_mid = cq.Vector(float(p_mid[0]), float(p_mid[1]), 0.0)
    v_cp2 = cq.Vector(float(cp2[0]), float(cp2[1]), 0.0)
    v_end = cq.Vector(float(p_end[0]), float(p_end[1]), 0.0)

    edge_line_start = cq.Edge.makeLine(v_start, v_cp1)

    taper_slope = (radius_out - radius_in) / length
    tangent_in = cq.Vector(1.0, taper_slope, 0.0)
    tangent_out = cq.Vector(1.0, taper_slope, 0.0)

    stenosis_points = [v_cp1, v_mid, v_cp2]
    edge_spline = cq.Edge.makeSpline(
        stenosis_points, tangents=[tangent_in, tangent_out]
    )

    edge_line_end = cq.Edge.makeLine(v_cp2, v_end)

    edge_cap_end = cq.Edge.makeLine(
        cq.Vector(length, r_base_end, 0), cq.Vector(length, 0, 0)
    )
    edge_axis = cq.Edge.makeLine(cq.Vector(length, 0, 0), cq.Vector(0, 0, 0))
    edge_cap_start = cq.Edge.makeLine(cq.Vector(0, 0, 0), cq.Vector(0, r_base_start, 0))

    wire = cq.Wire.assembleEdges(
        [
            edge_line_start,
            edge_spline,
            edge_line_end,
            edge_cap_end,
            edge_axis,
            edge_cap_start,
        ]
    )

    # revolve
    solid_local = (
        cq.Workplane("XY")
        .newObject([wire])
        .toPending()
        .revolve(360, (0, 0, 0), (1, 0, 0))
    )

    # orient the solid from local (x-axis) to global (Start->End)

    # translate
    solid_moved = solid_local.translate(tuple(float(x) for x in start_v))

    # rotate
    target_dir = vector / length
    initial_dir = np.array([1, 0, 0])

    # Axis of rotation is cross product
    rot_axis = np.cross(initial_dir, target_dir)
    msg = np.linalg.norm(rot_axis)

    if msg > 1e-6:
        dot = np.dot(initial_dir, target_dir)
        angle_deg = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))
        solid_final = solid_moved.rotate(
            tuple(start_v), tuple(start_v + rot_axis), angle_deg
        )
    else:
        # parallel or anti-parallel
        if np.dot(initial_dir, target_dir) < 0:
            # 180 degrees flip (anti-parallel)
            solid_final = solid_moved.rotate(
                tuple(start_v), tuple(start_v + np.array([0, 1, 0])), 180
            )
        else:
            solid_final = solid_moved

    return solid_final


def mesh_and_export(solid, filename_brep, filename_msh, start_pt, end_pt):
    cq.exporters.export(solid, filename_brep)
    print("[INFO] BREP exported to {}".format(filename_brep))

    gmsh.initialize()
    gmsh.model.add("Stenosis")

    gmsh.merge(filename_brep)
    try:
        gmsh.model.occ.synchronize()
    except:
        pass

    # tagging
    volumes = gmsh.model.getEntities(3)
    if not volumes:
        print("[ERROR] No volumes found.")
        return

    fluid_vol_tag = volumes[0][1]
    gmsh.model.addPhysicalGroup(3, [fluid_vol_tag], FLUID_TAG)
    gmsh.model.setPhysicalName(3, FLUID_TAG, "Fluid")

    surfaces = gmsh.model.getBoundary(volumes)

    def get_center_dist(tag, point):
        bbox = gmsh.model.getBoundingBox(2, tag)
        center = np.array(
            [(bbox[0] + bbox[3]) / 2, (bbox[1] + bbox[4]) / 2, (bbox[2] + bbox[5]) / 2]
        )
        return np.linalg.norm(center - np.array(point))

    # Identify Inlet and Outlet robustly
    surf_data = []  # (tag, dist_start, dist_end)
    for dim, tag in surfaces:
        d_in = get_center_dist(tag, start_pt)
        d_out = get_center_dist(tag, end_pt)
        surf_data.append((tag, d_in, d_out))
        print(
            "[DEBUG] Surface {} (tag {}): dist_start={:.4f}, dist_end={:.4f}".format(
                tag, tag, d_in, d_out
            )
        )

    # Sort by distance to start to find inlet
    surf_data.sort(key=lambda x: x[1])
    inlet_surf = surf_data[0][0]

    # Sort by distance to end to find outlet
    # Be careful not to pick the same if geometry is very short, but usually distinct.
    surf_data.sort(key=lambda x: x[2])
    outlet_surf = surf_data[0][0]

    # Just in case they match (degenerate), usually implies length ~ 0
    if inlet_surf == outlet_surf and len(surfaces) > 1:
        print(
            "[WARN] Inlet and Outlet resolved to same surface. Checking next closest..."
        )
        if surf_data[1][0] != inlet_surf:
            outlet_surf = surf_data[1][0]

    wall_surfaces = []
    for dim, tag in surfaces:
        if tag == inlet_surf:
            gmsh.model.addPhysicalGroup(2, [tag], INLET_TAG)
            gmsh.model.setPhysicalName(2, INLET_TAG, "Inlet")
        elif tag == outlet_surf:
            gmsh.model.addPhysicalGroup(2, [tag], OUTLET_TAG)
            gmsh.model.setPhysicalName(2, OUTLET_TAG, "Outlet")
        else:
            wall_surfaces.append(tag)

    if wall_surfaces:
        gmsh.model.addPhysicalGroup(2, wall_surfaces, WALL_TAG)
        gmsh.model.setPhysicalName(2, WALL_TAG, "Wall")

    print(
        "[INFO] Tags Assigned: Inlet={}, Outlet={}, Walls={}".format(
            inlet_surf, outlet_surf, wall_surfaces
        )
    )

    # meshing
    gmsh.option.setNumber("Mesh.Smoothing", 10)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 1)
    gmsh.option.setNumber("Mesh.MinimumElementsPerTwoPi", 20)

    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.optimize("Netgen")

    gmsh.write(filename_msh)
    gmsh.finalize()
    print(f"[INFO] Mesh written to {filename_msh}")


if __name__ == "__main__":
    args = parse_args()

    try:
        if args.radius_out > args.radius_in:
            raise ValueError(
                f"radius_out ({args.radius_out}) must be <= radius_in ({args.radius_in})"
            )

        if not 0 <= args.severity <= 1:
            raise ValueError(f"Severity must be in [0, 1], got {args.severity}")
        r_base_mid = (args.radius_in + args.radius_out) / 2
        min_radius = (1 - args.severity) * r_base_mid
        print(
            "[INFO] radius_in={:.4f}, radius_out={:.4f}".format(
                args.radius_in, args.radius_out
            )
        )
        print(
            "[INFO] Severity η={:.4f} → R_min = {:.4f} (at center)".format(
                args.severity, min_radius
            )
        )

        solid = generate_stenosis_geometry(
            args.start,
            args.end,
            args.radius_in,
            args.radius_out,
            min_radius,
            args.pendiente,
        )

        brep_path = os.path.join(args.output, "stenosis.brep")
        msh_path = os.path.join(args.output, "stenosis.msh")

        mesh_and_export(solid, brep_path, msh_path, args.start, args.end)

    except Exception as e:
        print(f"[ERROR] {e}")
        traceback.print_exc()
        sys.exit(1)
