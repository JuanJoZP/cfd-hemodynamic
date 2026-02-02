import argparse
import cadquery as cq
import gmsh
import numpy as np
import os
import sys
import traceback

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
        "--radius", type=float, required=True, help="Healthy vessel radius (R_max)"
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

    return parser.parse_args()


def generate_stenosis_geometry(start, end, max_radius, min_radius, slope):
    start_v = np.array(start)
    end_v = np.array(end)
    vector = end_v - start_v
    length = np.linalg.norm(vector)

    if length == 0:
        raise ValueError("Start and End points are identical.")

    # Local 2D coordinates for the profile
    # X axis -> Along the vessel (0 to L)
    # Y axis -> Radial direction (Normal)

    # Points definition based on user prompt:
    # Initial P: (0, max_radius)
    # End P: (L, max_radius)
    # Mid P: (L/2, -min_radius)  <-- "direccion opuesta al desfase inicial"

    p_start = (0, max_radius)
    p_end = (length, max_radius)
    p_mid = (
        length / 2,
        min_radius,
    )  # Stenosis narrows to min_radius (positive, toward axis)

    # Control Points (CP)
    # y_coord = same as initial/final -> max_radius
    # x_offset from center (L/2)
    # Ratio: (Stenosis Height) / (Dist to Center) = Slope
    # Stenosis Height = Y_max - Y_mid = max_radius - min_radius
    # Dist to Center (tangent dir) = Height / Slope

    height = max_radius - min_radius
    dist_x = height / slope

    cp1 = (length / 2 - dist_x, max_radius)
    cp2 = (length / 2 + dist_x, max_radius)

    # Validate CP order to ensure no loop-backs
    # We need 0 <= cp1_x <= mid_x <= cp2_x <= L
    if cp1[0] < 0 or cp2[0] > length:
        print(
            f"[WARN] Slope {slope} results in control points outside the segment length."
        )

    points_2d = [(float(p[0]), float(p[1])) for p in [p_start, cp1, p_mid, cp2, p_end]]
    print(f"[INFO] Profile Points (Local 2D): {points_2d}")

    # Build geometry with CadQuery
    # We construct the profile in XY plane, then revolve around X axis
    # Use piecewise approach: straight lines at extremes, spline only for stenosis
    # This prevents the spline from curving outward at control points

    # Profile segments:
    # 1. Line: Start (0, Rmax) -> CP1
    # 2. Spline: CP1 -> Mid -> CP2 (the stenosis section)
    # 3. Line: CP2 -> End (L, Rmax)
    # 4. Cap: End -> (L, 0)
    # 5. Axis: (L, 0) -> (0, 0)
    # 6. Cap: (0, 0) -> Start

    # Convert points to 3D vectors
    v_start = cq.Vector(float(p_start[0]), float(p_start[1]), 0.0)
    v_cp1 = cq.Vector(float(cp1[0]), float(cp1[1]), 0.0)
    v_mid = cq.Vector(float(p_mid[0]), float(p_mid[1]), 0.0)
    v_cp2 = cq.Vector(float(cp2[0]), float(cp2[1]), 0.0)
    v_end = cq.Vector(float(p_end[0]), float(p_end[1]), 0.0)

    # Create edges
    edge_line_start = cq.Edge.makeLine(v_start, v_cp1)  # Horizontal line at max_radius

    # Spline for the stenosis section with tangents pointing inward/horizontal
    stenosis_points = [v_cp1, v_mid, v_cp2]
    tangent_in = cq.Vector(1.0, 0.0, 0.0)  # Horizontal at entry
    tangent_out = cq.Vector(1.0, 0.0, 0.0)  # Horizontal at exit
    edge_spline = cq.Edge.makeSpline(
        stenosis_points, tangents=[tangent_in, tangent_out]
    )

    edge_line_end = cq.Edge.makeLine(v_cp2, v_end)  # Horizontal line at max_radius

    edge_cap_end = cq.Edge.makeLine(
        cq.Vector(length, max_radius, 0), cq.Vector(length, 0, 0)
    )
    edge_axis = cq.Edge.makeLine(cq.Vector(length, 0, 0), cq.Vector(0, 0, 0))
    edge_cap_start = cq.Edge.makeLine(cq.Vector(0, 0, 0), cq.Vector(0, max_radius, 0))

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

    # Revolve
    # Revolve around X-axis (0,0,0) to (1,0,0)
    solid_local = (
        cq.Workplane("XY")
        .newObject([wire])
        .toPending()
        .revolve(360, (0, 0, 0), (1, 0, 0))
    )

    # Now we need to orient this solid from Local (X-axis) to Global (Start->End)
    # Local Start is (0,0,0). Global Start is args.start.
    # Local Dir is (1,0,0). Global Dir is (end - start).

    # Translation
    solid_moved = solid_local.translate(tuple(float(x) for x in start_v))

    # Rotation
    # We need to rotate (1,0,0) to match normalized(vector)
    target_dir = vector / length
    initial_dir = np.array([1, 0, 0])

    # Axis of rotation is cross product
    rot_axis = np.cross(initial_dir, target_dir)
    msg = np.linalg.norm(rot_axis)

    if msg > 1e-6:
        # Check angle
        # dot = cos(theta)
        dot = np.dot(initial_dir, target_dir)
        angle_deg = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))

        # Rotate
        # CadQuery rotate about an axis at a point (Start)
        solid_final = solid_moved.rotate(
            tuple(start_v), tuple(start_v + rot_axis), angle_deg
        )
    else:
        # Parallel or anti-parallel
        if np.dot(initial_dir, target_dir) < 0:
            # 180 degrees flip (anti-parallel)
            # any axis perpendicular to X works, e.g., Y or Z
            solid_final = solid_moved.rotate(
                tuple(start_v), tuple(start_v + np.array([0, 1, 0])), 180
            )
        else:
            solid_final = solid_moved

    return solid_final


def mesh_and_export(solid, filename_brep, filename_msh, start_pt, end_pt):
    # Export BREP
    cq.exporters.export(solid, filename_brep)
    print(f"[INFO] BREP exported to {filename_brep}")

    # Initialize GMSH
    gmsh.initialize()
    gmsh.model.add("Stenosis")

    # Import BREP
    gmsh.merge(filename_brep)
    try:
        gmsh.model.occ.synchronize()
    except:
        pass

    # Tagging
    # We need to identify Inlet, Outlet, and Wall faces
    # Strategy: Find faces closest to Start Point (Inlet) and End Point (Outlet)

    volumes = gmsh.model.getEntities(3)
    if not volumes:
        print("[ERROR] No volumes found.")
        return

    fluid_vol_tag = volumes[0][1]
    gmsh.model.addPhysicalGroup(3, [fluid_vol_tag], FLUID_TAG)
    gmsh.model.setPhysicalName(3, FLUID_TAG, "Fluid")

    surfaces = gmsh.model.getBoundary(volumes)  # List of (dim, tag)

    # Helper for centroid distance
    def get_center_dist(tag, point):
        bbox = gmsh.model.getBoundingBox(2, tag)
        center = np.array(
            [(bbox[0] + bbox[3]) / 2, (bbox[1] + bbox[4]) / 2, (bbox[2] + bbox[5]) / 2]
        )
        return np.linalg.norm(center - np.array(point))

    inlet_surf = None
    outlet_surf = None
    min_dist_in = float("inf")
    min_dist_out = float("inf")

    # Identify Inlet and Outlet
    for dim, tag in surfaces:
        d_in = get_center_dist(tag, start_pt)
        d_out = get_center_dist(tag, end_pt)

        # Heuristic: Closest face to start/end points
        if d_in < min_dist_in:
            min_dist_in = d_in
            inlet_surf = tag

        if d_out < min_dist_out:
            min_dist_out = d_out
            outlet_surf = tag

    # Assign Tags
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
        f"[INFO] Tags Assigned: Inlet={inlet_surf}, Outlet={outlet_surf}, Walls={wall_surfaces}"
    )

    # Mesh Generation - curvature-based sizing and smoothing
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
        # Calculate min_radius from severity: R_min = (1 - η) * R
        if not 0 <= args.severity <= 1:
            raise ValueError(f"Severity must be in [0, 1], got {args.severity}")
        min_radius = (1 - args.severity) * args.radius
        print(f"[INFO] Severity η={args.severity} → R_min = {min_radius:.4f}")

        solid = generate_stenosis_geometry(
            args.start, args.end, args.radius, min_radius, args.pendiente
        )

        brep_path = "stenosis.brep"
        msh_path = "stenosis.msh"

        mesh_and_export(solid, brep_path, msh_path, args.start, args.end)

    except Exception as e:
        print(f"[ERROR] {e}")
        traceback.print_exc()
        sys.exit(1)
