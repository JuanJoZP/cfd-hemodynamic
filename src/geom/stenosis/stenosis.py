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
    parser.add_argument(
        "--artery_mesh_size_from_curvature",
        dest="artery_mesh_size_from_curvature",
        type=int,
        default=20,
        help="Number of elements per 2*pi (curvature based sizing)",
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


def label_and_group(start_pt, end_pt):
    vols = gmsh.model.getEntities(3)
    if not vols:
        return []

    gmsh.model.addPhysicalGroup(3, [v[1] for v in vols], FLUID_TAG)
    gmsh.model.setPhysicalName(3, FLUID_TAG, "Fluid")

    surf_count = {}
    for vol in vols:
        bnd = gmsh.model.getBoundary([vol], oriented=False)
        for dim, tag in bnd:
            surf_count[tag] = surf_count.get(tag, 0) + 1

    outer_surfs = [tag for tag, count in surf_count.items() if count == 1]

    surf_data = []
    for tag in outer_surfs:
        bbox = gmsh.model.getBoundingBox(2, tag)
        center = np.array(
            [(bbox[0] + bbox[3]) / 2, (bbox[1] + bbox[4]) / 2, (bbox[2] + bbox[5]) / 2]
        )
        d_in = np.linalg.norm(center - np.array(start_pt))
        d_out = np.linalg.norm(center - np.array(end_pt))
        surf_data.append((tag, d_in, d_out))

    # Sort by distance to start_pt (inlet) and end_pt (outlet) and pick top 5
    surf_data_in = sorted(surf_data, key=lambda x: x[1])
    inlets = [d[0] for d in surf_data_in[:5]]

    surf_data_out = sorted(surf_data, key=lambda x: x[2])
    outlets = [d[0] for d in surf_data_out[:5]]
    walls = [t for t in outer_surfs if t not in inlets and t not in outlets]

    if inlets:
        gmsh.model.addPhysicalGroup(2, inlets, INLET_TAG)
        gmsh.model.setPhysicalName(2, INLET_TAG, "Inlet")
    if outlets:
        gmsh.model.addPhysicalGroup(2, outlets, OUTLET_TAG)
        gmsh.model.setPhysicalName(2, OUTLET_TAG, "Outlet")
    if walls:
        gmsh.model.addPhysicalGroup(2, walls, WALL_TAG)
        gmsh.model.setPhysicalName(2, WALL_TAG, "Wall")

    return outer_surfs


def get_radial_scale_matrix(start_pt, end_pt, scale=0.8):
    vec = np.array(end_pt) - np.array(start_pt)
    length = np.linalg.norm(vec)
    direction = vec / length if length > 0 else np.array([1, 0, 0])

    S = np.eye(3) * scale + (1 - scale) * np.outer(direction, direction)
    offset = np.array(start_pt) - S @ np.array(start_pt)

    M = np.eye(4)
    M[:3, :3] = S
    M[:3, 3] = offset
    return M.flatten().tolist()


def mesh_and_export(
    solid,
    filename_brep,
    filename_msh,
    start_pt,
    end_pt,
    artery_mesh_size_from_curvature=20,
    radius_in=2.0,
    radius_out=2.0,
):
    cq.exporters.export(solid, filename_brep)
    print("[INFO] BREP exported to {}".format(filename_brep))

    gmsh.initialize()
    gmsh.model.add("Stenosis")

    vols = gmsh.model.occ.importShapes(filename_brep)

    import math

    target_mesh_size = (2 * math.pi * radius_out) / artery_mesh_size_from_curvature

    gmsh.initialize()
    gmsh.model.add("Stenosis_S2_Structured")
    vols = gmsh.model.occ.importShapes(filename_brep)
    gmsh.model.occ.synchronize()

    # Get initial bounding box to dimension cutting planes
    v_tags = [v[1] for v in vols]
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.occ.getBoundingBox(3, v_tags[0])
    L = max(xmax - xmin, ymax - ymin, zmax - zmin) * 2.5

    start_v = np.array(start_pt)
    end_v = np.array(end_pt)
    vector = end_v - start_v
    length = np.linalg.norm(vector)
    center_pt = start_v + vector * 0.5
    target_dir = vector / length
    internal_centers = set()
    for c in gmsh.model.getEntities(1):
        if "Circle" in gmsh.model.getType(1, c[1]):
            bb = gmsh.model.getBoundingBox(1, c[1])
            c_pt = np.array(
                [(bb[0] + bb[3]) / 2, (bb[1] + bb[4]) / 2, (bb[2] + bb[5]) / 2]
            )
            dist_s = np.linalg.norm(c_pt - start_v)
            dist_e = np.linalg.norm(c_pt - end_v)
            if dist_s > 1e-3 and dist_e > 1e-3:
                internal_centers.add(tuple(np.round(c_pt, 4)))

    # Ensure topological dimensions for central box and cutting planes
    a = 0.5 * radius_out
    L_ext = max(xmax - xmin, ymax - ymin, zmax - zmin) * 2.5
    W = radius_out * 3.0

    surfs_to_rotate = []

    # 1. Four inner planes replacing the volume box (prevents axial extension outside cylinder caps)
    box_pts = [(a / 2, a / 2), (-a / 2, a / 2), (-a / 2, -a / 2), (a / 2, -a / 2)]
    for i in range(4):
        p1 = gmsh.model.occ.addPoint(-L_ext / 2, box_pts[i][0], box_pts[i][1])
        p2 = gmsh.model.occ.addPoint(
            -L_ext / 2, box_pts[(i + 1) % 4][0], box_pts[(i + 1) % 4][1]
        )
        l_box = gmsh.model.occ.addLine(p1, p2)
        ext = gmsh.model.occ.extrude([(1, l_box)], L_ext, 0, 0)
        for e in ext:
            if e[0] == 2:
                surfs_to_rotate.append((2, e[1]))

    # 2. Four rectangles at angles from corners (Butterfly O-grid cuts)
    dirs = [(1, 1), (-1, 1), (-1, -1), (1, -1)]
    for p, d in zip(box_pts, dirs):
        p1 = gmsh.model.occ.addPoint(-L_ext / 2, p[0], p[1])
        p2 = gmsh.model.occ.addPoint(-L_ext / 2, p[0] + d[0] * W, p[1] + d[1] * W)
        l_diag = gmsh.model.occ.addLine(p1, p2)
        ext = gmsh.model.occ.extrude([(1, l_diag)], L_ext, 0, 0)
        for e in ext:
            if e[0] == 2:
                surfs_to_rotate.append((2, e[1]))

    # 3. Transverse planes matching inner B-Spline seams
    for c_tuple in internal_centers:
        axial_pos = np.dot(np.array(c_tuple) - start_v, target_dir)
        local_x = axial_pos - (length / 2.0)
        r_pl = gmsh.model.occ.addRectangle(-W * 2, -W * 2, 0, W * 4, W * 4)
        gmsh.model.occ.rotate([(2, r_pl)], 0, 0, 0, 0, 1, 0, math.pi / 2)
        gmsh.model.occ.translate([(2, r_pl)], local_x, 0, 0)
        surfs_to_rotate.append((2, r_pl))

    tools = surfs_to_rotate

    # Rotate by 45 degrees around X to perfectly align the O-grid diagonal cuts
    # with the parametric CAD revolution seams (typically lying on Y=0 or Z=0 axes)
    gmsh.model.occ.rotate(tools, 0, 0, 0, 1, 0, 0, math.pi / 4)

    # Rotate the whole toolkit to align with the artery cylinder
    target_dir = vector / length
    initial_dir = np.array([1, 0, 0])
    rot_axis = np.cross(initial_dir, target_dir)
    axis_len = np.linalg.norm(rot_axis)

    if axis_len > 1e-6:
        rot_axis = rot_axis / axis_len
        dot = np.dot(initial_dir, target_dir)
        angle = np.arccos(np.clip(dot, -1.0, 1.0))
        gmsh.model.occ.rotate(
            tools, 0, 0, 0, rot_axis[0], rot_axis[1], rot_axis[2], angle
        )
    elif np.dot(initial_dir, target_dir) < 0:
        gmsh.model.occ.rotate(tools, 0, 0, 0, 0, 1, 0, math.pi)

    # Translate back to the center of the segment
    gmsh.model.occ.translate(tools, center_pt[0], center_pt[1], center_pt[2])

    # Fragment the volume with the O-grid butterfly toolkit
    gmsh.model.occ.fragment(vols, tools)
    gmsh.model.occ.synchronize()

    all_vols = gmsh.model.getEntities(3)
    bnd = gmsh.model.getBoundary(all_vols, oriented=False)
    bnd_tags = [s[1] for s in bnd]

    all_surfs = gmsh.model.getEntities(2)
    excess_surfs = [(2, s[1]) for s in all_surfs if s[1] not in bnd_tags]

    if excess_surfs:
        gmsh.model.removeEntities(excess_surfs, recursive=True)

    # The fragment makes new objects -> re-label
    outer_surfs = label_and_group(start_pt, end_pt)

    # Refinar N_azimutal y N_radial según el target_mesh_size y la topología O-grid
    elem_circ = max(2, int(round(artery_mesh_size_from_curvature / 4.0)))
    N_azimutal = elem_circ

    # Radio promedio para el cálculo de divisiones radiales
    # a es el lado del cuadrado central (Butterfly box)
    rad_dist_avg = ((radius_in + radius_out) / 2.0) - (a * math.sqrt(2) / 2.0)
    elem_rad = max(2, int(round(rad_dist_avg / target_mesh_size)))
    N_radial = elem_rad

    curves = gmsh.model.getEntities(1)
    axial_N_cache = {}
    axial_tol = max(
        float(1e-4 * length), 1e-5
    )  # Tolerancia para agrupar tramos longitudinales

    for dim, tag in curves:
        bnd = gmsh.model.getBoundary([(1, tag)], oriented=False)
        if len(bnd) < 2:
            continue

        p1 = np.array(gmsh.model.getValue(0, bnd[0][1], []))
        p2 = np.array(gmsh.model.getValue(0, bnd[1][1], []))
        vec = p2 - p1
        dist = np.linalg.norm(vec)
        if dist < 1e-8:
            continue

        t_vec = vec / dist
        dot_axial = abs(np.dot(t_vec, target_dir))

        # Proyección axial del segmento
        axial_proj = abs(np.dot(vec, target_dir))

        if dot_axial > 0.7:  # Longitudinal / Axial
            # Asegurar consistencia entre tramos paralelos mediante una caché con tolerancia
            matched_N = None
            for p_val, n_val in axial_N_cache.items():
                if abs(p_val - axial_proj) < axial_tol:
                    matched_N = n_val
                    break

            if matched_N is not None:
                N_curr = matched_N
            else:
                N_curr = max(2, int(round(axial_proj / target_mesh_size))) + 1
                axial_N_cache[axial_proj] = N_curr

            gmsh.model.mesh.setTransfiniteCurve(tag, N_curr)

        else:
            # Determinar si es Radial o Azimutal usando el vector desde el eje
            mid_pt = (p1 + p2) / 2.0
            r_vec_axis = mid_pt - start_v
            r_vec_axis = r_vec_axis - np.dot(r_vec_axis, target_dir) * target_dir
            r_norm = np.linalg.norm(r_vec_axis)

            if r_norm > 1e-6:
                r_unit = r_vec_axis / r_norm
                dot_radial = abs(np.dot(t_vec, r_unit))
            else:
                dot_radial = 0

            if dot_radial > 0.8:  # Radial
                gmsh.model.mesh.setTransfiniteCurve(tag, N_radial)
            else:  # Azimutal (Arco o lado de caja central)
                gmsh.model.mesh.setTransfiniteCurve(tag, N_azimutal)

    gmsh.option.setNumber("Mesh.RecombineAll", 1)

    surfaces = gmsh.model.getEntities(2)
    for s in surfaces:
        gmsh.model.mesh.setTransfiniteSurface(s[1])
        # gmsh.model.mesh.setRecombine(2, s[5]) # Alternativa si no usas RecombineAll

    volumes = gmsh.model.getEntities(3)
    for v in volumes:
        gmsh.model.mesh.setTransfiniteVolume(v[1])

    # If pure transfinite generation fails due to topological mismatches, it falls back to Delaunay
    try:
        gmsh.model.mesh.generate(3)
    except Exception as e:
        print("[WARN] Pure transfinite failed, attempting automatic hybrid meshing:", e)
        # Disable strict transfinite volumes and use 3D Hexa subdivision
        gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)  # All Hexa
        gmsh.option.setNumber("Mesh.Algorithm", 8)  # Frontal-Delaunay Quads
        gmsh.model.mesh.generate(3)

    gmsh.write(filename_msh)
    print(f"[INFO] Strategy 2 (Structured) mesh written to {filename_msh}")
    gmsh.clear()
    gmsh.finalize()


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

        mesh_and_export(
            solid,
            brep_path,
            msh_path,
            args.start,
            args.end,
            artery_mesh_size_from_curvature=args.artery_mesh_size_from_curvature,
            radius_in=args.radius_in,
            radius_out=args.radius_out,
        )

    except Exception as e:
        print(f"[ERROR] {e}")
        traceback.print_exc()
        sys.exit(1)
