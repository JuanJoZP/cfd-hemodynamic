import os

import cadquery as cq
import gmsh
import numpy as np


def generate_coupling_geometry(start_pt, direction, r_start, r_end, length_ratio=1.0):
    """
    Generate a simple lofted transition (truncated cone) between two radii.

    Args:
        start_pt: tuple/array of starting coordinates (x, y, z)
        direction: tuple/array normalized direction vector
        r_start: radius at the start
        r_end: radius at the end
        length_ratio: controls length relative to radius difference.
                      L = abs(r_start - r_end) * length_ratio.
                      Min effective length is enforced to avoid degenerate solids.
    """
    start_v = cq.Vector(*start_pt)
    dir_v = cq.Vector(*direction).normalized()

    # Calculate length based on radius difference and ratio
    # If radii are equal, use a small default length or just length_ratio if seen as absolute length
    delta_r = abs(r_start - r_end)
    if delta_r < 1e-6:
        # constant radius, just a small cylinder or specified length
        length = max(0.5, length_ratio)  # Default small length
    else:
        length = delta_r * length_ratio

    # Ensure minimum length for meshability
    length = max(length, 0.5)

    end_v = start_v + dir_v * length

    # Create the loft
    # We construct it on the standard XY plane then orient it

    # Local construction: Starts at (0,0,0) going +X (or +Z, CadQuery default is Z for workplanes usually)
    # Let's use Workplane("XY") -> normal is Z.

    # However, CadQuery's loft is easier if we place sketches on planes.

    # Method: Create a plane at start_pt with normal=direction
    plane_start = cq.Plane(origin=start_v, xDir=None, normal=dir_v)

    # Plane at end
    plane_end = cq.Plane(origin=end_v, xDir=None, normal=dir_v)

    # Circle 1
    c1 = cq.Workplane(plane_start).circle(r_start)

    # Circle 2
    c2 = cq.Workplane(plane_end).circle(r_end)

    # Loft
    # To loft in CQ, we need the wires in the same stack or use .loft() on a workplane having multiple pending wires
    # But across different planes it is trickier to chain nicely in one liner if planes are arbitrary.
    # Easier strategy: Define on local Z, then rotate/translate.

    solid_local = (
        cq.Workplane("XY")
        .workplane(offset=0)
        .circle(r_start)
        .workplane(offset=length)
        .circle(r_end)
        .loft(combine=True)
    )

    # Now transform solid_local to match start_pt and direction
    # Initial direction is +Z (0,0,1)

    # 1. Rotate
    initial_dir = np.array([0, 0, 1])
    target_dir = np.array(direction)
    target_dir = target_dir / np.linalg.norm(target_dir)

    rot_axis = np.cross(initial_dir, target_dir)
    norm_axis = np.linalg.norm(rot_axis)

    if norm_axis > 1e-6:
        dot = np.dot(initial_dir, target_dir)
        angle_deg = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))
        # Rotate around the origin (which is the center of the first circle)
        solid_oriented = solid_local.rotate((0, 0, 0), tuple(rot_axis), angle_deg)
    else:
        # parallel or anti-parallel
        if np.dot(initial_dir, target_dir) < 0:
            # -Z direction
            solid_oriented = solid_local.rotate((0, 0, 0), (1, 0, 0), 180)
        else:
            solid_oriented = solid_local

    # 2. Translate to start point
    solid_final = solid_oriented.translate(tuple(start_pt))

    return solid_final, length


def mesh_coupling(solid, filename_msh):
    """
    Mesh the coupling solid.
    Tags the lateral surface (cone) as 'Wall' and the volume as 'Fluid'.
    Planar surfaces (caps) are left untagged (internal interfaces).
    """
    import os

    import gmsh

    # Generate BREP temp
    brep_tmp = filename_msh.replace(".msh", ".brep")

    # Export using exporters
    from cadquery import exporters

    exporters.export(solid, brep_tmp)

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.merge(brep_tmp)

    # Sync to get entities
    try:
        gmsh.model.occ.synchronize()
    except:
        pass

    # Tag Volume
    vols = gmsh.model.getEntities(3)
    if vols:
        gmsh.model.addPhysicalGroup(3, [vols[0][1]], 4)  # Fluid
        gmsh.model.setPhysicalName(3, 4, "Fluid")

    # Tag Surfaces
    surfs = gmsh.model.getBoundary(vols)
    wall_surfs = []

    for _, tag in surfs:
        # Check surface type
        # Plane = curvature is 0. Cone/Cylinder = curvature > 0
        # Gmsh occ getType: "Plane", "Cylinder", "Cone", "BSplineSurface", etc.
        try:
            stype = gmsh.model.getType(2, abs(tag))
            print(f"[DEBUG] Surface {tag} has type {stype}")

            # If it's a Plane, it's likely a cap (interface) -> Skip tagging
            if stype == "Plane":
                continue
            # If Cone, Cylinder, BSpline -> Wall
            wall_surfs.append(tag)

        except Exception as e:
            # Fallback based on bounding box or just assume it's wall if not sure?
            # Safe bet: if we can't determine, maybe don't tag to avoid blocking flow?
            # Or tag as wall? A cylinder wall is curved.
            print(f"[WARN] Could not determine type for surface {tag}: {e}")
            pass

    if wall_surfs:
        # Create physical group for Wall
        # Note: In other parts of code 'walls' might have ID 3. Let's stick to that.
        p_tag = gmsh.model.addPhysicalGroup(2, wall_surfs, 3)
        gmsh.model.setPhysicalName(2, 3, "walls")  # consistent with graph_to_mesh

    # Mesh generation
    gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay typically robust
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 1)
    gmsh.option.setNumber("Mesh.Smoothing", 10)

    # Generate 2D then 3D
    gmsh.model.mesh.generate(3)

    # Check if 2D mesh exists? generate(3) implies 2

    gmsh.write(filename_msh)
    gmsh.finalize()

    if os.path.exists(brep_tmp):
        os.remove(brep_tmp)
    print(f"[OK] Coupling mesh written to {filename_msh}")
