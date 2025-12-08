import numpy as np
from lxml import etree  # type: ignore
import gmsh
import cadquery as cq

GXL_FILE = "src/geom/tree_structure.xml"
VOXEL_WIDTH = 0.04
OUT_MSH = "src/geom/vessels.msh"

inlet_tag = 1
outlet_tag = 2
wall_tag = 3


def magnitude(v: cq.Vector) -> float:
    return (v.x**2 + v.y**2 + v.z**2) ** 0.5


def create_root_bifurcation(root_node, bif_node, r1, end1_node, end2_node, r2, r3):
    offset = 0.2

    start_inlet = cq.Vector(*root_node)
    end_inlet = (
        cq.Vector(*bif_node) - (cq.Vector(*bif_node) - cq.Vector(*root_node)) * offset
    )
    start_b = cq.Vector(*bif_node)
    end_b1 = cq.Vector(*end1_node)
    end_b2 = cq.Vector(*end2_node)

    # root cylinder
    plane_circle = cq.Plane(
        origin=start_inlet, xDir=(1, 0, 0), normal=end_inlet - start_inlet
    )
    print(end_inlet - start_inlet)
    print(start_inlet)
    inlet_line = cq.Workplane("XY").spline([start_inlet, end_inlet])
    inlet_volume = (
        cq.Workplane(plane_circle).workplane(offset=0).circle(r1).sweep(inlet_line)
    )

    # loft to biger radius at bifurcation
    loft_result = (
        inlet_volume.faces(cq.selectors.NearestToPointSelector(end_inlet))
        .circle(r1)
        .workplane(offset=magnitude(start_b - start_inlet) * offset)
        .circle(r2 if r2 >= r3 else r3)
        .loft(combine=True)
    )
    result = inlet_volume.union(loft_result)

    # wider branch
    end_b = end_b1 if r2 >= r3 else end_b2
    r = r2 if r2 >= r3 else r3

    plane_circle = cq.Plane(origin=start_b, xDir=(1, 0, 0), normal=start_b - start_inlet)
    bif1_line = cq.Workplane("XY").spline(
        [start_b, end_b], tangents=[start_b - start_inlet, end_b - start_b]
    )
    bif1_volume = (
        cq.Workplane(plane_circle).workplane(offset=0).circle(r).sweep(bif1_line)
    )
    result = result.union(bif1_volume)

    # thinner branch
    end_b = end_b1 if r2 < r3 else end_b2
    r = r2 if r2 < r3 else r3

    bif2_line = cq.Workplane("XY").spline(
        [start_b, end_b], tangents=[start_b - start_inlet, end_b - start_b]
    )
    bif2_volume = (
        cq.Workplane(plane_circle).workplane(offset=0).circle(r).sweep(bif2_line)
    )
    result = result.union(bif2_volume)

    wps = [bif1_volume]
    wps.insert(1 if r2 > r3 else 0, bif2_volume)

    return result, wps


def create_bifurcation(bif_node, bif_wp, end1_node, end2_node, r0, r1, r2):
    offset = 0.2

    start_b = cq.Vector(*bif_node)
    end_b1 = cq.Vector(*end1_node)
    end_b2 = cq.Vector(*end2_node)

    # loft to biger radius at bifurcation
    face = bif_wp.faces(cq.selectors.NearestToPointSelector(bif_node)).val()
    normal = face.normalAt(face.Center())
    center = face.Center()

    face2 = bif_wp.faces(cq.selectors.DirectionMinMaxSelector(normal, False)).val()
    center2 = face2.Center()
    height = magnitude(
        cq.Vector(center2.x - center.x, center2.y - center.y, center2.z - center.z)
    )

    result = (
        cq.Workplane(cq.Plane(origin=center, normal=normal))
        .circle(r0)
        .workplane(offset=offset * height)
        .circle(max(r1, r2))
        .loft(combine=True)
    )

    # wider branch
    end_b = end_b1 if r1 >= r2 else end_b2
    r = r1 if r1 >= r2 else r2

    plane_circle = cq.Plane(
        origin=center + offset * height * normal, xDir=(1, 0, 0), normal=normal
    )
    bif1_line = cq.Workplane("XY").spline(
        [start_b + offset * height * normal, end_b], tangents=[normal, end_b - start_b]
    )
    bif1_volume = (
        cq.Workplane(plane_circle).workplane(offset=0).circle(r).sweep(bif1_line)
    )
    result = result.union(bif1_volume)

    # thinner branch
    end_b = end_b1 if r1 < r2 else end_b2
    r = r1 if r1 < r2 else r2

    plane_circle = cq.Plane(origin=center, xDir=(1, 0, 0), normal=normal)
    # TODO: aca la tangente mejor del mismo branch
    bif2_line = cq.Workplane("XY").spline(
        [start_b, end_b], tangents=[normal, end_b - start_b]
    )
    bif2_volume = (
        cq.Workplane(plane_circle).workplane(offset=0).circle(r).sweep(bif2_line)
    )
    result = result.union(bif2_volume)

    wps = [bif1_volume]
    wps.insert(1 if r1 > r2 else 0, bif2_volume)

    return result, wps


def parse_gxl(gxl_path):
    tree = etree.parse(gxl_path)
    root = tree.getroot()

    nodes = {}
    node_types = {}
    edges = []

    for n in root.xpath(".//node"):
        nid = n.get("id")
        # posición
        pos_tup = n.xpath(".//attr[contains(@name,'position')]/tup")
        floats = pos_tup[0].xpath(".//float/text()")
        coords = tuple(float(f) for f in floats)
        nodes[nid] = np.array(coords) * VOXEL_WIDTH
        t = n.xpath(".//attr[contains(@name,'nodeType')]/string/text()")
        node_types[nid] = t[0]

    for e in root.xpath(".//edge"):
        frm = e.get("from")
        to = e.get("to")
        r_attr = e.xpath(".//attr[contains(@name,'radius')]/float/text()")
        radius = float(r_attr[0]) * VOXEL_WIDTH
        edges.append((frm, to, radius))

    return nodes, node_types, edges


def build_mesh(
    nodes: dict[str, tuple[float, float, float]],
    node_types: dict,
    edges: list[tuple[str, str, float]],
):
    def get_edges_from_node(node_id):
        connected = []
        for edge in edges:
            if edge[0] == node_id:
                connected.append(edge)
        assert len(connected) == 2, f"node {node_id} does not have 2 outgoing edges."

        return (
            connected[0][1],
            connected[1][1],
            connected[0][2],
            connected[1][2],
        )

    root = next(filter(lambda n: node_types[n] == " root node ", nodes.keys()))

    edge0 = next(filter(lambda edge: edge[0] == root, edges))
    bif0 = edge0[1]
    edge0_radius = edge0[2]
    end1, end2, r1, r2 = get_edges_from_node(bif0)

    result, wps = create_root_bifurcation(
        nodes[root],
        nodes[bif0],
        edge0_radius,
        nodes[end1],
        nodes[end2],
        r1,
        r2,
    )
    nodes_wps = {end1: wps[0], end2: wps[1]}

    queue = [(end1, r1), (end2, r2)]
    while len(queue) > 0:
        current_node, current_radius = queue.pop(0)
        if node_types[current_node] == " terminal node ":
            continue

        end1, end2, r1, r2 = get_edges_from_node(current_node)
        bif_result, wps = create_bifurcation(
            nodes[current_node],
            nodes_wps[current_node],
            nodes[end1],
            nodes[end2],
            current_radius,
            r1,
            r2,
        )
        result = result.union(bif_result)
        nodes_wps[end1] = wps[0]
        nodes_wps[end2] = wps[1]
        queue.append((end1, r1))
        queue.append((end2, r2))

    return result


def tag_and_mesh_with_gmsh(
    brep_path: str, nodes: dict, node_types: dict, tol: float = VOXEL_WIDTH * 0.6
):
    """
    Import the brep into gmsh, find surfaces nearest the requested node points (inlet + terminals),
    create physical groups for inlet, outlets and walls, then mesh and write out_msh.
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.merge(brep_path)

    # gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 2e-3)
    # gmsh.option.setNumber("Mesh.Smoothing", 3)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 1)
    gmsh.option.setNumber("Mesh.MinimumElementsPerTwoPi", 11)

    # get all 2D entities (surfaces)
    surfaces = gmsh.model.getEntities(2)  # list of (2, tag)

    # precompute surface centers (bounding-box centers)
    surf_centers = {}
    for dim, s in surfaces:
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(dim, s)
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        cz = 0.5 * (zmin + zmax)
        surf_centers[s] = (cx, cy, cz)

    assigned = {}  # node_id -> surface tag

    # helper to find nearest surface to a point
    def find_nearest_surface(pt):
        best_s = None
        best_d2 = float("inf")
        for s, c in surf_centers.items():
            d2 = (c[0] - pt[0]) ** 2 + (c[1] - pt[1]) ** 2 + (c[2] - pt[2]) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_s = s
        return best_s, best_d2**0.5

    # inlet: root node
    root_id = next(filter(lambda n: node_types[n] == " root node ", nodes.keys()))
    inlet_pt = tuple(nodes[root_id])
    s_inlet, d_in = find_nearest_surface(inlet_pt)
    if s_inlet is None or d_in > tol:
        print(f"[WARN] inlet surface not found within tol ({d_in:.4g} > {tol}).")
    else:
        assigned[root_id] = s_inlet

    # outlets: terminal nodes
    for nid, ntype in node_types.items():
        if ntype == " terminal node ":
            pt = tuple(nodes[nid])
            s_out, d_out = find_nearest_surface(pt)
            if s_out is None or d_out > tol:
                print(f"[WARN] outlet {nid} not found within tol ({d_out:.4g} > {tol}).")
            else:
                assigned[nid] = s_out

    # create physical groups
    used_surfaces = set(assigned.values())
    # inlet
    if root_id in assigned:
        phys_in = gmsh.model.addPhysicalGroup(2, [assigned[root_id]], inlet_tag)
        gmsh.model.setPhysicalName(2, inlet_tag, "inlet")
    # outlets (name by node id)
    outlet_surfaces = [s for nid, s in assigned.items() if nid != root_id]
    if outlet_surfaces:
        phys_outlets = gmsh.model.addPhysicalGroup(2, outlet_surfaces, outlet_tag)
        gmsh.model.setPhysicalName(2, outlet_tag, "outlets")

    # walls = all surfaces not used
    all_surface_tags = [s for (_, s) in surfaces]
    wall_surfaces = [s for s in all_surface_tags if s not in used_surfaces]
    if wall_surfaces:
        phys_walls = gmsh.model.addPhysicalGroup(2, wall_surfaces, wall_tag)
        gmsh.model.setPhysicalName(2, wall_tag, "walls")

    # generate 3D mesh (volumes must exist in the imported brep)
    try:
        gmsh.model.mesh.generate(3)
    except Exception as e:
        print(f"[ERROR] gmsh mesh generation failed: {e}")
    gmsh.write(OUT_MSH)
    gmsh.finalize()
    print(f"[OK] Mesh with physical groups written to {OUT_MSH}")


if __name__ == "__main__":
    nodes, node_types, edges = parse_gxl(GXL_FILE)
    geom = build_mesh(nodes, node_types, edges)
    cq.exporters.export(geom, "src/geom/vessels.brep")
    tag_and_mesh_with_gmsh("src/geom/vessels.brep", nodes, node_types)
