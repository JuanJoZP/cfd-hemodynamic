import os
from lxml import etree  # type: ignore
import numpy as np
import pygmsh
import meshio


GXL_FILE = "tree_structure.xml"
VOXEL_WIDTH = 0.04
OUT_MSH = "vessels.msh"
OUT_XDMF = "vessels.xdmf"
ELEMENT_SIZE_FACTOR = 0.5


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


def build_mesh(nodes, node_types, edges):
    with pygmsh.occ.geometry.Geometry() as geom:
        cyls = []
        inlet_caps = []
        outlet_caps = []

        for frm, to, r in edges:

            p0, p1 = nodes[frm], nodes[to]
            vec = p1 - p0
            if to == "n25":
                print("hello")
                p0 = p0 + 0.013 * vec

            # cilindro
            # lc = max(r * ELEMENT_SIZE_FACTOR, 0.01)
            cyl = geom.add_cylinder(p0.tolist(), vec.tolist(), r)  # ,mesh_size=lc)
            cyls.append(cyl)

            # # inlet
            # if "root node" in node_types[frm]:
            #     nvec = (vec / L).tolist()
            #     cap = geom.add_disk(p0.tolist(), r, nvec) #, mesh_size=lc)
            #     inlet_caps.append(cap)

            # # outlet
            # if "terminal node" in node_types[to]:
            #     nvec = (vec / L).tolist()
            #     cap = geom.add_disk(p1.tolist(), r, nvec) #, mesh_size=lc)
            #     outlet_caps.append(cap)

        # Unión booleana
        vessels = geom.boolean_union(cyls)

        # # Physical groups
        # geom.add_physical(vessels, "Fluid")
        # if inlet_caps:
        #     geom.add_physical(inlet_caps, "Inlet")
        # if outlet_caps:
        #     geom.add_physical(outlet_caps, "Outlet")
        # # todas las superficies externas = paredes
        # geom.add_physical(vessels, "Walls")

        geom.save_geometry("hola.brep")
        mesh = geom.generate_mesh(dim=3)
        mesh.write(OUT_MSH)
        print(f"[OK] Malla guardada en {OUT_MSH}")

    return OUT_MSH


def convert_to_xdmf(msh_file):
    m = meshio.read(msh_file)
    cells = {c.type: c.data for c in m.cells}
    if "tetra" in cells:
        tet_mesh = meshio.Mesh(
            points=m.points,
            cells={"tetra": cells["tetra"]},
            cell_data={"gmsh:physical": [m.cell_data_dict["gmsh:physical"]["tetra"]]},
        )
        meshio.write(OUT_XDMF, tet_mesh)
        print(f"[OK] XDMF guardado en {OUT_XDMF}")
    else:
        print("[WARN] No tetrahedra en el msh, revisa la generación.")


if __name__ == "__main__":
    # nodes, node_types, edges = parse_gxl(GXL_FILE)
    # import json

    # with open("nodes.json", mode="w") as file:
    #     file.write(str(nodes))
    # with open("nodes_types.json", mode="w") as file:
    #     file.write(json.dumps(node_types))
    # with open("edges.json", mode="w") as file:
    #     file.write(json.dumps(edges))
    # print(f"Leídos {len(nodes)} nodos, {len(edges)} aristas")
    # msh = build_mesh(nodes, node_types, edges)
    # raise
    # convert_to_xdmf(msh)
    nodes = {
        "n0": np.array([0.0, 2.0, 2.0]),
        "n24": np.array([0.0995216, 1.868072, 2.019692]),
        "n25": np.array([0.08, 2.2, 1.88]),
    }

    nodes_types = {
        "n0": " root node ",
        "n24": " bifurication ",
        "n25": " terminal node ",
    }

    edges = [
        ["n0", "n24", 0.003918604],
        ["n24", "n25", 0.000922768],
    ]

    msh = build_mesh(nodes, nodes_types, edges)
