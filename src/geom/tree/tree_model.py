import random

from .graph_to_mesh import build_mesh, parse_gxl, tag_and_mesh_with_gmsh


class VascularTree:
    def __init__(self, nodes, edges, node_types, params):
        self.nodes = nodes
        self.edges = edges  # List of dicts
        self.node_types = node_types
        self.params = params

    @classmethod
    def from_xml(cls, xml_path, params):
        voxel_width = params.get("voxel_width", 0.04)
        nodes, node_types, edges_tuples = parse_gxl(xml_path, voxel_width=voxel_width)

        # Convertir edges_tuples (frm, to, radius) a lista de dicts
        # para compatibilidad con apply_modifications
        edges = []
        for frm, to, radius in edges_tuples:
            edges.append({"from": frm, "to": to, "radius": radius})

        return cls(nodes, edges, node_types, params)

    def apply_modifications(self):
        levels = self._calculate_levels()
        max_level = max(levels.values()) if levels else 1
        factor_p = self.params.get("factor_perdida_vasos", 0.0)
        if factor_p > 0:
            self._prune_tree(factor_p, levels, max_level)

        is_hyper = self.params.get("hiperemia", False)
        hyper_f = self.params.get("factor_dilatacion_hiperemia", 1.0)
        thick_f = self.params.get("factor_engrosamiento_lumen", 1.0)
        thick_threshold = self.params.get("thickening_level_threshold", 0)

        for edge in self.edges:
            if is_hyper and "root node" not in self.node_types[edge["from"]]:
                edge["radius"] *= hyper_f
            if levels.get(edge["from"], 0) >= thick_threshold:
                edge["radius"] *= thick_f

    def _calculate_levels(self):
        levels = {}
        root_id = next(
            (nid for nid, nt in self.node_types.items() if "root node" in nt), None
        )
        if not root_id:
            return {}
        adj = {}
        for e in self.edges:
            adj.setdefault(e["from"], []).append(e["to"])
        queue = [(root_id, 0)]
        while queue:
            curr, lvl = queue.pop(0)
            levels[curr] = lvl
            for child in adj.get(curr, []):
                queue.append((child, lvl + 1))
        return levels

    def _prune_tree(self, factor, levels, max_level):
        to_remove = set()
        for nid, lvl in levels.items():
            if lvl > 1 and random.random() < factor * (lvl / max_level):
                to_remove.add(nid)

        adj = {}
        for e in self.edges:
            adj.setdefault(e["from"], []).append(e["to"])

        final_remove = set()

        def collect(nid):
            if nid not in final_remove:
                final_remove.add(nid)
                for c in adj.get(nid, []):
                    collect(c)

        for r in to_remove:
            collect(r)

        self.edges = [
            e
            for e in self.edges
            if e["from"] not in final_remove and e["to"] not in final_remove
        ]
        active = {e["from"] for e in self.edges} | {e["to"] for e in self.edges}
        self.nodes = {nid: pos for nid, pos in self.nodes.items() if nid in active}
        self.node_types = {
            nid: t for nid, t in self.node_types.items() if nid in active
        }

    def build_solid(self):
        # Convertir dicts a tuplas para build_mesh
        edges_tuples = [(e["from"], e["to"], e["radius"]) for e in self.edges]
        return build_mesh(self.nodes, self.node_types, edges_tuples)

    def mesh_and_tag(self, brep_path, out_msh):
        """
        Exporta a BREP temporal, meshea y tagea con Gmsh.
        Las etiquetas son: 1=inlet, 2=outlets, 3=walls, 4=fluid.
        """

        voxel_width = self.params.get("voxel_width", 0.04)
        tol = voxel_width * 0.6
        tag_and_mesh_with_gmsh(brep_path, self.nodes, self.node_types, tol, out_msh)
