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
        factor_p = self.params.get("vessel_loss_factor", 0.0)
        if factor_p > 0:
            self._prune_tree(factor_p, levels, max_level)

        is_hyper = self.params.get("hyperemia", False)
        hyper_f = self.params.get("hyperemia_dilation_factor", 1.0)
        thick_severity = self.params.get("wall_thickening_severity", 0.0)
        thick_threshold = self.params.get("thickening_level_threshold", 0)

        for edge in self.edges:
            if is_hyper and "root node" not in self.node_types[edge["from"]]:
                edge["radius"] *= hyper_f
            if thick_severity > 0 and levels.get(edge["from"], 0) >= thick_threshold:
                edge["radius"] *= 1.0 - thick_severity

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
        """Remove edges until count ≤ original * (1 - factor).

        The geometry mesher requires every internal node to have exactly 2
        outgoing edges.  To honour this invariant, pruning always removes
        **both** children of a bifurcation (turning the parent into a terminal
        leaf).  Bifurcations are pruned deepest-first so that the most distal
        vessels are lost first, which is biologically plausible for vessel loss.

        Parameters
        ----------
        factor : float ∈ (0, 1]
            Fraction of edges to remove.  The algorithm stops as soon as the
            remaining edge count is at or below ``original_edges * (1 - factor)``.
        """
        original_count = len(self.edges)
        target_count = original_count * (1.0 - factor)

        # Build adjacency (parent -> children).
        adj = {}
        for e in self.edges:
            adj.setdefault(e["from"], []).append(e["to"])

        removed = set()

        def count_subtree(nid):
            """Count nid + all descendants not yet removed."""
            if nid in removed:
                return 0
            total = 1
            for c in adj.get(nid, []):
                total += count_subtree(c)
            return total

        def collect_subtree(nid):
            """Recursively add nid and all its descendants to removed."""
            if nid in removed:
                return
            removed.add(nid)
            for c in adj.get(nid, []):
                collect_subtree(c)

        current_count = original_count

        # Identify "leaf bifurcations": nodes whose both children are leaves
        # (i.e. have no children themselves).  These are the safest to prune
        # because removing both children only affects the parent.
        # We iterate deepest-first until we hit the target edge count.
        while current_count > target_count:
            # Rebuild leaf-bifurcation candidates each iteration (some parents
            # may become new leaf-bifurcations after their children were pruned).
            candidates = []
            for p, children in adj.items():
                if p in removed:
                    continue
                live = [c for c in children if c not in removed]
                if len(live) != 2:
                    continue
                # Both live children must themselves be leaves (0 live children)
                both_leaves = all(
                    (
                        all(gc in removed for gc in adj.get(c, [])) and True
                        if adj.get(c)
                        else True
                    )
                    for c in live
                )
                if both_leaves:
                    # Use max level of the two children for ordering
                    lvl = max(levels.get(c, 0) for c in live)
                    candidates.append((lvl, p, live))

            if not candidates:
                # No more leaf-bifurcations available; stop.
                break

            # Sort deepest first, randomise within same level.
            random.shuffle(candidates)
            candidates.sort(key=lambda x: x[0], reverse=True)

            # Prune one bifurcation at a time, re-check target after each.
            lvl, p, live = candidates[0]
            # Count how many edges will be removed (edges whose from or to
            # is in the subtrees of both children).
            edges_lost = 0
            for c in live:
                edges_lost += count_subtree(c)
            for c in live:
                collect_subtree(c)
            current_count -= edges_lost

        # Rebuild edges, nodes, node_types from survivors.
        self.edges = [
            e for e in self.edges if e["from"] not in removed and e["to"] not in removed
        ]
        active = {e["from"] for e in self.edges} | {e["to"] for e in self.edges}
        # Always keep root nodes, even if they have no edges after pruning
        # (build_mesh requires the root/inlet to always be present).
        for nid, t in self.node_types.items():
            if "root node" in t:
                active.add(nid)
        self.nodes = {nid: pos for nid, pos in self.nodes.items() if nid in active}
        self.node_types = {
            nid: t for nid, t in self.node_types.items() if nid in active
        }

        # Reclassify nodes that lost all children as terminal nodes.
        # build_mesh expects every non-terminal, non-root node to have
        # exactly 2 outgoing edges; a former bifurcation with 0 children
        # must become a terminal (outlet) so the mesher skips it.
        outgoing = {}
        for e in self.edges:
            outgoing.setdefault(e["from"], []).append(e["to"])
        for nid in list(self.node_types):
            if "root node" in self.node_types[nid]:
                continue
            if nid not in outgoing or len(outgoing[nid]) == 0:
                self.node_types[nid] = " terminal node "

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
