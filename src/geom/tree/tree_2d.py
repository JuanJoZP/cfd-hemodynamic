"""
Simple 2D vascular tree generator inspired by VascuSynth.

Keeps the hemodynamic constraints (Murray's law, flow conservation,
Poiseuille resistance) but drops the oxygen-demand map and iterative
optimisation.  Bifurcations are placed at the end of each branch with
fixed angular spread.

Usage
-----
    tree = VascularTree2D(
        r_root=1.2,          # mm
        n_generations=3,
        gamma=3.0,
        bifurcation_angle=35.0,
        length_ratio=8.0,
        asymmetry=0.5,
    )
    tree.generate(origin=(138.0, 1.57), direction=0.0)
    # tree.nodes   -> dict[int, np.array([x, y])]
    # tree.edges   -> list[dict] with 'from', 'to', 'radius', 'r_parent'
    # tree.terminals -> list[int]  (leaf node ids)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class VascularTree2D:
    """Binary vascular tree built generation-by-generation.

    Parameters
    ----------
    r_root : float
        Radius of the root branch (mm).
    n_generations : int
        Number of bifurcation levels (e.g. 3 -> 8 terminal branches).
    gamma : float
        Murray's law exponent (typically 3.0).
    bifurcation_angle : float
        Half-angle (degrees) between child branches at each bifurcation.
    length_ratio : float
        Branch length = length_ratio * branch_radius.
    asymmetry : float
        Flow split ratio for the *left* child: Q_left = asymmetry * Q_parent.
        0.5 = symmetric.  Must be in (0, 1).
    """

    r_root: float = 1.2
    n_generations: int = 3
    gamma: float = 3.0
    bifurcation_angle: float = 35.0
    length_ratio: float = 8.0
    asymmetry: float = 0.5

    # Populated after generate()
    nodes: dict[int, np.ndarray] = field(default_factory=dict, repr=False)
    edges: list[dict] = field(default_factory=list, repr=False)
    terminals: list[int] = field(default_factory=list, repr=False)

    _next_id: int = field(default=0, repr=False)

    def _new_id(self) -> int:
        nid = self._next_id
        self._next_id += 1
        return nid

    def generate(self, origin: tuple[float, float], direction: float = 0.0):
        """Build the tree starting at *origin* heading in *direction* (degrees).

        Parameters
        ----------
        origin : (x, y)
            Position where the root branch starts (outlet of the stenosis).
        direction : float
            Angle in degrees of the root branch (0 = +x).
        """
        self.nodes.clear()
        self.edges.clear()
        self.terminals.clear()
        self._next_id = 0

        ox, oy = origin
        root_start = self._new_id()
        self.nodes[root_start] = np.array([ox, oy])

        # Place root branch endpoint
        root_end = self._new_id()
        root_len = self.length_ratio * self.r_root
        theta = np.radians(direction)
        self.nodes[root_end] = np.array(
            [
                ox + root_len * np.cos(theta),
                oy + root_len * np.sin(theta),
            ]
        )
        self.edges.append(
            {
                "from": root_start,
                "to": root_end,
                "radius": self.r_root,
                "r_parent": self.r_root,  # root has no parent; same radius
            }
        )

        # Recurse
        self._bifurcate(root_end, self.r_root, direction, generation=1)

    def _child_radii(self, r_parent: float) -> tuple[float, float]:
        """Compute child radii from Murray's law + asymmetry.

        Murray: r_p^gamma = r_l^gamma + r_r^gamma
        Flow split: Q_l / Q_r = asymmetry / (1 - asymmetry)
        Poiseuille: Q ~ r^4 / L, with L ~ r  =>  Q ~ r^3
        So r_l^3 / r_r^3 = asymmetry / (1 - asymmetry)
        => r_l / r_r = (asymmetry / (1-asymmetry))^(1/3)

        Then from Murray:
            r_l = r_p * (1 + (r_r/r_l)^gamma)^(-1/gamma)
        """
        a = self.asymmetry
        g = self.gamma

        # ratio = r_left / r_right
        ratio = (a / (1.0 - a)) ** (1.0 / 3.0)

        # Murray: r_p^g = r_l^g + r_r^g = r_l^g * (1 + ratio^(-g))
        r_left = r_parent * (1.0 + ratio ** (-g)) ** (-1.0 / g)
        r_right = r_left / ratio

        return r_left, r_right

    def _bifurcate(
        self, parent_node: int, r_parent: float, parent_angle: float, generation: int
    ):
        if generation > self.n_generations:
            self.terminals.append(parent_node)
            return

        r_left, r_right = self._child_radii(r_parent)
        half_angle = self.bifurcation_angle

        # Scale angle by radius asymmetry: thinner branch deflects more
        # (keeps total momentum-ish balance)
        angle_left = parent_angle + half_angle * (r_right / r_parent)
        angle_right = parent_angle - half_angle * (r_left / r_parent)

        for r_child, angle in [(r_left, angle_left), (r_right, angle_right)]:
            branch_len = self.length_ratio * r_child
            theta = np.radians(angle)
            pos_parent = self.nodes[parent_node]

            child_id = self._new_id()
            self.nodes[child_id] = pos_parent + branch_len * np.array(
                [
                    np.cos(theta),
                    np.sin(theta),
                ]
            )
            self.edges.append(
                {
                    "from": parent_node,
                    "to": child_id,
                    "radius": r_child,
                    "r_parent": r_parent,
                }
            )
            self._bifurcate(child_id, r_child, angle, generation + 1)

    @property
    def bifurcation_points(self) -> list[dict]:
        """Return bifurcation info for fillet placement.

        Each entry: {"node": id, "pos": np.array, "r_min": float}
        where r_min is the smallest child radius at that bifurcation.
        """
        # Nodes that are "to" of one edge and "from" of at least two edges
        children_of = {}
        for e in self.edges:
            children_of.setdefault(e["from"], []).append(e)

        result = []
        for nid, child_edges in children_of.items():
            if len(child_edges) < 2:
                continue
            r_min = min(e["radius"] for e in child_edges)
            result.append(
                {
                    "node": nid,
                    "pos": self.nodes[nid],
                    "r_min": r_min,
                }
            )
        return result
