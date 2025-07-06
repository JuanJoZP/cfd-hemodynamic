from dolfinx.fem import (
    Function,
    FunctionSpace,
    dirichletbc,
    DirichletBC,
    locate_dofs_geometrical,
    locate_dofs_topological,
)
from typing import Callable
from numpy import ndarray, dtype, int32


class BoundaryCondition:
    def __init__(self, f: Function):
        self._topological = False
        self._geometrical = False
        self.f = f

    def initTopological(
        self, entity_dim: int, entities: ndarray[None, dtype[int32]]
    ) -> None:
        assert not (self._topological or self._geometrical)
        self.entity_dim = entity_dim
        self.entities = entities
        self._topological = True

    def initGeometrical(self, marker: Callable) -> None:
        assert not (self._topological or self._geometrical)
        self.marker = marker
        self._geometrical = True

    def _getDofs(self, V: FunctionSpace) -> ndarray:
        assert self._topological or self._geometrical
        if self._topological:
            return locate_dofs_topological(V, self.entity_dim, self.entities)

        if self._geometrical:
            return locate_dofs_geometrical(V, self.marker)

    def getBC(self, V: FunctionSpace) -> DirichletBC:
        dofs = self._getDofs(V)
        f_V = Function(V)
        f_V.interpolate(self.f)
        return dirichletbc(f_V, dofs)
