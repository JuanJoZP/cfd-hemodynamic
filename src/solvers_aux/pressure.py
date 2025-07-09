# solve stationary momentum equation for pressure given a velocity

from typing import Callable
from dolfinx.fem import Function, form
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc
from petsc4py import PETSc
import numpy as np
from dolfinx.mesh import Mesh, locate_entities_boundary
from ufl import (
    inner,
    dot,
    ds,
    FacetNormal,
    TrialFunction,
    TestFunction,
    dx,
    nabla_grad,
    Identity,
)
from src.boundaryCondition import BoundaryCondition
from src.solverBase import SolverBase


class PressureSolver(SolverBase):
    def __init__(
        self,
        mesh: Mesh,
        rho: float,
        mu: float,
        f: list[float],
        velocity: Callable[[np.ndarray], np.ndarray],
    ):
        super().__init__(mesh, 0.0, rho, mu, f)

        super().initVelocitySpace("Lagrange", mesh.topology.cell_name(), 3, shape=(2,))
        super().initPressureSpace("Lagrange", mesh.topology.cell_name(), 2)

        u = self.u_sol
        u.interpolate(velocity)
        p = TrialFunction(self.Q)
        v = TestFunction(self.V)

        # weak form
        n = FacetNormal(mesh)

        a = inner(p, dot(n, v)) * ds
        a -= inner(p * Identity(len(u)), self.epsilon(v)) * dx

        L = dot(self.f, v) * dx
        L -= self.rho * dot(dot(u, nabla_grad(u)), v) * dx
        L -= 2 * self.mu * inner(self.epsilon(u), self.epsilon(v)) * dx
        L += dot(self.mu * nabla_grad(u) * n, v) * ds

        self.a = form(a)
        self.L = form(L)

    def setup(self, bcp: list[BoundaryCondition] = None):
        bcp_d = [bc.getBC(self.Q) for bc in bcp]

        A = assemble_matrix(self.a, bcs=bcp_d)
        A.assemble()

        b = assemble_vector(self.L)
        apply_lifting(b, [self.a], [bcp_d])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b, bcp_d)

        with self.p_sol.x.petsc_vec.duplicate().localForm() as vec_const:
            vec_const.set(1.0)
            norm = vec_const.norm(PETSc.NormType.NORM_2)
            vec_const.scale(1.0 / norm)  # normalize

            self.nullsp = PETSc.NullSpace().create(
                vectors=[vec_const], comm=self.mesh.comm
            )

        A.setNullSpace(self.nullsp)

        solver = PETSc.KSP().create(self.mesh.comm)
        solver.setOperators(A)
        solver.setType("lsqr")
        solver.getPC().setType("none")
        solver.setTolerances(rtol=1e-16, atol=1e-8, max_it=1000)

        self.solver = solver
        self.b = b

    def solveStep(self):
        solver = self.solver
        solver.solve(self.b, self.p_sol.x.petsc_vec)

        assert solver.is_converged, (
            f"Pressure solver did not converge.\n"
            f"Reason: {solver.reason}.\n"
            f"Iterations: {solver.getIterationNumber()}\n"
            f"Residual norm: {solver.getResidualNorm():.1e}."
        )
