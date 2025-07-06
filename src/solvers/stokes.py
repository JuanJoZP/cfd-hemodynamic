# solve the Stokes problem to use as initial guess for navier-stokes newton iterations

from typing import Callable
from petsc4py import PETSc
import numpy as np

from basix.ufl import element
from dolfinx.mesh import Mesh
from dolfinx.fem import (
    DirichletBC,
    Constant,
    Function,
    functionspace,
    form,
    bcs_by_block,
    extract_function_spaces,
)
from ufl import (
    MixedFunctionSpace,
    TrialFunctions,
    inner,
    grad,
    div,
    TestFunctions,
    TrialFunction,
    TestFunction,
    dx,
)
from dolfinx.fem.petsc import (
    assemble_matrix_nest,
    assemble_vector_nest,
    apply_lifting_nest,
    set_bc_nest,
)


class StokesSolver:
    MAX_ITER = 20

    def __init__(
        self,
        mesh: Mesh,
        rho: Constant,
        mu: Constant,
        f: Constant,
    ):
        self.mesh = mesh
        self.rho = rho
        self.mu = mu
        self.f = f

        element_velocity = element(
            "Lagrange",
            mesh.topology.cell_name(),
            3,
            shape=(mesh.geometry.dim,),
        )
        element_pressure = element("Lagrange", mesh.topology.cell_name(), 2)

        self.V = functionspace(mesh, element_velocity)
        self.Q = functionspace(mesh, element_pressure)
        self.VQ = MixedFunctionSpace(self.V, self.Q)

        u, p = TrialFunctions(self.VQ)
        v, q = TestFunctions(self.VQ)

        a_uu = inner(grad(u), grad(v)) * dx
        a_up = inner(p, div(v)) * dx
        a_pu = inner(div(u), q) * dx
        a_pp = Constant(self.mesh, 0.0) * inner(p, q) * dx
        self.A_form = form([[a_uu, a_up], [a_pu, a_pp]])

        self.L_form = form(
            [inner(self.f, v) * dx, inner(Constant(self.mesh, 0.0), q) * dx]
        )

    def solve(self, bcs: list[DirichletBC] = None):
        # block preconditioner
        p, q = TrialFunction(self.Q), TestFunction(self.Q)
        a_p11 = form(inner(p, q) * dx)
        a_p = form([[self.A_form[0][0], None], [None, a_p11]])
        P = assemble_matrix_nest(a_p, bcs)
        P.assemble()

        # assemble the system matrix
        A = assemble_matrix_nest(self.A_form, bcs=bcs)
        A.assemble()

        b = assemble_vector_nest(self.L_form)
        apply_lifting_nest(b, self.A_form, bcs=bcs)
        for b_sub in b.getNestSubVecs():
            b_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        spaces = extract_function_spaces(self.L_form)
        bcs0 = bcs_by_block(spaces, bcs)
        set_bc_nest(b, bcs0)

        # create the linear solver
        ksp = PETSc.KSP().create(self.mesh.comm)
        ksp.setOperators(A, P)
        ksp.setType("minres")
        ksp.setTolerances(rtol=1e-9)
        ksp.getPC().setType("fieldsplit")
        ksp.getPC().setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)

        nested_IS = P.getNestISs()
        ksp.getPC().setFieldSplitIS(("u", nested_IS[0][0]), ("p", nested_IS[0][1]))

        ksp_u, ksp_p = ksp.getPC().getFieldSplitSubKSP()
        ksp_u.setType("preonly")
        ksp_u.getPC().setType("gamg")
        ksp_p.setType("preonly")
        ksp_p.getPC().setType("jacobi")

        ksp.setFromOptions()

        # solve
        self.u_sol, self.p_sol = Function(self.V), Function(self.Q)
        w = PETSc.Vec().createNest([self.u_sol.x.petsc_vec, self.p_sol.x.petsc_vec])
        ksp.solve(b, w)
        assert ksp.getConvergedReason() > 0
        self.u_sol.x.scatter_forward()

        self.p_sol.x.petsc_vec.scale(-1.0)
        self.p_sol.x.scatter_forward()
