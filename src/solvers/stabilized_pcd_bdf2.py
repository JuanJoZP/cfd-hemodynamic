from typing import Callable

import dolfinx.fem as fem
import numpy as np
from dolfinx.fem import Constant, DirichletBC, Function, form, functionspace
from dolfinx.fem.petsc import (
    assemble_matrix_block,
    assemble_vector_block,
    create_matrix_block,
    create_vector_block,
)
from dolfinx.mesh import Mesh
from fenicsx_pctools.mat import create_splittable_matrix_block
from petsc4py import PETSc
from petsc4py import typing as petsc_typing
from ufl import (
    FacetNormal,
    Measure,
    MixedFunctionSpace,
    TestFunctions,
    TrialFunctions,
    conditional,
    derivative,
    div,
    dot,
    ds,
    dx,
    extract_blocks,
    ge,
    grad,
    inner,
    le,
    nabla_grad,
    sqrt,
)

from src.boundaryCondition import BoundaryCondition
from src.solverBase import SolverBase


class Solver(SolverBase):
    MAX_ITER = 20

    def __init__(
        self,
        mesh: Mesh,
        dt: float,
        rho: float,
        mu: float,
        f: list,
        initial_velocity: Callable[[np.ndarray], np.ndarray] = None,
    ):
        super().__init__(mesh, dt, rho, mu, f)

        super().initVelocitySpace(
            "Lagrange", mesh.topology.cell_name(), 1, shape=(mesh.geometry.dim,)
        )
        super().initPressureSpace("Lagrange", mesh.topology.cell_name(), 1)

        self.VQ = MixedFunctionSpace(self.V, self.Q)

        v, q = TestFunctions(self.VQ)

        if initial_velocity:
            self.u_prev.interpolate(initial_velocity)

        # BDF2 state: u at time n-1
        self.u_prev2 = Function(self.V)

        # BDF coefficients as updateable Constants (start with BDF1)
        self.bdf_a0 = Constant(mesh, PETSc.ScalarType(1.0))
        self.bdf_a1 = Constant(mesh, PETSc.ScalarType(-1.0))
        self.bdf_a2 = Constant(mesh, PETSc.ScalarType(0.0))

        self.step_count = 0

        # weak form
        u_sol = self.u_sol
        p_sol = self.p_sol
        u_prev = self.u_prev
        u_prev2 = self.u_prev2
        n = FacetNormal(self.mesh)

        F = self.rho * inner(
            v,
            (self.bdf_a0 * u_sol + self.bdf_a1 * u_prev + self.bdf_a2 * u_prev2)
            / self.dt,
        ) * dx
        F += self.rho * dot(v, dot(u_sol, nabla_grad(u_sol))) * dx
        F -= inner(v, self.rho * self.f) * dx
        F += inner(self.epsilon(v), self.sigma(u_sol, p_sol, self.mu)) * dx
        F += dot(p_sol * n, v) * ds - dot(self.mu * nabla_grad(u_sol) * n, v) * ds
        F += inner(q, div(u_sol)) * dx

        # stabilization terms
        V_dg0 = functionspace(mesh, ("DG", 0))
        h = Function(V_dg0)
        h.x.array[:] = mesh.h(
            mesh.topology.dim,
            np.arange(h.x.index_map.size_local + h.x.index_map.num_ghosts),
        )

        vnorm = sqrt(
            inner(u_prev, u_prev)
        )  # u_prev instead of u_sol to avoid nonlinearity after derivation

        R = self.rho * (
            (self.bdf_a0 * u_sol + self.bdf_a1 * u_prev + self.bdf_a2 * u_prev2)
            / self.dt
            + dot(u_sol, nabla_grad(u_sol))
        )
        R -= div(self.sigma(u_sol, p_sol, self.mu))
        R -= self.rho * self.f

        # SUPG
        eps = Constant(self.mesh, np.finfo(PETSc.ScalarType()).resolution)
        tau_supg1 = h / conditional(
            ge((2.0 * vnorm), eps), (2.0 * vnorm), eps
        )  # avoid division by zero
        tau_supg2 = self.dt / 2.0
        tau_supg3 = (h * h) / (4.0 * (self.mu / self.rho))
        tau_supg = (
            1 / (tau_supg1**2) + 1 / (tau_supg2**2) + 1 / (tau_supg3**2)
        ) ** (-1 / 2)
        F_supg = inner(tau_supg * R, dot(u_sol, nabla_grad(v))) * dx

        # PSPG
        tau_pspg = tau_supg
        F_pspg = (1 / self.rho) * inner(tau_pspg * R, grad(q)) * dx

        # LSIC
        Re = (vnorm * h) / (2.0 * (self.mu / self.rho))
        z = conditional(le(Re, 3), Re / 3, 1.0)
        tau_lsic = (vnorm * h * z) / 2.0
        F_lsic = tau_lsic * inner(div(u_sol), self.rho * div(v)) * dx

        F += F_supg + F_lsic
        F += F_pspg
        self.F = F

    def updateSolution(self, x: petsc_typing.Vec) -> None:
        "Updates the solution functions u_sol and p_sol with the values in x."
        start_u, end_u = self.u_prev.x.petsc_vec.getOwnershipRange()
        start_p, end_p = self.p_prev.x.petsc_vec.getOwnershipRange()
        u_size_local = self.u_prev.x.petsc_vec.getLocalSize()

        self.u_sol.x.petsc_vec.setValues(
            range(start_u, end_u), x.array_r[:u_size_local]
        )
        self.p_sol.x.petsc_vec.setValues(
            range(start_p, end_p), x.array_r[u_size_local:]
        )
        self.u_sol.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )
        self.p_sol.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

    def assembleJacobian(
        self,
        snes: petsc_typing.SNES,
        x: petsc_typing.Vec,
        J: petsc_typing.Mat,
        P: petsc_typing.Mat,
        bcs: list[DirichletBC] = [],
    ) -> None:
        "Assembles the Jacobian matrix evaluated at u_sol and p_sol."
        J_mat = J.getPythonContext().Mat if J.getType() == "python" else J
        J.zeroEntries()
        fem.petsc.assemble_matrix_block(J_mat, self.J_form, bcs, diagonal=1.0)
        J.assemble()
        if P is not None and P != J:
            P_mat = P.getPythonContext().Mat if P.getType() == "python" else P
            P.zeroEntries()
            fem.petsc.assemble_matrix_block(P_mat, self.J_form, bcs, diagonal=1.0)
            P.assemble()

    def assembleResidual(
        self,
        snes: PETSc.SNES,
        x: PETSc.Vec,
        F_vec: PETSc.Vec,
        bcs: list[DirichletBC] = [],
    ) -> None:
        "Assembles the residual vector evaluated at u_sol and p_sol, applies lifting and set_bcs so that the constrained dofs are = x_n - g."
        with F_vec.localForm() as F_local:
            F_local.set(0.0)

        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        self.updateSolution(x)
        [bc.update() for bc in bcs]

        assemble_vector_block(
            F_vec, self.F_form, self.J_form, bcs=bcs, x0=x, alpha=-1.0
        )
        F_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    def setup(
        self,
        bcu: list[BoundaryCondition],
        bcp: list[BoundaryCondition],
        facet_tags=None,
        tags=None,
    ) -> None:
        # create linealizated problem
        du, dp = TrialFunctions(self.VQ)

        J = derivative(self.F, (self.u_sol, self.p_sol), (du, dp))
        self.F_form = form(extract_blocks(self.F))
        self.J_form = form(extract_blocks(J))

        self.A = create_matrix_block(self.J_form)
        self.b = create_vector_block(self.F_form)
        self.x_n = self.b.duplicate()  # solution to the nth newton iteration
        self.offset = (
            self.V.dofmap.index_map.size_local + self.V.dofmap.index_map.num_ghosts
        ) * self.V.dofmap.index_map_bs  # after this index values of x correspond to pressure, before to velocity

        self.bcu_d = [bc.getBC(self.V) for bc in bcu]
        self.bcp_d = [bc.getBC(self.Q) for bc in bcp]
        bcs = [*self.bcu_d, *self.bcp_d]

        # PCD secondary boundary conditions
        inlet_dofs_p = fem.locate_dofs_topological(
            self.Q, self.mesh.topology.dim - 1, facet_tags.find(tags["inlet"])
        )
        outlet_dofs_p = fem.locate_dofs_topological(
            self.Q, self.mesh.topology.dim - 1, facet_tags.find(tags["outlet"])
        )

        pcd_type = "PCDPC_vY"
        bcs_pcd = {
            "PCDPC_vX": [fem.dirichletbc(fem.Function(self.Q), inlet_dofs_p)],
            "PCDPC_vY": [fem.dirichletbc(fem.Function(self.Q), outlet_dofs_p)],
        }[pcd_type]

        ds_in = Measure(
            "ds",
            domain=self.mesh,
            subdomain_data=facet_tags,
            subdomain_id=tags["inlet"],
        )
        appctx = {
            "nu": self.mu / self.rho,
            "v": self.u_sol,
            "bcs_pcd": bcs_pcd,
            "ds_in": ds_in,
        }

        # Initialize Jacobian matrix
        assemble_matrix_block(self.A, self.J_form, bcs)
        self.A.assemble()

        # Wrap matrix for FieldSplit / PCD
        J_splittable = create_splittable_matrix_block(
            self.A, extract_blocks(J), **appctx
        )

        # PETSc options configuration
        problem_prefix = "ns_"
        opts = PETSc.Options()
        opts.prefixPush(problem_prefix)
        opts["snes_type"] = "newtonls"
        opts["snes_rtol"] = 1.0e-04
        opts["snes_max_it"] = 50
        opts["snes_ksp_ew"] = True

        opts["ksp_converged_reason"] = ""
        opts["ksp_max_it"] = 10000
        opts["ksp_type"] = "fgmres"
        opts["ksp_gmres_restart"] = 150
        opts["ksp_pc_side"] = "right"

        opts["pc_type"] = "python"
        opts["pc_python_type"] = "fenicsx_pctools.pc.WrappedPC"
        opts.prefixPush("wrapped_")
        opts["pc_type"] = "fieldsplit"
        opts["pc_fieldsplit_type"] = "schur"
        opts["pc_fieldsplit_schur_fact_type"] = "upper"
        opts["pc_fieldsplit_schur_precondition"] = "user"
        opts["pc_fieldsplit_0_fields"] = 0  # velocity
        opts["pc_fieldsplit_1_fields"] = 1  # pressure

        opts["fieldsplit_0_ksp_type"] = "gmres"
        opts["fieldsplit_0_pc_type"] = "hypre"

        opts["fieldsplit_1_ksp_type"] = "preonly"
        opts["fieldsplit_1_pc_type"] = "python"
        opts["fieldsplit_1_pc_python_type"] = f"fenicsx_pctools.pc.{pcd_type}"
        opts["fieldsplit_1_pcd_Mp_ksp_type"] = "preonly"
        opts["fieldsplit_1_pcd_Mp_pc_type"] = "jacobi"
        opts["fieldsplit_1_pcd_Ap_ksp_type"] = "cg"
        opts["fieldsplit_1_pcd_Ap_pc_type"] = "hypre"
        opts.prefixPop()  # wrapped_
        opts.prefixPop()  # ns_

        # Newton solver
        snes = PETSc.SNES().create(self.mesh.comm)
        snes.setFunction(self.assembleResidual, f=self.b, kargs={"bcs": bcs})
        snes.setJacobian(
            self.assembleJacobian, J=J_splittable, P=None, kargs={"bcs": bcs}
        )
        snes.setOptionsPrefix(problem_prefix)
        snes.setFromOptions()

        # x is the initial guess for the newton iteration = solution at previous time step
        start, end = self.x_n.getOwnershipRange()
        u_size_local = self.u_prev.x.petsc_vec.getLocalSize()
        self.x_n.setValues(range(start, start + u_size_local), self.u_prev.x.petsc_vec)
        self.x_n.setValues(
            range(start + u_size_local, end),
            self.p_prev.x.petsc_vec,
        )
        self.x_n.assemble()

        snes.setUp()

        print("\n--- CONFIGURACIÓN DEL SOLVER ---")
        viewer = PETSc.Viewer.STDOUT(self.mesh.comm)
        snes.view(viewer)
        print("--------------------------------\n")

        self.solver = snes

        # ojo aca faltan cosas de null space

    def solveStep(self):
        # Set BDF coefficients: BDF1 for first step, BDF2 thereafter
        if self.step_count == 0:
            self.bdf_a0.value = 1.0
            self.bdf_a1.value = -1.0
            self.bdf_a2.value = 0.0
        else:
            self.bdf_a0.value = 1.5
            self.bdf_a1.value = -2.0
            self.bdf_a2.value = 0.5

        PETSc.Sys.Print("Solving the nonlinear problem with SNES")
        self.solver.solve(None, self.x_n)
        its_snes = self.solver.getIterationNumber()
        its_ksp = self.solver.getLinearSolveIterations()
        PETSc.Sys.Print(
            f"Solver converged in {its_snes} nonlinear iterations"
            f" (with total number of {its_ksp} linear iterations)"
        )
        self.updateSolution(self.x_n)

        reason = self.solver.getConvergedReason()
        if reason < 0:
            raise RuntimeError(f"Did not converge, reason: {reason}.")

        # Save current u_prev (= u^n) into u_prev2 for the next step (= u^(n-1) there)
        self.u_prev2.x.array[:] = self.u_prev.x.array[:]
        self.u_prev2.x.scatter_forward()

        self.step_count += 1
