# DFG 2D-1 Benchmark Solver
# Based on stabilized_schur_pressure_backflow.py
# Reference: stationary cases use U_max = 0.3, U_mean = 0.2, L = 0.1, rho = 1.0.

from typing import Callable, Optional

import numpy as np
from dolfinx.fem import Constant, Function, assemble_scalar, form, functionspace
from dolfinx.fem.petsc import (
    assemble_matrix_block,
    assemble_vector_block,
    create_matrix_block,
    create_vector_block,
)
from dolfinx.mesh import Mesh
from mpi4py import MPI
from petsc4py import PETSc
from ufl import (
    FacetNormal,
    Identity,
    Measure,
    MixedFunctionSpace,
    TestFunctions,
    TrialFunctions,
    conditional,
    derivative,
    div,
    dot,
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
    def __init__(
        self,
        mesh: Mesh,
        dt: float,
        rho: float,
        mu: float,
        f: list,
        initial_velocity: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        beta_backflow: float = 0.2,
        p_grade: int = 1,
    ):
        self.beta_backflow = float(beta_backflow)

        super().__init__(mesh, dt, rho, mu, f)

        super().initVelocitySpace(
            "Lagrange", mesh.topology.cell_name(), p_grade, shape=(mesh.geometry.dim,)
        )
        super().initPressureSpace("Lagrange", mesh.topology.cell_name(), p_grade)

        self.VQ = MixedFunctionSpace(self.V, self.Q)
        v, q = TestFunctions(self.VQ)

        if initial_velocity:
            self.u_prev.interpolate(initial_velocity)

        u_sol = self.u_sol
        p_sol = self.p_sol
        u_prev = self.u_prev
        u_mid = 0.5 * (u_sol + u_prev)
        self.u_mid = u_mid
        n = FacetNormal(self.mesh)

        # Standard Navier-Stokes residual (matches stabilized_schur_pressure_backflow)
        F = self.rho * inner(v, (u_sol - u_prev) / self.dt) * dx
        F += self.rho * dot(v, dot(u_mid, nabla_grad(u_mid))) * dx
        F -= inner(v, self.rho * self.f) * dx
        F += inner(self.epsilon(v), self.sigma(u_mid, p_sol, self.mu)) * dx
        F += inner(q, div(u_mid)) * dx

        # Stabilization: SUPG/PSPG/LSIC
        V_dg0 = functionspace(mesh, ("DG", 0))
        h = Function(V_dg0)
        h.x.array[:] = mesh.h(
            mesh.topology.dim,
            np.arange(h.x.index_map.size_local + h.x.index_map.num_ghosts),
        )
        vnorm = sqrt(inner(u_prev, u_prev))
        R = self.rho * ((u_sol - u_prev) / self.dt + dot(u_mid, nabla_grad(u_mid)))
        R -= div(self.sigma(u_mid, p_sol, self.mu))
        R -= self.rho * self.f

        eps = Constant(self.mesh, np.finfo(PETSc.ScalarType()).resolution)
        tau_supg1 = h / conditional(ge((2.0 * vnorm), eps), (2.0 * vnorm), eps)
        tau_supg2 = self.dt / 2.0
        tau_supg3 = (h * h) / (4.0 * (self.mu / self.rho))
        tau_supg = (
            1 / (tau_supg1**2) + 1 / (tau_supg2**2) + 1 / (tau_supg3**2)
        ) ** (-1 / 2)

        F += inner(tau_supg * R, dot(u_mid, nabla_grad(v))) * dx  # SUPG
        F += (1 / self.rho) * inner(tau_supg * R, grad(q)) * dx  # PSPG

        Re_local = (vnorm * h) / (2.0 * (self.mu / self.rho))
        z = conditional(le(Re_local, 3), Re_local / 3, 1.0)
        tau_lsic = (vnorm * h * z) / 2.0
        F += tau_lsic * inner(div(u_mid), self.rho * div(v)) * dx  # LSIC

        self.F = F
        self._v_test = v
        self._h = h

    def setup(
        self,
        bcu: list[BoundaryCondition],
        bcp: list[BoundaryCondition],
        facet_tags=None,
        tags=None,
    ) -> None:
        if tags is None or "obstacle" not in tags:
            raise ValueError(
                "The 'obstacle' tag is required in DFG solver for drag/lift calculation."
            )

        ds_out = Measure(
            "ds",
            domain=self.mesh,
            subdomain_data=facet_tags,
            subdomain_id=tags["outlet"],
        )
        ds_obs = Measure(
            "ds",
            domain=self.mesh,
            subdomain_data=facet_tags,
            subdomain_id=tags["obstacle"],
        )
        n = FacetNormal(self.mesh)
        u = self.u_mid
        v = self._v_test

        # 1. Symmetric viscous traction at outlet (Do-nothing)
        self.F -= dot(dot(2 * self.mu * self.epsilon(u), n), v) * ds_out

        # 2. Backflow stabilization
        u_prev = self.u_prev
        un_prev = dot(u_prev, n)
        un_minus = 0.5 * (un_prev - abs(un_prev))
        self.F -= self.beta_backflow * self.rho * un_minus * dot(u, v) * ds_out

        # 3. Linearize and compile
        du, dp = TrialFunctions(self.VQ)
        J = derivative(self.F, (self.u_sol, self.p_sol), (du, dp))
        self.F_form = form(extract_blocks(self.F))
        self.J_form = form(extract_blocks(J))
        self.A = create_matrix_block(self.J_form)
        self.b = create_vector_block(self.F_form)
        self.x_n = self.b.duplicate()

        # 4. Dirichlet BCs
        self.bcu_d = [bc.getBC(self.V) for bc in bcu]
        self.bcp_d = [bc.getBC(self.Q) for bc in bcp]

        # 5. Drag and Lift coefficients forms
        # Cd = 2 * Fx / (rho * U_mean^2 * L)
        # Force exerted by fluid on cylinder: F = integral( -sigma * n ) where n points INTO cylinder.
        self.drag_form = form(
            dot(
                -self.sigma(self.u_sol, self.p_sol, self.mu) * n,
                Constant(self.mesh, PETSc.ScalarType((1, 0))),
            )
            * ds_obs
        )
        self.lift_form = form(
            dot(
                -self.sigma(self.u_sol, self.p_sol, self.mu) * n,
                Constant(self.mesh, PETSc.ScalarType((0, 1))),
            )
            * ds_obs
        )

        # 6. SNES Solver Setup
        snes = PETSc.SNES().create(self.mesh.comm)
        snes.setOptionsPrefix("dfg_nonlinear_")
        snes.setType("newtonls")
        snes.setTolerances(rtol=1e-10, atol=1e-12, stol=1e-12, max_it=50)

        snes.setFunction(
            self.assembleResidual, f=self.b, kargs={"bcs": [*self.bcu_d, *self.bcp_d]}
        )
        snes.setJacobian(
            self.assembleJacobian,
            J=self.A,
            P=None,
            kargs={"bcs": [*self.bcu_d, *self.bcp_d]},
        )

        start, end = self.x_n.getOwnershipRange()
        u_size_local = self.u_prev.x.petsc_vec.getLocalSize()
        self.x_n.setValues(range(start, start + u_size_local), self.u_prev.x.petsc_vec)
        self.x_n.setValues(range(start + u_size_local, end), self.p_prev.x.petsc_vec)
        self.x_n.assemble()

        ksp = snes.getKSP()
        ksp.setType("fgmres")
        pc = ksp.getPC()
        pc.setType("fieldsplit")
        pc.setFieldSplitType(PETSc.PC.CompositeType.SCHUR)
        pc.setFieldSplitSchurFactType(PETSc.PC.SchurFactType.FULL)

        V_map, Q_map = self.V.dofmap.index_map, self.Q.dofmap.index_map
        u_bs = self.V.dofmap.index_map_bs
        offset_u = V_map.local_range[0] * u_bs + Q_map.local_range[0]
        offset_p = offset_u + V_map.size_local * u_bs
        is_u = PETSc.IS().createStride(
            V_map.size_local * u_bs, offset_u, 1, comm=self.mesh.comm
        )
        is_p = PETSc.IS().createStride(
            Q_map.size_local, offset_p, 1, comm=self.mesh.comm
        )
        pc.setFieldSplitIS(("u", is_u), ("p", is_p))

        self.solver = snes

    def solveStep(self):
        self.solver.solve(None, self.x_n)

        # Monitor SNES
        its = self.solver.getIterationNumber()
        reason = self.solver.getConvergedReason()
        PETSc.Sys.Print(f"    SNES iterations: {its}, Reason: {reason}")

        self.updateSolution(self.x_n)

        # Coefs
        fd = self.mesh.comm.allreduce(assemble_scalar(self.drag_form), op=MPI.SUM)
        fl = self.mesh.comm.allreduce(assemble_scalar(self.lift_form), op=MPI.SUM)
        # Scaling for U_mean = 0.2 -> 2 / (0.04 * 0.1) = 500
        cd = 500.0 * fd
        cl = 500.0 * fl
        PETSc.Sys.Print(
            f"    DFG Metrics: Fd={fd:.6e}, Fl={fl:.6e} | Cd={cd:.6f}, Cl={cl:.66f}"
        )

    def updateSolution(self, x: PETSc.Vec) -> None:
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

    def assembleJacobian(self, snes, x, J_mat, P_mat, bcs=[]):
        J_mat.zeroEntries()
        assemble_matrix_block(J_mat, self.J_form, bcs)
        J_mat.assemble()

    def assembleResidual(self, snes, x, F_vec, bcs=[]):
        F_vec.set(0.0)
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        self.updateSolution(x)
        [bc.update() for bc in bcs]
        assemble_vector_block(
            F_vec, self.F_form, self.J_form, bcs=bcs, x0=x, alpha=-1.0
        )
        F_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
