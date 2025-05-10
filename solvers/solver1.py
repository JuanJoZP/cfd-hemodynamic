# IPCS con método del punto medio y linealización del término convectivo mediante la solución en el paso anterior

from typing import Callable

from petsc4py import PETSc
import numpy as np

from basix.ufl import element
from dolfinx.mesh import Mesh
from dolfinx.fem import form, DirichletBC, Constant, Function, functionspace
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, create_vector, set_bc
from ufl import FacetNormal, dx, ds, dot, inner, sym, nabla_grad, Identity, lhs, rhs, div, TrialFunction, TestFunction


class SolverIPCS():
    def __init__(self,
                 domain: Mesh,
                 dt: float,
                 rho: float,
                 mu: float,
                 f: list,
                 initial_velocity: Callable[[np.ndarray], np.ndarray] = None
                ):
        self.domain = domain
        self.dt = Constant(domain, PETSc.ScalarType(dt))
        self.rho = Constant(domain, PETSc.ScalarType(rho))
        self.mu = Constant(domain, PETSc.ScalarType(mu))
        self.f = Constant(domain, PETSc.ScalarType(f))

        element_velocity = element("Lagrange", domain.topology.cell_name(), 2, shape=(domain.geometry.dim,))
        element_pressure = element("Lagrange", domain.topology.cell_name(), 1)
        self.velocity_space = functionspace(domain, element_velocity) 
        self.pressure_space = functionspace(domain, element_pressure)

        u = TrialFunction(self.velocity_space)
        p = TrialFunction(self.pressure_space)
        
        v = TestFunction(self.velocity_space)
        q = TestFunction(self.pressure_space)
        
        self.u_sol = Function(self.velocity_space)
        self.u_prev = Function(self.velocity_space)
        self.p_sol = Function(self.pressure_space) 
        self.p_prev = Function(self.pressure_space)
        
        # condiciones iniciales
        if initial_velocity:
            self.u_prev.interpolate(initial_velocity)
            self.u_sol.interpolate(initial_velocity)

        # forma variacional
        u_midpoint = 0.5*(self.u_prev + u)
        n = FacetNormal(domain)

        F1 = self.rho*dot((u - self.u_prev) / self.dt, v)*dx 
        F1 += self.rho*dot(dot(self.u_prev, nabla_grad(self.u_prev)), v)*dx 
        F1 += inner(self.sigma(u_midpoint, self.p_prev), self.epsilon(v))*dx 
        F1 += dot(self.p_prev*n, v)*ds - dot(mu*nabla_grad(u_midpoint)*n, v)*ds 
        F1 -= dot(self.f, v)*dx
        self.a1 = form(lhs(F1))
        self.L1 = form(rhs(F1))
        
        F2 = dot(nabla_grad(p), nabla_grad(q))*dx 
        F2 -= dot(nabla_grad(self.p_prev), nabla_grad(q))*dx 
        F2 += (self.rho/self.dt)*div(self.u_sol)*q*dx
        self.a2 = form(lhs(F2))
        self.L2 = form(rhs(F2))
        
        F3 = self.rho*dot((u - self.u_sol), v)*dx 
        F3 += self.dt*dot(nabla_grad(self.p_sol - self.p_prev), v)*dx
        self.a3 = form(lhs(F3))
        self.L3 = form(rhs(F3))

    
    def assembleTimeIndependent(self, bcu: list[DirichletBC], bcp: list[DirichletBC]):
        self.A1 = assemble_matrix(self.a1, bcs=bcu)
        self.A1.assemble()
        self.b1 = create_vector(self.L1)

        self.A2 = assemble_matrix(self.a2, bcs=bcp)
        self.A2.assemble()
        self.b2 = create_vector(self.L2)

        self.A3 = assemble_matrix(self.a3)
        self.A3.assemble()
        self.b3 = create_vector(self.L3)

        # inicializar solvers de PETSc
        self.solver1 = PETSc.KSP().create(self.domain.comm)
        self.solver1.setOperators(self.A1)
        self.solver1.setType(PETSc.KSP.Type.BCGS)
        pc1 = self.solver1.getPC()
        pc1.setType(PETSc.PC.Type.HYPRE)
        pc1.setHYPREType("boomeramg")
        
        self.solver2 = PETSc.KSP().create(self.domain.comm)
        self.solver2.setOperators(self.A2)
        self.solver2.setType(PETSc.KSP.Type.BCGS)
        pc2 = self.solver2.getPC()
        pc2.setType(PETSc.PC.Type.HYPRE)
        pc2.setHYPREType("boomeramg")
        
        self.solver3 = PETSc.KSP().create(self.domain.comm)
        self.solver3.setOperators(self.A3)
        self.solver3.setType(PETSc.KSP.Type.CG)
        pc3 = self.solver3.getPC()
        pc3.setType(PETSc.PC.Type.SOR)

    def solveStep(self, bcu: list[DirichletBC], bcp: list[DirichletBC]):
        # paso 1
        with self.b1.localForm() as loc_1:
            loc_1.set(0)
        assemble_vector(self.b1, self.L1)
        apply_lifting(self.b1, [self.a1], [bcu])
        self.b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(self.b1, bcu)
        self.solver1.solve(self.b1, self.u_sol.x.petsc_vec)
        self.u_sol.x.scatter_forward()
    
        # paso 2
        with self.b2.localForm() as loc_2:
            loc_2.set(0)
        assemble_vector(self.b2, self.L2)
        apply_lifting(self.b2, [self.a2], [bcp])
        self.b2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(self.b2, bcp)
        self.solver2.solve(self.b2, self.p_sol.x.petsc_vec)
        self.p_sol.x.scatter_forward()
    
        # paso 3
        with self.b3.localForm() as loc_3:
            loc_3.set(0)
        assemble_vector(self.b3, self.L3)
        self.b3.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        self.solver3.solve(self.b3, self.u_sol.x.petsc_vec)
        self.u_sol.x.scatter_forward()
        
        # actualizar solucion previa para el siguiente t
        self.u_prev.x.array[:] = self.u_sol.x.array[:]
        self.p_prev.x.array[:] = self.p_sol.x.array[:]

    
    def epsilon(self, u):
        return sym(nabla_grad(u))

    def sigma(self, u, p):
        return 2*self.mu*self.epsilon(u) - p*Identity(len(u))
    
    