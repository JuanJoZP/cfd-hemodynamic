from typing import Callable

from petsc4py import PETSc
import numpy as np

from dolfinx.mesh import Mesh
from dolfinx.fem import form, DirichletBC, Constant, Function
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, create_vector, set_bc
from ufl import FacetNormal, dx, ds, dot, inner, sym, nabla_grad, Identity, lhs, rhs, div, TrialFunction, TestFunction


class SolverIPCS():
    def __init__(self,
                 domain: Mesh,
                 dt: float,
                 rho: float,
                 mu: float,
                 f: np.ndarray,
                 initial_velocity: Callable[[np.ndarray], np.ndarray],
                 bcu: list[DirichletBC],
                 bcp: list[DirichletBC]
                ):
        self.dt = Constant(domain, PETSc.ScalarType(dt))
        self.rho = Constant(domain, PETSc.ScalarType(rho))
        self.mu = Constant(domain, PETSc.ScalarType(mu))
        self.f = Constant(domain, PETSc.ScalarType(f))

        u = TrialFunction(velocity_function_space)
        p = TrialFunction(pressure_function_space)
        
        v = TestFunction(velocity_function_space)
        q = TestFunction(pressure_function_space)
        
        self.u_sol = Function(velocity_function_space)
        self.u_prev = Function(velocity_function_space) 
        self.p_sol = Function(pressure_function_space) 
        self.p_prev = Function(pressure_function_space)
        
        # condiciones iniciales
        u_prev.interpolate(initial_velocity)

        # forma variacional
        u_midpoint = 0.5*(u_prev + u)
        n = FacetNormal(domain)

        F1 = self.rho*dot((u - self.u_prev) / self.dt, v)*dx 
        F1 += self.rho*dot(dot(self.u_prev, nabla_grad(self.u_prev)), v)*dx 
        F1 += inner(sigma(u_midpoint, self.p_prev), epsilon(v))*dx 
        F1 += dot(self.p_prev*n, v)*ds - dot(mu*nabla_grad(u_midpoint)*n, v)*ds 
        F1 -= dot(f, v)*dx
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

        # ensamblar formas bilineales (son independientes del tiempo)
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
        solver1 = PETSc.KSP().create(domain.comm)
        solver1.setOperators(self.A1)
        solver1.setType(PETSc.KSP.Type.BCGS)
        pc1 = solver1.getPC()
        pc1.setType(PETSc.PC.Type.HYPRE)
        pc1.setHYPREType("boomeramg")
        
        solver2 = PETSc.KSP().create(domain.comm)
        solver2.setOperators(self.A2)
        solver2.setType(PETSc.KSP.Type.BCGS)
        pc2 = solver2.getPC()
        pc2.setType(PETSc.PC.Type.HYPRE)
        pc2.setHYPREType("boomeramg")
        
        solver3 = PETSc.KSP().create(domain.comm)
        solver3.setOperators(self.A3)
        solver3.setType(PETSc.KSP.Type.CG)
        pc3 = solver3.getPC()
        pc3.setType(PETSc.PC.Type.SOR)

    def solveStep(self, bcu: list[DirichletBC], bcp: list[DirichletBC])
        # paso 1
        with self.b1.localForm() as loc_1:
            loc_1.set(0)
        assemble_vector(self.b1, self.L1)
        apply_lifting(self.b1, [self.a1], [bcu])
        b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(self.b1, bcu)
        solver1.solve(b1, self.u_sol.x.petsc_vec)
        self.u_sol.x.scatter_forward()
    
        # paso 2
        with self.b2.localForm() as loc_2:
            loc_2.set(0)
        assemble_vector(self.b2, self.L2)
        apply_lifting(self.b2, [self.a2], [bcp])
        b2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(self.b2, bcp)
        solver2.solve(self.b2, p_sol.x.petsc_vec)
        p_sol.x.scatter_forward()
    
        # paso 3
        with self.b3.localForm() as loc_3:
            loc_3.set(0)
        assemble_vector(self.b3, self.L3)
        b3.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        solver3.solve(self.b3, self.u_sol.x.petsc_vec)
        self.u_sol.x.scatter_forward()
        
        # actualizar solucion previa para el siguiente t
        self.u_prev.x.array[:] = self.u_sol.x.array[:]
        self.p_prev.x.array[:] = self.p_sol.x.array[:]

        return (self.u_sol, self.p_sol) # retorna solucion para calcular errores u otros procesos
    
    @staticmethod
    def epsilon(u):
        return sym(nabla_grad(u))

    @staticmethod
    def sigma(u, p):
        return 2*mu*epsilon(u) - p*Identity(len(u))
    
    