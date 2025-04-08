#!/usr/bin/env python
# coding: utf-8

# In[1]:


#https://jsdokken.com/dolfinx-tutorial/chapter2/navierstokes.html#variational-formulation

from dolfinx import mesh, fem, io
import ufl
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np

T = 1 
num_steps = 100
n_cells = 32


# In[2]:


domain = mesh.create_unit_cube(MPI.COMM_WORLD, n_cells, n_cells, n_cells)
velocity_function_space = fem.functionspace(domain, ("Lagrange", 2, (3,))) 
pressure_function_space = fem.functionspace(domain, ("Lagrange", 1))


# In[3]:


f = fem.Constant(domain, PETSc.ScalarType((0,0,0)))
dt = fem.Constant(domain, T/num_steps)
mu = fem.Constant(domain, PETSc.ScalarType(1/50)) # Re = 50
rho = fem.Constant(domain, PETSc.ScalarType(1))


# In[4]:


u = ufl.TrialFunction(velocity_function_space)
p = ufl.TrialFunction(pressure_function_space)

v = ufl.TestFunction(velocity_function_space)
q = ufl.TestFunction(pressure_function_space)

u_sol = fem.Function(velocity_function_space) # function to store u solved
u_prev = fem.Function(velocity_function_space) # u from previous time step
p_sol = fem.Function(pressure_function_space) 
p_prev = fem.Function(pressure_function_space)


# In[5]:


# lid-driven cavity flow
def lid(x):
    return np.isclose(x[2], 1)

def walls(x):
    return np.logical_or.reduce((np.isclose(x[2], 0), np.isclose(x[0], 0), np.isclose(x[0], 1), np.isclose(x[1], 0), np.isclose(x[1], 1)))

def corner(x):
    return np.logical_and.reduce((np.isclose(x[0], 0), np.isclose(x[1], 0), np.isclose(x[2], 0)))


fdim = domain.topology.dim - 1

lid_facets = mesh.locate_entities_boundary(domain, fdim, lid)
dofs_lid = fem.locate_dofs_topological(velocity_function_space, fdim, lid_facets)
bc_lid  = fem.dirichletbc(fem.Constant(domain, PETSc.ScalarType((1, 0, 0))), dofs_lid, velocity_function_space)

cavity_facets = mesh.locate_entities_boundary(domain, fdim, walls)
dofs_cavity = fem.locate_dofs_topological(velocity_function_space, fdim, cavity_facets)
bc_cavity  = fem.dirichletbc(fem.Constant(domain, PETSc.ScalarType((0, 0, 0))), dofs_cavity, velocity_function_space)

corner_facets = mesh.locate_entities_boundary(domain, fdim, corner)
dofs_corner = fem.locate_dofs_topological(pressure_function_space, fdim, corner_facets)
bc_p_fix = fem.dirichletbc(fem.Constant(domain, PETSc.ScalarType(0)), dofs_corner, pressure_function_space)

bc_u = [bc_lid, bc_cavity]
bc_p = [bc_p_fix]


# In[6]:


from ufl import FacetNormal, dx, ds, dot, inner, sym, nabla_grad, Identity, lhs, rhs, div

u_midpoint = 0.5*(u_prev + u)
n = FacetNormal(domain)

def epsilon(u):
    return sym(nabla_grad(u))

def sigma(u, p):
    return 2*mu*epsilon(u) - p*Identity(len(u))

# step 1
form1 = rho*dot((u - u_prev) / dt, v)*dx \
      + rho*dot(dot(u_prev, nabla_grad(u_prev)), v)*dx \
      + inner(sigma(u_midpoint, p_prev), epsilon(v))*dx \
      + dot(p_prev*n, v)*ds - dot(mu*nabla_grad(u_midpoint)*n, v)*ds \
      - dot(f, v)*dx
bilinear1 = lhs(form1)
linear1 = rhs(form1)

# step 2
form2 = dot(nabla_grad(p), nabla_grad(q))*dx \
      - dot(nabla_grad(p_prev), nabla_grad(q))*dx \
      + (rho/dt)*div(u_sol)*q*dx
bilinear2 = lhs(form2)
linear2 = rhs(form2)

# step 3
form3 = rho*dot((u - u_sol), v)*dx \
      + dt*dot(nabla_grad(p_sol - p_prev), v)*dx
bilinear3 = lhs(form3)
linear3 = rhs(form3)


# In[7]:


from dolfinx.fem.petsc import LinearProblem
from tqdm.notebook import tqdm
from datetime import date

t = 0
u_file = io.VTXWriter(domain.comm, f"{date.today()}/velocity.bp", u_sol)
p_file = io.VTXWriter(domain.comm, f"{date.today()}/pressure.bp", p_sol)
u_file.write(t)
p_file.write(t)

progress = tqdm(desc="Resolviendo navier-stokes", total=num_steps)

for n in range(num_steps):
    progress.update()
    t += dt
    
    problem1 = LinearProblem(bilinear1, linear1, bc_u, u_sol)
    problem1.solve()

    problem2 = LinearProblem(bilinear2, linear2, bc_p, p_sol)
    problem2.solve()

    problem3 = LinearProblem(bilinear3, linear3, bc_u, u_sol)
    problem3.solve()

    u_file.write(t)
    p_file.write(t)

    u_prev.x.array[:] = u_sol.x.array
    p_prev.x.array[:] = p_sol.x.array

