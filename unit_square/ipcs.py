#!/usr/bin/env python
# coding: utf-8

# # Poisellieu flow

# In[1]:


from dolfinx import mesh, fem, io
import ufl
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np


# In[2]:


T = 5
num_steps = 1000
n_cells = 32
domain = mesh.create_unit_square(MPI.COMM_WORLD, n_cells, n_cells)

f = (0, 0)
dt = T/num_steps
mu = 1
rho = 1


# In[3]:


import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))
from solver1 import SolverIPCS

def initial_velocity(x):
    values = np.zeros((domain.geometry.dim, x.shape[1]), dtype=PETSc.ScalarType)
    values[1] = 10
    return values

solver = SolverIPCS(domain, dt, rho, mu, f, initial_velocity)


# In[4]:


# bounda
def inflow(x):
    return np.isclose(x[0], 0)

def outflow(x):
    return np.isclose(x[0], 1)

def walls(x):
    return np.logical_or(
        np.isclose(x[1], 0), np.isclose(x[1], 1)
   )

fdim = domain.topology.dim - 1
inflow_facets = mesh.locate_entities_boundary(domain, fdim, inflow)
dofs_inflow = fem.locate_dofs_topological(solver.pressure_space, fdim, inflow_facets)
bc_inflow  = fem.dirichletbc(fem.Constant(domain, PETSc.ScalarType(8)), dofs_inflow, solver.pressure_space)

outflow_facets = mesh.locate_entities_boundary(domain, fdim, outflow)
dofs_outflow = fem.locate_dofs_topological(solver.pressure_space, fdim, outflow_facets)
bc_outflow  = fem.dirichletbc(fem.Constant(domain, PETSc.ScalarType(0)), dofs_outflow, solver.pressure_space)
bc_p = [bc_inflow, bc_outflow]

walls_facets = mesh.locate_entities_boundary(domain, fdim, walls)
dofs_walls = fem.locate_dofs_topological(solver.velocity_space, fdim, walls_facets)
bc_noslip  = fem.dirichletbc(fem.Constant(domain, PETSc.ScalarType((0, 0))), dofs_walls, solver.velocity_space)
bc_u = [bc_noslip]


# In[5]:


solver.assembleTimeIndependent(bc_u, bc_p)


# In[ ]:


def get_tqdm():
    try:
        # Check if inside Jupyter notebook
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        if shell in ["ZMQInteractiveShell"]:
            from tqdm.notebook import tqdm as notebook_tqdm
            return notebook_tqdm
    except:
        pass
    from tqdm import tqdm  # fallback for scripts
    return tqdm


# In[6]:


from dolfinx.fem import assemble_scalar
from ufl import inner, dx
from datetime import datetime, timezone, timedelta
tqdm = get_tqdm()

t = 0
progress = tqdm(desc="Resolviendo navier-stokes", total=num_steps) if domain.comm.rank == 0 else None
error_log = open(f"{folder}/error.txt", "w") if domain.comm.rank == 0 else None
folder = datetime.now(tz=timezone(-timedelta(hours=5))).isoformat(timespec='seconds') if domain.comm.rank == 0 else None
folder = domain.comm.bcast(folder, root=0)
u_file = io.VTXWriter(domain.comm, f"{folder}/velocity.bp", solver.u_sol)
p_file = io.VTXWriter(domain.comm, f"{folder}/pressure.bp", solver.p_sol)
u_file.write(t)
p_file.write(t)

u_e = fem.Function(solver.velocity_space)
u_e.interpolate(lambda x: np.vstack((4.0*x[1]*(1.0 - x[1]), 0.0*x[0])))

for n in range(num_steps):
    solver.solveStep(bc_u, bc_p)
    error_abs_integral = fem.form(inner(u_e - solver.u_sol, u_e - solver.u_sol) * dx)
    error_abs = np.sqrt(solver.domain.comm.allreduce(assemble_scalar(error_abs_integral), op=MPI.SUM))
    norm_u_e_integral = fem.form(inner(u_e, u_e) * dx)
    norm_u_e = np.sqrt(solver.domain.comm.allreduce(assemble_scalar(norm_u_e_integral), op=MPI.SUM))
    error = error_abs / norm_u_e

    t += dt
    if error_log:
        error_log.write('t = %.2f: error = %.3g' % (t, error) + "\n")
    if progress:
        progress.update()
    
    u_file.write(t)
    p_file.write(t)

u_file.close()
p_file.close()
if error_log:
    error_log.close()
if progress:
    progress.close()


# In[ ]:




