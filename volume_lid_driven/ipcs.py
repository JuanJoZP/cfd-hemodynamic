#!/usr/bin/env python
# coding: utf-8

# In[1]:


from dolfinx import mesh, fem, io
import ufl
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np


# In[2]:


T = 1 
num_steps = 100
n_cells = 32
domain = mesh.create_unit_cube(MPI.COMM_WORLD, n_cells, n_cells, n_cells) 

f = (0,0,0)
dt = T/num_steps
mu = 1/50 # Re = 50
rho = 1


# In[3]:


import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))
from solver1 import SolverIPCS

solver = SolverIPCS(domain, dt, rho, mu, f)


# In[4]:


# lid-driven cavity flow
def lid(x):
    return np.isclose(x[2], 1)

def walls(x):
    return np.logical_or.reduce((np.isclose(x[2], 0), np.isclose(x[0], 0), np.isclose(x[0], 1), np.isclose(x[1], 0), np.isclose(x[1], 1)))

def corner(x):
    return np.logical_and.reduce((np.isclose(x[0], 0), np.isclose(x[1], 0), np.isclose(x[2], 0)))


fdim = domain.topology.dim - 1

lid_facets = mesh.locate_entities_boundary(domain, fdim, lid)
dofs_lid = fem.locate_dofs_topological(solver.velocity_space, fdim, lid_facets)
bc_lid  = fem.dirichletbc(fem.Constant(domain, PETSc.ScalarType((1, 0, 0))), dofs_lid, solver.velocity_space)

cavity_facets = mesh.locate_entities_boundary(domain, fdim, walls)
dofs_cavity = fem.locate_dofs_topological(solver.velocity_space, fdim, cavity_facets)
bc_cavity  = fem.dirichletbc(fem.Constant(domain, PETSc.ScalarType((0, 0, 0))), dofs_cavity, solver.velocity_space)

corner_facets = mesh.locate_entities_boundary(domain, fdim, corner)
dofs_corner = fem.locate_dofs_topological(solver.pressure_space, fdim, corner_facets)
bc_p_fix = fem.dirichletbc(fem.Constant(domain, PETSc.ScalarType(0)), dofs_corner, solver.pressure_space)

bc_u = [bc_lid, bc_cavity]
bc_p = [bc_p_fix]


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


tqdm = get_tqdm()
from datetime import datetime, timezone, timedelta

t = 0
progress = tqdm(desc="Resolviendo navier-stokes", total=num_steps) if domain.comm.rank == 0 else None
folder = datetime.now(tz=timezone(-timedelta(hours=5))).isoformat(timespec='seconds') if domain.comm.rank == 0 else None
folder = domain.comm.bcast(folder, root=0)
u_file = io.VTXWriter(domain.comm, f"{folder}/velocity.bp", solver.u_sol)
p_file = io.VTXWriter(domain.comm, f"{folder}/pressure.bp", solver.p_sol)
u_file.write(t)
p_file.write(t)

for n in range(num_steps):
    solver.solveStep(bc_u, bc_p)
    t += dt
    u_file.write(t)
    p_file.write(t)
    
    if progress:
        progress.update()

u_file.close()
p_file.close()
if progress:
    progress.close()

