#!/usr/bin/env python
# coding: utf-8

# # Benchmark Taylor green

# In[1]:


from dolfinx import mesh, fem, io
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np


# In[ ]:


solver_name = "solver2"
T = 0.1
num_steps = 100
n_cells = 32
domain = mesh.create_unit_cube(MPI.COMM_WORLD, n_cells, n_cells, n_cells) 

f = (0,0,0)
dt = T/num_steps
mu = 1/50 # Re = 50
rho = 1


# In[4]:


# solución analitica de: https://www.ljll.fr/~frey/papers/Navier-Stokes/Ethier%20C.R.,%20Steinman%20D.A.,%20Exact%20fully%203d%20Navier-Stokes%20solutions%20for%20benchmarking.pdf
a = np.pi/4
d = np.pi/2

def u_analytic(x, y, z, t):
    return np.vstack((-a*(np.exp(a*x)*np.sin(a*y + d*z) + np.exp(a*z)*np.cos(a*x + d*y))*np.exp(-1*d*d*t),
              -a*(np.exp(a*y)*np.sin(a*z + d*x) + np.exp(a*x)*np.cos(a*y + d*z))*np.exp(-1*d*d*t),
              -a*(np.exp(a*z)*np.sin(a*x + d*y) + np.exp(a*y)*np.cos(a*z + d*x))*np.exp(-1*d*d*t)
                     ))

def p_analytic(x, y, z, t):
    return -1*a*a*(1/2)*(np.exp(2*a*x) + np.exp(2*a*y) + np.exp(2*a*z) + 2*np.sin(a*x + d*y)*np.cos(a*z + d*x)*np.exp(a*y + a*z) \
                        + 2*np.sin(a*y + d*z)*np.cos(a*x + d*y)*np.exp(a*z + a*x) + 2*np.sin(a*z + d*x)*np.cos(a*y + d*z)*np.exp(a*x + a*y)) \
            *np.exp(-2*d*d*t)

u_sol_analytic = fem.Function(velocity_function_space)
P_sol_analytic = fem.Function(pressure_function_space)


# In[5]:


import sys
import os
from importlib import import_module
sys.path.append(os.path.dirname(os.getcwd()))
SolverIPCS = getattr(import_module(f"solvers.{solver_name}"), "SolverIPCS")

solver = SolverIPCS(domain, dt, rho, mu, f, lambda x: u_analytic(*x, 0))

# puede no ser necesario
#p_prev.interpolate(lambda x: p_analytic(*x, 0))


# In[6]:


# Dirichlet BC en todas las fronteras, dada por la solución analítica
domain.topology.create_connectivity(domain.topology.dim-1, domain.topology.dim)
boundary_facets = mesh.exterior_facet_indices(domain.topology) # acá hay un tema con los procesos y es que devuelve los indices locales (del proceso)
dofs_boundary_u = fem.locate_dofs_topological(solver.velocity_space, domain.topology.dim-1, boundary_facets)
dofs_boundary_p = fem.locate_dofs_topological(solver.pressure_space, domain.topology.dim-1, boundary_facets)

u_bc = fem.Function(solver.velocity_space)
p_bc = fem.Function(solver.pressure_space)
bcu  = [fem.dirichletbc(u_bc, dofs_boundary_u)]
bcp  = [fem.dirichletbc(p_bc, dofs_boundary_p)]


# In[ ]:


solver.assembleTimeIndependent(bcu, bcp)


# In[8]:


# !! para debugear y checkear valores de la funcion en un punto
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells

tree = bb_tree(domain, domain.geometry.dim)
points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 1]])
cell_candidates = compute_collisions_points(tree, points)
colliding_cells = compute_colliding_cells(domain, cell_candidates, points)
cells = [] 
cells.append(colliding_cells.links(0)[0])
cells.append(colliding_cells.links(1)[0])
cells.append(colliding_cells.links(2)[0])


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


# In[18]:


tqdm = get_tqdm()
from dolfinx.fem import assemble_scalar
from datetime import datetime, timezone, timedelta
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells

t = 0
i = 0
progress = tqdm(desc="Resolviendo navier-stokes", total=num_steps) if domain.comm.rank == 0 else None
error_log = open(f"{folder}/error.txt", "w") if domain.comm.rank == 0 else None
date = datetime.now(tz=timezone(-timedelta(hours=5))).isoformat(timespec='seconds') if domain.comm.rank == 0 else None
date = domain.comm.bcast(folder, root=0)
u_file = io.VTXWriter(domain.comm, f"{solver_name}/{date}/velocity.bp", solver.u_sol)
p_file = io.VTXWriter(domain.comm, f"{solver_name}/{date}/pressure.bp", solver.p_sol)
u_file.write(t)
p_file.write(t)

for n in range(num_steps):
    if progress:
        progress.update()
        
    t += dt.value
    i += 1

    u_bc.interpolate(lambda x: u_analytic(*x, t))
    p_bc.interpolate(lambda x: p_analytic(*x, t))
    
    solver.solveStep(bc_u, bc_p)
    
    u_file.write(t)
    p_file.write(t)

    # error relativo: |u_sol - u_analitica| / |u_analitica|
    u_sol_analytic.interpolate(lambda x: u_analytic(*x, t))

    # !! DEBUG
    #print("t=",t)
    #for (j, point) in enumerate(points):
    #    print("aprox", u_sol.eval(point, cells[j]))
    #    print("aprox2", u_prev.eval(point, cells[j]))
    #    print("inter", u_sol_analytic.eval(point, cells[j]))
    #    print("exact", u_analytic(*point,t).flatten())
    #    print("error", np.sum(u_sol.eval(point, cells[j]) -u_analytic(*point,t).flatten()))
    # !! DEBUG
    
    error_abs_integral = fem.form(inner(u_sol_analytic - u_sol, u_sol_analytic - u_sol) * dx)
    error_abs = np.sqrt(comm.allreduce(assemble_scalar(error_abs_integral), op=MPI.SUM))
    norm_u_analytic_integral = fem.form(inner(u_sol_analytic, u_sol_analytic) * dx)
    norm_u_analytic = np.sqrt(comm.allreduce(assemble_scalar(norm_u_analytic_integral), op=MPI.SUM))
    error = error_abs / norm_u_analytic

    if error_log:
        error_log.write('t = %.3f: error = %.3g' % (t, error) + "\n")

u_file.close()
p_file.close()
if progress:
    progress.close()
if error_log:
    error_log.close()


# In[ ]:




