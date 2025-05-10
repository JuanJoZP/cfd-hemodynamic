import sys
import os
from importlib import import_module
sys.path.append(os.path.dirname(os.getcwd()))

solver_name = "solver2"
SolverIPCS = getattr(import_module(f"solvers.{solver_name}"), "SolverIPCS")

SolverIPCS()