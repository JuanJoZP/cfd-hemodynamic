import argparse
import ast
import sys
from src.simulation import Simulation

def main():
    parser = argparse.ArgumentParser(description="Run CFD Simulations")
    parser.add_argument("--simulation", required=True, help="Scenario name (e.g., dfg_1)")
    parser.add_argument("--solver", required=True, help="Solver name (e.g., stabilized_schur)")
    parser.add_argument("--mu", type=float, default=None, help="Viscosity (optional, uses scenario default if not provided)")
    parser.add_argument("--rho", type=float, default=None, help="Density (optional, uses scenario default if not provided)")
    parser.add_argument("--T", type=float, required=True, help="Total time")
    parser.add_argument("--dt", type=float, required=True, help="Time step")
    parser.add_argument("--name", required=True, help="Name of the run")
    parser.add_argument("--output_dir", default="results", help="Output directory")

    # Use parse_known_args to handle arbitrary Scenario arguments
    args, unknown = parser.parse_known_args()

    kwargs = {}
    
    # Process unknown arguments
    i = 0
    while i < len(unknown):
        arg = unknown[i]
        if arg.startswith("--"):
            key = arg[2:]
            val = None
            if i + 1 < len(unknown) and not unknown[i+1].startswith("--"):
                val = unknown[i+1]
                i += 1
            else:
                # Flag without value, assume True? or missing value?
                val = True
            
            # Try to convert to proper type
            if isinstance(val, str):
                try:
                    # try evaluating as literal (int, float, list, tuple)
                    val = ast.literal_eval(val)
                except:
                    # keep as string
                    pass
            
            kwargs[key] = val
        i += 1
        
    # Handle f specifically if it was passed generically or needs generic logic
    # If passed as --f "(0,0)", ast.literal_eval above handles it.

    # Only include mu/rho if explicitly provided
    if args.mu is not None:
        kwargs['mu'] = args.mu
    if args.rho is not None:
        kwargs['rho'] = args.rho

    from mpi4py import MPI
    if MPI.COMM_WORLD.Get_rank() == 0:
        print(f"Running simulation with extra args: {kwargs}")

    try:
        sim = Simulation(
            name=args.name,
            simulation=args.simulation,
            solver=args.solver,
            T=args.T,
            dt=args.dt,
            output_dir=args.output_dir,
            **kwargs
        )
    except ValueError as e:
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(f"\n[ERROR] Invalid configuration: {e}")
        return 1
    except ImportError as e:
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(f"\n[ERROR] Failed to load module: {e}")
        return 1
    except SyntaxError as e:
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(f"\n[ERROR] Syntax error in module: {e}")
        return 1
    except RuntimeError as e:
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(f"\n[ERROR] Initialization failed: {e}")
        return 1
    except Exception as e:
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(f"\n[ERROR] Unexpected error during setup: {type(e).__name__}: {e}")
        raise
    
    try:
        sim.run()
    except Exception as e:
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(f"\n[ERROR] Simulation failed: {type(e).__name__}: {e}")
        raise
    
    return 0

if __name__ == "__main__":
    sys.exit(main() or 0)
