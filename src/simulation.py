import inspect
import os
import subprocess
from datetime import datetime
from importlib import import_module
from pathlib import Path
from typing import Any, Optional, Type, Union

from mpi4py import MPI

from src.scenario import Scenario


class Simulation:
    def __init__(
        self,
        name: str,
        simulation: Union[str, Type[Scenario]],
        solver: str,
        T: Union[float, str],
        dt: Union[float, str],
        output_dir: Union[str, Path],
        **kwargs: Any,
    ):
        """
        Orchestrates the simulation execution.

        Args:
            name (str): Name of the specific run
            simulation (str): Name of the scenario module to load (e.g. 'dfg_1')
            solver (str): Name of the solver to use (e.g. 'stabilized_schur')
            T (float): Total simulation time
            dt (float): Time step
            output_dir (str): Base output directory
            **kwargs: Additional arguments for the scenario (e.g. mu, rho, f)

        Raises:
            ValueError: If arguments are invalid
            ImportError: If scenario module cannot be loaded
        """
        # Validate required arguments
        if not name or not isinstance(name, str):
            raise ValueError("'name' must be a non-empty string.")
        if not simulation:
            raise ValueError(
                "'simulation' must be provided (string or Scenario class)."
            )
        if not solver or not isinstance(solver, str):
            raise ValueError(
                "'solver' must be a non-empty string specifying the solver module."
            )

        self.name = name
        self.scenario_name = (
            simulation
            if isinstance(simulation, str)
            else getattr(simulation, "__name__", "custom_scenario")
        )
        self.solver_name = solver
        self.output_dir = output_dir
        self.kwargs = kwargs

        # If simulation is a class (Scenario subclass), store it
        self._scenario_class_override = None
        if inspect.isclass(simulation) and issubclass(simulation, Scenario):
            self._scenario_class_override = simulation

        # Validate and convert T and dt
        try:
            self.T = float(T)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"'T' (total time) must be a valid number, got: {T!r}"
            ) from e

        try:
            self.dt = float(dt)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"'dt' (time step) must be a valid number, got: {dt!r}"
            ) from e

        if self.T <= 0:
            raise ValueError(f"'T' (total time) must be positive, got: {self.T}")
        if self.dt <= 0:
            raise ValueError(f"'dt' (time step) must be positive, got: {self.dt}")
        if self.dt > self.T:
            raise ValueError(f"'dt' ({self.dt}) cannot be greater than 'T' ({self.T})")

        self.mu = kwargs.get("mu")
        self.rho = kwargs.get("rho")

        self.scenario_instance = self._load_scenario()

    def _load_scenario(self) -> Scenario:
        """Load and instantiate the scenario class.

        Returns:
            Scenario: An instance of the requested scenario

        Raises:
            ImportError: If the scenario module cannot be imported
            ValueError: If no Scenario subclass is found or required parameters are missing
            RuntimeError: If scenario instantiation fails
        """
        scenario_class = self._scenario_class_override

        if not scenario_class:
            # Import the scenario module
            try:
                module = import_module(f"src.scenarios.{self.scenario_name}")
            except ImportError as e:
                available = self._list_available_scenarios()
                raise ImportError(
                    f"Could not import scenario '{self.scenario_name}'. "
                    f"Ensure src/scenarios/{self.scenario_name}.py exists.\n"
                    f"Available scenarios: {available}"
                ) from e
            except SyntaxError as e:
                raise SyntaxError(
                    f"Syntax error in scenario module 'src/scenarios/{self.scenario_name}.py': {e}"
                ) from e

            # Find the Scenario subclass
            for member_name, obj in inspect.getmembers(module):
                if (
                    inspect.isclass(obj)
                    and issubclass(obj, Scenario)
                    and obj is not Scenario
                ):
                    scenario_class = obj
                    break

        if not scenario_class:
            raise ValueError(
                f"No Scenario subclass found for '{self.scenario_name}'. "
                f"Ensure the module exists and defines a class that inherits from Scenario."
            )

        # Prepare arguments for the scenario constructor
        sig = inspect.signature(scenario_class.__init__)

        # Default available params (always provided)
        available_params = {
            "solver_name": self.solver_name,
            "dt": self.dt,
            "T": self.T,
        }
        # Merge kwargs (may contain mu, rho, f, etc.)
        available_params.update(self.kwargs)

        # Build valid params dict and check for missing required params
        init_args = {}
        missing_params = []
        has_var_keyword = False  # Does the scenario accept **kwargs?

        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            # Check if scenario accepts **kwargs
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                has_var_keyword = True
                continue
            if param_name in available_params:
                init_args[param_name] = available_params[param_name]
            elif param.default == inspect.Parameter.empty:
                missing_params.append(param_name)

        if missing_params:
            raise ValueError(
                f"Missing required parameter(s) for scenario '{self.scenario_name}': {missing_params}. "
                f"Pass them via command line arguments, e.g.: "
                + " ".join(f"--{p}=<value>" for p in missing_params)
            )

        # If scenario accepts **kwargs, pass any extra params not already matched
        if has_var_keyword:
            for key, value in available_params.items():
                if key not in init_args:
                    init_args[key] = value

        # Instantiate the scenario
        try:
            return scenario_class(**init_args)
        except TypeError as e:
            raise RuntimeError(
                f"Failed to instantiate scenario '{self.scenario_name}' with arguments {init_args}: {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Error while initializing scenario '{self.scenario_name}': {type(e).__name__}: {e}"
            ) from e

    def _list_available_scenarios(self) -> list[str]:
        """List available scenario modules."""
        import os

        scenarios_dir = os.path.join(os.path.dirname(__file__), "scenarios")
        try:
            files = os.listdir(scenarios_dir)
            scenarios = [
                f[:-3] for f in files if f.endswith(".py") and not f.startswith("_")
            ]
            return scenarios if scenarios else ["(none found)"]
        except OSError:
            return ["(could not list)"]

    def run(self, save_path: Optional[Union[str, Path]] = None) -> str:
        # Create output path structure: <output_dir>/<scenario>/<datetime>_<name>/
        # datetime-without-tz format example: 2023-01-01T12:00:00
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        if save_path is None:
            timestamp = datetime.now().strftime("%Y-%m-%dT%H.%M.%S")
            folder_name = f"{timestamp}_{self.name}"
            save_path = os.path.abspath(
                os.path.join(self.output_dir, self.scenario_name, folder_name)
            )
        else:
            save_path = os.path.abspath(save_path)

        if rank == 0:
            os.makedirs(save_path, exist_ok=True)

            # Save simulation_params.txt
            with open(os.path.join(save_path, "simulation_params.txt"), "w") as f:
                f.write(f"Scenario: {self.scenario_name}\n")
                f.write(f"Run Name: {self.name}\n")
                f.write(f"Solver: {self.solver_name}\n")
                if self.mu is not None:
                    f.write(f"mu: {self.mu}\n")
                else:
                    f.write("mu: (scenario default)\n")
                if self.rho is not None:
                    f.write(f"rho: {self.rho}\n")
                else:
                    f.write("rho: (scenario default)\n")
                f.write(f"T: {self.T}\n")
                f.write(f"dt: {self.dt}\n")
                for k, v in self.kwargs.items():
                    if k not in ("mu", "rho"):  # Already logged above
                        f.write(f"{k}: {v}\n")

                try:
                    commit_id = (
                        subprocess.check_output(
                            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
                        )
                        .strip()
                        .decode("utf-8")
                    )
                    f.write(f"Source Code Version (Git Commit): {commit_id}\n")
                except:
                    f.write("Source Code Version: Unknown (git not valid)\n")

            print(
                f"Initializing simulation '{self.name}' with scenario '{self.scenario_name}'..."
            )

        comm.barrier()

        self.scenario_instance.setup()
        result_path = self.scenario_instance.solve(output_folder=save_path)

        if rank == 0:
            print(f"Simulation completed. Results saved to: {result_path}")

        return result_path
