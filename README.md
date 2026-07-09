# CFD Hemodynamic

Simulaciones de dinámica de fluidos computacional (CFD) para aplicaciones hemodinámicas utilizando FEniCSx.

## Tabla de Contenidos

- [Descripción](#descripción)
- [Requisitos](#requisitos)
- [Ejecución Local con Docker](#ejecución-local-con-docker)
- [Ejecución en HPC con SLURM](#ejecución-en-hpc-con-slurm)
- [Escenarios](#escenarios-srcscenarios)
- [Solvers](#solvers-srcsolvers)
- [Uso del CLI](#uso-del-cli)

## Descripción

Este proyecto proporciona un framework para ejecutar simulaciones CFD hemodinámicas utilizando la biblioteca FEniCSx. Soporta múltiples escenarios de simulación y solvers, con capacidad de ejecución tanto en entornos locales (Docker) como en clusters HPC.

## Requisitos

### Para ejecución local

- [Docker](https://www.docker.com/) instalado

### Para ejecución en HPC

- Acceso a un cluster con [SLURM](https://slurm.schedmd.com/)
- [Singularity](https://sylabs.io/singularity/) instalado en el cluster
- Imagen de Singularity `fenicsx.sif` (ver sección de construcción)

---

## Ejecución Local con Docker

1. **Construir la imagen Docker:**

   ```bash
   docker build -t cfd-hemodynamic .
   ```

2. **Ejecutar el contenedor de forma interactiva:**

   ```bash
   docker run -it --rm \
     -v $(pwd):/app \
     -w /app \
     cfd-hemodynamic bash
   ```

3. **Dentro del contenedor, ejecutar la simulación:**
   ```bash
   python main.py --simulation dfg_1 --solver stabilized_schur --T 1.0 --dt 0.01 --name test_run
   ```

### Ejecución con MPI (múltiples procesos)

Para ejecutar con MPI dentro de Docker:

```bash
docker run --rm \
  -v $(pwd):/app \
  -v $(pwd)/results:/app/results \
  -w /app \
  cfd-hemodynamic \
  mpirun -n 4 python main.py --simulation dfg_1 --solver stabilized_schur --T 1.0 --dt 0.01 --name mpi_test
```

---

## Ejecución en HPC con SLURM

### Prerrequisitos

1. **Construir la imagen de Singularity** (si no existe `fenicsx.sif`):

   ```bash
   singularity build fenicsx.sif singularity.def
   ```

2. **Transferir archivos al cluster:**

   ```bash
   scp -r cfd-hemodynamic usuario@cluster:~/
   scp fenicsx.sif usuario@cluster:~/
   ```

3. **Crear directorios de salida en el cluster:**
   ```bash
   mkdir -p ~/data/logs ~/data/results
   ```

### Configuración de `src/simulation_hpc.sh`

El archivo `src/simulation_hpc.sh` está configurado para SLURM. Antes de usarlo, **edita las siguientes variables** según tu entorno:

| Variable | Descripción |
|---|---|
| `image` | Ruta a la imagen `fenicsx.sif` construida con `singularity.def` |
| `mpich` | Ruta al ejecutable `mpirun` del cluster (ver `which mpirun`) |
| `--mail-user` | Tu correo para notificaciones SLURM |

El script asume la siguiente estructura de directorios en tu `$HOME` del cluster:

```
~/fenicsx.sif              # Imagen de Singularity (construida con singularity build fenicsx.sif singularity.def)
~/data/logs/               # Logs de SLURM (crear con mkdir -p ~/data/logs)
~/data/results/            # Resultados de simulación (crear con mkdir -p ~/data/results)
```

> **Nota:** Por defecto, el job usa **4 cores** (`--ntasks=4`). Puedes cambiar este valor al enviar el trabajo usando el flag `--ntasks`.

### Enviar un trabajo al cluster

```bash
sbatch src/simulation_hpc.sh --simulation dfg_1 --solver stabilized_schur --T 10.0 --dt 0.001 --name hpc_run_01
```

---

## Escenarios (`src/scenarios/`)

| Escenario | Descripción |
|---|---|
| `dfg_1` | Flow around a cylinder (DFG 2D-1 benchmark, Re=20). Malla generada con Gmsh, inlet parabólico, paredes no-slip, outlet do-nothing. |
| `dfg_2d_1` | Variante estacionaria del DFG 2D-1 benchmark con inlet parabólico. |
| `lid_driven2D` | Cavidad cuadrada con tapa móvil (lid-driven cavity). Útil para validación de códigos Navier-Stokes. |
| `pipe_cylinder` | Tubería 2D con obstáculo cilíndrico, inlet parabólico, similar a DFG pero con geometría paramétrica. |
| `pipe_cylinder_pressurebc` | Misma geometría que `pipe_cylinder` pero con condición de presión débil en inlet en vez de velocidad. |
| `simple_bifurcation` | Bifurcación microvascular simple (un inlet, dos outlets). |
| `stenosis` | Estenosis (estrechamiento) en canal 2D con severidad configurable (mild/moderate/severe). Paredes Bezier, inlet parabólico. |
| `stenosis_mesh_variable` | Misma estenosis que `stenosis` pero permite refinar la malla selectivamente (mesh size variable). |
| `stenosis_pressure` | Estenosis con inlet por presión débil (p_inlet) + condición tangencial Nitsche y outlet con resistencia (p = R·Q) + backflow stabilization. |
| `stenosis_pressure_structured` | Mismo caso que `stenosis_pressure` con malla estructurada transfinita para simetría radial perfecta. |
| `stenosis_with_tree` | Estenosis 2D con árbol vascular generado por VascuSynth acoplado al outlet. |
| `stenosis_with_tree_2d` | Estenosis + árbol vascular 2D generado proceduralmente (pure Python, Murray's law, sin VascuSynth). |
| `stenosis_with_tree_2d_pressure` | Mismo caso que `stenosis_with_tree_2d` pero con inlet por presión débil en vez de velocidad. |
| `taylor_green` | Vórtice de Taylor-Green 3D (solución analítica conocida, decaimiento exponencial). Validación de convergencia. |
| `unit_cube_pipe` | Flujo 3D pressure-driven en ducto rectangular con malla hexaédrica. |
| `unit_square_pipe` | Flujo 2D pressure-driven en canal rectangular con malla quadrilateral. |
| `unit_square` | Cuadrado unitario con condiciones Dirichlet simples. Caso mínimo para debugging. |
| `vascular_tree` | Árbol vascular (microvasculatura) 3D generado con Gmsh. |

## Solvers (`src/solvers/`)

Los solvers comparten el núcleo `stabilized_schur` (SUPG/PSPG/LSIC, Newton, Schur fieldsplit). Las variantes modifican condiciones de frontera, integración temporal o el precondicionador:

### Familia `stabilized_schur`

| Solver | Variante |
|---|---|
| `stabilized_schur` | Solver base: Euler implícito, inlet Dirichlet fuerte, outlet do-nothing. |
| `stabilized_schur_backflow` | + estabilización de backflow en outlet (Moghadam et al. 2011). |
| `stabilized_schur_bdf2` | + integración temporal BDF2 (segundo orden). |
| `stabilized_schur_adaptive` | + paso de tiempo adaptativo. |
| `stabilized_schur_aspin` | + ASPIN (Additive Schwarz preconditioned inexact Newton) para escalabilidad en paralelo. |
| `stabilized_schur_pressurebc` | Formulación curl-curl con condición de presión natural en inlet + Nitsche tangencial. |
| `stabilized_schur_pressure_backflow` | Inlet presión débil + outlet resistencia (R·Q) + backflow stabilization. |
| `stabilized_schur_ramping` | + rampa en parámetros (Re, etc.) para arranque suave. |
| `stabilized_schur_stokes` | Stokes (sin término convectivo). |
| `stabilized_schur_velocity_vascular_backflow` | Inlet Dirichlet velocidad + outlet resistencia (R·Q) + backflow. |

### Familia vascular BC (formulación curl-curl, Euler implícito)

| Solver | Condición de outlet |
|---|---|
| `stabilized_schur_vascularbc` | Resistencia R·Q con iteración de punto fijo. |
| `stabilized_schur_vascularbc_backflow` | Resistencia + backflow stabilization, presión FFR fija. |
| `stabilized_schur_vascularbc_cbc` | Convective BC (CBC), outlet do-nothing con estabilización convectiva. |
| `stabilized_schur_vascularbc_ddn` | Directional do-nothing (DDN), activa solo en backflow. |
| `stabilized_schur_vascularbc_strong` | Dirichlet fuerte de presión en outlet (R·Q). |
| `stabilized_schur_vascularbc_weak` | Presión débil en outlet (R·Q), sin Nitsche. |

### Familia LSC (precondicionador Least Squares Commutator)

| Solver | Variante |
|---|---|
| `stabilized_lsc` | Base con precondicionador LSC en vez de Schur. |
| `stabilized_lsc_bdf2` | LSC + BDF2. |
| `stabilized_lsc_pressurebc` | LSC + curl-curl + presión natural inlet. |

### Familia PCD (precondicionador Pressure Convection-Diffusion)

| Solver | Variante |
|---|---|
| `stabilized_pcd` | Base con precondicionador PCD (fenicsx-pctools). |
| `stabilized_pcd_bdf2` | PCD + BDF2. |
| `stabilized_pcd_pressurebc` | PCD + curl-curl + presión natural inlet. |

### Otros

| Solver | Descripción |
|---|---|
| `stabilized_staggered` | Aproximación segregada (staggered) en vez de monolítica. |
| `ipcs_midpoint` | IPCS (Incremental Pressure Correction Scheme) con método del punto medio. |
| `ipcs_bdf2` | IPCS con extrapolación BDF2 del término convectivo. |
| `dfg_2d_1` | Solver dedicado para el benchmark DFG 2D-1 estacionario (basado en `stabilized_schur_pressure_backflow`). |

## Uso del CLI

El script `main.py` acepta los siguientes argumentos:

```
Argumentos requeridos:
  --simulation    Nombre del escenario (e.g., dfg_1)
  --solver        Nombre del solver (e.g., stabilized_schur)
  --T             Tiempo total de simulación
  --dt            Paso de tiempo
  --name          Nombre del run/experimento

Argumentos opcionales:
  --output_dir    Directorio de salida (default: results)
  --mu            Viscosidad (usa valor por defecto del escenario si no se especifica)
  --rho           Densidad (usa valor por defecto del escenario si no se especifica)
```
