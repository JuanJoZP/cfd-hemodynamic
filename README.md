# CFD Hemodynamic

Simulaciones de dinámica de fluidos computacional (CFD) para aplicaciones hemodinámicas utilizando FEniCSx.

## Tabla de Contenidos

- [Descripción](#descripción)
- [Requisitos](#requisitos)
- [Ejecución Local con Docker](#ejecución-local-con-docker)
- [Ejecución en HPC con SLURM](#ejecución-en-hpc-con-slurm)
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
   scp -r cfd-hemodynamic usuario@cluster:/home/usuario/
   scp fenicsx.sif usuario@cluster:/home/usuario/cfd-hemodynamic/
   ```

3. **Crear directorios de salida en el cluster:**
   ```bash
   mkdir -p ~/data/logs ~/data/results
   ```

### Configuración del job.sh

El archivo `job.sh` está configurado para SLURM. Antes de usarlo, **edita las rutas** según tu entorno:

```bash
#!/bin/bash

#SBATCH --job-name=cfd-hemodynamic
#SBATCH --output=/home/TU_USUARIO/data/logs/output_%j.log
#SBATCH --error=/home/TU_USUARIO/data/logs/error_%j.log
#SBATCH --ntasks=4
#SBATCH --time=12:00:00

mpirun -n $SLURM_NTASKS singularity exec fenicsx.sif python3 main.py --output_dir=/home/TU_USUARIO/data/results $@
```

> **Nota:** Por defecto, el job usa **4 cores** (`--ntasks=4`). Puedes cambiar este valor al enviar el trabajo usando el flag `--ntasks`:

### Enviar un trabajo al cluster

```bash
cd /home/usuario/cfd-hemodynamic

sbatch --ntasks=8 job.sh --simulation dfg_1 --solver stabilized_schur --T 10.0 --dt 0.001 --name hpc_run_01
```

---

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
