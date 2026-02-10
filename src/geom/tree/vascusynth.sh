#!/bin/bash
set -e

temp_dir="src/geom/tree/tmp"  # default, overridable via --temp_dir
bind_hpc="/home/juanjo.zuluaga/simulatio.nova"

BIND_PATH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bind)
      BIND_PATH="$2"
      shift 2
      ;;
    --voxel_width)
      VOXEL_WIDTH="$2"
      shift 2
      ;;
    --temp_dir)
      temp_dir="$2"
      shift 2
      ;;
    *)
      echo "Opción desconocida: $1"
      exit 1
      ;;
  esac
done

if [[ -z "$BIND_PATH" ]]; then
  BIND_PATH="$bind_hpc"
elif [[ "$BIND_PATH" == "." ]]; then
  BIND_PATH="$(pwd)"
fi

if [[ ! -d "$BIND_PATH" ]]; then
  echo "El directorio a bindear no existe: $BIND_PATH"
  exit 1
fi

if [[ ! -d "$temp_dir" ]]; then
  echo "temp_dir no existe, hay un problema con src/geom/tree/ , el programa que invoca este script debe crear la carpeta temporal para pasar los archivos de input y leer el output"
  mkdir -p "$temp_dir"
fi

if [[ -z "$VOXEL_WIDTH" ]]; then
  echo "El voxel_width no fue especificado"
  exit 1
fi

if [[ ! -f "$temp_dir/configs.txt" ]]; then
  echo "configs.txt no existe, hay un problema con src/geom/tree/ , el programa que invoca este script debe crear la carpeta temporal para pasar los archivos de input y leer el output"
  exit 1
fi

if [[ ! -f "$temp_dir/output_dir.txt" ]]; then
  echo "output_dir.txt no existe, hay un problema con src/geom/tree/ , el programa que invoca este script debe crear la carpeta temporal para pasar los archivos de input y leer el output"
  exit 1
fi

singularity exec \
    --bind "$BIND_PATH:/work" \
    --pwd "/work/$temp_dir" \
    vascusynth.sif \
    /VascuSynth/VascuSynth configs.txt output_dir.txt $VOXEL_WIDTH