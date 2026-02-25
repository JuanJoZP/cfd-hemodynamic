#!/bin/bash
# Script para empaquetar y copiar archivos al HPC

set -e

# Configuración
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
PEM_FILE="$HOME/Documents/juanjo.zuluaga.pem"
HPC_HOST="loginpub-hpc.urosario.edu.co"
HPC_PORT="53841"
HPC_USER="juanjo.zuluaga"
HPC_DEST="/home/juanjo.zuluaga/simulatio.nova"
OUTPUT_FILE="hpc_package.tar.gz"

cd "$PROJECT_DIR"

# Verificar que existe el archivo PEM
if [ ! -f "$PEM_FILE" ]; then
    echo "❌ Error: No se encontró el archivo PEM en: $PEM_FILE"
    exit 1
fi

echo "📦 Empaquetando archivos para HPC..."

# Crear el archivo tar.gz con los archivos necesarios
tar -czvf "$OUTPUT_FILE" \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.git' \
    --exclude='.venv' \
    --exclude='.devcontainer' \
    --exclude='results' \
    --exclude='*.sif' \
    src/ \
    *.py \

echo ""
echo "✅ Archivo creado: $OUTPUT_FILE"
echo "   Tamaño: $(du -h "$OUTPUT_FILE" | cut -f1)"
echo ""

echo "📤 Copiando al HPC..."
scp -i "$PEM_FILE" -P "$HPC_PORT" "$PROJECT_DIR/$OUTPUT_FILE" "$HPC_USER@$HPC_HOST:$HPC_DEST/"

echo ""
echo "✅ Archivo copiado exitosamente a: $HPC_DEST/"
echo ""
echo "🖥️  En el HPC, descomprime con:"
echo ""
echo "   cd $HPC_DEST"
echo "   tar -xzvf hpc_package.tar.gz"
echo ""
echo "📤 Si necesitas copiar el contenedor Singularity, ejecuta:"
echo ""
echo "   scp -i $PEM_FILE -P $HPC_PORT $PROJECT_DIR/fenicsx.sif $HPC_USER@$HPC_HOST:$HPC_DEST/"
