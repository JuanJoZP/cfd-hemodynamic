#!/bin/bash
# extract_convergence_data.sh
# Extract convergence metrics from results folders and log files.

RESULTS_DIR=$1
LOGS_DIR=${2:-$HOME/data/logs}
OUTPUT_CSV="./convergence_data.csv"

if [ -z "$RESULTS_DIR" ]; then
    echo "Uso: $0 <results_dir> [logs_dir]"
    echo "Ejemplo: $0 results/dfg_1 ~/data/logs"
    exit 1
fi

echo "folder,dofs,l2_velocity,l2_pressure,cd,cl" > "$OUTPUT_CSV"

echo "Buscando datos en $RESULTS_DIR..."
found_count=0

# Iterar sobre las carpetas de resultados
for folder_path in "$RESULTS_DIR"/*; do
    if [ -d "$folder_path" ]; then
        folder_name=$(basename "$folder_path")
        norms_file="$folder_path/norms.txt"
        
        if [ -f "$norms_file" ]; then
            # Extraer normas L2
            l2_vel=$(grep "L2 norm of velocity:" "$norms_file" | awk '{print $NF}')
            l2_pres=$(grep "L2 norm of pressure:" "$norms_file" | awk '{print $NF}')
            
            # Buscar el archivo de log correspondiente
            # Se busca un log que mencione que los resultados se guardaron en esta carpeta
            log_file=$(grep -l "Results saved to:.*$folder_name" "$LOGS_DIR"/output_*.log 2>/dev/null | head -n 1)
            
            if [ -n "$log_file" ]; then
                # Extraer DOFs (ej: DOFs: 36918 (Velocity: 24612, Pressure: 12306))
                dofs=$(grep "DOFs:" "$log_file" | head -n 1 | awk '{print $2}')
                
                # Extraer coeficientes (ej: Cd=-4.111041, Cl=-0.000901...)
                # Usamos awk para mayor robustez ante variaciones en el formato
                metrics_line=$(grep "Cd=" "$log_file" | tail -n 1)
                cd_val=$(echo "$metrics_line" | sed -n 's/.*Cd=\([-0-9.]*\).*/\1/p')
                cl_val=$(echo "$metrics_line" | sed -n 's/.*Cl=\([-0-9.]*\).*/\1/p')
                
                # Si fallan los scripts de sed (ej: por notación científica), probamos con awk
                if [ -z "$cd_val" ]; then
                    cd_val=$(echo "$metrics_line" | awk -F'Cd=' '{print $2}' | awk -F'[, ]' '{print $1}')
                fi
                if [ -z "$cl_val" ]; then
                    cl_val=$(echo "$metrics_line" | awk -F'Cl=' '{print $2}' | awk -F'[, ]' '{print $1}')
                fi

                echo "$folder_name,$dofs,$l2_vel,$l2_pres,$cd_val,$cl_val" >> "$OUTPUT_CSV"
                echo "[OK] Procesado: $folder_name (DOFs: $dofs)"
                ((found_count++))
            else
                echo "[WARN] No se encontró log para $folder_name en $LOGS_DIR"
            fi
        fi
    fi
done

echo "Finalizado. Se procesaron $found_count carpetas."
echo "Datos guardados en $OUTPUT_CSV"
