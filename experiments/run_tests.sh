#!/bin/bash
#set -e

echo "=== DBSCAN FRAMEWORK TEST SUITE ==="

DATASETS=("data/test.csv")
BACKENDS=("brute" "kdtree" "balltree")
MODES=("serial" "omp")

LOGFILE="results.log"
echo "==== RESULTADOS DE PRUEBAS ====" > "$LOGFILE"
echo "Fecha: $(date)" >> "$LOGFILE"
echo "" >> "$LOGFILE"
TIME_CMD="/usr/bin/time"

for ds in "${DATASETS[@]}"; do
    for be in "${BACKENDS[@]}"; do
        for mode in "${MODES[@]}"; do

            echo
            echo "==============================================="
            echo "   Dataset: $ds"
            echo "   Backend: $be"
            echo "   Modo:    $mode"
            echo "==============================================="

            cat > config.yaml <<EOF
data:
  path: "$ds"
  start_col: 0

backend: $be

parallel:
  mode: $mode
  num_threads: 4

dbscan:
  eps: 0.3
  minPts: 1
EOF

            echo "[INFO] Ejecutando..."
            $TIME_CMD -f "REAL=%e MEM=%M" -o .tmp_time ./dbscan_app > .tmp_output 2>&1

            METRIC_OUTPUT=$(cat .tmp_time)
            # Ejemplo: "REAL=0.00 MEM=4332"
            REAL_TIME=$(echo "$METRIC_OUTPUT" | sed 's/.*REAL=\([^ ]*\).*/\1/')
            MEM_KB=$(echo "$METRIC_OUTPUT" | sed 's/.*MEM=\([^ ]*\).*/\1/')

            echo "[OK] Tiempo real: ${REAL_TIME}s"
            echo "[OK] Memoria máx: ${MEM_KB} KB"
            echo

            # Escribir al log consolidado
            {
                echo "DATASET=$ds"
                echo "BACKEND=$be"
                echo "MODE=$mode"
                echo "TIME=$REAL_TIME"
                echo "MEMORY_KB=$MEM_KB"

                # Extraer clusters encontrados del archivo clusters.csv
                if [ -f clusters.csv ]; then
                    N_POINTS=$(( $(wc -l < clusters.csv) - 1 ))
                    N_NOISE=$(awk -F',' 'NR>1 && $2==-1 {c++} END{print c+0}' clusters.csv)
                    N_CLUSTERS=$(awk -F',' 'NR>1 && $2!=-1 {seen[$2]=1} END{print length(seen)}' clusters.csv)

                    echo "CLUSTERS=$N_CLUSTERS"
                    echo "NOISE=$N_NOISE"
                    echo "POINTS=$N_POINTS"
                else
                    echo "CLUSTERS=ERROR"
                    echo "NOISE=ERROR"
                    echo "POINTS=ERROR"
                fi

                echo "-----"
            } >> "$LOGFILE"

            cat .tmp_output
        done
    done
done

rm .tmp_output .tmp_time clusters.csv config.yaml