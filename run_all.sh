#!/bin/bash
set -e

EXEC="./dbscan_app"

DATASETS=("data/df.csv")
BACKENDS=("brute" "kdtree" "balltree")
MODES=("serial" "omp")
TASKS=("dbscan" "kmeans")

echo "=== DBSCAN + KMEANS EXPERIMENTS ==="

for ds in "${DATASETS[@]}"; do
    for mode in "${MODES[@]}"; do
        for backend in "${BACKENDS[@]}"; do
            for task in "${TASKS[@]}"; do
                LOG_FILE="results_$task-$backend-$mode.log"
                touch "$LOG_FILE"

                # para kmeans el backend no importa, pero lo dejamos por simetría
                echo
                echo "==============================================="
                echo "   Dataset: $ds"
                echo "   Task:    $task"
                echo "   Backend: $backend"
                echo "   Mode:    $mode"
                echo "==============================================="

                # config.yaml para esta corrida
                cat > config.yaml <<EOF
task: $task
backend: $backend

data:
  path: $ds
  start_col: 2

preprocess:
  normalize: true
  skew_thr: 0.5
  kurt_thr: 1.0

parallel:
  mode: $mode
  num_threads: 8

dbscan:
  eps: 2.05
  minPts: 50

kmeans:
  k: 4
  max_iters: 100

metrics:
  enabled: true
  list: "time,memory,silhouette"
  output: $LOG_FILE
EOF

                # correr el binario; él solito:
                # - imprime cosas a stdout
                # - escribe métricas a $METRICS_FILE
                "$EXEC"

                # opcional: limpiar los CSV de clusters si no los necesitas
                rm -f clusters_dbscan.csv clusters_kmeans.csv

            done
        done
    done
done

rm -f config.yaml
