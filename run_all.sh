#!/bin/bash
#set -e

EXEC="./dbscan_app"

DATASETS=("data/test.csv")
BACKENDS=("brute" "kdtree" "balltree")
MODES=("serial" "omp")
TASKS=("dbscan" "kmeans")

METRICS_FILE="experiments.log"

echo "=== DBSCAN + KMEANS EXPERIMENTS ==="
echo "Log: $METRICS_FILE"

# limpiar log anterior
: > "$METRICS_FILE"

for ds in "${DATASETS[@]}"; do
    for mode in "${MODES[@]}"; do
        for backend in "${BACKENDS[@]}"; do
            for task in "${TASKS[@]}"; do

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
  start_col: 0

preprocess:
  normalize: true
  skew_thr: 0.5
  kurt_thr: 1.0

parallel:
  mode: $mode
  num_threads: 8

dbscan:
  eps: 0.5
  minPts: 5

kmeans:
  k: 4
  max_iters: 100

metrics:
  enabled: true
  list: "time,memory,silhouette"
  output: "$METRICS_FILE"
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

echo
echo "=== EXPERIMENTOS COMPLETADOS. Revisa: $METRICS_FILE ==="
