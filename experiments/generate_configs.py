import os

CONFIGS_DIR = os.path.join("configs")

os.makedirs(CONFIGS_DIR, exist_ok=True)

DATA_PATH = "df.csv"
START_COL = 2
NORMALIZE = True
SKEW_THR = 0.5
KURT_THR = 1.0
DBSCAN_EPS = 2.05
DBSCAN_MINPTS = 50
KMEANS_K = 4
KMEANS_MAX_ITERS = 100
METRICS_ENABLED = True
METRICS_LIST = "time,memory"
METRICS_OUTPUT = "results.log"

OMP_THREADS = [2, 4, 8, 12, 16]

def make_text(task, backend, mode, num_threads):
    if mode == "serial":
        num_threads = 1

    yaml = f"""
task: {task}
backend: {backend}

data:
    path: {DATA_PATH}
    start_col: {START_COL}

preprocess:
    normalize: {"true" if NORMALIZE else "false"}
    skew_thr: {SKEW_THR}
    kurt_thr: {KURT_THR}

parallel:
    mode: {mode}
    num_threads: {num_threads}

dbscan:
    eps: {DBSCAN_EPS}
    minPts: {DBSCAN_MINPTS}

kmeans:
    k: {KMEANS_K}
    max_iters: {KMEANS_MAX_ITERS}

metrics:
    enabled: {"true" if METRICS_ENABLED else "false"}
    list: "{METRICS_LIST}"
    output: {METRICS_OUTPUT}
    """
    return yaml

def write_config(filename, text):
    path = os.path.join(CONFIGS_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def generate_dbscan():
    backends = ["brute", "kdtree", "balltree"]
    for backend in backends:
        # serial
        name = f"exp-dbscan-{backend}-serial.yaml"
        txt = make_text(task="dbscan", backend=backend,
                               mode="serial", num_threads=1)
        write_config(name, txt)

        # omp con distintos hilos
        for th in OMP_THREADS:
            name = f"exp-dbscan-{backend}-omp{th}.yaml"
            txt = make_text(task="dbscan", backend=backend,
                                   mode="omp", num_threads=th)
            write_config(name, txt)

def generate_kmeans():
    backend = "brute"

    # serial
    name = "exp-kmeans-serial.yaml"
    txt = make_text(task="kmeans", backend=backend,
                           mode="serial", num_threads=1)
    write_config(name, txt)

    # omp con distintos hilos
    for th in OMP_THREADS:
        name = f"exp-kmeans-omp{th}.yaml"
        txt = make_text(task="kmeans", backend=backend,
                               mode="omp", num_threads=th)
        write_config(name, txt)

def main():
    generate_dbscan()
    generate_kmeans()

if __name__ == "__main__":
    main()