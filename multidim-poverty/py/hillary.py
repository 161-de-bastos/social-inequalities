# %%
import numpy as np
import pandas as pd
import time
import datetime
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from threadpoolctl import threadpool_limits
from multiprocessing import Pool
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")


# %%
# Dataset
df = pd.read_csv("/content/df_multidimensional_2014_2024_clustering.csv")

cols_num = df.drop(columns=["Unnamed: 0", "CODIGO_UNICO_EXT"])
X = cols_num.values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# %%
# Utilidades
def get_timestamp():
    return datetime.datetime.now().strftime('%H:%M:%S')

# KMeans serial
def kmeans_serial(X, k):
    print(f"[{get_timestamp()}] Ejecutando KMeans SERIAL con k={k}")
    start = time.time()
    with threadpool_limits(limits=1, user_api="blas"):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
    duration = time.time() - start
    score = silhouette_score(X, labels)
    print(f"Serial -> Tiempo: {duration:.4f}s | Silhouette: {score:.4f}")
    return ("Serial", duration, score)

# KMeans paralelo
def kmeans_parallel(X, k):
    print(f"[{get_timestamp()}] Ejecutando KMeans PARALELO con k={k}")
    start = time.time()
    with threadpool_limits(limits=None, user_api="blas"):  # usa todos los cores
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
    duration = time.time() - start
    score = silhouette_score(X, labels)
    print(f"Paralelo -> Tiempo: {duration:.4f}s | Silhouette: {score:.4f}")
    return ("Paralelo", duration, score)

# %%
# Buscar número óptimo de clusters (k_opt)
def buscar_k_optimo(X, rango=(2, 10)):
    print("\nBuscando k óptimo con Silhouette...")
    sil_scores = {}
    for k in range(rango[0], rango[1]+1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        sil_scores[k] = silhouette_score(X, labels)
        print(f"k={k} -> Silhouette={sil_scores[k]:.4f}")
    k_opt = max(sil_scores, key=sil_scores.get)
    print(f"\n>>> Mejor número de clusters: k={k_opt} (Silhouette={sil_scores[k_opt]:.4f})")
    return k_opt

# %%
# Comparación secuencial
def comparar_secuencial(X, k):
    serial = kmeans_serial(X, k)
    paralelo = kmeans_parallel(X, k)
    return [serial, paralelo]

# %%
# Visualización
def graficar_resultados(results, k):
    methods = [r[0] for r in results]
    times = [r[1] for r in results]
    silhouettes = [r[2] for r in results]

    fig, ax1 = plt.subplots(figsize=(8,6))
    ax2 = ax1.twinx()

    bars = ax1.bar(methods, times, color='skyblue', alpha=0.7, label="Tiempo (s)")
    line = ax2.plot(methods, silhouettes, color='red', marker='o', label="Silhouette")

    for bar, val in zip(bars, times):
        ax1.text(bar.get_x() + bar.get_width()/2., val + 0.01, f'{val:.3f}s',
                 ha='center', va='bottom')
    for i, val in enumerate(silhouettes):
        ax2.annotate(f'{val:.3f}', (methods[i], silhouettes[i]),
                     textcoords="offset points", xytext=(0,10), ha='center')

    ax1.set_ylabel("Tiempo (s)", color='skyblue')
    ax2.set_ylabel("Silhouette", color='red')
    ax1.set_title(f"Comparación KMeans Serial vs Paralelo (k={k})")

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

# %%
# Pipeline principal
def ejecutar_pipeline():
    print("Dataset Shape:", X_scaled.shape)

    inicio = get_timestamp()
    print(f"===== INICIANDO PIPELINE KMEANS ===== [INICIO: {inicio}]")

    # 1) buscar k óptimo
    k_opt = buscar_k_optimo(X_scaled, rango=(2, 10))

    # 2) correr comparaciones
    results_seq = comparar_secuencial(X_scaled, k_opt)

    # 3) graficar
    graficar_resultados(results_seq, k_opt)

    fin = get_timestamp()
    print(f"\n===== PIPELINE FINALIZADO ===== [FIN: {fin}]")

# %%
if __name__ == "__main__":
    ejecutar_pipeline()