import time
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
from .utils import get_timestamp

# KMeans serial
def kmeans_serial(X, k):
    print(f"[{get_timestamp()}] Ejecutando KMeans SERIAL con k={k}")
    start = time.time()
    with threadpool_limits(limits=1, user_api="blas"):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
    duration = time.time() - start
    score = silhouette_score(X, labels)
    print(f"Serial -> Tiempo: {duration:.4f}s | Silhouette: {score:.4f}\n")
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
    print(f"Paralelo -> Tiempo: {duration:.4f}s | Silhouette: {score:.4f}\n")
    return ("Paralelo", duration, score)

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
    print(f"\n>>> Mejor número de clusters: k={k_opt} (Silhouette={sil_scores[k_opt]:.4f})\n")
    return k_opt