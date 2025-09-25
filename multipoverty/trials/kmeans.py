import pandas as pd
import time
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
from .visuals import plot_speedup
from .utils import get_timestamp
from multiprocessing import Pool

# Restricción de KMeans en threads
def timed_kmeans(X, k: int, user_api: str, parallel: bool, getscore: bool = True):
    limits = None if parallel else 1
    tag = 'Paralelo' if parallel else 'Serial'

    print(f"[{get_timestamp()}] Ejecutando KMeans {tag} con k={k}, bloqueando {user_api}")
    start = time.time()

    with threadpool_limits(limits = limits, user_api = user_api):
        km = KMeans(n_clusters = k, random_state = 42, n_init = 10)
        labels = km.fit_predict(X)

    duration = time.time() - start
    print(f"{tag} -> Tiempo: {duration:.4f}s")

    if getscore:
        score = silhouette_score(X, labels)
        print(f"Silhouette: {score:.4f}\n")
        return (tag, duration, score)
    else:
        return duration

def speedup_k(X, ks=range(2,11), user_apis = ['blas','openmp']):
    results = [] 
    for user_api in user_apis:
        for k in ks:
            result = {
                'k': k,
                'user_api': user_api,
                'serial_time': timed_kmeans(X, k, user_api=user_api, parallel=False, getscore=False),
                'parallel_time': timed_kmeans(X, k, user_api=user_api, parallel=True, getscore=False)
            }
            results.append(result)
    results = pd.DataFrame(results)
    results['speedup'] = results['serial_time'] / results['parallel_time']

    plot_speedup(results)

# Scoring to be paralellized
def ksearch(X,k):
        km = KMeans(n_clusters = k, random_state = 42, n_init = 10)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels)
        print(f"k={k} -> Silhouette={score:.4f}")
        return score

# Buscar número óptimo de clusters (k_opt)
def buscar_k_optimo(X, rango = range(2,11), cores = 4):    
    with Pool(processes = cores) as p:
        sil_scores = dict(zip(
            rango,
            p.starmap(ksearch, [(X, i) for i in rango])
        ))

    k_opt = max(sil_scores, key = sil_scores.get)
    print(f"\n>>> Mejor número de clusters: k={k_opt} (Silhouette={sil_scores[k_opt]:.4f})\n")
    return k_opt

def kmeans(X, k: int, user_api: str):
    serial = timed_kmeans(X, k, user_api, False)
    paralelo = timed_kmeans(X, k, user_api, True)
    return [serial, paralelo]