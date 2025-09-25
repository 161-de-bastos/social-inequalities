import numpy as np
from .trials.utils import get_timestamp
from .trials.visuals import graficar_kmeans, graficar_dbscan
from .trials.kmeans import buscar_k_optimo, kmeans
from .trials.dbscan import dbscan

def pipeline(X: np.ndarray, tag: str):
    tag = tag.upper()
    assert tag in ['KMEANS','DBSCAN'] 
    print("Dataset Shape:", X.shape)

    inicio = get_timestamp()
    print(f"===== INICIANDO PIPELINE {tag} ===== [INICIO: {inicio}]")

    if tag == 'KMEANS':
        k = buscar_k_optimo(X)

        for user_api in ['blas', 'openmp']:
            results = kmeans(X, k, user_api)
            graficar_kmeans(results, k, user_api)
        
    elif tag == 'DBSCAN':
        eps = 0.5
        min_samples = 5
        
        for user_api in ['blas', 'openmp']:
            results = dbscan(X, eps=eps, min_samples=min_samples, user_api=user_api)
            graficar_dbscan(results, eps, min_samples, user_api)

    fin = get_timestamp()
    print(f"\n===== PIPELINE FINALIZADO {tag} ===== [FIN: {fin}]")

if __name__=='__main__':
    pipeline()