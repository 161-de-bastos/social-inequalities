import numpy as np
from .utils import get_timestamp
from .visuals import graficar_resultados
from .kmeans import kmeans_serial, kmeans_parallel, buscar_k_optimo

def kmeans(X: np.ndarray):
    # 1) buscar k óptimo
    k = buscar_k_optimo(X,rango=(2,10))

    # 2) correr comparaciones
    serial = kmeans_serial(X, k)
    paralelo = kmeans_parallel(X, k)

    return [serial, paralelo], k

def pipeline(X: np.ndarray, tag: str):
    tag = tag.upper()
    assert tag in ['KMEANS','DBSCAN'] 
    print("Dataset Shape:", X.shape)

    inicio = get_timestamp()
    print(f"===== INICIANDO PIPELINE {tag} ===== [INICIO: {inicio}]")

    if tag == 'KMEANS':
        results, k = kmeans(X)
        
    elif tag == 'DBSCAN':
        pass

    fin = get_timestamp()
    print(f"\n===== PIPELINE FINALIZADO {tag} ===== [FIN: {fin}]")

    graficar_resultados(results,k)