import numpy as np
from .utils import get_timestamp
from .visuals import graficar_resultados
from .kmeans import buscar_k_optimo, kmeans

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
            graficar_resultados(results, k, user_api)
        
    elif tag == 'DBSCAN':
        pass

    fin = get_timestamp()
    print(f"\n===== PIPELINE FINALIZADO {tag} ===== [FIN: {fin}]")

if __name__=='__main__':
    pipeline()