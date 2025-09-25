import time
from threadpoolctl import threadpool_limits
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from .utils import get_timestamp

def safe_silhouette(X, labels):
    uniq = set(labels)
    if len([u for u in uniq if u != -1]) < 2:
        return float("nan")
    try:
        return float(silhouette_score(X, labels))
    except Exception:
        return float("nan")

def timed_dbscan(X, eps: float, min_samples: int, user_api: str = None, p_parallel: bool = False, t_parallel: bool = False, getscore: bool = True):
    assert not (t_parallel and p_parallel)
    limits = None if t_parallel else 1
    tag = 'Paralelo de alto nivel' if p_parallel else 'Paralelo de bajo nivel' if t_parallel else 'Serial'

    print(f"[{get_timestamp()}] Ejecutando DBSCAN {tag} con eps={eps},min_samples={min_samples}")
    start = time.time()

    if t_parallel:
        with threadpool_limits(limits = limits, user_api = user_api):
            db = DBSCAN(eps = eps, min_samples = min_samples)
            labels = db.fit_predict(X)

    else:
        if p_parallel:
            db = DBSCAN(eps = eps, min_samples = min_samples, n_jobs = 4)
        else:
            db = DBSCAN(eps = eps, min_samples = min_samples, n_jobs = 1)
        labels = db.fit_predict(X)

    duration = time.time() - start
    print(f"{tag} -> Tiempo: {duration:.4f}s")

    if getscore:
        score = safe_silhouette(X, labels)
        print(f"Silhouette: {score:.4f}\n")
        return (tag, duration, score)
    else:
        return duration

def dbscan(X, eps: float, min_samples: int, user_api: str):
    serial = timed_dbscan(X, eps, min_samples, user_api=user_api, p_parallel=False, t_parallel=False)
    low_level = timed_dbscan(X, eps, min_samples, user_api=user_api, p_parallel=False, t_parallel=True)
    high_level = timed_dbscan(X, eps, min_samples, user_api=user_api, p_parallel=True, t_parallel=False)
    return [serial, low_level, high_level]