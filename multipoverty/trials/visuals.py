import matplotlib.pyplot as plt

def graficar_dbscan(results, eps, min_samples, user_api=None):
    methods = [r[0] for r in results]
    times = [r[1] for r in results]
    silhouettes = [r[2] for r in results]

    fig, ax1 = plt.subplots(figsize=(16,12))
    ax2 = ax1.twinx()

    bars = ax1.bar(methods, times, alpha=0.7, label="Tiempo (s)")
    line = ax2.plot(methods, silhouettes, marker='o', label="Silhouette")

    for bar, val in zip(bars, times):
        ax1.text(bar.get_x() + bar.get_width()/2., val + 0.01, f'{val:.3f}s',
                 ha='center', va='bottom')
    for i, val in enumerate(silhouettes):
        ax2.annotate(f'{val:.3f}' if val==val else 'N/A', (methods[i], silhouettes[i] if silhouettes[i]==silhouettes[i] else 0),
                     textcoords="offset points", xytext=(0,10), ha='center')

    ax1.set_ylabel("Tiempo (s)")
    ax2.set_ylabel("Silhouette")
    title = f"Comparación DBSCAN (eps={eps}, min_samples={min_samples}" + (f", {user_api}" if user_api else "") + ")"
    ax1.set_title(title)

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

def graficar_kmeans(results, k, user_api):
    methods = [r[0] for r in results]
    times = [r[1] for r in results]
    silhouettes = [r[2] for r in results]

    fig, ax1 = plt.subplots(figsize=(16,12))
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
    ax1.set_title(f"Comparación Serial vs Paralelo (k={k}, {user_api})")

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

def plot_speedup(df, title: str = "KMeans Speedup (Serial / Parallel)") -> None:
    plt.figure(figsize=(16, 12))
    for backend in df["user_api"].unique():
        sub = df[df["user_api"] == backend].sort_values("k")
        plt.plot(sub["k"], sub["speedup"], marker="o", label=backend)
    plt.axhline(1.0, linestyle="--", linewidth=1)
    plt.xlabel("Cluster count (k)")
    plt.ylabel("Speedup (serial_time / parallel_time)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()