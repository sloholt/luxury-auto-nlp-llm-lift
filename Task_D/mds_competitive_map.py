import os
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.spatial import procrustes

TASK_B_LIFT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Task_B", "output", "brand_lift_matrix.csv")
TASK_C_LIFT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Task_C", "output", "semantic_lift_matrix.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


CANONICAL_BRAND = {
    "bmw": "BMW", "acura": "Acura", "cadillac": "Cadillac", "audi": "Audi",
    "lexus": "Lexus", "gm": "GM", "infiniti": "Infiniti",
    "mercedes-benz": "Mercedes-Benz", "volvo": "Volvo", "nissan": "Nissan",
}


def load_lift_matrix(path):
    df = pd.read_csv(path, index_col=0)
    df.index = [CANONICAL_BRAND.get(i.lower(), i) for i in df.index]
    df.columns = [CANONICAL_BRAND.get(c.lower(), c) for c in df.columns]
    return df


def lift_to_distance(lift_df, method="reciprocal", eps=0.05):
    d = lift_df.copy().astype(float)
    if method == "reciprocal":
        dist = 1.0 / d.clip(lower=eps)
    elif method == "neglog":
        dist = (-np.log(d.clip(lower=eps))).clip(lower=0)
    else:
        raise ValueError(method)
    for b in dist.index:
        dist.loc[b, b] = 0.0
    return dist


def run_mds(distance_df, n_components=2, random_state=42):
    mds = MDS(n_components=n_components, dissimilarity="precomputed",
              random_state=random_state, normalized_stress="auto", n_init=8, init="random")
    coords = mds.fit_transform(distance_df.values)
    coords_df = pd.DataFrame(coords, index=distance_df.index, columns=[f"dim{i+1}" for i in range(n_components)])
    return coords_df, mds.stress_


def run_clustering(distance_df, n_clusters=3, method="average"):
    dm = distance_df.values.copy()
    np.fill_diagonal(dm, 0)
    Z = linkage(squareform(dm, checks=False), method=method)
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    return dict(zip(distance_df.index, labels)), Z


def plot_mds(coords_df, cluster_labels, title, path):
    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.get_cmap("tab10")
    for brand in coords_df.index:
        c = cluster_labels.get(brand, 0)
        ax.scatter(coords_df.loc[brand, "dim1"], coords_df.loc[brand, "dim2"],
                   s=140, color=cmap(c % 10), edgecolor="black", zorder=3)
        ax.annotate(brand, (coords_df.loc[brand, "dim1"], coords_df.loc[brand, "dim2"]),
                    xytext=(6, 6), textcoords="offset points", fontsize=10)
    ax.set_title(title)
    ax.set_xlabel("MDS dim 1")
    ax.set_ylabel("MDS dim 2")
    ax.axhline(0, color="grey", lw=0.5)
    ax.axvline(0, color="grey", lw=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def mds_sensitivity(lift_df_a, lift_df_b, method="reciprocal"):
    coords_a, _ = run_mds(lift_to_distance(lift_df_a, method=method))
    coords_b, _ = run_mds(lift_to_distance(lift_df_b, method=method))
    common = [b for b in coords_a.index if b in coords_b.index]
    _, _, disparity = procrustes(coords_a.loc[common].values, coords_b.loc[common].values)
    return {"procrustes_disparity": disparity, "brands_compared": common}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    nlp_lift_df = load_lift_matrix(TASK_B_LIFT_PATH)
    brands = list(nlp_lift_df.index)

    dist_nlp = lift_to_distance(nlp_lift_df, method="reciprocal")
    dist_nlp.to_csv(os.path.join(OUTPUT_DIR, "distance_matrix_nlp.csv"))

    coords_nlp, stress_nlp = run_mds(dist_nlp)
    coords_nlp.to_csv(os.path.join(OUTPUT_DIR, "mds_coords_nlp.csv"))

    clusters_nlp, Z_nlp = run_clustering(dist_nlp, n_clusters=3)
    pd.Series(clusters_nlp, name="cluster").to_csv(os.path.join(OUTPUT_DIR, "clusters_nlp.csv"))

    plot_mds(coords_nlp, clusters_nlp, "Competitive map — NLP lexical lift",
              os.path.join(OUTPUT_DIR, "mds_map_nlp.png"))

    print(f"NLP MDS stress: {stress_nlp:.4f}")
    print(clusters_nlp)

    if os.path.exists(TASK_C_LIFT_PATH):
        llm_lift_df = load_lift_matrix(TASK_C_LIFT_PATH)
        llm_lift_df = llm_lift_df.loc[brands, brands]

        dist_llm = lift_to_distance(llm_lift_df, method="reciprocal")
        dist_llm.to_csv(os.path.join(OUTPUT_DIR, "distance_matrix_llm.csv"))

        coords_llm, stress_llm = run_mds(dist_llm)
        coords_llm.to_csv(os.path.join(OUTPUT_DIR, "mds_coords_llm.csv"))

        clusters_llm, Z_llm = run_clustering(dist_llm, n_clusters=3)
        pd.Series(clusters_llm, name="cluster").to_csv(os.path.join(OUTPUT_DIR, "clusters_llm.csv"))

        plot_mds(coords_llm, clusters_llm, "Competitive map — LLM semantic lift",
                  os.path.join(OUTPUT_DIR, "mds_map_llm.png"))

        print(f"LLM MDS stress: {stress_llm:.4f}")
        print(clusters_llm)

        sensitivity = mds_sensitivity(nlp_lift_df, llm_lift_df)
        pd.DataFrame([sensitivity]).to_csv(os.path.join(OUTPUT_DIR, "mds_sensitivity.csv"), index=False)
        print(sensitivity)
    else:
        print(f"Task C output not found at {TASK_C_LIFT_PATH} — run Task_C/semantic_lift.py first for the sensitivity comparison.")

    print(f"Wrote outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
