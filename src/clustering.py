import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pyarrow.parquet as pq # Added since data.py writes the cleaned pull requests as a parquet file
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

# Overview of the clustering process this file does:
# It takes a sample of the cleaned pull requests, converts their titles and bodies to 
# TF-IDF features, and transforms those features with PCA. 
# It then runs both K-means and hierarchical clustering on the PCA-transformed sample and 
# compares their results.

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_PATH = ROOT / "data" / "processed" / "prs.parquet"
SAMPLE_SIZE = 2000
BATCH_SIZE = 65536
REQUIRED_COLUMNS = ["title", "body", "task_type", "agent", "merged"]


# Load a sample in batches so the full parquet file does
# not need to be held in memory. The other columns are retained for analysis.
def load_sample(path=PROCESSED_PATH, sample_size=SAMPLE_SIZE, random_state=42):
    if sample_size < 1:
        raise ValueError("sample_size must be at least 1.")

    parquet_file = pq.ParquetFile(path)
    missing_columns = set(REQUIRED_COLUMNS) - set(parquet_file.schema.names)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    total_rows = parquet_file.metadata.num_rows
    sample_size = min(sample_size, total_rows)
    if sample_size == total_rows:
        return parquet_file.read(columns=REQUIRED_COLUMNS).to_pandas()

    sampled_frames = []
    rows_seen = 0
    samples_taken = 0

    for batch_number, batch in enumerate(
        parquet_file.iter_batches(batch_size=BATCH_SIZE, columns=REQUIRED_COLUMNS)
    ):
        frame = batch.to_pandas()
        rows_seen += len(frame)
        target_samples = round(sample_size * rows_seen / total_rows)
        batch_samples = target_samples - samples_taken
        if batch_samples:
            sampled_frames.append(
                frame.sample(
                    n=batch_samples,
                    random_state=random_state + batch_number,
                )
            )
            samples_taken += batch_samples

    if not sampled_frames:
        return parquet_file.read(columns=REQUIRED_COLUMNS).to_pandas()
    return pd.concat(sampled_frames, ignore_index=True)


# Returns the most frequent label in a cluster or "Unknown" if
# that column is missing for every row in the cluster.
def _most_common(values):
    counts = values.fillna("Unknown").value_counts()
    return counts.index[0] if not counts.empty else "Unknown"


def cluster_prs(prs, n_clusters=5, random_state=42):
    missing_columns = set(REQUIRED_COLUMNS) - set(prs.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
    if len(prs) < 2:
        raise ValueError("At least two pull requests are required for clustering.")
    if not 2 <= n_clusters <= len(prs):
        raise ValueError("n_clusters must be between 2 and the number of pull requests.")

    # Build text features from the task description only. Task type, agent, and
    # merge outcome are kept out of the model inputs for post-clustering analysis.
    text = (
        prs["title"].fillna("").astype(str)
        + " "
        + prs["body"].fillna("").astype(str)
    )
    if not text.str.strip().any():
        raise ValueError("No title or body text is available to cluster.")

    # TF-IDF represents each pull request by the terms in its title/body
    vectorizer = TfidfVectorizer(
        max_features=1000,
        min_df=2,
        max_df=0.95,
        stop_words="english",
    )
    tfidf = vectorizer.fit_transform(text)
    if tfidf.shape[1] < 2:
        raise ValueError("At least two distinct text features are needed for PCA.")

    # PCA compresses the text vectors into a smaller set of numerical features.
    # Its first two components are also used to visualize the cluster assignments.
    n_components = min(50, len(prs) - 1, tfidf.shape[1])
    pca = PCA(n_components=n_components, random_state=random_state)
    reduced_features = pca.fit_transform(tfidf.toarray())

    results = prs.copy().reset_index(drop=True)

    # K-means assigns each pull request to the nearest of n_clusters centroids.
    results["kmeans_cluster"] = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
    ).fit_predict(reduced_features)

    # Hierarchical clustering merges groups to limit the increase in
    # within-cluster variance. It uses the same PCA features as the K-means.
    results["hierarchical_cluster"] = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage="ward",
    ).fit_predict(reduced_features)

    # Compares the resulting groups with task type, agent, and merge rates.
    # Done for descriptive summaries to potentially include in our report, 
    # has no effect on the actual clustering predictions.
    for algorithm in ("kmeans", "hierarchical"):
        summary = results.groupby(f"{algorithm}_cluster").agg(
            pull_requests=("merged", "size"),
            merge_rate=("merged", "mean"),
            most_common_task_type=("task_type", _most_common),
            most_common_agent=("agent", _most_common),
        )
        print(f"\n{algorithm.title()} cluster summary:")
        print(summary.to_string())

    # Reports how much feature variance the PCA components preserve, then plots
    # both algorithms' assignments on the first two principal components.
    print(
        "\nPCA variance retained by the components used for clustering:",
        round(pca.explained_variance_ratio_.sum(), 4),
    )
    if n_components >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for axis, algorithm in zip(axes, ("kmeans", "hierarchical")):
            for cluster in sorted(results[f"{algorithm}_cluster"].unique()):
                mask = results[f"{algorithm}_cluster"] == cluster
                axis.scatter(
                    reduced_features[mask, 0],
                    reduced_features[mask, 1],
                    alpha=0.6,
                    label=f"Cluster {cluster}",
                )
            axis.set_xlabel("Principal component 1")
            axis.set_ylabel("Principal component 2")
            axis.set_title(f"{algorithm.title()} clusters")
            axis.legend()
        fig.tight_layout()
        plt.show()

    return results, pca, vectorizer


def main():
    # Command-line entry point for sampling the processed data and running both
    # clustering algorithms with a specified cluster count.
    parser = argparse.ArgumentParser(
        description="Cluster pull-request text and compare task, agent, and merge outcomes."
    )
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--clusters", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    if args.sample_size < 2:
        parser.error("--sample-size must be at least 2")
    prs = load_sample(sample_size=args.sample_size, random_state=args.random_state)
    print(f"Clustering {len(prs):,} sampled pull requests.")
    cluster_prs(prs, n_clusters=args.clusters, random_state=args.random_state)


if __name__ == "__main__":
    main()
