#!/usr/bin/env -S uv run --quiet --script
"""
USearch Clustering Benchmarking

Benchmarks clustering performance comparing USearch k-means implementation
against FAISS baseline with various datasets and configurations.

Usage:
    uv run python/scripts/bench_cluster.py --help
    uv run python/scripts/bench_cluster.py --data datasets/wiki_1M/base.1M.fbin
    
Dependencies listed in the script header for uv to resolve automatically.
"""
# /// script
# dependencies = [
#   "numpy",
#   "faiss-cpu", 
#   "usearch",
#   "tqdm"
# ]
# ///
import os
import argparse

import numpy as np
import faiss
from tqdm import tqdm

import usearch
from usearch.index import kmeans
from usearch.io import load_matrix


def evaluate_clustering_euclidean(X, labels, centroids):
    """Evaluate clustering quality as average distance to centroids"""
    distances = np.linalg.norm(X - centroids[labels], axis=1)
    return np.mean(distances)


def evaluate_clustering_cosine(X, labels, centroids):
    """Evaluate clustering quality as average cosine distance to centroids"""

    # Normalize both data points and centroids
    X_normalized = X / np.linalg.norm(X, axis=1, keepdims=True)
    centroids_normalized = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)

    # Compute cosine similarity using dot product
    cosine_similarities = np.sum(X_normalized * centroids_normalized[labels], axis=1)

    # Convert cosine similarity to cosine distance
    cosine_distances = 1 - cosine_similarities

    # Return the average cosine distance
    return np.mean(cosine_distances)


def numpy_initialize_centroids(X, k):
    """Randomly choose k data points as initial centroids"""
    indices = np.random.choice(X.shape[0], k, replace=False)
    return X[indices]


def numpy_assign_clusters(X, centroids):
    """Assign each data point to the nearest centroid (numpy NumPy implementation)"""
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)


def numpy_update_centroids(X, labels, k):
    """Compute new centroids as the mean of all data points assigned to each cluster"""
    return np.array([X[labels == i].mean(axis=0) for i in range(k)])


def cluster_with_numpy(X, k, max_iters=100, tol=1e-4):
    centroids = numpy_initialize_centroids(X, k)

    for i in tqdm(range(max_iters), desc="KMeans Iterations"):
        labels = numpy_assign_clusters(X, centroids)
        new_centroids = numpy_update_centroids(X, labels, k)

        if np.linalg.norm(new_centroids - centroids) < tol:
            break

        centroids = new_centroids

    return labels, centroids


def cluster_with_faiss(X, k, max_iters=100):
    # Docs: https://github.com/facebookresearch/faiss/wiki/Faiss-building-blocks:-clustering,-PCA,-quantization
    # Header: https://github.com/facebookresearch/faiss/blob/main/faiss/Clustering.h
    # Source: https://github.com/facebookresearch/faiss/blob/main/faiss/Clustering.cpp
    verbose = False
    d: int = X.shape[1]
    kmeans = faiss.Kmeans(d, k, niter=max_iters, verbose=verbose)
    kmeans.train(X)
    D, I = kmeans.index.search(X, 1)
    return I.flatten(), kmeans.centroids


def cluster_with_usearch(X, k, max_iters=100):
    assignments, _, centroids = kmeans(X, k, max_iterations=max_iters)
    return assignments, centroids


def main():
    parser = argparse.ArgumentParser(description="Compare KMeans clustering algorithms")
    parser.add_argument("--vectors", type=str, required=True, help="Path to binary matrix file")
    parser.add_argument("-k", default=10, type=int, required=True, help="Number of centroids")
    parser.add_argument("-i", default=100, type=int, help="Upper bound on number of iterations")
    parser.add_argument("-n", type=int, help="Upper bound on number of points to use")
    parser.add_argument(
        "--method",
        type=str,
        choices=["numpy", "faiss", "usearch"],
        default="numpy",
        help="Clustering backend",
    )

    args = parser.parse_args()

    max_iters = args.i
    X = load_matrix(args.vectors, count_rows=args.n)
    k = args.k
    method = args.method

    time_before = os.times()
    if method == "usearch":
        labels, centroids = cluster_with_usearch(X, k, max_iters=max_iters)
    elif method == "faiss":
        labels, centroids = cluster_with_faiss(X, k, max_iters=max_iters)
    else:
        labels, centroids = cluster_with_numpy(X, k, max_iters=max_iters)
    time_after = os.times()
    time_duration = time_after[0] - time_before[0]
    print(f"Time: {time_duration:.2f}s, {time_duration / max_iters:.2f}s per iteration")

    quality = evaluate_clustering_euclidean(X, labels, centroids)
    quality_cosine = evaluate_clustering_cosine(X, labels, centroids)
    print(f"Clustering quality (average distance to centroids): {quality:.4f}, cosine: {quality_cosine:.4f}")

    # Let's compare it to some random uniform assignment
    random_labels = np.random.randint(0, k, size=X.shape[0])
    random_quality = evaluate_clustering_euclidean(X, random_labels, centroids)
    random_quality_cosine = evaluate_clustering_cosine(X, random_labels, centroids)
    print(f"... while random assignment quality: {random_quality:.4f}, cosine: {random_quality_cosine:.4f}")

    cluster_sizes = np.unique(labels, return_counts=True)[1]
    cluster_sizes_mean = np.mean(cluster_sizes)
    cluster_sizes_stddev = np.std(cluster_sizes)
    print(f"Cluster sizes: {cluster_sizes_mean:.2f} ± {cluster_sizes_stddev:.2f}")
    print(cluster_sizes)


if __name__ == "__main__":
    main()
