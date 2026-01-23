#!/usr/bin/env -S uv run --quiet --script
"""
USearch Index Joining Utility

Script for joining and benchmarking multimodal datasets (images and texts)
using different embedding models and distance metrics. Includes support for
cross-modal search and evaluation.

Usage:
    uv run python/scripts/join.py

To download the required datasets, run:
wget -nc https://huggingface.co/datasets/unum-cloud/ann-cc-3m/resolve/main/clip_images.fbin -P datasets/cc_3M/
wget -nc https://huggingface.co/datasets/unum-cloud/ann-cc-3m/resolve/main/clip_texts.fbin -P datasets/cc_3M/
wget -nc https://huggingface.co/datasets/unum-cloud/ann-cc-3m/resolve/main/images.fbin -P datasets/cc_3M/
wget -nc https://huggingface.co/datasets/unum-cloud/ann-cc-3m/resolve/main/texts.fbin -P datasets/cc_3M/

Dependencies listed in the script header for uv to resolve automatically.
"""
# /// script
# dependencies = [
#   "numpy",
#   "simsimd",
#   "usearch",
#   "tqdm"
# ]
# ///

from numpy import dot
from numpy.linalg import norm
from tqdm import tqdm
from simsimd import cos_f32x4_neon, to_int

import usearch
from usearch.index import Index, MetricKind, CompiledMetric, MetricSignature
from usearch.io import load_matrix
from usearch.eval import measure_seconds

count = 10
exact = False
batch_size = 1024 * 4
max_elements = 1000000

a_name = "cc_3M/texts"
b_name = "cc_3M/images"

a_mat = load_matrix(f"datasets/{a_name}.fbin", view=True)
b_mat = load_matrix(f"datasets/{b_name}.fbin", view=True)

a_mat = a_mat[:max_elements]
b_mat = b_mat[:max_elements]

print(f"Loaded two datasets of shape: {a_mat.shape}, {b_mat.shape}")
print("--------------------------------------")
print("---------------Indexing---------------")
print("--------------------------------------")

metric = CompiledMetric(
    pointer=to_int(cos_f32x4_neon),
    kind=MetricKind.Cos,
    signature=MetricSignature.ArrayArraySize,
)

a = Index(
    a_mat.shape[1],
    metric=metric,
    path=f"datasets/{a_name}-{max_elements}.f32.usearch",
    dtype="f32",
)
b = Index(
    b_mat.shape[1],
    metric=metric,
    path=f"datasets/{b_name}-{max_elements}.f32.usearch",
    dtype="f32",
)

if len(a) != a_mat.shape[0]:
    a.clear()
    a.add(None, a_mat, log=True, batch_size=batch_size)
    a.save()

if len(b) != b_mat.shape[0]:
    b.clear()
    b.add(None, b_mat, log=True, batch_size=batch_size)
    b.save()


print(f"Loaded two indexes of size: {len(a):,} for {a_name} and {len(b):,} for {b_name}")
min_elements = min(len(a), len(b))

run_diagnostics = input("Would you like to run diagnostics? [Y/n]: ")
if len(run_diagnostics) == 0 or run_diagnostics.lower() == "y":
    print("--------------------------------------")
    print("-------------Diagnostics--------------")
    print("--------------------------------------")

    mean_similarity = 0.0
    mean_recovered_similarity = 0.0

    for i in tqdm(range(min_elements), desc="Pairwise Similarity"):
        a_vec = a_mat[i]
        b_vec = b_mat[i]
        cos_similarity = dot(a_vec, b_vec) / (norm(a_vec) * norm(b_vec))
        mean_similarity += cos_similarity

        a_vec = a[i]
        b_vec = b[i]
        cos_similarity = dot(a_vec, b_vec) / (norm(a_vec) * norm(b_vec))
        mean_recovered_similarity += cos_similarity

    mean_similarity /= min_elements
    mean_recovered_similarity /= min_elements
    print(
        f"Average vector similarity is {mean_similarity:.4f} in original dataset, "
        f"and {mean_recovered_similarity:.4f} in recovered state in index"
    )

    dt = measure_seconds
    args = dict(
        count=count,
        batch_size=batch_size,
        log=True,
        exact=exact,
    )

    secs, a_self_recall = dt(lambda: a.search(a.vectors, **args).recall(a.keys))
    print("Self-recall @{} of {} index: {:.2f}%, took {:.2f}s".format(count, a_name, a_self_recall * 100, secs))

    secs, b_self_recall = dt(lambda: b.search(b.vectors, **args).recall(b.keys))
    print("Self-recall @{} of {} index: {:.2f}%, took {:.2f}s".format(count, b_name, b_self_recall * 100, secs))

    secs, ab_recall = dt(lambda: b.search(a.vectors, **args).recall(b.keys))
    print("Cross-recall @{} of {} in {}: {:.2f}%, took {:.2f}s".format(count, a_name, b_name, ab_recall * 100, secs))

    secs, ba_recall = dt(lambda: a.search(b.vectors, **args).recall(a.keys))
    print("Cross-recall @{} of {} in {}: {:.2f}%, took {:.2f}s".format(count, b_name, a_name, ba_recall * 100, secs))


print("--------------------------------------")
print("-----------------Join-----------------")
print("--------------------------------------")

secs, bimapping = measure_seconds(lambda: a.join(b, max_proposals=100))
mapping_size = len(bimapping)
recall = 0
for i, j in bimapping.items():
    recall += i == j

recall *= 100.0 / min_elements
print(f"Took {secs:.2f}s to find {mapping_size:,} pairings with {recall:.2f}% being exact")
