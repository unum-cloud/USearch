import numpy as np

import usearch

count_vectors = 100
count_dimensions = 96

index = usearch.Index(dim=96)
print(index)

vectors = np.random.uniform(
    0, 0.3, (count_vectors, count_dimensions)).astype(np.float32)
labels = np.array(range(count_vectors), dtype=np.longlong)
print('will insert', labels.shape, vectors.shape)

index.add(labels, vectors, copy=True)

print('will search')
results = index.search(vectors, 10)
print('found', results[0].shape, results[1].shape, results[2].shape)
