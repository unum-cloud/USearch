# USearch for Python

## Installation

```sh
pip install usearch
```

## Quickstart

```python
import numpy as np
from usearch.index import Index, Matches

index = Index(
    ndim=3, # Define the number of dimensions in input vectors
    metric='cos', # Choose 'l2sq', 'haversine' or other metric, default = 'ip'
    dtype='f32', # Quantize to 'f16' or 'i8' if needed, default = 'f32'
    connectivity=16, # How frequent should the connections in the graph be, optional
    expansion_add=128, # Control the recall of indexing, optional
    expansion_search=64, # Control the quality of search, optional
)

vector = np.array([0.2, 0.6, 0.4])
index.add(42, vector)
matches: Matches = index.search(vector, 10)

assert len(index) == 1
assert len(matches) == 1
assert matches[0].key == 42
assert matches[0].distance <= 0.001
assert np.allclose(index[42], vector)
```

Python bindings are implemented with [`pybind/pybind11`](https://github.com/pybind/pybind11).
Assuming the presence of Global Interpreter Lock in Python, we spawn threads in the C++ layer on large insertions.

## Serialization

```py
index.save('index.usearch')
index.load('index.usearch') # Copy the whole index into memory
index.view('index.usearch') # View from disk without loading in memory
```

If you don't know anything about the index except its path, there are two more endpoints to know:

```py
Index.metadata('index.usearch') -> IndexMetadata
Index.restore('index.usearch', view=False) -> Index
```

## Batch Operations

Adding or querying a batch of entries is identical to adding a single vector.
The difference would be in the shape of the tensors.

```py
n = 100
keys = np.arange(n)
vectors = np.random.uniform(0, 0.3, (n, index.ndim)).astype(np.float32)

index.add(keys, vectors, threads=..., copy=...)
matches: BatchMatches = index.search(vectors, 10, threads=...)

first_query_matches: Matches = matches[0]
assert matches[0].key == 0
assert matches[0].distance <= 0.001

assert len(matches) == vectors.shape[0]
assert len(matches[0]) <= 10
```

You can also override the default `threads` and `copy` arguments in bulk workloads.
The first controls the number of threads spawned for the task.
The second controls whether the vector itself will be persisted inside the index.
If you can preserve the lifetime of the vector somewhere else, you can avoid the copy.

> When using `BatchMatches`, be aware that unused positions in the results arrays will be filled with sentinel values, like the signaling `NaN` for floating-point distances.
> Don't forget to check for `matches.counts` if you are using the recommended batch interfaces.

## User-Defined Metrics and JIT in Python

### [Numba][numba]

Assuming the language boundary exists between Python user code and C++ implementation, there are more efficient solutions than passing a Python callable to the engine.
Luckily, with the help of [Numba][numba], we can JIT compile a function with a matching signature and pass it down to the engine.

```py
from numba import cfunc, types, carray

ndim = 256
signature = types.float32(
    types.CPointer(types.float32),
    types.CPointer(types.float32))

@cfunc(signature)
def inner_product(a, b):
    a_array = carray(a, ndim)
    b_array = carray(b, ndim)
    c = 0.0
    for i in range(ndim):
        c += a_array[i] * b_array[i]
    return 1 - c

index = Index(ndim=ndim, metric=CompiledMetric(
    pointer=inner_product.address,
    kind=MetricKind.IP,
    signature=MetricSignature.ArrayArray,
))
```

Alternatively, you can avoid pre-defining the number of dimensions, and pass it separately:

```py
signature = types.float32(
    types.CPointer(types.float32),
    types.CPointer(types.float32),
    types.uint64)

@cfunc(signature)
def inner_product(a, b, ndim):
    a_array = carray(a, ndim)
    b_array = carray(b, ndim)
    c = 0.0
    for i in range(ndim):
        c += a_array[i] * b_array[i]
    return 1 - c

index = Index(ndim=ndim, metric=CompiledMetric(
    pointer=inner_product.address,
    kind=MetricKind.IP,
    signature=MetricSignature.ArrayArraySize,
))
```

```sh
pip install numba
```

### [Cppyy][cppyy]

Similarly, you can use Cppyy with Cling to JIT-compile native C or C++ code and pass it to USearch, which may be a good idea, if you want to explicitly request loop-unrolling or other low-level optimizations!

```py
import cppyy
import cppyy.ll

cppyy.cppdef("""
float inner_product(float *a, float *b) {
    float result = 0;
#pragma unroll
    for (size_t i = 0; i != ndim; ++i)
        result += a[i] * b[i];
    return 1 - result;
}
""".replace("ndim", str(ndim)))

function = cppyy.gbl.inner_product
index = Index(ndim=ndim, metric=CompiledMetric(
    pointer=cppyy.ll.addressof(function),
    kind=MetricKind.IP,
    signature=MetricSignature.ArrayArraySize,
))
```

```sh
conda install -c conda-forge cppyy
```

### [PeachPy][peachpy]

We have covered JIT-ing Python with Numba and C++ with Cppyy and Cling.
How about writing Assembly directly?
That is also possible.
Below is an example of constructing the "Inner Product" distance for 8-dimensional `f32` vectors for x86 using [PeachPy][peachpy].

```py
from peachpy import (
    Argument,
    ptr,
    float_,
    const_float_,
)
from peachpy.x86_64 import (
    abi,
    Function,
    uarch,
    isa,
    GeneralPurposeRegister64,
    LOAD,
    YMMRegister,
    VSUBPS,
    VADDPS,
    VHADDPS,
    VMOVUPS,
    VFMADD231PS,
    VPERM2F128,
    VXORPS,
    RETURN,
)

a = Argument(ptr(const_float_), name="a")
b = Argument(ptr(const_float_), name="b")

with Function(
    "inner_product", (a, b), float_, target=uarch.default + isa.avx2
) as asm_function:
    
    # Request two 64-bit general-purpose registers for addresses
    reg_a, reg_b = GeneralPurposeRegister64(), GeneralPurposeRegister64()
    LOAD.ARGUMENT(reg_a, a)
    LOAD.ARGUMENT(reg_b, b)

    # Load the vectors
    ymm_a = YMMRegister()
    ymm_b = YMMRegister()
    VMOVUPS(ymm_a, [reg_a])
    VMOVUPS(ymm_b, [reg_b])

    # Prepare the accumulator
    ymm_c = YMMRegister()
    ymm_one = YMMRegister()
    VXORPS(ymm_c, ymm_c, ymm_c)
    VXORPS(ymm_one, ymm_one, ymm_one)

    # Accumulate A and B product into C
    VFMADD231PS(ymm_c, ymm_a, ymm_b)

    # Reduce the contents of a YMM register
    ymm_c_permuted = YMMRegister()
    VPERM2F128(ymm_c_permuted, ymm_c, ymm_c, 1)
    VADDPS(ymm_c, ymm_c, ymm_c_permuted)
    VHADDPS(ymm_c, ymm_c, ymm_c)
    VHADDPS(ymm_c, ymm_c, ymm_c)

    # Negate the values, to go from "similarity" to "distance"
    VSUBPS(ymm_c, ymm_one, ymm_c)

    # A common convention is to return floats in XMM registers
    RETURN(ymm_c.as_xmm)

python_function = asm_function.finalize(abi.detect()).encode().load()
metric = CompiledMetric(
    pointer=python_function.loader.code_address,
    kind=MetricKind.IP,
    signature=MetricSignature.ArrayArray,
)
index = Index(ndim=ndim, metric=metric)
```

[numba]: https://numba.readthedocs.io/en/stable/reference/jit-compilation.html#c-callbacks
[cppyy]: https://cppyy.readthedocs.io/en/latest/
[peachpy]: https://github.com/Maratyszcza/PeachPy

## Tooling

To work with `bbin`, `fbin`, `ibin`, `hbin` matrix files USearch provides `load_matrix` and `save_matrix`.
Such files are standard in k-ANN tasks and represent a binary object with all the scalars, prepended by two 32-bit integers - the number of rows and columns in the matrix.

```py
from usearch.index import Index
from usearch.io import load_matrix, save_matrix

vectors = load_matrix('deep1B.fbin')
index = Index(ndim=vectors.shape[1])
index.add(keys, vectors)
```

One may often want to evaluate the quality of the constructed index before running in production.
The trivial way is to measure `recall@1` on the entries already present in the index.

```py
from usearch.eval import self_recall

stats: SearchStats = self_recall(index, exact=True)
assert stats.visited_members == 0, "Exact search won't attend index nodes"
assert stats.computed_distances == len(index), "And will compute the distance to every node"

stats: SearchStats = self_recall(index, exact=False)
assert stats.visited_members > 0
assert stats.computed_distances <= len(index)
```

In case you have some ground-truth data for more than one entry, you compare search results against expected values:

```py
from usearch.eval import relevance, dcg, ndcg, random_vectors

vectors = random_vectors(index=index)
matches_approximate = index.search(vectors)
matches_exact = index.search(vectors, exact=True)
relevance_scores = relevance(matches_exact, matches_approximate)
print(dcg(relevance_scores), ndcg(relevance_scores))
```
