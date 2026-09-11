"""Billion-Scale Multi-Modal Search, Clustering, & Sparse Algebra Engine for CPUs, GPUs, & NVMe Devices, bound over its C ABI.

USearch indexes dense embeddings in a navigable graph and answers approximate nearest-neighbor
queries over it, with SIMD kernels dispatched to the machine it runs on. This package binds the
dense modality: the index, its snapshot cursor, its resumable finder, and the three linear-algebra
verbs that read products, attention, and late-interaction scores off the same walk.

The engine is reached by `dlopen`, so a consumer builds with no linker flags at all:

```mojo
from usearch import (
    DenseIndex, DenseOptions, Executor, Key, Library, MatchesSpan, MatrixView,
    Metric, RamOptions, ScalarKind, TraversalStats,
)

def main() raises:
    var library = Library()
    var executor = Executor(library, RamOptions(), threads=4)
    var index = DenseIndex(library, DenseOptions(Metric.ANGULAR, ScalarKind.F32, dimensions=3))
    index.reserve(2)

    var vectors = [Float32(1), 0, 0, Float32(0), 1, 0]
    var keys = [Key(integer=10), Key(integer=11)]
    var stats = TraversalStats()
    var published = index.add(executor, keys, MatrixView.of(vectors, 3, ScalarKind.F32), stats)

    var query = [Float32(1), 0, 0]
    var found_keys = List[Key](length=1, fill=Key(integer=0))
    var found_scores = List[Float64](length=1, fill=0)
    var found_counts = List[UInt](length=1, fill=0)
    index.search(
        executor,
        MatrixView.of(query, 3, ScalarKind.F32),
        MatchesSpan.of(found_keys, found_scores, found_counts),
        stats,
    )
    print(t"published {published}, nearest = {found_keys[0]}") # published 2, nearest = key(10)
```

An `Executor` is mandatory, never optional: every verb that fans out takes one, and the engine
reads a null executor as a residency mismatch and refuses the call. Its kind is fixed by the
constructor that produced it and must match the residency of every handle it drives.

Every fallible entry point raises `Error` and nothing else, because Mojo allows one error type per
function and never widens a typed `raises`. A function raises if and only if it can fail, so a
refusal is never folded into an `Optional` a caller has to interpret.

No callback is written in Mojo. Progress handlers are not exposed at all, and the two slots that do
take one - the admission predicate of `search_if` and the custom metric of `DenseOptions` - cross
as raw addresses, so a kernel compiled in C installs unchanged. Mojo 1.0 cannot author either: a
closure capturing anything cannot be handed to C, and a `def ... abi("C")` that returns a value
does not lower to the C calling convention, which both of those entry points need.

This package root re-exports every public symbol from the modules that mirror the C headers, so
`from usearch import X` resolves whatever module X lives in.
"""

# library
from .library import Library, Symbols

# types
from .types import (
    Address,
    BytesSpan,
    BytesView,
    Completion,
    Distance,
    Error,
    ErrorSlot,
    ExecutionOptions,
    Executor,
    Key,
    KeysSpan,
    KeysView,
    Mood,
    NULL_ADDRESS,
    NvmeOptions,
    RamOptions,
    Residency,
    ScalarKind,
    Status,
    VramOptions,
    text_at,
)

# dense
from .dense import (
    Admission,
    AttentionOptions,
    DenseCursor,
    DenseFinder,
    DenseIndex,
    DenseMetadata,
    DenseOptions,
    InsertionOptions,
    MatchesSpan,
    MatrixSpan,
    MatrixView,
    Metric,
    SearchOptions,
    SoftmaxState,
    TraversalStats,
    metadata,
    metadata_buffer,
)
