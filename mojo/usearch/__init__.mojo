"""Billion-Scale Multi-Modal Search, Clustering, & Sparse Algebra Engine for CPUs, GPUs, & NVMe Devices, bound over its C ABI.

USearch indexes dense embeddings in a navigable graph and answers approximate nearest-neighbor
queries over it, with SIMD kernels dispatched to the machine it runs on. This package binds the
dense modality: the index, its row cursor, its search stream, and the three linear-algebra verbs
that read products, attention, and late-interaction scores off the same walk.

The engine is reached by `dlopen`, so a consumer builds with no linker flags at all:

```mojo
from usearch import Arena, DenseIndex, DenseMetric, DenseOptions, Executor, Key, Library, RowsView, ScalarKind

def main() raises:
    var library = Library()
    var index = DenseIndex.init_ram(library, DenseOptions(DenseMetric.ANGULAR, ScalarKind.F32, dimensions=3))
    index.reserve(2)

    var vectors = [Float32(1), 0, 0, Float32(0), 1, 0]
    var keys = [Key(integer=10), Key(integer=11)]
    index.insert(keys, RowsView.of(vectors, 3))                     # on the calling thread

    var executor = Executor.for_ram(library, threads=4)
    var arena = Arena(executor)
    var query = [Float32(1), 0, 0]
    var matches = index.on(executor).search(RowsView.of(query, 3), 1, arena)  # on an executor
    print(t"stored {index.describe().size}, nearest = {matches.row_keys(0)[0]}")
```

The receiver chooses where a verb runs and how it completes, and the verb stays the same: a verb
on the index runs on the calling thread, `index.on(executor)` blocks on that executor, and
`index.deferred(executor)` submits the call there and answers a `Deferred` at once, whose `wait`
blocks until it is final. `Executor(library)` owns nothing and runs each call on the caller's own
thread; the spawning constructors fix a kind that must match the residency of every handle they
drive.

Every output whose length scales with the call lands in an `Arena` the verb takes as its last
required argument, and is read in place. `Arena(library)` holds host pages for calls on the calling
thread, and `Arena(executor)` places its memory where that executor computes. Every receiver takes
it the same way, deferred ones too, and the result borrows it in turn; each call rewinds and refills
it, so a steady loop allocates once. A result refuses every read once a later call rewound its
arena, and a VRAM arena refuses host reads.

Every fallible entry point raises `Error` and nothing else, because Mojo allows one error type per
function, so a second would force every caller raising `Error` to catch and convert. A function raises if and only if it can fail, so a
refusal is never folded into an `Optional` a caller has to interpret.

A closure capturing anything cannot be handed to C, and a `def ... abi("C")` that returns a value
does not lower to the C calling convention, so only a callback returning nothing is written in
Mojo: the `ValueLookup` of `attend`, its state crossing beside it as an address. Progress handlers
are not exposed at all, and the admission predicate of `search` crosses as a raw address, so a
kernel compiled in C installs unchanged.

This package root re-exports every public symbol from the modules that mirror the C headers, so
`from usearch import X` resolves whatever module X lives in.
"""

# library
from .library import Library

# types
from .types import (
    Address,
    Admission,
    Arena,
    BytesView,
    Deferred,
    Distance,
    Error,
    Executor,
    Key,
    Matches,
    Monolith,
    Mood,
    NULL_ADDRESS,
    NvmeOptions,
    RamOptions,
    Residency,
    RowsView,
    ScalarKind,
    Status,
    VramOptions,
)

# dense
from .dense import (
    DenseAttendConfig,
    DenseBatch,
    DenseCursor,
    DenseIndex,
    DenseIndexDeferredOn,
    DenseIndexDescription,
    DenseIndexOn,
    DenseInsertConfig,
    DenseMaxsimConfig,
    DenseMetadata,
    DenseMetric,
    DenseMultiplyConfig,
    DenseOptions,
    DenseProducts,
    DenseReachability,
    DenseRefineConfig,
    DenseSearchConfig,
    DenseSearchStream,
    DenseSearchStreamInitConfig,
    DenseSoftmaxState,
    DenseTraversalStats,
    DenseVectors,
    DenseVerbs,
    ValueLookup,
    dense_metadata_read,
    dense_metadata_read_buffer,
)
