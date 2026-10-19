"""Vector search over equidimensional embeddings: the index, its receivers, its cursor, and its search stream.

The type is `DenseIndex` rather than `Index` because the engine has sibling modalities that will
bind beside it, and a bare name would have to be renamed the day one of them lands.

Every verb takes one argument structure named after it, and every one of them is mirrored below as
a `...Arguments` struct laid out exactly as the library reads it, materialized in a
`stack_allocation` slot and crossed by address. The receiver picks the completion: verbs on the
index and on `index.on(executor)` pass blocking completion and return with their outcome final and
their outputs readable, while `index.deferred(executor)` passes deferred completion and answers a
`Deferred` at once. Each verb is one `Call` all three route, so the receivers stay symmetric.

Batches are rows. `RowsView` carries the base pointer, the row count, the column count, the byte
stride, and the scalar kind as one value, so a single vector is a one-row view. The row count
becomes the verb's `count`, and the column count crosses with the rows, so the engine checks it
against the geometry the index froze.

Every output whose length scales with the call - the matches of `search` and `maxsim`, the rows of
`get` and `attend`, the pairs of `multiply`, the monolith of `save_buffer`, and every cursor and
stream pull - lands in an `Arena` the verb requires, and is read in place. Every receiver takes the arena
as the verb's last required argument, deferred ones too, and the result borrows it in turn, named by
the arena's origin.

Two Mojo 1.1 facts shape the callback slots. A closure capturing anything cannot be handed to C at
all, so a callback's state crosses beside it as an `Address`; and a `def ... abi("C")` that returns
a value does not in fact lower to the C calling convention, so the engine calling one jumps into a
corrupted frame. The value lookup of `attend` returns nothing, so it is a Mojo function - a
`ValueLookup`. The admission predicate of `search` crosses as a raw `Address` that a kernel built in
C fills; progress handlers return a verdict, so every `progress` slot crosses with a `NULL_ADDRESS`
callback.

Telemetry is a lent output. Every config of a verb that reports traversal statistics takes an
optional `stats` pointer, which the engine overwrites whole on every call it is passed to.
"""

from std.ffi import c_int, c_size_t
from std.memory import stack_allocation
from std.sys.info import size_of

from usearch.library import Library
from usearch.types import (
    Address,
    Arena,
    BytesView,
    Call,
    Completion,
    Deferred,
    Distance,
    Error,
    Executor,
    Key,
    KeyBytes,
    KeysTapeViewArguments,
    KeysView,
    Matches,
    MatchesTapeViewArguments,
    MatchesViewArguments,
    Monolith,
    NULL_ADDRESS,
    NvmeOptions,
    Outcome,
    PredicateArguments,
    ProgressArguments,
    RamOptions,
    RowsView,
    RowsViewArguments,
    ScalarKind,
    Status,
    Traversal,
    Unwritten,
    Verb,
    VramOptions,
    _anchor,
    _block,
    _in_arena,
    _readable,
    _refuse,
    _served,
    text_at,
)

# region Configuration


@fieldwise_init
struct DenseMetric(Equatable, ImplicitlyCopyable, TrivialRegisterPassable):
    """The built-in dense metrics - `usearch_dense_metric_t`, each with its own SIMD kernel chain.

    The last two read their rows as `ScalarKind.U1X8` bit sets and refuse any other quantization; the
    first four refuse that one. The three linear-algebra verbs run on `INNER` alone, since a
    product and a softmax both need logits rather than distances. `Library.dense_metric_name` spells
    each one as the engine does.
    """

    var code: c_int
    """The raw `usearch_dense_metric_t` value the ABI exchanges."""
    comptime UNKNOWN = Self(0)
    """No kind named - an unset field, refused by every constructor."""
    comptime INNER = Self(1)
    """Sum of elementwise products, unnormalized and scale-sensitive."""
    comptime ANGULAR = Self(2)
    """Cosine distance in [0, 2], blind to magnitude."""
    comptime EUCLIDEAN = Self(3)
    """L2 distance, zero only for identical rows."""
    comptime HAMMING = Self(4)
    """Count of differing bits, so its range is the dimensionality."""
    comptime JACCARD = Self(5)
    """Differing bits over union bits in [0, 1], ignoring the zeros both rows share."""


@fieldwise_init
struct DenseSoftmaxState(ImplicitlyCopyable, TrivialRegisterPassable):
    """The online-softmax carry of one query row - `usearch_dense_softmax_state_t`.

    Flash Attention's running maximum and partition. The two are one concept: a maximum without
    its partition is a number nothing can rescale, so they share an element rather than travelling
    as two runs a caller could supply half of. Carrying them folds one attention call into another
    over the same queries.
    """

    var maximum_score: Float32
    """The largest logit seen so far - the recurrence's `m`."""
    var partition: Float32
    """The sum of exponentials taken against that maximum - the recurrence's `l`."""

    def __init__(out self):
        """Zeroes both, which is what an unfolded first branch starts from."""
        self.maximum_score = Float32(0)
        self.partition = Float32(0)


comptime ValueLookup = def(
    Address, Pointer[Key, ImmUntrackedOrigin], c_size_t, ScalarKind, c_size_t, Pointer[Address, MutAnyOrigin]
) thin abi("C") -> None
"""`usearch_value_lookup_t`: resolves `count` keys to one value-row address each, `columns` values of
`scalar` apiece - the index's own width and kind - alive until the call returns; a `NULL_ADDRESS` row
refuses the call.

Called with the `value_lookup_state` the verb was handed, once per scored batch and concurrently from
the executor's threads. A lookup whose rows hold another kind or width answers `NULL_ADDRESS` for
every key rather than handing the engine rows it would misread.

A VRAM index calls it inside the device walk instead, so there it must be the address of a
`__device__` function, as `cudaMemcpyFromSymbol` reads it, with its state and rows in memory that
device reads. Nothing tells the two kinds apart, so a host function there is undefined, never refused."""


@fieldwise_init
struct DenseTraversalStats(ImplicitlyCopyable):
    """Traversal telemetry - `usearch_traversal_stats_t`, a pure output.

    Fully overwritten by the call it is passed to, so a caller reusing one slot across calls reads
    the last call's figures rather than a running total.
    """

    var visited_members: c_size_t
    """Graph members visited while resolving the batch."""
    var computed_distances: c_size_t
    """Distances computed while resolving the batch, diversity re-selections included."""
    var restarts: c_size_t
    """Re-seedings after the frontier drained prematurely - a connectivity health signal."""

    def __init__(out self):
        """Zeroes every counter, so an unwritten slot reads as an untaken walk."""
        self.visited_members = c_size_t(0)
        self.computed_distances = c_size_t(0)
        self.restarts = c_size_t(0)


@fieldwise_init
struct DenseReachability(Defaultable, ImplicitlyCopyable):
    """One reachability audit - `usearch_reachability_t`: the population it judged, and how much
    of it a search reaches."""

    var audited: c_size_t
    """Entries published when the audit started; compare `reachable` against this, never a later size."""
    var reachable: c_size_t
    """Of those, the entries a search reaches; the rest no search finds until a `refine` re-links them."""
    var sweeps: c_size_t
    """Levels the sweep took to converge - a diameter proxy."""

    def __init__(out self):
        """Zeroes the audit, which is what a failed call leaves."""
        self.audited = c_size_t(0)
        self.reachable = c_size_t(0)
        self.sweeps = c_size_t(0)


@fieldwise_init
struct DenseOptions(ImplicitlyCopyable):
    """The geometry an empty index freezes - what `usearch_dense_init_ram_t` and
    `usearch_dense_init_vram_t` share, without the placement each names its own way.

    Zeroed optional fields derive from the geometry.
    """

    var metric: DenseMetric
    """The metric kind used for distance calculations between vectors."""
    var quantization: ScalarKind
    """The scalar type vectors are stored and quantized into."""
    var dimensions: c_size_t
    """Dimensions in the indexed vectors."""
    var connectivity: c_size_t
    """The connections-per-node bound; zero derives."""
    var values_per_key: c_size_t
    """Upper bound on entries sharing one key; zero derives to one - unique keys."""

    def __init__(
        out self,
        dimensions: Int,
        metric: DenseMetric = DenseMetric.ANGULAR,
        quantization: ScalarKind = ScalarKind.BF16,
        connectivity: Int = 0,
        values_per_key: Int = 0,
    ):
        """Describes an index by its geometry, angular over `BF16` rows unless told otherwise.

        Args:
            dimensions: Dimensions per vector.
            metric: The metric kind the graph is frozen under.
            quantization: The scalar type vectors are stored as.
            connectivity: The connections-per-node bound; zero derives.
            values_per_key: Upper bound on entries sharing one key; zero derives to one.
        """
        self.metric = metric
        self.quantization = quantization
        self.dimensions = c_size_t(dimensions)
        self.connectivity = c_size_t(connectivity)
        self.values_per_key = c_size_t(values_per_key)


@fieldwise_init
struct DenseMetadata(Defaultable, ImplicitlyCopyable):
    """The geometry and metric a checkpoint froze - `usearch_dense_metadata_t`, a pure output.

    Everything a reader learns without opening the checkpoint, and nothing a reader must choose for
    itself. Placement is absent because a checkpoint records where it was written rather than where
    a reader must keep it, and `values_per_key` is absent because it bounds a live handle's mutable
    state rather than anything the serialized form carries.

    Never marked register-passable: its address crosses to C, and Mojo 1.1 mislays a struct once it
    is, so the probes would read back nothing.
    """

    var metric: DenseMetric
    """The metric kind the checkpoint scores with."""
    var quantization: ScalarKind
    """The scalar type its rows are stored and quantized into."""
    var dimensions: c_size_t
    """Scalars per stored vector."""
    var connectivity: c_size_t
    """Connections per node the graph was built with."""

    def __init__(out self):
        """Names no kind and no geometry, so an unwritten slot reads as an unanswered probe."""
        self.metric = DenseMetric.UNKNOWN
        self.quantization = ScalarKind.UNKNOWN
        self.dimensions = c_size_t(0)
        self.connectivity = c_size_t(0)


@fieldwise_init
struct DenseIndexDescription(Copyable, Movable):
    """`usearch_dense_index_description_t`: a live index's frozen metadata and current counters.

    Each counter is read once, so under concurrent insertions they may come from slightly
    different moments.
    """

    var metadata: DenseMetadata
    """The frozen metric, quantization, dimensions and connectivity, as a checkpoint reports them."""
    var size: c_size_t
    """Stored entries."""
    var capacity: c_size_t
    """Entries hosted without growing again."""
    var memory_usage: c_size_t
    """Bytes the handle holds; traversal scratch is the executor's."""
    var capabilities: String
    """Codename of the kernels the index dispatched to on this machine."""


@fieldwise_init
struct DenseIndexDescriptionOutputs(Defaultable, ImplicitlyCopyable):
    """`usearch_dense_index_description_t`, laid out as the ABI declares it."""

    var metadata: DenseMetadata
    var size: c_size_t
    var capacity: c_size_t
    var memory_usage: c_size_t
    var capabilities: Address

    def __init__(out self):
        """Zeroes every field, so an unwritten slot reads as an empty index."""
        self.metadata = DenseMetadata()
        self.size = c_size_t(0)
        self.capacity = c_size_t(0)
        self.memory_usage = c_size_t(0)
        self.capabilities = NULL_ADDRESS


@fieldwise_init
struct DenseVectors[dtype: DType, origin: MutOrigin](Movable, Sized):
    """Rows a call wrote into its arena - `usearch_rows_view_t` - read in place: the entries `get`
    exports, or the rows `attend` attends.

    Rows are `stride` bytes apart and `columns` values wide, both as the engine reported them, so a
    padded row never reads as extra values.

    Parameters:
        dtype: The element type each row is read as.
        origin: The arena the rows live in, which this result borrows.
    """

    var arena: Pointer[Arena, Self.origin]
    """The arena holding the rows."""
    var generation: Int
    """The arena's call these rows came from, so a read after a later call is refused."""
    var view: RowsViewArguments
    """The first row, the stride, the width and the kind, as the engine wrote them."""
    var rows: Int
    """Rows written; zero when the key is absent."""

    def __len__(self) -> Int:
        """The rows written.

        Returns:
            Zero when the key is absent.
        """
        return self.rows

    def __getitem__(ref self, row: Int) raises Error -> Span[Scalar[Self.dtype], origin_of(self)]:
        """One row, borrowed from the arena this result reads. Binding-only.

        Args:
            row: Which row, below `len(self)`.

        Returns:
            The row's elements - `columns` of them, or its packed bytes for a sub-byte kind.

        Raises:
            `Error` with `UNINITIALIZED` once a later call rewound the arena, or `DEVICE` over VRAM.
        """
        _readable(self.arena, self.generation, "DenseVectors.__getitem__")
        return _in_arena[Scalar[Self.dtype], origin_of(self)](
            self.view.data + row * Int(self.view.stride),
            self.view.scalar.bytes_for_columns(Int(self.view.columns)) // size_of[Scalar[Self.dtype]](),
        )

    def columns(self) -> Int:
        """Values in each row, as the engine reported them.

        Returns:
            The width the rows were written at.
        """
        return Int(self.view.columns)


@fieldwise_init
struct DenseBatch[origin: MutOrigin](Movable, Sized):
    """One cursor pull, read in place from its arena: the keys, and the keys each links to when asked for.

    Parameters:
        origin: The arena the batch lives in, which this result borrows.
    """

    var arena: Pointer[Arena, Self.origin]
    """The arena holding the batch."""
    var generation: Int
    """The arena's call this batch came from, so a read after a later call is refused."""
    var keys_address: Address
    """The first key, inside the arena."""
    var linked: Bool
    """Whether the pull exported links."""
    var links_view: KeysTapeViewArguments
    """The keys each entry links to and their offsets, null for a pull of keys alone."""
    var written: Int
    """Entries the pull wrote; zero when the walk is exhausted."""

    def __len__(self) -> Int:
        """The entries the pull wrote.

        Returns:
            Zero when the walk is exhausted.
        """
        return self.written

    def keys(ref self) raises Error -> Span[Key, origin_of(self)]:
        """Every entry's key, in walk order.

        Returns:
            The keys, borrowed from the arena this result reads.

        Raises:
            `Error` with `UNINITIALIZED` once a later call rewound the arena, or `DEVICE` over VRAM.
        """
        _readable(self.arena, self.generation, "DenseBatch.keys")
        return _in_arena[Key, origin_of(self)](self.keys_address, self.written)

    def links(ref self, entry: Int) raises Error -> Span[Key, origin_of(self)]:
        """The keys one entry links to, as the graph stored them when the pull ran. Binding-only.

        Args:
            entry: Which entry, below `len(self)`.

        Returns:
            The linked keys, borrowed from the arena this result reads.

        Raises:
            `Error` with `INVALID_ARGUMENT` for a pull of keys alone or an entry past the batch, with
            `UNINITIALIZED` once a later call rewound the arena, or `DEVICE` over VRAM.
        """
        _readable(self.arena, self.generation, "DenseBatch.links")
        if not self.linked or entry < 0 or entry >= self.written:
            raise Error(Status.INVALID_ARGUMENT, "DenseBatch.links")
        var offsets = _in_arena[c_size_t, origin_of(self)](self.links_view.offsets, self.written + 1)
        var begin = Int(offsets[entry])
        return _in_arena[Key, origin_of(self)](
            self.links_view.keys + begin * size_of[Key](), Int(offsets[entry + 1]) - begin
        )


@fieldwise_init
struct DenseProducts[origin: MutOrigin](Movable):
    """The ragged output of `multiply` - `usearch_matches_tape_view_t` - read in place from its arena:
    row `i` spans pairs `offsets()[i]` up to `offsets()[i + 1]` of `keys()` and `scores()`.

    Parameters:
        origin: The arena the pairs live in, which this result borrows.
    """

    var arena: Pointer[Arena, Self.origin]
    """The arena holding the pairs."""
    var generation: Int
    """The arena's call these pairs came from, so a read after a later call is refused."""
    var view: MatchesTapeViewArguments
    """The keys, products and offsets, as the engine wrote them."""
    var rows: Int
    """Left-hand rows the call multiplied."""
    var pairs: Int
    """Pairs written across every row."""
    var required: Int
    """Pairs the walks scored before `wanted` cut any; above `pairs` means rows were truncated."""

    def offsets(ref self) raises Error -> Span[c_size_t, origin_of(self)]:
        """One more offset than left-hand rows, as Apache Arrow lays a ragged batch out.

        Returns:
            The offsets into `keys()` and `scores()`, borrowed from the arena this result reads.

        Raises:
            `Error` with `UNINITIALIZED` once a later call rewound the arena, or `DEVICE` over VRAM.
        """
        _readable(self.arena, self.generation, "DenseProducts.offsets")
        return _in_arena[c_size_t, origin_of(self)](self.view.offsets, self.rows + 1)

    def keys(ref self) raises Error -> Span[Key, origin_of(self)]:
        """Each pair's indexed row, borrowed from the arena this result reads.

        Returns:
            One key per pair.

        Raises:
            `Error` with `UNINITIALIZED` once a later call rewound the arena, or `DEVICE` over VRAM.
        """
        _readable(self.arena, self.generation, "DenseProducts.keys")
        return _in_arena[Key, origin_of(self)](self.view.keys, self.pairs)

    def scores(ref self) raises Error -> Span[Distance, origin_of(self)]:
        """Each pair's product, positionally paired with `keys()`.

        Returns:
            One score per pair.

        Raises:
            `Error` with `UNINITIALIZED` once a later call rewound the arena, or `DEVICE` over VRAM.
        """
        _readable(self.arena, self.generation, "DenseProducts.scores")
        return _in_arena[Distance, origin_of(self)](self.view.scores, self.pairs)


# endregion Configuration

# region Configs


def _stats_address[origin: MutOrigin](stats: Optional[Pointer[DenseTraversalStats, origin]]) -> Address:
    """The telemetry slot a config lends, or `NULL_ADDRESS` to skip reporting."""
    return Int(stats.value()) if stats else NULL_ADDRESS


struct DenseInsertConfig[stats_origin: MutOrigin = MutUntrackedOrigin](ImplicitlyCopyable):
    """The knobs of `usearch_dense_insert_t`; zeroed or absent ones derive what the C interface derives.

    Parameters:
        stats_origin: The lent telemetry slot's origin, inferred where one is lent.
    """

    var expansion_factor: Int
    """The breadth of the candidates the build walk explores; zero derives."""
    var stats: Optional[Pointer[DenseTraversalStats, Self.stats_origin]]
    """Receives this call's traversal telemetry, fully overwritten; absent skips reporting."""

    def __init__(
        out self,
        *,
        expansion_factor: Int = 0,
        stats: Optional[Pointer[DenseTraversalStats, Self.stats_origin]] = Optional[
            Pointer[DenseTraversalStats, Self.stats_origin]
        ](),
    ):
        """Names only the knobs that differ from the derived ones.

        Args:
            expansion_factor: The breadth of the candidates the build walk explores; zero derives.
            stats: Receives this call's traversal telemetry; absent skips reporting.
        """
        self.expansion_factor = expansion_factor
        self.stats = stats


struct DenseRefineConfig[stats_origin: MutOrigin = MutUntrackedOrigin](ImplicitlyCopyable):
    """The knobs of `usearch_dense_refine_t`; zeroed or absent ones derive what the C interface derives.

    Parameters:
        stats_origin: The lent telemetry slot's origin, inferred where one is lent.
    """

    var expansion_factor: Int
    """The breadth of the candidates the re-selection walk explores; zero derives."""
    var stats: Optional[Pointer[DenseTraversalStats, Self.stats_origin]]
    """Receives this call's traversal telemetry, fully overwritten; absent skips reporting."""

    def __init__(
        out self,
        *,
        expansion_factor: Int = 0,
        stats: Optional[Pointer[DenseTraversalStats, Self.stats_origin]] = Optional[
            Pointer[DenseTraversalStats, Self.stats_origin]
        ](),
    ):
        """Names only the knobs that differ from the derived ones.

        Args:
            expansion_factor: The breadth of the candidates the re-selection walk explores; zero derives.
            stats: Receives this call's traversal telemetry; absent skips reporting.
        """
        self.expansion_factor = expansion_factor
        self.stats = stats


struct DenseSearchConfig[stats_origin: MutOrigin = MutUntrackedOrigin](ImplicitlyCopyable):
    """The knobs of `usearch_dense_search_t`; zeroed or absent ones derive what the C interface derives.

    A `predicate` narrows the matches to the entries it admits: the address of a
    `usearch_predicate_callback_t` compiled in C, which installs unchanged. Non-matching entries still
    route the traversal, so the predicate judges every scored candidate, a quad of them per call,
    concurrently from the executor's threads.

    Parameters:
        stats_origin: The lent telemetry slot's origin, inferred where one is lent.
    """

    var expansion_factor: Int
    """The breadth of the candidates the walk explores; zero derives."""
    var traversal: Traversal
    """A walk, or an exact scan of every row on RAM and NVMe handles; VRAM answers `UNSUPPORTED`."""
    var predicate: Address
    """A `usearch_predicate_callback_t` that must outlive the call; `NULL_ADDRESS` admits all."""
    var predicate_state: Address
    """The opaque state forwarded into every predicate call."""
    var stats: Optional[Pointer[DenseTraversalStats, Self.stats_origin]]
    """Receives this call's traversal telemetry, fully overwritten; absent skips reporting."""

    def __init__(
        out self,
        *,
        expansion_factor: Int = 0,
        traversal: Traversal = Traversal.WALK,
        predicate: Address = NULL_ADDRESS,
        predicate_state: Address = NULL_ADDRESS,
        stats: Optional[Pointer[DenseTraversalStats, Self.stats_origin]] = Optional[
            Pointer[DenseTraversalStats, Self.stats_origin]
        ](),
    ):
        """Names only the knobs that differ from the derived ones.

        Args:
            expansion_factor: The breadth of the candidates the walk explores; zero derives.
            traversal: A walk, or an exact scan of every row on RAM and NVMe handles.
            predicate: A `usearch_predicate_callback_t` that must outlive the call; `NULL_ADDRESS` admits all.
            predicate_state: The opaque state forwarded into every predicate call.
            stats: Receives this call's traversal telemetry; absent skips reporting.
        """
        self.expansion_factor = expansion_factor
        self.traversal = traversal
        self.predicate = predicate
        self.predicate_state = predicate_state
        self.stats = stats


struct DenseMaxsimConfig[stats_origin: MutOrigin = MutUntrackedOrigin](ImplicitlyCopyable):
    """The knobs of `usearch_dense_maxsim_t`; zeroed or absent ones derive what the C interface derives.

    Parameters:
        stats_origin: The lent telemetry slot's origin, inferred where one is lent.
    """

    var expansion_factor: Int
    """The breadth of the candidates; zero derives."""
    var stats: Optional[Pointer[DenseTraversalStats, Self.stats_origin]]
    """Receives this call's traversal telemetry, fully overwritten; absent skips reporting."""

    def __init__(
        out self,
        *,
        expansion_factor: Int = 0,
        stats: Optional[Pointer[DenseTraversalStats, Self.stats_origin]] = Optional[
            Pointer[DenseTraversalStats, Self.stats_origin]
        ](),
    ):
        """Names only the knobs that differ from the derived ones.

        Args:
            expansion_factor: The breadth of the candidates; zero derives.
            stats: Receives this call's traversal telemetry; absent skips reporting.
        """
        self.expansion_factor = expansion_factor
        self.stats = stats


struct DenseMultiplyConfig[stats_origin: MutOrigin = MutUntrackedOrigin](ImplicitlyCopyable):
    """The knobs of `usearch_dense_multiply_t`; zeroed or absent ones derive what the C interface derives.

    Parameters:
        stats_origin: The lent telemetry slot's origin, inferred where one is lent.
    """

    var wanted: Int
    """Pairs kept per row, the first ones its walk scored; zero keeps every pair."""
    var expansion_factor: Int
    """The breadth of the candidates, which widens the emitted pattern; zero derives."""
    var stats: Optional[Pointer[DenseTraversalStats, Self.stats_origin]]
    """Receives this call's traversal telemetry, fully overwritten; absent skips reporting."""

    def __init__(
        out self,
        *,
        wanted: Int = 0,
        expansion_factor: Int = 0,
        stats: Optional[Pointer[DenseTraversalStats, Self.stats_origin]] = Optional[
            Pointer[DenseTraversalStats, Self.stats_origin]
        ](),
    ):
        """Names only the knobs that differ from the derived ones.

        Args:
            wanted: Pairs kept per row, the first ones its walk scored; zero keeps every pair.
            expansion_factor: The breadth of the candidates, which widens the emitted pattern; zero derives.
            stats: Receives this call's traversal telemetry; absent skips reporting.
        """
        self.wanted = wanted
        self.expansion_factor = expansion_factor
        self.stats = stats


struct DenseAttendConfig[carry_origin: MutOrigin = MutUntrackedOrigin, stats_origin: MutOrigin = MutUntrackedOrigin](
    ImplicitlyCopyable
):
    """The knobs of `usearch_dense_attend_t`; zeroed or absent ones derive what the C interface derives.

    Parameters:
        carry_origin: The lent softmax states' origin, inferred where any are lent.
        stats_origin: The lent telemetry slot's origin, inferred where one is lent.
    """

    var head_count: Int
    """The query heads sharing the index's key-value head; it must divide the rows, and zero means one."""
    var carry: Span[DenseSoftmaxState, Self.carry_origin]
    """The softmax states, one per query row; empty skips them."""
    var expansion_factor: Int
    """The breadth of the candidates; zero derives. Wider candidates walk further, folding more of the tail."""
    var score_scale: Float32
    """The logit scale, conventionally the inverse square root of the head width; zero derives it."""
    var stats: Optional[Pointer[DenseTraversalStats, Self.stats_origin]]
    """Receives this call's traversal telemetry, fully overwritten; absent skips reporting."""

    def __init__(
        out self,
        *,
        head_count: Int = 0,
        carry: Span[DenseSoftmaxState, Self.carry_origin] = Span[DenseSoftmaxState, Self.carry_origin](),
        expansion_factor: Int = 0,
        score_scale: Float32 = 0,
        stats: Optional[Pointer[DenseTraversalStats, Self.stats_origin]] = Optional[
            Pointer[DenseTraversalStats, Self.stats_origin]
        ](),
    ):
        """Names only the knobs that differ from the derived ones.

        Args:
            head_count: The query heads sharing the index's key-value head; zero means one.
            carry: The softmax states, one per query row; empty skips them.
            expansion_factor: The breadth of the candidates; zero derives.
            score_scale: The logit scale; zero derives it from the index's frozen geometry.
            stats: Receives this call's traversal telemetry; absent skips reporting.
        """
        self.head_count = head_count
        self.carry = carry
        self.expansion_factor = expansion_factor
        self.score_scale = score_scale
        self.stats = stats


struct DenseSearchStreamInitConfig(ImplicitlyCopyable):
    """The knobs of `usearch_dense_search_stream_init_t`; zeroed ones derive what the C interface derives."""

    var expansion_factor: Int
    """The breadth of the candidates the walk explores; zero derives."""

    def __init__(out self, *, expansion_factor: Int = 0):
        """Names only the knobs that differ from the derived ones.

        Args:
            expansion_factor: The breadth of the candidates the walk explores; zero derives.
        """
        self.expansion_factor = expansion_factor


# endregion Configs

# region Arguments


@fieldwise_init
struct DenseInitRamArguments(ImplicitlyCopyable):
    """`usearch_dense_init_ram_t`."""

    var index: Address
    var outcome: Address
    var metric: DenseMetric
    var quantization: ScalarKind
    var dimensions: c_size_t
    var connectivity: c_size_t
    var values_per_key: c_size_t
    var ram: RamOptions[ImmUntrackedOrigin]


@fieldwise_init
struct DenseInitVramArguments(ImplicitlyCopyable):
    """`usearch_dense_init_vram_t`."""

    var index: Address
    var outcome: Address
    var metric: DenseMetric
    var quantization: ScalarKind
    var dimensions: c_size_t
    var connectivity: c_size_t
    var values_per_key: c_size_t
    var vram: VramOptions[ImmUntrackedOrigin]


@fieldwise_init
struct DenseLoadRamArguments(ImplicitlyCopyable):
    """`usearch_dense_load_ram_t`, and `usearch_dense_load_ram_buffer_t` with `source` a buffer."""

    var index: Address
    var executor: Address
    var outcome: Address
    var completion: Completion
    var source: BytesView[ImmUntrackedOrigin]
    var ram: RamOptions[ImmUntrackedOrigin]
    var values_per_key: c_size_t
    var progress: ProgressArguments


@fieldwise_init
struct DenseLoadVramArguments(ImplicitlyCopyable):
    """`usearch_dense_load_vram_t`, and `usearch_dense_load_vram_buffer_t` with `source` a buffer."""

    var index: Address
    var executor: Address
    var outcome: Address
    var completion: Completion
    var source: BytesView[ImmUntrackedOrigin]
    var vram: VramOptions[ImmUntrackedOrigin]
    var values_per_key: c_size_t
    var progress: ProgressArguments


@fieldwise_init
struct DenseViewRamArguments(ImplicitlyCopyable):
    """`usearch_dense_view_ram_t`."""

    var index: Address
    var outcome: Address
    var path: BytesView[ImmUntrackedOrigin]


@fieldwise_init
struct DenseViewNvmeArguments(ImplicitlyCopyable):
    """`usearch_dense_view_nvme_t`."""

    var index: Address
    var executor: Address
    var outcome: Address
    var completion: Completion
    var path: BytesView[ImmUntrackedOrigin]
    var nvme: NvmeOptions[ImmUntrackedOrigin]


@fieldwise_init
struct DenseReserveArguments(ImplicitlyCopyable):
    """`usearch_dense_reserve_t`."""

    var index: Address
    var executor: Address
    var outcome: Address
    var completion: Completion
    var capacity: c_size_t


@fieldwise_init
struct DenseClearArguments(ImplicitlyCopyable):
    """`usearch_dense_clear_t`."""

    var index: Address
    var executor: Address
    var outcome: Address
    var completion: Completion


@fieldwise_init
struct DenseDescribeArguments(ImplicitlyCopyable):
    """`usearch_dense_describe_t`."""

    var index: Address
    var executor: Address
    var outcome: Address
    var completion: Completion
    var description: Address


@fieldwise_init
struct DenseInsertArguments(ImplicitlyCopyable):
    """`usearch_dense_insert_t`."""

    var index: Address
    var executor: Address
    var outcome: Address
    var completion: Completion
    var count: c_size_t
    var keys: KeysView[ImmUntrackedOrigin]
    var vectors: RowsViewArguments
    var expansion_factor: c_size_t
    var progress: ProgressArguments
    var stats: Address


@fieldwise_init
struct DenseRefineArguments(ImplicitlyCopyable):
    """`usearch_dense_refine_t`."""

    var index: Address
    var executor: Address
    var outcome: Address
    var completion: Completion
    var changed: Address
    var reachability: Address
    var expansion_factor: c_size_t
    var progress: ProgressArguments
    var stats: Address


@fieldwise_init
struct DenseHealthArguments(ImplicitlyCopyable):
    """`usearch_dense_health_t`."""

    var index: Address
    var executor: Address
    var outcome: Address
    var completion: Completion
    var reachability: Address


@fieldwise_init
struct DenseAbsorbArguments(ImplicitlyCopyable):
    """`usearch_dense_absorb_t`."""

    var index: Address
    var executor: Address
    var outcome: Address
    var completion: Completion
    var count: c_size_t
    var donors: Address
    var changed: Address
    var reachability: Address
    var expansion_factor: c_size_t
    var progress: ProgressArguments
    var stats: Address


@fieldwise_init
struct DenseSearchArguments(ImplicitlyCopyable):
    """`usearch_dense_search_t`."""

    var index: Address
    var executor: Address
    var outcome: Address
    var completion: Completion
    var count: c_size_t
    var queries: RowsViewArguments
    var matches: Address
    var arena: Address
    var wanted: c_size_t
    var expansion_factor: c_size_t
    var traversal: Traversal
    var predicate: PredicateArguments
    var progress: ProgressArguments
    var stats: Address


@fieldwise_init
struct DenseCountArguments(ImplicitlyCopyable):
    """`usearch_dense_count_t`."""

    var index: Address
    var executor: Address
    var outcome: Address
    var completion: Completion
    var key: KeyBytes
    var entries: Address


@fieldwise_init
struct DenseGetArguments(ImplicitlyCopyable):
    """`usearch_dense_get_t`."""

    var index: Address
    var executor: Address
    var outcome: Address
    var completion: Completion
    var key: KeyBytes
    var vectors: Address
    var written: Address
    var arena: Address
    var outputs_scalar: ScalarKind


@fieldwise_init
struct DenseCursorInitArguments(ImplicitlyCopyable):
    """`usearch_dense_cursor_init_t`."""

    var cursor: Address
    var index: Address
    var outcome: Address
    var starts_count: c_size_t
    var starts: Address


@fieldwise_init
struct DenseCursorNextArguments(ImplicitlyCopyable):
    """`usearch_dense_cursor_next_t`."""

    var index: Address
    var executor: Address
    var outcome: Address
    var completion: Completion
    var cursor: Address
    var keys: Address
    var links: Address
    var written: Address
    var arena: Address
    var wanted: c_size_t


@fieldwise_init
struct DenseSearchStreamInitArguments(ImplicitlyCopyable):
    """`usearch_dense_search_stream_init_t`."""

    var stream: Address
    var index: Address
    var outcome: Address
    var query: RowsViewArguments
    var expansion_factor: c_size_t


@fieldwise_init
struct DenseSearchStreamNextArguments(ImplicitlyCopyable):
    """`usearch_dense_search_stream_next_t`."""

    var index: Address
    var executor: Address
    var outcome: Address
    var completion: Completion
    var stream: Address
    var matches: Address
    var arena: Address
    var wanted: c_size_t


@fieldwise_init
struct DenseSaveArguments(ImplicitlyCopyable):
    """`usearch_dense_save_t`."""

    var index: Address
    var executor: Address
    var outcome: Address
    var completion: Completion
    var path: BytesView[ImmUntrackedOrigin]
    var progress: ProgressArguments


@fieldwise_init
struct DenseSaveBufferArguments(ImplicitlyCopyable):
    """`usearch_dense_save_buffer_t`."""

    var index: Address
    var executor: Address
    var outcome: Address
    var completion: Completion
    var monolith: Address
    var arena: Address
    var progress: ProgressArguments


@fieldwise_init
struct DenseMetadataReadArguments(ImplicitlyCopyable):
    """`usearch_dense_metadata_read_t`, and `usearch_dense_metadata_read_buffer_t` with `source` a
    buffer."""

    var executor: Address
    var outcome: Address
    var completion: Completion
    var source: BytesView[ImmUntrackedOrigin]
    var metadata: Address


@fieldwise_init
struct DenseMultiplyArguments(ImplicitlyCopyable):
    """`usearch_dense_multiply_t`."""

    var index: Address
    var executor: Address
    var outcome: Address
    var completion: Completion
    var count: c_size_t
    var left: RowsViewArguments
    var products: Address
    var written: Address
    var required: Address
    var arena: Address
    var wanted: c_size_t
    var expansion_factor: c_size_t
    var progress: ProgressArguments
    var stats: Address


@fieldwise_init
struct DenseAttendArguments(ImplicitlyCopyable):
    """`usearch_dense_attend_t`."""

    var index: Address
    var executor: Address
    var outcome: Address
    var completion: Completion
    var count: c_size_t
    var queries: RowsViewArguments
    var head_count: c_size_t
    var value_lookup: ValueLookup
    var value_lookup_state: Address
    var carry: Address
    var outputs: Address
    var arena: Address
    var outputs_scalar: ScalarKind
    var expansion_factor: c_size_t
    var score_scale: Float32
    var progress: ProgressArguments
    var stats: Address


@fieldwise_init
struct DenseMaxsimArguments(ImplicitlyCopyable):
    """`usearch_dense_maxsim_t`."""

    var index: Address
    var executor: Address
    var outcome: Address
    var completion: Completion
    var count: c_size_t
    var queries: RowsViewArguments
    var rows_per_query: c_size_t
    var matches: Address
    var arena: Address
    var wanted: c_size_t
    var expansion_factor: c_size_t
    var progress: ProgressArguments
    var stats: Address


comptime DenseInitRam = Verb[DenseInitRamArguments]
comptime DenseInitVram = Verb[DenseInitVramArguments]
comptime DenseLoadRam = Verb[DenseLoadRamArguments]
comptime DenseLoadRamBuffer = Verb[DenseLoadRamArguments]
comptime DenseLoadVram = Verb[DenseLoadVramArguments]
comptime DenseLoadVramBuffer = Verb[DenseLoadVramArguments]
comptime DenseViewRam = Verb[DenseViewRamArguments]
comptime DenseViewNvme = Verb[DenseViewNvmeArguments]
comptime DenseReserve = Verb[DenseReserveArguments]
comptime DenseClear = Verb[DenseClearArguments]
comptime DenseDescribe = Verb[DenseDescribeArguments]
comptime DenseInsert = Verb[DenseInsertArguments]
comptime DenseRefine = Verb[DenseRefineArguments]
comptime DenseHealth = Verb[DenseHealthArguments]
comptime DenseAbsorb = Verb[DenseAbsorbArguments]
comptime DenseSearch = Verb[DenseSearchArguments]
comptime DenseCount = Verb[DenseCountArguments]
comptime DenseGet = Verb[DenseGetArguments]
comptime DenseCursorInit = Verb[DenseCursorInitArguments]
comptime DenseCursorNext = Verb[DenseCursorNextArguments]
comptime DenseSearchStreamInit = Verb[DenseSearchStreamInitArguments]
comptime DenseSearchStreamNext = Verb[DenseSearchStreamNextArguments]
comptime DenseSave = Verb[DenseSaveArguments]
comptime DenseSaveBuffer = Verb[DenseSaveBufferArguments]
comptime DenseMetadataRead = Verb[DenseMetadataReadArguments]
comptime DenseMetadataReadBuffer = Verb[DenseMetadataReadArguments]
comptime DenseMultiply = Verb[DenseMultiplyArguments]
comptime DenseAttend = Verb[DenseAttendArguments]
comptime DenseMaxsim = Verb[DenseMaxsimArguments]


# endregion Arguments

# region Calls


def _cross[Arguments: ImplicitlyCopyable & Deinitable, //](arguments: Arguments, verb: Verb[Arguments]):
    """Materializes one argument structure in a stack slot and hands its address to `verb`."""
    var slot = stack_allocation[1, Arguments]()
    slot[unsafe_offset=0] = arguments
    verb(slot.unsafe_origin_cast[MutAnyOrigin]())


@fieldwise_init
struct DenseCountOutputs(Defaultable, ImplicitlyCopyable):
    """The one `size_t` `usearch_dense_count` and `usearch_dense_refine` write."""

    var value: c_size_t

    def __init__(out self):
        """Zero, which is what a failed call leaves."""
        self.value = c_size_t(0)


def _no_rows() -> RowsViewArguments:
    """A rows view nothing was written to, which is what a failed call leaves."""
    return RowsViewArguments(NULL_ADDRESS, c_size_t(0), c_size_t(0), ScalarKind.UNKNOWN)


@fieldwise_init
struct DenseGetOutputs(Defaultable, ImplicitlyCopyable):
    """What `usearch_dense_get` writes through its pointers."""

    var vectors: RowsViewArguments
    var written: c_size_t

    def __init__(out self):
        """Nothing exported."""
        self.vectors = _no_rows()
        self.written = c_size_t(0)


@fieldwise_init
struct DenseCursorNextOutputs(Defaultable, ImplicitlyCopyable):
    """What `usearch_dense_cursor_next` writes through its pointers; the keys are a `usearch_keys_view_t`."""

    var keys: Address
    var keys_stride: c_size_t
    var links: KeysTapeViewArguments
    var written: c_size_t

    def __init__(out self):
        """Nothing pulled."""
        self.keys = NULL_ADDRESS
        self.keys_stride = c_size_t(0)
        self.links = KeysTapeViewArguments()
        self.written = c_size_t(0)


@fieldwise_init
struct DenseMultiplyOutputs(Defaultable, ImplicitlyCopyable):
    """What `usearch_dense_multiply` writes through its pointers."""

    var products: MatchesTapeViewArguments
    var written: c_size_t
    var required: c_size_t

    def __init__(out self):
        """No pairs."""
        self.products = MatchesTapeViewArguments()
        self.written = c_size_t(0)
        self.required = c_size_t(0)


@fieldwise_init
struct DenseAttendOutputs(Defaultable, ImplicitlyCopyable):
    """What `usearch_dense_attend` writes through its pointer."""

    var outputs: RowsViewArguments

    def __init__(out self):
        """No rows attended."""
        self.outputs = _no_rows()


@fieldwise_init
struct DenseSaveBufferOutputs(Defaultable, ImplicitlyCopyable):
    """What `usearch_dense_save_buffer` writes through its pointer: a `usearch_bytes_view_t`."""

    var data: Address
    var length: c_size_t

    def __init__(out self):
        """No bytes."""
        self.data = NULL_ADDRESS
        self.length = c_size_t(0)


@fieldwise_init
struct _DenseReserveCall(Call):
    """The crossing of `usearch_dense_reserve`."""

    comptime Outputs = Unwritten
    comptime Result = NoneType
    comptime symbol: StaticString = "usearch_dense_reserve"
    var capacity: Int

    def submit(
        mut self,
        library: Library,
        index: Address,
        executor: Address,
        completion: Completion,
        outcome: Address,
        outputs: Address,
    ):
        _cross(
            DenseReserveArguments(index, executor, outcome, completion, c_size_t(self.capacity)),
            library.symbols().dense_reserve,
        )

    def conclude(deinit self, outputs: Unwritten, generation: Int) -> NoneType:
        return None


@fieldwise_init
struct _DenseClearCall(Call):
    """The crossing of `usearch_dense_clear`."""

    comptime Outputs = Unwritten
    comptime Result = NoneType
    comptime symbol: StaticString = "usearch_dense_clear"

    def submit(
        mut self,
        library: Library,
        index: Address,
        executor: Address,
        completion: Completion,
        outcome: Address,
        outputs: Address,
    ):
        _cross(DenseClearArguments(index, executor, outcome, completion), library.symbols().dense_clear)

    def conclude(deinit self, outputs: Unwritten, generation: Int) -> NoneType:
        return None


@fieldwise_init
struct _DenseDescribeCall(Call):
    """The crossing of `usearch_dense_describe`."""

    comptime Outputs = DenseIndexDescriptionOutputs
    comptime Result = DenseIndexDescription
    comptime symbol: StaticString = "usearch_dense_describe"

    def submit(
        mut self,
        library: Library,
        index: Address,
        executor: Address,
        completion: Completion,
        outcome: Address,
        outputs: Address,
    ):
        _cross(DenseDescribeArguments(index, executor, outcome, completion, outputs), library.symbols().dense_describe)

    def conclude(deinit self, outputs: DenseIndexDescriptionOutputs, generation: Int) -> DenseIndexDescription:
        return DenseIndexDescription(
            outputs.metadata, outputs.size, outputs.capacity, outputs.memory_usage, text_at(outputs.capabilities)
        )


@fieldwise_init
struct _DenseCountCall(Call):
    """The crossing of `usearch_dense_count`."""

    comptime Outputs = DenseCountOutputs
    comptime Result = Int
    comptime symbol: StaticString = "usearch_dense_count"
    var key: Key

    def submit(
        mut self,
        library: Library,
        index: Address,
        executor: Address,
        completion: Completion,
        outcome: Address,
        outputs: Address,
    ):
        _cross(
            DenseCountArguments(index, executor, outcome, completion, KeyBytes.of(self.key), outputs),
            library.symbols().dense_count,
        )

    def conclude(deinit self, outputs: DenseCountOutputs, generation: Int) -> Int:
        return Int(outputs.value)


@fieldwise_init
struct _DenseGetCall[dtype: DType, origin: MutOrigin](Call):
    """The crossing of `usearch_dense_get`, which borrows the arena its rows land in."""

    comptime Outputs = DenseGetOutputs
    comptime Result = DenseVectors[Self.dtype, Self.origin]
    comptime symbol: StaticString = "usearch_dense_get"
    var key: Key
    var scalar: ScalarKind
    var arena: Pointer[Arena, Self.origin]

    def submit(
        mut self,
        library: Library,
        index: Address,
        executor: Address,
        completion: Completion,
        outcome: Address,
        outputs: Address,
    ):
        if not self.scalar.stores[Self.dtype]():
            return _refuse(outcome, Status.SHAPE_MISMATCH)
        var landing = Pointer[DenseGetOutputs, MutAnyOrigin](unsafe_from_address=outputs)
        _cross(
            DenseGetArguments(
                index,
                executor,
                outcome,
                completion,
                KeyBytes.of(self.key),
                Int(Pointer(to=landing[].vectors)),
                Int(Pointer(to=landing[].written)),
                self.arena[]._address,
                self.scalar,
            ),
            library.symbols().dense_get,
        )

    def served(self, status: Status) -> Int:
        return _served(self.arena, status)

    def conclude(deinit self, outputs: DenseGetOutputs, generation: Int) -> Self.Result:
        return Self.Result(self.arena, generation, outputs.vectors, Int(outputs.written))


@fieldwise_init
struct _DenseInsertCall(Call):
    """The crossing of `usearch_dense_insert`."""

    comptime Outputs = Unwritten
    comptime Result = NoneType
    comptime symbol: StaticString = "usearch_dense_insert"
    var keys: KeysView[ImmUntrackedOrigin]
    var keys_count: Int
    var vectors: RowsViewArguments
    var count: Int
    var expansion_factor: Int
    var stats: Address

    def submit(
        mut self,
        library: Library,
        index: Address,
        executor: Address,
        completion: Completion,
        outcome: Address,
        outputs: Address,
    ):
        if self.keys_count != self.count:
            return _refuse(outcome, Status.SHAPE_MISMATCH)
        _cross(
            DenseInsertArguments(
                index,
                executor,
                outcome,
                completion,
                c_size_t(self.count),
                self.keys,
                self.vectors,
                c_size_t(self.expansion_factor),
                ProgressArguments(),
                self.stats,
            ),
            library.symbols().dense_insert,
        )

    def conclude(deinit self, outputs: Unwritten, generation: Int) -> NoneType:
        return None


@fieldwise_init
struct _DenseRefineCall(Call):
    """The crossing of `usearch_dense_refine`."""

    comptime Outputs = DenseCountOutputs
    comptime Result = Int
    comptime symbol: StaticString = "usearch_dense_refine"
    var expansion_factor: Int
    var stats: Address

    def submit(
        mut self,
        library: Library,
        index: Address,
        executor: Address,
        completion: Completion,
        outcome: Address,
        outputs: Address,
    ):
        _cross(
            DenseRefineArguments(
                index,
                executor,
                outcome,
                completion,
                outputs,
                NULL_ADDRESS,
                c_size_t(self.expansion_factor),
                ProgressArguments(),
                self.stats,
            ),
            library.symbols().dense_refine,
        )

    def conclude(deinit self, outputs: DenseCountOutputs, generation: Int) -> Int:
        return Int(outputs.value)


@fieldwise_init
struct _DenseHealthCall(Call):
    """The crossing of `usearch_dense_health`."""

    comptime Outputs = DenseReachability
    comptime Result = DenseReachability
    comptime symbol: StaticString = "usearch_dense_health"

    def submit(
        mut self,
        library: Library,
        index: Address,
        executor: Address,
        completion: Completion,
        outcome: Address,
        outputs: Address,
    ):
        _cross(DenseHealthArguments(index, executor, outcome, completion, outputs), library.symbols().dense_health)

    def conclude(deinit self, outputs: DenseReachability, generation: Int) -> DenseReachability:
        return outputs


struct _DenseAbsorbCall(Call):
    """The crossing of `usearch_dense_absorb`, which owns the donors until the call is final."""

    comptime Outputs = Unwritten
    comptime Result = NoneType
    comptime symbol: StaticString = "usearch_dense_absorb"
    var donors: List[DenseIndex]
    var addresses: List[Address]

    def __init__(out self, var donors: List[DenseIndex]):
        self.addresses = List[Address](capacity=len(donors))
        for position in range(len(donors)):
            self.addresses.append(donors[position].address)
        self.donors = donors^

    def submit(
        mut self,
        library: Library,
        index: Address,
        executor: Address,
        completion: Completion,
        outcome: Address,
        outputs: Address,
    ):
        _cross(
            DenseAbsorbArguments(
                index,
                executor,
                outcome,
                completion,
                c_size_t(len(self.addresses)),
                Int(self.addresses.unsafe_ptr()),
                NULL_ADDRESS,
                NULL_ADDRESS,
                c_size_t(0),
                ProgressArguments(),
                NULL_ADDRESS,
            ),
            library.symbols().dense_absorb,
        )

    def conclude(deinit self, outputs: Unwritten, generation: Int) -> NoneType:
        # The engine nulled every donor it consumed, so only the survivors are freed with the list.
        var surviving = Pointer[Address, MutAnyOrigin](unsafe_from_address=Int(self.addresses.unsafe_ptr()))
        for position in range(len(self.donors)):
            self.donors[position].address = surviving[unsafe_offset=position]
        return None


@fieldwise_init
struct _DenseSearchCall[origin: MutOrigin](Call):
    """The crossing of `usearch_dense_search`, which borrows the arena its matches land in."""

    comptime Outputs = MatchesViewArguments
    comptime Result = Matches[Self.origin]
    comptime symbol: StaticString = "usearch_dense_search"
    var queries: RowsViewArguments
    var count: Int
    var wanted: Int
    var arena: Pointer[Arena, Self.origin]
    var expansion_factor: Int
    var traversal: Traversal
    var predicate: Address
    var predicate_state: Address
    var stats: Address

    def submit(
        mut self,
        library: Library,
        index: Address,
        executor: Address,
        completion: Completion,
        outcome: Address,
        outputs: Address,
    ):
        _cross(
            DenseSearchArguments(
                index,
                executor,
                outcome,
                completion,
                c_size_t(self.count),
                self.queries,
                outputs,
                self.arena[]._address,
                c_size_t(self.wanted),
                c_size_t(self.expansion_factor),
                self.traversal,
                PredicateArguments(self.predicate, self.predicate_state),
                ProgressArguments(),
                self.stats,
            ),
            library.symbols().dense_search,
        )

    def served(self, status: Status) -> Int:
        return _served(self.arena, status)

    def conclude(deinit self, outputs: MatchesViewArguments, generation: Int) -> Self.Result:
        return Self.Result(self.arena, generation, outputs, self.count)


@fieldwise_init
struct _DenseMaxsimCall[origin: MutOrigin](Call):
    """The crossing of `usearch_dense_maxsim`, which borrows the arena its matches land in."""

    comptime Outputs = MatchesViewArguments
    comptime Result = Matches[Self.origin]
    comptime symbol: StaticString = "usearch_dense_maxsim"
    var queries: RowsViewArguments
    var rows: Int
    var rows_per_query: Int
    var wanted: Int
    var arena: Pointer[Arena, Self.origin]
    var expansion_factor: Int
    var stats: Address

    def submit(
        mut self,
        library: Library,
        index: Address,
        executor: Address,
        completion: Completion,
        outcome: Address,
        outputs: Address,
    ):
        if self.rows_per_query <= 0 or self.rows % self.rows_per_query != 0:
            return _refuse(outcome, Status.SHAPE_MISMATCH)
        _cross(
            DenseMaxsimArguments(
                index,
                executor,
                outcome,
                completion,
                c_size_t(self.rows // self.rows_per_query),
                self.queries,
                c_size_t(self.rows_per_query),
                outputs,
                self.arena[]._address,
                c_size_t(self.wanted),
                c_size_t(self.expansion_factor),
                ProgressArguments(),
                self.stats,
            ),
            library.symbols().dense_maxsim,
        )

    def served(self, status: Status) -> Int:
        return _served(self.arena, status)

    def conclude(deinit self, outputs: MatchesViewArguments, generation: Int) -> Self.Result:
        var queries = self.rows // self.rows_per_query if self.rows_per_query > 0 else 0
        return Self.Result(self.arena, generation, outputs, queries)


@fieldwise_init
struct _DenseMultiplyCall[origin: MutOrigin](Call):
    """The crossing of `usearch_dense_multiply`, which borrows the arena its pairs land in."""

    comptime Outputs = DenseMultiplyOutputs
    comptime Result = DenseProducts[Self.origin]
    comptime symbol: StaticString = "usearch_dense_multiply"
    var left: RowsViewArguments
    var count: Int
    var arena: Pointer[Arena, Self.origin]
    var wanted: Int
    var expansion_factor: Int
    var stats: Address

    def submit(
        mut self,
        library: Library,
        index: Address,
        executor: Address,
        completion: Completion,
        outcome: Address,
        outputs: Address,
    ):
        var landing = Pointer[DenseMultiplyOutputs, MutAnyOrigin](unsafe_from_address=outputs)
        _cross(
            DenseMultiplyArguments(
                index,
                executor,
                outcome,
                completion,
                c_size_t(self.count),
                self.left,
                Int(Pointer(to=landing[].products)),
                Int(Pointer(to=landing[].written)),
                Int(Pointer(to=landing[].required)),
                self.arena[]._address,
                c_size_t(self.wanted),
                c_size_t(self.expansion_factor),
                ProgressArguments(),
                self.stats,
            ),
            library.symbols().dense_multiply,
        )

    def served(self, status: Status) -> Int:
        return _served(self.arena, status)

    def conclude(deinit self, outputs: DenseMultiplyOutputs, generation: Int) -> Self.Result:
        return Self.Result(
            self.arena, generation, outputs.products, self.count, Int(outputs.written), Int(outputs.required)
        )


@fieldwise_init
struct _DenseAttendCall[dtype: DType, origin: MutOrigin](Call):
    """The crossing of `usearch_dense_attend`, which borrows the arena its attended rows land in."""

    comptime Outputs = DenseAttendOutputs
    comptime Result = DenseVectors[Self.dtype, Self.origin]
    comptime symbol: StaticString = "usearch_dense_attend"
    var queries: RowsViewArguments
    var count: Int
    var head_count: Int
    var value_lookup: ValueLookup
    var value_lookup_state: Address
    var carry: Address
    var carry_count: Int
    var arena: Pointer[Arena, Self.origin]
    var expansion_factor: Int
    var score_scale: Float32
    var stats: Address

    def submit(
        mut self,
        library: Library,
        index: Address,
        executor: Address,
        completion: Completion,
        outcome: Address,
        outputs: Address,
    ):
        # The engine reads a zero kind as the queries' own, which `dtype` might not hold.
        if ScalarKind.of[Self.dtype]() == ScalarKind.UNKNOWN:
            return _refuse(outcome, Status.SHAPE_MISMATCH)
        # The carry crosses as a bare pointer, so only this side can see it fall short of the rows.
        if self.carry != NULL_ADDRESS and self.carry_count < self.count:
            return _refuse(outcome, Status.SHAPE_MISMATCH)
        _cross(
            DenseAttendArguments(
                index,
                executor,
                outcome,
                completion,
                c_size_t(self.count),
                self.queries,
                c_size_t(self.head_count),
                self.value_lookup,
                self.value_lookup_state,
                self.carry,
                outputs,
                self.arena[]._address,
                ScalarKind.of[Self.dtype](),
                c_size_t(self.expansion_factor),
                self.score_scale,
                ProgressArguments(),
                self.stats,
            ),
            library.symbols().dense_attend,
        )

    def served(self, status: Status) -> Int:
        return _served(self.arena, status)

    def conclude(deinit self, outputs: DenseAttendOutputs, generation: Int) -> Self.Result:
        return Self.Result(self.arena, generation, outputs.outputs, self.count)


@fieldwise_init
struct _DenseSaveCall(Call):
    """The crossing of `usearch_dense_save`."""

    comptime Outputs = Unwritten
    comptime Result = NoneType
    comptime symbol: StaticString = "usearch_dense_save"
    var path: BytesView[ImmUntrackedOrigin]

    def submit(
        mut self,
        library: Library,
        index: Address,
        executor: Address,
        completion: Completion,
        outcome: Address,
        outputs: Address,
    ):
        _cross(
            DenseSaveArguments(index, executor, outcome, completion, self.path, ProgressArguments()),
            library.symbols().dense_save,
        )

    def conclude(deinit self, outputs: Unwritten, generation: Int) -> NoneType:
        return None


@fieldwise_init
struct _DenseSaveBufferCall[origin: MutOrigin](Call):
    """The crossing of `usearch_dense_save_buffer`, which borrows the arena the monolith lands in."""

    comptime Outputs = DenseSaveBufferOutputs
    comptime Result = Monolith[Self.origin]
    comptime symbol: StaticString = "usearch_dense_save_buffer"
    var arena: Pointer[Arena, Self.origin]

    def submit(
        mut self,
        library: Library,
        index: Address,
        executor: Address,
        completion: Completion,
        outcome: Address,
        outputs: Address,
    ):
        _cross(
            DenseSaveBufferArguments(
                index, executor, outcome, completion, outputs, self.arena[]._address, ProgressArguments()
            ),
            library.symbols().dense_save_buffer,
        )

    def served(self, status: Status) -> Int:
        return _served(self.arena, status)

    def conclude(deinit self, outputs: DenseSaveBufferOutputs, generation: Int) -> Self.Result:
        return Self.Result(self.arena, generation, outputs.data, Int(outputs.length))


@fieldwise_init
struct _DenseCursorNextCall[origin: MutOrigin](Call):
    """The crossing of `usearch_dense_cursor_next`, which borrows the arena its batch lands in."""

    comptime Outputs = DenseCursorNextOutputs
    comptime Result = DenseBatch[Self.origin]
    comptime symbol: StaticString = "usearch_dense_cursor_next"
    var cursor: Address
    var links: Bool
    var wanted: Int
    var arena: Pointer[Arena, Self.origin]

    def submit(
        mut self,
        library: Library,
        index: Address,
        executor: Address,
        completion: Completion,
        outcome: Address,
        outputs: Address,
    ):
        var landing = Pointer[DenseCursorNextOutputs, MutAnyOrigin](unsafe_from_address=outputs)
        var links = self.links
        _cross(
            DenseCursorNextArguments(
                index,
                executor,
                outcome,
                completion,
                self.cursor,
                Int(Pointer(to=landing[].keys)),
                Int(Pointer(to=landing[].links)) if links else NULL_ADDRESS,
                Int(Pointer(to=landing[].written)),
                self.arena[]._address,
                c_size_t(self.wanted),
            ),
            library.symbols().dense_cursor_next,
        )

    def served(self, status: Status) -> Int:
        return _served(self.arena, status)

    def conclude(deinit self, outputs: DenseCursorNextOutputs, generation: Int) -> Self.Result:
        return Self.Result(
            self.arena, generation, outputs.keys, self.links, outputs.links, Int(outputs.written)
        )


@fieldwise_init
struct _DenseSearchStreamNextCall[origin: MutOrigin](Call):
    """The crossing of `usearch_dense_search_stream_next`, which borrows the arena its pull lands in."""

    comptime Outputs = MatchesViewArguments
    comptime Result = Matches[Self.origin]
    comptime symbol: StaticString = "usearch_dense_search_stream_next"
    var stream: Address
    var wanted: Int
    var arena: Pointer[Arena, Self.origin]

    def submit(
        mut self,
        library: Library,
        index: Address,
        executor: Address,
        completion: Completion,
        outcome: Address,
        outputs: Address,
    ):
        _cross(
            DenseSearchStreamNextArguments(
                index, executor, outcome, completion, self.stream, outputs, self.arena[]._address, c_size_t(self.wanted)
            ),
            library.symbols().dense_search_stream_next,
        )

    def served(self, status: Status) -> Int:
        return _served(self.arena, status)

    def conclude(deinit self, outputs: MatchesViewArguments, generation: Int) -> Self.Result:
        return Self.Result(self.arena, generation, outputs, 1)


# endregion Calls

# region Index


struct DenseIndex(DenseVerbs, Movable):
    """A dense vector index, released when it goes out of scope.

    Insertions and searches overlap freely on one handle: a search never writes to the graph, and
    an insertion publishes a row only once it is reachable, so a concurrent walk sees a consistent
    graph at every instant. Calls on one executor run in submission order.

    Verbs called on the index run on the calling thread; `index.on(executor)` runs the same verbs
    blocking on an executor, and `index.deferred(executor)` submits them there. The cursor and the
    stream's opening name no executor, and their pulls run on the receiver's.
    """

    var library: Library
    """The loaded engine, held so its symbol table outlives every call made through the handle."""
    var address: Address
    """The opaque `usearch_dense_index_t`, or zero once an absorb has consumed it."""

    # region Lifetime

    def __init__(out self, *, library: Library, address: Address):
        """Adopts a handle the engine just created.

        Args:
            library: The loaded engine the handle was created through, and is freed through.
            address: A `usearch_dense_index_t` whose lifetime this index takes over.
        """
        self.library = library
        self.address = address

    @staticmethod
    def init_ram[
        ram_origin: ImmOrigin, //
    ](library: Library, options: DenseOptions, ram: RamOptions[ram_origin] = RamOptions()) raises Error -> Self:
        """Initializes an empty, mutable index in host memory. Calls `usearch_dense_init_ram`.

        Parameters:
            ram_origin: The options' borrowed node-id array, inferred from the argument.

        Args:
            library: The loaded engine, kept so the shared object stays mapped.
            options: The geometry and metric; read only during this call.
            ram: The memory domains to place one shard on apiece.

        Returns:
            The new index.

        Raises:
            `Error` if the geometry is refused, the placement cannot be resolved, or the
            allocation fails.
        """
        var created = stack_allocation[1, Address]()
        created[unsafe_offset=0] = NULL_ADDRESS
        var outcome = stack_allocation[1, Outcome]()
        outcome[unsafe_offset=0] = Outcome()
        _cross(
            DenseInitRamArguments(
                Int(created),
                Int(outcome),
                options.metric,
                options.quantization,
                options.dimensions,
                options.connectivity,
                options.values_per_key,
                ram.untracked(),
            ),
            library.symbols().dense_init_ram,
        )
        library._check(outcome.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0], "usearch_dense_init_ram")
        return Self(library=library, address=created.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0])

    @staticmethod
    def init_vram[
        vram_origin: ImmOrigin, //
    ](library: Library, options: DenseOptions, vram: VramOptions[vram_origin] = VramOptions()) raises Error -> Self:
        """Initializes an empty, mutable index in device memory. Calls `usearch_dense_init_vram`.

        Parameters:
            vram_origin: The options' borrowed ordinal array, inferred from the argument.

        Args:
            library: The loaded engine, kept so the shared object stays mapped.
            options: The geometry and metric; read only during this call.
            vram: The devices to place one shard on apiece.

        Returns:
            The new index.

        Raises:
            `Error` if the geometry is refused, or a device cannot be selected or claimed.
        """
        var created = stack_allocation[1, Address]()
        created[unsafe_offset=0] = NULL_ADDRESS
        var outcome = stack_allocation[1, Outcome]()
        outcome[unsafe_offset=0] = Outcome()
        _cross(
            DenseInitVramArguments(
                Int(created),
                Int(outcome),
                options.metric,
                options.quantization,
                options.dimensions,
                options.connectivity,
                options.values_per_key,
                vram.untracked(),
            ),
            library.symbols().dense_init_vram,
        )
        library._check(outcome.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0], "usearch_dense_init_vram")
        return Self(library=library, address=created.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0])

    @staticmethod
    def load_ram[
        path_origin: ImmOrigin, ram_origin: ImmOrigin, //
    ](
        executor: Executor,
        path: StringSlice[path_origin],
        ram: RamOptions[ram_origin] = RamOptions(),
        values_per_key: Int = 0,
    ) raises Error -> Self:
        """Copies a checkpoint into a new, mutable index in host memory, shard by shard.
        Calls `usearch_dense_load_ram`.

        Point the path at the manifest, never at one of the parts it enumerates - naming a part
        loads exactly that part, which is a fragment of the index rather than the index. Part `p`
        lands whole on shard `p` modulo the shard count, wherever it was written from.

        Parameters:
            path_origin: The path bytes' origin, inferred from the argument.
            ram_origin: The options' borrowed node-id array, inferred from the argument.

        Args:
            executor: The executor the read fans out over, one shard per thread; it also lends its engine.
            path: The manifest of a sharded checkpoint, or a single `.safegraphs` monolith.
            ram: The memory domains to place the parts on.
            values_per_key: Upper bound on entries sharing one key, which no checkpoint records.

        Returns:
            The new index; a failed load creates none.

        Raises:
            `Error` if the checkpoint cannot be read or decoded.
        """
        return Self._load_ram(
            executor,
            executor.library.symbols().dense_load_ram,
            BytesView.of(path).untracked(),
            ram.untracked(),
            values_per_key,
            "usearch_dense_load_ram",
        )

    @staticmethod
    def load_ram_buffer[
        buffer_origin: ImmOrigin, ram_origin: ImmOrigin, //
    ](
        executor: Executor,
        buffer: Span[UInt8, buffer_origin],
        ram: RamOptions[ram_origin] = RamOptions(),
        values_per_key: Int = 0,
    ) raises Error -> Self:
        """Copies a monolith from a memory buffer into a new, mutable index in host memory.
        Calls `usearch_dense_load_ram_buffer`.

        Parameters:
            buffer_origin: The source bytes' origin, inferred from the argument.
            ram_origin: The options' borrowed node-id array, inferred from the argument.

        Args:
            executor: The executor the copy fans out over; it also lends its engine.
            buffer: Source holding one serialized monolith.
            ram: The memory domains to place shards on; the monolith lands whole on the first.
            values_per_key: Upper bound on entries sharing one key, which no checkpoint records.

        Returns:
            The new index; a failed load creates none.

        Raises:
            `Error` if the bytes cannot be decoded.
        """
        return Self._load_ram(
            executor,
            executor.library.symbols().dense_load_ram_buffer,
            BytesView.of(buffer).untracked(),
            ram.untracked(),
            values_per_key,
            "usearch_dense_load_ram_buffer",
        )

    @staticmethod
    def _load_ram(
        executor: Executor,
        verb: DenseLoadRam,
        source: BytesView[ImmUntrackedOrigin],
        ram: RamOptions[ImmUntrackedOrigin],
        values_per_key: Int,
        detail: StaticString,
    ) raises Error -> Self:
        """The path and buffer forms share one argument layout, differing only in the verb."""
        var created = stack_allocation[1, Address]()
        created[unsafe_offset=0] = NULL_ADDRESS
        var outcome = stack_allocation[1, Outcome]()
        outcome[unsafe_offset=0] = Outcome()
        _cross(
            DenseLoadRamArguments(
                Int(created),
                executor.address,
                Int(outcome),
                Completion.BLOCKING,
                source,
                ram,
                c_size_t(values_per_key),
                ProgressArguments(),
            ),
            verb,
        )
        executor.library._check(outcome.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0], detail)
        return Self(library=executor.library, address=created.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0])

    @staticmethod
    def load_vram[
        path_origin: ImmOrigin, vram_origin: ImmOrigin, //
    ](
        executor: Executor,
        path: StringSlice[path_origin],
        vram: VramOptions[vram_origin] = VramOptions(),
        values_per_key: Int = 0,
    ) raises Error -> Self:
        """Copies a checkpoint into a new, mutable index in device memory, with the placement rules
        of `load_ram`. Calls `usearch_dense_load_vram`.

        Parameters:
            path_origin: The path bytes' origin, inferred from the argument.
            vram_origin: The options' borrowed ordinal array, inferred from the argument.

        Args:
            executor: The executor the upload runs on, one stream per device; it also lends its engine.
            path: The manifest of a sharded checkpoint, or a single `.safegraphs` monolith.
            vram: The devices to place the parts on.
            values_per_key: Upper bound on entries sharing one key, which no checkpoint records.

        Returns:
            The new index; a failed load creates none.

        Raises:
            `Error` if the checkpoint cannot be read, or a device cannot hold it.
        """
        return Self._load_vram(
            executor,
            executor.library.symbols().dense_load_vram,
            BytesView.of(path).untracked(),
            vram.untracked(),
            values_per_key,
            "usearch_dense_load_vram",
        )

    @staticmethod
    def load_vram_buffer[
        buffer_origin: ImmOrigin, vram_origin: ImmOrigin, //
    ](
        executor: Executor,
        buffer: Span[UInt8, buffer_origin],
        vram: VramOptions[vram_origin] = VramOptions(),
        values_per_key: Int = 0,
    ) raises Error -> Self:
        """Copies a monolith from a host or device buffer into a new, mutable index in device memory.
        Calls `usearch_dense_load_vram_buffer`.

        Parameters:
            buffer_origin: The source bytes' origin, inferred from the argument.
            vram_origin: The options' borrowed ordinal array, inferred from the argument.

        Args:
            executor: The executor the upload runs on, one stream per device; it also lends its engine.
            buffer: Source holding one serialized monolith.
            vram: The devices to place shards on; the monolith lands whole on the first.
            values_per_key: Upper bound on entries sharing one key, which no checkpoint records.

        Returns:
            The new index; a failed load creates none.

        Raises:
            `Error` if the bytes cannot be decoded, or a device cannot hold them.
        """
        return Self._load_vram(
            executor,
            executor.library.symbols().dense_load_vram_buffer,
            BytesView.of(buffer).untracked(),
            vram.untracked(),
            values_per_key,
            "usearch_dense_load_vram_buffer",
        )

    @staticmethod
    def _load_vram(
        executor: Executor,
        verb: DenseLoadVram,
        source: BytesView[ImmUntrackedOrigin],
        vram: VramOptions[ImmUntrackedOrigin],
        values_per_key: Int,
        detail: StaticString,
    ) raises Error -> Self:
        """The path and buffer forms share one argument layout, differing only in the verb."""
        var created = stack_allocation[1, Address]()
        created[unsafe_offset=0] = NULL_ADDRESS
        var outcome = stack_allocation[1, Outcome]()
        outcome[unsafe_offset=0] = Outcome()
        _cross(
            DenseLoadVramArguments(
                Int(created),
                executor.address,
                Int(outcome),
                Completion.BLOCKING,
                source,
                vram,
                c_size_t(values_per_key),
                ProgressArguments(),
            ),
            verb,
        )
        executor.library._check(outcome.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0], detail)
        return Self(library=executor.library, address=created.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0])

    @staticmethod
    def view_ram[path_origin: ImmOrigin, //](library: Library, path: StringSlice[path_origin]) raises Error -> Self:
        """Serves a checkpoint in place through a memory mapping, as a read-only index.
        Calls `usearch_dense_view_ram`.

        Nothing is copied, so opening costs the headers alone. Every mutating verb on the handle
        raises with `READ_ONLY`, and the files must stay unmodified until the handle is released.

        Parameters:
            path_origin: The path bytes' origin, inferred from the argument.

        Args:
            library: The loaded engine, kept so the shared object stays mapped.
            path: The manifest of a sharded checkpoint, or a single `.safegraphs` monolith.

        Returns:
            The read-only index.

        Raises:
            `Error` if the checkpoint cannot be mapped or decoded.
        """
        var created = stack_allocation[1, Address]()
        created[unsafe_offset=0] = NULL_ADDRESS
        var outcome = stack_allocation[1, Outcome]()
        outcome[unsafe_offset=0] = Outcome()
        _cross(
            DenseViewRamArguments(Int(created), Int(outcome), BytesView.of(path).untracked()),
            library.symbols().dense_view_ram,
        )
        library._check(outcome.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0], "usearch_dense_view_ram")
        return Self(library=library, address=created.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0])

    @staticmethod
    def view_nvme[
        path_origin: ImmOrigin, nvme_origin: ImmOrigin, //
    ](
        executor: Executor,
        path: StringSlice[path_origin],
        nvme: NvmeOptions[nvme_origin] = NvmeOptions(),
    ) raises Error -> Self:
        """Serves a checkpoint in place from NVMe drives, read through the executor's rings.
        Calls `usearch_dense_view_nvme`.

        Parameters:
            path_origin: The path bytes' origin, inferred from the argument.
            nvme_origin: The options' borrowed path array, inferred from the argument.

        Args:
            executor: The NVMe executor whose rings read the checkpoint's headers; it also lends its engine.
            path: The manifest of a sharded checkpoint, or a single `.safegraphs` monolith.
            nvme: The devices the parts are served from; naming none serves each where it points.

        Returns:
            The read-only index.

        Raises:
            `Error` with `UNSUPPORTED` until NVMe checkpoints carry keys and a shard wrapper.
        """
        var created = stack_allocation[1, Address]()
        created[unsafe_offset=0] = NULL_ADDRESS
        var outcome = stack_allocation[1, Outcome]()
        outcome[unsafe_offset=0] = Outcome()
        _cross(
            DenseViewNvmeArguments(
                Int(created),
                executor.address,
                Int(outcome),
                Completion.BLOCKING,
                BytesView.of(path).untracked(),
                nvme.untracked(),
            ),
            executor.library.symbols().dense_view_nvme,
        )
        executor.library._check(outcome.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0], "usearch_dense_view_nvme")
        return Self(library=executor.library, address=created.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0])

    def __deinit__(deinit self):
        """Releases the index and everything it owns, unless an absorb already took the handle.
        Calls `usearch_dense_free`.

        The loader reference is surrendered only after the call: reading the entry point is the
        last use of `library`, so without the trailing move Mojo drops the last reference - and
        `dlclose`s the engine - between the load and the branch to it.
        """
        self.library.symbols().dense_free(self.address)
        _ = self.library^

    # endregion Lifetime

    # region Receivers

    def on(ref self, ref executor: Executor) -> DenseIndexOn[ImmOrigin(origin_of(self, executor))]:
        """The same verbs, blocking on `executor` after every call submitted there before them. Binding-only.

        Args:
            executor: The executor every verb runs on; its kind must match the handle's residency.

        Returns:
            A receiver borrowing both, which neither may outlive.
        """
        return DenseIndexOn[ImmOrigin(origin_of(self, executor))](
            Pointer[DenseIndex, ImmOrigin(origin_of(self, executor))](unsafe_from_address=Int(Pointer(to=self))),
            executor.address,
        )

    def deferred(ref self, ref executor: Executor) -> DenseIndexDeferredOn[ImmOrigin(origin_of(self, executor))]:
        """The same verbs, each submitted to `executor` and answering a `Deferred` at once. Binding-only.

        Args:
            executor: The executor every verb is submitted to; its kind must match the handle's residency.

        Returns:
            A receiver borrowing both, which neither it nor any `Deferred` it answers may outlive.
        """
        return DenseIndexDeferredOn[ImmOrigin(origin_of(self, executor))](
            Pointer[DenseIndex, ImmOrigin(origin_of(self, executor))](unsafe_from_address=Int(Pointer(to=self))),
            executor.address,
        )

    def _route[C: Call](self, var call: C) raises Error -> C.Result:
        """Runs `call` on the calling thread."""
        return _block(self.library, call^, self.address, NULL_ADDRESS)

    def cursor(ref self) raises Error -> DenseCursor[ImmOrigin(origin_of(self))]:
        """Opens a walk over every row as stored, its pulls on the calling thread.
        Calls `usearch_dense_cursor_init`.

        Each shard's publication watermark is read as the walk enters that shard, so a row once
        passed is never revisited.

        Returns:
            The cursor, which borrows the index and cannot outlive it.

        Raises:
            `Error` if the residency has no cursor, as VRAM does not.
        """
        return DenseCursor[ImmOrigin(origin_of(self))](self.library, self.address, NULL_ADDRESS, 0, NULL_ADDRESS)

    def cursor[
        starts_origin: ImmOrigin, //
    ](ref self, starts: Span[Key, starts_origin]) raises Error -> DenseCursor[ImmOrigin(origin_of(self))]:
        """Opens a walk from `starts`, its pulls on the calling thread. Calls `usearch_dense_cursor_init`.

        It yields the start keys' own rows first, in order, then the rows those link to
        breadth-first, each once.

        Parameters:
            starts_origin: The start keys' origin, inferred from the argument.

        Args:
            starts: The keys to walk from.

        Returns:
            The cursor, which borrows the index and cannot outlive it.

        Raises:
            `Error` if the residency has no cursor, as VRAM does not.
        """
        return DenseCursor[ImmOrigin(origin_of(self))](
            self.library, self.address, NULL_ADDRESS, len(starts), Int(starts.unsafe_ptr())
        )

    def search_stream[
        query_origin: ImmOrigin, //
    ](
        ref self, query: RowsView[query_origin], config: DenseSearchStreamInitConfig = DenseSearchStreamInitConfig()
    ) raises Error -> DenseSearchStream[ImmOrigin(origin_of(self, _anchor[query_origin]()))]:
        """Opens a resumable nearest-neighbor stream for one query, its pulls on the calling thread.
        Calls `usearch_dense_search_stream_init`.

        The stream keeps walk state rather than a thread, so any number may stay open at once.

        Parameters:
            query_origin: The query row's origin, inferred from the argument.

        Args:
            query: The single query row; a view of more rows has its first row read.
            config: The breadth of the candidates the walk explores.

        Returns:
            A stream yielding matches lazily, which borrows the index and the query.

        Raises:
            `Error` with `UNSUPPORTED` while no residency implements the stream.
        """
        return DenseSearchStream[ImmOrigin(origin_of(self, _anchor[query_origin]()))](
            self.library, self.address, NULL_ADDRESS, query.arguments(), config.expansion_factor
        )

    # endregion Receivers


trait DenseVerbs:
    """The verbs every blocking receiver offers - the index on the calling thread, and
    `index.on(executor)` on an executor - each written once over the receiver's `_route`."""

    def _route[C: Call](self, var call: C) raises Error -> C.Result:
        """Runs `call` to its final status on this receiver's executor and answers its result."""
        ...

    def reserve(self, capacity: Int) raises Error:
        """Reserves capacity for a number of incoming vectors ahead of insertion. Calls `usearch_dense_reserve`.

        Args:
            capacity: The desired total capacity, including the current size.

        Raises:
            `Error` if the reservation cannot be satisfied, or contradicts a frozen geometry.
        """
        _ = self._route(_DenseReserveCall(capacity))

    def clear(self) raises Error:
        """Erases every entry, keeping the reserved capacity and the frozen geometry. Calls `usearch_dense_clear`.

        Never invoked concurrently with any other verb on the same handle.

        Raises:
            `Error` with `READ_ONLY` if the handle serves its checkpoint in place.
        """
        _ = self._route(_DenseClearCall())

    def describe(self) raises Error -> DenseIndexDescription:
        """Reports the frozen metadata and the current counters in one call. Calls `usearch_dense_describe`.

        Returns:
            The description.

        Raises:
            `Error` if the engine refuses the query.
        """
        return self._route(_DenseDescribeCall())

    def insert[
        keys_origin: ImmOrigin, vectors_origin: ImmOrigin, stats_origin: MutOrigin, //
    ](
        self,
        keys: Span[Key, keys_origin],
        vectors: RowsView[vectors_origin],
        config: DenseInsertConfig[stats_origin] = DenseInsertConfig(),
    ) raises Error:
        """Adds a run of vectors under their keys - the native batch operation. Calls `usearch_dense_insert`.

        A batch is all-or-nothing, because this binding installs no progress handler: every
        fallible step runs before the first entry identity is claimed, so a rejected batch leaves
        the index untouched and publishes no row.

        Parameters:
            keys_origin: The key run's origin, inferred from the argument.
            vectors_origin: The row storage's origin, inferred with it.
            stats_origin: The lent telemetry slot's origin, inferred with it.

        Args:
            keys: One key per row of `vectors`; keys need not be unique.
            vectors: The rows to insert, converted on ingest to the stored type.
            config: The breadth of the candidates and the telemetry slot.

        Raises:
            `Error` if the shapes disagree or the handle is read-only.
        """
        _ = self._route(_insert_call(keys, vectors, config))

    def refine[
        stats_origin: MutOrigin, //
    ](self, config: DenseRefineConfig[stats_origin] = DenseRefineConfig()) raises Error -> Int:
        """Re-selects every entry's neighbors against the finished graph. Calls `usearch_dense_refine`.

        A build step rather than a repair: a cheap insertion pass followed by one refinement
        reaches a better graph, in less total time, than a single expensive insertion pass - so a
        wide expansion factor belongs here rather than in `insert`. Searches may run throughout;
        insertions may not.

        Parameters:
            stats_origin: The lent telemetry slot's origin, inferred from the argument.

        Args:
            config: The breadth of the candidates the re-selection walk explores, and the telemetry slot.

        Returns:
            How many nodes' neighbors changed; zero means the graph has settled.

        Raises:
            `Error` with `READ_ONLY` if the handle serves its checkpoint in place.
        """
        return self._route(_DenseRefineCall(config.expansion_factor, _stats_address(config.stats)))

    def health(self) raises Error -> DenseReachability:
        """Audits how much of the graph a search reaches from its landmarks. Calls `usearch_dense_health`.

        Read-only, so concurrent insertions merely leave fresh entries unaudited.

        Returns:
            The audit; entries it misses stay unfound until a `refine` re-links them.

        Raises:
            `Error` if the engine refuses the audit.
        """
        return self._route(_DenseHealthCall())

    def absorb(self, var donors: List[DenseIndex]) raises Error:
        """Absorbs every donor into this index, keeping their graphs rather than re-inserting their
        vectors. Calls `usearch_dense_absorb`.

        Each donor the engine consumes is emptied here, so dropping the list releases only the
        donors a stopped absorb left standing.

        Args:
            donors: The handles to absorb, consumed by the call.

        Raises:
            `Error` with `CONFIG_MISMATCH` for a donor of another residency, scalar, metric or
            geometry, or with `READ_ONLY` if the handle serves its checkpoint in place.
        """
        _ = self._route(_DenseAbsorbCall(donors^))

    def search[
        queries_origin: ImmOrigin, stats_origin: MutOrigin, //
    ](
        self,
        queries: RowsView[queries_origin],
        wanted: Int,
        mut arena: Arena,
        config: DenseSearchConfig[stats_origin] = DenseSearchConfig(),
    ) raises Error -> Matches[origin_of(arena)]:
        """Performs k-approximate nearest-neighbor search for a batch of queries, landing the matches
        in a lent arena. Calls `usearch_dense_search`.

        Never writes to the graph, so any number of these may overlap with concurrent insertions.

        Parameters:
            queries_origin: The query rows' origin, inferred from the argument.
            stats_origin: The lent telemetry slot's origin, inferred with it.

        Args:
            queries: The query rows, one per search.
            wanted: Matches per query - the "k"; zero lands only the counts.
            arena: The arena to rewind and refill, which the result borrows; a failed call leaves it valid.
            config: The breadth of the candidates, the predicate, and the telemetry slot.

        Returns:
            Every query's matches, read in place from the arena.

        Raises:
            `Error` if the shapes disagree, or a predicate is passed to a VRAM handle.
        """
        return self._route(_search_call(queries, wanted, Pointer(to=arena), config))

    def count(self, key: Key) raises Error -> Int:
        """How many entries the index stores under one key. Calls `usearch_dense_count`.

        Args:
            key: The key whose entries are counted.

        Returns:
            The number of entries sharing the key; zero means the index holds none.

        Raises:
            `Error` if the engine refuses the query.
        """
        return self._route(_DenseCountCall(key))

    def get[
        dtype: DType
    ](self, key: Key, mut arena: Arena, scalar: ScalarKind = ScalarKind.of[dtype]()) raises Error -> DenseVectors[
        dtype, origin_of(arena)
    ]:
        """Exports every vector stored under one key into a lent arena, one row per entry, in
        insertion order. Calls `usearch_dense_get`.

        Parameters:
            dtype: The element type the rows are read as, which `scalar` must store whole.

        Args:
            key: The key whose entries are exported.
            arena: The arena to rewind and refill, which the result borrows; a failed call leaves it valid.
            scalar: The scalar type to export as, converted from storage where it differs; the kind
                `dtype` stores by default.

        Returns:
            The rows, read in place from the arena; none when the key is absent.

        Raises:
            `Error` with `SHAPE_MISMATCH` if `dtype` cannot hold `scalar`, or if the engine refuses
            the export.
        """
        return self._route(_DenseGetCall[dtype](key, scalar, Pointer(to=arena)))

    def save[path_origin: ImmOrigin, //](self, path: StringSlice[path_origin]) raises Error:
        """Writes a checkpoint, the path extension picking the layout. Calls `usearch_dense_save`.

        A `.safegraphs` path is one monolith; a `.safegraphs.json` path is a manifest naming one
        monolith per shard. Demands a mutation-free window; concurrent readers keep searching
        throughout. A failed save leaves a file without its header, which every reader rejects, and
        that leftover is the caller's to remove.

        Parameters:
            path_origin: The path bytes' origin, inferred from the argument.

        Args:
            path: Destination path, whose extension picks the layout.

        Raises:
            `Error` if the destination cannot be written.
        """
        _ = self._route(_DenseSaveCall(BytesView.of(path).untracked()))

    def save_buffer(self, mut arena: Arena) raises Error -> Monolith[origin_of(arena)]:
        """Serializes the index as one monolith into a lent arena - the same bytes the path-taking
        sibling writes. Calls `usearch_dense_save_buffer`.

        Args:
            arena: The arena to rewind and refill, which the result borrows; a failed call leaves it valid.

        Returns:
            The monolith, read in place from the arena.

        Raises:
            `Error` if the engine refuses the write.
        """
        return self._route(_DenseSaveBufferCall(Pointer(to=arena)))

    def multiply[
        left_origin: ImmOrigin, stats_origin: MutOrigin, //
    ](
        self,
        left: RowsView[left_origin],
        mut arena: Arena,
        config: DenseMultiplyConfig[stats_origin] = DenseMultiplyConfig(),
    ) raises Error -> DenseProducts[origin_of(arena)]:
        """Approximate sparse product of a run of rows against the indexed rows, into a lent arena.
        Calls `usearch_dense_multiply`.

        Reports every pair the traversal scored instead of only the nearest, in traversal order
        rather than sorted, and with no duplicates. Cost is roughly
        `rows x expansion x connectivity x dimensions`, sublinear in the indexed rows, which is the
        whole point of reading a product off a graph. Runs on `DenseMetric.INNER` alone.

        Parameters:
            left_origin: The left-hand rows' origin, inferred from the argument.
            stats_origin: The lent telemetry slot's origin, inferred with it.

        Args:
            left: The left-hand rows.
            arena: The arena to rewind and refill, which the result borrows; a failed call leaves it valid.
            config: The pairs kept per row, the breadth of the candidates, and the telemetry slot.

        Returns:
            The ragged pairs and their offsets, read in place from the arena.

        Raises:
            `Error` with `CONFIG_MISMATCH` if the graph was frozen under a distance kind.
        """
        return self._route(_multiply_call(left, Pointer(to=arena), config))

    def attend[
        queries_origin: ImmOrigin, carry_origin: MutOrigin, stats_origin: MutOrigin, //, dtype: DType
    ](
        self,
        queries: RowsView[queries_origin],
        value_lookup: ValueLookup,
        value_lookup_state: Address,
        mut arena: Arena,
        config: DenseAttendConfig[carry_origin, stats_origin] = DenseAttendConfig(),
    ) raises Error -> DenseVectors[dtype, origin_of(arena)]:
        """Softmax attention of a run of query heads over every indexed key their graph walks score,
        landing in a lent arena. Calls `usearch_dense_attend`.

        Every key a walk scores folds into each head's running maximum, partition, and weighted
        value sum, as Flash Attention folds key tiles, with no top-k cut: the long tail the walk
        scores is what brings the result toward exact dense attention. The outputs come back
        already divided by the partition.

        Row `t · head_count + h` of `queries`, the outputs, and `carry` is head `h` of token `t`. One
        index holds one layer's one key-value head, so the `head_count` query heads of grouped-query
        attention share its keys and one walk per token.

        The index stores no values: `value_lookup` resolves the keys of every scored batch to their
        value rows, so only keys a walk scores are ever resolved. `carry` is the softmax's `m` and
        `l`, one `DenseSoftmaxState` per query row, lent by the caller: supplying it folds this branch
        into another over the same queries, such as the recent window kept out of the index, by
        partial-softmax rescaling.

        Parameters:
            queries_origin: The query rows' origin, inferred from the argument.
            carry_origin: The softmax states' origin, inferred with it where any are supplied.
            stats_origin: The lent telemetry slot's origin, inferred with it.
            dtype: The element type the attended rows land as, which must name a scalar kind.

        Args:
            queries: The query rows, `head_count` per token.
            value_lookup: Resolves every scored batch of keys to their value rows - a host function on
                RAM, and a device one on VRAM.
            value_lookup_state: The opaque state forwarded into every `value_lookup` call.
            arena: The arena to rewind and refill, which the result borrows; a failed call leaves it valid.
            config: The heads per token, the carry, the breadth of the candidates, the logit scale,
                and the telemetry slot.

        Returns:
            One attended row per query row, as wide as the keys, read in place from the arena.

        Raises:
            `Error` with `SHAPE_MISMATCH` if `dtype` names no scalar kind, `head_count` does not
            divide the rows, the carry holds fewer states than there are rows, or a lookup yields a
            null row; with `CONFIG_MISMATCH` if the graph was frozen under a distance kind.
        """
        return self._route(_attend_call[dtype](queries, value_lookup, value_lookup_state, Pointer(to=arena), config))

    def maxsim[
        queries_origin: ImmOrigin, stats_origin: MutOrigin, //
    ](
        self,
        queries: RowsView[queries_origin],
        rows_per_query: Int,
        wanted: Int,
        mut arena: Arena,
        config: DenseMaxsimConfig[stats_origin] = DenseMaxsimConfig(),
    ) raises Error -> Matches[origin_of(arena)]:
        """Late-interaction scoring: every query row takes its best match within a key, and those
        maxima sum to that key's score, landing in a lent arena. Calls `usearch_dense_maxsim`.

        This is what `values_per_key` above one exists for. A document is one key holding its token
        rows, and the walk stays over individual tokens under a plain inner product. Approximate in
        the usual direction: a key can score below its exact value but never above it.

        Parameters:
            queries_origin: The query rows' origin, inferred from the argument.
            stats_origin: The lent telemetry slot's origin, inferred with it.

        Args:
            queries: The token rows of every query, laid end to end.
            rows_per_query: Rows in every query; uniform, so ColBERT's padding applies as is.
            wanted: Keys per query - the "k" over documents rather than rows; zero lands only the counts.
            arena: The arena to rewind and refill, which the result borrows; a failed call leaves it valid.
            config: The breadth of the candidates and the telemetry slot.

        Returns:
            Every query's keys, summed maxima largest first, read in place from the arena.

        Raises:
            `Error` with `SHAPE_MISMATCH` if `rows_per_query` does not divide the rows; with
            `CONFIG_MISMATCH` if the graph was frozen under a distance kind.
        """
        return self._route(_maxsim_call(queries, rows_per_query, wanted, Pointer(to=arena), config))


struct DenseIndexOn[origin: ImmOrigin](DenseVerbs, ImplicitlyCopyable):
    """An index bound to the executor its verbs block on, as `index.on(executor)` answers it. Binding-only.

    Parameters:
        origin: The index and the executor, which this receiver borrows.
    """

    var index: Pointer[DenseIndex, Self.origin]
    """The index every verb addresses."""
    var executor: Address
    """The `usearch_executor_t` every verb runs on."""

    def __init__(out self, index: Pointer[DenseIndex, Self.origin], executor: Address):
        """Binds an index to an executor.

        Args:
            index: The index every verb addresses.
            executor: The `usearch_executor_t` every verb runs on.
        """
        self.index = index
        self.executor = executor

    def _route[C: Call](self, var call: C) raises Error -> C.Result:
        """Runs `call` on the bound executor, after every call submitted there before it."""
        return _block(self.index[].library, call^, self.index[].address, self.executor)

    def cursor(self) raises Error -> DenseCursor[Self.origin]:
        """Opens a walk over every row as stored, its pulls on the bound executor.
        Calls `usearch_dense_cursor_init`.

        Returns:
            The cursor, which borrows the index and the executor and cannot outlive them.

        Raises:
            `Error` if the residency has no cursor, as VRAM does not.
        """
        return DenseCursor[Self.origin](self.index[].library, self.index[].address, self.executor, 0, NULL_ADDRESS)

    def cursor[
        starts_origin: ImmOrigin, //
    ](self, starts: Span[Key, starts_origin]) raises Error -> DenseCursor[Self.origin]:
        """Opens a walk from `starts`, its pulls on the bound executor. Calls `usearch_dense_cursor_init`.

        Parameters:
            starts_origin: The start keys' origin, inferred from the argument.

        Args:
            starts: The keys to walk from, the walk yielding their rows first.

        Returns:
            The cursor, which borrows the index and the executor and cannot outlive them.

        Raises:
            `Error` if the residency has no cursor, as VRAM does not.
        """
        return DenseCursor[Self.origin](
            self.index[].library, self.index[].address, self.executor, len(starts), Int(starts.unsafe_ptr())
        )

    def search_stream[
        query_origin: ImmOrigin, //
    ](
        self, query: RowsView[query_origin], config: DenseSearchStreamInitConfig = DenseSearchStreamInitConfig()
    ) raises Error -> DenseSearchStream[ImmOrigin(origin_of(_anchor[Self.origin](), _anchor[query_origin]()))]:
        """Opens a resumable nearest-neighbor stream for one query, its pulls on the bound executor.
        Calls `usearch_dense_search_stream_init`.

        Parameters:
            query_origin: The query row's origin, inferred from the argument.

        Args:
            query: The single query row; a view of more rows has its first row read.
            config: The breadth of the candidates the walk explores.

        Returns:
            A stream yielding matches lazily, which borrows the index, the executor and the query.

        Raises:
            `Error` with `UNSUPPORTED` while no residency implements the stream.
        """
        return DenseSearchStream[ImmOrigin(origin_of(_anchor[Self.origin](), _anchor[query_origin]()))](
            self.index[].library, self.index[].address, self.executor, query.arguments(), config.expansion_factor
        )


def _insert_call[
    keys_origin: ImmOrigin, vectors_origin: ImmOrigin, stats_origin: MutOrigin
](
    keys: Span[Key, keys_origin], vectors: RowsView[vectors_origin], config: DenseInsertConfig[stats_origin]
) -> _DenseInsertCall:
    """Lowers an insertion's borrowed inputs into its crossing."""
    return _DenseInsertCall(
        KeysView.of(keys).untracked(),
        len(keys),
        vectors.arguments(),
        Int(vectors.rows),
        config.expansion_factor,
        _stats_address(config.stats),
    )


def _search_call[
    origin: MutOrigin, queries_origin: ImmOrigin, stats_origin: MutOrigin
](
    queries: RowsView[queries_origin],
    wanted: Int,
    arena: Pointer[Arena, origin],
    config: DenseSearchConfig[stats_origin],
) -> _DenseSearchCall[origin]:
    """Lowers a search's borrowed inputs into its crossing."""
    return _DenseSearchCall[origin](
        queries.arguments(),
        Int(queries.rows),
        wanted,
        arena,
        config.expansion_factor,
        config.traversal,
        config.predicate,
        config.predicate_state,
        _stats_address(config.stats),
    )


def _maxsim_call[
    origin: MutOrigin, queries_origin: ImmOrigin, stats_origin: MutOrigin
](
    queries: RowsView[queries_origin],
    rows_per_query: Int,
    wanted: Int,
    arena: Pointer[Arena, origin],
    config: DenseMaxsimConfig[stats_origin],
) -> _DenseMaxsimCall[origin]:
    """Lowers a late-interaction call's borrowed inputs into its crossing."""
    return _DenseMaxsimCall[origin](
        queries.arguments(),
        Int(queries.rows),
        rows_per_query,
        wanted,
        arena,
        config.expansion_factor,
        _stats_address(config.stats),
    )


def _multiply_call[
    origin: MutOrigin, left_origin: ImmOrigin, stats_origin: MutOrigin
](
    left: RowsView[left_origin], arena: Pointer[Arena, origin], config: DenseMultiplyConfig[stats_origin]
) -> _DenseMultiplyCall[origin]:
    """Lowers a product's borrowed inputs into its crossing."""
    return _DenseMultiplyCall[origin](
        left.arguments(),
        Int(left.rows),
        arena,
        config.wanted,
        config.expansion_factor,
        _stats_address(config.stats),
    )


def _attend_call[
    dtype: DType, origin: MutOrigin, queries_origin: ImmOrigin, carry_origin: MutOrigin, stats_origin: MutOrigin
](
    queries: RowsView[queries_origin],
    value_lookup: ValueLookup,
    value_lookup_state: Address,
    arena: Pointer[Arena, origin],
    config: DenseAttendConfig[carry_origin, stats_origin],
) -> _DenseAttendCall[dtype, origin]:
    """Lowers an attention call's borrowed inputs into its crossing."""
    return _DenseAttendCall[dtype, origin](
        queries.arguments(),
        Int(queries.rows),
        config.head_count,
        value_lookup,
        value_lookup_state,
        Int(config.carry.unsafe_ptr()) if len(config.carry) else NULL_ADDRESS,
        len(config.carry),
        arena,
        config.expansion_factor,
        config.score_scale,
        _stats_address(config.stats),
    )


struct DenseIndexDeferredOn[origin: ImmOrigin](ImplicitlyCopyable):
    """An index bound to the executor its verbs are submitted to, as `index.deferred(executor)` answers
    it: every verb answers a `Deferred` at once. Binding-only.

    The verbs and their arguments are those of `DenseVerbs`, the arena borrowed as theirs is; only the
    blocking receivers add the cursor and the search stream, whose every pull blocks.

    Parameters:
        origin: The index and the executor, which this receiver borrows.
    """

    var index: Pointer[DenseIndex, Self.origin]
    """The index every verb addresses."""
    var executor: Address
    """The `usearch_executor_t` every verb is submitted to."""

    def __init__(out self, index: Pointer[DenseIndex, Self.origin], executor: Address):
        """Binds an index to an executor.

        Args:
            index: The index every verb addresses.
            executor: The `usearch_executor_t` every verb is submitted to.
        """
        self.index = index
        self.executor = executor

    def reserve(self, capacity: Int) -> Deferred[_DenseReserveCall, Self.origin]:
        """Submits `DenseVerbs.reserve`. Calls `usearch_dense_reserve`."""
        return {self.index[].library, _DenseReserveCall(capacity), self.index[].address, self.executor}

    def clear(self) -> Deferred[_DenseClearCall, Self.origin]:
        """Submits `DenseVerbs.clear`. Calls `usearch_dense_clear`."""
        return {self.index[].library, _DenseClearCall(), self.index[].address, self.executor}

    def describe(self) -> Deferred[_DenseDescribeCall, Self.origin]:
        """Submits `DenseVerbs.describe`. Calls `usearch_dense_describe`."""
        return {self.index[].library, _DenseDescribeCall(), self.index[].address, self.executor}

    def insert[
        keys_origin: ImmOrigin, vectors_origin: ImmOrigin, stats_origin: MutOrigin, //
    ](
        self,
        keys: Span[Key, keys_origin],
        vectors: RowsView[vectors_origin],
        config: DenseInsertConfig[stats_origin] = DenseInsertConfig(),
    ) -> Deferred[
        _DenseInsertCall,
        ImmOrigin(
            origin_of(
                _anchor[Self.origin](), _anchor[keys_origin](), _anchor[vectors_origin](), _anchor[stats_origin]()
            )
        ),
    ]:
        """Submits `DenseVerbs.insert`; the keys, rows and telemetry slot stay borrowed until final.
        Calls `usearch_dense_insert`."""
        return {self.index[].library, _insert_call(keys, vectors, config), self.index[].address, self.executor}

    def refine[
        stats_origin: MutOrigin, //
    ](self, config: DenseRefineConfig[stats_origin] = DenseRefineConfig()) -> Deferred[
        _DenseRefineCall, ImmOrigin(origin_of(_anchor[Self.origin](), _anchor[stats_origin]()))
    ]:
        """Submits `DenseVerbs.refine`. Calls `usearch_dense_refine`."""
        return {
            self.index[].library,
            _DenseRefineCall(config.expansion_factor, _stats_address(config.stats)),
            self.index[].address,
            self.executor,
        }

    def health(self) -> Deferred[_DenseHealthCall, Self.origin]:
        """Submits `DenseVerbs.health`. Calls `usearch_dense_health`."""
        return {self.index[].library, _DenseHealthCall(), self.index[].address, self.executor}

    def absorb(self, var donors: List[DenseIndex]) -> Deferred[_DenseAbsorbCall, Self.origin]:
        """Submits `DenseVerbs.absorb`; the donors are owned by the call. Calls `usearch_dense_absorb`."""
        return {self.index[].library, _DenseAbsorbCall(donors^), self.index[].address, self.executor}

    def search[
        queries_origin: ImmOrigin, stats_origin: MutOrigin, //
    ](
        self,
        queries: RowsView[queries_origin],
        wanted: Int,
        mut arena: Arena,
        config: DenseSearchConfig[stats_origin] = DenseSearchConfig(),
    ) -> Deferred[
        _DenseSearchCall[origin_of(arena)],
        ImmOrigin(origin_of(_anchor[Self.origin](), _anchor[queries_origin](), _anchor[stats_origin]())),
    ]:
        """Submits `DenseVerbs.search`; the queries, telemetry slot and arena stay borrowed until final.
        Calls `usearch_dense_search`."""
        return {
            self.index[].library,
            _search_call(queries, wanted, Pointer(to=arena), config),
            self.index[].address,
            self.executor,
        }

    def count(self, key: Key) -> Deferred[_DenseCountCall, Self.origin]:
        """Submits `DenseVerbs.count`. Calls `usearch_dense_count`."""
        return {self.index[].library, _DenseCountCall(key), self.index[].address, self.executor}

    def get[
        dtype: DType
    ](self, key: Key, mut arena: Arena, scalar: ScalarKind = ScalarKind.of[dtype]()) -> Deferred[
        _DenseGetCall[dtype, origin_of(arena)], Self.origin
    ]:
        """Submits `DenseVerbs.get`; the arena stays borrowed until final. Calls `usearch_dense_get`."""
        return {
            self.index[].library,
            _DenseGetCall[dtype](key, scalar, Pointer(to=arena)),
            self.index[].address,
            self.executor,
        }

    def save[
        path_origin: ImmOrigin, //
    ](self, path: StringSlice[path_origin]) -> Deferred[
        _DenseSaveCall, ImmOrigin(origin_of(_anchor[Self.origin](), _anchor[path_origin]()))
    ]:
        """Submits `DenseVerbs.save`; the path stays borrowed until final. Calls `usearch_dense_save`."""
        return {
            self.index[].library,
            _DenseSaveCall(BytesView.of(path).untracked()),
            self.index[].address,
            self.executor,
        }

    def save_buffer(self, mut arena: Arena) -> Deferred[_DenseSaveBufferCall[origin_of(arena)], Self.origin]:
        """Submits `DenseVerbs.save_buffer`; the arena stays borrowed until final. Calls `usearch_dense_save_buffer`."""
        return {
            self.index[].library,
            _DenseSaveBufferCall(Pointer(to=arena)),
            self.index[].address,
            self.executor,
        }

    def multiply[
        left_origin: ImmOrigin, stats_origin: MutOrigin, //
    ](
        self,
        left: RowsView[left_origin],
        mut arena: Arena,
        config: DenseMultiplyConfig[stats_origin] = DenseMultiplyConfig(),
    ) -> Deferred[
        _DenseMultiplyCall[origin_of(arena)],
        ImmOrigin(origin_of(_anchor[Self.origin](), _anchor[left_origin](), _anchor[stats_origin]())),
    ]:
        """Submits `DenseVerbs.multiply`; the rows, telemetry slot and arena stay borrowed until final.
        Calls `usearch_dense_multiply`."""
        return {
            self.index[].library,
            _multiply_call(left, Pointer(to=arena), config),
            self.index[].address,
            self.executor,
        }

    def attend[
        queries_origin: ImmOrigin, carry_origin: MutOrigin, stats_origin: MutOrigin, //, dtype: DType
    ](
        self,
        queries: RowsView[queries_origin],
        value_lookup: ValueLookup,
        value_lookup_state: Address,
        mut arena: Arena,
        config: DenseAttendConfig[carry_origin, stats_origin] = DenseAttendConfig(),
    ) -> Deferred[
        _DenseAttendCall[dtype, origin_of(arena)],
        ImmOrigin(
            origin_of(
                _anchor[Self.origin](), _anchor[queries_origin](), _anchor[carry_origin](), _anchor[stats_origin]()
            )
        ),
    ]:
        """Submits `DenseVerbs.attend`; the queries, carry, telemetry slot and arena stay borrowed until
        final, and the lookup state must outlive the call too. Calls `usearch_dense_attend`."""
        return {
            self.index[].library,
            _attend_call[dtype](queries, value_lookup, value_lookup_state, Pointer(to=arena), config),
            self.index[].address,
            self.executor,
        }

    def maxsim[
        queries_origin: ImmOrigin, stats_origin: MutOrigin, //
    ](
        self,
        queries: RowsView[queries_origin],
        rows_per_query: Int,
        wanted: Int,
        mut arena: Arena,
        config: DenseMaxsimConfig[stats_origin] = DenseMaxsimConfig(),
    ) -> Deferred[
        _DenseMaxsimCall[origin_of(arena)],
        ImmOrigin(origin_of(_anchor[Self.origin](), _anchor[queries_origin](), _anchor[stats_origin]())),
    ]:
        """Submits `DenseVerbs.maxsim`; the queries, telemetry slot and arena stay borrowed until final.
        Calls `usearch_dense_maxsim`."""
        return {
            self.index[].library,
            _maxsim_call(queries, rows_per_query, wanted, Pointer(to=arena), config),
            self.index[].address,
            self.executor,
        }


struct DenseCursor[origin: ImmOrigin]:
    """A walk over the index contents - `usearch_dense_cursor_t`.

    Export, migration, and visualization feed off it. It borrows the index and the executor its
    pulls run on, so neither can be released while it lives.

    Parameters:
        origin: The index and the executor, which the cursor borrows.
    """

    var library: Library
    """The loaded engine, held so the release can still reach `usearch_dense_cursor_free`."""
    var index: Address
    """The `usearch_dense_index_t` the cursor was opened on, which every pull names again."""
    var executor: Address
    """The `usearch_executor_t` every pull runs on."""
    var address: Address
    """The `usearch_dense_cursor_t` this struct owns."""

    def __init__(
        out self, library: Library, index: Address, executor: Address, starts_count: Int, starts: Address
    ) raises Error:
        """Opens a cursor over `index`. Calls `usearch_dense_cursor_init`.

        Args:
            library: The loaded engine the cursor is opened through.
            index: The handle to scan.
            executor: The executor every pull runs on.
            starts_count: The start keys at `starts`; zero walks every row as stored.
            starts: The first start key, or `NULL_ADDRESS` for none.

        Raises:
            `Error` if the residency has no cursor, as VRAM does not.
        """
        var created = stack_allocation[1, Address]()
        created[unsafe_offset=0] = NULL_ADDRESS
        var outcome = stack_allocation[1, Outcome]()
        outcome[unsafe_offset=0] = Outcome()
        _cross(
            DenseCursorInitArguments(Int(created), index, Int(outcome), c_size_t(starts_count), starts),
            library.symbols().dense_cursor_init,
        )
        library._check(outcome.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0], "usearch_dense_cursor_init")
        self.library = library
        self.index = index
        self.executor = executor
        self.address = created.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0]

    def __deinit__(deinit self):
        """Releases the cursor. Calls `usearch_dense_cursor_free`.

        The loader reference is surrendered only after the call: reading the entry point is the
        last use of `library`, so without the trailing move Mojo drops the last reference - and
        `dlclose`s the engine - between the load and the branch to it.
        """
        self.library.symbols().dense_cursor_free(self.address)
        _ = self.library^

    def next_keys(self, wanted: Int, mut arena: Arena) raises Error -> DenseBatch[origin_of(arena)]:
        """Copies up to `wanted` next keys into a lent arena, skipping their links. Calls `usearch_dense_cursor_next`.

        Args:
            wanted: The upper bound on keys this pull writes.
            arena: The arena to rewind and refill, which the batch borrows; a failed call leaves it valid.

        Returns:
            The keys, read in place from the arena; an empty batch once the walk is exhausted.

        Raises:
            `Error` if the engine refuses the copy.
        """
        return self._next(_DenseCursorNextCall(self.address, False, wanted, Pointer(to=arena)))

    def next_links(self, wanted: Int, mut arena: Arena) raises Error -> DenseBatch[origin_of(arena)]:
        """Copies up to `wanted` next keys into a lent arena, each with the keys it links to.
        Calls `usearch_dense_cursor_next`.

        Args:
            wanted: The upper bound on entries this pull writes.
            arena: The arena to rewind and refill, which the batch borrows; a failed call leaves it valid.

        Returns:
            The keys and their links, read in place from the arena; an empty batch once the walk is exhausted.

        Raises:
            `Error` if the engine refuses the copy.
        """
        return self._next(_DenseCursorNextCall(self.address, True, wanted, Pointer(to=arena)))

    def _next[C: Call](self, var call: C) raises Error -> C.Result:
        """One blocking pull on the cursor's executor."""
        return _block(self.library, call^, self.index, self.executor)


struct DenseSearchStream[origin: ImmOrigin]:
    """A resumable nearest-neighbor stream - `usearch_dense_search_stream_t`.

    Best-first traversal is naturally pausable, so the stream keeps the walk's frontier and visited
    set between pulls. Callers that post-filter aggressively pull until satisfied instead of
    guessing "k" upfront. Ordering beyond the expansion horizon is best-effort nearest rather than
    globally exact.

    Parameters:
        origin: The index, the executor, and the query row, which the stream borrows.
    """

    var library: Library
    """The loaded engine, held so the release can still reach `usearch_dense_search_stream_free`."""
    var index: Address
    """The `usearch_dense_index_t` the stream was opened on, which every pull names again."""
    var executor: Address
    """The `usearch_executor_t` every pull runs on."""
    var address: Address
    """The `usearch_dense_search_stream_t` this struct owns."""

    def __init__(
        out self, library: Library, index: Address, executor: Address, query: RowsViewArguments, expansion_factor: Int
    ) raises Error:
        """Opens a stream over `index` for one query row. Calls `usearch_dense_search_stream_init`.

        Args:
            library: The loaded engine the stream is opened through.
            index: The handle to query.
            executor: The executor every pull runs on.
            query: The single query row.
            expansion_factor: The breadth of the candidates the walk explores; zero derives.

        Raises:
            `Error` with `UNSUPPORTED` while no residency implements the stream.
        """
        var created = stack_allocation[1, Address]()
        created[unsafe_offset=0] = NULL_ADDRESS
        var outcome = stack_allocation[1, Outcome]()
        outcome[unsafe_offset=0] = Outcome()
        _cross(
            DenseSearchStreamInitArguments(Int(created), index, Int(outcome), query, c_size_t(expansion_factor)),
            library.symbols().dense_search_stream_init,
        )
        library._check(outcome.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0], "usearch_dense_search_stream_init")
        self.library = library
        self.index = index
        self.executor = executor
        self.address = created.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0]

    def __deinit__(deinit self):
        """Releases the stream and the walk state it keeps. Calls `usearch_dense_search_stream_free`.

        The loader reference is surrendered only after the call: reading the entry point is the
        last use of `library`, so without the trailing move Mojo drops the last reference - and
        `dlclose`s the engine - between the load and the branch to it.
        """
        self.library.symbols().dense_search_stream_free(self.address)
        _ = self.library^

    def next(self, wanted: Int, mut arena: Arena) raises Error -> Matches[origin_of(arena)]:
        """Pulls up to `wanted` next-nearest matches into a lent arena, as the one row of a search.
        Calls `usearch_dense_search_stream_next`.

        Args:
            wanted: The upper bound on matches this pull writes.
            arena: The arena to rewind and refill, which the pull borrows; a failed call leaves it valid.

        Returns:
            One row of matches, read in place from the arena; an empty row once the walk is drained.

        Raises:
            `Error` if the engine refuses to advance.
        """
        return _block(
            self.library,
            _DenseSearchStreamNextCall(self.address, wanted, Pointer(to=arena)),
            self.index,
            self.executor,
        )


# endregion Index

# region Metadata


def _read_metadata(
    executor: Executor, verb: DenseMetadataRead, source: BytesView[ImmUntrackedOrigin], detail: StaticString
) raises Error -> DenseMetadata:
    """The path and buffer forms share one argument layout, differing only in the verb."""
    var described = stack_allocation[1, DenseMetadata]()
    described[unsafe_offset=0] = DenseMetadata()
    var outcome = stack_allocation[1, Outcome]()
    outcome[unsafe_offset=0] = Outcome()
    _cross(
        DenseMetadataReadArguments(executor.address, Int(outcome), Completion.BLOCKING, source, Int(described)),
        verb,
    )
    executor.library._check(outcome.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0], detail)
    return described.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0]


def dense_metadata_read[
    path_origin: ImmOrigin, //
](executor: Executor, path: StringSlice[path_origin]) raises Error -> DenseMetadata:
    """Reads a checkpoint's geometry and metric without loading it. Calls `usearch_dense_metadata_read`.

    A manifest answers from its summary and its first part's header, so the cost does not grow with
    the index.

    Parameters:
        path_origin: The path bytes' origin, inferred from the argument.

    Args:
        executor: The executor the read runs on; it also lends its engine.
        path: The manifest of a sharded checkpoint, or a single monolith.

    Returns:
        The geometry and metric the checkpoint froze.

    Raises:
        `Error` if the checkpoint cannot be opened or its header cannot be decoded.
    """
    return _read_metadata(
        executor,
        executor.library.symbols().dense_metadata_read,
        BytesView.of(path).untracked(),
        "usearch_dense_metadata_read",
    )


def dense_metadata_read_buffer[
    buffer_origin: ImmOrigin, //
](executor: Executor, buffer: Span[UInt8, buffer_origin]) raises Error -> DenseMetadata:
    """Reads a serialized monolith's geometry and metric without loading it.
    Calls `usearch_dense_metadata_read_buffer`.

    Parameters:
        buffer_origin: The source bytes' origin, inferred from the argument.

    Args:
        executor: The executor the read runs on; it also lends its engine.
        buffer: Source holding one serialized monolith.

    Returns:
        The geometry and metric the monolith froze.

    Raises:
        `Error` if the bytes cannot be decoded.
    """
    return _read_metadata(
        executor,
        executor.library.symbols().dense_metadata_read_buffer,
        BytesView.of(buffer).untracked(),
        "usearch_dense_metadata_read_buffer",
    )


# endregion Metadata
