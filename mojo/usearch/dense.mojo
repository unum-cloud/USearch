"""Vector search over equidimensional embeddings: the index, its receivers, its cursor, and its search stream.

The type is `DenseIndex` rather than `Index` because the engine has sibling modalities that will
bind beside it, and a bare name would have to be renamed the day one of them lands.

Every verb takes one argument structure named after it, and every one of them is mirrored below as
a `...Arguments` struct laid out exactly as the library reads it, materialized in a
`stack_allocation` slot and crossed by address. The receiver picks the completion: verbs on the
index and on `index.on(executor)` pass blocking completion and return with their outcome final and
their outputs readable, while `index.deferred(executor)` passes deferred completion and answers a
`Deferred` at once. Each verb is one `Call` all three route, so the receivers stay symmetric.

Batches are rows. `RowsView` holds the base pointer, the row count, the column count, the byte
stride, and the scalar kind as one value, so a single vector is a one-row view. The row count
becomes the verb's `count`, and the column count crosses with the rows, so the engine checks it
against the geometry the index froze.

Every output whose length scales with the call - the matches of `search` and `maxsim`, the rows of
`get` and `attend`, the pairs of `multiply`, the monolith of `save_buffer`, and every cursor and
stream pull - lands in an `Arena` the verb requires, and is read in place. Every receiver takes the arena
as the verb's last required argument, deferred ones too, and the result views it in turn, named by
the arena's origin.

A closure capturing anything cannot be handed to C at all, so a callback's state crosses beside it
as an `Address`, and a Mojo handler is a comptime parameter of a capture-free `abi("C")`
trampoline that types the state again. The value lookup of `attend` is such a function, a
`ValueLookup`; the predicate of `search` and every `progress` slot are installed the same way by
`PredicateArguments.each`, `PredicateArguments.batch` and `ProgressArguments.of`, or take a handler
compiled in C.

Telemetry is a passed output. Every config of a verb that reports traversal statistics takes an
optional `stats` pointer, which the engine overwrites whole on every call it is passed to.

File: mojo/usearch/dense.mojo
Author: Ash Vardanian
Date: September 7, 2026
"""

from std.ffi import c_int, c_size_t
from std.memory import stack_allocation
from std.sys.info import size_of

from usearch.library import Library
from usearch.types import (
    Address,
    Arena,
    ArenaRead,
    BytesView,
    Call,
    CallHead,
    Completion,
    CPUClass,
    Deferred,
    Distance,
    Error,
    Executor,
    GPUClass,
    Key,
    KeyBytes,
    KeysTapeViewArguments,
    KeysView,
    LoadPath,
    Matches,
    MatchesTapeViewArguments,
    MatchesViewArguments,
    Monolith,
    NULL_ADDRESS,
    NvmeOptions,
    Outcome,
    PredicateArguments,
    PredicateCost,
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
    _cross,
    _in_arena,
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
    """L2 distance, zero only for identical vectors."""
    comptime HAMMING = Self(4)
    """Count of differing bits, so its range is the dimensionality."""
    comptime JACCARD = Self(5)
    """Differing bits over union bits in [0, 1], ignoring the zeros both vectors share."""


@fieldwise_init
struct DenseSoftmaxState(ImplicitlyCopyable, TrivialRegisterPassable):
    """The online-softmax carry of one query row - `usearch_dense_softmax_state_t`.

    Flash Attention's running maximum and partition. The two are one concept: a maximum without
    its partition is a number nothing can rescale, so they share an element rather than travelling
    as two runs a caller could supply half of. Holding them folds one attention call into another
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
    var missing_hints: c_size_t
    """Hint keys no shard held an entry under; zero unless a search was handed hints."""

    def __init__(out self):
        """Zeroes every counter, so an unwritten slot reads as an untaken walk."""
        self.visited_members = c_size_t(0)
        self.computed_distances = c_size_t(0)
        self.restarts = c_size_t(0)
        self.missing_hints = c_size_t(0)


@fieldwise_init
struct DenseReachability(Defaultable, ImplicitlyCopyable):
    """One reachability audit - `usearch_reachability_t`: the population it judged, and how much
    of it a search reaches."""

    var audited: c_size_t
    """Entries published when the audit started; compare `reachable` against this, never a later size."""
    var reachable: c_size_t
    """Of those, the entries a search reaches; the rest no search finds until a `refine` re-links them."""
    var depth_from_root: c_size_t
    """Depths the walk from the root took to converge."""
    var depth_to_root: c_size_t
    """Depths the walk back toward the root took, where a health check walks both ways."""
    var unlinked: c_size_t
    """Entries no reached entry links to."""
    var underlinked: c_size_t
    """Entries linked to by fewer entries than they link to."""
    var beyond_most_populated_depth: c_size_t
    """Entries deeper than the depth holding the most entries."""
    var unreached_islands: c_size_t
    """Islands no entry outside them links into."""
    var stranded_islands: c_size_t
    """Islands with no link out."""
    var components: c_size_t
    """Components when direction is ignored."""

    def __init__(out self):
        """Zeroes the audit, which is what a failed call leaves."""
        self.audited = c_size_t(0)
        self.reachable = c_size_t(0)
        self.depth_from_root = c_size_t(0)
        self.depth_to_root = c_size_t(0)
        self.unlinked = c_size_t(0)
        self.underlinked = c_size_t(0)
        self.beyond_most_populated_depth = c_size_t(0)
        self.unreached_islands = c_size_t(0)
        self.stranded_islands = c_size_t(0)
        self.components = c_size_t(0)


@fieldwise_init
struct RefineVerdict(Equatable, ImplicitlyCopyable, TrivialRegisterPassable):
    """What a round concluded - `usearch_refine_verdict_t`, so no caller picks a change rate."""

    var code: c_int
    """The raw `usearch_refine_verdict_t` value the ABI exchanges."""
    comptime NONE = Self(0)
    """No round concluded: a report no round filled yet."""
    comptime IMPROVED = Self(1)
    """The round changed some entry's link set, so another round can help."""
    comptime CONVERGED = Self(2)
    """The round changed no entry's link set."""
    comptime STALLED = Self(3)
    """The round lowered nothing while something stands, as full entries leave no slot."""
    comptime STOPPED = Self(4)
    """The caller stopped the round; the phase names where."""


@fieldwise_init
struct Phase(Equatable, ImplicitlyCopyable, TrivialRegisterPassable):
    """The kind of work a plan is doing - `usearch_phase_t`, one value shared by refine and reshard.
    `Library.phase_name` spells each one as the engine does."""

    var code: c_int
    """The raw `usearch_phase_t` value the ABI exchanges."""
    comptime NONE = Self(0)
    """No plan, or one that has not started."""
    comptime COPY = Self(1)
    """A reshard moving keys and values into a target, with the links it keeps."""
    comptime BUILD = Self(2)
    """A reshard on NVMe building blocks of a target's rows in RAM and writing their links."""
    comptime PARTITION = Self(3)
    """A reshard cutting a source into one part per target."""
    comptime SCATTER = Self(4)
    """A reshard moving each part into its target."""
    comptime HEALTH = Self(5)
    """A round's health check, opening or closing it."""
    comptime HEAL = Self(6)
    """A round linking each island with the root's component."""
    comptime HUBS = Self(7)
    """A round drawing the hubs its gather streams entries past."""
    comptime GATHER = Self(8)
    """A round gathering every entry into a neighborhood."""
    comptime RESTITCH = Self(9)
    """A round re-stitching each neighborhood."""
    comptime WRITE = Self(10)
    """A round on NVMe rewriting each tile of links from what its restitch changed."""
    comptime DONE = Self(11)
    """The plan ended its round, and nothing more is pending."""


@fieldwise_init
struct PhaseProgress(Defaultable, ImplicitlyCopyable):
    """Where a plan stands - `usearch_phase_progress_t`: its phase, that phase's position, and its own count."""

    var phase: Phase
    """The kind of work in hand."""
    var ordinal: c_size_t
    """This phase's position among the ones the plan runs."""
    var completed: c_size_t
    """Units of this phase done, entries or islands."""
    var total: c_size_t
    """Units of this phase in all."""

    def __init__(out self):
        """No plan started, which is what an unwritten slot reads as."""
        self.phase = Phase.NONE
        self.ordinal = c_size_t(0)
        self.completed = c_size_t(0)
        self.total = c_size_t(0)


struct DenseRefinePlan(Movable):
    """One refine in flight - `usearch_dense_refine_plan_t`. Binding-only.

    Passed through `DenseRefineConfig.plan`, it opens on the first refine, stays live past a cancel or
    the end of a round, continues on every later refine, and closes with `refine_plan_finish`. While
    it lives, the index admits no insertion and no save.
    """

    var address: Address
    """The opaque handle, `NULL_ADDRESS` until a refine opens it."""

    def __init__(out self):
        """A plan no refine opened yet."""
        self.address = NULL_ADDRESS


struct DenseReshardPlan(Movable):
    """One reshard in flight - `usearch_dense_reshard_plan_t`, with its own lists of sources and targets.
    Binding-only.

    Passed through `DenseReshardConfig.plan`, it opens on the first reshard, stays live past a cancel,
    moves what is still pending and continues the rounds on every later one, and closes with
    `dense_reshard_plan_finish`. While it lives, every index it names admits no insertion and no save.
    """

    var address: Address
    """The opaque handle, `NULL_ADDRESS` until a reshard opens it."""

    def __init__(out self):
        """A plan no reshard opened yet."""
        self.address = NULL_ADDRESS


@fieldwise_init
struct DenseRefineReport(Defaultable, ImplicitlyCopyable):
    """What one `refine` or `reshard` did to the graph - `usearch_refine_report_t`."""

    var reselected: c_size_t
    """Entries whose neighbors were selected again, the islands of every heal round included."""
    var changed: c_size_t
    """Of those, the entries whose neighbor set gained a member it lacked; zero means settled."""
    var islands_found: c_size_t
    """Entries the first audit found not reached both ways."""
    var islands_reconnected: c_size_t
    """Links written between an island and an entry the audit reached both ways."""
    var neighborhoods: c_size_t
    """Neighborhoods the round re-stitched."""
    var widest_neighborhood: c_size_t
    """Entries of the widest neighborhood."""
    var added_links: c_size_t
    """Links the round added."""
    var dropped_links: c_size_t
    """Links the round dropped."""
    var verdict: RefineVerdict
    """What the round concluded."""
    var opening: DenseReachability
    """The health check opening the round."""
    var reachability: DenseReachability
    """The health check closing the heal, as a fresh `health` right after would take it."""

    def __init__(out self):
        """Zeroes the report, which is what a failed call leaves."""
        self.reselected = c_size_t(0)
        self.changed = c_size_t(0)
        self.islands_found = c_size_t(0)
        self.islands_reconnected = c_size_t(0)
        self.neighborhoods = c_size_t(0)
        self.widest_neighborhood = c_size_t(0)
        self.added_links = c_size_t(0)
        self.dropped_links = c_size_t(0)
        self.verdict = RefineVerdict.NONE
        self.opening = DenseReachability()
        self.reachability = DenseReachability()


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
    """Upper bound on entries sharing one key; zero is unbounded."""

    def __init__(
        out self,
        dimensions: Int,
        metric: DenseMetric = DenseMetric.ANGULAR,
        quantization: ScalarKind = ScalarKind.BF16,
        connectivity: Int = 0,
        values_per_key: Int = 0,
    ):
        """Describes an index by its geometry, angular over `BF16` vectors unless told otherwise.

        Args:
            dimensions: Dimensions per vector.
            metric: The metric kind the graph is frozen under.
            quantization: The scalar type vectors are stored as.
            connectivity: The connections-per-node bound; zero derives.
            values_per_key: Upper bound on entries sharing one key; zero is unbounded.
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
    state rather than anything the serialized form holds.

    Never marked register-passable: its address crosses to C, and Mojo mislays a struct once it is,
    so the probes would read back nothing.
    """

    var metric: DenseMetric
    """The metric kind the checkpoint scores with."""
    var quantization: ScalarKind
    """The scalar type its vectors are stored and quantized into."""
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
struct DenseIndexDescription(Defaultable, ImplicitlyCopyable):
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
    var cpu_class: CPUClass
    """The CPU class dispatch picked on this machine; `UNKNOWN` for a device index."""
    var gpu_class: GPUClass
    """The GPU class dispatch picked on the index's devices; `UNKNOWN` for a host index."""
    var shards_count: c_size_t
    """Live shards."""
    var load_path: LoadPath
    """How the last load from a path read the parts onto the devices."""

    def __init__(out self):
        """Zeroes every field, so an unwritten slot reads as an empty index."""
        self.metadata = DenseMetadata()
        self.size = c_size_t(0)
        self.capacity = c_size_t(0)
        self.memory_usage = c_size_t(0)
        self.cpu_class = CPUClass.UNKNOWN
        self.gpu_class = GPUClass.UNKNOWN
        self.shards_count = c_size_t(0)
        self.load_path = LoadPath.NONE


@fieldwise_init
struct DenseVectors[dtype: DType, origin: MutOrigin](Movable, Sized):
    """Rows a call wrote into its arena - `usearch_rows_view_t` - read in place: the entries `get`
    exports, or the rows `attend` attends.

    Rows are `stride` bytes apart and `columns` values wide, both as the engine reported them, so a
    padded row never reads as extra values.

    Parameters:
        dtype: The element type each row is read as.
        origin: The arena the rows live in, which this result views.
    """

    var view: RowsViewArguments
    """The first row, the stride, the width and the kind, as the engine wrote them."""
    var rows: Int
    """Rows written; zero when the key is absent."""

    var viewing: ArenaRead
    """The arena the result reads and the generation the call that wrote it brought it to."""

    def __len__(self) -> Int:
        """The rows written.

        Returns:
            Zero when the key is absent.
        """
        return self.rows

    def __getitem__(ref self, row: Int) raises Error -> Span[Scalar[Self.dtype], origin_of(self)]:
        """One row, viewed from the arena this result reads. Binding-only.

        Args:
            row: Which row, below `len(self)`.

        Returns:
            The row's elements - `columns` of them, or its packed bytes for a sub-byte kind.
        """
        self.viewing.open("DenseVectors.__getitem__")
        var viewed = _in_arena[Scalar[Self.dtype], origin_of(self)](
            self.view.data + row * Int(self.view.stride),
            self.view.scalar.bytes_for_columns(Int(self.view.columns)) // size_of[Scalar[Self.dtype]](),
        )
        self.viewing.close()
        return viewed

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
        origin: The arena the batch lives in, which this result views.
    """

    var keys_address: Address
    """The first key, inside the arena."""
    var linked: Bool
    """Whether the pull exported links."""
    var links_view: KeysTapeViewArguments
    """The keys each entry links to and their offsets, null for a pull of keys alone."""
    var written: Int
    """Entries the pull wrote; zero when the walk is exhausted."""

    var viewing: ArenaRead
    """The arena the result reads and the generation the call that wrote it brought it to."""

    def __len__(self) -> Int:
        """The entries the pull wrote.

        Returns:
            Zero when the walk is exhausted.
        """
        return self.written

    def keys(ref self) raises Error -> Span[Key, origin_of(self)]:
        """Every entry's key, in walk order.

        Returns:
            The keys, viewed from the arena this result reads.
        """
        self.viewing.open("DenseBatch.keys")
        var viewed = _in_arena[Key, origin_of(self)](self.keys_address, self.written)
        self.viewing.close()
        return viewed

    def links(ref self, entry: Int) raises Error -> Span[Key, origin_of(self)]:
        """The keys one entry links to, as the graph stored them when the pull ran. Binding-only.

        Args:
            entry: Which entry, below `len(self)`.

        Returns:
            The linked keys, viewed from the arena this result reads.

        Raises:
            `Error` with `INVALID_ARGUMENT` for a pull of keys alone or an entry past the batch.
        """
        if not self.linked or entry < 0 or entry >= self.written:
            raise Error(Status.INVALID_ARGUMENT, "DenseBatch.links")
        self.viewing.open("DenseBatch.links")
        var offsets = _in_arena[c_size_t, origin_of(self)](self.links_view.offsets, self.written + 1)
        var begin = Int(offsets[entry])
        var viewed = _in_arena[Key, origin_of(self)](
            self.links_view.keys + begin * size_of[Key](), Int(offsets[entry + 1]) - begin
        )
        self.viewing.close()
        return viewed


@fieldwise_init
struct DenseDraws[origin: MutOrigin](Movable, Sized):
    """The draws `sample` wrote into its arena, read in place: each draw's key, which of that key's
    entries it is, the shown or earlier draw it hangs from, and where it is placed.

    Parameters:
        origin: The arena the draws live in, which this result views.
    """

    var outputs: DenseSampleOutputs
    """Where the engine wrote every output, inside the arena."""

    var viewing: ArenaRead
    """The arena the result reads and the generation the call that wrote it brought it to."""

    def __len__(self) -> Int:
        """The draws written.

        Returns:
            Fewer than wanted only once the region is exhausted.
        """
        return Int(self.outputs.written)

    def keys(ref self) raises Error -> Span[Key, origin_of(self)]:
        """Every draw's key.

        Returns:
            The keys, viewed from the arena this result reads.
        """
        self.viewing.open("DenseDraws.keys")
        var viewed = _in_arena[Key, origin_of(self)](self.outputs.keys, len(self))
        self.viewing.close()
        return viewed

    def entry_ordinals(ref self) raises Error -> Span[c_size_t, origin_of(self)]:
        """Which of its key's entries each draw is, in the order `get` writes them.

        Returns:
            The entry ordinals, viewed from the arena this result reads.
        """
        self.viewing.open("DenseDraws.entry_ordinals")
        var viewed = _in_arena[c_size_t, origin_of(self)](self.outputs.entry_ordinals, len(self))
        self.viewing.close()
        return viewed

    def parents(self) -> Matches[Self.origin]:
        """Each draw's parent and its distance, one column per draw, a root holding none.

        Returns:
            The parents, read in place from the same arena.
        """
        return {self.outputs.parents, len(self), self.viewing}

    def coordinates(self) -> DenseVectors[DType.float32, Self.origin]:
        """Where each draw is placed, one row of three values apiece.

        Returns:
            The places, read in place from the same arena.
        """
        return {self.outputs.coordinates, len(self), self.viewing}


@fieldwise_init
struct DenseProducts[origin: MutOrigin](Movable):
    """The ragged output of `multiply` and `maxsim` - `usearch_matches_tape_view_t` - read in place
    from its arena: query `i` spans pairs `offsets()[i]` up to `offsets()[i + 1]` of `keys()` and
    `scores()`.

    Parameters:
        origin: The arena the pairs live in, which this result views.
    """

    var view: MatchesTapeViewArguments
    """The keys, products and offsets, as the engine wrote them."""
    var queries: Int
    """Left-hand rows the call multiplied, or queries it scored."""
    var pairs: Int
    """Pairs written across every query."""
    var required: Int
    """Pairs the walks scored, or keys the queries scored for `maxsim`, before `wanted` cut any;
    above `pairs` means some lists were truncated."""

    var viewing: ArenaRead
    """The arena the result reads and the generation the call that wrote it brought it to."""

    def offsets(ref self) raises Error -> Span[c_size_t, origin_of(self)]:
        """One more offset than queries, as Apache Arrow lays a ragged batch out.

        Returns:
            The offsets into `keys()` and `scores()`, viewed from the arena this result reads.
        """
        self.viewing.open("DenseProducts.offsets")
        var viewed = _in_arena[c_size_t, origin_of(self)](self.view.offsets, self.queries + 1)
        self.viewing.close()
        return viewed

    def keys(ref self) raises Error -> Span[Key, origin_of(self)]:
        """Each pair's indexed entry, viewed from the arena this result reads.

        Returns:
            One key per pair.
        """
        self.viewing.open("DenseProducts.keys")
        var viewed = _in_arena[Key, origin_of(self)](self.view.keys, self.pairs)
        self.viewing.close()
        return viewed

    def scores(ref self) raises Error -> Span[Distance, origin_of(self)]:
        """Each pair's product, positionally paired with `keys()`.

        Returns:
            One score per pair.
        """
        self.viewing.open("DenseProducts.scores")
        var viewed = _in_arena[Distance, origin_of(self)](self.view.scores, self.pairs)
        self.viewing.close()
        return viewed


# endregion Configuration

# region Configs


def _slot[
    T: AnyType, origin: MutOrigin
](passed: Optional[Pointer[T, origin]]) -> Optional[Pointer[T, MutUntrackedOrigin]]:
    """A slot a config passes with its origin erased, or `None` for one it passes none."""
    if case .Some(slot) = passed:
        return slot.unsafe_origin_cast[MutUntrackedOrigin]()
    return None


struct DenseInsertConfig[stats_origin: MutOrigin = MutUntrackedOrigin](ImplicitlyCopyable):
    """The knobs of `usearch_dense_insert_t`; zeroed or absent ones derive what the C interface derives.

    Parameters:
        stats_origin: The passed telemetry slot's origin, inferred where one is passed.
    """

    var expansion_factor: Int
    """The breadth of the candidates the build walk explores; zero derives."""
    var progress: ProgressArguments
    """Hears the entries admitted so far; an insertion reports but never cancels."""
    var stats: Optional[Pointer[DenseTraversalStats, Self.stats_origin]]
    """Receives this call's traversal telemetry, fully overwritten; absent skips reporting."""

    def __init__(
        out self,
        *,
        expansion_factor: Int = 0,
        progress: ProgressArguments = ProgressArguments(),
        stats: Optional[Pointer[DenseTraversalStats, Self.stats_origin]] = Optional[
            Pointer[DenseTraversalStats, Self.stats_origin]
        ](),
    ):
        """Names only the knobs that differ from the derived ones.

        Args:
            expansion_factor: The breadth of the candidates the build walk explores; zero derives.
            progress: Hears the entries admitted so far; an insertion reports but never cancels.
            stats: Receives this call's traversal telemetry; absent skips reporting.
        """
        self.expansion_factor = expansion_factor
        self.progress = progress
        self.stats = stats


struct DenseRefineConfig[
    stats_origin: MutOrigin = MutUntrackedOrigin,
    plan_origin: MutOrigin = MutUntrackedOrigin,
    phase_origin: MutOrigin = MutUntrackedOrigin,
](ImplicitlyCopyable):
    """The knobs of `usearch_dense_refine_t`; zeroed or absent ones derive what the C interface derives.

    Parameters:
        stats_origin: The passed telemetry slot's origin, inferred where one is passed.
        plan_origin: The passed plan's origin, inferred where one is passed.
        phase_origin: The passed phase slot's origin, inferred where one is passed.
    """

    var expansion_factor: Int
    """The breadth of any walk beside the round; zero derives."""
    var memory_limit: Int
    """The bytes a loaded NVMe index holds for the one part it rewrites; zero derives."""
    var progress: ProgressArguments
    """Hears every phase's units at its barriers; a cancel stops the round at that barrier."""
    var stats: Optional[Pointer[DenseTraversalStats, Self.stats_origin]]
    """Receives this call's traversal telemetry, fully overwritten; absent skips reporting."""
    var plan: Optional[Pointer[DenseRefinePlan, Self.plan_origin]]
    """The plan this call opens or continues; absent runs one round to its end."""
    var phase: Optional[Pointer[PhaseProgress, Self.phase_origin]]
    """Receives where this call stopped; absent skips it."""

    def __init__(
        out self,
        *,
        expansion_factor: Int = 0,
        memory_limit: Int = 0,
        progress: ProgressArguments = ProgressArguments(),
        stats: Optional[Pointer[DenseTraversalStats, Self.stats_origin]] = Optional[
            Pointer[DenseTraversalStats, Self.stats_origin]
        ](),
        plan: Optional[Pointer[DenseRefinePlan, Self.plan_origin]] = Optional[
            Pointer[DenseRefinePlan, Self.plan_origin]
        ](),
        phase: Optional[Pointer[PhaseProgress, Self.phase_origin]] = Optional[
            Pointer[PhaseProgress, Self.phase_origin]
        ](),
    ):
        """Names only the knobs that differ from the derived ones.

        Args:
            expansion_factor: The breadth of any walk beside the round; zero derives.
            memory_limit: The bytes a loaded NVMe index holds for the one part it rewrites; zero derives.
            progress: Hears every phase's units at its barriers; a cancel stops the round at that barrier.
            stats: Receives this call's traversal telemetry; absent skips reporting.
            plan: The plan this call opens or continues; absent runs one round to its end.
            phase: Receives where this call stopped; absent skips it.
        """
        self.expansion_factor = expansion_factor
        self.memory_limit = memory_limit
        self.progress = progress
        self.stats = stats
        self.plan = plan
        self.phase = phase


struct DenseReshardConfig[
    ram_origin: ImmOrigin = ImmUntrackedOrigin,
    vram_origin: ImmOrigin = ImmUntrackedOrigin,
    nvme_origin: ImmOrigin = ImmUntrackedOrigin,
    stats_origin: MutOrigin = MutUntrackedOrigin,
    plan_origin: MutOrigin = MutUntrackedOrigin,
    phase_origin: MutOrigin = MutUntrackedOrigin,
](ImplicitlyCopyable):
    """The knobs of `usearch_dense_reshard_t`; zeroed or absent ones derive what the C interface derives.

    Parameters:
        ram_origin: The RAM placement's viewed domain array, inferred where one is named.
        vram_origin: The VRAM placement's viewed ordinal array, inferred where one is named.
        nvme_origin: The NVMe placement's viewed path array, inferred where one is named.
        stats_origin: The passed telemetry slot's origin, inferred where one is passed.
        plan_origin: The passed plan's origin, inferred where one is passed.
        phase_origin: The passed phase slot's origin, inferred where one is passed.
    """

    var ram: Optional[RamOptions[Self.ram_origin]]
    """The placement of a RAM index recut in place; absent keeps it."""
    var vram: Optional[VramOptions[Self.vram_origin]]
    """The placement of a VRAM index recut in place; absent keeps it."""
    var nvme: Optional[NvmeOptions[Self.nvme_origin]]
    """The placement of an NVMe index recut in place; absent keeps it."""
    var expansion_factor: Int
    """The breadth of any walk beside the closing rounds; zero derives."""
    var memory_limit: Int
    """The bytes a loaded NVMe index holds for the one part it rewrites; zero derives."""
    var seed: UInt64
    """Picks the part roots of a source cut among several targets; equal seeds cut equally."""
    var progress: ProgressArguments
    """Hears the closing rounds' phases at their barriers; a cancel stops them, never a move."""
    var stats: Optional[Pointer[DenseTraversalStats, Self.stats_origin]]
    """Receives this call's traversal telemetry, fully overwritten; absent skips reporting."""
    var plan: Optional[Pointer[DenseReshardPlan, Self.plan_origin]]
    """The plan this call opens or continues; absent moves and runs to the end."""
    var phase: Optional[Pointer[PhaseProgress, Self.phase_origin]]
    """Receives where this call stopped; absent skips it."""

    def __init__(
        out self,
        *,
        ram: Optional[RamOptions[Self.ram_origin]] = Optional[RamOptions[Self.ram_origin]](),
        vram: Optional[VramOptions[Self.vram_origin]] = Optional[VramOptions[Self.vram_origin]](),
        nvme: Optional[NvmeOptions[Self.nvme_origin]] = Optional[NvmeOptions[Self.nvme_origin]](),
        expansion_factor: Int = 0,
        memory_limit: Int = 0,
        seed: UInt64 = 0,
        progress: ProgressArguments = ProgressArguments(),
        stats: Optional[Pointer[DenseTraversalStats, Self.stats_origin]] = Optional[
            Pointer[DenseTraversalStats, Self.stats_origin]
        ](),
        plan: Optional[Pointer[DenseReshardPlan, Self.plan_origin]] = Optional[
            Pointer[DenseReshardPlan, Self.plan_origin]
        ](),
        phase: Optional[Pointer[PhaseProgress, Self.phase_origin]] = Optional[
            Pointer[PhaseProgress, Self.phase_origin]
        ](),
    ):
        """Names only the knobs that differ from the derived ones.

        Args:
            ram: The placement of a RAM index recut in place; absent keeps it.
            vram: The placement of a VRAM index recut in place; absent keeps it.
            nvme: The placement of an NVMe index recut in place; absent keeps it.
            expansion_factor: The breadth of any walk beside the closing rounds; zero derives.
            memory_limit: The bytes a loaded NVMe index holds for the one part it rewrites; zero derives.
            seed: Picks the part roots of a source cut among several targets.
            progress: Hears the closing rounds' phases at their barriers; a cancel stops them, never a move.
            stats: Receives this call's traversal telemetry; absent skips reporting.
            plan: The plan this call opens or continues; absent moves and runs to the end.
            phase: Receives where this call stopped; absent skips it.
        """
        self.ram = ram
        self.vram = vram
        self.nvme = nvme
        self.expansion_factor = expansion_factor
        self.memory_limit = memory_limit
        self.seed = seed
        self.progress = progress
        self.stats = stats
        self.plan = plan
        self.phase = phase


struct DenseSearchConfig[stats_origin: MutOrigin = MutUntrackedOrigin](ImplicitlyCopyable):
    """The knobs of `usearch_dense_search_t`; zeroed or absent ones derive what the C interface derives.

    A `predicate` narrows the matches to the entries it admits: a Mojo handler installed by
    `PredicateArguments.each` or `PredicateArguments.batch`, or a `usearch_predicate_callback_t`
    compiled in C. Non-matching entries still route the traversal, so the predicate judges every
    scored candidate, concurrently from the executor's threads: a cheap one as candidates are
    scored, an expensive one once per round.

    Parameters:
        stats_origin: The passed telemetry slot's origin, inferred where one is passed.
    """

    var expansion_factor: Int
    """The breadth of the candidates the walk explores; zero derives."""
    var traversal: Traversal
    """A walk, or an exact scan of every entry on RAM and NVMe handles; VRAM answers `UNSUPPORTED`."""
    var predicate: PredicateArguments
    """The handler and state judging which candidates are reported; the default admits all."""
    var progress: ProgressArguments
    """Hears the queries resolved so far; a cancel stops the search with no matches."""
    var stats: Optional[Pointer[DenseTraversalStats, Self.stats_origin]]
    """Receives this call's traversal telemetry, fully overwritten; absent skips reporting."""

    def __init__(
        out self,
        *,
        expansion_factor: Int = 0,
        traversal: Traversal = Traversal.WALK,
        predicate: PredicateArguments = PredicateArguments(),
        progress: ProgressArguments = ProgressArguments(),
        stats: Optional[Pointer[DenseTraversalStats, Self.stats_origin]] = Optional[
            Pointer[DenseTraversalStats, Self.stats_origin]
        ](),
    ):
        """Names only the knobs that differ from the derived ones.

        Args:
            expansion_factor: The breadth of the candidates the walk explores; zero derives.
            traversal: A walk, or an exact scan of every entry on RAM and NVMe handles.
            predicate: The handler and state judging which candidates are reported; the default admits all.
            progress: Hears the queries resolved so far; a cancel stops the search with no matches.
            stats: Receives this call's traversal telemetry; absent skips reporting.
        """
        self.expansion_factor = expansion_factor
        self.traversal = traversal
        self.predicate = predicate
        self.progress = progress
        self.stats = stats


struct DenseMaxsimConfig[stats_origin: MutOrigin = MutUntrackedOrigin](ImplicitlyCopyable):
    """The knobs of `usearch_dense_maxsim_t`; zeroed or absent ones derive what the C interface derives.

    Parameters:
        stats_origin: The passed telemetry slot's origin, inferred where one is passed.
    """

    var expansion_factor: Int
    """The breadth of the candidates; zero derives."""
    var progress: ProgressArguments
    """Hears the queries resolved so far, and may cancel the call."""
    var stats: Optional[Pointer[DenseTraversalStats, Self.stats_origin]]
    """Receives this call's traversal telemetry, fully overwritten; absent skips reporting."""

    def __init__(
        out self,
        *,
        expansion_factor: Int = 0,
        progress: ProgressArguments = ProgressArguments(),
        stats: Optional[Pointer[DenseTraversalStats, Self.stats_origin]] = Optional[
            Pointer[DenseTraversalStats, Self.stats_origin]
        ](),
    ):
        """Names only the knobs that differ from the derived ones.

        Args:
            expansion_factor: The breadth of the candidates; zero derives.
            progress: Hears the queries resolved so far, and may cancel the call.
            stats: Receives this call's traversal telemetry; absent skips reporting.
        """
        self.expansion_factor = expansion_factor
        self.progress = progress
        self.stats = stats


struct DenseMultiplyConfig[stats_origin: MutOrigin = MutUntrackedOrigin](ImplicitlyCopyable):
    """The knobs of `usearch_dense_multiply_t`; zeroed or absent ones derive what the C interface derives.

    Parameters:
        stats_origin: The passed telemetry slot's origin, inferred where one is passed.
    """

    var wanted: Int
    """Pairs kept per left row, the first ones its walk scored; zero keeps every pair."""
    var expansion_factor: Int
    """The breadth of the candidates, which widens the emitted pattern; zero derives."""
    var progress: ProgressArguments
    """Hears the queries resolved so far, and may cancel the call."""
    var stats: Optional[Pointer[DenseTraversalStats, Self.stats_origin]]
    """Receives this call's traversal telemetry, fully overwritten; absent skips reporting."""

    def __init__(
        out self,
        *,
        wanted: Int = 0,
        expansion_factor: Int = 0,
        progress: ProgressArguments = ProgressArguments(),
        stats: Optional[Pointer[DenseTraversalStats, Self.stats_origin]] = Optional[
            Pointer[DenseTraversalStats, Self.stats_origin]
        ](),
    ):
        """Names only the knobs that differ from the derived ones.

        Args:
            wanted: Pairs kept per left row, the first ones its walk scored; zero keeps every pair.
            expansion_factor: The breadth of the candidates, which widens the emitted pattern; zero derives.
            progress: Hears the queries resolved so far, and may cancel the call.
            stats: Receives this call's traversal telemetry; absent skips reporting.
        """
        self.wanted = wanted
        self.expansion_factor = expansion_factor
        self.progress = progress
        self.stats = stats


struct DenseAttendConfig[carry_origin: MutOrigin = MutUntrackedOrigin, stats_origin: MutOrigin = MutUntrackedOrigin](
    ImplicitlyCopyable
):
    """The knobs of `usearch_dense_attend_t`; zeroed or absent ones derive what the C interface derives.

    Parameters:
        carry_origin: The passed softmax states' origin, inferred where any are passed.
        stats_origin: The passed telemetry slot's origin, inferred where one is passed.
    """

    var head_count: Int
    """The query heads sharing the index's key-value head; it must divide the rows, and zero means one."""
    var carry: Span[DenseSoftmaxState, Self.carry_origin]
    """The softmax states, one per query row; empty skips them."""
    var expansion_factor: Int
    """The breadth of the candidates; zero derives. Wider candidates walk further, folding more of the tail."""
    var score_scale: Float32
    """The logit scale, conventionally the inverse square root of the head width; zero derives it."""
    var progress: ProgressArguments
    """Hears the queries resolved so far, and may cancel the call."""
    var stats: Optional[Pointer[DenseTraversalStats, Self.stats_origin]]
    """Receives this call's traversal telemetry, fully overwritten; absent skips reporting."""

    def __init__(
        out self,
        *,
        head_count: Int = 0,
        carry: Span[DenseSoftmaxState, Self.carry_origin] = Span[DenseSoftmaxState, Self.carry_origin](),
        expansion_factor: Int = 0,
        score_scale: Float32 = 0,
        progress: ProgressArguments = ProgressArguments(),
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
            progress: Hears the queries resolved so far, and may cancel the call.
            stats: Receives this call's traversal telemetry; absent skips reporting.
        """
        self.head_count = head_count
        self.carry = carry
        self.expansion_factor = expansion_factor
        self.score_scale = score_scale
        self.progress = progress
        self.stats = stats


struct DenseSampleConfig[
    shown_origin: ImmOrigin = ImmUntrackedOrigin,
    coordinates_origin: ImmOrigin = ImmUntrackedOrigin,
    focus_origin: ImmOrigin = ImmUntrackedOrigin,
](ImplicitlyCopyable):
    """The knobs of `usearch_dense_sample_t`; left alone, it draws across the whole index with nothing shown.

    Parameters:
        shown_origin: The shown keys' origin, inferred where any are passed.
        coordinates_origin: Their coordinates' origin, inferred with them.
        focus_origin: The focus keys' origin, inferred where any are passed.
    """

    var shown: Span[Key, Self.shown_origin]
    """The keys already shown, never drawn again; empty shows nothing yet."""
    var shown_coordinates: Span[Float32, Self.coordinates_origin]
    """Where each `shown` key sits, three values apiece."""
    var focus: Span[Key, Self.focus_origin]
    """The keys to look around; empty looks around the whole index."""
    var seed: UInt64
    """Equal seeds, shown keys and focus over an unchanged index draw equally."""
    var progress: ProgressArguments
    """Hears the draws written so far out of those wanted, and may cancel the call."""

    def __init__(
        out self,
        *,
        shown: Span[Key, Self.shown_origin] = Span[Key, Self.shown_origin](),
        shown_coordinates: Span[Float32, Self.coordinates_origin] = Span[Float32, Self.coordinates_origin](),
        focus: Span[Key, Self.focus_origin] = Span[Key, Self.focus_origin](),
        seed: UInt64 = 0,
        progress: ProgressArguments = ProgressArguments(),
    ):
        """Names only the knobs that differ from the defaults.

        Args:
            shown: The keys already shown, never drawn again; empty shows nothing yet.
            shown_coordinates: Where each `shown` key sits, three values apiece.
            focus: The keys to look around; empty looks around the whole index.
            seed: Equal seeds, shown keys and focus over an unchanged index draw equally.
            progress: Hears the draws written so far out of those wanted, and may cancel the call.
        """
        self.shown = shown
        self.shown_coordinates = shown_coordinates
        self.focus = focus
        self.seed = seed
        self.progress = progress


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

    var index: Pointer[Address, MutUntrackedOrigin]
    var outcome: Pointer[Outcome, MutUntrackedOrigin]
    var metric: DenseMetric
    var quantization: ScalarKind
    var dimensions: c_size_t
    var connectivity: c_size_t
    var values_per_key: c_size_t
    var ram: RamOptions[ImmUntrackedOrigin]


@fieldwise_init
struct DenseInitVramArguments(ImplicitlyCopyable):
    """`usearch_dense_init_vram_t`."""

    var index: Pointer[Address, MutUntrackedOrigin]
    var outcome: Pointer[Outcome, MutUntrackedOrigin]
    var metric: DenseMetric
    var quantization: ScalarKind
    var dimensions: c_size_t
    var connectivity: c_size_t
    var values_per_key: c_size_t
    var vram: VramOptions[ImmUntrackedOrigin]


@fieldwise_init
struct DenseLoadRamArguments(ImplicitlyCopyable):
    """`usearch_dense_load_ram_t`, and `usearch_dense_load_ram_buffer_t` with `source` a buffer."""

    var index: Pointer[Address, MutUntrackedOrigin]
    var executor: Address
    var outcome: Pointer[Outcome, MutUntrackedOrigin]
    var completion: Completion
    var source: BytesView[ImmUntrackedOrigin]
    var ram: RamOptions[ImmUntrackedOrigin]
    var connectivity: c_size_t
    var values_per_key: c_size_t
    var progress: ProgressArguments


@fieldwise_init
struct DenseLoadVramArguments(ImplicitlyCopyable):
    """`usearch_dense_load_vram_t`, and `usearch_dense_load_vram_buffer_t` with `source` a buffer."""

    var index: Pointer[Address, MutUntrackedOrigin]
    var executor: Address
    var outcome: Pointer[Outcome, MutUntrackedOrigin]
    var completion: Completion
    var source: BytesView[ImmUntrackedOrigin]
    var vram: VramOptions[ImmUntrackedOrigin]
    var values_per_key: c_size_t
    var progress: ProgressArguments


@fieldwise_init
struct DenseViewRamArguments(ImplicitlyCopyable):
    """`usearch_dense_view_ram_t`."""

    var index: Pointer[Address, MutUntrackedOrigin]
    var outcome: Pointer[Outcome, MutUntrackedOrigin]
    var path: BytesView[ImmUntrackedOrigin]


@fieldwise_init
struct DenseViewNvmeArguments(ImplicitlyCopyable):
    """`usearch_dense_view_nvme_t`."""

    var index: Pointer[Address, MutUntrackedOrigin]
    var executor: Address
    var outcome: Pointer[Outcome, MutUntrackedOrigin]
    var completion: Completion
    var path: BytesView[ImmUntrackedOrigin]
    var nvme: NvmeOptions[ImmUntrackedOrigin]
    var ram: Optional[Pointer[RamOptions[ImmUntrackedOrigin], ImmUntrackedOrigin]]


@fieldwise_init
struct DenseLoadNvmeArguments(ImplicitlyCopyable):
    """`usearch_dense_load_nvme_t`."""

    var index: Pointer[Address, MutUntrackedOrigin]
    var executor: Address
    var outcome: Pointer[Outcome, MutUntrackedOrigin]
    var completion: Completion
    var path: BytesView[ImmUntrackedOrigin]
    var nvme: NvmeOptions[ImmUntrackedOrigin]
    var ram: Optional[Pointer[RamOptions[ImmUntrackedOrigin], ImmUntrackedOrigin]]


@fieldwise_init
struct DenseReserveArguments(ImplicitlyCopyable):
    """`usearch_dense_reserve_t`."""

    var index: Address
    var executor: Address
    var outcome: Pointer[Outcome, MutUntrackedOrigin]
    var completion: Completion
    var capacity: c_size_t


@fieldwise_init
struct DenseClearArguments(ImplicitlyCopyable):
    """`usearch_dense_clear_t`."""

    var index: Address
    var executor: Address
    var outcome: Pointer[Outcome, MutUntrackedOrigin]
    var completion: Completion


@fieldwise_init
struct DenseDescribeArguments(ImplicitlyCopyable):
    """`usearch_dense_describe_t`."""

    var index: Address
    var executor: Address
    var outcome: Pointer[Outcome, MutUntrackedOrigin]
    var completion: Completion
    var description: Pointer[DenseIndexDescription, MutUntrackedOrigin]


@fieldwise_init
struct DenseInsertArguments(ImplicitlyCopyable):
    """`usearch_dense_insert_t`."""

    var index: Address
    var executor: Address
    var outcome: Pointer[Outcome, MutUntrackedOrigin]
    var completion: Completion
    var count: c_size_t
    var keys: KeysView[ImmUntrackedOrigin]
    var vectors: RowsViewArguments
    var expansion_factor: c_size_t
    var progress: ProgressArguments
    var stats: Optional[Pointer[DenseTraversalStats, MutUntrackedOrigin]]


@fieldwise_init
struct RefineOptionsArguments(ImplicitlyCopyable):
    """`usearch_refine_options_t`."""

    var expansion_factor: c_size_t
    var memory_limit: c_size_t


@fieldwise_init
struct DenseRefineArguments(ImplicitlyCopyable):
    """`usearch_dense_refine_t`."""

    var index: Address
    var executor: Address
    var outcome: Pointer[Outcome, MutUntrackedOrigin]
    var completion: Completion
    var report: Pointer[DenseRefineReport, MutUntrackedOrigin]
    var refine: RefineOptionsArguments
    var progress: ProgressArguments
    var stats: Optional[Pointer[DenseTraversalStats, MutUntrackedOrigin]]
    var plan: Optional[Pointer[DenseRefinePlan, MutUntrackedOrigin]]
    var phase: Optional[Pointer[PhaseProgress, MutUntrackedOrigin]]


@fieldwise_init
struct DenseRefinePlanFinishArguments(ImplicitlyCopyable):
    """`usearch_dense_refine_plan_finish_t`."""

    var index: Address
    var executor: Address
    var outcome: Pointer[Outcome, MutUntrackedOrigin]
    var completion: Completion
    var plan: Address
    var report: Pointer[DenseRefineReport, MutUntrackedOrigin]


@fieldwise_init
struct DenseHealthArguments(ImplicitlyCopyable):
    """`usearch_dense_health_t`."""

    var index: Address
    var executor: Address
    var outcome: Pointer[Outcome, MutUntrackedOrigin]
    var completion: Completion
    var reachability: Pointer[DenseReachability, MutUntrackedOrigin]


@fieldwise_init
struct DenseReshardArguments(ImplicitlyCopyable):
    """`usearch_dense_reshard_t`."""

    var sources_count: c_size_t
    var sources: Pointer[Address, ImmUntrackedOrigin]
    var targets_count: c_size_t
    var targets: Pointer[Address, ImmUntrackedOrigin]
    var executor: Address
    var outcome: Pointer[Outcome, MutUntrackedOrigin]
    var completion: Completion
    var ram: Optional[Pointer[RamOptions[ImmUntrackedOrigin], ImmUntrackedOrigin]]
    var vram: Optional[Pointer[VramOptions[ImmUntrackedOrigin], ImmUntrackedOrigin]]
    var nvme: Optional[Pointer[NvmeOptions[ImmUntrackedOrigin], ImmUntrackedOrigin]]
    var report: Pointer[DenseRefineReport, MutUntrackedOrigin]
    var refine: RefineOptionsArguments
    var seed: UInt64
    var progress: ProgressArguments
    var stats: Optional[Pointer[DenseTraversalStats, MutUntrackedOrigin]]
    var plan: Optional[Pointer[DenseReshardPlan, MutUntrackedOrigin]]
    var phase: Optional[Pointer[PhaseProgress, MutUntrackedOrigin]]


@fieldwise_init
struct DenseReshardPlanFinishArguments(ImplicitlyCopyable):
    """`usearch_dense_reshard_plan_finish_t`."""

    var executor: Address
    var outcome: Pointer[Outcome, MutUntrackedOrigin]
    var completion: Completion
    var plan: Address
    var report: Pointer[DenseRefineReport, MutUntrackedOrigin]


@fieldwise_init
struct DenseSampleArguments(ImplicitlyCopyable):
    """`usearch_dense_sample_t`."""

    var index: Address
    var executor: Address
    var outcome: Pointer[Outcome, MutUntrackedOrigin]
    var completion: Completion
    var shown_count: c_size_t
    var shown: KeysView[ImmUntrackedOrigin]
    var shown_coordinates: RowsViewArguments
    var focus_count: c_size_t
    var focus: KeysView[ImmUntrackedOrigin]
    var keys: Pointer[Address, MutUntrackedOrigin]
    var entry_ordinals: Pointer[Address, MutUntrackedOrigin]
    var parents: Pointer[MatchesViewArguments, MutUntrackedOrigin]
    var coordinates: Pointer[RowsViewArguments, MutUntrackedOrigin]
    var written: Pointer[c_size_t, MutUntrackedOrigin]
    var arena: Address
    var wanted: c_size_t
    var seed: UInt64
    var expansion_factor: c_size_t
    var progress: ProgressArguments
    var stats: Optional[Pointer[DenseTraversalStats, MutUntrackedOrigin]]


@fieldwise_init
struct DenseSearchArguments(ImplicitlyCopyable):
    """`usearch_dense_search_t`."""

    var index: Address
    var executor: Address
    var outcome: Pointer[Outcome, MutUntrackedOrigin]
    var completion: Completion
    var count: c_size_t
    var queries: RowsViewArguments
    var matches: Pointer[MatchesViewArguments, MutUntrackedOrigin]
    var arena: Address
    var wanted: c_size_t
    var expansion_factor: c_size_t
    var traversal: Traversal
    var predicate: PredicateArguments
    var hints: Optional[Pointer[MatchesViewArguments, ImmUntrackedOrigin]]
    var progress: ProgressArguments
    var stats: Optional[Pointer[DenseTraversalStats, MutUntrackedOrigin]]


@fieldwise_init
struct DenseCountArguments(ImplicitlyCopyable):
    """`usearch_dense_count_t`."""

    var index: Address
    var executor: Address
    var outcome: Pointer[Outcome, MutUntrackedOrigin]
    var completion: Completion
    var key: KeyBytes
    var entries: Pointer[c_size_t, MutUntrackedOrigin]


@fieldwise_init
struct DenseGetArguments(ImplicitlyCopyable):
    """`usearch_dense_get_t`."""

    var index: Address
    var executor: Address
    var outcome: Pointer[Outcome, MutUntrackedOrigin]
    var completion: Completion
    var key: KeyBytes
    var vectors: Pointer[RowsViewArguments, MutUntrackedOrigin]
    var written: Pointer[c_size_t, MutUntrackedOrigin]
    var arena: Address
    var outputs_scalar: ScalarKind


@fieldwise_init
struct DenseCursorInitArguments(ImplicitlyCopyable):
    """`usearch_dense_cursor_init_t`."""

    var index: Address
    var executor: Address
    var outcome: Pointer[Outcome, MutUntrackedOrigin]
    var starts_count: c_size_t
    var starts: Optional[Pointer[Key, ImmUntrackedOrigin]]
    var cursor: Pointer[Address, MutUntrackedOrigin]


@fieldwise_init
struct DenseCursorNextArguments(ImplicitlyCopyable):
    """`usearch_dense_cursor_next_t`."""

    var index: Address
    var executor: Address
    var outcome: Pointer[Outcome, MutUntrackedOrigin]
    var completion: Completion
    var cursor: Address
    var keys: Pointer[Address, MutUntrackedOrigin]
    var links: Optional[Pointer[KeysTapeViewArguments, MutUntrackedOrigin]]
    var written: Pointer[c_size_t, MutUntrackedOrigin]
    var arena: Address
    var wanted: c_size_t


@fieldwise_init
struct DenseSearchStreamInitArguments(ImplicitlyCopyable):
    """`usearch_dense_search_stream_init_t`."""

    var stream: Pointer[Address, MutUntrackedOrigin]
    var index: Address
    var outcome: Pointer[Outcome, MutUntrackedOrigin]
    var query: RowsViewArguments
    var expansion_factor: c_size_t


@fieldwise_init
struct DenseSearchStreamNextArguments(ImplicitlyCopyable):
    """`usearch_dense_search_stream_next_t`."""

    var index: Address
    var executor: Address
    var outcome: Pointer[Outcome, MutUntrackedOrigin]
    var completion: Completion
    var stream: Address
    var matches: Pointer[MatchesViewArguments, MutUntrackedOrigin]
    var arena: Address
    var wanted: c_size_t


@fieldwise_init
struct DenseSaveArguments(ImplicitlyCopyable):
    """`usearch_dense_save_t`."""

    var index: Address
    var executor: Address
    var outcome: Pointer[Outcome, MutUntrackedOrigin]
    var completion: Completion
    var path: BytesView[ImmUntrackedOrigin]
    var progress: ProgressArguments


@fieldwise_init
struct DenseSaveBufferArguments(ImplicitlyCopyable):
    """`usearch_dense_save_buffer_t`."""

    var index: Address
    var executor: Address
    var outcome: Pointer[Outcome, MutUntrackedOrigin]
    var completion: Completion
    var monolith: Pointer[DenseSaveBufferOutputs, MutUntrackedOrigin]
    var arena: Address
    var progress: ProgressArguments


@fieldwise_init
struct DenseMetadataReadArguments(ImplicitlyCopyable):
    """`usearch_dense_metadata_read_t`, and `usearch_dense_metadata_read_buffer_t` with `source` a
    buffer."""

    var executor: Address
    var outcome: Pointer[Outcome, MutUntrackedOrigin]
    var completion: Completion
    var source: BytesView[ImmUntrackedOrigin]
    var metadata: Pointer[DenseMetadata, MutUntrackedOrigin]


@fieldwise_init
struct DenseMultiplyArguments(ImplicitlyCopyable):
    """`usearch_dense_multiply_t`."""

    var index: Address
    var executor: Address
    var outcome: Pointer[Outcome, MutUntrackedOrigin]
    var completion: Completion
    var count: c_size_t
    var left: RowsViewArguments
    var products: Pointer[MatchesTapeViewArguments, MutUntrackedOrigin]
    var written: Pointer[c_size_t, MutUntrackedOrigin]
    var required: Pointer[c_size_t, MutUntrackedOrigin]
    var arena: Address
    var wanted: c_size_t
    var expansion_factor: c_size_t
    var traversal: Traversal
    var predicate: PredicateArguments
    var progress: ProgressArguments
    var stats: Optional[Pointer[DenseTraversalStats, MutUntrackedOrigin]]


@fieldwise_init
struct DenseAttendArguments(ImplicitlyCopyable):
    """`usearch_dense_attend_t`."""

    var index: Address
    var executor: Address
    var outcome: Pointer[Outcome, MutUntrackedOrigin]
    var completion: Completion
    var count: c_size_t
    var queries: RowsViewArguments
    var head_count: c_size_t
    var value_lookup: ValueLookup
    var value_lookup_state: Address
    var carry: Optional[Pointer[DenseSoftmaxState, MutUntrackedOrigin]]
    var outputs: Pointer[RowsViewArguments, MutUntrackedOrigin]
    var arena: Address
    var outputs_scalar: ScalarKind
    var expansion_factor: c_size_t
    var score_scale: Float32
    var traversal: Traversal
    var predicate: PredicateArguments
    var progress: ProgressArguments
    var stats: Optional[Pointer[DenseTraversalStats, MutUntrackedOrigin]]


@fieldwise_init
struct DenseMaxsimArguments(ImplicitlyCopyable):
    """`usearch_dense_maxsim_t`."""

    var index: Address
    var executor: Address
    var outcome: Pointer[Outcome, MutUntrackedOrigin]
    var completion: Completion
    var count: c_size_t
    var queries: RowsViewArguments
    var query_offsets: Optional[Pointer[c_size_t, ImmUntrackedOrigin]]
    var rows_per_query: c_size_t
    var matches: Pointer[MatchesTapeViewArguments, MutUntrackedOrigin]
    var written: Pointer[c_size_t, MutUntrackedOrigin]
    var required: Pointer[c_size_t, MutUntrackedOrigin]
    var arena: Address
    var wanted: c_size_t
    var expansion_factor: c_size_t
    var traversal: Traversal
    var predicate: PredicateArguments
    var hints: Optional[Pointer[MatchesViewArguments, ImmUntrackedOrigin]]
    var progress: ProgressArguments
    var stats: Optional[Pointer[DenseTraversalStats, MutUntrackedOrigin]]


comptime DenseInitRam = Verb[DenseInitRamArguments]
comptime DenseInitVram = Verb[DenseInitVramArguments]
comptime DenseLoadRam = Verb[DenseLoadRamArguments]
comptime DenseLoadRamBuffer = Verb[DenseLoadRamArguments]
comptime DenseLoadVram = Verb[DenseLoadVramArguments]
comptime DenseLoadVramBuffer = Verb[DenseLoadVramArguments]
comptime DenseViewRam = Verb[DenseViewRamArguments]
comptime DenseViewNvme = Verb[DenseViewNvmeArguments]
comptime DenseLoadNvme = Verb[DenseLoadNvmeArguments]
comptime DenseReserve = Verb[DenseReserveArguments]
comptime DenseClear = Verb[DenseClearArguments]
comptime DenseDescribe = Verb[DenseDescribeArguments]
comptime DenseInsert = Verb[DenseInsertArguments]
comptime DenseRefine = Verb[DenseRefineArguments]
comptime DenseRefinePlanFinish = Verb[DenseRefinePlanFinishArguments]
comptime DenseHealth = Verb[DenseHealthArguments]
comptime DenseReshard = Verb[DenseReshardArguments]
comptime DenseReshardPlanFinish = Verb[DenseReshardPlanFinishArguments]
comptime DenseSample = Verb[DenseSampleArguments]
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


@fieldwise_init
struct DenseCountOutputs(Defaultable, ImplicitlyCopyable):
    """The one `size_t` `usearch_dense_count` writes."""

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
struct DenseSampleOutputs(Defaultable, ImplicitlyCopyable):
    """What `usearch_dense_sample` writes through its pointers; the keys are a `usearch_keys_view_t`."""

    var keys: Address
    var keys_stride: c_size_t
    var entry_ordinals: Address
    var parents: MatchesViewArguments
    var coordinates: RowsViewArguments
    var written: c_size_t

    def __init__(out self):
        """Nothing drawn."""
        self.keys = NULL_ADDRESS
        self.keys_stride = c_size_t(0)
        self.entry_ordinals = NULL_ADDRESS
        self.parents = MatchesViewArguments()
        self.coordinates = _no_rows()
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
    """What `usearch_dense_multiply` and `usearch_dense_maxsim` write through their pointers."""

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


def _passed[T: Copyable](list: List[T]) -> Optional[Pointer[T, ImmUntrackedOrigin]]:
    """The first element a list keeps on the heap for a deferred call, or `None` for an empty one."""
    if len(list) == 0:
        return None
    return Pointer[T, ImmUntrackedOrigin](unsafe_from_address=Int(list.unsafe_ptr()))


@fieldwise_init
struct _DenseReserveCall(Call):
    """The crossing of `usearch_dense_reserve`."""

    comptime Arguments = DenseReserveArguments
    comptime Outputs = Unwritten
    comptime Result = NoneType
    comptime symbol: StaticString = "usearch_dense_reserve"
    var capacity: Int

    @staticmethod
    def verb(library: Library) -> Verb[Self.Arguments]:
        return library.symbols().dense_reserve

    def arguments(self, head: CallHead, outputs: Pointer[Self.Outputs, MutUntrackedOrigin]) -> Self.Arguments:
        return {head.index, head.executor, head.outcome, head.completion, c_size_t(self.capacity)}

    def conclude(deinit self, outputs: Unwritten, viewing: ArenaRead) -> NoneType:
        return None


@fieldwise_init
struct _DenseClearCall(Call):
    """The crossing of `usearch_dense_clear`."""

    comptime Arguments = DenseClearArguments
    comptime Outputs = Unwritten
    comptime Result = NoneType
    comptime symbol: StaticString = "usearch_dense_clear"

    @staticmethod
    def verb(library: Library) -> Verb[Self.Arguments]:
        return library.symbols().dense_clear

    def arguments(self, head: CallHead, outputs: Pointer[Self.Outputs, MutUntrackedOrigin]) -> Self.Arguments:
        return {head.index, head.executor, head.outcome, head.completion}

    def conclude(deinit self, outputs: Unwritten, viewing: ArenaRead) -> NoneType:
        return None


@fieldwise_init
struct _DenseDescribeCall(Call):
    """The crossing of `usearch_dense_describe`."""

    comptime Arguments = DenseDescribeArguments
    comptime Outputs = DenseIndexDescription
    comptime Result = DenseIndexDescription
    comptime symbol: StaticString = "usearch_dense_describe"

    @staticmethod
    def verb(library: Library) -> Verb[Self.Arguments]:
        return library.symbols().dense_describe

    def arguments(self, head: CallHead, outputs: Pointer[Self.Outputs, MutUntrackedOrigin]) -> Self.Arguments:
        return {head.index, head.executor, head.outcome, head.completion, outputs}

    def conclude(deinit self, outputs: DenseIndexDescription, viewing: ArenaRead) -> DenseIndexDescription:
        return outputs


@fieldwise_init
struct _DenseCountCall(Call):
    """The crossing of `usearch_dense_count`."""

    comptime Arguments = DenseCountArguments
    comptime Outputs = DenseCountOutputs
    comptime Result = Int
    comptime symbol: StaticString = "usearch_dense_count"
    var key: Key

    @staticmethod
    def verb(library: Library) -> Verb[Self.Arguments]:
        return library.symbols().dense_count

    def arguments(self, head: CallHead, outputs: Pointer[Self.Outputs, MutUntrackedOrigin]) -> Self.Arguments:
        return {
            head.index,
            head.executor,
            head.outcome,
            head.completion,
            KeyBytes.of(self.key),
            Pointer(to=outputs[].value),
        }

    def conclude(deinit self, outputs: DenseCountOutputs, viewing: ArenaRead) -> Int:
        return Int(outputs.value)


@fieldwise_init
struct _DenseGetCall[dtype: DType, kind: ScalarKind, origin: MutOrigin](Call):
    """The crossing of `usearch_dense_get`, which uses the arena its rows land in."""

    comptime Arguments = DenseGetArguments
    comptime Outputs = DenseGetOutputs
    comptime Result = DenseVectors[Self.dtype, Self.origin]
    comptime symbol: StaticString = "usearch_dense_get"
    var key: Key
    var arena: Address

    @staticmethod
    def verb(library: Library) -> Verb[Self.Arguments]:
        return library.symbols().dense_get

    def arguments(self, head: CallHead, outputs: Pointer[Self.Outputs, MutUntrackedOrigin]) -> Self.Arguments:
        return {
            head.index,
            head.executor,
            head.outcome,
            head.completion,
            KeyBytes.of(self.key),
            Pointer(to=outputs[].vectors),
            Pointer(to=outputs[].written),
            self.arena,
            Self.kind,
        }

    def arena_handle(self) -> Address:
        return self.arena

    def conclude(deinit self, outputs: DenseGetOutputs, viewing: ArenaRead) -> Self.Result:
        return {outputs.vectors, Int(outputs.written), viewing}


@fieldwise_init
struct _DenseInsertCall(Call):
    """The crossing of `usearch_dense_insert`."""

    comptime Arguments = DenseInsertArguments
    comptime Outputs = Unwritten
    comptime Result = NoneType
    comptime symbol: StaticString = "usearch_dense_insert"
    var keys: KeysView[ImmUntrackedOrigin]
    var keys_count: Int
    var vectors: RowsViewArguments
    var count: Int
    var expansion_factor: Int
    var progress: ProgressArguments
    var stats: Optional[Pointer[DenseTraversalStats, MutUntrackedOrigin]]

    @staticmethod
    def verb(library: Library) -> Verb[Self.Arguments]:
        return library.symbols().dense_insert

    def arguments(self, head: CallHead, outputs: Pointer[Self.Outputs, MutUntrackedOrigin]) -> Self.Arguments:
        return {
            head.index,
            head.executor,
            head.outcome,
            head.completion,
            c_size_t(self.count),
            self.keys,
            self.vectors,
            c_size_t(self.expansion_factor),
            self.progress,
            self.stats,
        }

    def lengths_agree(self) -> Bool:
        return self.keys_count == self.count

    def conclude(deinit self, outputs: Unwritten, viewing: ArenaRead) -> NoneType:
        return None


@fieldwise_init
struct _DenseRefineCall(Call):
    """The crossing of `usearch_dense_refine`."""

    comptime Arguments = DenseRefineArguments
    comptime Outputs = DenseRefineReport
    comptime Result = DenseRefineReport
    comptime symbol: StaticString = "usearch_dense_refine"
    var refine: RefineOptionsArguments
    var progress: ProgressArguments
    var stats: Optional[Pointer[DenseTraversalStats, MutUntrackedOrigin]]
    var plan: Optional[Pointer[DenseRefinePlan, MutUntrackedOrigin]]
    var phase: Optional[Pointer[PhaseProgress, MutUntrackedOrigin]]

    @staticmethod
    def verb(library: Library) -> Verb[Self.Arguments]:
        return library.symbols().dense_refine

    def arguments(self, head: CallHead, outputs: Pointer[Self.Outputs, MutUntrackedOrigin]) -> Self.Arguments:
        return {
            head.index,
            head.executor,
            head.outcome,
            head.completion,
            outputs,
            self.refine,
            self.progress,
            self.stats,
            self.plan,
            self.phase,
        }

    def conclude(deinit self, outputs: DenseRefineReport, viewing: ArenaRead) -> DenseRefineReport:
        return outputs


@fieldwise_init
struct _DenseRefinePlanFinishCall(Call):
    """The crossing of `usearch_dense_refine_plan_finish`, which frees the plan whatever it answers."""

    comptime Arguments = DenseRefinePlanFinishArguments
    comptime Outputs = DenseRefineReport
    comptime Result = DenseRefineReport
    comptime symbol: StaticString = "usearch_dense_refine_plan_finish"
    var plan: Address

    @staticmethod
    def verb(library: Library) -> Verb[Self.Arguments]:
        return library.symbols().dense_refine_plan_finish

    def arguments(self, head: CallHead, outputs: Pointer[Self.Outputs, MutUntrackedOrigin]) -> Self.Arguments:
        return {head.index, head.executor, head.outcome, head.completion, self.plan, outputs}

    def conclude(deinit self, outputs: DenseRefineReport, viewing: ArenaRead) -> DenseRefineReport:
        return outputs


@fieldwise_init
struct _DenseHealthCall(Call):
    """The crossing of `usearch_dense_health`."""

    comptime Arguments = DenseHealthArguments
    comptime Outputs = DenseReachability
    comptime Result = DenseReachability
    comptime symbol: StaticString = "usearch_dense_health"

    @staticmethod
    def verb(library: Library) -> Verb[Self.Arguments]:
        return library.symbols().dense_health

    def arguments(self, head: CallHead, outputs: Pointer[Self.Outputs, MutUntrackedOrigin]) -> Self.Arguments:
        return {head.index, head.executor, head.outcome, head.completion, outputs}

    def conclude(deinit self, outputs: DenseReachability, viewing: ArenaRead) -> DenseReachability:
        return outputs


struct _DenseReshardCall(Call):
    """The crossing of `usearch_dense_reshard`, which keeps the handle lists and each placement it passes
    on the heap until the call is final; an empty placement list passes none."""

    comptime Arguments = DenseReshardArguments
    comptime Outputs = DenseRefineReport
    comptime Result = DenseRefineReport
    comptime symbol: StaticString = "usearch_dense_reshard"
    var sources: List[Address]
    var targets: List[Address]
    var ram: List[RamOptions[ImmUntrackedOrigin]]
    var vram: List[VramOptions[ImmUntrackedOrigin]]
    var nvme: List[NvmeOptions[ImmUntrackedOrigin]]
    var refine: RefineOptionsArguments
    var seed: UInt64
    var progress: ProgressArguments
    var stats: Optional[Pointer[DenseTraversalStats, MutUntrackedOrigin]]
    var plan: Optional[Pointer[DenseReshardPlan, MutUntrackedOrigin]]
    var phase: Optional[Pointer[PhaseProgress, MutUntrackedOrigin]]

    def __init__[
        sources_origin: ImmOrigin,
        targets_origin: ImmOrigin,
        ram_origin: ImmOrigin,
        vram_origin: ImmOrigin,
        nvme_origin: ImmOrigin,
        stats_origin: MutOrigin,
        plan_origin: MutOrigin,
        phase_origin: MutOrigin,
        //,
    ](
        out self,
        sources: Span[DenseIndex, sources_origin],
        targets: Span[DenseIndex, targets_origin],
        config: DenseReshardConfig[ram_origin, vram_origin, nvme_origin, stats_origin, plan_origin, phase_origin],
    ):
        self.sources = [sources[position].address for position in range(len(sources))]
        self.targets = [targets[position].address for position in range(len(targets))]
        self.ram = List[RamOptions[ImmUntrackedOrigin]]()
        self.vram = List[VramOptions[ImmUntrackedOrigin]]()
        self.nvme = List[NvmeOptions[ImmUntrackedOrigin]]()
        if config.ram:
            self.ram.append(config.ram.value().arguments())
        if config.vram:
            self.vram.append(config.vram.value().arguments())
        if config.nvme:
            self.nvme.append(config.nvme.value().arguments())
        self.refine = RefineOptionsArguments(c_size_t(config.expansion_factor), c_size_t(config.memory_limit))
        self.seed = config.seed
        self.progress = config.progress
        self.stats = _slot(config.stats)
        self.plan = _slot(config.plan)
        self.phase = _slot(config.phase)

    @staticmethod
    def verb(library: Library) -> Verb[Self.Arguments]:
        return library.symbols().dense_reshard

    def arguments(self, head: CallHead, outputs: Pointer[Self.Outputs, MutUntrackedOrigin]) -> Self.Arguments:
        return {
            c_size_t(len(self.sources)),
            Pointer[Address, ImmUntrackedOrigin](unsafe_from_address=Int(self.sources.unsafe_ptr())),
            c_size_t(len(self.targets)),
            Pointer[Address, ImmUntrackedOrigin](unsafe_from_address=Int(self.targets.unsafe_ptr())),
            head.executor,
            head.outcome,
            head.completion,
            _passed(self.ram),
            _passed(self.vram),
            _passed(self.nvme),
            outputs,
            self.refine,
            self.seed,
            self.progress,
            self.stats,
            self.plan,
            self.phase,
        }

    def conclude(deinit self, outputs: DenseRefineReport, viewing: ArenaRead) -> DenseRefineReport:
        return outputs


@fieldwise_init
struct _DenseReshardPlanFinishCall(Call):
    """The crossing of `usearch_dense_reshard_plan_finish`, which frees the plan whatever it answers."""

    comptime Arguments = DenseReshardPlanFinishArguments
    comptime Outputs = DenseRefineReport
    comptime Result = DenseRefineReport
    comptime symbol: StaticString = "usearch_dense_reshard_plan_finish"
    var plan: Address

    @staticmethod
    def verb(library: Library) -> Verb[Self.Arguments]:
        return library.symbols().dense_reshard_plan_finish

    def arguments(self, head: CallHead, outputs: Pointer[Self.Outputs, MutUntrackedOrigin]) -> Self.Arguments:
        return {head.executor, head.outcome, head.completion, self.plan, outputs}

    def conclude(deinit self, outputs: DenseRefineReport, viewing: ArenaRead) -> DenseRefineReport:
        return outputs


@fieldwise_init
struct _DenseSampleCall[origin: MutOrigin](Call):
    """The crossing of `usearch_dense_sample`, which uses the arena its draws are written in."""

    comptime Arguments = DenseSampleArguments
    comptime Outputs = DenseSampleOutputs
    comptime Result = DenseDraws[Self.origin]
    comptime symbol: StaticString = "usearch_dense_sample"
    var shown: KeysView[ImmUntrackedOrigin]
    var shown_count: Int
    var shown_coordinates: RowsViewArguments
    var shown_coordinates_count: Int
    var focus: KeysView[ImmUntrackedOrigin]
    var focus_count: Int
    var wanted: Int
    var seed: UInt64
    var progress: ProgressArguments
    var arena: Address

    @staticmethod
    def verb(library: Library) -> Verb[Self.Arguments]:
        return library.symbols().dense_sample

    def arguments(self, head: CallHead, outputs: Pointer[Self.Outputs, MutUntrackedOrigin]) -> Self.Arguments:
        return {
            head.index,
            head.executor,
            head.outcome,
            head.completion,
            c_size_t(self.shown_count),
            self.shown,
            self.shown_coordinates,
            c_size_t(self.focus_count),
            self.focus,
            Pointer(to=outputs[].keys),
            Pointer(to=outputs[].entry_ordinals),
            Pointer(to=outputs[].parents),
            Pointer(to=outputs[].coordinates),
            Pointer(to=outputs[].written),
            self.arena,
            c_size_t(self.wanted),
            self.seed,
            c_size_t(0),
            self.progress,
            None,
        }

    def lengths_agree(self) -> Bool:
        return self.shown_coordinates_count == 3 * self.shown_count

    def arena_handle(self) -> Address:
        return self.arena

    def conclude(deinit self, outputs: DenseSampleOutputs, viewing: ArenaRead) -> Self.Result:
        return {outputs, viewing}


@fieldwise_init
struct _DenseSearchCall[origin: MutOrigin](Call):
    """The crossing of `usearch_dense_search`, which uses the arena its matches land in."""

    comptime Arguments = DenseSearchArguments
    comptime Outputs = MatchesViewArguments
    comptime Result = Matches[Self.origin]
    comptime symbol: StaticString = "usearch_dense_search"
    var queries: RowsViewArguments
    var count: Int
    var wanted: Int
    var arena: Address
    var expansion_factor: Int
    var traversal: Traversal
    var predicate: PredicateArguments
    var progress: ProgressArguments
    var stats: Optional[Pointer[DenseTraversalStats, MutUntrackedOrigin]]

    @staticmethod
    def verb(library: Library) -> Verb[Self.Arguments]:
        return library.symbols().dense_search

    def arguments(self, head: CallHead, outputs: Pointer[Self.Outputs, MutUntrackedOrigin]) -> Self.Arguments:
        return {
            head.index,
            head.executor,
            head.outcome,
            head.completion,
            c_size_t(self.count),
            self.queries,
            outputs,
            self.arena,
            c_size_t(self.wanted),
            c_size_t(self.expansion_factor),
            self.traversal,
            self.predicate,
            None,
            self.progress,
            self.stats,
        }

    def arena_handle(self) -> Address:
        return self.arena

    def conclude(deinit self, outputs: MatchesViewArguments, viewing: ArenaRead) -> Self.Result:
        return {outputs, self.count, viewing}


@fieldwise_init
struct _DenseMaxsimCall[origin: MutOrigin](Call):
    """The crossing of `usearch_dense_maxsim`, which uses the arena its matches land in."""

    comptime Arguments = DenseMaxsimArguments
    comptime Outputs = DenseMultiplyOutputs
    comptime Result = DenseProducts[Self.origin]
    comptime symbol: StaticString = "usearch_dense_maxsim"
    var queries: RowsViewArguments
    var rows: Int
    var rows_per_query: Int
    var wanted: Int
    var arena: Address
    var expansion_factor: Int
    var progress: ProgressArguments
    var stats: Optional[Pointer[DenseTraversalStats, MutUntrackedOrigin]]

    @staticmethod
    def verb(library: Library) -> Verb[Self.Arguments]:
        return library.symbols().dense_maxsim

    def arguments(self, head: CallHead, outputs: Pointer[Self.Outputs, MutUntrackedOrigin]) -> Self.Arguments:
        return {
            head.index,
            head.executor,
            head.outcome,
            head.completion,
            c_size_t(self.rows // self.rows_per_query),
            self.queries,
            None,
            c_size_t(self.rows_per_query),
            Pointer(to=outputs[].products),
            Pointer(to=outputs[].written),
            Pointer(to=outputs[].required),
            self.arena,
            c_size_t(self.wanted),
            c_size_t(self.expansion_factor),
            Traversal.WALK,
            PredicateArguments(),
            None,
            self.progress,
            self.stats,
        }

    def lengths_agree(self) -> Bool:
        return self.rows_per_query > 0 and self.rows % self.rows_per_query == 0

    def arena_handle(self) -> Address:
        return self.arena

    def conclude(deinit self, outputs: DenseMultiplyOutputs, viewing: ArenaRead) -> Self.Result:
        var queries = self.rows // self.rows_per_query if self.rows_per_query > 0 else 0
        return {outputs.products, queries, Int(outputs.written), Int(outputs.required), viewing}


@fieldwise_init
struct _DenseMultiplyCall[origin: MutOrigin](Call):
    """The crossing of `usearch_dense_multiply`, which uses the arena its pairs land in."""

    comptime Arguments = DenseMultiplyArguments
    comptime Outputs = DenseMultiplyOutputs
    comptime Result = DenseProducts[Self.origin]
    comptime symbol: StaticString = "usearch_dense_multiply"
    var left: RowsViewArguments
    var count: Int
    var arena: Address
    var wanted: Int
    var expansion_factor: Int
    var progress: ProgressArguments
    var stats: Optional[Pointer[DenseTraversalStats, MutUntrackedOrigin]]

    @staticmethod
    def verb(library: Library) -> Verb[Self.Arguments]:
        return library.symbols().dense_multiply

    def arguments(self, head: CallHead, outputs: Pointer[Self.Outputs, MutUntrackedOrigin]) -> Self.Arguments:
        return {
            head.index,
            head.executor,
            head.outcome,
            head.completion,
            c_size_t(self.count),
            self.left,
            Pointer(to=outputs[].products),
            Pointer(to=outputs[].written),
            Pointer(to=outputs[].required),
            self.arena,
            c_size_t(self.wanted),
            c_size_t(self.expansion_factor),
            Traversal.WALK,
            PredicateArguments(),
            self.progress,
            self.stats,
        }

    def arena_handle(self) -> Address:
        return self.arena

    def conclude(deinit self, outputs: DenseMultiplyOutputs, viewing: ArenaRead) -> Self.Result:
        return {outputs.products, self.count, Int(outputs.written), Int(outputs.required), viewing}


@fieldwise_init
struct _DenseAttendCall[dtype: DType, kind: ScalarKind, origin: MutOrigin](Call):
    """The crossing of `usearch_dense_attend`, which uses the arena its attended rows land in."""

    comptime Arguments = DenseAttendArguments
    comptime Outputs = DenseAttendOutputs
    comptime Result = DenseVectors[Self.dtype, Self.origin]
    comptime symbol: StaticString = "usearch_dense_attend"
    var queries: RowsViewArguments
    var count: Int
    var head_count: Int
    var value_lookup: ValueLookup
    var value_lookup_state: Address
    var carry: Optional[Pointer[DenseSoftmaxState, MutUntrackedOrigin]]
    var carry_count: Int
    var arena: Address
    var expansion_factor: Int
    var score_scale: Float32
    var progress: ProgressArguments
    var stats: Optional[Pointer[DenseTraversalStats, MutUntrackedOrigin]]

    @staticmethod
    def verb(library: Library) -> Verb[Self.Arguments]:
        return library.symbols().dense_attend

    def arguments(self, head: CallHead, outputs: Pointer[Self.Outputs, MutUntrackedOrigin]) -> Self.Arguments:
        return {
            head.index,
            head.executor,
            head.outcome,
            head.completion,
            c_size_t(self.count),
            self.queries,
            c_size_t(self.head_count),
            self.value_lookup,
            self.value_lookup_state,
            self.carry,
            Pointer(to=outputs[].outputs),
            self.arena,
            Self.kind,
            c_size_t(self.expansion_factor),
            self.score_scale,
            Traversal.WALK,
            PredicateArguments(),
            self.progress,
            self.stats,
        }

    def lengths_agree(self) -> Bool:
        # The carry crosses as a bare pointer, so only this side can see it fall short of the rows.
        return not self.carry or self.carry_count >= self.count

    def arena_handle(self) -> Address:
        return self.arena

    def conclude(deinit self, outputs: DenseAttendOutputs, viewing: ArenaRead) -> Self.Result:
        var written = outputs
        return {written.outputs, self.count, viewing}


@fieldwise_init
struct _DenseSaveCall(Call):
    """The crossing of `usearch_dense_save`."""

    comptime Arguments = DenseSaveArguments
    comptime Outputs = Unwritten
    comptime Result = NoneType
    comptime symbol: StaticString = "usearch_dense_save"
    var path: BytesView[ImmUntrackedOrigin]

    @staticmethod
    def verb(library: Library) -> Verb[Self.Arguments]:
        return library.symbols().dense_save

    def arguments(self, head: CallHead, outputs: Pointer[Self.Outputs, MutUntrackedOrigin]) -> Self.Arguments:
        return {head.index, head.executor, head.outcome, head.completion, self.path, ProgressArguments()}

    def conclude(deinit self, outputs: Unwritten, viewing: ArenaRead) -> NoneType:
        return None


@fieldwise_init
struct _DenseSaveBufferCall[origin: MutOrigin](Call):
    """The crossing of `usearch_dense_save_buffer`, which uses the arena the monolith lands in."""

    comptime Arguments = DenseSaveBufferArguments
    comptime Outputs = DenseSaveBufferOutputs
    comptime Result = Monolith[Self.origin]
    comptime symbol: StaticString = "usearch_dense_save_buffer"
    var arena: Address

    @staticmethod
    def verb(library: Library) -> Verb[Self.Arguments]:
        return library.symbols().dense_save_buffer

    def arguments(self, head: CallHead, outputs: Pointer[Self.Outputs, MutUntrackedOrigin]) -> Self.Arguments:
        return {head.index, head.executor, head.outcome, head.completion, outputs, self.arena, ProgressArguments()}

    def arena_handle(self) -> Address:
        return self.arena

    def conclude(deinit self, outputs: DenseSaveBufferOutputs, viewing: ArenaRead) -> Self.Result:
        return {outputs.data, Int(outputs.length), viewing}


@fieldwise_init
struct _DenseCursorNextCall[origin: MutOrigin](Call):
    """The crossing of `usearch_dense_cursor_next`, which uses the arena its batch lands in."""

    comptime Arguments = DenseCursorNextArguments
    comptime Outputs = DenseCursorNextOutputs
    comptime Result = DenseBatch[Self.origin]
    comptime symbol: StaticString = "usearch_dense_cursor_next"
    var cursor: Address
    var links: Bool
    var wanted: Int
    var arena: Address

    @staticmethod
    def verb(library: Library) -> Verb[Self.Arguments]:
        return library.symbols().dense_cursor_next

    def arguments(self, head: CallHead, outputs: Pointer[Self.Outputs, MutUntrackedOrigin]) -> Self.Arguments:
        return {
            head.index,
            head.executor,
            head.outcome,
            head.completion,
            self.cursor,
            Pointer(to=outputs[].keys),
            Optional(Pointer(to=outputs[].links)) if self.links else None,
            Pointer(to=outputs[].written),
            self.arena,
            c_size_t(self.wanted),
        }

    def arena_handle(self) -> Address:
        return self.arena

    def conclude(deinit self, outputs: DenseCursorNextOutputs, viewing: ArenaRead) -> Self.Result:
        return {outputs.keys, self.links, outputs.links, Int(outputs.written), viewing}


@fieldwise_init
struct _DenseSearchStreamNextCall[origin: MutOrigin](Call):
    """The crossing of `usearch_dense_search_stream_next`, which uses the arena its pull lands in."""

    comptime Arguments = DenseSearchStreamNextArguments
    comptime Outputs = MatchesViewArguments
    comptime Result = Matches[Self.origin]
    comptime symbol: StaticString = "usearch_dense_search_stream_next"
    var stream: Address
    var wanted: Int
    var arena: Address

    @staticmethod
    def verb(library: Library) -> Verb[Self.Arguments]:
        return library.symbols().dense_search_stream_next

    def arguments(self, head: CallHead, outputs: Pointer[Self.Outputs, MutUntrackedOrigin]) -> Self.Arguments:
        return {
            head.index,
            head.executor,
            head.outcome,
            head.completion,
            self.stream,
            outputs,
            self.arena,
            c_size_t(self.wanted),
        }

    def arena_handle(self) -> Address:
        return self.arena

    def conclude(deinit self, outputs: MatchesViewArguments, viewing: ArenaRead) -> Self.Result:
        return {outputs, 1, viewing}


# endregion Calls

# region Index


struct DenseIndex(DenseVerbs, Movable):
    """A dense vector index, released when it goes out of scope.

    Insertions and searches overlap freely on one handle: a search never writes to the graph, and
    an insertion publishes an entry only once it is reachable, so a concurrent walk sees a consistent
    graph at every instant. One call holds an executor at a time.

    Verbs called on the index run on the calling thread; `index.on(executor)` runs the same verbs
    blocking on an executor, and `index.deferred(executor)` submits them there. The cursor and the
    stream's opening name no executor, and their pulls run on the receiver's.
    """

    var library: Library
    """The loaded engine, held so its symbol table outlives every call made through the handle."""
    var address: Address
    """The opaque `usearch_dense_index_t`."""

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
            ram_origin: The options' viewed node-id array, inferred from the argument.

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
                created.unsafe_origin_cast[MutUntrackedOrigin](),
                outcome.unsafe_origin_cast[MutUntrackedOrigin](),
                options.metric,
                options.quantization,
                options.dimensions,
                options.connectivity,
                options.values_per_key,
                ram.arguments(),
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
            vram_origin: The options' viewed ordinal array, inferred from the argument.

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
                created.unsafe_origin_cast[MutUntrackedOrigin](),
                outcome.unsafe_origin_cast[MutUntrackedOrigin](),
                options.metric,
                options.quantization,
                options.dimensions,
                options.connectivity,
                options.values_per_key,
                vram.arguments(),
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
            ram_origin: The options' viewed node-id array, inferred from the argument.

        Args:
            executor: The executor the read fans out over, one shard per thread; it also supplies its queues.
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
            BytesView.of(path).arguments(),
            ram.arguments(),
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
            ram_origin: The options' viewed node-id array, inferred from the argument.

        Args:
            executor: The executor the copy fans out over; it also supplies its queues.
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
            BytesView.of(buffer).arguments(),
            ram.arguments(),
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
                created.unsafe_origin_cast[MutUntrackedOrigin](),
                executor.address,
                outcome.unsafe_origin_cast[MutUntrackedOrigin](),
                Completion.BLOCKING,
                source,
                ram,
                c_size_t(0),
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
            vram_origin: The options' viewed ordinal array, inferred from the argument.

        Args:
            executor: The executor the upload runs on, one stream per device; it also supplies its queues.
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
            BytesView.of(path).arguments(),
            vram.arguments(),
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
            vram_origin: The options' viewed ordinal array, inferred from the argument.

        Args:
            executor: The executor the upload runs on, one stream per device; it also supplies its queues.
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
            BytesView.of(buffer).arguments(),
            vram.arguments(),
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
                created.unsafe_origin_cast[MutUntrackedOrigin](),
                executor.address,
                outcome.unsafe_origin_cast[MutUntrackedOrigin](),
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
            DenseViewRamArguments(
                created.unsafe_origin_cast[MutUntrackedOrigin](),
                outcome.unsafe_origin_cast[MutUntrackedOrigin](),
                BytesView.of(path).arguments(),
            ),
            library.symbols().dense_view_ram,
        )
        library._check(outcome.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0], "usearch_dense_view_ram")
        return Self(library=library, address=created.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0])

    @staticmethod
    def view_nvme[
        path_origin: ImmOrigin, nvme_origin: ImmOrigin, key_table_origin: ImmOrigin, //
    ](
        executor: Executor,
        path: StringSlice[path_origin],
        nvme: NvmeOptions[nvme_origin] = NvmeOptions(),
        key_table: Optional[RamOptions[key_table_origin]] = Optional[RamOptions[ImmUntrackedOrigin]](),
    ) raises Error -> Self:
        """Serves a checkpoint in place from NVMe drives, read through the executor's rings.
        Calls `usearch_dense_view_nvme`.

        Parameters:
            path_origin: The path bytes' origin, inferred from the argument.
            nvme_origin: The options' viewed path array, inferred from the argument.
            key_table_origin: The key table's viewed domain array, inferred from the argument.

        Args:
            executor: The NVMe executor whose rings read the checkpoint's headers; it also supplies its queues.
            path: The manifest of a sharded checkpoint, or a single `.safegraphs` monolith.
            nvme: The devices the parts are served from; naming none serves each where it points.
            key_table: The memory domains of a key table held in RAM, which `count`, `get`, `sample`,
                MaxSim and hinted searches read; `None` holds none.

        Returns:
            The read-only index.

        Raises:
            `Error` with `IO` if the checkpoint cannot be opened.
        """
        var created = stack_allocation[1, Address]()
        created[unsafe_offset=0] = NULL_ADDRESS
        var outcome = stack_allocation[1, Outcome]()
        outcome[unsafe_offset=0] = Outcome()
        _cross(
            DenseViewNvmeArguments(
                created.unsafe_origin_cast[MutUntrackedOrigin](),
                executor.address,
                outcome.unsafe_origin_cast[MutUntrackedOrigin](),
                Completion.BLOCKING,
                BytesView.of(path).arguments(),
                nvme.arguments(),
                Optional(
                    Pointer(to=key_table.value())
                    .unsafe_bitcast[RamOptions[ImmUntrackedOrigin]]()
                    .unsafe_origin_cast[ImmUntrackedOrigin]()
                ) if key_table else None,
            ),
            executor.library.symbols().dense_view_nvme,
        )
        executor.library._check(outcome.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0], "usearch_dense_view_nvme")
        return Self(library=executor.library, address=created.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0])

    @staticmethod
    def load_nvme[
        path_origin: ImmOrigin, nvme_origin: ImmOrigin, key_table_origin: ImmOrigin, //
    ](
        executor: Executor,
        path: StringSlice[path_origin],
        nvme: NvmeOptions[nvme_origin] = NvmeOptions(),
        key_table: Optional[RamOptions[key_table_origin]] = Optional[RamOptions[ImmUntrackedOrigin]](),
    ) raises Error -> Self:
        """Opens a checkpoint's manifest from NVMe drives as a writable index: it answers every read
        a view answers, and `refine` rewrites its parts one at a time. Calls `usearch_dense_load_nvme`.

        Parameters:
            path_origin: The path bytes' origin, inferred from the argument.
            nvme_origin: The options' viewed path array, inferred from the argument.
            key_table_origin: The key table's viewed domain array, inferred from the argument.

        Args:
            executor: The NVMe executor whose rings read the checkpoint's headers; it also supplies its queues.
            path: The manifest of a sharded checkpoint.
            nvme: The class to dispatch as; its paths must name none.
            key_table: The memory domains of a key table held in RAM, as `view_nvme` takes them.

        Returns:
            The writable index.

        Raises:
            `Error` with `UNSUPPORTED` for a lone monolith, which has no manifest to replace, or for
            paths naming any.
        """
        var created = stack_allocation[1, Address]()
        created[unsafe_offset=0] = NULL_ADDRESS
        var outcome = stack_allocation[1, Outcome]()
        outcome[unsafe_offset=0] = Outcome()
        _cross(
            DenseLoadNvmeArguments(
                created.unsafe_origin_cast[MutUntrackedOrigin](),
                executor.address,
                outcome.unsafe_origin_cast[MutUntrackedOrigin](),
                Completion.BLOCKING,
                BytesView.of(path).arguments(),
                nvme.arguments(),
                Optional(
                    Pointer(to=key_table.value())
                    .unsafe_bitcast[RamOptions[ImmUntrackedOrigin]]()
                    .unsafe_origin_cast[ImmUntrackedOrigin]()
                ) if key_table else None,
            ),
            executor.library.symbols().dense_load_nvme,
        )
        executor.library._check(outcome.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0], "usearch_dense_load_nvme")
        return Self(library=executor.library, address=created.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0])

    def __deinit__(deinit self):
        """Releases the index and everything it owns. Calls `usearch_dense_free`.

        The loader reference is surrendered only after the call: reading the entry point is the
        last use of `library`, so without the trailing move Mojo drops the last reference - and
        `dlclose`s the engine - between the load and the branch to it.
        """
        self.library.symbols().dense_free(self.address)
        _ = self.library^

    # endregion Lifetime

    # region Receivers

    def on(ref self, ref executor: Executor) -> DenseIndexOn[ImmOrigin(origin_of(self, executor))]:
        """The same verbs, blocking on `executor`, refused while another call holds it. Binding-only.

        Args:
            executor: The executor every verb runs on; its kind must match the handle's residency.

        Returns:
            A receiver viewing both, which neither may outlive.
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
            A receiver viewing both, which neither it nor any `Deferred` it answers may outlive.
        """
        return DenseIndexDeferredOn[ImmOrigin(origin_of(self, executor))](
            Pointer[DenseIndex, ImmOrigin(origin_of(self, executor))](unsafe_from_address=Int(Pointer(to=self))),
            executor.address,
        )

    def _route[C: Call](self, var call: C) raises Error -> C.Result:
        """Runs `call` on the calling thread."""
        return _block(self.library, call^, self.address, NULL_ADDRESS)

    def cursor(ref self) raises Error -> DenseCursor[ImmOrigin(origin_of(self))]:
        """Opens a walk over every entry as stored, its pulls on the calling thread.
        Calls `usearch_dense_cursor_init`.

        Each shard's publication watermark is read as the walk enters that shard, so an entry once
        passed is never revisited.

        Returns:
            The cursor, which uses the index and cannot outlive it.

        Raises:
            `Error` if the residency has no cursor, as VRAM does not.
        """
        return DenseCursor[ImmOrigin(origin_of(self))](self.library, self.address, 0, None)

    def cursor[
        starts_origin: ImmOrigin, //
    ](ref self, starts: Span[Key, starts_origin]) raises Error -> DenseCursor[ImmOrigin(origin_of(self))]:
        """Opens a walk from `starts`, its pulls on the calling thread. Calls `usearch_dense_cursor_init`.

        It yields the start keys' own entries first, in order, then the entries those link to
        breadth-first, each once.

        Parameters:
            starts_origin: The start keys' origin, inferred from the argument.

        Args:
            starts: The keys to walk from.

        Returns:
            The cursor, which uses the index and cannot outlive it.

        Raises:
            `Error` if the residency has no cursor, as VRAM does not.
        """
        return DenseCursor[ImmOrigin(origin_of(self))](
            self.library,
            self.address,
            len(starts),
            starts.unsafe_ptr().unsafe_origin_cast[ImmUntrackedOrigin](),
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
            A stream yielding matches lazily, which uses the index and the query.

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

        A batch is all-or-nothing, because an insertion's progress handler reports but never
        cancels: every fallible step runs before the first entry identity is claimed, so a rejected
        batch leaves the index untouched and publishes no entry.

        Parameters:
            keys_origin: The key run's origin, inferred from the argument.
            vectors_origin: The row storage's origin, inferred with it.
            stats_origin: The passed telemetry slot's origin, inferred with it.

        Args:
            keys: One key per row of `vectors`; keys need not be unique.
            vectors: The rows to insert, converted on ingest to the stored type.
            config: The breadth of the candidates and the telemetry slot.

        Raises:
            `Error` if the shapes disagree or the handle is read-only.
        """
        _ = self._route(_insert_call(keys, vectors, config))

    def refine[
        stats_origin: MutOrigin, plan_origin: MutOrigin, phase_origin: MutOrigin, //
    ](
        self, config: DenseRefineConfig[stats_origin, plan_origin, phase_origin] = DenseRefineConfig()
    ) raises Error -> DenseRefineReport:
        """Runs one round over the whole graph: heals every island a search cannot reach, gathers
        every entry into a neighborhood of near entries, and re-stitches each one. Calls
        `usearch_dense_refine`.

        A build step rather than a repair: a cheap insertion pass followed by rounds reaches a better
        graph, in less total time, than a single expensive insertion pass. A caller repeats rounds
        until the verdict reads converged. Searches may run throughout on RAM; insertions may not.

        Parameters:
            stats_origin: The passed telemetry slot's origin, inferred from the argument.
            plan_origin: The passed plan's origin, inferred with it.
            phase_origin: The passed phase slot's origin, inferred with it.

        Args:
            config: The round's spending, the handler, the telemetry slot, and the plan and phase slots.

        Returns:
            What the round found and changed, its verdict, and its opening and closing health checks.

        Raises:
            `Error` with `READ_ONLY` if the handle serves its checkpoint in place, or `CANCELLED` once
            the handler stops the round.
        """
        return self._route(_refine_call(config))

    def refine_plan_finish(self, var plan: DenseRefinePlan) raises Error -> DenseRefineReport:
        """Closes and frees a refine plan whatever the call answers, reopening the index.
        Calls `usearch_dense_refine_plan_finish`.

        Args:
            plan: The plan a refine on this index and executor opened, consumed by the call.

        Returns:
            What the whole plan did to the graph.

        Raises:
            `Error` with `INVALID_ARGUMENT` for a plan no refine opened, or one opened on another index
            or executor.
        """
        return self._route(_DenseRefinePlanFinishCall(plan.address))

    def health(self) raises Error -> DenseReachability:
        """Measures how much of the graph a search reaches from the root. Calls `usearch_dense_health`.

        Read-only, so concurrent insertions merely leave fresh entries uncounted.

        Returns:
            The audit; entries it misses stay unfound until a `refine` re-links them.

        Raises:
            `Error` if the engine refuses the audit.
        """
        return self._route(_DenseHealthCall())

    def sample[
        shown_origin: ImmOrigin, coordinates_origin: ImmOrigin, focus_origin: ImmOrigin, //
    ](
        self,
        wanted: Int,
        mut arena: Arena,
        config: DenseSampleConfig[shown_origin, coordinates_origin, focus_origin] = DenseSampleConfig(),
    ) raises Error -> DenseDraws[origin_of(arena)]:
        """Draws up to `wanted` keys around `config.focus`, or across the whole index, each hanging
        from a shown or earlier draw and placed in three dimensions beside the shown ones. Calls
        `usearch_dense_sample`.

        A larger `wanted` draws a superset, so a view grows one connected sample as it zooms. Only
        RAM samples yet.

        Parameters:
            shown_origin: The shown keys' origin, inferred from the config.
            coordinates_origin: Their coordinates' origin, inferred with it.
            focus_origin: The focus keys' origin, inferred with it.

        Args:
            wanted: The upper bound on draws.
            arena: The arena to rewind and refill, which the result views; a failed call leaves it valid.
            config: The shown keys and their coordinates, the focus and the seed.

        Returns:
            The draws, read in place from the arena.

        Raises:
            `Error` with `SHAPE_MISMATCH` unless there are three coordinates per shown key, or
            `UNSUPPORTED` off RAM.
        """
        return self._route(_sample_call[origin_of(arena)](wanted, arena._address, config))

    def search[
        queries_origin: ImmOrigin, stats_origin: MutOrigin, //
    ](
        self,
        queries: RowsView[queries_origin],
        wanted: Int,
        mut arena: Arena,
        config: DenseSearchConfig[stats_origin] = DenseSearchConfig(),
    ) raises Error -> Matches[origin_of(arena)]:
        """Performs k-approximate nearest-neighbor search for a batch of queries, writing the matches
        in a passed arena. Calls `usearch_dense_search`.

        Never writes to the graph, so any number of these may overlap with concurrent insertions.

        Parameters:
            queries_origin: The query rows' origin, inferred from the argument.
            stats_origin: The passed telemetry slot's origin, inferred with it.

        Args:
            queries: The query rows, one per search.
            wanted: Matches per query - the "k"; zero lands only the counts.
            arena: The arena to rewind and refill, which the result views; a failed call leaves it valid.
            config: The breadth of the candidates, the predicate, and the telemetry slot.

        Returns:
            Every query's matches, read in place from the arena.

        Raises:
            `Error` if the shapes disagree, or a predicate is passed to a VRAM handle.
        """
        return self._route(_search_call[origin_of(arena)](queries, wanted, arena._address, config))

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
        dtype: DType, kind: ScalarKind = ScalarKind.of[dtype]()
    ](self, key: Key, mut arena: Arena) raises Error -> DenseVectors[dtype, origin_of(arena)]:
        """Exports every vector stored under one key into a passed arena, one row per entry, in
        insertion order. Calls `usearch_dense_get`.

        Parameters:
            dtype: The element type the rows are read as.
            kind: The scalar kind to export as, converted from storage where it differs, which `dtype`
                must store whole; the kind `dtype` stores by default. `DType` names no e4m3, e5m2,
                u1x8 or i4, so those kinds travel here, read as the bytes `dtype` holds.

        Args:
            key: The key whose entries are exported.
            arena: The arena to rewind and refill, which the result views; a failed call leaves it valid.

        Returns:
            The rows, read in place from the arena; none when the key is absent.

        Raises:
            `Error` if the engine refuses the export.
        """
        comptime assert kind.stores[dtype](), "the element type must store the kind whole"
        return self._route(_DenseGetCall[dtype, kind, origin_of(arena)](key, arena._address))

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
        _ = self._route(_DenseSaveCall(BytesView.of(path).arguments()))

    def save_buffer(self, mut arena: Arena) raises Error -> Monolith[origin_of(arena)]:
        """Serializes the index as one monolith into a passed arena - the same bytes the path-taking
        sibling writes. Calls `usearch_dense_save_buffer`.

        Args:
            arena: The arena to rewind and refill, which the result views; a failed call leaves it valid.

        Returns:
            The monolith, read in place from the arena.

        Raises:
            `Error` if the engine refuses the write.
        """
        return self._route(_DenseSaveBufferCall[origin_of(arena)](arena._address))

    def multiply[
        left_origin: ImmOrigin, stats_origin: MutOrigin, //
    ](
        self,
        left: RowsView[left_origin],
        mut arena: Arena,
        config: DenseMultiplyConfig[stats_origin] = DenseMultiplyConfig(),
    ) raises Error -> DenseProducts[origin_of(arena)]:
        """Approximate sparse product of a run of rows against the indexed vectors, into a passed arena.
        Calls `usearch_dense_multiply`.

        Reports every pair the traversal scored instead of only the nearest, in traversal order
        rather than sorted, and with no duplicates. Cost is roughly
        `rows x expansion x connectivity x dimensions`, sublinear in the indexed entries, which is the
        whole point of reading a product off a graph. Runs on `DenseMetric.INNER` alone.

        Parameters:
            left_origin: The left-hand rows' origin, inferred from the argument.
            stats_origin: The passed telemetry slot's origin, inferred with it.

        Args:
            left: The left-hand rows.
            arena: The arena to rewind and refill, which the result views; a failed call leaves it valid.
            config: The pairs kept per row, the breadth of the candidates, and the telemetry slot.

        Returns:
            The ragged pairs and their offsets, read in place from the arena.

        Raises:
            `Error` with `CONFIG_MISMATCH` if the graph was frozen under a distance kind.
        """
        return self._route(_multiply_call[origin_of(arena)](left, arena._address, config))

    def attend[
        queries_origin: ImmOrigin,
        carry_origin: MutOrigin,
        stats_origin: MutOrigin,
        //,
        dtype: DType,
        kind: ScalarKind = ScalarKind.of[dtype](),
    ](
        self,
        queries: RowsView[queries_origin],
        value_lookup: ValueLookup,
        value_lookup_state: Address,
        mut arena: Arena,
        config: DenseAttendConfig[carry_origin, stats_origin] = DenseAttendConfig(),
    ) raises Error -> DenseVectors[dtype, origin_of(arena)]:
        """Softmax attention of a run of query heads over every indexed key their graph walks score,
        written in a passed arena. Calls `usearch_dense_attend`.

        Every key a walk scores folds into each head's running maximum, partition, and weighted
        value sum, as Flash Attention folds key tiles, with no top-k cut: the long tail the walk
        scores is what brings the result toward exact dense attention. The outputs come back
        already divided by the partition.

        Row `t · head_count + h` of `queries`, the outputs, and `carry` is head `h` of token `t`. One
        index holds one layer's one key-value head, so the `head_count` query heads of grouped-query
        attention share its keys and one walk per token.

        The index stores no values: `value_lookup` resolves the keys of every scored batch to their
        value rows, so only keys a walk scores are ever resolved. `carry` is the softmax's `m` and
        `l`, one `DenseSoftmaxState` per query row, passed by the caller: supplying it folds this branch
        into another over the same queries, such as the recent window kept out of the index, by
        partial-softmax rescaling.

        Parameters:
            queries_origin: The query rows' origin, inferred from the argument.
            carry_origin: The softmax states' origin, inferred with it where any are supplied.
            stats_origin: The passed telemetry slot's origin, inferred with it.
            dtype: The element type the attended rows are read as.
            kind: The scalar kind they land as, which `dtype` must store whole; the kind `dtype` stores
                by default.

        Args:
            queries: The query rows, `head_count` per token.
            value_lookup: Resolves every scored batch of keys to their value rows - a host function on
                RAM, and a device one on VRAM.
            value_lookup_state: The opaque state forwarded into every `value_lookup` call.
            arena: The arena to rewind and refill, which the result views; a failed call leaves it valid.
            config: The heads per token, the carry, the breadth of the candidates, the logit scale,
                and the telemetry slot.

        Returns:
            One attended row per query row, as wide as the keys, read in place from the arena.

        Raises:
            `Error` with `SHAPE_MISMATCH` if `head_count` does not divide the rows, the carry holds
            fewer states than there are rows, or a lookup yields a null row; with `CONFIG_MISMATCH` if
            the graph was frozen under a distance kind.
        """
        comptime assert kind.stores[dtype](), "the element type must store the kind whole"
        return self._route(
            _attend_call[dtype, kind, origin_of(arena)](
                queries, value_lookup, value_lookup_state, arena._address, config
            )
        )

    def maxsim[
        queries_origin: ImmOrigin, stats_origin: MutOrigin, //
    ](
        self,
        queries: RowsView[queries_origin],
        rows_per_query: Int,
        wanted: Int,
        mut arena: Arena,
        config: DenseMaxsimConfig[stats_origin] = DenseMaxsimConfig(),
    ) raises Error -> DenseProducts[origin_of(arena)]:
        """Late-interaction scoring: every query row takes its best match within a key, and those
        maxima sum to that key's score, written in a passed arena. Calls `usearch_dense_maxsim`.

        This is what `values_per_key` above one exists for. A document is one key holding its token
        rows, and the walk stays over individual tokens under a plain inner product. Approximate in
        the usual direction: a key can score below its exact value but never above it.

        Parameters:
            queries_origin: The query rows' origin, inferred from the argument.
            stats_origin: The passed telemetry slot's origin, inferred with it.

        Args:
            queries: The token rows of every query, laid end to end.
            rows_per_query: Rows in every query; uniform, so ColBERT's padding applies as is.
            wanted: Keys per query - the "k" over documents rather than rows; zero keeps every key
                scored.
            arena: The arena to rewind and refill, which the result views; a failed call leaves it valid.
            config: The breadth of the candidates and the telemetry slot.

        Returns:
            Every query's keys, summed maxima largest first, read in place from the arena.

        Raises:
            `Error` with `SHAPE_MISMATCH` if `rows_per_query` does not divide the rows; with
            `CONFIG_MISMATCH` if the graph was frozen under a distance kind.
        """
        return self._route(_maxsim_call[origin_of(arena)](queries, rows_per_query, wanted, arena._address, config))


struct DenseIndexOn[origin: ImmOrigin](DenseVerbs, ImplicitlyCopyable):
    """An index bound to the executor its verbs block on, as `index.on(executor)` answers it. Binding-only.

    Parameters:
        origin: The index and the executor, which this receiver views.
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
        """Runs `call` on the bound executor, refused while another call holds it."""
        return _block(self.index[].library, call^, self.index[].address, self.executor)

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
            A stream yielding matches lazily, which uses the index, the executor and the query.

        Raises:
            `Error` with `UNSUPPORTED` while no residency implements the stream.
        """
        return DenseSearchStream[ImmOrigin(origin_of(_anchor[Self.origin](), _anchor[query_origin]()))](
            self.index[].library, self.index[].address, self.executor, query.arguments(), config.expansion_factor
        )


def _refine_call[
    stats_origin: MutOrigin, plan_origin: MutOrigin, phase_origin: MutOrigin
](config: DenseRefineConfig[stats_origin, plan_origin, phase_origin]) -> _DenseRefineCall:
    """Lowers a refinement's passed slots into its crossing."""
    return _DenseRefineCall(
        RefineOptionsArguments(c_size_t(config.expansion_factor), c_size_t(config.memory_limit)),
        config.progress,
        _slot(config.stats),
        _slot(config.plan),
        _slot(config.phase),
    )


def _insert_call[
    keys_origin: ImmOrigin, vectors_origin: ImmOrigin, stats_origin: MutOrigin
](
    keys: Span[Key, keys_origin], vectors: RowsView[vectors_origin], config: DenseInsertConfig[stats_origin]
) -> _DenseInsertCall:
    """Lowers an insertion's viewed inputs into its crossing."""
    return _DenseInsertCall(
        KeysView.of(keys).arguments(),
        len(keys),
        vectors.arguments(),
        Int(vectors.rows),
        config.expansion_factor,
        config.progress,
        _slot(config.stats),
    )


def _sample_call[
    origin: MutOrigin, shown_origin: ImmOrigin, coordinates_origin: ImmOrigin, focus_origin: ImmOrigin
](
    wanted: Int,
    arena: Address,
    config: DenseSampleConfig[shown_origin, coordinates_origin, focus_origin],
) -> _DenseSampleCall[origin]:
    """Lowers a sample's viewed inputs into its crossing."""
    return _DenseSampleCall[origin](
        KeysView.of(config.shown).arguments(),
        len(config.shown),
        RowsViewArguments(
            Int(config.shown_coordinates.unsafe_ptr()), c_size_t(3 * size_of[Float32]()), c_size_t(3), ScalarKind.F32
        ),
        len(config.shown_coordinates),
        KeysView.of(config.focus).arguments(),
        len(config.focus),
        wanted,
        config.seed,
        config.progress,
        arena,
    )


def _search_call[
    origin: MutOrigin, queries_origin: ImmOrigin, stats_origin: MutOrigin
](
    queries: RowsView[queries_origin],
    wanted: Int,
    arena: Address,
    config: DenseSearchConfig[stats_origin],
) -> _DenseSearchCall[origin]:
    """Lowers a search's viewed inputs into its crossing."""
    return _DenseSearchCall[origin](
        queries.arguments(),
        Int(queries.rows),
        wanted,
        arena,
        config.expansion_factor,
        config.traversal,
        config.predicate,
        config.progress,
        _slot(config.stats),
    )


def _maxsim_call[
    origin: MutOrigin, queries_origin: ImmOrigin, stats_origin: MutOrigin
](
    queries: RowsView[queries_origin],
    rows_per_query: Int,
    wanted: Int,
    arena: Address,
    config: DenseMaxsimConfig[stats_origin],
) -> _DenseMaxsimCall[origin]:
    """Lowers a late-interaction call's viewed inputs into its crossing."""
    return _DenseMaxsimCall[origin](
        queries.arguments(),
        Int(queries.rows),
        rows_per_query,
        wanted,
        arena,
        config.expansion_factor,
        config.progress,
        _slot(config.stats),
    )


def _multiply_call[
    origin: MutOrigin, left_origin: ImmOrigin, stats_origin: MutOrigin
](left: RowsView[left_origin], arena: Address, config: DenseMultiplyConfig[stats_origin]) -> _DenseMultiplyCall[origin]:
    """Lowers a product's viewed inputs into its crossing."""
    return _DenseMultiplyCall[origin](
        left.arguments(),
        Int(left.rows),
        arena,
        config.wanted,
        config.expansion_factor,
        config.progress,
        _slot(config.stats),
    )


def _attend_call[
    dtype: DType,
    kind: ScalarKind,
    origin: MutOrigin,
    queries_origin: ImmOrigin,
    carry_origin: MutOrigin,
    stats_origin: MutOrigin,
](
    queries: RowsView[queries_origin],
    value_lookup: ValueLookup,
    value_lookup_state: Address,
    arena: Address,
    config: DenseAttendConfig[carry_origin, stats_origin],
) -> _DenseAttendCall[dtype, kind, origin]:
    """Lowers an attention call's viewed inputs into its crossing."""
    return _DenseAttendCall[dtype, kind, origin](
        queries.arguments(),
        Int(queries.rows),
        config.head_count,
        value_lookup,
        value_lookup_state,
        Optional(config.carry.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()) if len(config.carry) else None,
        len(config.carry),
        arena,
        config.expansion_factor,
        config.score_scale,
        config.progress,
        _slot(config.stats),
    )


struct DenseIndexDeferredOn[origin: ImmOrigin](ImplicitlyCopyable):
    """An index bound to the executor its verbs are submitted to, as `index.deferred(executor)` answers
    it: every verb answers a `Deferred` at once. Binding-only.

    The verbs and their arguments are those of `DenseVerbs`, the arena viewed as theirs is; only the
    blocking receivers add the cursor and the search stream, whose every pull blocks.

    Parameters:
        origin: The index and the executor, which this receiver views.
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
        """Submits `DenseVerbs.insert`; the keys, rows and telemetry slot must stay allocated and unchanged until final.
        Calls `usearch_dense_insert`."""
        return {self.index[].library, _insert_call(keys, vectors, config), self.index[].address, self.executor}

    def refine[
        stats_origin: MutOrigin, plan_origin: MutOrigin, phase_origin: MutOrigin, //
    ](self, config: DenseRefineConfig[stats_origin, plan_origin, phase_origin] = DenseRefineConfig()) -> Deferred[
        _DenseRefineCall,
        ImmOrigin(
            origin_of(_anchor[Self.origin](), _anchor[stats_origin](), _anchor[plan_origin](), _anchor[phase_origin]())
        ),
    ]:
        """Submits `DenseVerbs.refine`; the telemetry, plan and phase slots must stay allocated and unchanged until final.
        Calls `usearch_dense_refine`."""
        return {self.index[].library, _refine_call(config), self.index[].address, self.executor}

    def refine_plan_finish(self, var plan: DenseRefinePlan) -> Deferred[_DenseRefinePlanFinishCall, Self.origin]:
        """Submits `DenseVerbs.refine_plan_finish`. Calls `usearch_dense_refine_plan_finish`."""
        return {self.index[].library, _DenseRefinePlanFinishCall(plan.address), self.index[].address, self.executor}

    def health(self) -> Deferred[_DenseHealthCall, Self.origin]:
        """Submits `DenseVerbs.health`. Calls `usearch_dense_health`."""
        return {self.index[].library, _DenseHealthCall(), self.index[].address, self.executor}

    def sample[
        shown_origin: ImmOrigin, coordinates_origin: ImmOrigin, focus_origin: ImmOrigin, //
    ](
        self,
        wanted: Int,
        mut arena: Arena,
        config: DenseSampleConfig[shown_origin, coordinates_origin, focus_origin] = DenseSampleConfig(),
    ) -> Deferred[
        _DenseSampleCall[origin_of(arena)],
        ImmOrigin(
            origin_of(
                _anchor[Self.origin](),
                _anchor[shown_origin](),
                _anchor[coordinates_origin](),
                _anchor[focus_origin](),
            )
        ),
    ]:
        """Submits `DenseVerbs.sample`; the shown keys, focus and arena must stay allocated and unchanged until final.
        Calls `usearch_dense_sample`."""
        return {
            self.index[].library,
            _sample_call[origin_of(arena)](wanted, arena._address, config),
            self.index[].address,
            self.executor,
        }

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
        """Submits `DenseVerbs.search`; the queries, telemetry slot and arena must stay allocated and unchanged until final.
        Calls `usearch_dense_search`."""
        return {
            self.index[].library,
            _search_call[origin_of(arena)](queries, wanted, arena._address, config),
            self.index[].address,
            self.executor,
        }

    def count(self, key: Key) -> Deferred[_DenseCountCall, Self.origin]:
        """Submits `DenseVerbs.count`. Calls `usearch_dense_count`."""
        return {self.index[].library, _DenseCountCall(key), self.index[].address, self.executor}

    def get[
        dtype: DType, kind: ScalarKind = ScalarKind.of[dtype]()
    ](self, key: Key, mut arena: Arena) -> Deferred[_DenseGetCall[dtype, kind, origin_of(arena)], Self.origin]:
        """Submits `DenseVerbs.get`; the arena must stay allocated and unchanged until final. Calls `usearch_dense_get`.
        """
        comptime assert kind.stores[dtype](), "the element type must store the kind whole"
        return {
            self.index[].library,
            _DenseGetCall[dtype, kind, origin_of(arena)](key, arena._address),
            self.index[].address,
            self.executor,
        }

    def save[
        path_origin: ImmOrigin, //
    ](self, path: StringSlice[path_origin]) -> Deferred[
        _DenseSaveCall, ImmOrigin(origin_of(_anchor[Self.origin](), _anchor[path_origin]()))
    ]:
        """Submits `DenseVerbs.save`; the path must stay allocated and unchanged until final. Calls `usearch_dense_save`.
        """
        return {
            self.index[].library,
            _DenseSaveCall(BytesView.of(path).arguments()),
            self.index[].address,
            self.executor,
        }

    def save_buffer(self, mut arena: Arena) -> Deferred[_DenseSaveBufferCall[origin_of(arena)], Self.origin]:
        """Submits `DenseVerbs.save_buffer`; the arena must stay allocated and unchanged until final. Calls `usearch_dense_save_buffer`.
        """
        return {
            self.index[].library,
            _DenseSaveBufferCall[origin_of(arena)](arena._address),
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
        """Submits `DenseVerbs.multiply`; the rows, telemetry slot and arena must stay allocated and unchanged until final.
        Calls `usearch_dense_multiply`."""
        return {
            self.index[].library,
            _multiply_call[origin_of(arena)](left, arena._address, config),
            self.index[].address,
            self.executor,
        }

    def attend[
        queries_origin: ImmOrigin,
        carry_origin: MutOrigin,
        stats_origin: MutOrigin,
        //,
        dtype: DType,
        kind: ScalarKind = ScalarKind.of[dtype](),
    ](
        self,
        queries: RowsView[queries_origin],
        value_lookup: ValueLookup,
        value_lookup_state: Address,
        mut arena: Arena,
        config: DenseAttendConfig[carry_origin, stats_origin] = DenseAttendConfig(),
    ) -> Deferred[
        _DenseAttendCall[dtype, kind, origin_of(arena)],
        ImmOrigin(
            origin_of(
                _anchor[Self.origin](), _anchor[queries_origin](), _anchor[carry_origin](), _anchor[stats_origin]()
            )
        ),
    ]:
        """Submits `DenseVerbs.attend`; the queries, carry, telemetry slot and arena must stay allocated and unchanged until
        final, and the lookup state must outlive the call too. Calls `usearch_dense_attend`."""
        comptime assert kind.stores[dtype](), "the element type must store the kind whole"
        return {
            self.index[].library,
            _attend_call[dtype, kind, origin_of(arena)](
                queries, value_lookup, value_lookup_state, arena._address, config
            ),
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
        """Submits `DenseVerbs.maxsim`; the queries, telemetry slot and arena must stay allocated and unchanged until final.
        Calls `usearch_dense_maxsim`."""
        return {
            self.index[].library,
            _maxsim_call[origin_of(arena)](queries, rows_per_query, wanted, arena._address, config),
            self.index[].address,
            self.executor,
        }


struct DenseCursor[origin: ImmOrigin]:
    """A walk over the index contents - `usearch_dense_cursor_t`.

    Export, migration, and visualization feed off it. It uses the index, so the index cannot be
    released while it lives, and it keeps no executor: its pulls run on the calling thread, or
    block on the executor `on` names for each of them.

    Parameters:
        origin: The index, which the cursor views.
    """

    var library: Library
    """The loaded engine, held so the release can still reach `usearch_dense_cursor_free`."""
    var index: Address
    """The `usearch_dense_index_t` the cursor was opened on, which every pull names again."""
    var address: Address
    """The `usearch_dense_cursor_t` this struct owns."""

    def __init__(
        out self,
        library: Library,
        index: Address,
        starts_count: Int,
        starts: Optional[Pointer[Key, ImmUntrackedOrigin]],
    ) raises Error:
        """Opens a cursor over `index`. Calls `usearch_dense_cursor_init`.

        Args:
            library: The loaded engine the cursor is opened through.
            index: The handle to scan.
            starts_count: The start keys at `starts`; zero walks every entry as stored.
            starts: The first start key, or `None` for none.

        Raises:
            `Error` if the residency has no cursor, as VRAM does not.
        """
        var created = stack_allocation[1, Address]()
        created[unsafe_offset=0] = NULL_ADDRESS
        var outcome = stack_allocation[1, Outcome]()
        outcome[unsafe_offset=0] = Outcome()
        _cross(
            DenseCursorInitArguments(
                index,
                NULL_ADDRESS,
                outcome.unsafe_origin_cast[MutUntrackedOrigin](),
                c_size_t(starts_count),
                starts,
                created.unsafe_origin_cast[MutUntrackedOrigin](),
            ),
            library.symbols().dense_cursor_init,
        )
        library._check(outcome.unsafe_origin_cast[MutAnyOrigin]()[unsafe_offset=0], "usearch_dense_cursor_init")
        self.library = library
        self.index = index
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
        """Copies up to `wanted` next keys into a passed arena, skipping their links. Calls `usearch_dense_cursor_next`.

        Args:
            wanted: The upper bound on keys this pull writes.
            arena: The arena to rewind and refill, which the batch views; a failed call leaves it valid.

        Returns:
            The keys, read in place from the arena; an empty batch once the walk is exhausted.

        Raises:
            `Error` if the engine refuses the copy.
        """
        return self._next(
            _DenseCursorNextCall[origin_of(arena)](self.address, False, wanted, arena._address), NULL_ADDRESS
        )

    def next_links(self, wanted: Int, mut arena: Arena) raises Error -> DenseBatch[origin_of(arena)]:
        """Copies up to `wanted` next keys into a passed arena, each with the keys it links to.
        Calls `usearch_dense_cursor_next`.

        Args:
            wanted: The upper bound on entries this pull writes.
            arena: The arena to rewind and refill, which the batch views; a failed call leaves it valid.

        Returns:
            The keys and their links, read in place from the arena; an empty batch once the walk is exhausted.

        Raises:
            `Error` if the engine refuses the copy.
        """
        return self._next(
            _DenseCursorNextCall[origin_of(arena)](self.address, True, wanted, arena._address), NULL_ADDRESS
        )

    def on(ref self, ref executor: Executor) -> DenseCursorOn[Self.origin, ImmOrigin(origin_of(self, executor))]:
        """The receiver whose pulls block on `executor`, named again for every pull. Binding-only.

        Args:
            executor: The executor every pull runs on; its kind must match the index's residency.

        Returns:
            A receiver viewing the cursor and the executor, which neither may outlive.
        """
        return DenseCursorOn[Self.origin, ImmOrigin(origin_of(self, executor))](
            Pointer[DenseCursor[Self.origin], ImmOrigin(origin_of(self, executor))](
                unsafe_from_address=Int(Pointer(to=self))
            ),
            executor.address,
        )

    def _next[C: Call](self, var call: C, executor: Address) raises Error -> C.Result:
        """One blocking pull on `executor`, or on the calling thread for `NULL_ADDRESS`."""
        return _block(self.library, call^, self.index, executor)


struct DenseCursorOn[cursor_origin: ImmOrigin, origin: ImmOrigin]:
    """A cursor's pulls blocking on one executor, as `DenseCursor.on` makes them. Binding-only.

    Parameters:
        cursor_origin: The index the cursor views.
        origin: The cursor and the executor, which this receiver views.
    """

    var cursor: Pointer[DenseCursor[Self.cursor_origin], Self.origin]
    """The cursor every pull advances."""
    var executor: Address
    """The `usearch_executor_t` every pull runs on."""

    def __init__(out self, cursor: Pointer[DenseCursor[Self.cursor_origin], Self.origin], executor: Address):
        """Binds a cursor to an executor.

        Args:
            cursor: The cursor every pull advances.
            executor: The `usearch_executor_t` every pull runs on.
        """
        self.cursor = cursor
        self.executor = executor

    def next_keys(self, wanted: Int, mut arena: Arena) raises Error -> DenseBatch[origin_of(arena)]:
        """Copies up to `wanted` next keys into a passed arena on the bound executor, skipping their
        links. Calls `usearch_dense_cursor_next`.

        Args:
            wanted: The upper bound on keys this pull writes.
            arena: The arena to rewind and refill, which the batch views; a failed call leaves it valid.

        Returns:
            The keys, read in place from the arena; an empty batch once the walk is exhausted.

        Raises:
            `Error` if the engine refuses the copy.
        """
        return self.cursor[]._next(
            _DenseCursorNextCall[origin_of(arena)](self.cursor[].address, False, wanted, arena._address), self.executor
        )

    def next_links(self, wanted: Int, mut arena: Arena) raises Error -> DenseBatch[origin_of(arena)]:
        """Copies up to `wanted` next keys into a passed arena on the bound executor, each with the
        keys it links to. Calls `usearch_dense_cursor_next`.

        Args:
            wanted: The upper bound on entries this pull writes.
            arena: The arena to rewind and refill, which the batch views; a failed call leaves it valid.

        Returns:
            The keys and their links, read in place from the arena; an empty batch once the walk is exhausted.

        Raises:
            `Error` if the engine refuses the copy.
        """
        return self.cursor[]._next(
            _DenseCursorNextCall[origin_of(arena)](self.cursor[].address, True, wanted, arena._address), self.executor
        )


struct DenseSearchStream[origin: ImmOrigin]:
    """A resumable nearest-neighbor stream - `usearch_dense_search_stream_t`.

    Best-first traversal is naturally pausable, so the stream keeps the walk's frontier and visited
    set between pulls. Callers that post-filter aggressively pull until satisfied instead of
    guessing "k" upfront. Ordering beyond the expansion horizon is best-effort nearest rather than
    globally exact.

    Parameters:
        origin: The index, the executor, and the query row, which the stream views.
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
            DenseSearchStreamInitArguments(
                created.unsafe_origin_cast[MutUntrackedOrigin](),
                index,
                outcome.unsafe_origin_cast[MutUntrackedOrigin](),
                query,
                c_size_t(expansion_factor),
            ),
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
        """Pulls up to `wanted` next-nearest matches into a passed arena, as the one query of a search.
        Calls `usearch_dense_search_stream_next`.

        Args:
            wanted: The upper bound on matches this pull writes.
            arena: The arena to rewind and refill, which the pull views; a failed call leaves it valid.

        Returns:
            One query's matches, read in place from the arena; an empty query once the walk is drained.

        Raises:
            `Error` if the engine refuses to advance.
        """
        return _block(
            self.library,
            _DenseSearchStreamNextCall[origin_of(arena)](self.address, wanted, arena._address),
            self.index,
            self.executor,
        )


# endregion Index

# region Reshard


def dense_reshard[
    sources_origin: ImmOrigin,
    targets_origin: ImmOrigin,
    ram_origin: ImmOrigin,
    vram_origin: ImmOrigin,
    nvme_origin: ImmOrigin,
    stats_origin: MutOrigin,
    plan_origin: MutOrigin,
    phase_origin: MutOrigin,
    //,
](
    executor: Executor,
    sources: Span[DenseIndex, sources_origin],
    targets: Span[DenseIndex, targets_origin],
    config: DenseReshardConfig[
        ram_origin, vram_origin, nvme_origin, stats_origin, plan_origin, phase_origin
    ] = DenseReshardConfig(),
) raises Error -> DenseRefineReport:
    """Moves the entries of `sources` into `targets`, keeping their links rather than re-inserting their
    vectors, then runs a round on every target that took entries. Calls `usearch_dense_reshard`.

    Several sources into one target among them gather into it; one source into several empty targets
    cuts it into one graph-local part apiece; one index as its only source and target recuts its own
    shards under `config.ram`. Every index stays the caller's, and a source the targets do not name
    ends empty and live.

    Parameters:
        sources_origin: The sources' origin, inferred from the argument.
        targets_origin: The targets' origin, inferred from the argument.
        ram_origin: The RAM placement's viewed domain array, inferred from the config.
        vram_origin: The VRAM placement's viewed ordinal array, inferred with it.
        nvme_origin: The NVMe placement's viewed path array, inferred with it.
        stats_origin: The passed telemetry slot's origin, inferred with it.
        plan_origin: The passed plan's origin, inferred with it.
        phase_origin: The passed phase slot's origin, inferred with it.

    Args:
        executor: The executor every move and the closing rounds fan out over; it also supplies its queues.
        sources: The indexes whose entries move, each named once.
        targets: The indexes the entries move into, each named once, a source itself or empty.
        config: The placement of a recut, the rounds' spending, the seed of a cut, the handler, and the
            telemetry, plan and phase slots.

    Returns:
        What the closing rounds did to the graphs.

    Raises:
        `Error` with `INVALID_ARGUMENT` for an index named twice, `CONFIG_MISMATCH` for indexes of
        different residencies or geometries or a target neither a source nor empty, `UNSUPPORTED` for
        several sources into several targets, or `READ_ONLY` for an NVMe view.
    """
    return _block(executor.library, _DenseReshardCall(sources, targets, config), NULL_ADDRESS, executor.address)


def dense_reshard_deferred[
    sources_origin: ImmOrigin,
    targets_origin: ImmOrigin,
    ram_origin: ImmOrigin,
    vram_origin: ImmOrigin,
    nvme_origin: ImmOrigin,
    stats_origin: MutOrigin,
    plan_origin: MutOrigin,
    phase_origin: MutOrigin,
    //,
](
    ref executor: Executor,
    sources: Span[DenseIndex, sources_origin],
    targets: Span[DenseIndex, targets_origin],
    config: DenseReshardConfig[
        ram_origin, vram_origin, nvme_origin, stats_origin, plan_origin, phase_origin
    ] = DenseReshardConfig(),
) -> Deferred[
    _DenseReshardCall,
    ImmOrigin(
        origin_of(
            executor,
            _anchor[sources_origin](),
            _anchor[targets_origin](),
            _anchor[ram_origin](),
            _anchor[vram_origin](),
            _anchor[nvme_origin](),
            _anchor[stats_origin](),
            _anchor[plan_origin](),
            _anchor[phase_origin](),
        )
    ),
]:
    """Submits `dense_reshard` to `executor`, answering a `Deferred` at once; the indexes, placements
    and slots must stay allocated and unchanged until final. Calls `usearch_dense_reshard`."""
    return {executor.library, _DenseReshardCall(sources, targets, config), NULL_ADDRESS, executor.address}


def dense_reshard_plan_finish(executor: Executor, var plan: DenseReshardPlan) raises Error -> DenseRefineReport:
    """Closes and frees a reshard plan whatever the call answers, reopening every index it names.
    Calls `usearch_dense_reshard_plan_finish`.

    Args:
        executor: The executor the plan was opened with; it also supplies its queues.
        plan: The plan a reshard opened, consumed by the call.

    Returns:
        What the whole plan did to the graphs; zeros while entries were still to move.

    Raises:
        `Error` with `INVALID_ARGUMENT` for a plan no reshard opened, or one opened with another executor.
    """
    return _block(executor.library, _DenseReshardPlanFinishCall(plan.address), NULL_ADDRESS, executor.address)


def dense_reshard_plan_finish_deferred(
    ref executor: Executor, var plan: DenseReshardPlan
) -> Deferred[_DenseReshardPlanFinishCall, ImmOrigin(origin_of(executor))]:
    """Submits `dense_reshard_plan_finish` to `executor`, answering a `Deferred` at once.
    Calls `usearch_dense_reshard_plan_finish`."""
    return {executor.library, _DenseReshardPlanFinishCall(plan.address), NULL_ADDRESS, executor.address}


# endregion Reshard

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
        DenseMetadataReadArguments(
            executor.address,
            outcome.unsafe_origin_cast[MutUntrackedOrigin](),
            Completion.BLOCKING,
            source,
            described.unsafe_origin_cast[MutUntrackedOrigin](),
        ),
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
        executor: The executor the read runs on; it also supplies its queues.
        path: The manifest of a sharded checkpoint, or a single monolith.

    Returns:
        The geometry and metric the checkpoint froze.

    Raises:
        `Error` if the checkpoint cannot be opened or its header cannot be decoded.
    """
    return _read_metadata(
        executor,
        executor.library.symbols().dense_metadata_read,
        BytesView.of(path).arguments(),
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
        executor: The executor the read runs on; it also supplies its queues.
        buffer: Source holding one serialized monolith.

    Returns:
        The geometry and metric the monolith froze.

    Raises:
        `Error` if the bytes cannot be decoded.
    """
    return _read_metadata(
        executor,
        executor.library.symbols().dense_metadata_read_buffer,
        BytesView.of(buffer).arguments(),
        "usearch_dense_metadata_read_buffer",
    )


# endregion Metadata
