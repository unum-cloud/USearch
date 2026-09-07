"""Vector search over equidimensional embeddings: the index, its cursor, and its resumable finder.

Mirrors @c include/usearch/dense_index.h. The type is `DenseIndex` rather than `Index` because the
engine has sibling modalities that will bind beside it, and a bare name would have to be renamed
the day one of them lands.

Every batch is a matrix. `MatrixView` and `MatrixSpan` carry the base pointer, the row count, the
column count, the byte stride, and the scalar kind as one value, so a single vector is a one-row
matrix and a caller never packs a special case. Both structs are laid out exactly as
`usearch_dense_matrix_view_t` and `usearch_dense_matrix_span_t` declare them, and cross by address
as the verbs take them - transposing two adjacent counts is a compile error rather than a silent
corruption. `MatchesSpan` is the same idea for the three outputs a ranked run writes.

No callback of this binding is written in Mojo, and the two slots that take one - the admission
predicate of `search_if` and the custom metric of `DenseOptions` - cross as raw `Address` values so
that a kernel built in C can still be installed. Two Mojo 1.0 facts force that. A closure capturing
anything cannot be handed to C at all, which is the familiar one; and a `def ... abi("C")` that
returns a value does not in fact lower to the C calling convention, so the engine calling one
jumps into a corrupted frame. Both of these entry points return a value - an admission and a
distance - so neither can be authored here. Progress handlers are not exposed for the same reason,
and every `progress` and `progress_state` slot crosses as `NULL_ADDRESS`.

Telemetry is an argument rather than an option. Every verb that reports traversal statistics takes
`mut stats`, so the engine always writes through a real slot and the binding never has to represent
a null one.
"""

from std.ffi import c_int, c_size_t
from std.memory import stack_allocation
from std.sys.info import size_of

from usearch.library import Library
from usearch.types import (
    Address,
    BytesSpan,
    BytesView,
    Distance,
    Error,
    ErrorSlot,
    Executor,
    Handle,
    Key,
    KeysSpan,
    KeysView,
    NULL_ADDRESS,
    Residency,
    ScalarKind,
    Status,
    text_at,
)

# region Signatures

comptime ErrorReport = Pointer[ErrorSlot, MutAnyOrigin]
"""The `usearch_error_t *` every entry point takes last."""

comptime ReadData = Pointer[NoneType, ImmUntrackedOrigin]
"""A `void const *` run of rows the engine reads."""

comptime WriteData = Pointer[NoneType, MutAnyOrigin]
"""A `void *` run of rows the engine fills."""

comptime StoredWriteData = Pointer[NoneType, MutUntrackedOrigin]
"""The same, as a struct may hold it: Mojo 1.0 refuses `AnyOrigin` in a field, so a stored pointer
carries an untracked origin and is widened at the call boundary."""

comptime StoredKeys = Pointer[Key, MutUntrackedOrigin]
"""A stored `usearch_key_t *` output run, widened at the call boundary."""

comptime StoredScores = Pointer[Distance, MutUntrackedOrigin]
"""A stored `usearch_distance_t *` output run, widened at the call boundary."""

comptime StoredCounts = Pointer[c_size_t, MutUntrackedOrigin]
"""A stored `size_t *` output run, widened at the call boundary."""

comptime ScoresOut = Pointer[Distance, MutAnyOrigin]
"""A `usearch_distance_t *` output run."""

comptime MatrixViewIn = Pointer[MatrixView[ImmUntrackedOrigin], MutAnyOrigin]
"""A `usearch_dense_matrix_view_t const *` run of rows the engine reads.

Every descriptor pointer below carries `MutAnyOrigin` even though C only reads through it: the
binding fills a stack slot and hands its address over, and a narrower origin lets the compiler
fold that store away, which leaves the engine reading whatever the frame held before. Every
descriptor here is the borrow-erased instantiation, since one C function pointer type cannot be
parametric; what keeps the caller's rows alive is the origin its own descriptor carries, which
lives for the duration of the binding method that erases it."""

comptime MatrixSpanIn = Pointer[MatrixSpan[MutUntrackedOrigin], MutAnyOrigin]
"""A `usearch_dense_matrix_span_t const *` run of rows the engine fills."""

comptime MatchesIn = Pointer[MatchesSpan[MutUntrackedOrigin, MutUntrackedOrigin, MutUntrackedOrigin], MutAnyOrigin]
"""A `usearch_dense_matches_span_t const *` triple of ranked outputs the engine fills."""

comptime OptionsIn = Pointer[DenseOptions[ImmUntrackedOrigin], MutAnyOrigin]
"""A `usearch_dense_options_t const *` the engine reads."""

comptime OptionsOut = Pointer[DenseOptions[ImmUntrackedOrigin], MutAnyOrigin]
"""A `usearch_dense_options_t *` a metadata probe fills."""

comptime ExpansionIn = Pointer[ExpansionConfig, MutAnyOrigin]
"""A `usearch_dense_expansion_config_t const *` the engine reads."""

comptime AttentionIn = Pointer[AttentionConfig, MutAnyOrigin]
"""A `usearch_dense_attention_config_t const *` the engine reads."""

comptime StatsOut = Pointer[TraversalStats, MutAnyOrigin]
"""A `usearch_dense_traversal_stats_t *` the engine overwrites."""


comptime DenseInit = def(OptionsIn, ErrorReport) thin abi("C") -> Address
"""`usearch_dense_init`: answers the handle's address, or zero on failure."""
comptime DenseFree = def(Handle) thin abi("C") -> None
"""`usearch_dense_free`: releases a handle; cannot fail."""
comptime DenseReserve = def(Handle, c_size_t, ErrorReport) thin abi("C") -> None
"""`usearch_dense_reserve`: reserves total capacity ahead of insertion."""
comptime DenseClear = def(Handle, ErrorReport) thin abi("C") -> None
"""`usearch_dense_clear`: erases every entry, keeping the capacity and the frozen geometry."""
comptime DenseMeasure = def(Handle, ErrorReport) thin abi("C") -> c_size_t
"""`usearch_dense_size` and every other whole-index count: one `size_t` returned by value."""
comptime DenseCapabilities = def(Handle, ErrorReport) thin abi("C") -> Address
"""`usearch_dense_capabilities`: the dispatched codename, as a static C string."""

comptime DenseAdd = def(
    Handle, Handle, KeysView[ImmUntrackedOrigin], MatrixViewIn, ExpansionIn, Address, Address, StatsOut, ErrorReport
) thin abi("C") -> c_size_t
"""`usearch_dense_add`: the rows and their keys, answering how many were published."""

comptime DenseRefine = def(Handle, Handle, ExpansionIn, Address, Address, StatsOut, ErrorReport) thin abi(
    "C"
) -> c_size_t
"""`usearch_dense_refine`: re-selects every entry's neighbors, answering how many changed."""

comptime DenseSearch = def(
    Handle, Handle, MatrixViewIn, MatchesIn, ExpansionIn, Address, Address, StatsOut, ErrorReport
) thin abi("C") -> None
"""`usearch_dense_search`: k-approximate nearest neighbors for a batch of queries."""

comptime DenseSearchIf = def(
    Handle, Handle, MatrixViewIn, MatchesIn, ExpansionIn, Address, Address, Address, Address, StatsOut, ErrorReport
) thin abi("C") -> None
"""`usearch_dense_search_if`: the same walk, admitting only what the predicate approves."""

comptime DenseCount = def(Handle, Key, ErrorReport) thin abi("C") -> c_size_t
"""`usearch_dense_count`: how many entries share one key."""

comptime DenseGet = def(Handle, Key, MatrixSpanIn, ErrorReport) thin abi("C") -> c_size_t
"""`usearch_dense_get`: exports the rows under one key, answering the true count."""

comptime DenseOpenCursor = def(Handle, ErrorReport) thin abi("C") -> Address
"""`usearch_dense_cursor`: opens a snapshot cursor, answering its address or zero."""
comptime DenseCursorNext = def(Handle, KeysSpan[MutUntrackedOrigin], MatrixSpanIn, ErrorReport) thin abi(
    "C"
) -> c_size_t
"""`usearch_dense_cursor_next`: copies the next rows out of the snapshot."""
comptime DenseCursorFree = def(Handle) thin abi("C") -> None
"""`usearch_dense_cursor_free`: releases the cursor."""

comptime DenseFind = def(Handle, Handle, MatrixViewIn, ErrorReport) thin abi("C") -> Address
"""`usearch_dense_find`: opens a resumable stream for one query, pinning one executor thread."""
comptime DenseFindNext = def(Handle, KeysSpan[MutUntrackedOrigin], ScoresOut, ErrorReport) thin abi("C") -> c_size_t
"""`usearch_dense_find_next`: pulls the next matches off the live frontier."""
comptime DenseFindFree = def(Handle) thin abi("C") -> None
"""`usearch_dense_find_free`: releases the finder and unpins its thread."""

comptime DenseSave = def(Handle, Handle, BytesView[ImmUntrackedOrigin], Address, Address, ErrorReport) thin abi(
    "C"
) -> None
"""`usearch_dense_save`: writes a checkpoint, the path extension picking the layout."""
comptime DenseLoad = def(Handle, Handle, BytesView[ImmUntrackedOrigin], Address, Address, ErrorReport) thin abi(
    "C"
) -> None
"""`usearch_dense_load`: reads a checkpoint into an initialized handle, shard by shard."""
comptime DenseMetadata = def(BytesView[ImmUntrackedOrigin], OptionsOut, ErrorReport) thin abi("C") -> None
"""`usearch_dense_metadata`: reads a checkpoint's geometry without loading it."""
comptime DenseSaveBuffer = def(Handle, Handle, BytesSpan[MutUntrackedOrigin], Address, Address, ErrorReport) thin abi(
    "C"
) -> None
"""`usearch_dense_save_buffer`: writes one monolith into a caller-owned buffer."""
comptime DenseLoadBuffer = def(Handle, Handle, BytesView[ImmUntrackedOrigin], Address, Address, ErrorReport) thin abi(
    "C"
) -> None
"""`usearch_dense_load_buffer`: reads one monolith out of a caller-owned buffer."""
comptime DenseMetadataBuffer = def(BytesView[ImmUntrackedOrigin], OptionsOut, ErrorReport) thin abi("C") -> None
"""`usearch_dense_metadata_buffer`: reads a monolith buffer's geometry without loading it."""

comptime DenseMultiply = def(
    Handle, Handle, MatrixViewIn, MatchesIn, ExpansionIn, Address, Address, StatsOut, ErrorReport
) thin abi("C") -> None
"""`usearch_dense_multiply`: every pair the walk scored, not only the nearest."""

comptime DenseAttend = def(
    Handle,
    Handle,
    MatrixViewIn,
    MatrixViewIn,
    MatrixSpanIn,
    Address,
    AttentionIn,
    Address,
    Address,
    StatsOut,
    ErrorReport,
) thin abi("C") -> None
"""`usearch_dense_attend`: softmax-weighted attention folded into the walk, in one pass."""

comptime DenseMaxsim = def(
    Handle, Handle, MatrixViewIn, c_size_t, MatchesIn, ExpansionIn, Address, Address, StatsOut, ErrorReport
) thin abi("C") -> None
"""`usearch_dense_maxsim`: late-interaction scoring, every query row taking its best match per key."""

# endregion Signatures

# region Matrices


@fieldwise_init
struct MatrixView[origin: ImmOrigin = ImmUntrackedOrigin](ImplicitlyCopyable, TrivialRegisterPassable):
    """A borrowed strided matrix - `usearch_dense_matrix_view_t`, the batch currency every verb reads.

    One vector is a one-row matrix. The row and column counts are scalars; `stride_bytes` alone is
    a byte count, which is why it says so in its name, and zero means rows sit contiguously at
    their natural width. `columns` is what the engine checks the frozen geometry against.

    Parameters:
        origin: The borrowed rows, named so they outlive every call this view crosses.
    """

    var data: ReadData
    """Base pointer of the first row; non-null, since a Mojo pointer cannot be otherwise."""
    var rows: c_size_t
    """Rows the matrix carries."""
    var columns: c_size_t
    """Logical scalars per row, as the caller counts them."""
    var stride_bytes: c_size_t
    """Bytes between the starts of consecutive rows; zero means contiguous."""
    var scalar_kind: ScalarKind
    """The scalar type the rows hold, converted on the way in where it differs."""

    @staticmethod
    def of[
        dtype: DType, rows_origin: ImmOrigin, //
    ](rows: Span[Scalar[dtype], rows_origin], columns: Int, scalar_kind: ScalarKind) raises Error -> MatrixView[
        rows_origin
    ]:
        """Reads a flat run of scalars as a contiguous matrix of the given width.

        The row count comes from the bytes the run spans divided by the bytes one row of
        `scalar_kind` occupies, so a kind that disagrees with `dtype` is refused here rather than
        read past the end by the engine. A sub-byte kind counts logical dimensions, so eight
        `U1X8` columns are one `UInt8`.

        Parameters:
            dtype: The scalar type of the run, inferred from the argument.
            rows_origin: The run's origin, inferred with it.

        Args:
            rows: The scalars, laid out row after row with no padding between them.
            columns: Logical scalars per row, which must divide the run into whole rows.
            scalar_kind: How the engine should read those scalars.

        Returns:
            A view of the whole run, with a derived contiguous stride.

        Raises:
            `Error` when the run does not divide into whole rows of that width and kind.
        """
        var row_bytes = scalar_kind.bytes_for_columns(columns)
        var spanned = len(rows) * size_of[Scalar[dtype]]()
        if columns <= 0 or row_bytes == 0 or spanned % row_bytes != 0:
            raise Error(Status.SHAPE_MISMATCH, "MatrixView.of", NULL_ADDRESS)
        return MatrixView[rows_origin](
            Pointer[NoneType, ImmUntrackedOrigin](unsafe_from_address=Int(rows.unsafe_ptr())),
            c_size_t(spanned // row_bytes),
            c_size_t(columns),
            c_size_t(0),
            scalar_kind,
        )

    def untracked(self) -> MatrixView[ImmUntrackedOrigin]:
        """The same matrix with its borrow erased - the one instantiation the ABI signatures name.

        Returns:
            A view of the same rows, whose type no longer keeps their storage alive.
        """
        return MatrixView[ImmUntrackedOrigin](self.data, self.rows, self.columns, self.stride_bytes, self.scalar_kind)


@fieldwise_init
struct MatrixSpan[origin: MutOrigin = MutUntrackedOrigin](ImplicitlyCopyable, TrivialRegisterPassable):
    """The writable twin - `usearch_dense_matrix_span_t`, carried by every verb that lands rows.

    The row count is the capacity the engine may fill rather than a promise about what it will.

    Parameters:
        origin: The lent rows, named so they outlive every call this span crosses.
    """

    var data: StoredWriteData
    """Base pointer of the first row the engine may write."""
    var capacity: c_size_t
    """Rows the matrix can hold, which a verb writes up to and reports back."""
    var columns: c_size_t
    """Logical scalars per row, as the caller counts them."""
    var stride_bytes: c_size_t
    """Bytes between the starts of consecutive rows; zero means contiguous."""
    var scalar_kind: ScalarKind
    """The scalar type to write, converted on the way out where it differs."""

    @staticmethod
    def of[
        dtype: DType, rows_origin: MutOrigin, //
    ](rows: Span[Scalar[dtype], rows_origin], columns: Int, scalar_kind: ScalarKind) raises Error -> MatrixSpan[
        rows_origin
    ]:
        """Writes into a flat run of scalars as a contiguous matrix of the given width.

        The capacity comes from the bytes the run spans divided by the bytes one row of
        `scalar_kind` occupies, so a kind that disagrees with `dtype` is refused here rather than
        written past the end by the engine.

        Parameters:
            dtype: The scalar type of the run, inferred from the argument.
            rows_origin: The run's origin, inferred with it.

        Args:
            rows: The storage, laid out row after row with no padding between them.
            columns: Logical scalars per row, which must divide the run into whole rows.
            scalar_kind: How the engine should write those scalars.

        Returns:
            A span over the whole run, with a derived contiguous stride.

        Raises:
            `Error` when the run does not divide into whole rows of that width and kind.
        """
        var row_bytes = scalar_kind.bytes_for_columns(columns)
        var spanned = len(rows) * size_of[Scalar[dtype]]()
        if columns <= 0 or row_bytes == 0 or spanned % row_bytes != 0:
            raise Error(Status.SHAPE_MISMATCH, "MatrixSpan.of", NULL_ADDRESS)
        return MatrixSpan[rows_origin](
            StoredWriteData(unsafe_from_address=Int(rows.unsafe_ptr())),
            c_size_t(spanned // row_bytes),
            c_size_t(columns),
            c_size_t(0),
            scalar_kind,
        )

    def untracked(self) -> MatrixSpan[MutUntrackedOrigin]:
        """The same matrix with its borrow erased - the one instantiation the ABI signatures name.

        Returns:
            A span over the same storage, whose type no longer keeps it alive.
        """
        return MatrixSpan[MutUntrackedOrigin](
            self.data, self.capacity, self.columns, self.stride_bytes, self.scalar_kind
        )


@fieldwise_init
struct MatchesSpan[
    keys_origin: MutOrigin = MutUntrackedOrigin,
    scores_origin: MutOrigin = MutUntrackedOrigin,
    counts_origin: MutOrigin = MutUntrackedOrigin,
](ImplicitlyCopyable, TrivialRegisterPassable):
    """Where a batch of matches lands: `rows x wanted` keys, their scores, and one count per row.

    Every ranking verb - both searches, the sparse product, and late interaction - fills exactly
    this triple, so it travels as one value rather than as six loose arguments. Each count reports
    the true number the walk found, never the number written, so a count above `wanted` means that
    row was truncated and the call should be repeated with more room.

    Parameters:
        keys_origin: The lent key storage, named so it outlives every call this span crosses.
        scores_origin: The lent score storage, named for the same reason.
        counts_origin: The lent per-row count storage, named for the same reason.
    """

    var keys: StoredKeys
    """First key of the first row."""
    var keys_stride_bytes: c_size_t
    """Bytes between the starts of consecutive key rows; zero means contiguous."""
    var scores: StoredScores
    """First score of the first row, paired positionally with the keys."""
    var scores_stride_bytes: c_size_t
    """Bytes between the starts of consecutive score rows; zero means contiguous."""
    var counts: StoredCounts
    """One true count per row, of at least as many entries as the batch has rows."""
    var wanted: c_size_t
    """Column capacity of both matrices - the "k" the verb is asked for."""

    @staticmethod
    def of[
        found_keys_origin: MutOrigin, found_scores_origin: MutOrigin, found_counts_origin: MutOrigin, //
    ](
        keys: Span[Key, found_keys_origin],
        scores: Span[Distance, found_scores_origin],
        counts: Span[c_size_t, found_counts_origin],
    ) raises Error -> MatchesSpan[found_keys_origin, found_scores_origin, found_counts_origin]:
        """Lands the batch in three runs the caller already sized to `rows x wanted`.

        The strides and `wanted` follow from the lengths, so they are derived here rather than
        asked for - three numbers a caller could contradict become one shape the lengths fix.

        Parameters:
            found_keys_origin: The key run's origin, inferred from the argument.
            found_scores_origin: The score run's origin, inferred with it.
            found_counts_origin: The count run's origin, inferred with it.

        Args:
            keys: Storage for the keys, row after row.
            scores: Storage for the scores, row after row and the same shape.
            counts: Storage for the per-row counts, one per row.

        Returns:
            A span over all three, with derived contiguous strides.

        Raises:
            `Error` when the three runs do not describe one `rows x wanted` rectangle.
        """
        if len(counts) == 0 or len(keys) != len(scores) or len(keys) % len(counts) != 0:
            raise Error(Status.SHAPE_MISMATCH, "MatchesSpan.of", NULL_ADDRESS)
        return MatchesSpan[found_keys_origin, found_scores_origin, found_counts_origin](
            StoredKeys(unsafe_from_address=Int(keys.unsafe_ptr())),
            c_size_t(0),
            StoredScores(unsafe_from_address=Int(scores.unsafe_ptr())),
            c_size_t(0),
            StoredCounts(unsafe_from_address=Int(counts.unsafe_ptr())),
            c_size_t(len(keys) // len(counts)),
        )

    def untracked(self) -> MatchesSpan[MutUntrackedOrigin, MutUntrackedOrigin, MutUntrackedOrigin]:
        """The same triple with its borrows erased - the one instantiation the ABI signatures name.

        Returns:
            A span over the same storage, whose type no longer keeps any of it alive.
        """
        return MatchesSpan[MutUntrackedOrigin, MutUntrackedOrigin, MutUntrackedOrigin](
            self.keys,
            self.keys_stride_bytes,
            self.scores,
            self.scores_stride_bytes,
            self.counts,
            self.wanted,
        )


# endregion Matrices

# region Configuration


@fieldwise_init
struct Metric(Equatable, ImplicitlyCopyable, TrivialRegisterPassable, Writable):
    """The built-in dense metrics - `usearch_dense_metric_t`, each with its own SIMD kernel chain.

    The last two read their rows as `ScalarKind.U1X8` bit sets and refuse any other quantization; the
    first four refuse that one. The three linear-algebra verbs run on `INNER` alone, since a
    product and a softmax both need logits rather than distances.
    """

    var code: c_int
    """The raw `usearch_dense_metric_t` value the ABI exchanges."""
    comptime UNKNOWN = Self(0)
    """No built-in kind - a custom callback supplies the scoring and forfeits the kernel chains."""
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

    def write_to(self, mut writer: Some[Writer]):
        """Names which of the six this is.

        Args:
            writer: The sink the name is appended to.
        """
        if self == Self.INNER:
            writer.write("inner")
        elif self == Self.ANGULAR:
            writer.write("angular")
        elif self == Self.EUCLIDEAN:
            writer.write("euclidean")
        elif self == Self.HAMMING:
            writer.write("hamming")
        elif self == Self.JACCARD:
            writer.write("jaccard")
        else:
            writer.write("unknown")


@fieldwise_init
struct Admission(Equatable, ImplicitlyCopyable, TrivialRegisterPassable):
    """Whether a filtered search reports one candidate key or passes over it - `usearch_admission_t`."""

    var code: c_int
    """The raw `usearch_admission_t` value the ABI exchanges."""
    comptime REJECT = Self(0)
    """Skip the key; the walk still routes through it."""
    comptime ADMIT = Self(1)
    """Report the key among the matches."""


@fieldwise_init
struct ExpansionConfig(ImplicitlyCopyable, TrivialRegisterPassable):
    """Per-call breadth, shared by insertion and both searches - `usearch_dense_expansion_config_t`.

    A pure input, never written. Every field zeroed derives every setting, which is what the
    default constructor leaves behind.
    """

    var expansion_factor: c_size_t
    """The candidate-pool breadth; zero derives."""

    def __init__(out self):
        """Derives every setting."""
        self.expansion_factor = c_size_t(0)

    def __init__(out self, *, expansion_factor: Int):
        """Pins the candidate-pool breadth for this call.

        Args:
            expansion_factor: How wide the pool is kept; zero derives.
        """
        self.expansion_factor = c_size_t(expansion_factor)


@fieldwise_init
struct AttentionConfig(ImplicitlyCopyable, TrivialRegisterPassable):
    """Per-call settings of the attention verb - `usearch_dense_attention_config_t`.

    A pure input, never written, and every field zeroed derives every setting.
    """

    var expansion_factor: c_size_t
    """The candidate-pool breadth; zero derives."""
    var score_scale: Float32
    """The logit scale, conventionally the inverse square root of the head dimensionality; zero derives."""
    var mass_tolerance: Float32
    """The residual-mass threshold the walk stops at; zero derives."""

    def __init__(out self):
        """Derives every setting."""
        self.expansion_factor = c_size_t(0)
        self.score_scale = Float32(0)
        self.mass_tolerance = Float32(0)


@fieldwise_init
struct SoftmaxState(ImplicitlyCopyable, TrivialRegisterPassable):
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


@fieldwise_init
struct TraversalStats(ImplicitlyCopyable):
    """Traversal telemetry - `usearch_dense_traversal_stats_t`, a pure output.

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
struct DenseOptions[origin: ImmOrigin = ImmUntrackedOrigin](ImplicitlyCopyable):
    """Construction and loading options - `usearch_dense_options_t`, laid out as the ABI declares.

    The `residency` field tags which arm of the C union the last two fields are; every arm is a
    borrowed array and its count, so the two fields serve all three and the placement factories
    below name which. Zeroed optional fields derive from the geometry.

    Never marked register-passable: its address crosses to C, and Mojo 1.0 mislays a struct this
    wide once it is, so the metadata probes would read back nothing.

    Parameters:
        origin: The borrowed placement array, named so it outlives the call that reads it.
    """

    var metric: Metric
    """The metric kind used for distance calculations between vectors."""
    var metric_callback: Address
    """A `usearch_dense_metric_callback_t`; `NULL_ADDRESS` unless `metric` is `Metric.UNKNOWN`.

    An address rather than a Mojo function, for the reason this module's docstring records: a
    `def ... abi("C")` returning a value does not lower to the C calling convention in Mojo 1.0,
    and a scoring kernel returns a distance. A kernel compiled in C installs here unchanged, and
    must outlive the index.
    """
    var metric_callback_state: Address
    """The opaque state forwarded into every custom metric call."""
    var quantization: ScalarKind
    """The scalar type vectors are stored and quantized into."""
    var dimensions: c_size_t
    """Dimensions in the indexed vectors; zero defers the geometry to a later load."""
    var connectivity: c_size_t
    """The connections-per-node bound; zero derives."""
    var values_per_key: c_size_t
    """Upper bound on entries sharing one key; zero derives to one - unique keys."""
    var residency: Residency
    """Selects the active arm of the two placement fields, and the meaning of "load"."""
    var placement: Address
    """The arm's borrowed array - NUMA node ids, device ordinals, or paths; `NULL_ADDRESS` derives."""
    var placement_count: c_size_t
    """How many entries that array holds; zero derives."""

    def __init__(
        out self,
        metric: Metric,
        quantization: ScalarKind,
        dimensions: Int,
        connectivity: Int = 0,
        values_per_key: Int = 0,
        residency: Residency = Residency.RAM,
    ):
        """Describes an index by its geometry, leaving placement to be derived.

        Args:
            metric: The metric kind the graph is frozen under.
            quantization: The scalar type vectors are stored as.
            dimensions: Dimensions per vector; zero defers the geometry to a later load.
            connectivity: The connections-per-node bound; zero derives.
            values_per_key: Upper bound on entries sharing one key; zero derives to one.
            residency: Where the payload lives and which engine traverses it.
        """
        self.metric = metric
        self.metric_callback = NULL_ADDRESS
        self.metric_callback_state = NULL_ADDRESS
        self.quantization = quantization
        self.dimensions = c_size_t(dimensions)
        self.connectivity = c_size_t(connectivity)
        self.values_per_key = c_size_t(values_per_key)
        self.residency = residency
        self.placement = NULL_ADDRESS
        self.placement_count = c_size_t(0)

    def on_memory_domains[
        domains_origin: ImmOrigin, //
    ](self, memory_domains: Span[Int32, domains_origin]) -> DenseOptions[domains_origin]:
        """Names the NUMA nodes a RAM-resident index places its shards in.

        Answers re-typed options rather than mutating in place, because naming the array is what
        binds its origin, and a value cannot gain a parameter after it is built.

        Parameters:
            domains_origin: The run's origin, inferred from the argument.

        Args:
            memory_domains: The node ids to place one shard on apiece.

        Returns:
            The same geometry, placed on those nodes and holding their storage alive.
        """
        return DenseOptions[domains_origin](
            self.metric,
            self.metric_callback,
            self.metric_callback_state,
            self.quantization,
            self.dimensions,
            self.connectivity,
            self.values_per_key,
            Residency.RAM,
            Int(memory_domains.unsafe_ptr()),
            c_size_t(len(memory_domains)),
        )

    def on_devices[
        devices_origin: ImmOrigin, //
    ](self, devices: Span[Int32, devices_origin]) -> DenseOptions[devices_origin]:
        """Names the CUDA ordinals a VRAM-resident index shards across.

        Parameters:
            devices_origin: The run's origin, inferred from the argument.

        Args:
            devices: The device ordinals to place one shard on apiece.

        Returns:
            The same geometry, placed on those devices and holding their storage alive.
        """
        return DenseOptions[devices_origin](
            self.metric,
            self.metric_callback,
            self.metric_callback_state,
            self.quantization,
            self.dimensions,
            self.connectivity,
            self.values_per_key,
            Residency.VRAM,
            Int(devices.unsafe_ptr()),
            c_size_t(len(devices)),
        )

    def on_paths[
        paths_origin: ImmOrigin, //
    ](self, paths: Span[BytesView[ImmUntrackedOrigin], paths_origin]) -> DenseOptions[paths_origin]:
        """Names the devices or directories an NVMe-resident index places its shards on.

        Parameters:
            paths_origin: The run's origin, inferred from the argument.

        Args:
            paths: The device or directory names, one shard apiece.

        Returns:
            The same geometry, placed on those paths and holding their storage alive.
        """
        return DenseOptions[paths_origin](
            self.metric,
            self.metric_callback,
            self.metric_callback_state,
            self.quantization,
            self.dimensions,
            self.connectivity,
            self.values_per_key,
            Residency.NVME,
            Int(paths.unsafe_ptr()),
            c_size_t(len(paths)),
        )

    def untracked(self) -> DenseOptions[ImmUntrackedOrigin]:
        """The same options with the borrow erased - the one instantiation the ABI signatures name.

        Returns:
            Options naming the same placement array, whose type no longer keeps it alive.
        """
        return DenseOptions[ImmUntrackedOrigin](
            self.metric,
            self.metric_callback,
            self.metric_callback_state,
            self.quantization,
            self.dimensions,
            self.connectivity,
            self.values_per_key,
            self.residency,
            self.placement,
            self.placement_count,
        )


# endregion Configuration

# region Index


struct DenseIndex:
    """A dense vector index. Released when the last reference to it goes out of use.

    Insertions and searches overlap freely on one handle: a search never writes to the graph, and
    an insertion publishes a row only once it is reachable, so a concurrent walk sees a consistent
    graph at every instant. What does not overlap is two calls driving one executor - the second
    reports `CONTENDED` before any work begins.
    """

    var library: Library
    """The loaded engine, held so its symbol table outlives every call made through the handle."""
    var address: Address
    """The opaque `usearch_dense_index_t`, or zero once a merge has consumed it."""

    # region Lifetime

    def __init__[
        options_origin: ImmOrigin, //
    ](out self, library: Library, options: DenseOptions[options_origin]) raises Error:
        """Initializes a handle for the residency the options select.

        Parameters:
            options_origin: The options' borrowed placement array, inferred from the argument.

        Args:
            library: The loaded engine, kept so the shared object stays mapped.
            options: The geometry, metric, and placement; read only during this call.

        Raises:
            `Error` if the geometry is refused, the placement cannot be resolved, or the
            allocation fails.
        """
        self.library = library
        var report = stack_allocation[1, ErrorSlot]()
        report[unsafe_offset=0] = ErrorSlot()
        var slot = report.unsafe_origin_cast[MutAnyOrigin]()
        var settings = stack_allocation[1, DenseOptions[ImmUntrackedOrigin]]()
        settings[unsafe_offset=0] = options.untracked()
        var address = library.symbols().dense_init(settings.unsafe_origin_cast[MutAnyOrigin](), slot)
        Error.raise_if_failed(slot, "usearch_dense_init")
        if address == 0:
            raise Error(Status.INTERNAL, "usearch_dense_init", 0)
        self.address = address

    def __deinit__(deinit self):
        """Releases the index and everything it owns, unless a merge already took the handle.

        The loader reference is surrendered only after the call: reading the entry point is the
        last use of `library`, so without the trailing move Mojo drops the last reference - and
        `dlclose`s the engine - between the load and the branch to it.
        """
        if self.address != 0:
            self.library.symbols().dense_free(Handle(unsafe_from_address=self.address))
        _ = self.library^

    def live(self) raises Error -> Handle:
        """The handle a verb crosses, refusing one a merge has already consumed.

        Returns:
            The engine's `usearch_dense_index_t`.

        Raises:
            `Error` if this handle was donated to a merge, which the engine has since freed.
        """
        if self.address == 0:
            raise Error(Status.UNINITIALIZED, "the index handle was consumed by a merge", 0)
        return Handle(unsafe_from_address=self.address)

    def reserve(self, capacity: Int) raises Error:
        """Reserves capacity for a number of incoming vectors ahead of insertion.

        Args:
            capacity: The desired total capacity, including the current size.

        Raises:
            `Error` if the reservation cannot be satisfied, or contradicts a frozen geometry.
        """
        var report = stack_allocation[1, ErrorSlot]()
        report[unsafe_offset=0] = ErrorSlot()
        var slot = report.unsafe_origin_cast[MutAnyOrigin]()
        self.library.symbols().dense_reserve(self.live(), c_size_t(capacity), slot)
        Error.raise_if_failed(slot, "usearch_dense_reserve")

    def clear(self) raises Error:
        """Erases every entry, keeping the reserved capacity and the frozen geometry.

        Never invoked concurrently with any other verb on the same handle.

        Raises:
            `Error` if the handle opens its checkpoint in place, or the engine refuses.
        """
        var report = stack_allocation[1, ErrorSlot]()
        report[unsafe_offset=0] = ErrorSlot()
        var slot = report.unsafe_origin_cast[MutAnyOrigin]()
        self.library.symbols().dense_clear(self.live(), slot)
        Error.raise_if_failed(slot, "usearch_dense_clear")

    # endregion Lifetime

    # region Introspection

    @always_inline
    def _read_integer_property(self, symbol: DenseMeasure, detail: StaticString) raises Error -> Int:
        """Reads one of the handle's integer properties, which every geometry and footprint getter
        below shares the shape of; `capabilities` is the one property that answers text instead.

        Args:
            symbol: The `(handle, error) -> size_t` entry point to call.
            detail: The entry point's name, carried into any raised error.

        Returns:
            The reading, which is meaningful only when nothing raised.

        Raises:
            `Error` if the engine refuses the query.
        """
        var report = stack_allocation[1, ErrorSlot]()
        report[unsafe_offset=0] = ErrorSlot()
        var slot = report.unsafe_origin_cast[MutAnyOrigin]()
        var answer = symbol(self.live(), slot)
        Error.raise_if_failed(slot, detail)
        return Int(answer)

    def size(self) raises Error -> Int:
        """How many entries the index stores.

        Returns:
            The count of published rows, which a concurrent insertion may already have grown.

        Raises:
            `Error` if the engine refuses the query.
        """
        return self._read_integer_property(self.library.symbols().dense_size, "usearch_dense_size")

    def capacity(self) raises Error -> Int:
        """How many entries the reservation left room for.

        Returns:
            The reserved capacity in entries, never below `size`.

        Raises:
            `Error` if the engine refuses the query.
        """
        return self._read_integer_property(self.library.symbols().dense_capacity, "usearch_dense_capacity")

    def dimensions(self) raises Error -> Int:
        """The frozen dimensionality of the stored vectors.

        Returns:
            Logical scalars per row, which every batch's `columns` must match.

        Raises:
            `Error` if the engine refuses the query.
        """
        return self._read_integer_property(self.library.symbols().dense_dimensions, "usearch_dense_dimensions")

    def connectivity(self) raises Error -> Int:
        """The frozen connections-per-node bound of the graph.

        Returns:
            The bound every node's neighbor list is sized by.

        Raises:
            `Error` if the engine refuses the query.
        """
        return self._read_integer_property(self.library.symbols().dense_connectivity, "usearch_dense_connectivity")

    def memory_usage(self) raises Error -> Int:
        """How many bytes the index occupies.

        Returns:
            The byte count, arenas and neighbor lists included.

        Raises:
            `Error` if the engine refuses the query.
        """
        return self._read_integer_property(self.library.symbols().dense_memory_usage, "usearch_dense_memory_usage")

    def capabilities(self) raises Error -> String:
        """The capability codename the index dispatched to on this machine.

        Returns:
            The codename the engine picked, copied out of its own static storage.

        Raises:
            `Error` if the engine refuses the query.
        """
        var report = stack_allocation[1, ErrorSlot]()
        report[unsafe_offset=0] = ErrorSlot()
        var slot = report.unsafe_origin_cast[MutAnyOrigin]()
        var address = self.library.symbols().dense_capabilities(self.live(), slot)
        Error.raise_if_failed(slot, "usearch_dense_capabilities")
        return text_at(address)

    def storage_requirements(self) raises Error -> Int:
        """Bytes one monolith of this index takes under the default layout.

        Returns:
            What `save_buffer` needs; a manifest checkpoint's parts sum to about the same.

        Raises:
            `Error` if the engine refuses the query.
        """
        return self._read_integer_property(self.library.symbols().dense_storage_requirements, "usearch_dense_storage_requirements")

    # endregion Introspection

    # region Population

    def add[
        keys_origin: ImmOrigin, vectors_origin: ImmOrigin, //
    ](
        self,
        executor: Executor,
        keys: Span[Key, keys_origin],
        vectors: MatrixView[vectors_origin],
        mut stats: TraversalStats,
        config: ExpansionConfig = ExpansionConfig(),
    ) raises Error -> Int:
        """Adds a strided matrix of vectors under their keys - the native batch operation.

        A batch is all-or-nothing, because this binding installs no progress handler: every
        fallible step runs before the first entry identity is claimed, so a rejected batch leaves
        the index untouched and publishes no row.

        Parameters:
            keys_origin: The key run's origin, inferred from the argument.
            vectors_origin: The row storage's origin, inferred with it.

        Args:
            executor: The executor driving the batch; its kind must match the handle's residency.
            keys: One key per row of `vectors`; keys need not be unique.
            vectors: The rows to insert, converted on ingest to the stored type.
            stats: Receives this call's traversal telemetry, fully overwritten.
            config: Per-call breadth; the default derives every setting.

        Returns:
            How many rows were published, which is every row unless the call failed.

        Raises:
            `Error` if the shapes disagree, the handle is read-only, or a second thread is already
            driving `executor`.
        """
        var report = stack_allocation[1, ErrorSlot]()
        report[unsafe_offset=0] = ErrorSlot()
        var slot = report.unsafe_origin_cast[MutAnyOrigin]()
        var settings = stack_allocation[1, ExpansionConfig]()
        settings[unsafe_offset=0] = config
        var view = stack_allocation[1, MatrixView[ImmUntrackedOrigin]]()
        view[unsafe_offset=0] = vectors.untracked()
        var published = self.library.symbols().dense_add(
            self.live(),
            executor.handle,
            KeysView.of(keys).untracked(),
            view.unsafe_origin_cast[MutAnyOrigin](),
            settings.unsafe_origin_cast[MutAnyOrigin](),
            NULL_ADDRESS,
            NULL_ADDRESS,
            Pointer(to=stats).unsafe_origin_cast[MutAnyOrigin](),
            slot,
        )
        Error.raise_if_failed(slot, "usearch_dense_add")
        return Int(published)

    def refine(
        self,
        executor: Executor,
        mut stats: TraversalStats,
        config: ExpansionConfig = ExpansionConfig(),
    ) raises Error -> Int:
        """Re-selects every entry's neighbors against the finished graph.

        A build step rather than a repair: a cheap insertion pass followed by one refinement
        reaches a better graph, in less total time, than a single expensive insertion pass - so a
        wide expansion factor belongs here rather than in `add`. Searches may run throughout;
        insertions may not.

        Args:
            executor: The executor to fan out over; its kind must match the residency.
            stats: Receives this call's traversal telemetry, fully overwritten.
            config: Per-call breadth; the default derives every setting.

        Returns:
            How many nodes' edges changed; zero means the graph has settled.

        Raises:
            `Error` if the handle is read-only, or a second thread is already driving `executor`.
        """
        var report = stack_allocation[1, ErrorSlot]()
        report[unsafe_offset=0] = ErrorSlot()
        var slot = report.unsafe_origin_cast[MutAnyOrigin]()
        var settings = stack_allocation[1, ExpansionConfig]()
        settings[unsafe_offset=0] = config
        var changed = self.library.symbols().dense_refine(
            self.live(),
            executor.handle,
            settings.unsafe_origin_cast[MutAnyOrigin](),
            NULL_ADDRESS,
            NULL_ADDRESS,
            Pointer(to=stats).unsafe_origin_cast[MutAnyOrigin](),
            slot,
        )
        Error.raise_if_failed(slot, "usearch_dense_refine")
        return Int(changed)

    # endregion Population

    # region Retrieval

    def search[
        queries_origin: ImmOrigin,
        found_keys_origin: MutOrigin,
        found_scores_origin: MutOrigin,
        found_counts_origin: MutOrigin,
        //,
    ](
        self,
        executor: Executor,
        queries: MatrixView[queries_origin],
        matches: MatchesSpan[found_keys_origin, found_scores_origin, found_counts_origin],
        mut stats: TraversalStats,
        config: ExpansionConfig = ExpansionConfig(),
    ) raises Error:
        """Performs k-approximate nearest-neighbor search for a batch of queries.

        Never writes to the graph, so any number of these may overlap with concurrent insertions.
        A failure leaves every output row indeterminate, since a query resolved before the stop is
        indistinguishable from one that never ran.

        Parameters:
            queries_origin: The query rows' origin, inferred from the argument.
            found_keys_origin: The landing keys' origin, inferred with it.
            found_scores_origin: The landing scores' origin, inferred with it.
            found_counts_origin: The landing counts' origin, inferred with it.

        Args:
            executor: The executor driving the batch; its kind must match the handle's residency.
            queries: The query rows, one per search.
            matches: Where the keys, distances, and per-query counts land.
            stats: Receives this call's traversal telemetry, fully overwritten.
            config: Per-call breadth; the default derives every setting.

        Raises:
            `Error` if the shapes disagree, or a second thread is already driving `executor`.
        """
        var report = stack_allocation[1, ErrorSlot]()
        report[unsafe_offset=0] = ErrorSlot()
        var slot = report.unsafe_origin_cast[MutAnyOrigin]()
        var settings = stack_allocation[1, ExpansionConfig]()
        settings[unsafe_offset=0] = config
        var view = stack_allocation[1, MatrixView[ImmUntrackedOrigin]]()
        view[unsafe_offset=0] = queries.untracked()
        var landing = stack_allocation[1, MatchesSpan[MutUntrackedOrigin, MutUntrackedOrigin, MutUntrackedOrigin]]()
        landing[unsafe_offset=0] = matches.untracked()
        self.library.symbols().dense_search(
            self.live(),
            executor.handle,
            view.unsafe_origin_cast[MutAnyOrigin](),
            landing.unsafe_origin_cast[MutAnyOrigin](),
            settings.unsafe_origin_cast[MutAnyOrigin](),
            NULL_ADDRESS,
            NULL_ADDRESS,
            Pointer(to=stats).unsafe_origin_cast[MutAnyOrigin](),
            slot,
        )
        Error.raise_if_failed(slot, "usearch_dense_search")

    def search_if[
        queries_origin: ImmOrigin,
        found_keys_origin: MutOrigin,
        found_scores_origin: MutOrigin,
        found_counts_origin: MutOrigin,
        //,
    ](
        self,
        executor: Executor,
        queries: MatrixView[queries_origin],
        matches: MatchesSpan[found_keys_origin, found_scores_origin, found_counts_origin],
        mut stats: TraversalStats,
        predicate: Address,
        predicate_state: Address = NULL_ADDRESS,
        config: ExpansionConfig = ExpansionConfig(),
    ) raises Error:
        """The filtered sibling of `search`, admitting only entries the predicate approves.

        The predicate is an address rather than a Mojo function, because a `def ... abi("C")` that
        returns a value does not lower to the C calling convention in Mojo 1.0 and the engine
        calling one jumps into a corrupted frame. A predicate compiled in C and reached through the
        same `dlopen` - or any other `usearch_predicate_t` the caller already holds - installs
        here unchanged.

        Non-matching entries still route the traversal rather than terminating it, so recall does
        not collapse at low selectivity. The cost is that the predicate is called once per scored
        candidate, far more often than `wanted`, and concurrently from the executor's threads - so
        it belongs on the caller's fast path and must tolerate that.

        Parameters:
            queries_origin: The query rows' origin, inferred from the argument.
            found_keys_origin: The landing keys' origin, inferred with it.
            found_scores_origin: The landing scores' origin, inferred with it.
            found_counts_origin: The landing counts' origin, inferred with it.

        Args:
            executor: The executor driving the batch; its kind must match the handle's residency.
            queries: The query rows, one per search.
            matches: Where the keys, distances, and per-query counts land.
            stats: Receives this call's traversal telemetry, fully overwritten.
            predicate: A `usearch_predicate_t`, which must outlive the call.
            predicate_state: The opaque state forwarded into every predicate call.
            config: Per-call breadth; the default derives every setting.

        Raises:
            `Error` if the shapes disagree, or a second thread is already driving `executor`.
        """
        var report = stack_allocation[1, ErrorSlot]()
        report[unsafe_offset=0] = ErrorSlot()
        var slot = report.unsafe_origin_cast[MutAnyOrigin]()
        var settings = stack_allocation[1, ExpansionConfig]()
        settings[unsafe_offset=0] = config
        var view = stack_allocation[1, MatrixView[ImmUntrackedOrigin]]()
        view[unsafe_offset=0] = queries.untracked()
        var landing = stack_allocation[1, MatchesSpan[MutUntrackedOrigin, MutUntrackedOrigin, MutUntrackedOrigin]]()
        landing[unsafe_offset=0] = matches.untracked()
        self.library.symbols().dense_search_if(
            self.live(),
            executor.handle,
            view.unsafe_origin_cast[MutAnyOrigin](),
            landing.unsafe_origin_cast[MutAnyOrigin](),
            settings.unsafe_origin_cast[MutAnyOrigin](),
            predicate,
            predicate_state,
            NULL_ADDRESS,
            NULL_ADDRESS,
            Pointer(to=stats).unsafe_origin_cast[MutAnyOrigin](),
            slot,
        )
        Error.raise_if_failed(slot, "usearch_dense_search_if")

    def count(self, key: Key) raises Error -> Int:
        """How many entries the index stores under one key.

        Pairing this with `get` is not atomic, but harmlessly so: entries are only ever added, so a
        stale count under-reports and `get` returns the true total.

        Args:
            key: The key whose entries are counted.

        Returns:
            The number of entries sharing the key; zero means the index holds none.

        Raises:
            `Error` if the engine refuses the query.
        """
        var report = stack_allocation[1, ErrorSlot]()
        report[unsafe_offset=0] = ErrorSlot()
        var slot = report.unsafe_origin_cast[MutAnyOrigin]()
        var found = self.library.symbols().dense_count(self.live(), key, slot)
        Error.raise_if_failed(slot, "usearch_dense_count")
        return Int(found)

    def get[vectors_origin: MutOrigin, //](self, key: Key, vectors: MatrixSpan[vectors_origin]) raises Error -> Int:
        """Exports the vectors stored under one key, one row per entry, in insertion order.

        Parameters:
            vectors_origin: The landing rows' origin, inferred from the argument.

        Args:
            key: The key whose entries are exported.
            vectors: Where the rows land; its capacity bounds the copy, never promising it.

        Returns:
            The true number of entries under the key, never the number written - a return above
            the span's capacity means the copy was truncated.

        Raises:
            `Error` if the shapes disagree, or the engine refuses the export.
        """
        var report = stack_allocation[1, ErrorSlot]()
        report[unsafe_offset=0] = ErrorSlot()
        var slot = report.unsafe_origin_cast[MutAnyOrigin]()
        var span = stack_allocation[1, MatrixSpan[MutUntrackedOrigin]]()
        span[unsafe_offset=0] = vectors.untracked()
        var found = self.library.symbols().dense_get(
            self.live(),
            key,
            span.unsafe_origin_cast[MutAnyOrigin](),
            slot,
        )
        Error.raise_if_failed(slot, "usearch_dense_get")
        return Int(found)

    # endregion Retrieval

    # region Serialization

    def save[path_origin: ImmOrigin, //](self, executor: Executor, path: BytesView[path_origin]) raises Error:
        """Writes a checkpoint, the path extension picking the layout.

        A `.safegraphs` path is one monolith; a `.safegraphs.json` path is a manifest naming one
        monolith per shard. Demands a mutation-free window; concurrent readers keep searching
        throughout. A failed save leaves a file without its header, which every reader rejects, and
        that leftover is the caller's to remove.

        Parameters:
            path_origin: The path bytes' origin, inferred from the argument.

        Args:
            executor: The executor the write fans out over, one shard per thread or stream.
            path: Destination path, whose extension picks the layout.

        Raises:
            `Error` if the destination cannot be written, or a second thread is driving `executor`.
        """
        var report = stack_allocation[1, ErrorSlot]()
        report[unsafe_offset=0] = ErrorSlot()
        var slot = report.unsafe_origin_cast[MutAnyOrigin]()
        self.library.symbols().dense_save(
            self.live(), executor.handle, path.untracked(), NULL_ADDRESS, NULL_ADDRESS, slot
        )
        Error.raise_if_failed(slot, "usearch_dense_save")

    def load[path_origin: ImmOrigin, //](self, executor: Executor, path: BytesView[path_origin]) raises Error:
        """Reads a checkpoint into this handle, shard by shard.

        The handle's residency decides what loading means: materialize into RAM, upload to the
        devices, or open in place on disk. A handle whose geometry was left zeroed freezes it here,
        from the checkpoint's own headers. Point the path at the manifest, never at one of the
        parts it enumerates - naming a part loads exactly that part, which is a fragment of the
        index rather than the index. A failed load publishes nothing and loads cleanly again.

        Parameters:
            path_origin: The path bytes' origin, inferred from the argument.

        Args:
            executor: The executor the read fans out over, one shard per thread or stream.
            path: The manifest of a sharded checkpoint, or a single monolith.

        Raises:
            `Error` if the checkpoint cannot be read, or its geometry does not fit this handle.
        """
        var report = stack_allocation[1, ErrorSlot]()
        report[unsafe_offset=0] = ErrorSlot()
        var slot = report.unsafe_origin_cast[MutAnyOrigin]()
        self.library.symbols().dense_load(
            self.live(), executor.handle, path.untracked(), NULL_ADDRESS, NULL_ADDRESS, slot
        )
        Error.raise_if_failed(slot, "usearch_dense_load")

    def save_buffer[
        buffer_origin: MutOrigin, //
    ](self, executor: Executor, buffer: BytesSpan[buffer_origin]) raises Error:
        """Writes one monolith into a caller-owned buffer of at least `storage_requirements` bytes.

        The bytes are the same the path-taking sibling writes, so the two are interchangeable.

        Parameters:
            buffer_origin: The destination bytes' origin, inferred from the argument.

        Args:
            executor: The executor to fan out over; its kind must match the residency.
            buffer: Destination, whose length the engine never exceeds.

        Raises:
            `Error` if the buffer is too small, or a second thread is driving `executor`.
        """
        var report = stack_allocation[1, ErrorSlot]()
        report[unsafe_offset=0] = ErrorSlot()
        var slot = report.unsafe_origin_cast[MutAnyOrigin]()
        self.library.symbols().dense_save_buffer(
            self.live(), executor.handle, buffer.untracked(), NULL_ADDRESS, NULL_ADDRESS, slot
        )
        Error.raise_if_failed(slot, "usearch_dense_save_buffer")

    def load_buffer[
        buffer_origin: ImmOrigin, //
    ](self, executor: Executor, buffer: BytesView[buffer_origin]) raises Error:
        """Reads one monolith out of a caller-owned buffer into this handle.

        The monolith lands whole on the first shard, and a failed load publishes nothing. The
        path-taking sibling reaches the same bytes through a mapping, so a mapped file needs no
        copy into a buffer first.

        Parameters:
            buffer_origin: The source bytes' origin, inferred from the argument.

        Args:
            executor: The executor to fan out over; its kind must match the residency.
            buffer: Source holding one serialized monolith.

        Raises:
            `Error` if the bytes cannot be decoded, or do not fit this handle's geometry.
        """
        var report = stack_allocation[1, ErrorSlot]()
        report[unsafe_offset=0] = ErrorSlot()
        var slot = report.unsafe_origin_cast[MutAnyOrigin]()
        self.library.symbols().dense_load_buffer(
            self.live(), executor.handle, buffer.untracked(), NULL_ADDRESS, NULL_ADDRESS, slot
        )
        Error.raise_if_failed(slot, "usearch_dense_load_buffer")

    # endregion Serialization

    # region Linear Algebra

    def multiply[
        rows_origin: ImmOrigin,
        pair_keys_origin: MutOrigin,
        pair_scores_origin: MutOrigin,
        pair_counts_origin: MutOrigin,
        //,
    ](
        self,
        executor: Executor,
        rows: MatrixView[rows_origin],
        pairs: MatchesSpan[pair_keys_origin, pair_scores_origin, pair_counts_origin],
        mut stats: TraversalStats,
        config: ExpansionConfig = ExpansionConfig(),
    ) raises Error:
        """Approximate sparse product of a matrix against the indexed rows.

        Reports every pair the traversal scored instead of only the nearest, in traversal order
        rather than sorted, and with no duplicates. Cost is roughly
        `rows x expansion x connectivity x dimensions`, sublinear in the indexed rows, which is the
        whole point of reading a product off a graph. Runs on `Metric.INNER` alone.

        `pairs.keys` names each pair's indexed row and `pairs.scores` carries the products.

        Parameters:
            rows_origin: The left-hand rows' origin, inferred from the argument.
            pair_keys_origin: The landing keys' origin, inferred with it.
            pair_scores_origin: The landing products' origin, inferred with it.
            pair_counts_origin: The landing counts' origin, inferred with it.

        Args:
            executor: The executor driving the batch; its kind must match the handle's residency.
            rows: The left-hand rows.
            pairs: Where the column keys, products, and per-row pair counts land.
            stats: Receives this call's traversal telemetry, fully overwritten.
            config: Per-call breadth, which is what widens the emitted pattern.

        Raises:
            `Error` with `CONFIG_MISMATCH` if the graph was frozen under a distance kind.
        """
        var report = stack_allocation[1, ErrorSlot]()
        report[unsafe_offset=0] = ErrorSlot()
        var slot = report.unsafe_origin_cast[MutAnyOrigin]()
        var settings = stack_allocation[1, ExpansionConfig]()
        settings[unsafe_offset=0] = config
        var view = stack_allocation[1, MatrixView[ImmUntrackedOrigin]]()
        view[unsafe_offset=0] = rows.untracked()
        var landing = stack_allocation[1, MatchesSpan[MutUntrackedOrigin, MutUntrackedOrigin, MutUntrackedOrigin]]()
        landing[unsafe_offset=0] = pairs.untracked()
        self.library.symbols().dense_multiply(
            self.live(),
            executor.handle,
            view.unsafe_origin_cast[MutAnyOrigin](),
            landing.unsafe_origin_cast[MutAnyOrigin](),
            settings.unsafe_origin_cast[MutAnyOrigin](),
            NULL_ADDRESS,
            NULL_ADDRESS,
            Pointer(to=stats).unsafe_origin_cast[MutAnyOrigin](),
            slot,
        )
        Error.raise_if_failed(slot, "usearch_dense_multiply")

    def attend[
        queries_origin: ImmOrigin,
        values_origin: ImmOrigin,
        outputs_origin: MutOrigin,
        //,
        carry_origin: MutOrigin = MutUntrackedOrigin,
    ](
        self,
        executor: Executor,
        queries: MatrixView[queries_origin],
        values: MatrixView[values_origin],
        outputs: MatrixSpan[outputs_origin],
        mut stats: TraversalStats,
        carry: Optional[Pointer[SoftmaxState, carry_origin]] = None,
        config: AttentionConfig = AttentionConfig(),
    ) raises Error:
        """Softmax-weighted attention over the indexed keys, folded into the walk in one pass.

        The index holds the keys and `values` holds the value rows, addressed by key: an entry's
        key is its row offset into `values`. Token positions already are such keys, so a context
        inserted in position order needs no mapping, and the publication watermark becomes the
        causal mask. The recurrence is Flash Attention's online softmax, so `outputs` comes back
        already divided by its partition; what differs is the sweep, which walks the graph and
        scores a magnitude-chosen subset rather than every key tile.

        `carry` is that recurrence's `m` and `l`, one `SoftmaxState` per query row: supplying it
        folds this branch into another over the same queries - a dense recent window, or a second
        shard of one context - by partial-softmax rescaling.

        The value table crosses whole, so `values.rows` bounds which keys may contribute and a key
        past it is refused rather than silently skipped.

        Parameters:
            queries_origin: The query rows' origin, inferred from the argument.
            values_origin: The value rows' origin, inferred with it.
            outputs_origin: The landing rows' origin, inferred with it.
            carry_origin: The softmax states' origin, inferred with it where any are supplied.

        Args:
            executor: The executor driving the batch; its kind must match the handle's residency.
            queries: The query rows; every query head is one more row.
            values: The value rows, whose column count is independent of the key dimensionality.
            outputs: Where the attended rows land, one per query.
            stats: Receives this call's traversal telemetry, fully overwritten.
            carry: The online-softmax states, one per query row; omitted skips them.
            config: Per-call breadth, logit scale, and stopping mass; the default derives all three.

        Raises:
            `Error` if the outputs cannot hold one row per query or are not as wide as the value
            rows, and with `CONFIG_MISMATCH` if the graph was frozen under a distance kind.
        """
        if outputs.capacity < queries.rows or outputs.columns != values.columns:
            raise Error(Status.SHAPE_MISMATCH, "usearch_dense_attend", NULL_ADDRESS)
        var report = stack_allocation[1, ErrorSlot]()
        report[unsafe_offset=0] = ErrorSlot()
        var slot = report.unsafe_origin_cast[MutAnyOrigin]()
        var settings = stack_allocation[1, AttentionConfig]()
        settings[unsafe_offset=0] = config
        var queries_view = stack_allocation[1, MatrixView[ImmUntrackedOrigin]]()
        queries_view[unsafe_offset=0] = queries.untracked()
        var values_view = stack_allocation[1, MatrixView[ImmUntrackedOrigin]]()
        values_view[unsafe_offset=0] = values.untracked()
        var outputs_span = stack_allocation[1, MatrixSpan[MutUntrackedOrigin]]()
        outputs_span[unsafe_offset=0] = outputs.untracked()
        var states = Int(carry.value()) if carry else NULL_ADDRESS
        self.library.symbols().dense_attend(
            self.live(),
            executor.handle,
            queries_view.unsafe_origin_cast[MutAnyOrigin](),
            values_view.unsafe_origin_cast[MutAnyOrigin](),
            outputs_span.unsafe_origin_cast[MutAnyOrigin](),
            states,
            settings.unsafe_origin_cast[MutAnyOrigin](),
            NULL_ADDRESS,
            NULL_ADDRESS,
            Pointer(to=stats).unsafe_origin_cast[MutAnyOrigin](),
            slot,
        )
        Error.raise_if_failed(slot, "usearch_dense_attend")

    def maxsim[
        queries_origin: ImmOrigin,
        found_keys_origin: MutOrigin,
        found_scores_origin: MutOrigin,
        found_counts_origin: MutOrigin,
        //,
    ](
        self,
        executor: Executor,
        queries: MatrixView[queries_origin],
        rows_per_query: Int,
        matches: MatchesSpan[found_keys_origin, found_scores_origin, found_counts_origin],
        mut stats: TraversalStats,
        config: ExpansionConfig = ExpansionConfig(),
    ) raises Error:
        """Late-interaction scoring: every query row takes its best match within a key, and those
        maxima sum to that key's score.

        This is what `values_per_key` above one exists for. A document is one key holding its token
        rows, and the walk stays over individual tokens under a plain inner product - no
        set-to-set metric is ever evaluated, which is why an entry's rows need no ordering and two
        documents need not share a row count. Approximate in the usual direction: a key can score
        below its exact value but never above it.

        `queries.rows` carries the total row count, from which the engine derives the query count
        by dividing out `rows_per_query`.

        Parameters:
            queries_origin: The query rows' origin, inferred from the argument.
            found_keys_origin: The landing keys' origin, inferred with it.
            found_scores_origin: The landing scores' origin, inferred with it.
            found_counts_origin: The landing counts' origin, inferred with it.

        Args:
            executor: The executor to fan out over; its kind must match the residency.
            queries: The query rows, queries laid end to end.
            rows_per_query: Rows in every query; uniform, so ColBERT's padding applies as is.
            matches: Where the keys, summed maxima, and per-query key counts land.
            stats: Receives this call's traversal telemetry, fully overwritten.
            config: Per-call breadth; the default derives every setting.

        Raises:
            `Error` with `CONFIG_MISMATCH` if the graph was frozen under a distance kind.
        """
        var report = stack_allocation[1, ErrorSlot]()
        report[unsafe_offset=0] = ErrorSlot()
        var slot = report.unsafe_origin_cast[MutAnyOrigin]()
        if rows_per_query <= 0 or Int(queries.rows) % rows_per_query != 0:
            raise Error(Status.SHAPE_MISMATCH, "usearch_dense_maxsim", NULL_ADDRESS)
        var settings = stack_allocation[1, ExpansionConfig]()
        settings[unsafe_offset=0] = config
        var view = stack_allocation[1, MatrixView[ImmUntrackedOrigin]]()
        view[unsafe_offset=0] = queries.untracked()
        var landing = stack_allocation[1, MatchesSpan[MutUntrackedOrigin, MutUntrackedOrigin, MutUntrackedOrigin]]()
        landing[unsafe_offset=0] = matches.untracked()
        self.library.symbols().dense_maxsim(
            self.live(),
            executor.handle,
            view.unsafe_origin_cast[MutAnyOrigin](),
            c_size_t(rows_per_query),
            landing.unsafe_origin_cast[MutAnyOrigin](),
            settings.unsafe_origin_cast[MutAnyOrigin](),
            NULL_ADDRESS,
            NULL_ADDRESS,
            Pointer(to=stats).unsafe_origin_cast[MutAnyOrigin](),
            slot,
        )
        Error.raise_if_failed(slot, "usearch_dense_maxsim")

    # endregion Linear Algebra

    # region Streams

    def cursor(self) raises Error -> DenseCursor:
        """Opens a snapshot cursor over the current contents.

        Returns:
            A cursor pinned to the publication watermark this call captured, so concurrent inserts
            never appear mid-iteration.

        Raises:
            `Error` if the snapshot cannot be taken.
        """
        var report = stack_allocation[1, ErrorSlot]()
        report[unsafe_offset=0] = ErrorSlot()
        var slot = report.unsafe_origin_cast[MutAnyOrigin]()
        var address = self.library.symbols().dense_cursor(self.live(), slot)
        Error.raise_if_failed(slot, "usearch_dense_cursor")
        if address == 0:
            raise Error(Status.INTERNAL, "usearch_dense_cursor", 0)
        return DenseCursor(library=self.library, handle=Handle(unsafe_from_address=address))

    def find[
        query_origin: ImmOrigin, //
    ](self, executor: Executor, query: MatrixView[query_origin]) raises Error -> DenseFinder:
        """Opens a resumable nearest-neighbor stream for one query.

        A live finder holds one of the executor's threads for as long as it exists, so an executor
        with few threads and many open finders starves. Release each one the moment its caller
        stops pulling, rather than at the end of the enclosing scope.

        Parameters:
            query_origin: The query row's origin, inferred from the argument.

        Args:
            executor: The executor the stream draws from; one of its threads stays pinned.
            query: The single query row; a matrix of more rows has its first row read.

        Returns:
            A finder yielding matches lazily off a live frontier.

        Raises:
            `Error` if the executor cannot spare a thread, or the shapes disagree.
        """
        var report = stack_allocation[1, ErrorSlot]()
        report[unsafe_offset=0] = ErrorSlot()
        var slot = report.unsafe_origin_cast[MutAnyOrigin]()
        var view = stack_allocation[1, MatrixView[ImmUntrackedOrigin]]()
        view[unsafe_offset=0] = query.untracked()
        var address = self.library.symbols().dense_find(
            self.live(), executor.handle, view.unsafe_origin_cast[MutAnyOrigin](), slot
        )
        Error.raise_if_failed(slot, "usearch_dense_find")
        if address == 0:
            raise Error(Status.INTERNAL, "usearch_dense_find", 0)
        return DenseFinder(library=self.library, handle=Handle(unsafe_from_address=address))

    # endregion Streams


@fieldwise_init
struct DenseCursor:
    """A consistent snapshot of the index contents - `usearch_dense_cursor_t`.

    Captures the publication watermark at creation, so concurrent inserts never appear
    mid-iteration, and the walk is a sequential row scan. Export, migration, and visualization feed
    off it.
    """

    var library: Library
    """The loaded engine, held so the release can still reach `usearch_dense_cursor_free`."""
    var handle: Handle
    """The `usearch_dense_cursor_t` this struct owns, non-null for as long as it lives."""

    def __deinit__(deinit self):
        """Releases the cursor and the snapshot it pinned.

        The loader reference is surrendered only after the call: reading the entry point is the
        last use of `library`, so without the trailing move Mojo drops the last reference - and
        `dlclose`s the engine - between the load and the branch to it.
        """
        self.library.symbols().dense_cursor_free(self.handle)
        _ = self.library^

    def next[
        keys_origin: MutOrigin, vectors_origin: MutOrigin, //
    ](self, keys: Span[Key, keys_origin], vectors: MatrixSpan[vectors_origin]) raises Error -> Int:
        """Copies the next batch of entries out of the snapshot.

        The key run bounds the batch, so a vector export must have room for as many rows.

        Parameters:
            keys_origin: The key run's origin, inferred from the argument.
            vectors_origin: The landing rows' origin, inferred with it.

        Args:
            keys: Where the keys land, whose length is the batch size this call may write.
            vectors: Where the rows land, of at least as many rows as `keys` holds.

        Returns:
            How many rows were written; zero when the snapshot is exhausted.

        Raises:
            `Error` if the engine refuses the copy.
        """
        var report = stack_allocation[1, ErrorSlot]()
        report[unsafe_offset=0] = ErrorSlot()
        var slot = report.unsafe_origin_cast[MutAnyOrigin]()
        var span = stack_allocation[1, MatrixSpan[MutUntrackedOrigin]]()
        span[unsafe_offset=0] = vectors.untracked()
        var written = self.library.symbols().dense_cursor_next(
            self.handle,
            KeysSpan.of(keys).untracked(),
            span.unsafe_origin_cast[MutAnyOrigin](),
            slot,
        )
        Error.raise_if_failed(slot, "usearch_dense_cursor_next")
        return Int(written)


@fieldwise_init
struct DenseFinder:
    """A resumable nearest-neighbor stream - `usearch_dense_finder_t`.

    Best-first traversal is naturally pausable, so the finder owns a live frontier and visited set
    and yields matches lazily. Callers that post-filter aggressively pull until satisfied instead
    of guessing "k" upfront, at the price of pinning one of the executor's threads until released.

    Ordering beyond the expansion horizon is best-effort nearest rather than globally exact, so
    pulling far past the configured breadth sees the ranking degrade rather than the stream end.
    """

    var library: Library
    """The loaded engine, held so the release can still reach `usearch_dense_find_free`."""
    var handle: Handle
    """The `usearch_dense_finder_t` this struct owns, non-null for as long as it lives."""

    def __deinit__(deinit self):
        """Releases the finder and the traversal context it pinned.

        The loader reference is surrendered only after the call: reading the entry point is the
        last use of `library`, so without the trailing move Mojo drops the last reference - and
        `dlclose`s the engine - between the load and the branch to it.
        """
        self.library.symbols().dense_find_free(self.handle)
        _ = self.library^

    def next[
        keys_origin: MutOrigin, distances_origin: MutOrigin, //
    ](self, keys: Span[Key, keys_origin], distances: Span[Distance, distances_origin]) raises Error -> Int:
        """Pulls the next matches off the live frontier.

        Parameters:
            keys_origin: The key run's origin, inferred from the argument.
            distances_origin: The distance run's origin, inferred with it.

        Args:
            keys: Where the keys land, which bounds how many this call may write.
            distances: Where the distances land, paired positionally with the keys.

        Returns:
            How many matches were written; zero when the traversal is drained.

        Raises:
            `Error` if the engine refuses to advance the stream.
        """
        if len(keys) != len(distances):
            raise Error(Status.SHAPE_MISMATCH, "usearch_dense_find_next", NULL_ADDRESS)
        var report = stack_allocation[1, ErrorSlot]()
        report[unsafe_offset=0] = ErrorSlot()
        var slot = report.unsafe_origin_cast[MutAnyOrigin]()
        var written = self.library.symbols().dense_find_next(
            self.handle,
            KeysSpan.of(keys).untracked(),
            ScoresOut(unsafe_from_address=Int(distances.unsafe_ptr())),
            slot,
        )
        Error.raise_if_failed(slot, "usearch_dense_find_next")
        return Int(written)


# endregion Index

# region Metadata


def metadata[
    path_origin: ImmOrigin, //
](library: Library, path: BytesView[path_origin]) raises Error -> DenseOptions[ImmUntrackedOrigin]:
    """Reads a checkpoint's geometry and metric without loading it.

    A manifest answers from its summary and its first part's header, so the cost does not grow with
    the index. The placement fields come back untouched, because a checkpoint records where it was
    written and never where a reader must keep it.

    Parameters:
        path_origin: The path bytes' origin, inferred from the argument.

    Args:
        library: The loaded engine that reads the header.
        path: The manifest of a sharded checkpoint, or a single monolith.

    Returns:
        The geometry and metric the checkpoint froze, with placement left to the caller.

    Raises:
        `Error` if the checkpoint cannot be opened or its header cannot be decoded.
    """
    var report = stack_allocation[1, ErrorSlot]()
    report[unsafe_offset=0] = ErrorSlot()
    var slot = report.unsafe_origin_cast[MutAnyOrigin]()
    var options = stack_allocation[1, DenseOptions[ImmUntrackedOrigin]]()
    options[unsafe_offset=0] = DenseOptions(Metric.UNKNOWN, ScalarKind.UNKNOWN, 0)
    library.symbols().dense_metadata(path.untracked(), options.unsafe_origin_cast[MutAnyOrigin](), slot)
    Error.raise_if_failed(slot, "usearch_dense_metadata")
    return options[unsafe_offset=0]


def metadata_buffer[
    buffer_origin: ImmOrigin, //
](library: Library, buffer: BytesView[buffer_origin]) raises Error -> DenseOptions[ImmUntrackedOrigin]:
    """Reads a serialized monolith's geometry and metric without loading it.

    Parameters:
        buffer_origin: The source bytes' origin, inferred from the argument.

    Args:
        library: The loaded engine that reads the header.
        buffer: Source holding one serialized monolith.

    Returns:
        The geometry and metric the monolith froze, with placement left to the caller.

    Raises:
        `Error` if the bytes cannot be decoded.
    """
    var report = stack_allocation[1, ErrorSlot]()
    report[unsafe_offset=0] = ErrorSlot()
    var slot = report.unsafe_origin_cast[MutAnyOrigin]()
    var options = stack_allocation[1, DenseOptions[ImmUntrackedOrigin]]()
    options[unsafe_offset=0] = DenseOptions(Metric.UNKNOWN, ScalarKind.UNKNOWN, 0)
    library.symbols().dense_metadata_buffer(buffer.untracked(), options.unsafe_origin_cast[MutAnyOrigin](), slot)
    Error.raise_if_failed(slot, "usearch_dense_metadata_buffer")
    return options[unsafe_offset=0]


# endregion Metadata
