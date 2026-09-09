"""The Mojo binding's tests, in the shape `c/test.c` already uses.

Mojo 1.0 has no `mojo test` subcommand and no test discovery, so every check below is registered in
`main` and one that is not silently never runs. The pure-logic group needs no engine loaded; the
rest exercise the C ABI directly and are skipped when nothing is on the loader path.
"""

from std.testing import TestSuite, assert_equal, assert_false, assert_true

from usearch import (
    AttentionOptions,
    DenseIndex,
    DenseOptions,
    Distance,
    Executor,
    InsertionOptions,
    Key,
    Library,
    MatchesSpan,
    MatrixSpan,
    MatrixView,
    Metric,
    RamOptions,
    Residency,
    ScalarKind,
    SearchOptions,
    SoftmaxState,
    TraversalStats,
)

# region Pure Logic


def test_scalar_widths_round_sub_byte_rows_up() raises:
    """A packed row is measured in whole bytes, however few bits a dimension takes."""
    assert_equal(
        ScalarKind.F32.bytes_for_columns(96),
        384,
        "f32 is four bytes a dimension",
    )
    assert_equal(
        ScalarKind.U1X8.bytes_for_columns(96),
        12,
        "eight bit dimensions share a byte",
    )
    assert_equal(
        ScalarKind.U1X8.bytes_for_columns(1),
        1,
        "a single bit still costs a byte",
    )
    assert_equal(ScalarKind.U4X2.bytes_for_columns(3), 2, "an odd nibble count rounds up")


def test_keys_carry_integer_identities() raises:
    """An integer identity occupies bytes 8 through 15, most significant byte first."""
    var key = Key(integer=42)
    assert_equal(key.as_integer().value(), UInt64(42), "the identity round-trips")
    assert_equal(key.leading, UInt64(0), "the leading bytes stay zeroed")
    assert_false(
        Key(1, 0).as_integer(),
        "a key using its leading bytes has no integer identity",
    )
    var bytes = Pointer(to=key).unsafe_bitcast[UInt8]()
    assert_equal(
        bytes[unsafe_offset=15],
        UInt8(42),
        "the identity's low byte is the key's last",
    )
    for index in range(15):
        assert_equal(
            bytes[unsafe_offset=index],
            UInt8(0),
            "every other byte stays zeroed",
        )
    assert_true(
        Key(integer=42) == Key(integer=42),
        "two keys of one identity compare equal",
    )


def test_matrices_derive_contiguous_strides() raises:
    """A flat list read as a matrix reports the row count its width implies."""
    var rows = [Float32(1), 2, 3, 4, 5, 6]
    var view = MatrixView.of(rows, 3, ScalarKind.F32)
    assert_equal(Int(view.rows), 2, "six scalars of width three are two rows")
    assert_equal(Int(view.columns), 3, "the width is carried verbatim")
    assert_equal(Int(view.stride_bytes), 0, "a zero stride means contiguous")

    var keys = List[Key](length=6, fill=Key(integer=0))
    var scores = List[Distance](length=6, fill=0)
    var counts = List[UInt](length=3, fill=0)
    var matches = MatchesSpan.of(keys, scores, counts)
    assert_equal(
        Int(matches.wanted),
        2,
        "three rows over six slots want two matches apiece",
    )


def test_shapes_the_lengths_cannot_support_are_refused() raises:
    """A width that does not divide its run is refused here rather than divided by."""
    var rows = [Float32(1), 2, 3, 4, 5, 6]
    var keys = List[Key](length=6, fill=Key(integer=0))
    var scores = List[Distance](length=6, fill=0)
    var counts = List[UInt](length=3, fill=0)
    var short_scores = List[Distance](length=5, fill=0)
    var wide_counts = List[UInt](length=4, fill=0)

    var refused = False
    try:
        _ = MatrixView.of(rows, 4, ScalarKind.F32)
    except:
        refused = True
    assert_true(refused, "six scalars are no whole rows of four")

    refused = False
    try:
        _ = MatrixSpan.of(rows, 0, ScalarKind.F32)
    except:
        refused = True
    assert_true(refused, "a zero width would be a division")

    refused = False
    try:
        _ = MatchesSpan.of(keys, scores, wide_counts)
    except:
        refused = True
    assert_true(refused, "six slots make no rectangle over four rows")

    refused = False
    try:
        _ = MatchesSpan.of(keys, short_scores, counts)
    except:
        refused = True
    assert_true(refused, "the keys and the scores must be one shape")


def test_configurations_derive_when_zeroed() raises:
    """Every optional setting left at zero is one the engine derives."""
    assert_equal(Int(InsertionOptions().expansion_factor), 0, "insertion breadth derives")
    assert_equal(Int(SearchOptions().expansion_factor), 0, "search breadth derives")
    assert_equal(Int(AttentionOptions().expansion_factor), 0, "attention breadth derives")
    assert_equal(AttentionOptions().score_scale, Float32(0), "the logit scale derives")
    var stats = TraversalStats()
    assert_equal(Int(stats.visited_members), 0, "telemetry starts at an untaken walk")


def test_options_tag_their_placement_arm() raises:
    """Naming a placement is what fixes the residency the union arm belongs to."""
    var options = DenseOptions(Metric.ANGULAR, ScalarKind.F32, dimensions=96)
    assert_true(options.residency == Residency.RAM, "the default residency is RAM")
    var devices = [Int32(0), Int32(1)]
    var placed = options.on_devices(devices)
    assert_true(placed.residency == Residency.VRAM, "naming devices moves it to VRAM")
    assert_equal(Int(placed.placement_count), 2, "both ordinals are named")
    var domains = [Int32(0)]
    assert_true(
        options.on_memory_domains(domains).residency == Residency.RAM,
        "naming domains keeps it in RAM",
    )


# endregion Pure Logic

# region Engine


def _engine_is_present() -> Bool:
    """Whether anything answers on the loader path, asked once so a skip cannot hide a failure."""
    try:
        _ = Library()
        return True
    except:
        return False


def _round_trip(library: Library) raises:
    """Builds a three-vector index, searches it, and reads one row back out."""
    var executor = Executor(library, RamOptions(), threads=2)
    var index = DenseIndex(library, DenseOptions(Metric.ANGULAR, ScalarKind.F32, dimensions=3))
    index.reserve(3)

    var rows = [Float32(1), 0, 0, Float32(0), 1, 0, Float32(0), 0, 1]
    var keys = [Key(integer=0), Key(integer=1), Key(integer=2)]
    var stats = TraversalStats()
    var published = index.add(executor, keys, MatrixView.of(rows, 3, ScalarKind.F32), stats)
    assert_equal(published, 3, "every row is published when nothing failed")
    assert_equal(index.size(), 3, "the index reports what it stored")
    assert_equal(index.dimensions(), 3, "the geometry froze at three")

    var query = [Float32(1), 0, 0]
    var found_keys = List[Key](length=2, fill=Key(integer=0))
    var found_scores = List[Distance](length=2, fill=0)
    var found_counts = List[UInt](length=1, fill=0)
    var matches = MatchesSpan.of(found_keys, found_scores, found_counts)
    index.search(executor, MatrixView.of(query, 3, ScalarKind.F32), matches, stats)
    assert_equal(Int(found_counts[0]), 2, "the one query filled its own output row")
    assert_true(Int(stats.computed_distances) != 0, "the walk reported what it computed")

    var exported = List[Float32](length=3, fill=0)
    var written = index.get(Key(integer=0), MatrixSpan.of(exported, 3, ScalarKind.F32))
    assert_equal(written, 1, "one entry sits under that key")
    assert_equal(index.count(Key(integer=0)), 1, "and the count agrees")
    assert_equal(
        exported[0],
        Float32(1),
        "the stored row comes back through the span it lends",
    )

    # A query the frozen geometry disagrees with is refused rather than read past its end.
    var narrow = [Float32(1), 0]
    var refused = False
    try:
        index.search(executor, MatrixView.of(narrow, 2, ScalarKind.F32), matches, stats)
    except:
        refused = True
    assert_true(refused, "a two-wide query cannot search a three-wide index")


def test_engine_round_trips_when_present() raises:
    """Skipped rather than failed when the engine is not on the loader path."""
    if not _engine_is_present():
        print("skipped: libusearch_shared is not on the loader path")
        return
    var library = Library()
    assert_true(library.version().byte_length() != 0, "the engine names its own version")
    _round_trip(library)


def test_get_reports_the_true_extent_when_truncated() raises:
    """Get reports what exists, never what it wrote, so a caller sizes from the answer."""
    if not _engine_is_present():
        print("skipped: libusearch_shared is not on the loader path")
        return
    var library = Library()
    var executor = Executor(library, RamOptions(), threads=2)
    var index = DenseIndex(
        library,
        DenseOptions(Metric.ANGULAR, ScalarKind.F32, dimensions=2, values_per_key=3),
    )
    index.reserve(3)

    var rows = [Float32(1), 0, Float32(2), 0, Float32(3), 0]
    var keys = [Key(integer=7), Key(integer=7), Key(integer=7)]
    var stats = TraversalStats()
    _ = index.add(executor, keys, MatrixView.of(rows, 2, ScalarKind.F32), stats)
    assert_equal(index.count(Key(integer=7)), 3, "three entries share the one key")

    var cramped = List[Float32](length=2, fill=0)
    var reported = index.get(Key(integer=7), MatrixSpan.of(cramped, 2, ScalarKind.F32))
    assert_equal(reported, 3, "a one-row span still reports the true extent")

    var roomy = List[Float32](length=6, fill=0)
    assert_equal(
        index.get(Key(integer=7), MatrixSpan.of(roomy, 2, ScalarKind.F32)),
        3,
        "sized from that answer",
    )


# endregion Engine

# region Linear Algebra


def _basis(library: Library, executor: Executor) raises -> DenseIndex:
    """Builds the two-row inner-product index over the identity basis every verb below scores against.

    Args:
        library: The loaded engine, kept so the shared object stays mapped.
        executor: The executor the insertion fans out over.

    Returns:
        An index holding `[1, 0]` under key zero and `[0, 1]` under key one.
    """
    var index = DenseIndex(
        library,
        DenseOptions(Metric.INNER, ScalarKind.F32, dimensions=2, values_per_key=2),
    )
    index.reserve(2)
    var rows = [Float32(1), 0, Float32(0), 1]
    var keys = [Key(integer=0), Key(integer=1)]
    var stats = TraversalStats()
    var published = index.add(executor, keys, MatrixView.of(rows, 2, ScalarKind.F32), stats)
    assert_equal(published, 2, "both basis rows are published")
    return index^


def test_multiply_reports_every_scored_pair() raises:
    """The sparse product counts every pair it scored, which may outrun the room it was given."""
    if not _engine_is_present():
        print("skipped: libusearch_shared is not on the loader path")
        return
    var library = Library()
    var executor = Executor(library, RamOptions(), threads=2)
    var index = _basis(library, executor)

    var left = [Float32(1), 0]
    var pair_keys = List[Key](length=1, fill=Key(integer=0))
    var pair_scores = List[Distance](length=1, fill=0)
    var pair_counts = List[UInt](length=1, fill=0)
    var stats = TraversalStats()
    index.multiply(
        executor,
        MatrixView.of(left, 2, ScalarKind.F32),
        MatchesSpan.of(pair_keys, pair_scores, pair_counts),
        stats,
    )
    assert_equal(
        Int(pair_counts[0]),
        2,
        "both columns are counted, though one slot was reserved",
    )


def test_maxsim_scores_over_documents_rather_than_rows() raises:
    """Two tokens of one query score against the indexed documents, not against each other."""
    if not _engine_is_present():
        print("skipped: libusearch_shared is not on the loader path")
        return
    var library = Library()
    var executor = Executor(library, RamOptions(), threads=2)
    var index = _basis(library, executor)

    var tokens = [Float32(1), 0, Float32(0), 1]
    var found_keys = List[Key](length=2, fill=Key(integer=0))
    var found_scores = List[Distance](length=2, fill=0)
    var found_counts = List[UInt](length=1, fill=0)
    var stats = TraversalStats()
    index.maxsim(
        executor,
        MatrixView.of(tokens, 2, ScalarKind.F32),
        2,
        MatchesSpan.of(found_keys, found_scores, found_counts),
        stats,
    )
    assert_equal(
        Int(found_counts[0]),
        2,
        "two token rows are one query over two documents",
    )


def test_attend_writes_one_context_row_per_query() raises:
    """The context row and the carry entry are written through the descriptors the caller lends."""
    if not _engine_is_present():
        print("skipped: libusearch_shared is not on the loader path")
        return
    var library = Library()
    var executor = Executor(library, RamOptions(), threads=2)
    var index = _basis(library, executor)

    var queries = [Float32(1), 0]
    var values = [Float32(10), 20, Float32(30), 40]
    var context = List[Float32](length=2, fill=0)
    var carry = SoftmaxState()
    var stats = TraversalStats()
    index.attend(
        executor,
        MatrixView.of(queries, 2, ScalarKind.F32),
        MatrixView.of(values, 2, ScalarKind.F32),
        MatrixSpan.of(context, 2, ScalarKind.F32),
        stats,
        Pointer(to=carry),
    )
    assert_true(
        carry.partition > Float32(0),
        "the partition counts the mass the walk accumulated",
    )


# endregion Linear Algebra


def main() raises:
    var suite = TestSuite()
    suite.test[test_scalar_widths_round_sub_byte_rows_up]()
    suite.test[test_keys_carry_integer_identities]()
    suite.test[test_matrices_derive_contiguous_strides]()
    suite.test[test_shapes_the_lengths_cannot_support_are_refused]()
    suite.test[test_configurations_derive_when_zeroed]()
    suite.test[test_options_tag_their_placement_arm]()
    suite.test[test_engine_round_trips_when_present]()
    suite.test[test_get_reports_the_true_extent_when_truncated]()
    suite.test[test_multiply_reports_every_scored_pair]()
    suite.test[test_maxsim_scores_over_documents_rather_than_rows]()
    suite.test[test_attend_writes_one_context_row_per_query]()
    suite^.run()
