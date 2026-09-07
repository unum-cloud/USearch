"""The Mojo binding's tests, in the shape `c/test.c` already uses.

Mojo 1.0 has no `mojo test` subcommand and no test discovery, but `std.testing.TestSuite` is a real
runner: register each check as a parameter and it reports and filters them. The pure-logic group
needs no engine loaded; the rest exercise the C ABI directly and are skipped when nothing is on the
loader path, so this file compiles and runs before `libusearch_c` does.
"""

from std.testing import TestSuite, assert_equal, assert_true

from usearch import (
    AttentionConfig,
    DenseIndex,
    DenseOptions,
    Distance,
    Executor,
    ExpansionConfig,
    Key,
    Library,
    MatchesSpan,
    MatrixSpan,
    MatrixView,
    Metric,
    RamOptions,
    Residency,
    ScalarKind,
    SoftmaxState,
    TraversalStats,
    VramOptions,
)

# region Pure Logic


def test_scalar_widths_round_sub_byte_rows_up() raises:
    """A packed row is measured in whole bytes, however few bits a dimension takes."""
    assert_equal(ScalarKind.F32.bytes_for_columns(96), 384, "f32 is four bytes a dimension")
    assert_equal(ScalarKind.U1X8.bytes_for_columns(96), 12, "eight bit dimensions share a byte")
    assert_equal(ScalarKind.U1X8.bytes_for_columns(1), 1, "a single bit still costs a byte")
    assert_equal(ScalarKind.U4X2.bytes_for_columns(3), 2, "an odd nibble count rounds up")


def test_keys_carry_integer_identities() raises:
    """An integer identity occupies bytes 8 through 15, most significant byte first.

    Asserted against the bytes rather than through `as_integer`, which would agree with itself
    whichever half the identity landed in - the engine and every sibling binding read byte 15.
    """
    var key = Key(integer=42)
    assert_equal(key.as_integer(), UInt64(42), "the identity round-trips")
    assert_equal(key.leading, UInt64(0), "the leading bytes stay zeroed")
    var bytes = Pointer(to=key).unsafe_bitcast[UInt8]()
    assert_equal(bytes[unsafe_offset=15], UInt8(42), "the identity's low byte is the key's last")
    for index in range(15):
        assert_equal(bytes[unsafe_offset=index], UInt8(0), "every other byte stays zeroed")
    assert_true(Key(integer=42) == Key(integer=42), "two keys of one identity compare equal")


def test_matrices_derive_contiguous_strides() raises:
    """A flat list read as a matrix reports the row count its width implies."""
    var rows = [Float32(1), 2, 3, 4, 5, 6]
    var view = MatrixView.of(rows, 3, ScalarKind.F32)
    assert_equal(Int(view.rows), 2, "six scalars of width three are two rows")
    assert_equal(Int(view.columns), 3, "the width is carried verbatim")
    assert_equal(Int(view.stride_bytes), 0, "a zero stride means contiguous")


def test_configurations_derive_when_zeroed() raises:
    """Every optional setting left at zero is one the engine derives."""
    assert_equal(Int(ExpansionConfig().expansion_factor), 0, "breadth derives")
    assert_equal(Int(AttentionConfig().expansion_factor), 0, "attention breadth derives")
    assert_equal(AttentionConfig().score_scale, Float32(0), "the logit scale derives")
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


# endregion Pure Logic

# region Engine


def _round_trip(library: Library) raises:
    """Builds a three-vector index, searches it, and filters the same walk."""
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
    assert_equal(found_keys[0].as_integer(), UInt64(0), "the identical row ranks first")

    var exported = List[Float32](length=3, fill=0)
    var written = index.get(Key(integer=0), MatrixSpan.of(exported, 3, ScalarKind.F32))
    assert_equal(written, 1, "one entry sits under that key")
    assert_equal(index.count(Key(integer=0)), 1, "and the count agrees")


def _engine_is_present() -> Bool:
    """Whether anything answers on the loader path, asked once so a skip cannot hide a failure."""
    try:
        var library = Library()
        return True
    except:
        return False


def test_engine_round_trips_when_present() raises:
    """Skipped rather than failed when the engine is not on the loader path."""
    if not _engine_is_present():
        print("skipped: libusearch_c is not on the loader path")
        return
    var library = Library()
    assert_true(library.version().byte_length() != 0, "the engine names its own version")
    _round_trip(library)


# endregion Engine

# region Linear Algebra


def _basis(library: Library, executor: Executor) raises -> DenseIndex:
    """Builds the two-row inner-product index over the identity basis every verb below scores against.

    A query of `[1, 0]` then has a logit of one against the first row and zero against the second.

    Args:
        library: The loaded engine, kept so the shared object stays mapped.
        executor: The executor the insertion fans out over.

    Returns:
        An index holding `[1, 0]` under key zero and `[0, 1]` under key one.
    """
    var index = DenseIndex(library, DenseOptions(Metric.INNER, ScalarKind.F32, dimensions=2, values_per_key=2))
    index.reserve(2)
    var rows = [Float32(1), 0, Float32(0), 1]
    var keys = [Key(integer=0), Key(integer=1)]
    var stats = TraversalStats()
    var published = index.add(executor, keys, MatrixView.of(rows, 2, ScalarKind.F32), stats)
    assert_equal(published, 2, "both basis rows are published")
    return index^


def test_multiply_ranks_the_aligned_row_first() raises:
    """The sparse product reports every scored pair, the aligned row leading them."""
    if not _engine_is_present():
        print("skipped: libusearch_c is not on the loader path")
        return
    var library = Library()
    var executor = Executor(library, RamOptions(), threads=2)
    var index = _basis(library, executor)

    var left = [Float32(1), 0]
    var pair_keys = List[Key](length=2, fill=Key(integer=0))
    var pair_scores = List[Distance](length=2, fill=0)
    var pair_counts = List[UInt](length=1, fill=0)
    var stats = TraversalStats()
    index.multiply(
        executor,
        MatrixView.of(left, 2, ScalarKind.F32),
        MatchesSpan.of(pair_keys, pair_scores, pair_counts),
        stats,
    )
    assert_true(pair_counts[0] != 0, "the walk scored at least one pair")
    assert_equal(pair_keys[0].as_integer(), UInt64(0), "the aligned row leads the pairs")


def test_maxsim_scores_over_documents_rather_than_rows() raises:
    """Two tokens of one query score against the indexed documents, not against each other."""
    if not _engine_is_present():
        print("skipped: libusearch_c is not on the loader path")
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
    assert_true(found_counts[0] != 0, "the one query scored at least one document")


def test_attend_writes_one_context_row_per_query() raises:
    """The context is a convex blend of the value rows, so it lands strictly between them."""
    if not _engine_is_present():
        print("skipped: libusearch_c is not on the loader path")
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
    assert_true(context[0] > Float32(10), "the blend sits above the nearer value row")
    assert_true(context[0] < Float32(30), "and below the farther one")
    assert_true(carry.partition > Float32(0), "the partition counts the mass the walk accumulated")


def test_refine_reports_settling() raises:
    """A two-row graph has nothing left to re-select, so the first pass already reports settled."""
    if not _engine_is_present():
        print("skipped: libusearch_c is not on the loader path")
        return
    var library = Library()
    var executor = Executor(library, RamOptions(), threads=2)
    var index = _basis(library, executor)

    var stats = TraversalStats()
    assert_equal(index.refine(executor, stats), 0, "the graph has already settled")


# endregion Linear Algebra


def main() raises:
    var suite = TestSuite()
    suite.test[test_scalar_widths_round_sub_byte_rows_up]()
    suite.test[test_keys_carry_integer_identities]()
    suite.test[test_matrices_derive_contiguous_strides]()
    suite.test[test_configurations_derive_when_zeroed]()
    suite.test[test_options_tag_their_placement_arm]()
    suite.test[test_engine_round_trips_when_present]()
    suite.test[test_multiply_ranks_the_aligned_row_first]()
    suite.test[test_maxsim_scores_over_documents_rather_than_rows]()
    suite.test[test_attend_writes_one_context_row_per_query]()
    suite.test[test_refine_reports_settling]()
    suite^.run()
