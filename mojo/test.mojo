"""The Mojo binding's tests, in the shape `c/test.c` already uses.

Mojo 1.1 has no `mojo test` subcommand and no test discovery, so every check below is registered in
`main` and one that is not silently never runs. The pure-logic group needs no engine loaded; the
rest exercise the C ABI directly and are skipped when nothing is on the loader path.
"""

from std.ffi import c_size_t
from std.math import exp, sqrt
from std.memory import stack_allocation
from std.testing import TestSuite, assert_almost_equal, assert_equal, assert_false, assert_true

from usearch import (
    Address,
    Arena,
    DenseAttendConfig,
    DenseIndex,
    DenseInsertConfig,
    DenseMetric,
    DenseMultiplyConfig,
    DenseOptions,
    DenseSearchConfig,
    DenseSoftmaxState,
    DenseTraversalStats,
    Error,
    Executor,
    Key,
    Library,
    Mood,
    NULL_ADDRESS,
    Residency,
    RowsView,
    ScalarKind,
    Status,
    Traversal,
    dense_metadata_read_buffer,
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


def test_scalar_kinds_follow_their_element_types() raises:
    """A Mojo element type names the kind it stores one-to-one, and reads only rows it can hold whole."""
    assert_true(ScalarKind.of[DType.float32]() == ScalarKind.F32, "float32 stores f32")
    assert_true(ScalarKind.of[DType.bfloat16]() == ScalarKind.BF16, "bfloat16 stores bf16")
    assert_true(ScalarKind.of[DType.float8_e4m3fn]() == ScalarKind.E4M3, "the FP8 dtypes store FP8")
    assert_true(ScalarKind.of[DType.int32]() == ScalarKind.I32, "int32 stores i32")
    assert_true(ScalarKind.of[DType.int64]() == ScalarKind.UNKNOWN, "int64 stores no kind")
    assert_true(ScalarKind.F16.stores[DType.float16](), "f16 rows read as float16")
    assert_true(ScalarKind.U1X8.stores[DType.uint8](), "packed bits read as bytes")
    assert_false(ScalarKind.U8.stores[DType.float32](), "bytes never read as floats")


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


def test_keys_render_as_dashed_uuids() raises:
    """A key prints as the lowercase dashed UUID the C header calls canonical."""
    assert_equal(String(Key(integer=42)), "00000000-0000-0000-0000-00000000002a", "an integer identity")
    assert_equal(String(Key(1, 0)), "01000000-0000-0000-0000-000000000000", "a leading byte")


def test_rows_derive_contiguous_strides() raises:
    """A flat list read as rows reports the row count its width implies."""
    var rows = [Float32(1), 2, 3, 4, 5, 6]
    var view = RowsView.of(rows, 3)
    assert_equal(Int(view.rows), 2, "six scalars of width three are two rows")
    assert_equal(Int(view.columns), 3, "the width is carried verbatim")
    assert_equal(Int(view.stride), 12, "the stride is the natural row width, since zero repeats a row")
    assert_true(view.scalar == ScalarKind.F32, "the kind follows the element type")


def test_shapes_the_lengths_cannot_support_are_refused() raises:
    """A width that does not divide its run is refused here rather than divided by."""
    var rows = [Float32(1), 2, 3, 4, 5, 6]
    var status = Status.SUCCESS
    try:
        _ = RowsView.of(rows, 4)
    except error:
        status = error.status
    assert_true(status == Status.SHAPE_MISMATCH, "six scalars are no whole rows of four")

    status = Status.SUCCESS
    try:
        _ = RowsView.of(rows, 0)
    except error:
        status = error.status
    assert_true(status == Status.SHAPE_MISMATCH, "a zero width would be a division")


def test_telemetry_starts_at_an_untaken_walk() raises:
    """An unwritten telemetry slot reads as a walk that never ran."""
    var stats = DenseTraversalStats()
    assert_equal(Int(stats.visited_members), 0, "nothing visited")
    assert_equal(Int(stats.computed_distances), 0, "nothing computed")



# endregion Pure Logic

# region Engine


def _engine_is_present() raises -> Bool:
    """Whether anything answers on the loader path; a library missing a symbol fails rather than skips."""
    try:
        _ = Library()
        return True
    except error:
        if error.status != Status.LIBRARY_MISSING:
            raise error
        return False


def _basis(library: Library) raises -> DenseIndex:
    """Builds the two-row inner-product index over the identity basis most verbs below score against.

    Args:
        library: The loaded engine, kept so the shared object stays mapped.

    Returns:
        An index holding `[1, 0]` under key zero and `[0, 1]` under key one.
    """
    var index = DenseIndex.init_ram(library, DenseOptions(2, DenseMetric.INNER, values_per_key=2))
    index.reserve(2)
    var rows = [Float32(1), 0, Float32(0), 1]
    var keys = [Key(integer=0), Key(integer=1)]
    index.insert(keys, RowsView.of(rows, 2))
    return index^


def test_engine_round_trips_when_present() raises:
    """Builds a three-vector index, searches it, and reads one row back out."""
    if not _engine_is_present():
        print("skipped: libusearch_shared is not on the loader path")
        return
    var library = Library()
    assert_equal(library.version_name(), "3.0.0", "the engine names its own version")
    var executor = Executor.for_ram(library, threads=2)
    var index = DenseIndex.init_ram(library, DenseOptions(3))
    var on = index.on(executor)
    on.reserve(3)

    var rows = [Float32(1), 0, 0, Float32(0), 1, 0, Float32(0), 0, 1]
    var keys = [Key(integer=0), Key(integer=1), Key(integer=2)]
    var stats = DenseTraversalStats()
    on.insert(keys, RowsView.of(rows, 3), DenseInsertConfig(stats=Pointer(to=stats)))
    var description = on.describe()
    assert_equal(Int(description.size), 3, "the index reports what it stored")
    assert_equal(Int(description.metadata.dimensions), 3, "the geometry froze at three")
    assert_true(description.metadata.quantization == ScalarKind.BF16, "an index left alone stores bf16")
    assert_true(description.metadata.metric == DenseMetric.ANGULAR, "and scores angular distances")
    assert_true(description.capabilities.byte_length() != 0, "the dispatch codename crosses as text")

    var query = [Float32(1), 0, 0]
    var arena = Arena(executor)
    var matches = on.search(RowsView.of(query, 3), 2, arena, DenseSearchConfig(stats=Pointer(to=stats)))
    assert_equal(len(matches), 1, "one query answered")
    assert_equal(len(matches.row_keys(0)), 2, "the one query filled its own output row")
    assert_true(matches.row_keys(0)[0] == Key(integer=0), "the nearest row is the query itself")
    assert_equal(Int(stats.computed_distances), 3, "the walk reported what it computed")
    var counted = on.search(RowsView.of(query, 3), 0, arena)
    assert_equal(len(counted.row_keys(0)), 0, "a zero wanted lands only the counts")

    var exported = on.get[DType.float32](Key(integer=0), arena)
    assert_equal(len(exported), 1, "one entry sits under that key")
    assert_equal(len(exported[0]), 3, "three values wide")
    assert_equal(exported[0][0], Float32(1), "the stored row is read in place from the arena")
    assert_equal(on.count(Key(integer=0)), 1, "and the count agrees")
    var absent = on.get[DType.float32](Key(integer=9), arena)
    assert_equal(len(absent), 0, "an absent key exports nothing")


def test_width_mismatch_is_refused_by_the_engine() raises:
    """The binding passes the caller's width through, and the engine refuses one that disagrees."""
    if not _engine_is_present():
        print("skipped: libusearch_shared is not on the loader path")
        return
    var library = Library()
    var index = _basis(library)
    var narrow = [Float32(1), 0, 0]
    var arena = Arena(library)
    var rendered = String()
    var status = Status.SUCCESS
    try:
        _ = index.search(RowsView.of(narrow, 3), 1, arena)
    except error:
        status = error.status
        rendered = String(error)
    assert_true(status == Status.SHAPE_MISMATCH, "a three-wide query cannot search a two-wide index")
    assert_equal(
        rendered,
        "USearch: shape-mismatch [usearch_dense_search]: The rows' width differs from the frozen geometry",
        "the failure reads with the engine's own name and message",
    )


def test_names_come_from_the_engine_tables() raises:
    """Every enum spells and parses through the C name tables, which own the vocabulary."""
    if not _engine_is_present():
        print("skipped: libusearch_shared is not on the loader path")
        return
    var library = Library()
    assert_equal(library.status_name(Status.READ_ONLY), "read-only", "a status")
    assert_equal(library.status_name(Status.DEFERRED), "deferred", "the one non-final status")
    assert_equal(library.status_name(Status(99)), "unrecognized", "a code no status carries")
    assert_true(library.status_from_name("config-mismatch") == Status.CONFIG_MISMATCH, "and back")
    assert_equal(library.scalar_name(ScalarKind.BF16), "bf16", "a scalar")
    assert_true(library.scalar_from_name("u1x8") == ScalarKind.U1X8, "and back")
    assert_true(library.scalar_from_name("i32") == ScalarKind.I32, "the geospatial coordinates")
    assert_equal(library.mood_name(Mood.SPINNING), "spinning", "a mood")
    assert_true(library.mood_from_name("parked") == Mood.PARKED, "and back")
    assert_equal(library.residency_name(Residency.VRAM), "vram", "a residency")
    assert_true(library.residency_from_name("nvme") == Residency.NVME, "and back")
    assert_equal(library.traversal_name(Traversal.SCAN), "scan", "a traversal")
    assert_true(library.traversal_from_name("walk") == Traversal.WALK, "and back")
    assert_equal(library.dense_metric_name(DenseMetric.JACCARD), "jaccard", "a metric")
    assert_true(library.dense_metric_from_name("euclidean") == DenseMetric.EUCLIDEAN, "and back")


def test_get_exports_every_entry_under_a_key() raises:
    """Get sizes its own output, so every entry sharing a key comes back in one call."""
    if not _engine_is_present():
        print("skipped: libusearch_shared is not on the loader path")
        return
    var library = Library()
    var index = DenseIndex.init_ram(library, DenseOptions(2, values_per_key=3))
    index.reserve(3)

    var rows = [Float32(1), 0, Float32(2), 0, Float32(3), 0]
    var keys = [Key(integer=7), Key(integer=7), Key(integer=7)]
    index.insert(keys, RowsView.of(rows, 2))
    assert_equal(index.count(Key(integer=7)), 3, "three entries share the one key")

    var arena = Arena(library)
    var exported = index.get[DType.float32](Key(integer=7), arena)
    assert_equal(len(exported), 3, "three entries come back")
    assert_equal(exported.columns(), 2, "each as wide as the engine reports")
    assert_equal(exported[2][0], Float32(3), "in insertion order")

    var status = Status.SUCCESS
    try:
        _ = index.get[DType.float32](Key(integer=7), arena, ScalarKind.U8)
    except error:
        status = error.status
    assert_true(status == Status.SHAPE_MISMATCH, "byte rows never read as floats")


def test_arena_is_reused_and_results_outlive_the_call() raises:
    """A result survives calls on other arenas, and the next call on its own arena rewinds and refills it,
    retiring the result."""
    if not _engine_is_present():
        print("skipped: libusearch_shared is not on the loader path")
        return
    var library = Library()
    var index = _basis(library)
    var arena = Arena(library)
    var other = Arena(library)

    var first = index.get[DType.float32](Key(integer=0), arena)
    var second = index.get[DType.float32](Key(integer=1), other)
    assert_equal(first[0][0], Float32(1), "the first result survives a later call")
    assert_equal(second[0][1], Float32(1), "which landed in an arena of its own")

    var reused = index.get[DType.float32](Key(integer=1), arena)
    assert_equal(reused[0][1], Float32(1), "the third call refilled the first arena")
    var status = Status.SUCCESS
    try:
        _ = first[0]
    except error:
        status = error.status
    assert_true(status == Status.UNINITIALIZED, "and the view it rewound refuses to read")


def test_arena_survives_a_failed_call_and_retires_stale_views() raises:
    """An arena stays valid through a refusal, which retires the views of earlier calls as a success would."""
    if not _engine_is_present():
        print("skipped: libusearch_shared is not on the loader path")
        return
    var library = Library()
    var index = _basis(library)
    var arena = Arena(library)

    var earlier = index.get[DType.float32](Key(integer=0), arena)
    assert_equal(earlier[0][0], Float32(1), "the rows land in the caller's arena")

    var narrow = [Float32(1), 0, 0]
    var status = Status.SUCCESS
    try:
        _ = index.multiply(RowsView.of(narrow, 3), arena)
    except error:
        status = error.status
    assert_true(status == Status.SHAPE_MISMATCH, "a three-wide row cannot multiply a two-wide index")
    status = Status.SUCCESS
    try:
        _ = earlier[0]
    except error:
        status = error.status
    assert_true(status == Status.UNINITIALIZED, "the failed call rewound the memory the earlier view read")

    var reused = index.get[DType.float32](Key(integer=1), arena)
    assert_equal(reused[0][1], Float32(1), "which the next call rewinds and refills")


def test_caller_thread_executor_drives_every_verb() raises:
    """The executor that owns nothing runs a verb inline instead of being refused."""
    if not _engine_is_present():
        print("skipped: libusearch_shared is not on the loader path")
        return
    var library = Library()
    var inline = Executor(library)
    var index = DenseIndex.init_ram(library, DenseOptions(2, DenseMetric.EUCLIDEAN))
    var on = index.on(inline)
    on.reserve(2)
    var rows = [Float32(0), 0, Float32(5), 5]
    var keys = [Key(integer=1), Key(integer=2)]
    on.insert(keys, RowsView.of(rows, 2))
    assert_equal(on.count(Key(integer=2)), 1, "the inline lookup sees the row")

    var query = [Float32(4), 4]
    var arena = Arena(inline)
    var matches = on.search(RowsView.of(query, 2), 1, arena)
    assert_true(matches.row_keys(0)[0] == Key(integer=2), "the inline walk still finds the nearest row")
    inline.drain(Mood.SPINNING)


def test_buffers_round_trip_through_the_arena() raises:
    """A monolith read in place from the arena loads back with the geometry and rows it froze."""
    if not _engine_is_present():
        print("skipped: libusearch_shared is not on the loader path")
        return
    var library = Library()
    var executor = Executor.for_ram(library, threads=2)
    var index = DenseIndex.init_ram(library, DenseOptions(2, DenseMetric.INNER))
    index.reserve(2)
    var rows = [Float32(1), 2, Float32(3), 4]
    var keys = [Key(integer=5), Key(integer=6)]
    index.insert(keys, RowsView.of(rows, 2))

    var arena = Arena(executor)
    var monolith = index.on(executor).save_buffer(arena)
    assert_true(len(monolith.bytes()) != 0, "the monolith carries bytes")
    var frozen = dense_metadata_read_buffer(executor, monolith.bytes())
    assert_equal(Int(frozen.dimensions), 2, "the header names the width")
    assert_true(frozen.metric == DenseMetric.INNER, "and the metric")

    var restored = DenseIndex.load_ram_buffer(executor, monolith.bytes())
    assert_equal(Int(restored.describe().size), 2, "both rows load back")
    assert_equal(restored.get[DType.float32](Key(integer=6), arena)[0][1], Float32(4), "with their values")


def test_search_stream_pulls_in_distance_order() raises:
    """Each pull resumes the walk where the last one stopped, until the stream drains."""
    if not _engine_is_present():
        print("skipped: libusearch_shared is not on the loader path")
        return
    var library = Library()
    var executor = Executor.for_ram(library, threads=2)
    var index = DenseIndex.init_ram(library, DenseOptions(2, DenseMetric.EUCLIDEAN))
    index.reserve(2)
    var rows = [Float32(0), 0, Float32(5), 5]
    var keys = [Key(integer=1), Key(integer=2)]
    index.insert(keys, RowsView.of(rows, 2))

    var query = [Float32(4), 4]
    try:
        _ = index.on(executor).search_stream(RowsView.of(query, 2))
    except error:
        assert_true(error.status == Status.UNSUPPORTED, "a residency that streams no matches yet refuses the stream")
        return
    var stream = index.on(executor).search_stream(RowsView.of(query, 2))
    var arena = Arena(executor)
    assert_true(stream.next(1, arena).row_keys(0)[0] == Key(integer=2), "the first pull lands the nearest")
    assert_true(stream.next(1, arena).row_keys(0)[0] == Key(integer=1), "the second resumes to the farther")
    assert_equal(len(stream.next(1, arena).row_keys(0)), 0, "then the stream is drained")


def test_cursor_walks_every_row_once() raises:
    """A cursor pulls keys and rows in batches the caller sizes, until the walk is exhausted."""
    if not _engine_is_present():
        print("skipped: libusearch_shared is not on the loader path")
        return
    var library = Library()
    var executor = Executor.for_ram(library, threads=2)
    var index = _basis(library)
    var cursor = index.on(executor).cursor()
    var arena = Arena(executor)
    var batch = cursor.next_batch(1, ScalarKind.F32, arena)
    assert_equal(len(batch), 1, "the first batch holds one row")
    assert_true(batch.keys()[0] == Key(integer=0), "the first key")
    assert_equal(batch.row[DType.float32](0)[0], Float32(1), "beside its row")
    var keys = cursor.next_keys(1, arena)
    assert_true(keys.keys()[0] == Key(integer=1), "the key column alone")
    var status = Status.SUCCESS
    try:
        _ = keys.row[DType.float32](0)
    except error:
        status = error.status
    assert_true(status == Status.SHAPE_MISMATCH, "a pull of keys alone has no rows to read")
    assert_equal(len(cursor.next_keys(1, arena)), 0, "then the walk is exhausted")


def test_health_audits_every_entry() raises:
    """A small graph is reached whole from its landmarks."""
    if not _engine_is_present():
        print("skipped: libusearch_shared is not on the loader path")
        return
    var library = Library()
    var index = _basis(library)
    var reachability = index.health()
    assert_equal(Int(reachability.audited), 2, "both entries audited")
    assert_equal(Int(reachability.reachable), 2, "and both reached")


def test_merge_consumes_its_donors() raises:
    """A merge absorbs every donor's rows, and the donors it consumed are freed exactly once."""
    if not _engine_is_present():
        print("skipped: libusearch_shared is not on the loader path")
        return
    var library = Library()
    var index = _basis(library)
    var donor = DenseIndex.init_ram(library, DenseOptions(2, DenseMetric.INNER, values_per_key=2))
    donor.reserve(1)
    var rows = [Float32(1), 1]
    var keys = [Key(integer=3)]
    donor.insert(keys, RowsView.of(rows, 2))
    var donors = List[DenseIndex]()
    donors.append(donor^)
    try:
        index.merge(donors^)
    except error:
        assert_true(error.status == Status.UNSUPPORTED, "no residency absorbs donors yet")
        return
    assert_equal(Int(index.describe().size), 3, "the donor's row joined")
    assert_equal(index.count(Key(integer=3)), 1, "under its own key")


def test_mutating_a_view_is_refused_as_read_only() raises:
    """A viewed checkpoint refuses every mutating verb under its own status."""
    if not _engine_is_present():
        print("skipped: libusearch_shared is not on the loader path")
        return
    var library = Library()
    var index = _basis(library)
    index.save("/tmp/usearch-mojo-view.safegraphs")
    try:
        _ = DenseIndex.view_ram(library, "/tmp/usearch-mojo-view.safegraphs")
    except error:
        assert_true(error.status == Status.UNSUPPORTED, "RAM serves no mapping in place yet")
        return
    var view = DenseIndex.view_ram(library, "/tmp/usearch-mojo-view.safegraphs")
    assert_equal(Int(view.describe().size), 2, "the view serves what was saved")
    var status = Status.SUCCESS
    try:
        view.clear()
    except error:
        status = error.status
    assert_true(status == Status.READ_ONLY, "clearing a view is refused as read-only")


def test_scans_find_every_query_itself() raises:
    """A scan scores every row rather than the ones a walk reaches, so every query finds itself first."""
    if not _engine_is_present():
        print("skipped: libusearch_shared is not on the loader path")
        return
    var library = Library()
    var index = DenseIndex.init_ram(library, DenseOptions(2))
    index.reserve(3)
    var rows = [Float32(1), 0, Float32(0), 1, Float32(1), 1]
    var keys = [Key(integer=0), Key(integer=1), Key(integer=2)]
    index.insert(keys, RowsView.of(rows, 2))
    var arena = Arena(library)
    var matches = index.search(RowsView.of(rows, 2), 1, arena, DenseSearchConfig(traversal=Traversal.SCAN))
    assert_equal(len(matches), 3, "every row is a query")
    for query in range(3):
        assert_true(matches.row_keys(query)[0] == keys[query], "each query finds itself first")


# endregion Engine

# region Deferred Calls


def test_deferred_calls_wait_until_final() raises:
    """A deferred call reads `DEFERRED` right after submission, and its wait answers repeatedly."""
    if not _engine_is_present():
        print("skipped: libusearch_shared is not on the loader path")
        return
    var library = Library()
    var executor = Executor.for_ram(library, threads=2)
    var index = _basis(library)
    var query = [Float32(0), 1]
    var arena = Arena(executor)
    var call = index.deferred(executor).search(RowsView.of(query, 2), 1, arena)
    assert_true(call.status() == Status.DEFERRED, "pending right after submission")
    assert_true(call.wait().row_keys(0)[0] == Key(integer=1), "the nearest row")
    assert_true(call.status() == Status.SUCCESS, "final once waited")
    assert_true(call.wait().row_keys(0)[0] == Key(integer=1), "and the same answer when asked again")


def test_deferred_failures_repeat() raises:
    """A deferred call the engine refuses raises the same failure on every wait."""
    if not _engine_is_present():
        print("skipped: libusearch_shared is not on the loader path")
        return
    var library = Library()
    var executor = Executor.for_ram(library, threads=2)
    var index = _basis(library)
    var narrow = [Float32(1), 0, 0]
    var arena = Arena(executor)
    var call = index.deferred(executor).search(RowsView.of(narrow, 3), 1, arena)
    for _ in range(2):
        var status = Status.SUCCESS
        try:
            _ = call.wait()
        except error:
            status = error.status
        assert_true(status == Status.SHAPE_MISMATCH, "every wait raises it")
    assert_true(call.status() == Status.SHAPE_MISMATCH, "and the status keeps reporting it")


def test_deferred_refusals_finish_at_submission() raises:
    """A failure the binding catches before crossing is final at once, with no drain."""
    if not _engine_is_present():
        print("skipped: libusearch_shared is not on the loader path")
        return
    var library = Library()
    var executor = Executor.for_ram(library, threads=2)
    var index = _basis(library)
    var rows = [Float32(1), 1]
    var keys = [Key(integer=4), Key(integer=5)]
    var call = index.deferred(executor).insert(keys, RowsView.of(rows, 2))
    assert_true(call.status() == Status.SHAPE_MISMATCH, "two keys never name one row")


def test_deferred_results_borrow_the_callers_arena() raises:
    """A deferred export holds the caller's arena until final, so a call meeting it is refused without
    retiring the export; a later call on that arena does retire it, and a deferred insertion publishes."""
    if not _engine_is_present():
        print("skipped: libusearch_shared is not on the loader path")
        return
    var library = Library()
    var executor = Executor.for_ram(library, threads=2)
    var index = _basis(library)
    var rows = [Float32(2), 2]
    var keys = [Key(integer=8)]
    var deferred = index.deferred(executor)
    var inserted = deferred.insert(keys, RowsView.of(rows, 2))
    var arena = Arena(executor)
    var exported = deferred.get[DType.float32](Key(integer=1), arena)
    var status = Status.SUCCESS
    try:
        _ = index.get[DType.float32](Key(integer=0), arena)
    except error:
        status = error.status
    assert_true(status == Status.CONTENDED, "the arena serves the deferred export alone")
    _ = inserted.wait()
    assert_equal(exported.wait()[0][1], Float32(1), "the refused call left the holder's view readable")

    _ = index.get[DType.float32](Key(integer=0), arena)
    status = Status.SUCCESS
    try:
        _ = exported.wait()[0]
    except error:
        status = error.status
    assert_true(status == Status.UNINITIALIZED, "a later call on the arena retires the deferred view")
    assert_equal(index.count(Key(integer=8)), 1, "and the deferred insertion published")


# endregion Deferred Calls

# region Linear Algebra


def test_multiply_reports_every_scored_pair() raises:
    """The sparse product keeps the pairs `wanted` allows, and still counts every pair it scored."""
    if not _engine_is_present():
        print("skipped: libusearch_shared is not on the loader path")
        return
    var library = Library()
    var executor = Executor.for_ram(library, threads=2)
    var index = _basis(library)
    var on = index.on(executor)

    var left = [Float32(1), 0]
    var arena = Arena(executor)
    var every = on.multiply(RowsView.of(left, 2), arena)
    assert_equal(len(every.keys()), 2, "both columns are scored")
    assert_equal(Int(every.offsets()[1]), 2, "and the one row spans both")

    var cut = on.multiply(RowsView.of(left, 2), arena, DenseMultiplyConfig(wanted=1))
    assert_equal(len(cut.keys()), 1, "one pair kept")
    assert_equal(cut.required, 2, "though both were counted")


def test_maxsim_scores_over_documents_rather_than_rows() raises:
    """Two tokens of one query score against the indexed documents, not against each other."""
    if not _engine_is_present():
        print("skipped: libusearch_shared is not on the loader path")
        return
    var library = Library()
    var index = _basis(library)

    var tokens = [Float32(1), 0, Float32(0), 1]
    var arena = Arena(library)
    var matches = index.maxsim(RowsView.of(tokens, 2), 2, 2, arena)
    assert_equal(len(matches), 1, "two token rows are one query")
    assert_equal(len(matches.row_keys(0)), 2, "over two documents")
    var counted = index.maxsim(RowsView.of(tokens, 2), 2, 0, arena)
    assert_equal(len(counted.row_keys(0)), 0, "a zero wanted lands only the counts")


@fieldwise_init
struct _ValueTable(ImplicitlyCopyable, TrivialRegisterPassable):
    """The `BFloat16` value rows an attend test resolves keys through, its first `rows` resolvable."""

    var data: Address
    var rows: Int
    var row_bytes: Int


def _value_rows(
    state: Address,
    keys: Pointer[Key, ImmUntrackedOrigin],
    count: c_size_t,
    scalar: ScalarKind,
    columns: c_size_t,
    rows: Pointer[Address, MutAnyOrigin],
) abi("C"):
    """Resolves the key of integer `k` to row `k` of the table, and to no row past its resolvable ones
    or for a kind or width other than the table's own."""
    var table = Pointer[_ValueTable, ImmUntrackedOrigin](unsafe_from_address=state)[]
    var shaped = scalar == ScalarKind.BF16 and Int(columns) * 2 == table.row_bytes
    for lane in range(Int(count)):
        var row = Int(keys[unsafe_offset=lane].as_integer().value())
        rows[unsafe_offset=lane] = table.data + row * table.row_bytes if shaped and row < table.rows else NULL_ADDRESS


def test_attend_is_exact_attention_over_every_key() raises:
    """A graph this small scores every key, so attention is the exact softmax over all of them, with
    one head per walk and with two sharing it; a key without a value row refuses the call, and so
    does a carry shorter than the rows."""
    if not _engine_is_present():
        print("skipped: libusearch_shared is not on the loader path")
        return
    var library = Library()
    var executor = Executor.for_ram(library, threads=2)
    var index = DenseIndex.init_ram(library, DenseOptions(2, DenseMetric.INNER))
    var on = index.on(executor)
    on.reserve(3)
    # Every value is exact in bf16, so the oracle below scores what the index stores.
    var rows = [Float32(1), 0, Float32(0), 1, Float32(0.5), Float32(0.75)]
    var keys = [Key(integer=0), Key(integer=1), Key(integer=2)]
    on.insert(keys, RowsView.of(rows, 2))

    # Value rows come in the index's own quantization.
    var values = [BFloat16(10), BFloat16(-1), BFloat16(20), BFloat16(-2), BFloat16(30), BFloat16(-3)]
    var table = stack_allocation[1, _ValueTable]()
    table[unsafe_offset=0] = _ValueTable(Int(values.unsafe_ptr()), 3, 4)
    var queries = [Float32(1), 0, Float32(0), 1]
    var arena = Arena(executor)
    for head_count in range(1, 3):
        var carry = List[DenseSoftmaxState](length=2, fill=DenseSoftmaxState())
        var outputs = on.attend[DType.float32](
            RowsView.of(queries, 2),
            _value_rows,
            Int(table),
            arena,
            DenseAttendConfig(head_count=head_count, carry=carry),
        )
        for query in range(2):
            var logits = List[Float64]()
            for key in range(3):
                var product = queries[query * 2] * rows[key * 2] + queries[query * 2 + 1] * rows[key * 2 + 1]
                logits.append(Float64(product) / sqrt(Float64(2)))
            var maximum = max(logits[0], max(logits[1], logits[2]))
            var partition = Float64(0)
            var mixed = [Float64(0), Float64(0)]
            for key in range(3):
                var weight = exp(logits[key] - maximum)
                partition += weight
                for column in range(2):
                    mixed[column] += weight * Float64(values[key * 2 + column])
            for column in range(2):
                var expected = mixed[column] / partition
                assert_almost_equal(Float64(outputs[query][column]), expected, atol=1e-3 * max(abs(expected), 1.0))
            assert_almost_equal(Float64(carry[query].maximum_score), maximum, atol=1e-3 * max(abs(maximum), 1.0))
            assert_almost_equal(Float64(carry[query].partition), partition, atol=1e-3 * max(abs(partition), 1.0))

    var short_carry = List[DenseSoftmaxState](length=1, fill=DenseSoftmaxState())
    var status = Status.SUCCESS
    try:
        _ = on.attend[DType.float32](
            RowsView.of(queries, 2), _value_rows, Int(table), arena, DenseAttendConfig(carry=short_carry)
        )
    except error:
        status = error.status
    assert_true(status == Status.SHAPE_MISMATCH, "one softmax state cannot carry two rows")

    table[unsafe_offset=0] = _ValueTable(Int(values.unsafe_ptr()), 2, 4)
    status = Status.SUCCESS
    try:
        _ = on.attend[DType.float32](RowsView.of(queries, 2), _value_rows, Int(table), arena)
    except error:
        status = error.status
    assert_true(status == Status.SHAPE_MISMATCH, "a key without a value row refuses the call")
    # The table reaches the rows by address, which is not a use, so they are kept past the last call.
    _ = values^


# endregion Linear Algebra


def main() raises:
    var suite = TestSuite()
    suite.test[test_scalar_widths_round_sub_byte_rows_up]()
    suite.test[test_scalar_kinds_follow_their_element_types]()
    suite.test[test_keys_carry_integer_identities]()
    suite.test[test_keys_render_as_dashed_uuids]()
    suite.test[test_rows_derive_contiguous_strides]()
    suite.test[test_shapes_the_lengths_cannot_support_are_refused]()
    suite.test[test_telemetry_starts_at_an_untaken_walk]()
    suite.test[test_engine_round_trips_when_present]()
    suite.test[test_width_mismatch_is_refused_by_the_engine]()
    suite.test[test_names_come_from_the_engine_tables]()
    suite.test[test_get_exports_every_entry_under_a_key]()
    suite.test[test_arena_is_reused_and_results_outlive_the_call]()
    suite.test[test_arena_survives_a_failed_call_and_retires_stale_views]()
    suite.test[test_caller_thread_executor_drives_every_verb]()
    suite.test[test_buffers_round_trip_through_the_arena]()
    suite.test[test_search_stream_pulls_in_distance_order]()
    suite.test[test_cursor_walks_every_row_once]()
    suite.test[test_health_audits_every_entry]()
    suite.test[test_merge_consumes_its_donors]()
    suite.test[test_mutating_a_view_is_refused_as_read_only]()
    suite.test[test_scans_find_every_query_itself]()
    suite.test[test_deferred_calls_wait_until_final]()
    suite.test[test_deferred_failures_repeat]()
    suite.test[test_deferred_refusals_finish_at_submission]()
    suite.test[test_deferred_results_borrow_the_callers_arena]()
    suite.test[test_multiply_reports_every_scored_pair]()
    suite.test[test_maxsim_scores_over_documents_rather_than_rows]()
    suite.test[test_attend_is_exact_attention_over_every_key]()
    suite^.run()
