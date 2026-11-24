"""The Mojo binding's tests, in the shape `c/test.c` already uses.

Mojo has no `mojo test` subcommand and no test discovery, so every check below is registered in
`main` and one that is not silently never runs. The pure-logic group needs no library loaded; the
rest exercise the C ABI directly and are skipped when nothing is on the loader path.
"""

from std.atomic import Atomic
from std.ffi import c_size_t
from std.math import exp, sqrt
from std.memory import stack_allocation
from std.os import getenv, remove, rmdir
from std.runtime._asyncrt import create_raising_task
from std.tempfile import mkdtemp
from std.testing import TestSuite, assert_almost_equal, assert_equal, assert_false, assert_true
from std.time import perf_counter_ns

from usearch import (
    Admission,
    CPUClass,
    GPUClass,
    SequenceIndex,
    SequenceMetric,
    SequenceOptions,
    SequenceRefineConfig,
    SequenceRefinePlan,
    SequenceReshardConfig,
    SequenceReshardPlan,
    Tape,
    sequence_metadata_read_buffer,
    sequence_reshard,
    sequence_reshard_plan_finish_deferred,
    Arena,
    Continuation,
    DenseAttendConfig,
    DenseIndex,
    DenseInsertConfig,
    DenseMetric,
    DenseMultiplyConfig,
    DenseOptions,
    DenseRefineConfig,
    DenseRefinePlan,
    DenseReshardConfig,
    DenseReshardPlan,
    DenseSampleConfig,
    DenseSearchConfig,
    DenseSoftmaxState,
    DenseTraversalStats,
    Error,
    Executor,
    Key,
    Library,
    Mood,
    Phase,
    PhaseProgress,
    PredicateArguments,
    PredicateCost,
    RefineVerdict,
    ProgressArguments,
    RamOptions,
    Residency,
    RowsView,
    ScalarKind,
    Status,
    Traversal,
    dense_metadata_read_buffer,
    dense_reshard,
    dense_reshard_plan_finish_deferred,
    GeospatialIndex,
    GeospatialObservation,
    GeospatialOptions,
    GeospatialRefineConfig,
    GeospatialRefinePlan,
    GeospatialRelation,
    GeospatialSearchConfig,
    GeospatialTerrain,
    ShapesIndex,
    ShapesMetric,
    ShapesOptions,
    ShapesRefineConfig,
    ShapesRefinePlan,
    TapeView,
    geospatial_metadata_read_buffer,
    geospatial_reshard,
    shapes_metadata_read_buffer,
    shapes_reshard,
)
from usearch.types import Address, NULL_ADDRESS


# region Shared Testing
def _skipped(reason: String):
    """Announces a skipped test, so a run shows what it left out rather than passing in silence."""
    print("skipped:", reason)


def _split_mix_bits(value: UInt64) -> UInt64:
    """SplitMix64's finalizer, a bijection that spreads every input bit over all 64 output bits."""
    var mixed = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9
    mixed = (mixed ^ (mixed >> 27)) * 0x94D049BB133111EB
    return mixed ^ (mixed >> 31)


def _stream_key(seed: UInt64, name: String, index: UInt64 = 0) -> UInt64:
    """The `index`-th stream of the family `name` under `seed`, identical in C, C++, Rust and Python."""
    var hashed: UInt64 = 0xCBF29CE484222325
    for byte in name.as_bytes():
        hashed = (hashed ^ UInt64(byte)) * 0x100000001B3
    return _split_mix_bits(_split_mix_bits(seed ^ hashed) + index)
# endregion Shared Testing


comptime library_missing_k = "the shared library is not on the loader path"


def _run_seed() raises -> UInt64:
    """`USEARCH_SEED`, 42 when unset or empty, so a draw reproduces under the seed every suite reports."""
    var text = getenv("USEARCH_SEED")
    return 42 if text == "" else UInt64(atol(text))


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


def test_keys_hold_integer_identities() raises:
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
    assert_equal(Int(view.columns), 3, "the width is held verbatim")
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


def _library_is_present() raises -> Bool:
    """Whether the shared library is on the loader path; any other failure to load it raises rather than skips."""
    try:
        _ = Library()
        return True
    except error:
        if error.status != Status.LIBRARY_MISSING:
            raise error
        return False


def _basis(library: Library) raises -> DenseIndex:
    """Builds the two-entry inner-product index over the identity basis most verbs below score against.

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
    if not _library_is_present():
        _skipped(library_missing_k)
        return
    var library = Library()
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
    assert_true(
        library.cpu_class_name(description.cpu_class) != "unknown", "a host index names the class it dispatched to"
    )

    var query = [Float32(1), 0, 0]
    var arena = Arena(executor)
    var matches = on.search(RowsView.of(query, 3), 2, arena, DenseSearchConfig(stats=Pointer(to=stats)))
    assert_equal(len(matches), 1, "one query answered")
    assert_equal(len(matches.query_keys(0)), 2, "the one query filled its own output")
    assert_true(matches.query_keys(0)[0] == Key(integer=0), "the nearest entry is the query itself")
    assert_equal(Int(stats.computed_distances), 3, "the walk reported what it computed")
    var counted = on.search(RowsView.of(query, 3), 0, arena)
    assert_equal(len(counted.query_keys(0)), 0, "a zero wanted lands only the counts")

    var exported = on.get[DType.float32](Key(integer=0), arena)
    assert_equal(len(exported), 1, "one entry sits under that key")
    assert_equal(len(exported[0]), 3, "three values wide")
    assert_equal(exported[0][0], Float32(1), "the stored vector is read in place from the arena")
    assert_equal(on.count(Key(integer=0)), 1, "and the count agrees")
    var absent = on.get[DType.float32](Key(integer=9), arena)
    assert_equal(len(absent), 0, "an absent key exports nothing")


def test_width_mismatch_is_refused_by_the_engine() raises:
    """The binding passes the caller's width through, and the engine refuses one that disagrees."""
    if not _library_is_present():
        _skipped(library_missing_k)
        return
    var library = Library()
    var index = _basis(library)
    var narrow = [Float32(1), 0, 0]
    var arena = Arena(library)
    var status = Status.SUCCESS
    try:
        _ = index.search(RowsView.of(narrow, 3), 1, arena)
    except error:
        status = error.status
    assert_true(status == Status.SHAPE_MISMATCH, "a three-wide query cannot search a two-wide index")


def test_names_round_trip_through_the_library_and_refuse_strangers() raises:
    """Every enum spells and parses through the C name tables, which own the vocabulary."""
    if not _library_is_present():
        _skipped(library_missing_k)
        return
    var library = Library()
    assert_equal(library.status_name(Status.READ_ONLY), "read-only", "a status")
    assert_equal(library.status_name(Status.DEFERRED), "deferred", "the one non-final status")
    assert_equal(library.status_name(Status(99)), "unrecognized", "a code no status holds")
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
    assert_equal(library.phase_name(Phase.RESTITCH), "restitch", "a phase")
    assert_true(library.phase_from_name("gather") == Phase.GATHER, "and back")
    assert_equal(library.refine_verdict_name(RefineVerdict.STALLED), "stalled", "a verdict")
    assert_true(library.refine_verdict_from_name("converged") == RefineVerdict.CONVERGED, "and back")


def test_get_exports_every_entry_under_a_key() raises:
    """Get sizes its own output, so every entry sharing a key comes back in one call."""
    if not _library_is_present():
        _skipped(library_missing_k)
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


def test_arena_is_reused_and_results_outlive_the_call() raises:
    """A result survives calls on other arenas, and the next call on its own arena rewinds and refills it."""
    if not _library_is_present():
        _skipped(library_missing_k)
        return
    var library = Library()
    var index = _basis(library)
    var arena = Arena(library)
    var other = Arena(library)
    var described = arena.describe()
    assert_true(described.residency == Residency.RAM and described.device == 0, "an inline arena is host pages")

    var first = index.get[DType.float32](Key(integer=0), arena)
    var second = index.get[DType.float32](Key(integer=1), other)
    assert_equal(first[0][0], Float32(1), "the first result survives a later call")
    assert_equal(second[0][1], Float32(1), "which was written in an arena of its own")

    var reused = index.get[DType.float32](Key(integer=1), arena)
    assert_equal(reused[0][1], Float32(1), "the third call refilled the first arena")


def test_arena_survives_a_failed_call() raises:
    """An arena stays valid through a refusal, and the next call rewinds and refills it."""
    if not _library_is_present():
        _skipped(library_missing_k)
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

    var reused = index.get[DType.float32](Key(integer=1), arena)
    assert_equal(reused[0][1], Float32(1), "which the next call rewinds and refills")


def test_caller_thread_executor_drives_every_verb() raises:
    """The executor that owns nothing runs a verb inline instead of being refused."""
    if not _library_is_present():
        _skipped(library_missing_k)
        return
    var library = Library()
    var inline = Executor(library)
    var index = DenseIndex.init_ram(library, DenseOptions(2, DenseMetric.EUCLIDEAN))
    var on = index.on(inline)
    on.reserve(2)
    var rows = [Float32(0), 0, Float32(5), 5]
    var keys = [Key(integer=1), Key(integer=2)]
    on.insert(keys, RowsView.of(rows, 2))
    assert_equal(on.count(Key(integer=2)), 1, "the inline lookup sees the entry")

    var query = [Float32(4), 4]
    var arena = Arena(inline)
    var matches = on.search(RowsView.of(query, 2), 1, arena)
    assert_true(matches.query_keys(0)[0] == Key(integer=2), "the inline walk still finds the nearest entry")
    inline.drain(Mood.SPINNING)


def test_buffers_round_trip_through_the_arena() raises:
    """A monolith read in place from the arena loads back with the geometry and entries it froze."""
    if not _library_is_present():
        _skipped(library_missing_k)
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
    assert_true(len(monolith.bytes()) != 0, "the monolith holds bytes")
    var frozen = dense_metadata_read_buffer(executor, monolith.bytes())
    assert_equal(Int(frozen.dimensions), 2, "the header names the width")
    assert_true(frozen.metric == DenseMetric.INNER, "and the metric")

    var restored = DenseIndex.load_ram_buffer(executor, monolith.bytes())
    assert_equal(Int(restored.describe().size), 2, "both entries load back")
    assert_equal(restored.get[DType.float32](Key(integer=6), arena)[0][1], Float32(4), "with their values")


def test_search_stream_pulls_in_distance_order() raises:
    """Each pull resumes the walk where the last one stopped, until the stream drains."""
    if not _library_is_present():
        _skipped(library_missing_k)
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
        _skipped("search streaming is refused on this machine")
        return
    var stream = index.on(executor).search_stream(RowsView.of(query, 2))
    var arena = Arena(executor)
    assert_true(stream.next(1, arena).query_keys(0)[0] == Key(integer=2), "the first pull lands the nearest")
    assert_true(stream.next(1, arena).query_keys(0)[0] == Key(integer=1), "the second resumes to the farther")
    assert_equal(len(stream.next(1, arena).query_keys(0)), 0, "then the stream is drained")


def test_cursor_walks_every_row_once() raises:
    """A cursor pulls keys and their links in batches the caller sizes, until the walk is exhausted."""
    if not _library_is_present():
        _skipped(library_missing_k)
        return
    var library = Library()
    var executor = Executor.for_ram(library, threads=2)
    var index = _basis(library)
    var cursor = index.cursor()
    var arena = Arena(executor)
    var batch = cursor.on(executor).next_links(1, arena)
    assert_equal(len(batch), 1, "the first batch holds one entry")
    assert_true(batch.keys()[0] == Key(integer=0), "the first key")
    _ = batch.links(0)
    var keys = cursor.next_keys(1, arena)
    assert_true(keys.keys()[0] == Key(integer=1), "the key column alone")
    var status = Status.SUCCESS
    try:
        _ = keys.links(0)
    except error:
        status = error.status
    assert_true(status == Status.INVALID_ARGUMENT, "a pull of keys alone has no links to read")
    assert_equal(len(cursor.next_keys(1, arena)), 0, "then the walk is exhausted")


def test_health_audits_every_entry() raises:
    """A small graph is reached whole from its root."""
    if not _library_is_present():
        _skipped(library_missing_k)
        return
    var library = Library()
    var index = _basis(library)
    var reachability = index.health()
    assert_equal(Int(reachability.audited), 2, "both entries audited")
    assert_equal(Int(reachability.reachable), 2, "and both reached")


def test_refine_plans_stay_live_until_finished() raises:
    """A refine passed a plan opens it and reports where it stopped, and finishing it reports the whole plan."""
    if not _library_is_present():
        _skipped(library_missing_k)
        return
    var library = Library()
    var executor = Executor.for_ram(library, threads=2)
    var index = _basis(library)
    var plan = DenseRefinePlan()
    var progress = PhaseProgress()
    var opened = index.on(executor).refine(DenseRefineConfig(plan=Pointer(to=plan), phase=Pointer(to=progress)))
    assert_true(plan.address != NULL_ADDRESS, "the first refine opens the plan")
    assert_true(progress.phase == Phase.DONE, "and reports its round ended")
    assert_equal(library.phase_name(progress.phase), "done", "as the engine spells it")
    var finished = index.deferred(executor).refine_plan_finish(plan^)
    var whole = finished.wait()
    assert_equal(Int(whole.reachability.reachable), 2, "the whole plan reaches both entries")
    assert_equal(Int(opened.reachability.audited), 2, "as the round audited them")


def test_reshard_gathers_its_sources() raises:
    """A planned gather empties every other source into the target, and its deferred finish reports the whole plan."""
    if not _library_is_present():
        _skipped(library_missing_k)
        return
    var library = Library()
    var executor = Executor.for_ram(library, threads=2)
    var shard = DenseIndex.init_ram(library, DenseOptions(2, DenseMetric.INNER, values_per_key=2))
    shard.reserve(1)
    var rows = [Float32(1), 1]
    var keys = [Key(integer=3)]
    shard.insert(keys, RowsView.of(rows, 2))
    var parts = List[DenseIndex]()
    parts.append(_basis(library))
    parts.append(shard^)
    var plan = DenseReshardPlan()
    var report = dense_reshard(executor, parts, Span(parts)[0:1], DenseReshardConfig(plan=Pointer(to=plan)))
    assert_true(plan.address != NULL_ADDRESS, "a reshard passed a plan opens it")
    assert_equal(Int(parts[0].describe().size), 3, "the shard's entry joined")
    assert_equal(parts[0].count(Key(integer=3)), 1, "under its own key")
    assert_equal(Int(parts[1].describe().size), 0, "and the shard is left empty")
    assert_true(report.changed <= report.reselected, "the closing round changes only what it re-selected")
    var finished = dense_reshard_plan_finish_deferred(executor, plan^)
    assert_equal(Int(finished.wait().reachability.audited), 3, "the whole plan reports the gathered target")
    _ = parts^  # every index a plan names outlives its finish


def test_reshard_cuts_and_sample_draws() raises:
    """A cut empties the source into its targets, and a sample draws held keys, each placed in 3D."""
    if not _library_is_present():
        _skipped(library_missing_k)
        return
    var library = Library()
    var index = _basis(library)
    var arena = Arena(library)
    var shown = List[Key]()
    var places = List[Float32]()
    var focus = List[Key]()
    var draws = index.sample(
        2,
        arena,
        DenseSampleConfig(shown=shown, shown_coordinates=places, focus=focus, seed=_stream_key(_run_seed(), "sample")),
    )
    assert_true(len(draws) <= 2, "no more draws than wanted")
    for key in draws.keys():
        assert_true(key == Key(integer=0) or key == Key(integer=1), "every draw is a held key")
    if len(draws) != 0:
        assert_equal(Int(draws.coordinates().view.columns), 3, "each draw is placed in three dimensions")

    var parts = List[DenseIndex]()
    parts.append(index^)
    for _ in range(2):
        var target = DenseIndex.init_ram(library, DenseOptions(2, DenseMetric.INNER, values_per_key=2))
        target.reserve(2)
        parts.append(target^)
    _ = dense_reshard(
        Executor(library),
        Span(parts)[0:1],
        Span(parts)[1:3],
        DenseReshardConfig(seed=_stream_key(_run_seed(), "reshard")),
    )
    assert_equal(Int(parts[0].describe().size), 0, "the source is left empty")
    assert_equal(Int(parts[1].describe().size + parts[2].describe().size), 2, "and its entries live in the targets")


def test_mutating_a_view_is_refused_as_read_only() raises:
    """A viewed checkpoint refuses every mutating verb under its own status."""
    if not _library_is_present():
        _skipped(library_missing_k)
        return
    var library = Library()
    var index = _basis(library)
    var directory = mkdtemp(prefix="usearch-mojo-")
    var path = directory + "/view.safegraphs"
    index.save(path)
    try:
        _ = DenseIndex.view_ram(library, path)
    except error:
        assert_true(error.status == Status.UNSUPPORTED, "RAM serves no mapping in place yet")
        remove(path)
        rmdir(directory)
        _skipped("mapping a checkpoint in place is refused on this machine")
        return
    var view = DenseIndex.view_ram(library, path)
    assert_equal(Int(view.describe().size), 2, "the view serves what was saved")
    var status = Status.SUCCESS
    try:
        view.clear()
    except error:
        status = error.status
    assert_true(status == Status.READ_ONLY, "clearing a view is refused as read-only")
    remove(path)
    rmdir(directory)


def test_nvme_views_resolve_keys_through_their_key_table() raises:
    """An NVMe view counts keys only through the key table it was opened with."""
    if not _library_is_present():
        _skipped(library_missing_k)
        return
    var library = Library()
    var index = _basis(library)
    var directory = mkdtemp(prefix="usearch-mojo-")
    var path = directory + "/nvme.safegraphs"
    index.save(path)
    var drives = Executor.for_nvme(library, threads=1)
    var domains = List[Int32]()
    var keyed = DenseIndex.view_nvme(drives, path, key_table=RamOptions.on(domains))
    assert_equal(keyed.on(drives).count(Key(integer=1)), 1, "a keyed view counts a saved key")
    var keyless = DenseIndex.view_nvme(drives, path)
    var status = Status.SUCCESS
    try:
        _ = keyless.on(drives).count(Key(integer=1))
    except error:
        status = error.status
    assert_true(status == Status.UNSUPPORTED, "a keyless view refuses to count")
    remove(path)
    rmdir(directory)


def test_reshards_in_place_and_loads_only_manifests() raises:
    """A recut keeps every entry, and a lone monolith has no manifest a loaded handle could replace."""
    if not _library_is_present():
        _skipped(library_missing_k)
        return
    var library = Library()
    var parts = List[DenseIndex]()
    parts.append(_basis(library))
    var placement = RamOptions()
    placement.shard_capacity = c_size_t(8)
    _ = dense_reshard(Executor(library), parts, parts, DenseReshardConfig(ram=placement))
    assert_equal(Int(parts[0].describe().size), 2, "every entry stays")
    var directory = mkdtemp(prefix="usearch-mojo-")
    var path = directory + "/monolith.safegraphs"
    parts[0].save(path)
    var drives = Executor.for_nvme(library, threads=1)
    var status = Status.SUCCESS
    try:
        _ = DenseIndex.load_nvme(drives, path)
    except error:
        status = error.status
    assert_true(status == Status.UNSUPPORTED, "a lone monolith is refused")
    remove(path)
    rmdir(directory)


def test_scans_find_every_query_itself() raises:
    """A scan scores every entry rather than the ones a walk reaches, so every query finds itself first."""
    if not _library_is_present():
        _skipped(library_missing_k)
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
        assert_true(matches.query_keys(query)[0] == keys[query], "each query finds itself first")


def _stop_at_first_report(completed: Int, total: Int, reports: Pointer[Int, MutAnyOrigin]) -> Continuation:
    """Counts its reports and stops the call at the first one."""
    reports[] += 1
    return Continuation.CANCEL


def test_progress_cancels_a_search() raises:
    """A handler answering `CANCEL` at its first report stops the search, which raises `CANCELLED`."""
    if not _library_is_present():
        _skipped(library_missing_k)
        return
    var library = Library()
    var index = _basis(library)
    var queries = List[Float32]()
    for _ in range(16):
        queries.append(0)
        queries.append(1)
    var reports = stack_allocation[1, Int]().unsafe_origin_cast[MutAnyOrigin]()
    reports[unsafe_offset=0] = 0
    var arena = Arena(library)
    var status = Status.SUCCESS
    try:
        _ = index.search(
            RowsView.of(queries, 2),
            1,
            arena,
            DenseSearchConfig(progress=ProgressArguments.of[_stop_at_first_report](reports)),
        )
    except error:
        status = error.status
    assert_true(status == Status.CANCELLED, "the search stops when its handler cancels")
    assert_equal(reports[unsafe_offset=0], 1, "and the handler hears no report after cancelling")


def _has_parity(key: Key, parity: Pointer[UInt64, MutAnyOrigin]) -> Bool:
    """Whether the key's integer identity has the given parity."""
    return key.as_integer().value() % 2 == parity[]


def _admit_parity(
    keys: Span[Key, ImmUntrackedOrigin],
    admissions: Span[Admission, MutUntrackedOrigin],
    parity: Pointer[UInt64, MutAnyOrigin],
):
    """Admits the keys whose integer identity has the given parity."""
    for position in range(len(keys)):
        admissions[position] = Admission.ADMIT if _has_parity(keys[position], parity) else Admission.REJECT


def test_predicate_filters_keys() raises:
    """A per-key and a batch Mojo predicate admitting odd keys each leave only those among every query's matches."""
    if not _library_is_present():
        _skipped(library_missing_k)
        return
    var library = Library()
    var index = DenseIndex.init_ram(library, DenseOptions(2))
    index.reserve(4)
    var rows = [Float32(1), 0, Float32(0), 1, Float32(1), 1, Float32(2), 1]
    var keys = [Key(integer=0), Key(integer=1), Key(integer=2), Key(integer=3)]
    index.insert(keys, RowsView.of(rows, 2))
    var parity = stack_allocation[1, UInt64]().unsafe_origin_cast[MutAnyOrigin]()
    parity[unsafe_offset=0] = 1
    var arena = Arena(library)
    var predicates = [PredicateArguments.each[_has_parity](parity), PredicateArguments.batch[_admit_parity](parity)]
    assert_true(predicates[0].cost == PredicateCost.CHEAP, "a per-key predicate is cheap")
    assert_true(predicates[1].cost == PredicateCost.EXPENSIVE, "and a batch one expensive unless told otherwise")
    for predicate in predicates:
        var matches = index.search(
            RowsView.of(rows, 2), 4, arena, DenseSearchConfig(traversal=Traversal.SCAN, predicate=predicate)
        )
        for query in range(4):
            var found = matches.query_keys(query)
            assert_equal(len(found), 2, "both odd keys and neither even one")
            for position in range(len(found)):
                assert_equal(found[position].as_integer().value() % 2, UInt64(1), "only odd keys are reported")


# endregion Engine

# region Deferred Calls


def test_deferred_calls_wait_until_final() raises:
    """A deferred call reads `DEFERRED` until final, and its wait answers repeatedly."""
    if not _library_is_present():
        _skipped(library_missing_k)
        return
    var library = Library()
    var executor = Executor.for_ram(library, threads=2)
    var index = _basis(library)
    var query = [Float32(0), 1]
    var arena = Arena(executor)
    var call = index.deferred(executor).search(RowsView.of(query, 2), 1, arena)
    assert_true(
        call.status() == Status.DEFERRED or call.status() == Status.SUCCESS,
        "pending or already finished by the pool's thread 0",
    )
    assert_true(call.wait().query_keys(0)[0] == Key(integer=1), "the nearest entry")
    assert_true(call.status() == Status.SUCCESS, "final once waited")
    assert_true(call.wait().query_keys(0)[0] == Key(integer=1), "and the same answer when asked again")


def test_deferred_failures_repeat() raises:
    """A deferred call the engine refuses raises the same failure on every wait."""
    if not _library_is_present():
        _skipped(library_missing_k)
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
    if not _library_is_present():
        _skipped(library_missing_k)
        return
    var library = Library()
    var executor = Executor.for_ram(library, threads=2)
    var index = _basis(library)
    var rows = [Float32(1), 1]
    var keys = [Key(integer=4), Key(integer=5)]
    var call = index.deferred(executor).insert(keys, RowsView.of(rows, 2))
    assert_true(call.status() == Status.SHAPE_MISMATCH, "two keys never name one row")


struct _Gate(Movable):
    """Holds the call reporting to it at its last progress report until the test releases it, so the
    test meets the call while it still holds its arena; a call run at submission stops waiting after
    a few seconds, since its submitter cannot release it."""

    var entered: Atomic[Int]
    var released: Atomic[Int]

    def __init__(out self):
        self.entered = Atomic[Int](0)
        self.released = Atomic[Int](0)

    def entered_soon(mut self) -> Bool:
        """Whether its call arrived within a few seconds."""
        var deadline = perf_counter_ns() + 3_000_000_000
        while self.entered.load() == 0 and perf_counter_ns() < deadline:
            pass
        return self.entered.load() != 0


def _hold_until_released(completed: Int, total: Int, gate: Pointer[_Gate, MutAnyOrigin]) -> Continuation:
    """Lets the test know at the last report, then waits for it to release the call."""
    if completed != total:
        return Continuation.CONTINUE
    gate[].entered.store(1)
    var deadline = perf_counter_ns() + 3_000_000_000
    while gate[].released.load() == 0 and perf_counter_ns() < deadline:
        pass
    return Continuation.CONTINUE


def test_deferred_results_view_the_callers_arena() raises:
    """A deferred search holds the caller's arena until final, so another call on it is refused meanwhile."""
    if not _library_is_present():
        _skipped(library_missing_k)
        return
    var library = Library()
    var executor = Executor.for_ram(library, threads=2)
    var index = _basis(library)
    var rows = [Float32(2), 2]
    var keys = [Key(integer=8)]
    var deferred = index.deferred(executor)
    var inserted = deferred.insert(keys, RowsView.of(rows, 2))
    _ = inserted.wait()  # one call holds the executor at a time
    var arena = Arena(executor)
    var gate = _Gate()
    var query = [Float32(1), 1]
    var searched = deferred.search(
        RowsView.of(query, 2),
        1,
        arena,
        DenseSearchConfig(
            progress=ProgressArguments.of[_hold_until_released](
                Pointer(to=gate).unsafe_origin_cast[MutAnyOrigin](), steps=1
            )
        ),
    )
    assert_true(gate.entered_soon(), "the deferred search reaches its progress handler")
    var status = Status.SUCCESS
    try:
        _ = index.get[DType.float32](Key(integer=0), arena)
    except error:
        status = error.status
    assert_true(status == Status.CONTENDED, "the arena serves the deferred search alone")
    gate.released.store(1)
    assert_true(searched.wait().query_keys(0)[0] == Key(integer=8), "the refused call left the holder's view readable")
    # The handler reaches the gate by address, which is not a use, so it is kept past the wait.
    _ = gate^
    assert_equal(index.count(Key(integer=8)), 1, "and the deferred insertion published")


async def _search_both[
    queries_origin: ImmOrigin, //
](
    east: DenseIndex, west: DenseIndex, east_executor: Executor, west_executor: Executor, queries: RowsView[queries_origin]
) raises -> UInt64:
    """Searches two shards at once, each on its own executor and arena, and adds their nearest keys."""
    var east_arena = Arena(east_executor)
    var west_arena = Arena(west_executor)
    var east_call = east.deferred(east_executor).search(queries, 1, east_arena)
    var west_call = west.deferred(west_executor).search(queries, 1, west_arena)
    ref east_matches = await east_call
    ref west_matches = await west_call
    return east_matches.query_keys(0)[0].as_integer().value() + west_matches.query_keys(0)[0].as_integer().value()


def test_awaited_fan_out_answers_both_shards() raises:
    """Two deferred searches submitted before either is awaited both answer inside one task."""
    if not _library_is_present():
        _skipped(library_missing_k)
        return
    var library = Library()
    var east = _basis(library)
    var west = DenseIndex.init_ram(library, DenseOptions(2, DenseMetric.INNER))
    var rows = [Float32(1), 0, Float32(0), 1]
    west.insert([Key(integer=10), Key(integer=11)], RowsView.of(rows, 2))
    var east_executor = Executor.for_ram(library, threads=2)
    var west_executor = Executor.for_ram(library, threads=2)
    var query = [Float32(0), 1]
    var both = create_raising_task(_search_both(east, west, east_executor, west_executor, RowsView.of(query, 2)))
    assert_equal(both^.wait(), UInt64(1 + 11), "each shard's nearest entry")


async def _search_once[
    queries_origin: ImmOrigin, //
](index: DenseIndex, executor: Executor, queries: RowsView[queries_origin]) raises -> Int:
    """Awaits one deferred search and answers the status code it failed with, or zero when it succeeded."""
    var arena = Arena(executor)
    var call = index.deferred(executor).search(queries, 1, arena)
    try:
        ref matches = await call
        _ = len(matches)
    except error:
        return Int(error.status.code)
    return 0


def test_awaited_failures_raise() raises:
    """A deferred call the engine refuses raises its failure through `await` and the task's wait."""
    if not _library_is_present():
        _skipped(library_missing_k)
        return
    var library = Library()
    var executor = Executor.for_ram(library, threads=2)
    var index = _basis(library)
    var narrow = [Float32(1), 0, 0]
    var failing = create_raising_task(_search_once(index, executor, RowsView.of(narrow, 3)))
    assert_equal(failing^.wait(), Int(Status.SHAPE_MISMATCH.code), "the refusal reaches the awaiting task")


def test_arenas_free_before_their_executor() raises:
    """An arena holds the origin of the executor that made it, so the executor outlives it past its own last use."""
    if not _library_is_present():
        _skipped(library_missing_k)
        return
    var library = Library()
    var index = _basis(library)
    var executor = Executor.for_ram(library, threads=2)
    var arena = Arena(executor)
    var query = [Float32(0), 1]
    for _ in range(2):
        var matches = index.search(RowsView.of(query, 2), 1, arena)
        assert_true(matches.query_keys(0)[0] == Key(integer=1), "the arena serves calls after its executor's last use")


# endregion Deferred Calls

# region Linear Algebra


def test_multiplication_reports_every_scored_pair() raises:
    """The sparse product keeps the pairs `wanted` allows, and still counts every pair it scored."""
    if not _library_is_present():
        _skipped(library_missing_k)
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


def test_late_interaction_scores_over_documents() raises:
    """Two tokens of one query score against the indexed documents, not against each other."""
    if not _library_is_present():
        _skipped(library_missing_k)
        return
    var library = Library()
    var index = _basis(library)

    var tokens = [Float32(1), 0, Float32(0), 1]
    var arena = Arena(library)
    var matches = index.maxsim(RowsView.of(tokens, 2), 2, 2, arena)
    assert_equal(matches.queries, 1, "two token rows are one query")
    var bounded = Int(matches.offsets()[1] - matches.offsets()[0])
    assert_equal(bounded, 2, "over two documents")
    var every = index.maxsim(RowsView.of(tokens, 2), 2, 0, arena)
    assert_true(Int(every.offsets()[1] - every.offsets()[0]) >= bounded, "a zero wanted keeps every scored key")


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


def test_attention_is_exact_over_every_key() raises:
    """Over a graph this small, attention is the exact softmax over every key, and malformed inputs are refused."""
    if not _library_is_present():
        _skipped(library_missing_k)
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
    assert_true(status == Status.SHAPE_MISMATCH, "one softmax state cannot hold two rows")

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


# region Sequences


def _words(count: Int, stream: Int) raises -> Tape[DType.uint8]:
    """`count` lowercase words of 8 to 16 letters, from the linear congruential stream `stream` keys."""
    var tape = Tape[DType.uint8]()
    var state = _stream_key(_run_seed(), "words", UInt64(stream))
    for _ in range(count):
        state = state * 6364136223846793005 + 1442695040888963407
        var letters = List[UInt8]()
        for _ in range(8 + Int((state >> 33) % 9)):
            state = state * 6364136223846793005 + 1442695040888963407
            letters.append(UInt8(97 + Int((state >> 33) % 26)))
        tape.append(Span(letters))
    return tape^


def _keys(first: Int, count: Int) -> List[Key]:
    """`count` consecutive integer keys from `first`."""
    var keys = List[Key]()
    for key in range(first, first + count):
        keys.append(Key(integer=UInt64(key)))
    return keys^


def _sequences(
    library: Library, executor: Executor, count: Int, first: Int = 0, stream: Int = 0
) raises -> SequenceIndex:
    """A host sequence index holding `count` random words under consecutive keys from `first`."""
    var index = SequenceIndex.init_ram(library)
    if count != 0:
        var tape = _words(count, stream)
        index.on(executor).insert(_keys(first, count), tape.view())
    return index^


def test_sequence_index_starts_empty_and_keeps_its_metric() raises:
    """A fresh host index is empty, on a CPU class, and reads back the kind it froze, spelled by the engine."""
    if not _library_is_present():
        _skipped(library_missing_k)
        return
    var library = Library()
    var index = SequenceIndex.init_ram(library, SequenceOptions(SequenceMetric.LEVENSHTEIN_UTF8))
    var described = index.describe()
    assert_equal(Int(described.size), 0, "a fresh index is empty")
    assert_true(described.cpu_class != CPUClass.UNKNOWN and described.gpu_class == GPUClass.UNKNOWN, "on a CPU")
    assert_true(described.metadata.metric == SequenceMetric.LEVENSHTEIN_UTF8, "the frozen kind reads back")
    var spelled = library.sequence_metric_name(SequenceMetric.LEVENSHTEIN_UTF8)
    assert_true(library.sequence_metric_from_name(spelled) == SequenceMetric.LEVENSHTEIN_UTF8, "names round-trip")
    var status = Status.SUCCESS
    try:
        _ = SequenceIndex.init_ram(library, SequenceOptions(SequenceMetric.OVERLAP))
    except error:
        status = error.status
    assert_true(status == Status.CONFIG_MISMATCH, "the overlap kind without window widths is the engine's refusal")


def test_sequence_search_get_and_count() raises:
    """Every word searched for comes back first under its own key at distance zero, and reads back whole."""
    if not _library_is_present():
        _skipped(library_missing_k)
        return
    var library = Library()
    var index = SequenceIndex.init_ram(library)
    var words = Tape[DType.uint8]()
    words.append("kitten")
    words.append("sitting")
    words.append("mitten")
    index.insert(_keys(1, 3), words.view())
    assert_equal(Int(index.describe().size), 3, "three words stored")

    var query = Tape[DType.uint8]()
    query.append("sitting")
    var arena = Arena(library)
    var matches = index.search(query.view(), 1, arena)
    assert_true(matches.query_keys(0)[0] == Key(integer=2), "the word finds itself first")
    assert_equal(matches.query_distances(0)[0], 0, "at distance zero")

    var stats = index.count(Key(integer=2))
    assert_equal(Int(stats.entries), 1, "one entry under the key")
    assert_equal(Int(stats.total_bytes), 7, "seven bytes long")
    var exported = index.get(Key(integer=2), arena)
    assert_equal(len(exported), 1, "one sequence exported")
    assert_equal(String(StringSlice(unsafe_from_utf8=exported[0])), "sitting", "byte for byte")
    assert_equal(len(index.get(Key(integer=9), arena)), 0, "an absent key exports nothing")


def test_sequence_saves_and_loads_a_buffer() raises:
    """A monolith saved into an arena loads back with every entry, and its header reads without loading."""
    if not _library_is_present():
        _skipped(library_missing_k)
        return
    var library = Library()
    var executor = Executor.for_ram(library, threads=2)
    var index = _sequences(library, executor, 16)
    var arena = Arena(executor)
    var monolith = index.on(executor).save_buffer(arena)
    var frozen = sequence_metadata_read_buffer(executor, monolith.bytes())
    assert_true(frozen.metric == SequenceMetric.LEVENSHTEIN_BYTES, "the header names the kind")
    var restored = SequenceIndex.load_ram_buffer(executor, monolith.bytes())
    assert_equal(Int(restored.describe().size), 16, "every entry loads back")


def test_sequence_refine_plan_resumes_and_finishes() raises:
    """A stopped round stays live in its plan, the next call ends it, and the plan finishes exactly once."""
    if not _library_is_present():
        _skipped(library_missing_k)
        return
    var library = Library()
    var executor = Executor.for_ram(library, threads=2)
    var index = _sequences(library, executor, 32)
    var round = index.on(executor).refine()
    assert_equal(Int(round.reachability.audited), 32, "a round audits every entry")
    assert_true(round.changed <= round.reselected, "and changes only what it re-selected")

    var plan = SequenceRefinePlan()
    var progress = PhaseProgress()
    var reports = stack_allocation[1, Int]().unsafe_origin_cast[MutAnyOrigin]()
    reports[unsafe_offset=0] = 0
    var status = Status.SUCCESS
    try:
        _ = index.on(executor).refine(
            SequenceRefineConfig(
                progress=ProgressArguments.of[_stop_at_first_report](reports),
                plan=Pointer(to=plan),
                phase=Pointer(to=progress),
            )
        )
    except error:
        status = error.status
    assert_true(status == Status.CANCELLED, "the handler stops the round")
    assert_true(progress.phase != Phase.DONE, "short of its end")
    _ = index.on(executor).refine(SequenceRefineConfig(plan=Pointer(to=plan), phase=Pointer(to=progress)))
    assert_true(progress.phase == Phase.DONE, "the next call ends it")
    var whole = index.on(executor).refine_plan_finish(plan^)
    assert_equal(Int(whole.reachability.audited), 32, "the whole plan reports every entry")
    status = Status.SUCCESS
    try:
        _ = index.on(executor).refine_plan_finish(SequenceRefinePlan())
    except error:
        status = error.status
    assert_true(status == Status.INVALID_ARGUMENT, "a plan no refine opened, or one finished, is refused")

    var inline = SequenceRefinePlan()
    status = Status.SUCCESS
    try:
        _ = index.refine(SequenceRefineConfig(plan=Pointer(to=inline)))
    except error:
        status = error.status
    assert_true(status == Status.INVALID_ARGUMENT, "a plan lives on an executor, so the calling thread opens none")


def test_sequence_reshard_gathers_cuts_and_refuses() raises:
    """Sources gather into a target, a source cuts into targets through a plan, and malformed lists move nothing."""
    if not _library_is_present():
        _skipped(library_missing_k)
        return
    var library = Library()
    var executor = Executor.for_ram(library, threads=2)
    var parts = List[SequenceIndex]()
    for stream in range(3):
        parts.append(_sequences(library, executor, 16, 16 * stream, stream))
    var report = sequence_reshard(executor, parts, Span(parts)[0:1])
    assert_true(report.changed <= report.reselected, "the closing round changes only what it re-selected")
    assert_equal(Int(parts[0].describe().size), 48, "every entry gathers into the target")
    assert_equal(Int(parts[0].count(Key(integer=40)).entries), 1, "under its own key")
    assert_equal(Int(parts[1].describe().size + parts[2].describe().size), 0, "the other sources end empty")

    var cut = List[SequenceIndex]()
    cut.append(_sequences(library, executor, 32))
    cut.append(SequenceIndex.init_ram(library))
    cut.append(SequenceIndex.init_ram(library))
    var plan = SequenceReshardPlan()
    var progress = PhaseProgress()
    var reports = stack_allocation[1, Int]().unsafe_origin_cast[MutAnyOrigin]()
    reports[unsafe_offset=0] = 0
    var seed = _stream_key(_run_seed(), "reshard")
    try:
        _ = sequence_reshard(
            executor,
            Span(cut)[0:1],
            Span(cut)[1:3],
            SequenceReshardConfig(
                seed=seed,
                progress=ProgressArguments.of[_stop_at_first_report](reports),
                plan=Pointer(to=plan),
                phase=Pointer(to=progress),
            ),
        )
    except error:
        assert_true(error.status == Status.CANCELLED, "only the handler stops the cut")
    assert_equal(Int(cut[1].describe().size + cut[2].describe().size), 32, "a stop keeps every entry moved")
    _ = sequence_reshard(
        executor,
        Span(cut)[0:1],
        Span(cut)[1:3],
        SequenceReshardConfig(plan=Pointer(to=plan), phase=Pointer(to=progress)),
    )
    assert_true(progress.phase == Phase.DONE, "the next call naming the plan ends its rounds")
    var finished = sequence_reshard_plan_finish_deferred(executor, plan^)
    _ = finished.wait()
    assert_equal(Int(cut[0].describe().size), 0, "the source is left empty and live")

    var first = _sequences(library, executor, 8)
    var pair = List[SequenceIndex]()
    pair.append(first^)
    pair.append(_sequences(library, executor, 8, 8, 1))
    var status = Status.SUCCESS
    try:
        _ = sequence_reshard(executor, Span(pair)[0:1], Span(pair)[0:1])
    except error:
        status = error.status
    assert_true(status == Status.UNSUPPORTED, "one index as its only source and target has nothing to recut")
    status = Status.SUCCESS
    try:
        _ = sequence_reshard(executor, Span(pair)[0:0], Span(pair)[0:1])
    except error:
        status = error.status
    assert_true(status == Status.INVALID_ARGUMENT, "an empty list is malformed")
    status = Status.SUCCESS
    try:
        _ = sequence_reshard(executor, Span(pair)[0:1], Span(pair)[1:2])
    except error:
        status = error.status
    assert_true(status == Status.CONFIG_MISMATCH, "a target holding foreign entries mismatches")
    assert_equal(Int(pair[0].describe().size + pair[1].describe().size), 16, "every refusal moves nothing")
    _ = cut^
    _ = parts^


# endregion Sequences

# region Shapes


def _clouds(count: Int, stream: Int) raises -> Tape[DType.float32, 3]:
    """`count` point clouds of 4 to 12 points in the unit cube, from the linear congruential stream `stream` keys."""
    var tape = Tape[DType.float32, 3]()
    var state = _stream_key(_run_seed(), "clouds", UInt64(stream))
    for _ in range(count):
        state = state * 6364136223846793005 + 1442695040888963407
        var scalars = List[Float32]()
        for _ in range(3 * (4 + Int((state >> 33) % 9))):
            state = state * 6364136223846793005 + 1442695040888963407
            scalars.append(Float32(Int((state >> 33) % 1000)) / 1000)
        tape.append(Span(scalars))
    return tape^


def _shapes(library: Library, executor: Executor, count: Int, first: Int = 0, stream: Int = 0) raises -> ShapesIndex:
    """A host shape index holding `count` random clouds under consecutive keys from `first`."""
    var index = ShapesIndex.init_ram(library)
    if count != 0:
        var tape = _clouds(count, stream)
        index.on(executor).insert(_keys(first, count), tape.view())
    return index^


def test_shapes_index_starts_empty_and_keeps_its_metric() raises:
    """A fresh host index is empty, on a CPU class, and reads back the kind it froze, spelled by the engine."""
    if not _library_is_present():
        _skipped(library_missing_k)
        return
    var library = Library()
    var index = ShapesIndex.init_ram(library, ShapesOptions(ShapesMetric.HAUSDORFF))
    var described = index.describe()
    assert_equal(Int(described.size), 0, "a fresh index is empty")
    assert_true(described.cpu_class != CPUClass.UNKNOWN and described.gpu_class == GPUClass.UNKNOWN, "on a CPU")
    assert_true(described.metadata.metric == ShapesMetric.HAUSDORFF, "the frozen kind reads back")
    assert_true(described.metadata.quantization == ScalarKind.F32, "over f32 points by default")
    var spelled = library.shapes_metric_name(ShapesMetric.WASSERSTEIN_PROCRUSTES)
    assert_true(library.shapes_metric_from_name(spelled) == ShapesMetric.WASSERSTEIN_PROCRUSTES, "names round-trip")
    var status = Status.SUCCESS
    try:
        _ = ShapesIndex.init_ram(library, ShapesOptions(quantization=ScalarKind.F64))
    except error:
        status = error.status
    assert_true(status == Status.UNSUPPORTED, "storage other than f16 or f32 is the engine's refusal")


def test_shapes_search_get_and_count() raises:
    """Clouds of different point counts share one tape, each finds itself first, and reads back in its pose."""
    if not _library_is_present():
        _skipped(library_missing_k)
        return
    var library = Library()
    var executor = Executor.for_ram(library, threads=2)
    var clouds = Tape[DType.float32, 3]()
    for points in [5, 9, 13]:
        var scalars = List[Float32]()
        for scalar in range(3 * points):
            scalars.append(Float32((scalar * 7 + points) % 11) / 11)
        clouds.append(Span(scalars))
    var index = ShapesIndex.init_ram(library)
    index.on(executor).reserve(3, 27)
    index.on(executor).insert(_keys(1, 3), clouds.view())
    var described = index.describe()
    assert_equal(Int(described.size), 3, "three clouds stored")
    assert_equal(Int(described.total_points), 27, "with every point counted")

    var arena = Arena(executor)
    var query = TapeView.of[width=3](Span(clouds.elements), Span(clouds.offsets)[2:4])
    var matches = index.on(executor).search(query, 1, arena)
    assert_true(matches.query_keys(0)[0] == Key(integer=3), "the cloud finds itself first")

    var stats = index.count(Key(integer=3))
    assert_equal(Int(stats.entries), 1, "one entry under the key")
    assert_equal(Int(stats.total_points), 13, "thirteen points")
    var exported = index.on(executor).get[DType.float32](Key(integer=3), arena)
    assert_equal(len(exported), 1, "one cloud exported")
    assert_equal(len(exported[0]), 39, "three scalars per point")
    assert_almost_equal(exported[0][4], clouds.elements[14 * 3 + 4], atol=1e-6, msg="in the pose it was inserted in")
    assert_equal(len(index.get[DType.float32](Key(integer=9), arena)), 0, "an absent key exports nothing")

    var cursor = index.cursor(_keys(2, 1))
    assert_true(cursor.on(executor).next_links(8, arena).keys()[0] == Key(integer=2), "a walk from a start key yields it first")
    var sampled = index.deferred(executor).sample(2, arena)
    assert_true(len(sampled.wait()) <= 2, "a deferred sample draws at most what it wants")


def test_shapes_checkpoints_refine_and_reshard() raises:
    """A monolith round-trips, a stopped round resumes, sources gather, and malformed lists move nothing."""
    if not _library_is_present():
        _skipped(library_missing_k)
        return
    var library = Library()
    var executor = Executor.for_ram(library, threads=2)
    var index = _shapes(library, executor, 16)
    var arena = Arena(executor)
    var monolith = index.on(executor).save_buffer(arena)
    var frozen = shapes_metadata_read_buffer(executor, monolith.bytes())
    assert_true(frozen.metric == ShapesMetric.CHAMFER and frozen.quantization == ScalarKind.F32, "the header")
    var restored = ShapesIndex.load_ram_buffer(executor, monolith.bytes())
    assert_equal(Int(restored.describe().total_points), Int(index.describe().total_points), "every point loads back")

    var plan = ShapesRefinePlan()
    var progress = PhaseProgress()
    var reports = stack_allocation[1, Int]().unsafe_origin_cast[MutAnyOrigin]()
    reports[unsafe_offset=0] = 0
    try:
        _ = index.on(executor).refine(
            ShapesRefineConfig(
                progress=ProgressArguments.of[_stop_at_first_report](reports),
                plan=Pointer(to=plan),
                phase=Pointer(to=progress),
            )
        )
    except error:
        assert_true(error.status == Status.CANCELLED, "only the handler stops the round")
    _ = index.on(executor).refine(ShapesRefineConfig(plan=Pointer(to=plan), phase=Pointer(to=progress)))
    assert_true(progress.phase == Phase.DONE, "the next call ends it")
    assert_equal(Int(index.on(executor).refine_plan_finish(plan^).reachability.audited), 16, "the plan saw all")

    var parts = List[ShapesIndex]()
    for stream in range(2):
        parts.append(_shapes(library, executor, 8, 8 * stream, stream))
    _ = shapes_reshard(executor, parts, Span(parts)[0:1])
    assert_equal(Int(parts[0].describe().size), 16, "every entry gathers into the target")
    assert_equal(Int(parts[1].describe().size), 0, "the other source ends empty")
    var status = Status.SUCCESS
    try:
        _ = shapes_reshard(executor, Span(parts)[0:1], Span(parts)[0:1])
    except error:
        status = error.status
    assert_true(status == Status.UNSUPPORTED, "one index as its only source and target has nothing to recut")
    status = Status.SUCCESS
    try:
        _ = shapes_reshard(executor, Span(parts)[0:0], Span(parts)[0:1])
    except error:
        status = error.status
    assert_true(status == Status.INVALID_ARGUMENT, "an empty list is malformed")
    parts[1].on(executor).insert(_keys(64, 1), _clouds(1, 9).view())
    status = Status.SUCCESS
    try:
        _ = shapes_reshard(executor, Span(parts)[0:1], Span(parts)[1:2])
    except error:
        status = error.status
    assert_true(status == Status.CONFIG_MISMATCH, "a target holding foreign entries mismatches")
    assert_equal(Int(parts[0].describe().size + parts[1].describe().size), 17, "every refusal moves nothing")


# endregion Shapes

# region Geospatial


def _stops(count: Int, stream: Int) raises -> List[GeospatialObservation]:
    """`count` positions spread over the inhabited latitudes, from the linear congruential stream `stream` keys."""
    var stops = List[GeospatialObservation]()
    var state = _stream_key(_run_seed(), "stops", UInt64(stream))
    for _ in range(count):
        state = state * 6364136223846793005 + 1442695040888963407
        var latitude = Float64(Int((state >> 33) % 120000)) / 1000 - 60
        state = state * 6364136223846793005 + 1442695040888963407
        stops.append(GeospatialObservation(latitude, Float64(Int((state >> 33) % 360000)) / 1000 - 180))
    return stops^


def _places(
    library: Library, executor: Executor, count: Int, first: Int = 0, stream: Int = 0
) raises -> GeospatialIndex:
    """A host geospatial index holding `count` random positions under consecutive keys from `first`."""
    var index = GeospatialIndex.init_ram(library)
    if count != 0:
        index.on(executor).insert(_keys(first, count), _stops(count, stream))
    return index^


def test_geospatial_index_starts_empty_and_refuses_what_it_lacks() raises:
    """A fresh host index is empty on a CPU class; sight lines need a terrain, and a terrain needs its steps."""
    if not _library_is_present():
        _skipped(library_missing_k)
        return
    var library = Library()
    var index = GeospatialIndex.init_ram(library)
    var described = index.describe()
    assert_equal(Int(described.size), 0, "a fresh index is empty")
    assert_true(described.cpu_class != CPUClass.UNKNOWN and described.gpu_class == GPUClass.UNKNOWN, "on a CPU")
    var stops = [GeospatialObservation(48.8566, 2.3522)]
    index.insert(_keys(1, 1), stops)
    var arena = Arena(library)
    var status = Status.SUCCESS
    try:
        _ = index.search(stops, 1, arena, GeospatialSearchConfig(relation=GeospatialRelation.VISIBLE))
    except error:
        status = error.status
    assert_true(status == Status.UNSUPPORTED, "a line of sight over no terrain is refused")

    var heights = [Float32(35), 35, 35, 35]
    status = Status.SUCCESS
    try:
        _ = GeospatialIndex.init_ram(
            library, GeospatialOptions(GeospatialTerrain.of(2, 2, 48.86, 2.35, -0.01, 0.01, Span(heights)))
        )
    except error:
        status = error.status
    assert_true(status == Status.CONFIG_MISMATCH, "a terrain whose rows step north mismatches")
    var hilly = GeospatialIndex.init_ram(
        library, GeospatialOptions(GeospatialTerrain.of(2, 2, 48.86, 2.35, 0.01, 0.01, Span(heights)))
    )
    hilly.insert(_keys(1, 1), stops)
    _ = hilly.search(stops, 1, arena, GeospatialSearchConfig(relation=GeospatialRelation.VISIBLE))


def test_geospatial_search_within_get_and_count() raises:
    """An observation finds itself at zero metres, a radius reports only its neighbors, and it reads back rounded."""
    if not _library_is_present():
        _skipped(library_missing_k)
        return
    var library = Library()
    var executor = Executor.for_ram(library, threads=2)
    var stops = [
        GeospatialObservation(48.8566, 2.3522),
        GeospatialObservation(48.8656, 2.3522),
        GeospatialObservation(52.52, 13.405),
    ]
    var index = GeospatialIndex.init_ram(library)
    index.on(executor).reserve(3)
    index.on(executor).insert(_keys(1, 3), stops)
    assert_equal(Int(index.describe().size), 3, "three observations stored")

    var arena = Arena(executor)
    var matches = index.on(executor).search(Span(stops)[1:2], 1, arena)
    assert_true(matches.query_keys(0)[0] == Key(integer=2), "the observation finds itself first")
    assert_equal(matches.query_distances(0)[0], 0, "at zero chord metres")

    var nearby = index.on(executor).within(Span(stops)[0:1], 2000, arena)
    assert_equal(len(nearby.query_keys(0)), 2, "a kilometre away falls inside two, Berlin does not")
    assert_equal(nearby.written, 2, "and the report counts both")
    var deferred = index.deferred(executor).within(Span(stops)[2:3], 2000, arena)
    assert_true(deferred.wait().query_keys(0)[0] == Key(integer=3), "a deferred radius finds Berlin alone")

    assert_equal(Int(index.count(Key(integer=1)).entries), 1, "one observation under the key")
    var exported = index.get(Key(integer=1), arena)
    assert_equal(len(exported), 1, "one observation exported")
    assert_almost_equal(exported.view()[0].latitude_degrees, 48.8566, atol=1e-6, msg="within centimetres")
    assert_equal(len(index.get(Key(integer=9), arena)), 0, "an absent key exports nothing")


def test_geospatial_checkpoints_refine_and_reshard() raises:
    """A monolith round-trips, a stopped round resumes, sources gather, and malformed lists move nothing."""
    if not _library_is_present():
        _skipped(library_missing_k)
        return
    var library = Library()
    var executor = Executor.for_ram(library, threads=2)
    var index = _places(library, executor, 16)
    var arena = Arena(executor)
    var monolith = index.on(executor).save_buffer(arena)
    assert_true(Int(geospatial_metadata_read_buffer(executor, monolith.bytes()).connectivity) > 0, "the header")
    var restored = GeospatialIndex.load_ram_buffer(executor, monolith.bytes())
    assert_equal(Int(restored.describe().size), 16, "every entry loads back")

    var plan = GeospatialRefinePlan()
    var progress = PhaseProgress()
    var reports = stack_allocation[1, Int]().unsafe_origin_cast[MutAnyOrigin]()
    reports[unsafe_offset=0] = 0
    try:
        _ = index.on(executor).refine(
            GeospatialRefineConfig(
                progress=ProgressArguments.of[_stop_at_first_report](reports),
                plan=Pointer(to=plan),
                phase=Pointer(to=progress),
            )
        )
    except error:
        assert_true(error.status == Status.CANCELLED, "only the handler stops the round")
    _ = index.on(executor).refine(GeospatialRefineConfig(plan=Pointer(to=plan), phase=Pointer(to=progress)))
    assert_true(progress.phase == Phase.DONE, "the next call ends it")
    assert_equal(Int(index.on(executor).refine_plan_finish(plan^).reachability.audited), 16, "the plan saw all")

    var parts = List[GeospatialIndex]()
    for stream in range(2):
        parts.append(_places(library, executor, 8, 8 * stream, stream))
    _ = geospatial_reshard(executor, parts, Span(parts)[0:1])
    assert_equal(Int(parts[0].describe().size), 16, "every entry gathers into the target")
    assert_equal(Int(parts[1].describe().size), 0, "the other source ends empty")
    var status = Status.SUCCESS
    try:
        _ = geospatial_reshard(executor, Span(parts)[0:1], Span(parts)[0:1])
    except error:
        status = error.status
    assert_true(status == Status.UNSUPPORTED, "one index as its only source and target has nothing to recut")
    status = Status.SUCCESS
    try:
        _ = geospatial_reshard(executor, Span(parts)[0:0], Span(parts)[0:1])
    except error:
        status = error.status
    assert_true(status == Status.INVALID_ARGUMENT, "an empty list is malformed")
    parts[1].on(executor).insert(_keys(64, 1), _stops(1, 9))
    status = Status.SUCCESS
    try:
        _ = geospatial_reshard(executor, Span(parts)[0:1], Span(parts)[1:2])
    except error:
        status = error.status
    assert_true(status == Status.CONFIG_MISMATCH, "a target holding foreign entries mismatches")
    assert_equal(Int(parts[0].describe().size + parts[1].describe().size), 17, "every refusal moves nothing")


def test_readme_quickstart() raises:
    """The sequence, shape and geospatial examples of `mojo/README.md` run line for line."""
    if not _library_is_present():
        _skipped(library_missing_k)
        return
    var library = Library()
    var arena = Arena(library)

    var sequences = SequenceIndex.init_ram(library, SequenceOptions(SequenceMetric.LEVENSHTEIN_UTF8))
    var words = Tape[DType.uint8]()
    for word in ["kitten", "sitting", "mitten"]:
        words.append(word)
    sequences.insert([Key(integer=1), Key(integer=2), Key(integer=3)], words.view())
    var query = Tape[DType.uint8]()
    query.append("sittin")
    var matches = sequences.search(query.view(), 1, arena)
    assert_true(matches.query_keys(0)[0] == Key(integer=2), "the nearest word")
    assert_equal(matches.query_distances(0)[0], 1, "one edit away")

    var shapes = ShapesIndex.init_ram(library)
    var clouds = Tape[DType.float32, 3]()
    clouds.append([Float32(0), 0, 0, 1.1, 0, 0])
    clouds.append([Float32(0), 0, 0, 1.5, 0, 0, 0, 1.5, 0])
    shapes.insert([Key(integer=1), Key(integer=2)], clouds.view())
    var probe = Tape[DType.float32, 3]()
    probe.append([Float32(0), 0, 0, 1, 0, 0])
    assert_true(shapes.search(probe.view(), 1, arena).query_keys(0)[0] == Key(integer=1), "the nearer pair")

    var places = GeospatialIndex.init_ram(library)
    var stops = [
        GeospatialObservation(48.8566, 2.3522),
        GeospatialObservation(48.8656, 2.3522),
        GeospatialObservation(52.52, 13.405),
    ]
    places.insert([Key(integer=1), Key(integer=2), Key(integer=3)], stops)
    var nearby = places.within(Span(stops)[0:1], 2000, arena)
    assert_equal(len(nearby.query_keys(0)), 2, "both Paris stops within two kilometres")


# endregion Geospatial


def main() raises:
    var suite = TestSuite()
    suite.test[test_scalar_widths_round_sub_byte_rows_up]()
    suite.test[test_scalar_kinds_follow_their_element_types]()
    suite.test[test_keys_hold_integer_identities]()
    suite.test[test_keys_render_as_dashed_uuids]()
    suite.test[test_rows_derive_contiguous_strides]()
    suite.test[test_shapes_the_lengths_cannot_support_are_refused]()
    suite.test[test_telemetry_starts_at_an_untaken_walk]()
    suite.test[test_engine_round_trips_when_present]()
    suite.test[test_width_mismatch_is_refused_by_the_engine]()
    suite.test[test_names_round_trip_through_the_library_and_refuse_strangers]()
    suite.test[test_get_exports_every_entry_under_a_key]()
    suite.test[test_arena_is_reused_and_results_outlive_the_call]()
    suite.test[test_arena_survives_a_failed_call]()
    suite.test[test_caller_thread_executor_drives_every_verb]()
    suite.test[test_buffers_round_trip_through_the_arena]()
    suite.test[test_search_stream_pulls_in_distance_order]()
    suite.test[test_cursor_walks_every_row_once]()
    suite.test[test_health_audits_every_entry]()
    suite.test[test_refine_plans_stay_live_until_finished]()
    suite.test[test_reshard_gathers_its_sources]()
    suite.test[test_reshard_cuts_and_sample_draws]()
    suite.test[test_mutating_a_view_is_refused_as_read_only]()
    suite.test[test_nvme_views_resolve_keys_through_their_key_table]()
    suite.test[test_reshards_in_place_and_loads_only_manifests]()
    suite.test[test_scans_find_every_query_itself]()
    suite.test[test_progress_cancels_a_search]()
    suite.test[test_predicate_filters_keys]()
    suite.test[test_deferred_calls_wait_until_final]()
    suite.test[test_deferred_failures_repeat]()
    suite.test[test_deferred_refusals_finish_at_submission]()
    suite.test[test_deferred_results_view_the_callers_arena]()
    suite.test[test_awaited_fan_out_answers_both_shards]()
    suite.test[test_awaited_failures_raise]()
    suite.test[test_arenas_free_before_their_executor]()
    suite.test[test_multiplication_reports_every_scored_pair]()
    suite.test[test_late_interaction_scores_over_documents]()
    suite.test[test_attention_is_exact_over_every_key]()
    suite.test[test_sequence_index_starts_empty_and_keeps_its_metric]()
    suite.test[test_sequence_search_get_and_count]()
    suite.test[test_sequence_saves_and_loads_a_buffer]()
    suite.test[test_sequence_refine_plan_resumes_and_finishes]()
    suite.test[test_sequence_reshard_gathers_cuts_and_refuses]()
    suite.test[test_shapes_index_starts_empty_and_keeps_its_metric]()
    suite.test[test_shapes_search_get_and_count]()
    suite.test[test_shapes_checkpoints_refine_and_reshard]()
    suite.test[test_geospatial_index_starts_empty_and_refuses_what_it_lacks]()
    suite.test[test_geospatial_search_within_get_and_count]()
    suite.test[test_geospatial_checkpoints_refine_and_reshard]()
    suite.test[test_readme_quickstart]()
    suite^.run()
