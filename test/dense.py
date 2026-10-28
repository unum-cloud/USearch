"""Vector search over equidimensional dense embeddings - the Python twin of `test/dense.cpp`.

Every recall claim is checked against `usearch.eval.exact_matches`, which scores with NumKong's
kernels in the index's own metric vocabulary, so a number here compares like with like.

Run:
    python -m pytest test/dense.py -v
"""

import asyncio
import gc
import os
import weakref
from collections.abc import Callable
from itertools import pairwise
from operator import attrgetter
from threading import Thread

import numpy as np
import numpy.typing as npt
import pytest
from base import (
    BATCH_SIZES,
    BF16_EPSILON,
    INPUT_DTYPES,
    QUANTIZATIONS,
    SETTINGS,
    SPATIAL_METRICS,
    bf16_rounded,
    bf16_words,
    build_index,
    distance_tolerances,
    measured_recall,
    recall_floor,
)

import numkong as nk
from usearch import (
    Arena,
    CancelledError,
    CapacityOverflowError,
    ConfigMismatchError,
    ContendedError,
    CPUClass,
    Deferred,
    DeferredOn,
    DenseIndex,
    DenseIndexDescription,
    DenseMetadata,
    DenseMetric,
    DenseProducts,
    DenseReachability,
    DenseTraversalStats,
    DenseVectors,
    Executor,
    GPUClass,
    InvalidArgumentError,
    Key,
    Match,
    Matches,
    Monolith,
    Mood,
    NVMeClass,
    On,
    ReadOnlyError,
    Residency,
    Scalar,
    ShapeMismatchError,
    Status,
    Traversal,
    UninitializedError,
    UnknownKindError,
    UnsupportedError,
)
from usearch.eval import random_vectors, self_recall
from usearch.keys import integers_to_keys, keys_to_integers

# region Construction


@pytest.mark.parametrize("dimensions", SETTINGS.dims)
@pytest.mark.parametrize("metric", SPATIAL_METRICS, ids=attrgetter("name"))
def test_construction_reports_its_geometry(dimensions: int, metric: DenseMetric):
    """Every setting the factory took reads back off the handle, the modes as their enums."""
    index = DenseIndex.init_ram(dimensions=dimensions, metric=metric, quantization=Scalar.BF16, connectivity=16)
    assert index.dimensions == dimensions
    assert index.connectivity == 16
    assert len(index) == 0
    assert isinstance(index.cpu_class, CPUClass) and index.gpu_class is None
    assert repr(index) == f"DenseIndex(dimensions={dimensions}, size=0)"
    description = index.describe()
    assert isinstance(description, DenseIndexDescription)
    assert description.metadata == (dimensions, metric, Scalar.BF16, 16)
    assert description.metadata.metric is metric
    assert description.size == 0
    assert description.shards_count == index.shards_count >= 1
    assert description.cpu_class is index.cpu_class and description.gpu_class is None


def test_construction_takes_a_pinned_class():
    """A serial pin runs on every CPU, and the index it opens reports the pinned class and its rows."""
    index = DenseIndex.init_ram(dimensions=8, cpu_class=CPUClass.SERIAL)
    index.reserve(2)
    index.insert([1, 2], np.ones((2, 8), dtype=np.float32))
    assert len(index) == 2 and index.cpu_class is CPUClass.SERIAL


def test_fresh_index_inserts_without_a_reservation():
    """Init freezes the geometry, so a fresh index takes rows and grows for them with no `reserve`."""
    index = DenseIndex.init_ram(dimensions=8)
    index.insert(np.arange(100), np.random.default_rng(42).random((100, 8), dtype=np.float32))
    assert len(index) == 100 and index.count(7) == 1


@pytest.mark.parametrize(
    ("options", "failure"),
    [
        ({"metric": "angular"}, TypeError),  # Modes are enums, never strings.
        ({"quantization": "bf16"}, TypeError),
        ({"quantization": True}, TypeError),  # A bool would read as an enumerator.
        ({"metric": 99}, UnknownKindError),  # An integer no enumerator carries.
        ({"quantization": Scalar.F32}, UnknownKindError),  # Rows cross the boundary as `f32` but are never stored so.
        ({"residency": Scalar.F32}, TypeError),  # The factory chooses the residency; no keyword does.
        ({"devices": [0]}, TypeError),  # A variant belonging to another residency.
        ({"threads": 2}, TypeError),  # An index owns rows; an `Executor` owns workers.
        ({"expansion_add": 64}, TypeError),  # A keyword the surface does not have.
        ({"ndim": 8}, TypeError),  # Renamed to 'dimensions'.
    ],
)
def test_construction_rejects_unknown_options(options: dict, failure: type):
    """A name or a mode the factory does not have is refused, not quietly ignored."""
    with pytest.raises(failure):
        DenseIndex.init_ram(dimensions=8, **options)


def test_construction_takes_three_positionals():
    """The three settings every caller names may be positional, and only once; the rest must be keywords."""
    index = DenseIndex.init_ram(32, DenseMetric.EUCLIDEAN, Scalar.BF16)
    assert index.dimensions == 32
    with pytest.raises(TypeError):
        DenseIndex.init_ram(32, DenseMetric.EUCLIDEAN, Scalar.BF16, 16)
    with pytest.raises(TypeError):
        DenseIndex.init_ram(32, dimensions=16)


@pytest.mark.parametrize("handle_type", [DenseIndex, Executor, On, DeferredOn])
def test_direct_construction_names_the_factories(handle_type: type):
    """No handle is built by calling its type; the refusal says what to call instead."""
    with pytest.raises(TypeError):
        handle_type()


# endregion Construction

# region Executors


def test_vram_executor_takes_no_thread_count():
    """One host thread drives one stream per device, so the device list is the pool's only width."""
    with pytest.raises(TypeError):
        Executor.for_vram(threads=4)


def test_executor_records_its_residency():
    """An executor names where its calls compute, which is where the outputs they size land."""
    assert Executor.for_ram(threads=1).residency is Residency.RAM


def test_ram_executor_takes_memory_domain_ordinals():
    """The NUMA nodes its workers stay local to arrive as ordinals, and nothing else is accepted."""
    Executor.for_ram(memory_domains=[0]).drain()
    with pytest.raises(TypeError):
        Executor.for_ram(memory_domains=["x"])
    with pytest.raises(ValueError):
        Executor.for_ram(memory_domains=[2**40])


def test_drain_takes_a_mood():
    """The idle mood is a `Mood`, and neither a name nor another integer is guessed at."""
    pool = Executor.for_ram(threads=1, deferred_limit=4)
    pool.drain(Mood.SPINNING)
    pool.drain()
    with pytest.raises(TypeError):
        pool.drain("parked")
    with pytest.raises(ValueError):
        pool.drain(7)


def test_class_enums_carry_the_library_spellings():
    """Every class member is named by the library's spelling of its value, in value order, from one up."""
    spellings = {
        CPUClass: "serial intel-haswell intel-sierra intel-skylake intel-icelake amd-genoa intel-sapphire intel-granite"
        " intel-diamond aws-g2 apple-m2 apple-m4 aws-g5 nvidia-vera alibaba-c930 wasm32-v128 wasm64-extended",
        GPUClass: "nvidia-ampere nvidia-ada nvidia-hopper nvidia-blackwell nvidia-blackwellrtx nvidia-rubin amd-cdna3"
        " amd-cdna4 amd-cdna5 apple-metal7 apple-metal9 apple-metal10",
        NVMeClass: "posix linux-aio linux-uring linux-uring-passthrough linux-spdk windows-ioring",
    }
    for enum, spelled in spellings.items():
        expected = [(code, name.upper().replace("-", "_")) for code, name in enumerate(spelled.split(), start=1)]
        assert [(member.value, member.name) for member in enum] == expected


def test_nvme_executor_takes_an_nvme_class():
    """The engine an NVMe executor submits through is an `NVMeClass`, and nothing else is guessed at."""
    assert Executor.for_nvme(threads=1, nvme_class=NVMeClass.POSIX).residency is Residency.NVME
    with pytest.raises(TypeError):
        Executor.for_nvme(threads=1, nvme_class="posix")


def test_borrowed_pool_must_be_named():
    """A pool arrives by address, and a null one is refused rather than dereferenced, where the build
    serves borrowed pools at all."""
    with pytest.raises((InvalidArgumentError, UnsupportedError)):
        Executor.on_pool(0)


def test_executor_closes_with_its_block():
    """Leaving a `with` block waits for the executor's calls and closes it, so a later receiver is refused."""
    index, _ = build_index(8, 8, Executor.for_ram(threads=1))
    with Executor.for_ram(threads=1) as pool:
        assert index.on(pool).count(0) == 1
    with pytest.raises(UninitializedError):
        index.on(pool)


_CALLER_THREAD_VERBS: list[tuple[str, Callable[[DenseIndex, npt.NDArray[np.float32]], object]]] = [
    ("insert", lambda index, rows: index.insert([9], rows[:1])),
    ("refine", lambda index, rows: index.refine()),
    ("search", lambda index, rows: index.search(rows[:1], 1, Arena())),
    ("search_stream", lambda index, rows: index.search_stream(rows[0]).next(1, Arena())),
    ("save_buffer", lambda index, rows: index.save_buffer(Arena())),
    ("get", lambda index, rows: index.get(0, Arena())),
    ("count", lambda index, rows: index.count(0)),
]
"""One entry per verb, called on the index itself, which runs it on the caller's thread."""


@pytest.mark.parametrize(("verb", "call"), _CALLER_THREAD_VERBS, ids=[name for name, _ in _CALLER_THREAD_VERBS])
def test_verbs_run_on_the_callers_thread(executor, verb: str, call):
    """The index itself is a receiver too: its verbs run on the caller's thread rather than refusing."""
    index, vectors = build_index(8, 8, executor)
    try:
        call(index, vectors)
    except UnsupportedError:
        assert verb == "search_stream", "only search streams are declared before they are served"


@pytest.mark.parametrize("wrong", [2, "ram", None])
def test_receivers_take_only_an_executor(executor, wrong):
    """A pool is a handle, not a thread count or a name, so anything else is refused where it is named."""
    index, _ = build_index(8, 8, executor)
    with pytest.raises(TypeError):
        index.on(wrong)
    with pytest.raises(TypeError):
        index.deferred(wrong)


def test_executor_is_borrowed_for_the_call(executor):
    """A verb borrows the workers for its call and never retains them, so a temporary one serves."""
    index, vectors = build_index(16, 8, Executor.for_ram())
    assert len(index) == 16
    assert len(index.on(executor).search(vectors[:1], 1, Arena(executor))) == 1


def test_one_executor_serves_several_indexes(executor):
    """An index owns rows and an executor owns workers, so one pool drives as many as asked."""
    first, vectors = build_index(16, 8, executor)
    second, _ = build_index(16, 8, executor, seed=1)
    assert len(first) == len(second) == 16
    assert len(first.on(executor).search(vectors[:2], 1, Arena(executor))) == 2


def test_concurrent_searches_take_one_executor_apiece(executor, spawn_executor):
    """A handle admits any number of callers, and each thread here blocks on an executor of its own."""
    index, vectors = build_index(128, 16, executor)
    pools = [spawn_executor() for _ in range(4)]
    answers: dict[int, Matches] = {}

    def resolve(thread_index: int):
        pool = pools[thread_index]
        answers[thread_index] = index.on(pool).search(vectors[:8], 4, Arena(pool))

    threads = [Thread(target=resolve, args=(thread_index,)) for thread_index in range(len(pools))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(answers) == len(pools), "every thread must have resolved its own batch"
    for matches in answers.values():
        assert np.asarray(matches.counts).shape == (8,)


def test_arena_is_made_by_an_executor_or_none():
    """An executor makes an arena where it computes, none makes one for the calling thread, and nothing else does."""
    with Executor.for_ram(threads=1) as pool:
        assert isinstance(Arena(pool), Arena) and isinstance(Arena(), Arena)
    with pytest.raises(UninitializedError):
        Arena(pool)
    with pytest.raises(TypeError):
        Arena(2)


@pytest.mark.parametrize("arena", [None, 0])
def test_output_verbs_require_an_arena(executor, arena):
    """No output verb allocates on its own, so one handed no arena, or anything else, is refused."""
    index, vectors = build_index(8, 8, executor)
    with pytest.raises(TypeError):
        index.search(vectors[:1], 1)
    with pytest.raises(TypeError):
        index.search(vectors[:1], 1, arena)
    with pytest.raises(TypeError):
        index.cursor().next(1, arena)


# endregion Executors

# region Receivers


def test_receivers_answer_alike(executor):
    """The receiver chooses where the call runs and the verb stays, so all three answer the same matches."""
    index, vectors = build_index(64, 16, executor)
    on_caller = keys_to_integers(index.search(vectors[:4], 5, Arena()).keys)
    on_executor = keys_to_integers(index.on(executor).search(vectors[:4], 5, Arena(executor)).keys)
    call = index.deferred(executor).search(vectors[:4], 5, Arena(executor))
    assert isinstance(call, Deferred)
    assert call.status == Status.DEFERRED
    deferred = keys_to_integers(call.wait().keys)
    assert np.array_equal(on_caller, on_executor)
    assert np.array_equal(on_caller, deferred)


def test_deferred_call_is_deferred_until_waited(executor):
    """A submitted call reads `DEFERRED` until it is final, then answers what a blocking one would."""
    index = DenseIndex.init_ram(16, DenseMetric.ANGULAR, Scalar.BF16)
    index.reserve(8)
    call = index.deferred(executor).insert(np.arange(8), random_vectors(8, 16))
    assert call.status is Status.DEFERRED
    assert repr(call) == "Deferred(status=deferred)"
    assert call.wait() is None
    assert call.status is Status.SUCCESS
    assert call.wait() is None, "a final call answers as many times as it is asked"
    assert len(index) == 8


def test_deferred_call_owns_its_inputs(executor):
    """The rows a deferred call borrows stay exported inside its `Deferred`, so the caller may drop its own names."""
    index = DenseIndex.init_ram(16, DenseMetric.ANGULAR, Scalar.BF16)
    index.reserve(4)
    rows = random_vectors(4, 16)
    call = index.deferred(executor).insert(list(range(4)), rows.copy())
    assert call.status == Status.DEFERRED
    call.wait()
    assert np.allclose(np.asarray(index.get(2, Arena())), rows[2:3], rtol=BF16_EPSILON, atol=BF16_EPSILON)


def test_deferred_call_raises_at_wait(executor):
    """Arguments the binding refuses raise at the call; the engine's own refusals raise once waited for."""
    index, vectors = build_index(16, 8, executor)
    with pytest.raises(ShapeMismatchError):
        index.deferred(executor).insert([1, 2, 3], vectors[:2])
    call = index.deferred(executor).search(vectors[:1, :4], 1, Arena(executor))
    assert call.status == Status.DEFERRED
    for _ in range(2):
        with pytest.raises(ShapeMismatchError) as raised:
            call.wait()
        assert raised.value.status is Status.SHAPE_MISMATCH
    assert call.status is Status.SHAPE_MISMATCH


def test_deferred_call_awaits(executor):
    """`await` runs `wait()` on the event loop's own executor, so a coroutine suspends rather than blocks."""
    index, vectors = build_index(32, 8, executor)

    async def search() -> Matches:
        call = index.deferred(executor).search(vectors[:2], 3, Arena(executor))
        assert call.status == Status.DEFERRED
        return await call

    matches = asyncio.run(search())
    assert keys_to_integers(matches.keys)[:, 0].tolist() == [0, 1]


def test_blocking_call_runs_after_earlier_deferred_ones(executor):
    """Calls on one executor start in submission order, so a blocking one sees what a deferred one wrote."""
    index = DenseIndex.init_ram(8, DenseMetric.ANGULAR, Scalar.BF16)
    index.reserve(4)
    call = index.deferred(executor).insert([7], random_vectors(1, 8))
    assert index.on(executor).count(7) == 1
    assert call.status is Status.SUCCESS


def test_full_deferred_table_answers_contended():
    """A submission finding the deferred table full finishes `CONTENDED` at once, and the earlier call still runs."""
    index, vectors = build_index(16, 8, Executor.for_ram(threads=1))
    pool = Executor.for_ram(threads=1, deferred_limit=1)
    first = index.deferred(pool).count(3)
    second = index.deferred(pool).count(4)
    assert second.status is Status.CONTENDED
    with pytest.raises(ContendedError):
        second.wait()
    assert first.wait() == 1


def test_dropping_an_unfinished_deferred_waits(executor):
    """Dropping a `Deferred` waits for its call, since the call's structure and buffers live inside it."""
    index = DenseIndex.init_ram(8, DenseMetric.ANGULAR, Scalar.BF16)
    index.reserve(4)
    call = index.deferred(executor).insert([1, 2], random_vectors(2, 8))
    assert call.status is Status.DEFERRED
    del call
    assert len(index) == 2


def test_dropped_deferred_in_a_cycle_is_collected_once_final(executor):
    """A predicate holding its own `Deferred` forms a cycle the collector breaks once the call is final."""
    index, vectors = build_index(4, 8, executor)
    witness = type("Witness", (), {})()
    alive = weakref.ref(witness)
    cycle: list = [witness]
    call = index.deferred(executor).search(vectors[:1], 1, Arena(executor), predicate=lambda key, cycle=cycle: True)
    cycle.append(call)
    del witness, cycle, call
    executor.drain()
    gc.collect()
    assert alive() is None


def test_deferred_call_borrows_its_arena(executor):
    """A deferred call borrows its arena as a blocking one does, refusing a second call until it is final."""
    index, vectors = build_index(4, 8, executor)
    arena = Arena(executor)
    call = index.deferred(executor).get(1, arena)
    assert call.status == Status.DEFERRED
    with pytest.raises(ContendedError):
        index.get(2, arena)
    rows = call.wait()
    assert np.allclose(np.asarray(rows), vectors[1:2], rtol=BF16_EPSILON, atol=BF16_EPSILON)
    assert np.asarray(index.get(2, arena)).shape == (1, 8)


def test_deferred_receiver_lacks_what_cannot_defer(executor):
    """Cursors and streams block on every pull and factories address no index, so the receivers that cannot run them
    lack them."""
    index, _ = build_index(4, 8, executor)
    assert not hasattr(index.deferred(executor), "cursor")
    assert not hasattr(index.deferred(executor), "search_stream")
    assert not hasattr(index.on(executor), "init_ram")
    assert hasattr(index.on(executor), "cursor")
    assert type(index.on(executor)) is On and type(index.deferred(executor)) is DeferredOn
    assert repr(index.on(executor)) == f"On({index!r}, {executor!r})"


# endregion Receivers

# region Ingest


@pytest.mark.parametrize("count", BATCH_SIZES)
@pytest.mark.parametrize("dimensions", SETTINGS.dims)
def test_insert_publishes_every_row(executor, count: int, dimensions: int):
    """A batch is all-or-nothing, so every row of an accepted batch is visible afterwards."""
    index, _ = build_index(count, dimensions, executor)
    assert len(index) == count
    assert index.capacity >= count


@pytest.mark.parametrize("quantization", QUANTIZATIONS, ids=attrgetter("name"))
def test_insert_accepts_every_quantization(executor, quantization: Scalar):
    """Every stored scalar type takes the same `float32` rows and quantizes them on ingest."""
    index, _ = build_index(64, 96, executor, quantization=quantization)
    assert len(index) == 64


@pytest.mark.parametrize("input_dtype", INPUT_DTYPES)
def test_insert_accepts_any_input_dtype(executor, input_dtype):
    """The query type is free to differ from the stored quantization."""
    index = DenseIndex.init_ram(dimensions=32, quantization=Scalar.BF16)
    index.reserve(16)
    vectors = (np.random.default_rng(0).random((16, 32)) * 10).astype(input_dtype)
    index.on(executor).insert(np.arange(16), vectors)
    assert len(index) == 16


def test_insert_accepts_packed_rows_by_logical_width(executor):
    """A `u1x8` row counts the bits it packs, so an eight-byte row is a sixty-four-bit vector."""
    index = DenseIndex.init_ram(dimensions=64, metric=DenseMetric.HAMMING, quantization=Scalar.U1X8)
    index.reserve(16)
    rows = random_vectors(8, 64, metric=DenseMetric.HAMMING)
    assert rows.shape == (8, 8), "sixty-four bits pack into eight bytes"
    index.on(executor).insert(np.arange(8), rows, dtype=Scalar.U1X8)
    assert len(index) == 8

    narrow = random_vectors(8, 32, metric=DenseMetric.HAMMING)
    with pytest.raises(ShapeMismatchError):
        index.on(executor).insert(np.arange(8, 16), narrow, dtype=Scalar.U1X8)


def test_insert_refuses_a_dtype_narrower_than_the_buffer(executor):
    """A named kind reinterprets the buffer's bytes, so one narrower than its items would misread every row."""
    index = DenseIndex.init_ram(dimensions=8, quantization=Scalar.BF16)
    index.reserve(4)
    with pytest.raises(UnknownKindError):
        index.insert(np.arange(4), random_vectors(4, 8), dtype=Scalar.F16)
    assert len(index) == 0


def test_insert_accepts_numkong_tensors_without_numpy(executor):
    """A NumKong tensor is a `memoryview`, so it reaches the index through PEP 3118 alone.

    This is the claim the README leads with, and the only test that exercises it: every other
    path in this file arrives as a NumPy array.
    """
    index = DenseIndex.init_ram(dimensions=16, quantization=Scalar.BF16)
    index.reserve(8)
    rows = nk.Tensor(random_vectors(8, 16))
    assert memoryview(rows).shape == (8, 16), "NumKong tensors export PEP 3118 directly"
    index.on(executor).insert(np.arange(8), rows)
    assert len(index) == 8


@pytest.mark.parametrize("keys", [[Key(11), 12], np.arange(2), integers_to_keys(np.arange(2))])
def test_insert_accepts_every_key_spelling(executor, keys):
    """Keys arrive as objects, as integers, or as the packed wire rows."""
    index = DenseIndex.init_ram(dimensions=4, quantization=Scalar.BF16)
    index.reserve(2)
    index.on(executor).insert(keys, random_vectors(2, 4))
    assert len(index) == 2


def test_insert_gathers_integer_vectors_as_keys(executor):
    """An integer vector of any width or stride reads as the identities its integers spell, negatives refused."""
    index = DenseIndex.init_ram(dimensions=4, quantization=Scalar.BF16)
    index.reserve(8)
    for keys in (
        np.array([7, 200], np.uint8),
        np.array([7, 300], np.int16),
        np.array([2**64 - 1, 5], np.uint64),
        np.arange(10, 14, dtype=np.int64)[::2],
    ):
        index.on(executor).insert(keys, random_vectors(2, 4))
    spelled = (7, 200, 300, 2**64 - 1, 5, 10, 12, 11)
    assert [index.count(key) for key in spelled] == [2, 1, 1, 1, 1, 1, 1, 0]
    with pytest.raises(OverflowError):
        index.on(executor).insert(np.array([3, -1], np.int32), random_vectors(2, 4))
    assert len(index) == 8


def test_insert_rejects_mismatched_key_count(executor):
    """A key per row, or the batch is refused before anything is published."""
    index = DenseIndex.init_ram(dimensions=4, quantization=Scalar.BF16)
    index.reserve(4)
    with pytest.raises(ShapeMismatchError):
        index.on(executor).insert([1, 2, 3], random_vectors(2, 4))
    assert len(index) == 0


def test_insert_reports_telemetry(executor):
    """A batch reports what it walked into the `DenseTraversalStats` a caller passes."""
    index = DenseIndex.init_ram(dimensions=16, quantization=Scalar.BF16)
    index.reserve(128)
    stats = DenseTraversalStats()
    index.on(executor).insert(np.arange(64), random_vectors(64, 16), stats=stats)
    assert stats.visited_members > 0
    assert stats.computed_distances > 0


def test_insert_reports_progress(executor):
    """Reports never go backwards, the last one says the whole batch finished, and a handler answering None carries on."""
    index = DenseIndex.init_ram(dimensions=16, quantization=Scalar.BF16)
    index.reserve(64)
    seen: list[tuple[int, int]] = []

    def watch(completed: int, total: int) -> None:
        seen.append((completed, total))

    index.on(executor).insert(np.arange(64), random_vectors(64, 16), progress=watch)
    assert len(index) == 64
    assert seen, "a watched batch reports at least once"
    assert all(earlier[0] <= later[0] for earlier, later in pairwise(seen)), "counts never decrease"
    assert {total for _, total in seen} == {64}, "every report names the same total"
    assert seen[-1] == (64, 64)


def test_insert_handler_reports_and_cannot_stop(executor):
    """Every row is claimed before the first report, so a handler answering False still lands the whole batch."""
    index = DenseIndex.init_ram(dimensions=16, quantization=Scalar.BF16)
    index.reserve(64)
    index.on(executor).insert(np.arange(64), random_vectors(64, 16), progress=lambda done, total: done < 8)
    assert len(index) == 64


def test_insert_surfaces_a_raising_handler(executor):
    """A handler's own exception outranks the cancellation it turned into."""
    index = DenseIndex.init_ram(dimensions=16, quantization=Scalar.BF16)
    index.reserve(8)

    failure = ValueError()

    def broken(completed: int, total: int) -> bool:
        raise failure

    with pytest.raises(ValueError) as raised:
        index.on(executor).insert(np.arange(8), random_vectors(8, 16), progress=broken)
    assert raised.value is failure
    assert len(index) == 8, "the batch landed before the handler's exception surfaced"


# endregion Ingest

# region Search


@pytest.mark.parametrize("dimensions", SETTINGS.dims)
def test_search_shapes(executor, dimensions: int):
    """The three outputs are shaped by the batch and the "k" that was asked for."""
    index, vectors = build_index(64, dimensions, executor)
    stats = DenseTraversalStats()
    matches: Matches = index.on(executor).search(vectors[:3], 5, Arena(executor), stats=stats)

    assert np.asarray(matches.keys).shape == (3, 5, 16)
    assert np.asarray(matches.distances).shape == (3, 5)
    assert np.asarray(matches.counts).shape == (3,)
    assert len(matches) == 3
    assert stats.computed_distances > 0


def test_search_rows_are_matches(executor):
    """A result row is a tuple of scored entries, nearest first."""
    index, vectors = build_index(64, 32, executor)
    row = index.on(executor).search(vectors[:1], 4, Arena(executor))[0]
    assert all(isinstance(match, Match) for match in row)
    assert isinstance(row[0].key, Key)
    assert row[0].distance <= row[-1].distance


def test_search_reports_fewer_matches_than_asked(executor):
    """A count bounds its own row, so an index holding less than `wanted` still answers."""
    index, vectors = build_index(3, 8, executor)
    matches = index.on(executor).search(vectors[:1], 10, Arena(executor))
    count = int(np.asarray(matches.counts)[0])
    assert count <= 3
    assert len(matches[0]) == count


def test_search_reuses_one_arena(executor):
    """One arena serves a whole sweep, each call rewinding what the one before it wrote."""
    index, vectors = build_index(64, 16, executor)
    arena = Arena(executor)
    first = keys_to_integers(index.on(executor).search(vectors[:2], 4, arena).keys)
    second = index.on(executor).search(vectors[4:6], 4, arena)
    assert not np.array_equal(first, keys_to_integers(second.keys))
    assert int(np.asarray(second.counts)[0]) == 4


def test_search_rejects_an_overflowing_request(executor):
    """An output matrix that cannot fit memory is refused rather than silently wrapped."""
    index, vectors = build_index(4, 8, executor)
    with pytest.raises(CapacityOverflowError) as raised:
        index.on(executor).search(vectors[:2], 2**62, Arena(executor))
    assert raised.value.status is Status.OVERFLOW


@pytest.mark.parametrize("dimensions", [97, 256])
@pytest.mark.parametrize("metric", SPATIAL_METRICS, ids=attrgetter("name"))
def test_recall_against_the_oracle(executor, dimensions: int, metric: DenseMetric):
    """The index must return most of what a brute-force scan would, in every metric."""
    index, vectors = build_index(512, dimensions, executor, metric=metric)
    stored = bf16_rounded(vectors)
    recall = measured_recall(index, stored, stored[:64], 10, executor, metric=metric)
    assert recall >= recall_floor(Scalar.BF16), f"{metric} recall collapsed to {recall}"


@pytest.mark.parametrize("quantization", QUANTIZATIONS, ids=attrgetter("name"))
def test_recall_survives_quantization(executor, quantization: Scalar):
    """Every stored scalar type still finds what it should, at its own floor."""
    index, _ = build_index(512, 96, executor, quantization=quantization)
    recall = self_recall(index.on(executor), Arena(executor)).mean_recall
    assert recall >= recall_floor(quantization), f"{quantization} recall collapsed to {recall}"


@pytest.mark.parametrize("quantization", [Scalar.BF16, Scalar.F16, Scalar.I8], ids=attrgetter("name"))
def test_distances_agree_with_the_oracle(executor, quantization: Scalar):
    """A reported distance matches what NumKong computes for the same pair, within tolerance."""
    index, vectors = build_index(128, 64, executor, quantization=quantization)
    queries = vectors[:8]
    matches = index.on(executor).search(queries, 1, Arena(executor))
    found = keys_to_integers(matches.keys)[:, 0]
    reported = np.asarray(matches.distances)[:, 0]
    truth = np.asarray(nk.cdist(queries, vectors, "angular", out_dtype="float64"))
    expected = truth[np.arange(len(queries)), found]
    atol, rtol = distance_tolerances(quantization)
    np.testing.assert_allclose(reported, expected, atol=atol, rtol=rtol)


def test_expansion_rides_the_call(executor):
    """The breadth of the candidates is a per-call setting, so one immutable index serves a whole sweep."""
    index, vectors = build_index(512, 64, executor)
    queries = vectors[:32]
    narrow, wide = DenseTraversalStats(), DenseTraversalStats()
    arena = Arena(executor)
    index.on(executor).search(queries, 10, arena, expansion_factor=16, stats=narrow)
    index.on(executor).search(queries, 10, arena, expansion_factor=256, stats=wide)
    assert wide.computed_distances > narrow.computed_distances


def test_telemetry_is_a_per_call_reading(executor):
    """A call overwrites the `DenseTraversalStats` it takes, and `+=` is how a sweep sums the readings."""
    index, vectors = build_index(128, 16, executor)
    queries = vectors[:8]
    stats, total = DenseTraversalStats(), DenseTraversalStats()
    arena = Arena(executor)

    index.on(executor).search(queries, 5, arena, stats=stats)
    first = stats.computed_distances
    assert first > 0
    total += stats

    index.on(executor).search(queries, 5, arena, stats=stats)
    assert stats.computed_distances == first, "a repeat replaces the reading rather than adding to it"
    total += stats
    assert total.computed_distances == 2 * first


def test_filtered_search_admits_only_approved_keys(executor):
    """A predicate narrows the answer without collapsing the walk that produced it."""
    index, vectors = build_index(128, 32, executor)
    matches = index.on(executor).search(vectors[:4], 8, Arena(executor), predicate=lambda key: int(key) % 2 == 0)
    found = keys_to_integers(matches.keys)
    counts = np.asarray(matches.counts)
    for query_index in range(len(counts)):
        assert np.all(found[query_index, : counts[query_index]] % 2 == 0)


def test_filtered_search_propagates_a_raising_predicate(executor):
    """A predicate that raises fails the search rather than silently narrowing it."""
    index, vectors = build_index(64, 16, executor)

    failure = ValueError()

    def broken(key):
        raise failure

    with pytest.raises(ValueError) as raised:
        index.on(executor).search(vectors[:2], 4, Arena(executor), predicate=broken)
    assert raised.value is failure


def test_scan_finds_every_row_itself(executor):
    """A scan scores every row exactly, so each self-query finds its own row first."""
    index, vectors = build_index(64, 16, executor)
    matches = index.on(executor).search(vectors, 4, Arena(executor), traversal=Traversal.SCAN)
    assert keys_to_integers(matches.keys)[:, 0].tolist() == list(range(64))


# endregion Search

# region Retrieval


def test_count_and_contains(executor):
    """Membership and multiplicity are the same question asked two ways."""
    index, _ = build_index(32, 16, executor)
    assert index.count(0) == 1
    assert index.count(10_000) == 0
    assert 0 in index
    assert 10_000 not in index


def test_get_returns_the_stored_rows(executor):
    """A `bf16` index hands back `f32` rows, each value within a `bf16` step of the one inserted."""
    index = DenseIndex.init_ram(dimensions=8, quantization=Scalar.BF16)
    index.reserve(2)
    vectors = random_vectors(2, 8)
    index.on(executor).insert([1, 2], vectors)

    rows = index.get(1, Arena(), outputs_dtype=Scalar.F32)
    assert isinstance(rows, DenseVectors) and not hasattr(rows, "__cuda_array_interface__")
    retrieved = np.asarray(rows)
    assert retrieved.shape == (1, 8)
    assert np.allclose(retrieved[0], vectors[0], rtol=BF16_EPSILON, atol=BF16_EPSILON)


def test_get_answers_empty_and_takes_no_subscript(executor):
    """`get` answers absence with no rows, and a subscript, having no arena to land in, is no verb at all."""
    index, _ = build_index(4, 8, executor)
    absent = index.get(10_000, Arena())
    assert isinstance(absent, DenseVectors) and len(absent) == 0 and not absent
    with pytest.raises(TypeError):
        index[0]


def test_get_leaves_unformattable_kinds_as_bytes(executor):
    """Sub-byte and mini-float kinds have no `memoryview` format, so they leave as raw runs."""
    index = DenseIndex.init_ram(dimensions=8, quantization=Scalar.BF16)
    index.reserve(1)
    index.on(executor).insert([1], random_vectors(1, 8))
    view = memoryview(index.get(1, Arena(), outputs_dtype=Scalar.BF16))
    assert view.format == "B"
    assert view.shape == (1, 16)


def test_arena_refuses_to_rewind_while_a_buffer_is_exported(executor):
    """Like a `bytearray`, an arena exporting a buffer over its outputs refuses the call that would rewind it,
    and the refused call leaves the holder's views readable."""
    index, vectors = build_index(4, 8, executor)
    arena = Arena(executor)
    rows = index.on(executor).get(1, arena)
    exported = memoryview(rows)
    with pytest.raises(ContendedError):
        index.on(executor).get(2, arena)
    assert np.allclose(np.asarray(rows), vectors[1:2], rtol=BF16_EPSILON, atol=BF16_EPSILON)
    exported.release()
    assert np.asarray(index.on(executor).get(2, arena)).shape == (1, 8)


def test_results_refuse_reads_once_their_arena_rewinds(executor):
    """A result kept past the next call over its arena is stale, so every read of it is refused."""
    index, vectors = build_index(16, 8, executor, metric=DenseMetric.INNER)
    arena = Arena(executor)
    matches = index.on(executor).search(vectors[:2], 4, arena)
    products = index.on(executor).multiply(vectors[:2], 4, arena)
    for read in (lambda: matches[0], lambda: memoryview(matches.keys), lambda: memoryview(matches.counts)):
        with pytest.raises(UninitializedError):
            read()
    rows = index.on(executor).get(1, arena)
    with pytest.raises(UninitializedError):
        products.row(0)
    assert np.asarray(rows).shape == (1, 8)


def test_duplicate_keys_stack(executor):
    """Keys need not be unique; several entries may share one and `count` reports how many."""
    index = DenseIndex.init_ram(dimensions=8, quantization=Scalar.BF16)
    index.reserve(3)
    index.on(executor).insert([5, 5, 5], random_vectors(3, 8))
    assert index.count(5) == 3
    assert np.asarray(index.get(5, Arena())).shape == (3, 8)


# endregion Retrieval

# region Iteration


def test_iteration_walks_every_key(executor):
    """A snapshot walk visits each published entry exactly once."""
    index, _ = build_index(300, 16, executor)
    cursor, arena, walked = index.cursor(), Arena(), []
    while batch := cursor.next(64, arena):
        walked.append(np.array(batch.keys))
    assert sorted(keys_to_integers(np.concatenate(walked)).tolist()) == list(range(300))


def test_iteration_carries_links(executor):
    """A pull asked for links carries them as offsets and keys, one asked for none carries keys alone, and each pull
    rewinds the batch before it."""
    index, _ = build_index(40, 12, executor)
    cursor, arena = index.cursor(), Arena()
    batch = cursor.next(16, arena, links=True)
    offsets, linked = batch.links
    view = memoryview(offsets)
    assert view.format == "N" and view.shape == (17,)
    assert len(batch) == 16 and np.asarray(batch.keys).shape == (16, 16)
    assert np.asarray(linked).shape == (view[16], 16)
    with pytest.raises(ContendedError):
        cursor.next(16, arena)
    view.release()
    assert cursor.next(16, arena).links is None
    with pytest.raises(UninitializedError):
        memoryview(offsets)


def test_iteration_walks_from_start_keys(executor):
    """A walk from start keys yields their own entries first, in the order given."""
    index, _ = build_index(40, 12, executor)
    cursor, arena = index.cursor([7, 3]), Arena()
    walked = keys_to_integers(np.array(cursor.next(2, arena).keys)).tolist()
    assert walked == [7, 3]


def test_search_stream_yields_nearest_first(executor):
    """A search stream is resumable, ordered by what the walk has reached, where the build serves one."""
    index, vectors = build_index(128, 24, executor)
    try:
        stream, arena = index.on(executor).search_stream(vectors[0]), Arena(executor)
    except UnsupportedError:
        return
    streamed = [match.distance for match in stream.next(8, arena)[0]]
    streamed += [match.distance for match in stream.next(8, arena)[0]]
    assert len(streamed) == 16 and streamed == sorted(streamed)


def test_walks_close_before_they_are_collected(executor):
    """`close()` and a `with` block release a cursor's snapshot or a stream's walk, after which a pull is refused."""
    index, vectors = build_index(32, 8, executor)
    arena = Arena()
    with index.cursor() as cursor:
        assert len(cursor.next(4, arena)) == 4
    with pytest.raises(UninitializedError):
        cursor.next(4, arena)
    try:
        stream = index.search_stream(vectors[0])
    except UnsupportedError:
        return
    with stream:
        assert int(stream.next(1, arena)[0][0].key) == 0
    with pytest.raises(UninitializedError):
        stream.next(1, arena)


# endregion Iteration

# region Composition


def test_absorb_absorbs_and_consumes_every_donor(executor):
    """Every donor's entries arrive and its handle stops answering, all at once, where the build absorbs."""
    first = DenseIndex.init_ram(dimensions=16, quantization=Scalar.BF16)
    first.reserve(96)
    first.on(executor).insert(np.arange(0, 32), random_vectors(32, 16))
    donors = []
    for seed in (1, 2):
        donor = DenseIndex.init_ram(dimensions=16, quantization=Scalar.BF16)
        donor.reserve(32)
        donor.on(executor).insert(np.arange(32 * seed, 32 * seed + 32), random_vectors(32, 16, seed=seed))
        donors.append(donor)

    try:
        first.on(executor).absorb(*donors)
    except UnsupportedError:
        return
    assert len(first) == 96
    assert [40 in first, 80 in first] == [True, True]
    for donor in donors:
        with pytest.raises(UninitializedError):
            len(donor)


def test_absorb_stopped_between_donors_leaves_the_rest_live(executor):
    """A donor the absorb never reached is handed back to its wrapper, still usable."""
    first, _ = build_index(8, 8, executor)
    second, _ = build_index(8, 8, executor, seed=1)
    third, _ = build_index(8, 8, executor, seed=2)
    with pytest.raises((CancelledError, UnsupportedError)) as raised:
        first.on(executor).absorb(second, third, progress=lambda completed, total: completed < 1)
    if raised.type is UnsupportedError:
        return
    assert len(first) == 16
    assert len(third) == 8
    with pytest.raises(UninitializedError):
        len(second)


def test_absorb_consumes_a_donor_still_walked(executor):
    """A cursor keeps no hold on its index, so an absorb consumes the donor and the cursor's next
    pull, naming a handle that is gone, is refused."""
    first, _ = build_index(16, 8, executor)
    second, _ = build_index(16, 8, executor, seed=1)
    cursor = second.cursor()
    try:
        first.on(executor).absorb(second)
    except UnsupportedError:
        return
    assert len(first) == 32
    with pytest.raises(UninitializedError):
        cursor.next(4, Arena())


def test_absorb_refuses_itself(executor):
    """An index cannot absorb the handle it already is."""
    index, _ = build_index(8, 8, executor)
    with pytest.raises(ConfigMismatchError):
        index.on(executor).absorb(index)


def test_clear_keeps_the_geometry(executor):
    """Clearing drops the entries and nothing else."""
    index, _ = build_index(32, 16, executor)
    capacity = index.capacity
    index.clear()
    assert len(index) == 0
    assert index.capacity == capacity
    assert index.dimensions == 16


# endregion Composition

# region Serialization


def test_save_load_roundtrip(executor, tmp_path):
    """A checkpoint restores what it stored, its geometry read from its own headers."""
    index, _ = build_index(128, 32, executor)
    path = str(tmp_path / "index.safegraphs")
    index.on(executor).save(path)

    metadata = DenseIndex.metadata_read(path, executor=executor)
    assert isinstance(metadata, DenseMetadata)
    assert metadata.dimensions == 32
    assert metadata.metric is DenseMetric.ANGULAR
    assert DenseMetadata.n_fields == 4, "a checkpoint fixes the geometry, never where it is hosted"

    reloaded = DenseIndex.load_ram(path, executor=executor)
    assert len(reloaded) == len(index)
    assert reloaded.dimensions == 32
    assert 7 in reloaded


def test_save_load_through_buffers(executor):
    """`save_buffer` returns the monolith a `load_ram_buffer` reads back."""
    index, _ = build_index(64, 16, executor)
    buffer = index.on(executor).save_buffer(Arena(executor))
    assert isinstance(buffer, Monolith) and memoryview(buffer).nbytes > 0
    assert DenseIndex.metadata_read_buffer(buffer).dimensions == 16

    reloaded = DenseIndex.load_ram_buffer(buffer)
    assert len(reloaded) == len(index)


def test_cancelled_load_creates_no_index(executor, tmp_path):
    """A handler that stops a load leaves nothing behind but the exception."""
    index, _ = build_index(8, 8, executor)
    path = str(tmp_path / "cancelled.safegraphs")
    index.on(executor).save(path)
    with pytest.raises(CancelledError):
        DenseIndex.load_ram(path, progress=lambda completed, total: False)


def test_view_serves_in_place_and_refuses_mutation(executor, tmp_path):
    """A viewed checkpoint answers searches and refuses every mutating verb."""
    index, vectors = build_index(32, 16, executor)
    path = str(tmp_path / "viewed.safegraphs")
    index.on(executor).save(path)

    try:
        viewed = DenseIndex.view_ram(path)
    except UnsupportedError:
        return
    assert len(viewed) == 32
    assert len(viewed.on(executor).search(vectors[:2], 3, Arena(executor))) == 2
    with pytest.raises(ReadOnlyError) as raised:
        viewed.on(executor).insert([99], vectors[:1])
    assert raised.value.status is Status.READ_ONLY


def test_encoded_path_is_read_as_a_path(executor, tmp_path):
    """`os.fsencode` returns `bytes`, which still names a file."""
    index, _ = build_index(8, 8, executor)
    path = str(tmp_path / "encoded.safegraphs")
    index.on(executor).save(path)
    assert DenseIndex.metadata_read(os.fsencode(path)).dimensions == 8


def test_metadata_roundtrips_through_a_factory(executor, tmp_path):
    """The metadata fields are the geometry every factory takes, so they rebuild the handle."""
    index = DenseIndex.init_ram(dimensions=48, metric=DenseMetric.EUCLIDEAN, quantization=Scalar.I8, connectivity=32)
    index.reserve(8)
    index.on(executor).insert(np.arange(8), random_vectors(8, 48))
    path = str(tmp_path / "typed.safegraphs")
    index.on(executor).save(path)

    metadata = DenseIndex.metadata_read(path)
    rebuilt = DenseIndex.init_ram(
        metadata.dimensions, metadata.metric, metadata.quantization, connectivity=metadata.connectivity
    )
    assert rebuilt.dimensions == 48
    assert rebuilt.connectivity == 32


# endregion Serialization

# region Refinement


def test_refine_reports_a_sweep_and_leaves_the_index_standing(executor):
    """A sweep re-selects edges without publishing or retiring an entry, so the rows survive it."""
    index, vectors = build_index(64, 16, executor)
    changed = index.on(executor).refine(expansion_factor=64)
    assert isinstance(changed, int)
    assert changed <= len(index), "a sweep cannot change more nodes than the graph holds"
    assert len(index) == 64
    assert len(index.on(executor).search(vectors[:4], 5, Arena(executor))) == 4


def test_refine_reports_telemetry(executor):
    """The sweep reports into the `DenseTraversalStats` a caller passes, as every other walk does."""
    index, _ = build_index(32, 16, executor)
    stats = DenseTraversalStats()
    index.on(executor).refine(stats=stats)
    assert stats.visited_members > 0


def test_refine_takes_no_positionals(executor):
    """Every setting a sweep has is a keyword, so a bare number cannot be mistaken for one."""
    index, _ = build_index(8, 8, executor)
    with pytest.raises(TypeError):
        index.refine(64)


def test_refine_rejects_unknown_keywords(executor):
    """A misspelled setting is refused rather than silently derived."""
    index, _ = build_index(8, 8, executor)
    with pytest.raises(TypeError):
        index.on(executor).refine(expansion_add=64)


def test_refine_refuses_a_consumed_handle(executor):
    """A donor absorbed by `absorb` has no graph left to sweep."""
    first, _ = build_index(8, 8, executor)
    second, _ = build_index(8, 8, executor, seed=1)
    try:
        first.on(executor).absorb(second)
    except UnsupportedError:
        return
    with pytest.raises(UninitializedError):
        second.on(executor).refine()


def test_health_reaches_every_entry_after_a_refine(executor):
    """A refine re-links whatever it strands, so the audit after it reaches the whole index."""
    index, _ = build_index(64, 16, executor)
    index.on(executor).refine()
    health = index.on(executor).health()
    assert isinstance(health, DenseReachability)
    assert health.audited == len(index)
    assert health.reachable == health.audited
    assert health.sweeps >= 1


# endregion Refinement

# region Late Interaction

_LATE_INTERACTION_DOCUMENTS: int = 6
"""The document keys a late-interaction index holds."""

_LATE_INTERACTION_TOKENS: int = 3
"""The token rows every document key carries."""


def late_interaction_index(executor, dimensions: int = 16) -> tuple[DenseIndex, npt.NDArray[np.float32]]:
    """An inner-product index whose entries are token rows, several sharing one document key."""
    rows = _LATE_INTERACTION_DOCUMENTS * _LATE_INTERACTION_TOKENS
    index = DenseIndex.init_ram(
        dimensions=dimensions,
        metric=DenseMetric.INNER,
        quantization=Scalar.BF16,
        values_per_key=_LATE_INTERACTION_TOKENS,
    )
    index.reserve(rows)
    documents = np.repeat(np.arange(_LATE_INTERACTION_DOCUMENTS, dtype=np.uint64), _LATE_INTERACTION_TOKENS)
    tokens = random_vectors(rows, dimensions, seed=5)
    index.on(executor).insert(integers_to_keys(documents), tokens)
    return index, tokens


def scored_documents(index, tokens, queries: int, wanted: int, executor, **kwargs) -> DenseProducts:
    """Runs `maxsim` over *queries* whole queries."""
    return index.on(executor).maxsim(
        tokens[: queries * _LATE_INTERACTION_TOKENS], _LATE_INTERACTION_TOKENS, wanted, Arena(executor), **kwargs
    )


def test_maxsim_sizes_its_outputs_over_documents(executor):
    """A query is a run of token rows, so the result has one run per query rather than per token."""
    index, tokens = late_interaction_index(executor)
    queries = 2
    stats = DenseTraversalStats()
    documents = scored_documents(index, tokens, queries, 4, executor, stats=stats)
    offsets = np.asarray(documents.offsets)

    assert isinstance(documents, DenseProducts) and len(documents) == queries
    assert offsets.shape == (queries + 1,)
    assert offsets[0] == 0 and offsets[-1] == len(np.asarray(documents.scores)) <= documents.required
    assert np.asarray(documents.keys).shape == (offsets[-1], 16)
    assert all(0 < end - begin <= 4 for begin, end in pairwise(offsets.tolist())), "each query keeps at most wanted"
    assert stats.computed_distances > 0


def test_maxsim_scores_each_document_once(executor):
    """The reduction groups by key, so a document holding many token rows still lands one entry."""
    index, tokens = late_interaction_index(executor)
    documents = scored_documents(index, tokens, 1, 4, executor)
    begin, end = np.asarray(documents.offsets).tolist()
    found = keys_to_integers(documents.keys)[begin:end]
    assert end > begin
    assert len(set(found.tolist())) == end - begin


def test_maxsim_orders_by_summed_maxima(executor):
    """Summed maxima are a similarity, so a query's run descends where a search's distances ascend."""
    index, tokens = late_interaction_index(executor)
    documents = scored_documents(index, tokens, 1, 4, executor)
    begin, end = np.asarray(documents.offsets).tolist()
    scored = np.asarray(documents.scores)[begin:end]
    assert list(scored) == sorted(scored, reverse=True)


def test_maxsim_refuses_ragged_queries(executor):
    """The query rows must divide into whole queries, or the grouping is a guess."""
    index, tokens = late_interaction_index(executor)
    with pytest.raises(ShapeMismatchError):
        index.on(executor).maxsim(tokens[: _LATE_INTERACTION_TOKENS + 1], _LATE_INTERACTION_TOKENS, 4, Arena())


def test_maxsim_scores_only_admitted_documents(executor):
    """A predicate is asked before any of a document's rows score, so a rejected one never lands."""
    index, tokens = late_interaction_index(executor)
    documents = scored_documents(index, tokens, 2, 4, executor, predicate=lambda key: int(key) % 2 == 0)
    found = keys_to_integers(documents.keys)
    for begin, end in pairwise(np.asarray(documents.offsets).tolist()):
        assert end > begin
        assert np.all(found[begin:end] % 2 == 0)


def test_maxsim_keeps_every_scored_document_when_wanted_is_zero(executor):
    """A zero bound keeps every document a query scored, so no query lands fewer than under a bound,
    and a bound cuts what lands but not what `required` counts."""
    index, tokens = late_interaction_index(executor)
    every = np.diff(np.asarray(scored_documents(index, tokens, 2, 0, executor).offsets))
    bounded = np.diff(np.asarray(scored_documents(index, tokens, 2, 2, executor).offsets))
    scanned = np.diff(np.asarray(scored_documents(index, tokens, 2, 0, executor, traversal=Traversal.SCAN).offsets))
    cut = scored_documents(index, tokens, 2, 2, executor, traversal=Traversal.SCAN)
    assert np.all(every >= bounded)
    assert scanned.tolist() == [_LATE_INTERACTION_DOCUMENTS] * 2, "a scan nominates every document"
    assert cut.required == 2 * _LATE_INTERACTION_DOCUMENTS, "every scanned document counts, kept or not"


def test_maxsim_refuses_a_distance_kind(executor):
    """Late interaction sums maxima over logits, which only the inner-product kinds score."""
    index, vectors = build_index(16, 8, executor, metric=DenseMetric.ANGULAR)
    with pytest.raises(ConfigMismatchError):
        index.on(executor).maxsim(vectors[:2], 2, 4, Arena())


def test_maxsim_reports_telemetry(executor):
    """The walk reports into the `DenseTraversalStats` a caller passes, as every other walk does."""
    index, tokens = late_interaction_index(executor)
    stats = DenseTraversalStats()
    scored_documents(index, tokens, 1, 4, executor, stats=stats)
    assert stats.computed_distances > 0


# endregion Late Interaction

# region Linear Algebra


def test_multiply_reports_every_scored_pair(executor):
    """A walk scores many more pairs than a search reports, and this verb hands them all back."""
    index, vectors = build_index(64, 32, executor, metric=DenseMetric.INNER)
    rows = vectors[:4]
    products = index.on(executor).multiply(rows, 0, Arena(executor))
    offsets = np.asarray(products.offsets)
    named = keys_to_integers(products.keys)
    scores = np.asarray(products.scores)

    assert isinstance(products, DenseProducts) and len(products) == len(rows)
    assert offsets.shape == (len(rows) + 1,)
    assert offsets[0] == 0 and offsets[-1] == len(scores) == products.required
    assert np.all(np.diff(offsets) > 0)
    row_keys, row_scores = products.row(1)
    assert np.array_equal(np.asarray(row_scores), scores[offsets[1] : offsets[2]])
    assert np.array_equal(keys_to_integers(row_keys), named[offsets[1] : offsets[2]])
    assert not hasattr(products.scores, "__cuda_array_interface__")

    # Every product is an exact dot against the row it names, which is the point of the verb.
    for row_index in range(len(rows)):
        for pair in range(int(offsets[row_index]), int(offsets[row_index + 1])):
            expected = float(rows[row_index] @ vectors[named[pair]])
            assert scores[pair] == pytest.approx(expected, rel=BF16_EPSILON, abs=BF16_EPSILON)


def test_multiply_reports_what_wanted_cut(executor):
    """A bound keeps the first pairs per row, and `required` still counts every pair scored."""
    index, vectors = build_index(64, 16, executor, metric=DenseMetric.INNER)
    products = index.on(executor).multiply(vectors[:2], 4, Arena(executor))
    assert np.diff(np.asarray(products.offsets)).tolist() == [4, 4]
    assert products.required > 8, "both rows scored more pairs than were kept"


def test_multiply_refuses_a_predicate(executor):
    """`multiply` has no filtered sibling, so the keyword is refused rather than ignored."""
    index, vectors = build_index(16, 8, executor, metric=DenseMetric.INNER)
    with pytest.raises(TypeError):
        index.on(executor).multiply(vectors[:1], 4, Arena(), predicate=lambda key: True)


@pytest.mark.parametrize("head_count", [1, 2])
def test_attend_over_every_key_is_full_attention(executor, head_count: int):
    """A graph this small walks every key, so each head's output and carry are exact attention's."""
    dimensions, context, tokens = 16, 32, 3
    index, keys_matrix = build_index(context, dimensions, executor, metric=DenseMetric.INNER)
    values = random_vectors(context, dimensions, seed=7)
    queries = random_vectors(tokens * head_count, dimensions, seed=3)
    carry = np.zeros((len(queries), 2), dtype=np.float32)
    outputs = index.on(executor).attend(
        queries,
        bf16_words(values),
        int,
        Arena(executor),
        head_count=head_count,
        expansion_factor=2 * context,
        values_dtype=Scalar.BF16,
        outputs_dtype=Scalar.F32,
        carry=carry,
    )

    logits = queries @ keys_matrix.T / np.sqrt(dimensions)
    weights = np.exp(logits - logits.max(axis=1, keepdims=True))
    expected = weights @ bf16_rounded(values) / weights.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(np.asarray(outputs), expected, rtol=BF16_EPSILON, atol=BF16_EPSILON)
    np.testing.assert_allclose(carry[:, 0], logits.max(axis=1), rtol=BF16_EPSILON, atol=BF16_EPSILON)
    np.testing.assert_allclose(carry[:, 1], weights.sum(axis=1), rtol=BF16_EPSILON)


def test_attend_lands_in_the_arena(executor):
    """The attended rows view the arena they were handed, so a decode loop reusing one allocates once."""
    index, keys_matrix = build_index(16, 8, executor, metric=DenseMetric.INNER)
    arena = Arena(executor)
    landed = index.on(executor).attend(
        keys_matrix[:2], bf16_words(keys_matrix), int, arena, values_dtype=Scalar.BF16, outputs_dtype=Scalar.F32
    )
    assert isinstance(landed, DenseVectors)
    assert np.any(np.asarray(landed) != 0)


def test_attend_refuses_a_missing_value_row(executor):
    """A missing row is a wrong softmax rather than a smaller one, and a lookup's own failure outranks it."""
    index, keys_matrix = build_index(8, 8, executor, metric=DenseMetric.INNER)
    values = bf16_words(keys_matrix)
    with pytest.raises(ShapeMismatchError):
        index.on(executor).attend(keys_matrix[:1], values, lambda key: None, Arena(), values_dtype=Scalar.BF16)
    with pytest.raises(ShapeMismatchError):
        index.on(executor).attend(keys_matrix[:1], values[:4], int, Arena(), values_dtype=Scalar.BF16)
    with pytest.raises(ShapeMismatchError):
        index.on(executor).attend(keys_matrix[:1], keys_matrix, int, Arena())
    with pytest.raises(ZeroDivisionError):
        index.on(executor).attend(keys_matrix[:1], values, lambda key: 1 / 0, Arena(), values_dtype=Scalar.BF16)


def test_attend_refuses_a_half_width_carry(executor):
    """A maximum alone is a number nothing can rescale, so the pair is sized exactly."""
    index, keys_matrix = build_index(8, 8, executor, metric=DenseMetric.INNER)
    values = bf16_words(keys_matrix)
    with pytest.raises(ValueError):
        index.on(executor).attend(
            keys_matrix[:1], values, int, Arena(), values_dtype=Scalar.BF16, carry=np.zeros(1, np.float32)
        )


def test_attend_refuses_a_distance_kind(executor):
    """A softmax weighs logits, which only the inner-product kinds score."""
    index, keys_matrix = build_index(8, 8, executor, metric=DenseMetric.ANGULAR)
    with pytest.raises(ConfigMismatchError):
        index.on(executor).attend(keys_matrix[:1], bf16_words(keys_matrix), int, Arena(), values_dtype=Scalar.BF16)


# endregion Linear Algebra
