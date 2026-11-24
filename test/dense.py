"""Vector search over equidimensional dense embeddings - the Python twin of `test/dense.cpp`.

Baselines:
    `usearch.eval.exact_matches` for every recall claim, which scores with NumKong's kernels in
    the index's own metric vocabulary, so a number here compares like with like.

Run:
    python -m pytest test/dense.py -v

File: test/dense.py
Author: Ash Vardanian
Date: August 16, 2026
"""

import asyncio
import contextlib
import gc
import os
import weakref
from collections.abc import Callable
from itertools import pairwise
from threading import Event, Thread

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
    init_ram_or_skip,
    measured_recall,
    recall_floor,
    stream_key,
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
    DenseRefinePlan,
    DenseReshardPlan,
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
    OutOfMemoryError,
    Phase,
    ReadOnlyError,
    RefineReport,
    Residency,
    Sample,
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


@pytest.mark.parametrize("ndim", SETTINGS.dims)
@pytest.mark.parametrize("metric", SPATIAL_METRICS)
def test_construction_reports_its_geometry(ndim: int, metric: DenseMetric):
    """Every setting the factory took reads back off the handle, the modes as their enums."""
    index = DenseIndex.init_ram(dimensions=ndim, metric=metric, quantization=Scalar.BF16, connectivity=16)
    assert index.dimensions == ndim
    assert index.connectivity == 16
    assert len(index) == 0
    assert isinstance(index.cpu_class, CPUClass)
    assert index.gpu_class is None
    assert repr(index) == f"DenseIndex(dimensions={ndim}, size=0)"
    description = index.describe()
    assert isinstance(description, DenseIndexDescription)
    assert description.metadata == (ndim, metric, Scalar.BF16, 16)
    assert description.metadata.metric is metric
    assert description.size == 0
    assert description.shards_count == index.shards_count >= 1
    assert description.cpu_class is index.cpu_class
    assert description.gpu_class is None


def test_construction_takes_a_class_ceiling():
    """A serial ceiling runs on every CPU, and the index it opens reports the class picked and its entries."""
    index = DenseIndex.init_ram(dimensions=8, cpu_class_ceiling=CPUClass.SERIAL)
    index.reserve(2)
    index.insert([1, 2], np.ones((2, 8), dtype=np.float32))
    assert len(index) == 2
    assert index.cpu_class is CPUClass.SERIAL


def test_fresh_index_inserts_without_a_reservation(seed: int):
    """Init freezes the geometry, so a fresh index takes rows and grows for them with no `reserve`."""
    index = DenseIndex.init_ram(dimensions=8)
    index.insert(np.arange(100), np.random.default_rng(seed).random((100, 8), dtype=np.float32))
    assert len(index) == 100
    assert index.count(7) == 1


@pytest.mark.parametrize(
    ("options", "failure"),
    [
        ({"metric": "angular"}, TypeError),  # Modes are enums, never strings.
        ({"quantization": "bf16"}, TypeError),
        ({"quantization": True}, TypeError),  # A bool would read as an enumerator.
        ({"metric": 99}, UnknownKindError),  # An integer no enumerator holds.
        ({"quantization": Scalar.F32}, UnknownKindError),  # Rows cross the boundary as `f32` but are never stored so.
        ({"residency": Scalar.F32}, TypeError),  # The factory chooses the residency; no keyword does.
        ({"devices": [0]}, TypeError),  # A variant belonging to another residency.
        ({"threads": 2}, TypeError),  # An index owns entries; an `Executor` owns workers.
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


def test_ram_executor_takes_memory_domain_ordinals():
    """The NUMA nodes its workers stay local to arrive as ordinals, and nothing else is accepted."""
    Executor.for_ram(memory_domains=[0]).drain()
    with pytest.raises(TypeError):
        Executor.for_ram(memory_domains=["x"])
    with pytest.raises(ValueError):
        Executor.for_ram(memory_domains=[2**40])


def test_drain_takes_a_mood():
    """The idle mood is a `Mood`, and neither a name nor another integer is guessed at."""
    pool = Executor.for_ram(threads=1)
    pool.drain(Mood.SPINNING)
    pool.drain()
    with pytest.raises(TypeError):
        pool.drain("parked")
    with pytest.raises(ValueError):
        pool.drain(7)


def test_class_enums_hold_the_library_spellings():
    """Every class member is named by the library's spelling of its value, in value order, and the values keep
    their registry numbers forever, so a new class renumbers none of its neighbours."""
    spellings = {
        CPUClass: "serial monoversioned intel-haswell intel-sierra intel-skylake intel-icelake amd-genoa intel-sapphire"
        " intel-granite intel-diamond aws-g2 apple-m2 apple-m4 aws-g5 nvidia-vera alibaba-c930 ibm-power10 wasm32-v128"
        " wasm64-extended",
        GPUClass: "nvidia-ampere nvidia-ada nvidia-hopper nvidia-blackwell nvidia-blackwellrtx nvidia-rubin amd-cdna3"
        " amd-cdna4 amd-cdna5 apple-metal7 apple-metal9 apple-metal10",
        NVMeClass: "posix linux-aio linux-uring linux-uring-passthrough linux-spdk windows-ioring",
    }
    for enum, spelled in spellings.items():
        assert [member.name for member in enum] == [name.upper().replace("-", "_") for name in spelled.split()]
    assert (CPUClass.SERIAL, CPUClass.MONOVERSIONED, CPUClass.IBM_POWER10) == (1, 2, 18)


def test_nvme_executor_takes_an_nvme_class():
    """The engine an NVMe executor submits through is an `NVMeClass`, and nothing else is guessed at."""
    assert isinstance(Executor.for_nvme(threads=1, nvme_class=NVMeClass.POSIX), Executor)
    with pytest.raises(TypeError):
        Executor.for_nvme(threads=1, nvme_class="posix")


def test_executor_closes_with_its_block(seed: int):
    """Leaving a `with` block waits for the executor's calls and closes it, so a later receiver is refused."""
    index, _ = build_index(8, 8, Executor.for_ram(threads=1), seed=seed)
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
def test_verbs_run_on_the_callers_thread(executor, verb: str, call, seed: int):
    """The index itself is a receiver too: its verbs run on the caller's thread rather than refusing."""
    index, vectors = build_index(8, 8, executor, seed=seed)
    try:
        call(index, vectors)
    except UnsupportedError:
        assert verb == "search_stream", "only search streams are declared before they are served"


@pytest.mark.parametrize("wrong", [2, "ram", None])
def test_receivers_take_only_an_executor(executor, wrong, seed: int):
    """A pool is a handle, not a thread count or a name, so anything else is refused where it is named."""
    index, _ = build_index(8, 8, executor, seed=seed)
    with pytest.raises(TypeError):
        index.on(wrong)
    with pytest.raises(TypeError):
        index.deferred(wrong)


def test_executor_is_viewed_for_the_call(executor, seed: int):
    """A verb uses the workers for its call and never retains them, so a temporary one serves."""
    index, vectors = build_index(16, 8, Executor.for_ram(), seed=seed)
    assert len(index) == 16
    assert len(index.on(executor).search(vectors[:1], 1, Arena(executor))) == 1


def test_one_executor_serves_several_indexes(executor, seed: int):
    """An index owns entries and an executor owns workers, so one pool drives as many as asked."""
    first, vectors = build_index(16, 8, executor, seed=seed)
    second, _ = build_index(16, 8, executor, seed=stream_key(seed, "second"))
    assert len(first) == len(second) == 16
    assert len(first.on(executor).search(vectors[:2], 1, Arena(executor))) == 2


def test_concurrent_searches_take_one_executor_apiece(executor, spawn_executor, seed: int):
    """A handle admits any number of callers, and each thread here blocks on an executor of its own."""
    index, vectors = build_index(128, 16, executor, seed=seed)
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
        assert isinstance(Arena(pool), Arena)
        assert isinstance(Arena(), Arena)
        assert Arena(pool).describe() == (Residency.RAM, None) == Arena().describe()
    with pytest.raises(UninitializedError):
        Arena(pool)
    with pytest.raises(TypeError):
        Arena(2)


@pytest.mark.parametrize(("arena", "failure"), [(None, InvalidArgumentError), (0, TypeError)])
def test_output_verbs_require_an_arena(executor, arena, failure: type, seed: int):
    """No output verb allocates on its own: the library refuses one handed no arena, and anything else is no arena."""
    index, vectors = build_index(8, 8, executor, seed=seed)
    with pytest.raises(TypeError):
        index.search(vectors[:1], 1)
    with pytest.raises(failure):
        index.search(vectors[:1], 1, arena)
    with pytest.raises(failure):
        index.cursor().next(1, arena)


# endregion Executors

# region Receivers


def test_receivers_answer_alike(executor, seed: int):
    """The receiver chooses where the call runs and the verb stays, so all three answer the same matches."""
    index, vectors = build_index(64, 16, executor, seed=seed)
    on_caller = keys_to_integers(index.search(vectors[:4], 5, Arena()).keys)
    on_executor = keys_to_integers(index.on(executor).search(vectors[:4], 5, Arena(executor)).keys)
    call = index.deferred(executor).search(vectors[:4], 5, Arena(executor))
    assert isinstance(call, Deferred)
    deferred = keys_to_integers(call.wait().keys)
    assert np.array_equal(on_caller, on_executor)
    assert np.array_equal(on_caller, deferred)


def test_deferred_call_is_deferred_until_waited(executor, seed: int):
    """A submitted call reads `DEFERRED` until it is final, then answers what a blocking one would."""
    index = DenseIndex.init_ram(16, DenseMetric.ANGULAR, Scalar.BF16)
    index.reserve(8)
    call = index.deferred(executor).insert(np.arange(8), random_vectors(8, 16, seed=seed))
    assert call.status in (Status.DEFERRED, Status.SUCCESS), "pool thread 0 may finish it before this reads"
    assert call.wait() is None
    assert call.status is Status.SUCCESS
    assert repr(call) == "Deferred(status=success)"
    assert call.wait() is None, "a final call answers as many times as it is asked"
    assert len(index) == 8


def test_deferred_call_owns_its_inputs(executor, seed: int):
    """The rows a deferred call views stay exported inside its `Deferred`, so the caller may drop its own names."""
    index = DenseIndex.init_ram(16, DenseMetric.ANGULAR, Scalar.BF16)
    index.reserve(4)
    rows = random_vectors(4, 16, seed=seed)
    call = index.deferred(executor).insert(list(range(4)), rows.copy())
    call.wait()
    assert np.allclose(np.asarray(index.get(2, Arena())), rows[2:3], rtol=BF16_EPSILON, atol=BF16_EPSILON)


def test_deferred_call_raises_at_wait(executor, seed: int):
    """Arguments the binding refuses raise at the call; the engine's own refusals raise once waited for."""
    index, vectors = build_index(16, 8, executor, seed=seed)
    with pytest.raises(ShapeMismatchError):
        index.deferred(executor).insert([1, 2, 3], vectors[:2])
    call = index.deferred(executor).search(vectors[:1, :4], 1, Arena(executor))
    assert call.status in (Status.DEFERRED, Status.SHAPE_MISMATCH)
    for _ in range(2):
        with pytest.raises(ShapeMismatchError) as raised:
            call.wait()
        assert raised.value.status is Status.SHAPE_MISMATCH
    assert call.status is Status.SHAPE_MISMATCH


def test_deferred_call_awaits(executor, seed: int):
    """`await` polls the status between short sleeps, so a coroutine suspends and no thread blocks."""
    index, vectors = build_index(32, 8, executor, seed=seed)

    async def search() -> Matches:
        call = index.deferred(executor).search(vectors[:2], 3, Arena(executor))
        return await call

    matches = asyncio.run(search())
    assert keys_to_integers(matches.keys)[:, 0].tolist() == [0, 1]


def test_polling_alone_finishes_a_deferred_call(executor, seed: int):
    """Pool thread 0 runs a deferred call in the caller's place, and a caller reading it final may resubmit at once."""
    index = DenseIndex.init_ram(8, DenseMetric.ANGULAR, Scalar.BF16)
    index.reserve(4)
    call = index.deferred(executor).insert([7], random_vectors(1, 8, seed=seed))
    while call.status is Status.DEFERRED:
        pass
    assert call.status is Status.SUCCESS
    assert index.on(executor).count(7) == 1


def test_held_executor_answers_contended(seed: int):
    """A call finding the executor held finishes `CONTENDED` at once, as nothing queues, and the holder still runs."""
    index, vectors = build_index(16, 8, Executor.for_ram(threads=1), seed=seed)
    pool = Executor.for_ram(threads=2)
    released = Event()
    first = index.deferred(pool).search(vectors[:1], 1, Arena(pool), predicate=lambda key: released.wait(10))
    second = index.deferred(pool).count(4)
    assert second.status is Status.CONTENDED
    with pytest.raises(ContendedError):
        second.wait()
    released.set()
    assert keys_to_integers(first.wait().keys)[0, 0] == 0


def test_dropping_an_unfinished_deferred_waits(executor, seed: int):
    """Dropping a `Deferred` waits for its call, since the call's structure and buffers live inside it."""
    index = DenseIndex.init_ram(8, DenseMetric.ANGULAR, Scalar.BF16)
    index.reserve(4)
    call = index.deferred(executor).insert([1, 2], random_vectors(2, 8, seed=seed))
    del call
    assert len(index) == 2


def test_dropped_deferred_in_a_cycle_is_collected_once_final(executor, seed: int):
    """A predicate holding its own `Deferred` forms a cycle the collector breaks once the call is final."""
    index, vectors = build_index(4, 8, executor, seed=seed)
    witness = type("Witness", (), {})()
    alive = weakref.ref(witness)
    cycle: list = [witness]
    call = index.deferred(executor).search(vectors[:1], 1, Arena(executor), predicate=lambda key, cycle=cycle: True)
    cycle.append(call)
    del witness, cycle, call
    executor.drain()
    gc.collect()
    assert alive() is None


def test_deferred_call_views_its_arena(executor, seed: int):
    """A deferred call uses its arena as a blocking one does, refusing a second call until it is final."""
    index, vectors = build_index(4, 8, executor, seed=seed)
    arena = Arena(executor)
    entered, released = Event(), Event()
    call = index.deferred(executor).search(
        vectors[:1], 1, arena, predicate=lambda key: (entered.set(), released.wait(10))[1]
    )
    assert entered.wait(10), "the call holds its arena once its body runs on pool thread 0"
    with pytest.raises(ContendedError):
        index.get(2, arena)
    released.set()
    assert keys_to_integers(call.wait().keys)[0, 0] == 0
    assert np.asarray(index.get(2, arena)).shape == (1, 8)


def test_deferred_receiver_lacks_what_cannot_defer(executor, seed: int):
    """Cursors and streams block on every pull and factories address no index, so the receivers that cannot run them
    lack them."""
    index, _ = build_index(4, 8, executor, seed=seed)
    assert not hasattr(index.deferred(executor), "cursor")
    assert not hasattr(index.deferred(executor), "search_stream")
    assert not hasattr(index.on(executor), "init_ram")
    assert not hasattr(index.on(executor), "cursor")
    assert type(index.on(executor)) is On
    assert type(index.deferred(executor)) is DeferredOn
    assert repr(index.on(executor)) == f"On({index!r}, {executor!r})"


# endregion Receivers

# region Ingest


@pytest.mark.parametrize("ndim", SETTINGS.dims)
@pytest.mark.parametrize("count", BATCH_SIZES)
def test_insert_publishes_every_row(executor, ndim: int, count: int, seed: int):
    """A batch is all-or-nothing, so every entry of an accepted batch is visible afterwards."""
    index, _ = build_index(count, ndim, executor, seed=seed)
    assert len(index) == count
    assert index.capacity >= count


@pytest.mark.parametrize("quantization", QUANTIZATIONS)
def test_insert_accepts_every_quantization(executor, quantization: Scalar, seed: int):
    """Every stored scalar type takes the same `float32` rows and quantizes them on ingest."""
    index, _ = build_index(64, 96, executor, quantization=quantization, seed=seed)
    assert len(index) == 64


@pytest.mark.parametrize("dtype", INPUT_DTYPES)
def test_insert_accepts_any_input_dtype(executor, dtype, seed: int):
    """The query type is free to differ from the stored quantization."""
    index = DenseIndex.init_ram(dimensions=32, quantization=Scalar.BF16)
    index.reserve(16)
    vectors = (np.random.default_rng(seed).random((16, 32)) * 10).astype(dtype)
    index.on(executor).insert(np.arange(16), vectors)
    assert len(index) == 16


def test_insert_accepts_packed_rows_by_logical_width(executor, seed: int):
    """A `u1x8` row counts the bits it packs, so an eight-byte row is a sixty-four-bit vector."""
    index = init_ram_or_skip(dimensions=64, metric=DenseMetric.HAMMING, quantization=Scalar.U1X8)
    index.reserve(16)
    rows = random_vectors(8, 64, metric=DenseMetric.HAMMING, seed=seed)
    assert rows.shape == (8, 8), "sixty-four bits pack into eight bytes"
    index.on(executor).insert(np.arange(8), rows, dtype=Scalar.U1X8)
    assert len(index) == 8

    narrow = random_vectors(8, 32, metric=DenseMetric.HAMMING, seed=seed)
    with pytest.raises(ShapeMismatchError):
        index.on(executor).insert(np.arange(8, 16), narrow, dtype=Scalar.U1X8)


def test_insert_refuses_a_dtype_narrower_than_the_buffer(executor, seed: int):
    """A named kind reinterprets the buffer's bytes, so one narrower than its items would misread every row."""
    index = DenseIndex.init_ram(dimensions=8, quantization=Scalar.BF16)
    index.reserve(4)
    with pytest.raises(UnknownKindError):
        index.insert(np.arange(4), random_vectors(4, 8, seed=seed), dtype=Scalar.F16)
    assert len(index) == 0


def test_insert_accepts_numkong_tensors_without_numpy(executor, seed: int):
    """A NumKong tensor is a `memoryview`, so it reaches the index through PEP 3118 alone.

    This is the claim the README leads with, and the only test that exercises it: every other
    path in this file arrives as a NumPy array.
    """
    index = DenseIndex.init_ram(dimensions=16, quantization=Scalar.BF16)
    index.reserve(8)
    rows = nk.Tensor(random_vectors(8, 16, seed=seed))
    assert memoryview(rows).shape == (8, 16), "NumKong tensors export PEP 3118 directly"
    index.on(executor).insert(np.arange(8), rows)
    assert len(index) == 8


@pytest.mark.parametrize("keys", [[Key(11), 12], np.arange(2), integers_to_keys(np.arange(2))])
def test_insert_accepts_every_key_spelling(executor, keys, seed: int):
    """Keys arrive as objects, as integers, or as the packed wire rows."""
    index = DenseIndex.init_ram(dimensions=4, quantization=Scalar.BF16)
    index.reserve(2)
    index.on(executor).insert(keys, random_vectors(2, 4, seed=seed))
    assert len(index) == 2


def test_insert_gathers_integer_vectors_as_keys(executor, seed: int):
    """An integer vector of any width or stride reads as the identities its integers spell, negatives refused."""
    index = DenseIndex.init_ram(dimensions=4, quantization=Scalar.BF16)
    index.reserve(8)
    for keys in (
        np.array([7, 200], np.uint8),
        np.array([7, 300], np.int16),
        np.array([2**64 - 1, 5], np.uint64),
        np.arange(10, 14, dtype=np.int64)[::2],
    ):
        index.on(executor).insert(keys, random_vectors(2, 4, seed=seed))
    spelled = (7, 200, 300, 2**64 - 1, 5, 10, 12, 11)
    assert [index.count(key) for key in spelled] == [2, 1, 1, 1, 1, 1, 1, 0]
    with pytest.raises(OverflowError):
        index.on(executor).insert(np.array([3, -1], np.int32), random_vectors(2, 4, seed=seed))
    assert len(index) == 8


def test_insert_rejects_mismatched_key_count(executor, seed: int):
    """A key per row, or the batch is refused before anything is published."""
    index = DenseIndex.init_ram(dimensions=4, quantization=Scalar.BF16)
    index.reserve(4)
    with pytest.raises(ShapeMismatchError):
        index.on(executor).insert([1, 2, 3], random_vectors(2, 4, seed=seed))
    assert len(index) == 0


def test_insert_reports_telemetry(executor, seed: int):
    """A batch reports what it walked into the `DenseTraversalStats` a caller passes."""
    index = DenseIndex.init_ram(dimensions=16, quantization=Scalar.BF16)
    index.reserve(128)
    stats = DenseTraversalStats()
    index.on(executor).insert(np.arange(64), random_vectors(64, 16, seed=seed), stats=stats)
    assert stats.visited_members > 0
    assert stats.computed_distances > 0


def test_insert_reports_progress(executor, seed: int):
    """Reports never go backwards, the last one says the whole batch finished, and a handler answering None continues."""
    index = DenseIndex.init_ram(dimensions=16, quantization=Scalar.BF16)
    index.reserve(64)
    seen: list[tuple[int, int]] = []

    def watch(completed: int, total: int) -> None:
        seen.append((completed, total))

    index.on(executor).insert(np.arange(64), random_vectors(64, 16, seed=seed), progress=watch)
    assert len(index) == 64
    assert seen, "a watched batch reports at least once"
    assert all(earlier[0] <= later[0] for earlier, later in pairwise(seen)), "counts never decrease"
    assert {total for _, total in seen} == {64}, "every report names the same total"
    assert seen[-1] == (64, 64)


def test_insert_handler_reports_and_cannot_stop(executor, seed: int):
    """Every entry is claimed before the first report, so a handler answering False still inserts the whole batch."""
    index = DenseIndex.init_ram(dimensions=16, quantization=Scalar.BF16)
    index.reserve(64)
    index.on(executor).insert(np.arange(64), random_vectors(64, 16, seed=seed), progress=lambda done, total: done < 8)
    assert len(index) == 64


def test_insert_reports_a_raising_handler(executor, unraisable, seed: int):
    """A handler never raises into the library: its exception goes to `sys.unraisablehook`, and the cancel it
    answers cannot stop an insertion."""
    index = DenseIndex.init_ram(dimensions=16, quantization=Scalar.BF16)
    index.reserve(8)

    failure = ValueError()

    def broken(completed: int, total: int) -> bool:
        raise failure

    index.on(executor).insert(np.arange(8), random_vectors(8, 16, seed=seed), progress=broken)
    assert unraisable
    assert all(reported.exc_value is failure for reported in unraisable)
    assert len(index) == 8


# endregion Ingest

# region Search


@pytest.mark.parametrize("ndim", SETTINGS.dims)
def test_search_shapes(executor, ndim: int, seed: int):
    """The three outputs are shaped by the batch and the "k" that was asked for."""
    index, vectors = build_index(64, ndim, executor, seed=seed)
    stats = DenseTraversalStats()
    matches: Matches = index.on(executor).search(vectors[:3], 5, Arena(executor), stats=stats)

    assert np.asarray(matches.keys).shape == (3, 5, 16)
    assert np.asarray(matches.distances).shape == (3, 5)
    assert np.asarray(matches.counts).shape == (3,)
    assert len(matches) == 3
    assert stats.computed_distances > 0


def test_search_queries_are_matches(executor, seed: int):
    """A query's result is a tuple of scored entries, nearest first."""
    index, vectors = build_index(64, 32, executor, seed=seed)
    found = index.on(executor).search(vectors[:1], 4, Arena(executor))[0]
    assert all(isinstance(match, Match) for match in found)
    assert isinstance(found[0].key, Key)
    assert found[0].distance <= found[-1].distance


def test_search_reports_fewer_matches_than_asked(executor, seed: int):
    """A count bounds its own row, so an index holding less than `wanted` still answers."""
    index, vectors = build_index(3, 8, executor, seed=seed)
    matches = index.on(executor).search(vectors[:1], 10, Arena(executor))
    count = int(np.asarray(matches.counts)[0])
    assert count <= 3
    assert len(matches[0]) == count


class ArrowArrayColumn:
    """A column exporting only the PyCapsule array protocol, as a foreign producer does."""

    def __init__(self, exported):
        self.exported = exported

    def __arrow_c_array__(self, requested_schema=None):
        return self.exported.__arrow_c_array__(requested_schema)


class ArrowStreamColumn:
    """A column exporting only the PyCapsule stream protocol, as a Polars Series does."""

    def __init__(self, exported):
        self.exported = exported

    def __arrow_c_stream__(self, requested_schema=None):
        return self.exported.__arrow_c_stream__(requested_schema)


@pytest.mark.parametrize("column", [ArrowArrayColumn, ArrowStreamColumn])
def test_insert_reads_its_own_arrow_columns(executor, column: type, seed: int):
    """Rows exported as Arrow insert through the converter, with no Arrow library loaded."""
    source, _ = build_index(1, 8, executor, seed=seed)
    exported = source.on(executor).get(0, Arena(executor))
    target = DenseIndex.init_ram(dimensions=8, quantization=Scalar.BF16)
    target.on(executor).insert([41], column(exported))
    assert np.array_equal(np.asarray(target.on(executor).get(41, Arena(executor))), np.asarray(exported))


def test_arrow_columns_read_as_pyarrow_writes_them(executor, seed: int):
    """PyArrow's fixed-size lists and integer keys insert, across chunks, and matches export compacted."""
    pa = pytest.importorskip("pyarrow", reason="PyArrow is not installed")
    rows = np.random.default_rng(seed).random((6, 8), dtype=np.float32)
    embedding = pa.chunked_array(
        [pa.FixedSizeListArray.from_arrays(pa.array(part.ravel()), 8) for part in np.array_split(rows, 2)]
    )
    identifiers = pa.chunked_array([pa.array([10, 11, 12], pa.int64()), pa.array([13, 14, 15], pa.int64())])
    index = DenseIndex.init_ram(dimensions=8)
    index.on(executor).insert(identifiers, embedding)
    assert len(index) == 6

    arena = Arena(executor)
    queries = pa.FixedSizeListArray.from_arrays(pa.array(rows[:2].ravel()), 8)
    matches = index.on(executor).search(queries, 2, arena)
    compacted = pa.array(matches)
    assert compacted.type == pa.large_list(compacted.type.value_field)
    assert compacted.offsets.to_pylist() == [0, 2, 4]
    assert pa.array(matches.distances).type == pa.list_(pa.field("item", pa.float64(), nullable=False), 2)


def test_search_reuses_one_arena(executor, seed: int):
    """One arena serves a whole sweep, each call rewinding what the one before it wrote."""
    index, vectors = build_index(64, 16, executor, seed=seed)
    arena = Arena(executor)
    first = keys_to_integers(index.on(executor).search(vectors[:2], 4, arena).keys)
    second = index.on(executor).search(vectors[4:6], 4, arena)
    assert not np.array_equal(first, keys_to_integers(second.keys))
    assert int(np.asarray(second.counts)[0]) == 4


def test_search_rejects_an_overflowing_request(executor, seed: int):
    """An output matrix that cannot fit memory is refused rather than silently wrapped."""
    index, vectors = build_index(4, 8, executor, seed=seed)
    with pytest.raises((CapacityOverflowError, OutOfMemoryError)):
        index.on(executor).search(vectors[:2], 2**62, Arena(executor))


@pytest.mark.parametrize("ndim", [97, 256])
@pytest.mark.parametrize("metric", SPATIAL_METRICS)
def test_recall_against_the_oracle(executor, ndim: int, metric: DenseMetric, seed: int):
    """The index must return most of what a brute-force scan would, in every metric."""
    index, vectors = build_index(512, ndim, executor, metric=metric, seed=seed)
    stored = bf16_rounded(vectors)
    recall = measured_recall(index, stored, stored[:64], 10, executor, metric=metric)
    assert recall >= recall_floor(Scalar.BF16), f"{metric} recall collapsed to {recall}"


@pytest.mark.parametrize("quantization", QUANTIZATIONS)
def test_recall_survives_quantization(executor, quantization: Scalar, seed: int):
    """Every stored scalar type still finds what it should, at its own floor."""
    index, _ = build_index(512, 96, executor, quantization=quantization, seed=seed)
    recall = self_recall(index, Arena(executor), executor=executor).mean_recall
    assert recall >= recall_floor(quantization), f"{quantization} recall collapsed to {recall}"


@pytest.mark.parametrize("quantization", [Scalar.BF16, Scalar.F16, Scalar.I8])
def test_distances_agree_with_the_oracle(executor, quantization: Scalar, seed: int):
    """A reported distance matches what NumKong computes for the same pair, within tolerance."""
    index, vectors = build_index(128, 64, executor, quantization=quantization, seed=seed)
    queries = vectors[:8]
    matches = index.on(executor).search(queries, 1, Arena(executor))
    found = keys_to_integers(matches.keys)[:, 0]
    reported = np.asarray(matches.distances)[:, 0]
    truth = np.asarray(nk.cdist(queries, vectors, "angular", out_dtype="float64"))
    expected = truth[np.arange(len(queries)), found]
    atol, rtol = distance_tolerances(quantization)
    np.testing.assert_allclose(reported, expected, atol=atol, rtol=rtol)


def test_expansion_rides_the_call(executor, seed: int):
    """The breadth of the candidates is a per-call setting, so one immutable index serves a whole sweep."""
    index, vectors = build_index(512, 64, executor, seed=seed)
    queries = vectors[:32]
    narrow, wide = DenseTraversalStats(), DenseTraversalStats()
    arena = Arena(executor)
    index.on(executor).search(queries, 10, arena, expansion_factor=16, stats=narrow)
    index.on(executor).search(queries, 10, arena, expansion_factor=256, stats=wide)
    assert wide.computed_distances > narrow.computed_distances


def test_telemetry_is_a_per_call_reading(executor, seed: int):
    """A call overwrites the `DenseTraversalStats` it takes, and `+=` is how a sweep sums the readings."""
    index, vectors = build_index(128, 16, executor, seed=seed)
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


def test_filtered_search_admits_only_approved_keys(executor, seed: int):
    """A predicate narrows the answer without collapsing the walk that produced it."""
    index, vectors = build_index(128, 32, executor, seed=seed)
    matches = index.on(executor).search(vectors[:4], 8, Arena(executor), predicate=lambda key: int(key) % 2 == 0)
    found = keys_to_integers(matches.keys)
    counts = np.asarray(matches.counts)
    for query_index in range(len(counts)):
        assert np.all(found[query_index, : counts[query_index]] % 2 == 0)


def test_filtered_search_rejects_what_a_raising_predicate_was_asked(executor, unraisable, seed: int):
    """A predicate never raises into the library: its exception goes to `sys.unraisablehook` and the key is rejected."""
    index, vectors = build_index(64, 16, executor, seed=seed)

    failure = ValueError()

    def broken(key):
        raise failure

    matches = index.on(executor).search(vectors[:2], 4, Arena(executor), predicate=broken)
    assert np.asarray(matches.counts).tolist() == [0, 0]
    assert unraisable
    assert all(reported.exc_value is failure for reported in unraisable)


def test_scan_finds_every_row_itself(executor, seed: int):
    """A scan scores every entry exactly, so each self-query finds itself first."""
    index, vectors = build_index(64, 16, executor, seed=seed)
    matches = index.on(executor).search(vectors, 4, Arena(executor), traversal=Traversal.SCAN)
    assert keys_to_integers(matches.keys)[:, 0].tolist() == list(range(64))


# endregion Search

# region Retrieval


def test_count_and_contains(executor, seed: int):
    """Membership and multiplicity are the same question asked two ways."""
    index, _ = build_index(32, 16, executor, seed=seed)
    assert index.count(0) == 1
    assert index.count(10_000) == 0
    assert 0 in index
    assert 10_000 not in index


def test_get_returns_the_stored_rows(executor, seed: int):
    """A `bf16` index hands back `f32` rows, each value within a `bf16` step of the one inserted."""
    index = DenseIndex.init_ram(dimensions=8, quantization=Scalar.BF16)
    index.reserve(2)
    vectors = random_vectors(2, 8, seed=seed)
    index.on(executor).insert([1, 2], vectors)

    rows = index.get(1, Arena(), outputs_dtype=Scalar.F32)
    assert isinstance(rows, DenseVectors)
    assert not hasattr(rows, "__cuda_array_interface__")
    retrieved = np.asarray(rows)
    assert retrieved.shape == (1, 8)
    assert np.allclose(retrieved[0], vectors[0], rtol=BF16_EPSILON, atol=BF16_EPSILON)


def test_get_answers_empty_and_takes_no_subscript(executor, seed: int):
    """`get` answers absence with no rows, and a subscript, having no arena to write into, is no verb at all."""
    index, _ = build_index(4, 8, executor, seed=seed)
    absent = index.get(10_000, Arena())
    assert isinstance(absent, DenseVectors)
    assert len(absent) == 0
    assert not absent
    with pytest.raises(TypeError):
        index[0]


def test_get_leaves_unformattable_kinds_as_bytes(executor, seed: int):
    """Sub-byte and mini-float kinds have no `memoryview` format, so they leave as raw runs."""
    index = DenseIndex.init_ram(dimensions=8, quantization=Scalar.BF16)
    index.reserve(1)
    index.on(executor).insert([1], random_vectors(1, 8, seed=seed))
    view = memoryview(index.get(1, Arena(), outputs_dtype=Scalar.BF16))
    assert view.format == "B"
    assert view.shape == (1, 16)


def test_outputs_export_dlpack_in_place(executor, seed: int):
    """An output exports a DLPack tensor over its arena's own memory, on the device the arena describes."""
    index, _ = build_index(4, 8, executor, seed=seed)
    rows = index.on(executor).get(1, Arena(executor))
    assert rows.__dlpack_device__() == (1, 0)
    tensor = np.from_dlpack(rows)
    assert np.array_equal(tensor, np.asarray(rows))
    assert tensor.ctypes.data == np.asarray(rows).ctypes.data


class DLPackOnly:
    """Shows an array through DLPack alone, recording what each consumer asked the producer for."""

    def __init__(self, array: npt.NDArray[np.float32]) -> None:
        self.array, self.requests = array, []

    def __dlpack__(self, **requested):
        self.requests.append(requested)
        return self.array.__dlpack__(**requested)

    def __dlpack_device__(self) -> tuple[int, int]:
        return self.array.__dlpack_device__()


def test_inputs_arrive_through_dlpack(executor, seed: int):
    """A tensor exporting only DLPack reaches an index in place, its producer asked for no stream."""
    index, vectors = build_index(16, 8, executor, seed=seed)
    queries = DLPackOnly(vectors[:2])
    found = keys_to_integers(index.on(executor).search(queries, 1, Arena(executor)).keys)
    assert found[:, 0].tolist() == [0, 1]
    assert queries.requests
    assert all(requested["stream"] is None for requested in queries.requests)


def test_duplicate_keys_stack(executor, seed: int):
    """Keys need not be unique; several entries may share one and `count` reports how many."""
    index = DenseIndex.init_ram(dimensions=8, quantization=Scalar.BF16)
    index.reserve(3)
    index.on(executor).insert([5, 5, 5], random_vectors(3, 8, seed=seed))
    assert index.count(5) == 3
    assert np.asarray(index.get(5, Arena())).shape == (3, 8)


# endregion Retrieval

# region Iteration


def test_iteration_walks_every_key(executor, seed: int):
    """A snapshot walk visits each published entry exactly once, each pull naming its executor."""
    index, _ = build_index(300, 16, executor, seed=seed)
    cursor, arena, walked = index.cursor(), Arena(), []
    while batch := cursor.on(executor).next(64, arena):
        walked.append(np.array(batch.keys))
    assert sorted(keys_to_integers(np.concatenate(walked)).tolist()) == list(range(300))


def test_iteration_holds_links(executor, seed: int):
    """A pull asked for links holds them as offsets and keys, and one asked for none holds keys alone."""
    index, _ = build_index(40, 12, executor, seed=seed)
    cursor, arena = index.cursor(), Arena()
    batch = cursor.next(16, arena, links=True)
    offsets, linked = batch.links
    view = memoryview(offsets)
    assert view.format == "N"
    assert view.shape == (17,)
    assert len(batch) == 16
    assert np.asarray(batch.keys).shape == (16, 16)
    assert np.asarray(linked).shape == (view[16], 16)
    view.release()
    assert cursor.next(16, arena).links is None


def test_iteration_walks_from_start_keys(executor, seed: int):
    """A walk from start keys yields their own entries first, in the order given."""
    index, _ = build_index(40, 12, executor, seed=seed)
    cursor, arena = index.cursor([7, 3]), Arena()
    walked = keys_to_integers(np.array(cursor.next(2, arena).keys)).tolist()
    assert walked == [7, 3]


def test_search_stream_yields_nearest_first(executor, seed: int):
    """A search stream is resumable, ordered by what the walk has reached, where the build serves one."""
    index, vectors = build_index(128, 24, executor, seed=seed)
    try:
        stream, arena = index.on(executor).search_stream(vectors[0]), Arena(executor)
    except UnsupportedError:
        return
    streamed = [match.distance for match in stream.next(8, arena)[0]]
    streamed += [match.distance for match in stream.next(8, arena)[0]]
    assert len(streamed) == 16
    assert streamed == sorted(streamed)


def test_walks_close_before_they_are_collected(executor, seed: int):
    """`close()` and a `with` block release a cursor's snapshot or a stream's walk, after which the library refuses a
    pull."""
    index, vectors = build_index(32, 8, executor, seed=seed)
    arena = Arena()
    with index.cursor() as cursor:
        assert len(cursor.next(4, arena)) == 4
    with pytest.raises(InvalidArgumentError):
        cursor.next(4, arena)
    try:
        stream = index.search_stream(vectors[0])
    except UnsupportedError:
        return
    with stream:
        assert int(stream.next(1, arena)[0][0].key) == 0
    with pytest.raises(InvalidArgumentError):
        stream.next(1, arena)


# endregion Iteration

# region Composition


def test_reshard_gathers_every_source_into_the_target(executor, seed: int):
    """Several sources into one target gather every entry there, and the other sources end empty and live."""
    first = DenseIndex.init_ram(dimensions=16, quantization=Scalar.BF16)
    first.reserve(96)
    first.on(executor).insert(np.arange(0, 32), random_vectors(32, 16, seed=seed))
    donors = []
    for ordinal in (1, 2):
        donor = DenseIndex.init_ram(dimensions=16, quantization=Scalar.BF16)
        donor.reserve(32)
        rows = random_vectors(32, 16, seed=stream_key(seed, "donors", ordinal))
        donor.on(executor).insert(np.arange(32 * ordinal, 32 * ordinal + 32), rows)
        donors.append(donor)

    try:
        report = first.on(executor).reshard([first, *donors], [first])
    except UnsupportedError:
        return
    assert isinstance(report, RefineReport)
    assert len(first) == 96
    assert [40 in first, 80 in first] == [True, True]
    assert [len(donor) for donor in donors] == [0, 0]


def test_reshard_stopped_still_moves_every_entry(executor, seed: int):
    """A stop cuts the closing rounds alone, so every entry has moved in."""
    first, _ = build_index(8, 8, executor, seed=seed)
    second, _ = build_index(8, 8, executor, seed=stream_key(seed, "donors", 1))
    third, _ = build_index(8, 8, executor, seed=stream_key(seed, "donors", 2))
    with pytest.raises((CancelledError, UnsupportedError)) as raised:
        first.on(executor).reshard([first, second, third], [first], progress=lambda completed, total: False)
    if raised.type is UnsupportedError:
        return
    assert [len(first), len(second), len(third)] == [24, 0, 0]


def test_reshard_plan_continues_where_a_stop_left_it(executor, seed: int):
    """With a plan a stop leaves the rounds live, the next call naming it ends them, and the receiver closes the plan
    deferred as well."""
    first, _ = build_index(8, 8, executor, seed=seed)
    second, _ = build_index(8, 8, executor, seed=stream_key(seed, "second"))
    plan = DenseReshardPlan()
    try:
        with contextlib.suppress(CancelledError):
            first.on(executor).reshard([first, second], [first], progress=lambda completed, total: False, plan=plan)
    except UnsupportedError:
        return
    assert len(first) == 16
    assert plan.phase is not Phase.DONE
    first.on(executor).reshard([first, second], [first], plan=plan)
    assert plan.phase is Phase.DONE
    assert isinstance(first.deferred(executor).reshard_plan_finish(plan).wait(), RefineReport)


def test_reshard_refuses_a_source_named_twice(executor, seed: int):
    """The library refuses an index named twice among the sources, whatever receiver the call runs through."""
    index, _ = build_index(8, 8, executor, seed=seed)
    with pytest.raises(InvalidArgumentError):
        index.on(executor).reshard([index, index], [index])


def test_reshards_in_place_and_loads_only_manifests(executor, tmp_path, seed: int):
    """A reshard recuts the shards in place, and a lone monolith has no manifest a loaded handle could replace."""
    index, _ = build_index(32, 16, executor, seed=seed)
    index.on(executor).reshard([index], [index], memory_domains=[], shard_capacity=64)
    assert len(index) == 32
    path = str(tmp_path / "monolith.safegraphs")
    index.on(executor).save(path)
    with pytest.raises(UnsupportedError):
        DenseIndex.load_nvme(path, executor=Executor.for_nvme(threads=1))


def test_clear_keeps_the_geometry(executor, seed: int):
    """Clearing drops the entries and nothing else."""
    index, _ = build_index(32, 16, executor, seed=seed)
    capacity = index.capacity
    index.clear()
    assert len(index) == 0
    assert index.capacity == capacity
    assert index.dimensions == 16


# endregion Composition

# region Serialization


def test_saves_and_loads_a_file(executor, tmp_path, seed: int):
    """A checkpoint restores what it stored, its geometry read from its own headers."""
    index, _ = build_index(128, 32, executor, seed=seed)
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


def test_save_load_through_buffers(executor, seed: int):
    """`save_buffer` returns the monolith a `load_ram_buffer` reads back."""
    index, _ = build_index(64, 16, executor, seed=seed)
    buffer = index.on(executor).save_buffer(Arena(executor))
    assert isinstance(buffer, Monolith)
    assert memoryview(buffer).nbytes > 0
    assert DenseIndex.metadata_read_buffer(buffer).dimensions == 16

    reloaded = DenseIndex.load_ram_buffer(buffer)
    assert len(reloaded) == len(index)


def test_cancelled_load_creates_no_index(executor, tmp_path, seed: int):
    """A handler that stops a load leaves nothing behind but the exception."""
    index, _ = build_index(8, 8, executor, seed=seed)
    path = str(tmp_path / "cancelled.safegraphs")
    index.on(executor).save(path)
    with pytest.raises(CancelledError):
        DenseIndex.load_ram(path, progress=lambda completed, total: False)


def test_view_serves_in_place_and_refuses_mutation(executor, tmp_path, seed: int):
    """A viewed checkpoint answers searches and refuses every mutating verb."""
    index, vectors = build_index(32, 16, executor, seed=seed)
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


def test_encoded_path_is_read_as_a_path(executor, tmp_path, seed: int):
    """`os.fsencode` returns `bytes`, which still names a file."""
    index, _ = build_index(8, 8, executor, seed=seed)
    path = str(tmp_path / "encoded.safegraphs")
    index.on(executor).save(path)
    assert DenseIndex.metadata_read(os.fsencode(path)).dimensions == 8


def test_metadata_round_trips_through_a_factory(executor, tmp_path, seed: int):
    """The metadata fields are the geometry every factory takes, so they rebuild the handle."""
    index = init_ram_or_skip(dimensions=48, metric=DenseMetric.EUCLIDEAN, quantization=Scalar.I8, connectivity=32)
    index.reserve(8)
    index.on(executor).insert(np.arange(8), random_vectors(8, 48, seed=seed))
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


def test_refine_reports_a_sweep_and_leaves_the_index_standing(executor, seed: int):
    """A sweep re-selects edges without publishing or retiring an entry, so the entries survive it."""
    index, vectors = build_index(64, 16, executor, seed=seed)
    report = index.on(executor).refine(expansion_factor=64)
    assert isinstance(report, RefineReport)
    assert report.changed <= report.reselected, "a sweep changes only the nodes it re-selected"
    assert report.reachability.audited == len(index) == 64
    assert len(index.on(executor).search(vectors[:4], 5, Arena(executor))) == 4


def test_refine_reports_telemetry(executor, seed: int):
    """A round reports the distances it computed into the `DenseTraversalStats` a caller passes, and walks nothing."""
    index, _ = build_index(32, 16, executor, seed=seed)
    stats = DenseTraversalStats()
    index.on(executor).refine(stats=stats)
    assert stats.computed_distances > 0
    assert stats.visited_members == 0


def test_refine_takes_no_positionals(executor, seed: int):
    """Every setting a sweep has is a keyword, so a bare number cannot be mistaken for one."""
    index, _ = build_index(8, 8, executor, seed=seed)
    with pytest.raises(TypeError):
        index.refine(64)


def test_refine_rejects_unknown_keywords(executor, seed: int):
    """A misspelled setting is refused rather than silently derived."""
    index, _ = build_index(8, 8, executor, seed=seed)
    with pytest.raises(TypeError):
        index.on(executor).refine(expansion_add=64)


def test_refine_plan_resumes_and_finishes(executor, seed: int):
    """A plan keeps a stopped round live for the next call, and the receiver closes it with the whole plan's report."""
    index, _ = build_index(64, 16, executor, seed=seed)
    plan = DenseRefinePlan()
    with contextlib.suppress(CancelledError):
        index.on(executor).refine(plan=plan, progress=lambda completed, total: False)
    assert plan.phase is not Phase.DONE
    index.on(executor).refine(plan=plan)
    assert plan.phase is Phase.DONE
    assert index.on(executor).refine_plan_finish(plan).reachability.audited == 64
    with pytest.raises(InvalidArgumentError):
        index.on(executor).refine_plan_finish(plan)


def test_health_reaches_every_entry_after_a_refine(executor, seed: int):
    """A refine re-links whatever it strands, so the audit after it reaches the whole index."""
    index, _ = build_index(64, 16, executor, seed=seed)
    index.on(executor).refine()
    health = index.on(executor).health()
    assert isinstance(health, DenseReachability)
    assert health.audited == len(index)
    assert health.reachable == health.audited
    assert health.depth_from_root >= 1


def test_reshard_partitions_every_entry_and_empties_the_source(executor, seed: int):
    """One source into several targets cuts it between them, and the source stays live but empty."""
    index, _ = build_index(64, 16, executor, seed=seed)
    receivers = [DenseIndex.init_ram(dimensions=16, quantization=Scalar.BF16) for _ in range(2)]
    for receiver in receivers:
        receiver.reserve(64)
    report = index.on(executor).reshard([index], receivers, seed=seed)
    assert isinstance(report, RefineReport)
    assert sum(len(receiver) for receiver in receivers) == 64
    assert len(index) == 0


def test_sample_grows_a_superset_as_wanted_grows(executor, seed: int):
    """Equal seeds over an unchanged index draw equally, so a larger draw extends a smaller one."""
    index, _ = build_index(64, 16, executor, seed=seed)
    smaller = index.on(executor).sample(8, Arena(executor), seed=seed)
    larger = index.on(executor).sample(16, Arena(executor), seed=seed)
    assert isinstance(smaller, Sample)
    assert len(smaller.keys) == 8
    assert len(larger.keys) == 16
    assert len(larger.parents) == 16
    assert np.asarray(larger.coordinates).shape == (16, 3)
    drawn = {bytes(row) for row in np.asarray(larger.keys)}
    assert {bytes(row) for row in np.asarray(smaller.keys)} <= drawn


# endregion Refinement

# region Late Interaction

_LATE_INTERACTION_DOCUMENTS: int = 6
"""The document keys a late-interaction index holds."""

_LATE_INTERACTION_TOKENS: int = 3
"""The token rows every document key holds."""


def late_interaction_index(executor, seed: int, dimensions: int = 16) -> tuple[DenseIndex, npt.NDArray[np.float32]]:
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
    tokens = random_vectors(rows, dimensions, seed=seed)
    index.on(executor).insert(integers_to_keys(documents), tokens)
    return index, tokens


def scored_documents(index, tokens, queries: int, wanted: int, executor, **kwargs) -> DenseProducts:
    """Runs `maxsim` over *queries* whole queries."""
    return index.on(executor).maxsim(
        tokens[: queries * _LATE_INTERACTION_TOKENS], _LATE_INTERACTION_TOKENS, wanted, Arena(executor), **kwargs
    )


def test_maxsim_sizes_its_outputs_over_documents(executor, seed: int):
    """A query is a run of token rows, so the result has one run per query rather than per token."""
    index, tokens = late_interaction_index(executor, seed)
    queries = 2
    stats = DenseTraversalStats()
    documents = scored_documents(index, tokens, queries, 4, executor, stats=stats)
    offsets = np.asarray(documents.offsets)

    assert isinstance(documents, DenseProducts)
    assert len(documents) == queries
    assert offsets.shape == (queries + 1,)
    assert offsets[0] == 0
    assert offsets[-1] == len(np.asarray(documents.scores)) <= documents.required
    assert np.asarray(documents.keys).shape == (offsets[-1], 16)
    assert all(0 < end - begin <= 4 for begin, end in pairwise(offsets.tolist())), "each query keeps at most wanted"
    assert stats.computed_distances > 0


def test_maxsim_scores_each_document_once(executor, seed: int):
    """The reduction groups by key, so a document holding many token rows still yields one entry."""
    index, tokens = late_interaction_index(executor, seed)
    documents = scored_documents(index, tokens, 1, 4, executor)
    begin, end = np.asarray(documents.offsets).tolist()
    found = keys_to_integers(documents.keys)[begin:end]
    assert end > begin
    assert len(set(found.tolist())) == end - begin


def test_maxsim_orders_by_summed_maxima(executor, seed: int):
    """Summed maxima are a similarity, so a query's run descends where a search's distances ascend."""
    index, tokens = late_interaction_index(executor, seed)
    documents = scored_documents(index, tokens, 1, 4, executor)
    begin, end = np.asarray(documents.offsets).tolist()
    scored = np.asarray(documents.scores)[begin:end]
    assert list(scored) == sorted(scored, reverse=True)


def test_maxsim_refuses_ragged_queries(executor, seed: int):
    """The query rows must divide into whole queries, or the grouping is a guess."""
    index, tokens = late_interaction_index(executor, seed)
    with pytest.raises(ShapeMismatchError):
        index.on(executor).maxsim(tokens[: _LATE_INTERACTION_TOKENS + 1], _LATE_INTERACTION_TOKENS, 4, Arena())


def test_maxsim_scores_only_admitted_documents(executor, seed: int):
    """A predicate is asked before any of a document's vectors score, so a rejected one is never written."""
    index, tokens = late_interaction_index(executor, seed)
    documents = scored_documents(index, tokens, 2, 4, executor, predicate=lambda key: int(key) % 2 == 0)
    found = keys_to_integers(documents.keys)
    for begin, end in pairwise(np.asarray(documents.offsets).tolist()):
        assert end > begin
        assert np.all(found[begin:end] % 2 == 0)


def test_maxsim_keeps_every_scored_document_when_wanted_is_zero(executor, seed: int):
    """A zero bound keeps every document a query scored, so no query keeps fewer than under a bound,
    and a bound cuts what is kept but not what `required` counts."""
    index, tokens = late_interaction_index(executor, seed)
    every = np.diff(np.asarray(scored_documents(index, tokens, 2, 0, executor).offsets))
    bounded = np.diff(np.asarray(scored_documents(index, tokens, 2, 2, executor).offsets))
    scanned = np.diff(np.asarray(scored_documents(index, tokens, 2, 0, executor, traversal=Traversal.SCAN).offsets))
    cut = scored_documents(index, tokens, 2, 2, executor, traversal=Traversal.SCAN)
    assert np.all(every >= bounded)
    assert scanned.tolist() == [_LATE_INTERACTION_DOCUMENTS] * 2, "a scan nominates every document"
    assert cut.required == 2 * _LATE_INTERACTION_DOCUMENTS, "every scanned document counts, kept or not"


def test_maxsim_refuses_a_distance_kind(executor, seed: int):
    """Late interaction sums maxima over logits, which only the inner-product kinds score."""
    index, vectors = build_index(16, 8, executor, metric=DenseMetric.ANGULAR, seed=seed)
    with pytest.raises(ConfigMismatchError):
        index.on(executor).maxsim(vectors[:2], 2, 4, Arena())


def test_maxsim_reports_telemetry(executor, seed: int):
    """The walk reports into the `DenseTraversalStats` a caller passes, as every other walk does."""
    index, tokens = late_interaction_index(executor, seed)
    stats = DenseTraversalStats()
    scored_documents(index, tokens, 1, 4, executor, stats=stats)
    assert stats.computed_distances > 0


# endregion Late Interaction

# region Linear Algebra


def test_multiplication_reports_every_scored_pair(executor, seed: int):
    """A walk scores many more pairs than a search reports, and this verb hands them all back."""
    index, vectors = build_index(64, 32, executor, metric=DenseMetric.INNER, seed=seed)
    rows = vectors[:4]
    products = index.on(executor).multiply(rows, 0, Arena(executor))
    offsets = np.asarray(products.offsets)
    named = keys_to_integers(products.keys)
    scores = np.asarray(products.scores)

    assert isinstance(products, DenseProducts)
    assert len(products) == len(rows)
    assert offsets.shape == (len(rows) + 1,)
    assert offsets[0] == 0
    assert offsets[-1] == len(scores) == products.required
    assert np.all(np.diff(offsets) > 0)
    row_keys, row_scores = products.query(1)
    assert np.array_equal(np.asarray(row_scores), scores[offsets[1] : offsets[2]])
    assert np.array_equal(keys_to_integers(row_keys), named[offsets[1] : offsets[2]])
    assert not hasattr(products.scores, "__cuda_array_interface__")

    # Every product is an exact dot against the row it names, which is the point of the verb.
    for row_index in range(len(rows)):
        for pair in range(int(offsets[row_index]), int(offsets[row_index + 1])):
            expected = float(rows[row_index] @ vectors[named[pair]])
            assert scores[pair] == pytest.approx(expected, rel=BF16_EPSILON, abs=BF16_EPSILON)


def test_multiply_reports_what_wanted_cut(executor, seed: int):
    """A bound keeps the first pairs per left row, and `required` still counts every pair scored."""
    index, vectors = build_index(64, 16, executor, metric=DenseMetric.INNER, seed=seed)
    products = index.on(executor).multiply(vectors[:2], 4, Arena(executor))
    assert np.diff(np.asarray(products.offsets)).tolist() == [4, 4]
    assert products.required > 8, "both rows scored more pairs than were kept"


def test_multiply_refuses_a_predicate(executor, seed: int):
    """`multiply` has no filtered sibling, so the keyword is refused rather than ignored."""
    index, vectors = build_index(16, 8, executor, metric=DenseMetric.INNER, seed=seed)
    with pytest.raises(TypeError):
        index.on(executor).multiply(vectors[:1], 4, Arena(), predicate=lambda key: True)


@pytest.mark.parametrize("head_count", [1, 2])
def test_attention_is_exact_over_every_key(executor, head_count: int, seed: int):
    """A graph this small walks every key, so each head's output and carry are exact attention's."""
    dimensions, context, tokens = 16, 32, 3
    index, keys_matrix = build_index(context, dimensions, executor, metric=DenseMetric.INNER, seed=seed)
    values = random_vectors(context, dimensions, seed=stream_key(seed, "values"))
    queries = random_vectors(tokens * head_count, dimensions, seed=stream_key(seed, "queries"))
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


def test_attend_writes_into_the_arena(executor, seed: int):
    """The attended rows view the arena they were handed, so a decode loop reusing one allocates once."""
    index, keys_matrix = build_index(16, 8, executor, metric=DenseMetric.INNER, seed=seed)
    arena = Arena(executor)
    attended = index.on(executor).attend(
        keys_matrix[:2], bf16_words(keys_matrix), int, arena, values_dtype=Scalar.BF16, outputs_dtype=Scalar.F32
    )
    assert isinstance(attended, DenseVectors)
    assert np.any(np.asarray(attended) != 0)


def test_attend_refuses_a_missing_value_row(executor, unraisable, seed: int):
    """A missing row is a wrong softmax rather than a smaller one, and a lookup that raises misses its row."""
    index, keys_matrix = build_index(8, 8, executor, metric=DenseMetric.INNER, seed=seed)
    values = bf16_words(keys_matrix)
    with pytest.raises(ShapeMismatchError):
        index.on(executor).attend(keys_matrix[:1], values, lambda key: None, Arena(), values_dtype=Scalar.BF16)
    with pytest.raises(ShapeMismatchError):
        index.on(executor).attend(keys_matrix[:1], values[:4], int, Arena(), values_dtype=Scalar.BF16)
    with pytest.raises(ShapeMismatchError):
        index.on(executor).attend(keys_matrix[:1], keys_matrix, int, Arena())
    with pytest.raises(ShapeMismatchError):
        index.on(executor).attend(keys_matrix[:1], values, lambda key: 1 / 0, Arena(), values_dtype=Scalar.BF16)
    assert unraisable
    assert all(isinstance(reported.exc_value, ZeroDivisionError) for reported in unraisable)


def test_attend_refuses_a_half_width_carry(executor, seed: int):
    """A maximum alone is a number nothing can rescale, so the pair is sized exactly."""
    index, keys_matrix = build_index(8, 8, executor, metric=DenseMetric.INNER, seed=seed)
    values = bf16_words(keys_matrix)
    with pytest.raises(ValueError):
        index.on(executor).attend(
            keys_matrix[:1], values, int, Arena(), values_dtype=Scalar.BF16, carry=np.zeros(1, np.float32)
        )


def test_attend_refuses_a_distance_kind(executor, seed: int):
    """A softmax weighs logits, which only the inner-product kinds score."""
    index, keys_matrix = build_index(8, 8, executor, metric=DenseMetric.ANGULAR, seed=seed)
    with pytest.raises(ConfigMismatchError):
        index.on(executor).attend(keys_matrix[:1], bf16_words(keys_matrix), int, Arena(), values_dtype=Scalar.BF16)


# endregion Linear Algebra
