"""Sequence, shapes and geospatial indexes, the modalities beside the dense one.

Each binding verb forwards to the C library and raises whatever its outcome says, so these tests
accept the library's statuses where a build or a machine cannot serve a request.

Run:
    python -m pytest test/modalities.py -v

File: test/modalities.py
Author: Ash Vardanian
Date: November 12, 2026
"""

import contextlib
from operator import attrgetter

import numpy as np
import numpy.typing as npt
import pytest

from usearch import (
    Arena,
    CancelledError,
    ConfigMismatchError,
    Error,
    GeospatialIndex,
    GeospatialIndexDescription,
    GeospatialRefinePlan,
    GeospatialRelation,
    GeospatialReshardPlan,
    InvalidArgumentError,
    Phase,
    RefineReport,
    RefineVerdict,
    SequenceIndex,
    SequenceIndexDescription,
    SequenceMetric,
    SequenceRefinePlan,
    SequenceReshardPlan,
    ShapesIndex,
    ShapesIndexDescription,
    ShapesMetric,
    ShapesRefinePlan,
    ShapesReshardPlan,
    Status,
    UnsupportedError,
)
from usearch.keys import keys_to_integers

MODALITIES = [SequenceIndex, ShapesIndex, GeospatialIndex]

PLANS = {
    SequenceIndex: (SequenceRefinePlan, SequenceReshardPlan),
    ShapesIndex: (ShapesRefinePlan, ShapesReshardPlan),
    GeospatialIndex: (GeospatialRefinePlan, GeospatialReshardPlan),
}
"""The refine and reshard plan types each modality's verbs take."""


def observations(latitudes: list[float], longitudes: list[float]) -> npt.NDArray[np.float64]:
    """An (N, 11) float64 batch placing each observation at a position, every other field unrecorded."""
    batch = np.full((len(latitudes), 11), np.nan)
    batch[:, 0], batch[:, 1] = latitudes, longitudes
    return batch


def entries(modality: type, count: int, seed: int) -> object:
    """A batch of `count` random entries of `modality`, in the form its `insert` takes."""
    generator = np.random.default_rng(seed)
    if modality is SequenceIndex:
        return [
            bytes(generator.integers(97, 123, size=int(generator.integers(8, 17)), dtype=np.uint8))
            for _ in range(count)
        ]
    if modality is ShapesIndex:
        return [generator.random((int(generator.integers(4, 13)), 3), dtype=np.float32) for _ in range(count)]
    return observations(generator.uniform(-60, 60, count).tolist(), generator.uniform(-180, 180, count).tolist())


def build(modality: type, count: int, executor, first_key: int = 0, seed: int = 0):
    """A host index of `modality` holding `count` random entries under consecutive keys from `first_key`."""
    index = modality.init_ram()
    if count:
        index.on(executor).insert(np.arange(first_key, first_key + count), entries(modality, count, seed))
    return index


@pytest.mark.parametrize("modality", MODALITIES, ids=attrgetter("__name__"))
def test_direct_construction_names_the_factories(modality: type):
    """No modality is built by calling its type; the refusal says what to call instead."""
    with pytest.raises(TypeError) as refusal:
        modality()
    assert str(refusal.value) == f"{modality.__name__} is built through init_*(), load_*(), view_ram() or view_nvme()"


@pytest.mark.parametrize(
    ("index", "description"),
    [
        (lambda: SequenceIndex.init_ram(SequenceMetric.LEVENSHTEIN_UTF8), SequenceIndexDescription),
        (lambda: ShapesIndex.init_ram(ShapesMetric.HAUSDORFF), ShapesIndexDescription),
        (lambda: GeospatialIndex.init_ram(), GeospatialIndexDescription),
    ],
    ids=["SequenceIndex", "ShapesIndex", "GeospatialIndex"],
)
def test_init_ram_starts_empty(index, description: type):
    """A fresh host index describes itself as empty, on a CPU class, with its frozen settings."""
    created = index()
    described = created.describe()
    assert isinstance(described, description)
    assert described.size == len(created) == 0
    assert described.cpu_class is not None and described.gpu_class is None


def test_init_ram_keeps_the_metric():
    """The metric a sequence or shapes index froze reads back as the enum it was given."""
    assert SequenceIndex.init_ram(SequenceMetric.LEVENSHTEIN_UTF8).describe().metadata.metric is (
        SequenceMetric.LEVENSHTEIN_UTF8
    )
    assert ShapesIndex.init_ram(ShapesMetric.HAUSDORFF).describe().metadata.metric is ShapesMetric.HAUSDORFF


def test_modality_enums_carry_the_library_spellings():
    """Every member is named by the library's dash-separated spelling of its value, in value order."""
    spellings = {
        SequenceMetric: "levenshtein levenshtein-utf8 needleman-wunsch overlap",
        ShapesMetric: "chamfer hausdorff gromov-wasserstein wasserstein-procrustes",
    }
    for enum, spelled in spellings.items():
        assert sorted(enum, key=int) == list(enum)
        assert {member.name for member in enum} == {name.upper().replace("-", "_") for name in spelled.split()}


@pytest.mark.parametrize("modality", MODALITIES, ids=attrgetter("__name__"))
def test_init_vram_surfaces_the_library_status(modality: type):
    """A VRAM index either opens on a GPU or raises the status the library answered, never one of the binding's."""
    try:
        index = modality.init_vram()
    except Error as refusal:
        assert isinstance(refusal.status, Status) and refusal.status not in (Status.SUCCESS, Status.DEFERRED)
        return
    assert index.describe().gpu_class is not None


def test_sequence_search_finds_the_inserted_entry():
    """Every sequence searched for comes back first under its own key, at distance zero."""
    index = SequenceIndex.init_ram()
    index.insert([1, 2, 3], ["kitten", b"sitting", bytearray(b"mitten")])
    assert len(index) == index.describe().size == 3
    matches = index.search(["sitting"], 1, Arena())
    assert keys_to_integers(np.asarray(matches.keys)[0]).tolist() == [2]
    assert np.asarray(matches.distances)[0, 0] == 0


def test_shapes_search_finds_the_inserted_entry():
    """Shapes of different point counts share one batch, and each finds itself first."""
    generator = np.random.default_rng(42)
    shapes = [generator.random((points, 3), dtype=np.float32) for points in (5, 9, 13)]
    index = ShapesIndex.init_ram()
    index.insert([1, 2, 3], shapes)
    assert len(index) == index.describe().size == 3
    assert index.describe().total_points == 27
    matches = index.search(shapes[2], 1, Arena())
    assert keys_to_integers(np.asarray(matches.keys)[0]).tolist() == [3]


def test_geospatial_search_finds_the_inserted_entry():
    """An observation searched for comes back first, at zero chord metres."""
    batch = observations([48.8566, 52.52, 40.7128], [2.3522, 13.405, -74.006])
    index = GeospatialIndex.init_ram()
    index.insert([1, 2, 3], batch)
    assert len(index) == index.describe().size == 3
    matches = index.search(batch[1], 1, Arena(), relation=GeospatialRelation.NEAR)
    assert keys_to_integers(np.asarray(matches.keys)[0]).tolist() == [2]
    assert np.asarray(matches.distances)[0, 0] == 0


def test_geospatial_within_reports_only_the_entries_inside_the_radius():
    """Two observations a kilometre apart fall inside a two-kilometre radius, and a far one does not."""
    batch = observations([48.8566, 48.8656, 52.52], [2.3522, 2.3522, 13.405])
    index = GeospatialIndex.init_ram()
    index.insert([1, 2, 3], batch)
    report = index.within(batch[0], Arena(), metres=2000.0)
    keys, _ = report.query(0)
    assert sorted(keys_to_integers(np.asarray(keys)).tolist()) == [1, 2]


# region Refinement


@pytest.mark.parametrize("modality", MODALITIES, ids=attrgetter("__name__"))
def test_refine_reports_a_round_over_every_entry(modality: type, executor):
    """A round reports what it did, its closing audit judging every entry, and leaves each entry findable."""
    index = build(modality, 64, executor)
    report = index.on(executor).refine()
    assert isinstance(report, RefineReport) and isinstance(report.verdict, RefineVerdict)
    assert report.changed <= report.reselected
    assert report.reachability.audited == len(index) == 64
    assert 17 in index


@pytest.mark.parametrize("modality", MODALITIES, ids=attrgetter("__name__"))
def test_refine_plan_resumes_and_finishes(modality: type, executor):
    """A plan keeps a stopped round live for the next call, and the receiver closes it once, with the whole plan's
    report."""
    index = build(modality, 64, executor)
    plan = PLANS[modality][0]()
    with contextlib.suppress(CancelledError):
        index.on(executor).refine(plan=plan, progress=lambda completed, total: False)
    assert plan.phase is not Phase.DONE and plan.completed <= plan.total
    index.on(executor).refine(plan=plan)
    assert plan.phase is Phase.DONE
    assert index.on(executor).refine_plan_finish(plan).reachability.audited == 64
    with pytest.raises(InvalidArgumentError):
        index.on(executor).refine_plan_finish(plan)


@pytest.mark.parametrize("modality", MODALITIES, ids=attrgetter("__name__"))
def test_refine_plan_needs_an_executor(modality: type):
    """A plan lives across calls on one executor, so a call on the calling thread cannot open one."""
    index = build(modality, 0, None)
    with pytest.raises(InvalidArgumentError):
        index.refine(plan=PLANS[modality][0]())


# endregion Refinement

# region Composition


@pytest.mark.parametrize("modality", MODALITIES, ids=attrgetter("__name__"))
def test_reshard_gathers_every_source_into_the_target(modality: type, executor):
    """Several sources into one target gather every entry there, and the other sources end empty and live."""
    first = build(modality, 32, executor)
    donors = [build(modality, 32, executor, first_key=32 * seed, seed=seed) for seed in (1, 2)]
    report = first.on(executor).reshard([first, *donors], [first])
    assert isinstance(report, RefineReport)
    assert len(first) == 96
    assert [40 in first, 80 in first] == [True, True]
    assert [len(donor) for donor in donors] == [0, 0]


@pytest.mark.parametrize("modality", MODALITIES, ids=attrgetter("__name__"))
def test_reshard_cuts_one_source_into_several_targets(modality: type, executor):
    """One source into several targets cuts it between them, and the source stays live but empty."""
    source = build(modality, 64, executor)
    targets = [modality.init_ram() for _ in range(2)]
    report = source.on(executor).reshard([source], targets, seed=7)
    assert isinstance(report, RefineReport)
    assert sum(len(target) for target in targets) == 64
    assert all(len(target) for target in targets)
    assert len(source) == 0


@pytest.mark.parametrize("modality", MODALITIES, ids=attrgetter("__name__"))
def test_reshard_cut_runs_through_a_plan_and_a_deferred_finish(modality: type, executor):
    """A cut a stop interrupts keeps every entry moved, the next call naming the plan ends its rounds, and a deferred
    receiver closes it."""
    source = build(modality, 64, executor)
    targets = [modality.init_ram() for _ in range(2)]
    plan = PLANS[modality][1]()
    with contextlib.suppress(CancelledError):
        source.on(executor).reshard([source], targets, progress=lambda completed, total: False, plan=plan)
    assert sum(len(target) for target in targets) == 64 and plan.phase is not Phase.DONE
    source.on(executor).reshard([source], targets, plan=plan)
    assert plan.phase is Phase.DONE
    assert isinstance(source.deferred(executor).reshard_plan_finish(plan).wait(), RefineReport)


@pytest.mark.parametrize("modality", MODALITIES, ids=attrgetter("__name__"))
def test_reshard_recut_of_one_handle_is_unsupported(modality: type, executor):
    """One index as its only source and target has a single shard to recut, which the library refuses."""
    index = build(modality, 16, executor)
    with pytest.raises(UnsupportedError):
        index.on(executor).reshard([index], [index])
    assert len(index) == 16


@pytest.mark.parametrize("modality", MODALITIES, ids=attrgetter("__name__"))
def test_reshard_refuses_malformed_lists_and_moves_nothing(modality: type, executor):
    """Empty or repeated lists are malformed and a target holding foreign entries mismatches; every refusal leaves
    the entries where they were."""
    first, second = build(modality, 8, executor), build(modality, 8, executor, first_key=8, seed=1)
    with pytest.raises(InvalidArgumentError):
        first.on(executor).reshard([], [first])
    with pytest.raises(InvalidArgumentError):
        first.on(executor).reshard([first, first], [first])
    with pytest.raises(ConfigMismatchError):
        first.on(executor).reshard([first], [second])
    with pytest.raises(TypeError):
        first.on(executor).reshard([first, object()], [first])
    assert [len(first), len(second)] == [8, 8]


# endregion Composition
