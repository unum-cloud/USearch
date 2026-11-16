/**
 *  @file javascript/usearch.ts
 *  @author Ash Vardanian
 *  @date March 30, 2024
 *  @brief One Proximity Graph for Search, Sparse Products & Attention, across RAM, VRAM & NVMe,
 *      for Node.
 *
 *  The package entry point: one module per modality underneath, re-exported here so a caller
 *  imports `usearch` and nothing deeper.
 */
export {
  Arena,
  Deferred,
  Executor,
  Matches,
  QueryMatches,
  Monolith,
  Mood,
  Residency,
  RowsView,
  Scalar,
  Status,
  Tape,
  Traversal,
  formatKey,
  integerFromKey,
  keyAt,
  keyBytes,
  keyFromInteger,
  parseKey,
} from "./types";
export type {
  ArenaConstructor,
  ArenaDescription,
  ExecutorConstructor,
  ExecutorForNvmeConfig,
  ExecutorForRamConfig,
  ExecutorForVramConfig,
  ExecutorOnPoolConfig,
  Key,
  KeyLike,
  KeysLike,
  USearchError,
  Vector,
} from "./types";

export {
  DenseCursor,
  DenseCursorBatch,
  DenseCursorLinks,
  DenseDraws,
  DenseIndex,
  DenseIndexDeferredOn,
  DenseIndexOn,
  DenseMetric,
  DenseProducts,
  DenseRefinePlan,
  DenseReshardPlan,
  DenseSearchStream,
  DenseTraversalStats,
  DenseVectors,
} from "./dense_index";
export type {
  DenseAttendConfig,
  DenseCursorNextConfig,
  DenseGetConfig,
  DenseIndexDescription,
  DenseInsertConfig,
  DenseLoadRamBufferConfig,
  DenseLoadRamConfig,
  DenseLoadVramBufferConfig,
  DenseLoadVramConfig,
  DenseMaxsimConfig,
  DenseMetadata,
  DenseMetadataReadBufferConfig,
  DenseMetadataReadConfig,
  DenseMultiplyConfig,
  DenseOptions,
  DensePhaseProgress,
  DenseReachability,
  DenseRefineConfig,
  DenseRefineReport,
  DenseReshardConfig,
  DenseSampleConfig,
  DenseSearchConfig,
  DenseSearchStreamInitConfig,
  DenseViewNvmeConfig,
  RefineVerdict,
  ValueRowOf,
} from "./dense_index";

export {
  SequenceIndex,
  SequenceIndexDeferredOn,
  SequenceIndexOn,
  SequenceMetric,
  SequenceRefinePlan,
  SequenceReshardPlan,
} from "./sequence_index";
export type {
  SequenceIndexDescription,
  SequenceKeyStats,
  SequenceLoadConfig,
  SequenceMetadata,
  SequenceOptions,
  SequencePhaseProgress,
  SequenceRefineConfig,
  SequenceReshardConfig,
  SequenceViewNvmeConfig,
  SequenceWeights,
  SequencesLike,
} from "./sequence_index";

export {
  ShapesIndex,
  ShapesIndexDeferredOn,
  ShapesIndexOn,
  ShapesMetric,
  ShapesRefinePlan,
  ShapesReshardPlan,
} from "./shapes_index";
export type {
  ShapesGetConfig,
  ShapesIndexDescription,
  ShapesKeyStats,
  ShapesLike,
  ShapesLoadConfig,
  ShapesMetadata,
  ShapesOptions,
  ShapesPhaseProgress,
  ShapesRefineConfig,
  ShapesReshardConfig,
  ShapesViewNvmeConfig,
} from "./shapes_index";

export {
  GeospatialIndex,
  GeospatialIndexDeferredOn,
  GeospatialIndexOn,
  GeospatialRefinePlan,
  GeospatialRelation,
  GeospatialReshardPlan,
  observationsOf,
} from "./geospatial_index";
export type {
  GeospatialIndexDescription,
  GeospatialKeyStats,
  GeospatialLoadConfig,
  GeospatialMetadata,
  GeospatialObservation,
  GeospatialOptions,
  GeospatialPhaseProgress,
  GeospatialRefineConfig,
  GeospatialReshardConfig,
  GeospatialSearchConfig,
  GeospatialViewNvmeConfig,
  GeospatialWithinConfig,
} from "./geospatial_index";

import { compiled } from "./library";
import {
  Arena,
  Deferred,
  Executor,
  Matches,
  QueryMatches,
  Monolith,
  Mood,
  Residency,
  RowsView,
  Scalar,
  Status,
  Tape,
  Traversal,
  formatKey,
  integerFromKey,
  keyAt,
  keyBytes,
  keyFromInteger,
  parseKey,
} from "./types";
import {
  DenseCursor,
  DenseCursorBatch,
  DenseCursorLinks,
  DenseDraws,
  DenseIndex,
  DenseIndexDeferredOn,
  DenseIndexOn,
  DenseMetric,
  DenseProducts,
  DenseRefinePlan,
  DenseReshardPlan,
  DenseSearchStream,
  DenseTraversalStats,
  DenseVectors,
} from "./dense_index";
import {
  SequenceIndex,
  SequenceIndexDeferredOn,
  SequenceIndexOn,
  SequenceMetric,
  SequenceRefinePlan,
  SequenceReshardPlan,
} from "./sequence_index";
import {
  ShapesIndex,
  ShapesIndexDeferredOn,
  ShapesIndexOn,
  ShapesMetric,
  ShapesRefinePlan,
  ShapesReshardPlan,
} from "./shapes_index";
import {
  GeospatialIndex,
  GeospatialIndexDeferredOn,
  GeospatialIndexOn,
  GeospatialRefinePlan,
  GeospatialRelation,
  GeospatialReshardPlan,
  observationsOf,
} from "./geospatial_index";

/** The library's version, as the release tooling stamped it. */
export function version(): string {
  return compiled.version();
}

/** The hardware capabilities this binary was compiled with, comma-separated. */
export function capabilitiesCompiled(): string {
  return compiled.capabilitiesCompiled();
}

/** The hardware capabilities the current machine offers at runtime, comma-separated. */
export function capabilitiesAvailable(): string {
  return compiled.capabilitiesAvailable();
}

const usearch = {
  Arena,
  Deferred,
  DenseCursor,
  DenseCursorBatch,
  DenseCursorLinks,
  DenseDraws,
  DenseIndex,
  DenseIndexDeferredOn,
  DenseIndexOn,
  DenseMetric,
  DenseProducts,
  DenseRefinePlan,
  DenseReshardPlan,
  DenseSearchStream,
  DenseTraversalStats,
  DenseVectors,
  Executor,
  GeospatialIndex,
  GeospatialIndexDeferredOn,
  GeospatialIndexOn,
  GeospatialRefinePlan,
  GeospatialRelation,
  GeospatialReshardPlan,
  Matches,
  Monolith,
  Mood,
  QueryMatches,
  Residency,
  RowsView,
  Scalar,
  SequenceIndex,
  SequenceIndexDeferredOn,
  SequenceIndexOn,
  SequenceMetric,
  SequenceRefinePlan,
  SequenceReshardPlan,
  ShapesIndex,
  ShapesIndexDeferredOn,
  ShapesIndexOn,
  ShapesMetric,
  ShapesRefinePlan,
  ShapesReshardPlan,
  Status,
  Tape,
  Traversal,
  capabilitiesAvailable,
  capabilitiesCompiled,
  formatKey,
  integerFromKey,
  keyAt,
  keyBytes,
  keyFromInteger,
  observationsOf,
  parseKey,
  version,
};
export default usearch;
