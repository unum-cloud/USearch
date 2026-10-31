/**
 *  @file javascript/usearch.ts
 *  @author Ash Vardanian
 *  @date March 30, 2024
 *  @brief Billion-Scale Multi-Modal Search, Clustering, & Sparse Algebra Engine for CPUs, GPUs, & NVMe Devices, for Node.
 *
 *  The package entry point: one module per C header underneath, re-exported here so a caller
 *  imports `usearch` and nothing deeper.
 */
export {
  Arena,
  ArenaView,
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
  DenseSearchStream,
  DenseTraversalStats,
  DenseVectors,
  DenseVerbs,
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
  DenseReachability,
  DenseRefineConfig,
  DenseRefineReport,
  DenseSampleConfig,
  DenseSearchConfig,
  DenseSearchStreamInitConfig,
  DenseSplitConfig,
  DenseViewNvmeConfig,
  ValueRowOf,
} from "./dense_index";

import { compiled } from "./library";
import {
  Arena,
  ArenaView,
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
  DenseSearchStream,
  DenseTraversalStats,
  DenseVectors,
  DenseVerbs,
} from "./dense_index";

/** The library's version, as the release tooling stamped it. Calls `usearch_version_name`. */
export function version(): string {
  return compiled.version();
}

/** The hardware capabilities this binary was compiled with, comma-separated. Calls `usearch_capabilities_compiled`. */
export function capabilitiesCompiled(): string {
  return compiled.capabilitiesCompiled();
}

/** The hardware capabilities the current machine offers at runtime, comma-separated. Calls `usearch_capabilities_available`. */
export function capabilitiesAvailable(): string {
  return compiled.capabilitiesAvailable();
}

const usearch = {
  Arena,
  ArenaView,
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
  DenseSearchStream,
  DenseTraversalStats,
  DenseVectors,
  DenseVerbs,
  Executor,
  Matches,
  QueryMatches,
  Monolith,
  Mood,
  Residency,
  RowsView,
  Scalar,
  Status,
  Traversal,
  capabilitiesAvailable,
  capabilitiesCompiled,
  formatKey,
  integerFromKey,
  keyAt,
  keyBytes,
  keyFromInteger,
  parseKey,
  version,
};
export default usearch;
