/**
 * Billion-Scale Multi-Modal Search, Clustering, & Sparse Algebra Engine for CPUs, GPUs, & NVMe Devices, for Node.
 *
 * The package entry point: one module per C header underneath, re-exported here so a caller
 * imports `usearch` and nothing deeper.
 */
export {
  Executor,
  MatchesRow,
  MatchesSpan,
  MetricKind,
  ScalarKind,
  formatKey,
  integerFromKey,
  keyAt,
  keyBytes,
  keyFromInteger,
  parseKey,
} from "./types";
export type {
  Key,
  KeyLike,
  KeysLike,
  Matrix,
  NvmeExecutorOptions,
  RamExecutorOptions,
  Vector,
  VectorOrMatrix,
  VramExecutorOptions,
} from "./types";

export { Cursor, DenseIndex, Finder, Stats } from "./dense_index";
export type {
  AttentionOptions,
  CursorBatch,
  DenseIndexOptions,
  FinderOptions,
  InsertionOptions,
  Metadata,
  NvmeIndexOptions,
  RamIndexOptions,
  SearchOptions,
  VramIndexOptions,
} from "./dense_index";

import { compiled } from "./library";
import {
  Executor,
  MatchesRow,
  MatchesSpan,
  MetricKind,
  ScalarKind,
  formatKey,
  integerFromKey,
  keyAt,
  keyBytes,
  keyFromInteger,
  parseKey,
} from "./types";
import { Cursor, DenseIndex, Finder, Stats } from "./dense_index";

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
  Cursor,
  DenseIndex,
  Executor,
  Finder,
  MatchesRow,
  MatchesSpan,
  MetricKind,
  ScalarKind,
  Stats,
  formatKey,
  capabilitiesAvailable,
  capabilitiesCompiled,
  integerFromKey,
  keyAt,
  keyBytes,
  keyFromInteger,
  parseKey,
  version,
};
export default usearch;
