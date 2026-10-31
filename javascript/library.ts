/**
 *  @file javascript/library.ts
 *  @author Ash Vardanian
 *  @date September 9, 2026
 *  @brief The native boundary: the search that locates the prebuilt addon, and the shapes it publishes.
 *
 *  Every signature here is positional and takes no defaults, so the wrapper classes decide what
 *  an omitted argument means rather than the addon guessing. Every instance verb opens with the
 *  executor and the C `completion`, and answers its result when blocking or a `CompiledDeferred`
 *  when deferred.
 */
import build from "node-gyp-build";
import * as path from "path";
import { existsSync } from "fs";
import { getFileName, getRoot } from "bindings";

/** The C `usearch_completion_t` a verb is submitted with: `0` blocks, `1` defers. */
export type Completion = 0 | 1;

/** The workers a fanned-out verb runs on; its own declaration lives with {@link Executor} in `types.ts`. */
export interface CompiledExecutor {
  drain(mood?: string): void;
  free(): void;
}

/** Memory every output whose length scales with a call lands in, made where its executor computes. */
export interface CompiledArena {
  readonly residency: string;
  free(): void;
}

/** One run of rows as it crosses: the array, its rows, its columns, the bytes between rows, and its scalar code. */
export type CompiledRows = [data: ArrayBufferView, rows: number, columns: number, stride: number, scalar: number];

/** A call submitted deferred, owning its outcome until it is final. */
export interface CompiledDeferred {
  status(): string;
  wait(): void;
  settled(): Promise<void>;
  result(): unknown;
}

/** The reading a walking verb overwrites with its traversal counters. */
export interface CompiledStats {
  visitedMembers: number;
  computedDistances: number;
  restarts: number;
}

/** One reachability audit. */
export interface CompiledReachability {
  audited: number;
  reachable: number;
  sweeps: number;
  backSweeps: number;
}

/** What a refine, absorb or split did to the graph. */
export interface CompiledRefineReport {
  reselected: number;
  changed: number;
  islandsFound: number;
  islandsReconnected: number;
  reachability: CompiledReachability;
}

/** The draws a sample wrote, viewed inside its arena. */
export interface CompiledDraws {
  keys: Uint8Array | null;
  entryOrdinals: BigUint64Array | null;
  parents: CompiledMatches;
  coordinates: CompiledVectors;
  draws: number;
}

/** The geometry and metric a checkpoint froze, in the canonical spellings. */
export interface CompiledMetadata {
  metric: string;
  quantization: string;
  dimensions: number;
  connectivity: number;
}

/** A live index's frozen metadata and current counters. */
export interface CompiledDescription {
  metadata: CompiledMetadata;
  size: number;
  capacity: number;
  memoryUsage: number;
  cpuClass: string;
  gpuClass: string;
  shardsCount: number;
}

/** A ranked run viewed inside its arena; every view is `null` over device memory. */
export interface CompiledMatches {
  keys: Uint8Array | null;
  scores: Float64Array | null;
  counts: BigUint64Array | null;
  columns: number;
}

/** Rows a call wrote inside its arena. */
export interface CompiledVectors {
  data: ArrayBufferView | null;
  rows: number;
  columns: number;
  stride: number;
}

/** The pairs a sparse product scored, viewed inside its arena. */
export interface CompiledProducts {
  offsets: BigUint64Array | null;
  keys: Uint8Array | null;
  scores: Float64Array | null;
  required: number;
}

/** The keys each entry of a cursor pull links to, viewed inside its arena as a ragged run. */
export interface CompiledLinks {
  offsets: BigUint64Array | null;
  keys: Uint8Array | null;
}

/** One cursor pull viewed inside its arena, the links present when they were asked for. */
export interface CompiledBatch {
  keys: Uint8Array | null;
  entries: number;
  links?: CompiledLinks;
}

/** A walk whose every pull lands in the arena it is handed. */
export interface CompiledCursor {
  next(wanted: number, arena: CompiledArena, links: boolean | undefined): CompiledBatch;
  free(): void;
}

/** A resumable nearest-neighbor walk over one query row. */
export interface CompiledSearchStream {
  next(wanted: number, arena: CompiledArena): CompiledMatches;
  free(): void;
}

/** The leading arguments of every instance verb. */
type On = [executor: CompiledExecutor | undefined, completion: Completion];

/** One dense index handle; every verb answers its result, or a `CompiledDeferred` of it when deferred. */
export interface CompiledIndex {
  free(): void;
  describe(...on: On): CompiledDescription | CompiledDeferred;
  health(...on: On): CompiledReachability | CompiledDeferred;
  reserve(...arguments_: [...On, capacity: number]): undefined | CompiledDeferred;
  clear(...on: On): undefined | CompiledDeferred;
  count(...arguments_: [...On, key: Uint8Array]): number | CompiledDeferred;
  insert(
    ...arguments_: [
      ...On,
      keys: Uint8Array | BigUint64Array,
      ...vectors: CompiledRows,
      expansion: number | undefined,
      stats: CompiledStats | undefined,
    ]
  ): undefined | CompiledDeferred;
  refine(
    ...arguments_: [...On, expansion: number | undefined, stats: CompiledStats | undefined]
  ): CompiledRefineReport | CompiledDeferred;
  search(
    ...arguments_: [
      ...On,
      ...queries: CompiledRows,
      arena: CompiledArena,
      wanted: number,
      expansion: number | undefined,
      traversal: string | undefined,
      stats: CompiledStats | undefined,
    ]
  ): CompiledMatches | CompiledDeferred;
  maxsim(
    ...arguments_: [
      ...On,
      ...queries: CompiledRows,
      rowsPerQuery: number,
      arena: CompiledArena,
      wanted: number,
      expansion: number | undefined,
      stats: CompiledStats | undefined,
    ]
  ): CompiledProducts | CompiledDeferred;
  get(
    ...arguments_: [...On, key: Uint8Array, scalar: number, arena: CompiledArena]
  ): CompiledVectors | CompiledDeferred;
  multiply(
    ...arguments_: [
      ...On,
      ...left: CompiledRows,
      arena: CompiledArena,
      wanted: number | undefined,
      expansion: number | undefined,
      stats: CompiledStats | undefined,
    ]
  ): CompiledProducts | CompiledDeferred;
  absorb(...arguments_: [...On, donors: CompiledIndex[]]): CompiledRefineReport | CompiledDeferred;
  split(
    ...arguments_: [...On, receivers: CompiledIndex[], seed: number | bigint | undefined, expansion: number | undefined]
  ): CompiledRefineReport | CompiledDeferred;
  sample(
    ...arguments_: [
      ...On,
      arena: CompiledArena,
      wanted: number,
      shown: Uint8Array | BigUint64Array | undefined,
      ...shownCoordinates: CompiledRows | [undefined, undefined, undefined, undefined, undefined],
      focus: Uint8Array | BigUint64Array | undefined,
      seed: number | bigint | undefined,
      expansion: number | undefined,
    ]
  ): CompiledDraws | CompiledDeferred;
  save(...arguments_: [...On, path: string]): undefined | CompiledDeferred;
  saveBuffer(...arguments_: [...On, arena: CompiledArena]): Uint8Array | null | CompiledDeferred;
  attend(
    ...arguments_: [
      ...queries: CompiledRows,
      ...values: CompiledRows,
      rowOf: (key: Uint8Array) => number,
      arena: CompiledArena,
      scalar: number | undefined,
      carry: Float32Array | undefined,
      headCount: number | undefined,
      expansion: number | undefined,
      scoreScale: number | undefined,
      stats: CompiledStats | undefined,
    ]
  ): CompiledVectors;
  cursor(executor: CompiledExecutor | undefined, starts: Uint8Array | BigUint64Array | undefined): CompiledCursor;
  searchStream(
    ...arguments_: [executor: CompiledExecutor | undefined, ...query: CompiledRows, expansion: number | undefined]
  ): CompiledSearchStream;
}

/** The addon's index class, whose lifecycle verbs are the only way to a live handle. */
export interface CompiledIndexConstructor {
  initRam(
    dimensions: number,
    metric: string,
    quantization: number,
    connectivity: number | undefined,
    valuesPerKey: number | undefined,
    memoryDomains: number[] | undefined,
    cpuClass: string | undefined,
  ): CompiledIndex;
  initVram(
    dimensions: number,
    metric: string,
    quantization: number,
    connectivity: number | undefined,
    valuesPerKey: number | undefined,
    devices: number[] | undefined,
    gpuClass: string | undefined,
  ): CompiledIndex;
  loadRam(
    path: string,
    executor: CompiledExecutor | undefined,
    memoryDomains: number[] | undefined,
    valuesPerKey: number | undefined,
    cpuClass: string | undefined,
  ): CompiledIndex;
  loadVram(
    path: string,
    executor: CompiledExecutor | undefined,
    devices: number[] | undefined,
    valuesPerKey: number | undefined,
    gpuClass: string | undefined,
  ): CompiledIndex;
  loadRamBuffer(
    buffer: ArrayBufferView,
    executor: CompiledExecutor | undefined,
    memoryDomains: number[] | undefined,
    valuesPerKey: number | undefined,
    cpuClass: string | undefined,
  ): CompiledIndex;
  loadVramBuffer(
    buffer: ArrayBufferView,
    executor: CompiledExecutor | undefined,
    devices: number[] | undefined,
    valuesPerKey: number | undefined,
    gpuClass: string | undefined,
  ): CompiledIndex;
  viewRam(path: string): CompiledIndex;
  viewNvme(
    path: string,
    executor: CompiledExecutor | undefined,
    paths: string[] | undefined,
    cpuClass: string | undefined,
  ): CompiledIndex;
  metadataRead(path: string, executor: CompiledExecutor | undefined): CompiledMetadata;
  metadataReadBuffer(buffer: ArrayBufferView, executor: CompiledExecutor | undefined): CompiledMetadata;
}

/** Everything the native addon publishes. */
export interface Compiled {
  CompiledIndex: CompiledIndexConstructor;
  Executor: unknown;
  Arena: new (executor?: CompiledExecutor) => CompiledArena;
  scalars: Readonly<Record<string, number>>;
  version(): string;
  capabilitiesCompiled(): string;
  capabilitiesAvailable(): string;
}

/** Walks up from `directory` to the first ancestor holding a build or a prebuild tree. */
function findBuildDirectory(directory: string): string {
  if (existsSync(path.join(directory, "build"))) return directory;
  if (existsSync(path.join(directory, "prebuilds"))) return directory;
  if (path.basename(directory) === ".next") {
    // Next.js on a custom node, where the addon sits beside the bundle rather than under it.
    const sideways = path.join(directory, "..", "node_modules", "usearch");
    if (existsSync(sideways)) return findBuildDirectory(sideways);
  }
  if (directory === "/") throw new Error("Could not find native build for usearch");
  return findBuildDirectory(path.join(directory, ".."));
}

/** The directory this module was loaded from, however the bundler spelled it. */
function currentDirectory(): string {
  try {
    if (__dirname) return __dirname;
  } catch (error) {}
  return getRoot(getFileName());
}

/** The loaded addon, shared by every wrapper class. */
export const compiled: Compiled = build(findBuildDirectory(currentDirectory()));

// Unreachable, and kept so `ncc` traces the native module into its bundle. The name is `node.napi.node`
// because that is what `node-gyp-build` parses tags out of.
if (process.uptime() < 0) {
  require(__dirname + "/../../../prebuilds/darwin-arm64/node.napi.node");
  require(__dirname + "/../../../prebuilds/darwin-x64/node.napi.node");
  require(__dirname + "/../../../prebuilds/linux-arm64/node.napi.node");
  require(__dirname + "/../../../prebuilds/linux-x64/node.napi.node");
  require(__dirname + "/../../../prebuilds/win32-x64/node.napi.node");
}
