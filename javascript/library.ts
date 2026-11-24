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
  describe(): { residency: string; device: number | null };
  free(): void;
}

/** One run of rows as it crosses: the array, its rows, its columns, the bytes between rows, and its scalar code. */
export type CompiledRows = [data: ArrayBufferView, rows: number, columns: number, stride: number, scalar: number];

/** A call submitted deferred, owning its outcome until it is final. */
export interface CompiledDeferred {
  status(): string;
  wait(): void;
  result(): unknown;
}

/** The reading a walking verb overwrites with its traversal counters. */
export interface CompiledStats {
  visitedMembers: number;
  computedDistances: number;
  restarts: number;
  missingHints: number;
}

/** One reachability audit. */
export interface CompiledReachability {
  audited: number;
  reachable: number;
  depthFromRoot: number;
  depthToRoot: number;
  unlinked: number;
  underlinked: number;
  beyondMostPopulatedDepth: number;
  unreachedIslands: number;
  strandedIslands: number;
  components: number;
}

/** What a refine or a reshard did to the graph. */
export interface CompiledRefineReport {
  reselected: number;
  changed: number;
  islandsFound: number;
  islandsReconnected: number;
  neighborhoods: number;
  widestNeighborhood: number;
  addedLinks: number;
  droppedLinks: number;
  verdict: string;
  opening: CompiledReachability;
  reachability: CompiledReachability;
}

/** Where a plan's last call stopped, the phase by its canonical name. */
export interface CompiledPhaseProgress {
  phase: string;
  ordinal: number;
  completed: number;
  total: number;
}

/** The draws a sample wrote, viewed inside its arena. */
export interface CompiledDraws {
  keys: Uint8Array;
  entryOrdinals: BigUint64Array;
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
  loadPath: string;
}

/** A ranked run viewed inside its arena. */
export interface CompiledMatches {
  keys: Uint8Array;
  scores: Float64Array;
  counts: BigUint64Array;
  columns: number;
}

/** Rows a call wrote inside its arena. */
export interface CompiledVectors {
  data: ArrayBufferView;
  rows: number;
  columns: number;
  stride: number;
}

/** The pairs a sparse product scored, viewed inside its arena. */
export interface CompiledProducts {
  offsets: BigUint64Array;
  keys: Uint8Array;
  scores: Float64Array;
  required: number;
}

/** The keys each entry of a cursor pull links to, viewed inside its arena as a ragged run. */
export interface CompiledLinks {
  offsets: BigUint64Array;
  keys: Uint8Array;
}

/** One cursor pull viewed inside its arena, the links present when they were asked for. */
export interface CompiledBatch {
  keys: Uint8Array;
  entries: number;
  links?: CompiledLinks;
}

/** A walk whose every pull lands in the arena it is handed. */
export interface CompiledCursor {
  next(
    executor: CompiledExecutor | undefined,
    wanted: number,
    arena: CompiledArena,
    links: boolean | undefined,
  ): CompiledBatch;
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
    ...arguments_: [
      ...On,
      expansion: number | undefined,
      memoryLimit: number | undefined,
      stats: CompiledStats | undefined,
      plan: ArrayBuffer | undefined,
    ]
  ): CompiledRefineReport | CompiledDeferred;
  refinePlanFinish(...arguments_: [...On, plan: ArrayBuffer]): CompiledRefineReport | CompiledDeferred;
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
  saveBuffer(...arguments_: [...On, arena: CompiledArena]): Uint8Array | CompiledDeferred;
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
  cursor(starts: Uint8Array | BigUint64Array | undefined): CompiledCursor;
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
    cpuClassCeiling: string | undefined,
  ): CompiledIndex;
  initVram(
    dimensions: number,
    metric: string,
    quantization: number,
    connectivity: number | undefined,
    valuesPerKey: number | undefined,
    devices: number[] | undefined,
    gpuClassCeiling: string | undefined,
  ): CompiledIndex;
  loadRam(
    path: string,
    executor: CompiledExecutor | undefined,
    memoryDomains: number[] | undefined,
    valuesPerKey: number | undefined,
    cpuClassCeiling: string | undefined,
  ): CompiledIndex;
  loadVram(
    path: string,
    executor: CompiledExecutor | undefined,
    devices: number[] | undefined,
    valuesPerKey: number | undefined,
    gpuClassCeiling: string | undefined,
  ): CompiledIndex;
  loadRamBuffer(
    buffer: ArrayBufferView,
    executor: CompiledExecutor | undefined,
    memoryDomains: number[] | undefined,
    valuesPerKey: number | undefined,
    cpuClassCeiling: string | undefined,
  ): CompiledIndex;
  loadVramBuffer(
    buffer: ArrayBufferView,
    executor: CompiledExecutor | undefined,
    devices: number[] | undefined,
    valuesPerKey: number | undefined,
    gpuClassCeiling: string | undefined,
  ): CompiledIndex;
  viewRam(path: string): CompiledIndex;
  viewNvme(
    path: string,
    executor: CompiledExecutor | undefined,
    paths: string[] | undefined,
    cpuClassCeiling: string | undefined,
    keyTableDomains: number[] | undefined,
  ): CompiledIndex;
  loadNvme(
    path: string,
    executor: CompiledExecutor | undefined,
    paths: string[] | undefined,
    cpuClassCeiling: string | undefined,
    keyTableDomains: number[] | undefined,
  ): CompiledIndex;
  metadataRead(path: string, executor: CompiledExecutor | undefined): CompiledMetadata;
  metadataReadBuffer(buffer: ArrayBufferView, executor: CompiledExecutor | undefined): CompiledMetadata;
  reshard(
    ...arguments_: [
      ...On,
      sources: CompiledIndex[],
      targets: CompiledIndex[],
      memoryDomains: number[] | undefined,
      devices: number[] | undefined,
      paths: string[] | undefined,
      shardCapacity: number | undefined,
      seed: number | bigint | undefined,
      expansion: number | undefined,
      memoryLimit: number | undefined,
      stats: CompiledStats | undefined,
      plan: ArrayBuffer | undefined,
    ]
  ): CompiledRefineReport | CompiledDeferred;
  reshardPlanFinish(...arguments_: [...On, plan: ArrayBuffer]): CompiledRefineReport | CompiledDeferred;
  plan(): ArrayBuffer;
  planProgress(plan: ArrayBuffer): CompiledPhaseProgress;
}

/** A ragged run of sequences or shapes a retrieval wrote, viewed inside its arena. */
export interface CompiledTape {
  data: ArrayBufferView;
  offsets: BigUint64Array | Uint32Array;
  entries: number;
}

/** One ragged batch as it crosses: its elements, their Apache Arrow offsets, its scalar code. */
export type CompiledTapeRows = [data: ArrayBufferView, offsets: Uint32Array | BigUint64Array, scalar: number];

/** The verbs every basic modality's handle shares, its inputs, outputs and lifecycle aside. */
interface CompiledBasicIndex<Description, KeyStats> {
  free(): void;
  describe(...on: On): Description | CompiledDeferred;
  health(...on: On): CompiledReachability | CompiledDeferred;
  clear(...on: On): undefined | CompiledDeferred;
  count(...arguments_: [...On, key: Uint8Array]): KeyStats | CompiledDeferred;
  refine(
    ...arguments_: [
      ...On,
      expansion: number | undefined,
      memoryLimit: number | undefined,
      stats: CompiledStats | undefined,
      plan: ArrayBuffer | undefined,
    ]
  ): CompiledRefineReport | CompiledDeferred;
  refinePlanFinish(...arguments_: [...On, plan: ArrayBuffer]): CompiledRefineReport | CompiledDeferred;
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
  saveBuffer(...arguments_: [...On, arena: CompiledArena]): Uint8Array | CompiledDeferred;
  cursor(starts: Uint8Array | BigUint64Array | undefined): CompiledCursor;
}

/** The lifecycle and reshard verbs every basic modality's class shares, its creation aside. */
interface CompiledBasicIndexConstructor<Index, Metadata> {
  metadataRead(path: string, executor: CompiledExecutor | undefined): Metadata;
  metadataReadBuffer(buffer: ArrayBufferView, executor: CompiledExecutor | undefined): Metadata;
  reshard(
    ...arguments_: [
      ...On,
      sources: Index[],
      targets: Index[],
      seed: number | bigint | undefined,
      expansion: number | undefined,
      memoryLimit: number | undefined,
      stats: CompiledStats | undefined,
      plan: ArrayBuffer | undefined,
    ]
  ): CompiledRefineReport | CompiledDeferred;
  reshardPlanFinish(...arguments_: [...On, plan: ArrayBuffer]): CompiledRefineReport | CompiledDeferred;
}

/** The metric and connectivity a sequence checkpoint froze. */
export interface CompiledSequenceMetadata {
  metric: string;
  connectivity: number;
}

/** A sequence index's frozen metadata and current counters. */
export type CompiledSequenceDescription = Omit<CompiledDescription, "metadata"> & {
  metadata: CompiledSequenceMetadata;
};

/** The entries under one key and the bytes they hold. */
export interface CompiledSequenceKeyStats {
  entries: number;
  totalBytes: number;
  largestBytes: number;
}

/** One sequence index handle. */
export interface CompiledSequenceIndex extends CompiledBasicIndex<
  CompiledSequenceDescription,
  CompiledSequenceKeyStats
> {
  reserve(...arguments_: [...On, entries: number, payloadBytes: number | undefined]): undefined | CompiledDeferred;
  insert(
    ...arguments_: [
      ...On,
      keys: Uint8Array | BigUint64Array,
      ...sequences: CompiledTapeRows,
      expansion: number | undefined,
      stats: CompiledStats | undefined,
    ]
  ): undefined | CompiledDeferred;
  search(
    ...arguments_: [
      ...On,
      ...queries: CompiledTapeRows,
      arena: CompiledArena,
      wanted: number,
      expansion: number | undefined,
      traversal: string | undefined,
      stats: CompiledStats | undefined,
    ]
  ): CompiledMatches | CompiledDeferred;
  get(...arguments_: [...On, key: Uint8Array, arena: CompiledArena]): CompiledTape | CompiledDeferred;
}

/** The weights an alignment kind scores with, as they cross. */
type CompiledWeights = [substitutionMatrix: Int8Array | undefined, gap: number | undefined];

/** The sequence index class. */
export interface CompiledSequenceIndexConstructor extends CompiledBasicIndexConstructor<
  CompiledSequenceIndex,
  CompiledSequenceMetadata
> {
  initRam(
    metric: string,
    ...arguments_: [
      ...CompiledWeights,
      windowWidths: number[] | undefined,
      connectivity: number | undefined,
      valuesPerKey: number | undefined,
      memoryDomains: number[] | undefined,
      cpuClassCeiling: string | undefined,
    ]
  ): CompiledSequenceIndex;
  initVram(
    metric: string,
    ...arguments_: [
      ...CompiledWeights,
      windowWidths: number[] | undefined,
      connectivity: number | undefined,
      valuesPerKey: number | undefined,
      devices: number[] | undefined,
      gpuClassCeiling: string | undefined,
    ]
  ): CompiledSequenceIndex;
  loadRam(...arguments_: [path: string, ...CompiledLoad, ...CompiledWeights]): CompiledSequenceIndex;
  loadVram(...arguments_: [path: string, ...CompiledLoad, ...CompiledWeights]): CompiledSequenceIndex;
  loadRamBuffer(...arguments_: [buffer: ArrayBufferView, ...CompiledLoad, ...CompiledWeights]): CompiledSequenceIndex;
  loadVramBuffer(...arguments_: [buffer: ArrayBufferView, ...CompiledLoad, ...CompiledWeights]): CompiledSequenceIndex;
  viewRam(path: string, cpuClassCeiling: string | undefined, ...weights: CompiledWeights): CompiledSequenceIndex;
  viewNvme(
    ...arguments_: [
      path: string,
      executor: CompiledExecutor | undefined,
      paths: string[] | undefined,
      cpuClassCeiling: string | undefined,
      ...CompiledWeights,
    ]
  ): CompiledSequenceIndex;
}

/** What every basic load takes last: executor, placement, key bound and class ceiling. */
type CompiledLoad = [
  executor: CompiledExecutor | undefined,
  placement: number[] | undefined,
  valuesPerKey: number | undefined,
  classCeiling: string | undefined,
];

/** The geometry and metric a shapes checkpoint froze. */
export interface CompiledShapesMetadata {
  metric: string;
  quantization: string;
  points: number;
  connectivity: number;
}

/** A shapes index's frozen metadata and current counters. */
export type CompiledShapesDescription = Omit<CompiledDescription, "metadata"> & {
  metadata: CompiledShapesMetadata;
  totalPoints: number;
};

/** The entries under one key and the points they hold. */
export interface CompiledShapesKeyStats {
  entries: number;
  totalPoints: number;
  largestPoints: number;
}

/** One shapes index handle. */
export interface CompiledShapesIndex extends CompiledBasicIndex<CompiledShapesDescription, CompiledShapesKeyStats> {
  reserve(...arguments_: [...On, entries: number, points: number | undefined]): undefined | CompiledDeferred;
  insert(
    ...arguments_: [
      ...On,
      keys: Uint8Array | BigUint64Array,
      ...shapes: CompiledTapeRows,
      expansion: number | undefined,
      stats: CompiledStats | undefined,
    ]
  ): undefined | CompiledDeferred;
  search(
    ...arguments_: [
      ...On,
      ...queries: CompiledTapeRows,
      arena: CompiledArena,
      wanted: number,
      expansion: number | undefined,
      traversal: string | undefined,
      stats: CompiledStats | undefined,
    ]
  ): CompiledMatches | CompiledDeferred;
  get(...arguments_: [...On, key: Uint8Array, scalar: number, arena: CompiledArena]): CompiledTape | CompiledDeferred;
}

/** The shapes index class. */
export interface CompiledShapesIndexConstructor extends CompiledBasicIndexConstructor<
  CompiledShapesIndex,
  CompiledShapesMetadata
> {
  initRam(
    metric: string,
    quantization: number,
    points: number | undefined,
    connectivity: number | undefined,
    valuesPerKey: number | undefined,
    memoryDomains: number[] | undefined,
    cpuClassCeiling: string | undefined,
  ): CompiledShapesIndex;
  initVram(
    metric: string,
    quantization: number,
    points: number | undefined,
    connectivity: number | undefined,
    valuesPerKey: number | undefined,
    devices: number[] | undefined,
    gpuClassCeiling: string | undefined,
  ): CompiledShapesIndex;
  loadRam(path: string, ...load: CompiledLoad): CompiledShapesIndex;
  loadVram(path: string, ...load: CompiledLoad): CompiledShapesIndex;
  loadRamBuffer(buffer: ArrayBufferView, ...load: CompiledLoad): CompiledShapesIndex;
  loadVramBuffer(buffer: ArrayBufferView, ...load: CompiledLoad): CompiledShapesIndex;
  viewRam(path: string, cpuClassCeiling: string | undefined): CompiledShapesIndex;
  viewNvme(
    path: string,
    executor: CompiledExecutor | undefined,
    paths: string[] | undefined,
    cpuClassCeiling: string | undefined,
  ): CompiledShapesIndex;
}

/** The connectivity a geospatial checkpoint froze. */
export interface CompiledGeospatialMetadata {
  connectivity: number;
}

/** A geospatial index's frozen metadata and current counters. */
export type CompiledGeospatialDescription = Omit<CompiledDescription, "metadata"> & {
  metadata: CompiledGeospatialMetadata;
};

/** The entries under one key. */
export interface CompiledGeospatialKeyStats {
  entries: number;
}

/** The relation, radius and time slack a geospatial query crosses with. */
type CompiledRelation = [relation: number, metres: number | undefined, seconds: number | undefined];

/** One geospatial index handle. */
export interface CompiledGeospatialIndex extends CompiledBasicIndex<
  CompiledGeospatialDescription,
  CompiledGeospatialKeyStats
> {
  reserve(...arguments_: [...On, entries: number]): undefined | CompiledDeferred;
  insert(
    ...arguments_: [
      ...On,
      keys: Uint8Array | BigUint64Array,
      observations: Float64Array,
      expansion: number | undefined,
      stats: CompiledStats | undefined,
    ]
  ): undefined | CompiledDeferred;
  search(
    ...arguments_: [
      ...On,
      queries: Float64Array,
      ...CompiledRelation,
      arena: CompiledArena,
      wanted: number,
      expansion: number | undefined,
      traversal: string | undefined,
      stats: CompiledStats | undefined,
    ]
  ): CompiledMatches | CompiledDeferred;
  within(
    ...arguments_: [
      ...On,
      queries: Float64Array,
      ...CompiledRelation,
      arena: CompiledArena,
      traversal: string | undefined,
      stats: CompiledStats | undefined,
    ]
  ): CompiledProducts | CompiledDeferred;
  get(...arguments_: [...On, key: Uint8Array, arena: CompiledArena]): Float64Array | CompiledDeferred;
}

/** The geospatial index class. */
export interface CompiledGeospatialIndexConstructor extends CompiledBasicIndexConstructor<
  CompiledGeospatialIndex,
  CompiledGeospatialMetadata
> {
  initRam(
    connectivity: number | undefined,
    valuesPerKey: number | undefined,
    memoryDomains: number[] | undefined,
    cpuClassCeiling: string | undefined,
  ): CompiledGeospatialIndex;
  initVram(
    connectivity: number | undefined,
    valuesPerKey: number | undefined,
    devices: number[] | undefined,
    gpuClassCeiling: string | undefined,
  ): CompiledGeospatialIndex;
  loadRam(path: string, ...load: CompiledLoad): CompiledGeospatialIndex;
  loadVram(path: string, ...load: CompiledLoad): CompiledGeospatialIndex;
  loadRamBuffer(buffer: ArrayBufferView, ...load: CompiledLoad): CompiledGeospatialIndex;
  loadVramBuffer(buffer: ArrayBufferView, ...load: CompiledLoad): CompiledGeospatialIndex;
  viewRam(path: string, cpuClassCeiling: string | undefined): CompiledGeospatialIndex;
  viewNvme(
    path: string,
    executor: CompiledExecutor | undefined,
    paths: string[] | undefined,
    cpuClassCeiling: string | undefined,
  ): CompiledGeospatialIndex;
}

/** Everything the native addon publishes. */
export interface Compiled {
  CompiledIndex: CompiledIndexConstructor;
  CompiledSequenceIndex: CompiledSequenceIndexConstructor;
  CompiledShapesIndex: CompiledShapesIndexConstructor;
  CompiledGeospatialIndex: CompiledGeospatialIndexConstructor;
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
