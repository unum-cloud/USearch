/**
 * The native boundary: the search that locates the prebuilt addon, and the shapes it publishes.
 *
 * Every signature here is positional and takes no defaults, so the wrapper classes decide what
 * an omitted argument means rather than the addon guessing.
 */
import build from "node-gyp-build";
import * as path from "path";
import { existsSync } from "fs";
import { getFileName, getRoot } from "bindings";

/** The workers a fanned-out verb runs on. */
export interface CompiledExecutor {
  drain(): void;
}

/** A cursor over the entries published when it was opened. */
export interface CompiledCursor {
  next(batch: number): [keys: Uint8Array, vectors: Float32Array] | null;
  free(): void;
}

/** A resumable nearest-neighbor stream over one query row. */
export interface CompiledFinder {
  next(wanted: number): [keys: Uint8Array, distances: Float64Array] | null;
  free(): void;
}

/** The geometry and metric a checkpoint froze, in the canonical spellings. */
export interface CompiledMetadata {
  dimensions: number;
  metric: string;
  quantization: string;
  connectivity: number;
  valuesPerKey: number;
}

/**
 * One dense index handle, with the scalar kind of every matrix argument named beside it.
 *
 * A walking verb takes its telemetry sink last: three slots the call overwrites, or `undefined`
 * to skip reporting.
 */
export interface CompiledIndex {
  dimensions(): number;
  connectivity(): number;
  size(): number;
  capacity(): number;
  memoryUsage(): number;
  storageRequirements(): number;
  capabilities(): string;
  reserve(capacity: number): void;
  clear(): void;
  add(
    keys: Uint8Array | BigUint64Array,
    vectors: ArrayBufferView,
    executor: CompiledExecutor,
    expansion: number,
    scalarKind: string | undefined,
    stats: Float64Array | undefined
  ): number;
  refine(
    executor: CompiledExecutor,
    expansion: number,
    stats: Float64Array | undefined
  ): number;
  search(
    queries: ArrayBufferView,
    keys: Uint8Array,
    distances: Float64Array,
    counts: BigUint64Array,
    executor: CompiledExecutor,
    expansion: number,
    scalarKind: string | undefined,
    stats: Float64Array | undefined
  ): void;
  maxsim(
    queries: ArrayBufferView,
    rowsPerQuery: number,
    keys: Uint8Array,
    distances: Float64Array,
    counts: BigUint64Array,
    executor: CompiledExecutor,
    expansion: number,
    scalarKind: string | undefined,
    stats: Float64Array | undefined
  ): void;
  multiply(
    left: ArrayBufferView,
    columns: Uint8Array,
    products: Float64Array,
    counts: BigUint64Array,
    executor: CompiledExecutor,
    expansion: number,
    scalarKind: string | undefined,
    stats: Float64Array | undefined
  ): void;
  attend(
    queries: ArrayBufferView,
    values: ArrayBufferView,
    valueDimensions: number,
    outputs: ArrayBufferView,
    carry: Float32Array | undefined,
    executor: CompiledExecutor,
    expansion: number,
    scoreScale: number,
    massTolerance: number,
    queriesKind: string | undefined,
    valuesKind: string | undefined,
    stats: Float64Array | undefined
  ): void;
  count(keys: Uint8Array | BigUint64Array): number[];
  contains(keys: Uint8Array | BigUint64Array): boolean[];
  get(key: Uint8Array | BigUint64Array): Float32Array;
  cursor(): CompiledCursor;
  find(
    query: ArrayBufferView,
    executor: CompiledExecutor,
    scalarKind: string | undefined
  ): CompiledFinder;
  save(path: string, executor: CompiledExecutor): void;
  load(path: string, executor: CompiledExecutor): void;
  saveBuffer(executor: CompiledExecutor): Uint8Array;
  loadBuffer(buffer: Uint8Array, executor: CompiledExecutor): void;
}

/**
 * The arm a residency selects: NUMA node ids for `ram`, CUDA ordinals for `vram`, device or
 * directory paths for `nvme`, and `undefined` to let the engine derive it.
 */
export type CompiledPlacement = number[] | string[] | undefined;

/** The addon's index constructor, and the checkpoint readers that need no index at all. */
export interface CompiledIndexConstructor {
  new (
    dimensions: number,
    metric: string,
    quantization: string,
    connectivity: number,
    valuesPerKey: number,
    residency: string,
    placement: CompiledPlacement
  ): CompiledIndex;
  metadata(path: string): CompiledMetadata;
  metadataBuffer(buffer: Uint8Array): CompiledMetadata;
}

/** The addon's executor constructor, whose residency decides what the other two arguments mean. */
export interface CompiledExecutorConstructor {
  new (
    residency: string,
    threads: number,
    placement: CompiledPlacement
  ): CompiledExecutor;
}

/** Everything the native addon publishes. */
export interface Compiled {
  CompiledIndex: CompiledIndexConstructor;
  Executor: CompiledExecutorConstructor;
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
  } catch (error) { }
  return getRoot(getFileName());
}

/** The loaded addon, shared by every wrapper class. */
export const compiled: Compiled = build(findBuildDirectory(currentDirectory()));

// Unreachable, and kept so `ncc` traces the native module into its bundle.
if (process.uptime() < 0) {
  require(__dirname + "/../../../prebuilds/darwin-arm64+x64/usearch.node");
  require(__dirname + "/../../../prebuilds/linux-arm64/usearch.node");
  require(__dirname + "/../../../prebuilds/linux-x64/usearch.node");
  require(__dirname + "/../../../prebuilds/win32-ia32/usearch.node");
  require(__dirname + "/../../../prebuilds/win32-x64/usearch.node");
  require(__dirname + "/../../../build/Release/usearch.node");
}
