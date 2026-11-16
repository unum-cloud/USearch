/**
 *  @file javascript/shapes_index.ts
 *  @author Ash Vardanian
 *  @date November 16, 2026
 *  @brief Similarity search over 3D point clouds of varying sizes, and the receivers its verbs run
 *      through.
 */
import {
  Completion,
  CompiledPhaseProgress,
  CompiledShapesDescription,
  CompiledShapesIndex,
  CompiledShapesKeyStats,
  CompiledShapesMetadata,
  CompiledTape,
  CompiledDraws,
  CompiledMatches,
  compiled,
} from "./library";
import {
  Arena,
  Deferred,
  Executor,
  KeyLike,
  KeysLike,
  Matches,
  Monolith,
  RowsView,
  Scalar,
  Status,
  Tape,
  Vector,
  crossing,
  fail,
  normalizeKey,
  normalizeKeys,
  scalarCodes,
  tapeCrossing,
} from "./types";
import {
  DenseCall,
  DenseCursor,
  DenseDraws,
  DenseInsertConfig,
  DenseReachability,
  DenseRefineConfig,
  DenseRefineReport,
  DenseReshardConfig,
  DenseSampleConfig,
  DenseSearchConfig,
  answerNothing,
  answerReport,
  block,
  defer,
  denseCall,
} from "./dense_index";

/** How a shapes index scores two point clouds, spelled as the library names each kind.
 *  @enum {string}
 *  @readonly */
export enum ShapesMetric {
  /** Mean distance to each point's nearest partner, both directions, halved. */
  Chamfer = "chamfer",
  /** The worst of all nearest-partner distances, both directions. */
  Hausdorff = "hausdorff",
  /** Transport over each side's internal geometry, the best ranker and the costliest. */
  GromovWasserstein = "gromov-wasserstein",
  /** Transport after the best rotation or reflection superposes the clouds. */
  WassersteinProcrustes = "wasserstein-procrustes",
}

/** The geometry every shapes index freezes at initialization, whatever its residency. */
export interface ShapesOptions {
  /** How clouds are scored; defaults to `chamfer`. */
  metric?: ShapesMetric;
  /** The scalar kind points are stored in, `f16` or `f32`; defaults to `f32`. */
  quantization?: Scalar;
  /** Points every shape holds; absent admits any count. */
  points?: number;
  /** Neighbors per node; absent derives it. */
  connectivity?: number;
  /** Entries one key may carry; absent is unbounded. */
  valuesPerKey?: number;
  /** The newest CPU class dispatch may pick, by canonical name; absent means no ceiling. */
  cpuClassCeiling?: string;
  /** The newest GPU class dispatch may pick, by canonical name; absent means no ceiling. */
  gpuClassCeiling?: string;
}

/** Optional settings of a load from a path or a buffer. */
export interface ShapesLoadConfig {
  /** The workers the read fans out over; absent reads on the calling thread. */
  executor?: Executor;
  /** NUMA node IDs for a RAM load, or CUDA ordinals for a VRAM one; absent derives them. */
  placement?: number[];
  /** Entries one key may carry; absent is unbounded. */
  valuesPerKey?: number;
  /** The newest CPU or GPU class dispatch may pick, by canonical name; absent means no ceiling. */
  classCeiling?: string;
}

/** Optional settings of a view from NVMe drives. */
export interface ShapesViewNvmeConfig {
  /** The NVMe executor whose rings read the checkpoint. */
  executor?: Executor;
  /** The drives to serve the parts from; absent serves each where its manifest points. */
  paths?: string[];
  /** The newest CPU class dispatch may pick, by canonical name; absent means no ceiling. */
  cpuClassCeiling?: string;
}

/** The geometry and metric a checkpoint froze. */
export interface ShapesMetadata {
  /** How the checkpoint scores clouds. */
  metric: ShapesMetric;
  /** The scalar kind points are stored in. */
  quantization: Scalar;
  /** Points every shape holds, or zero for any count. */
  points: number;
  /** Neighbors per node the graph was built with. */
  connectivity: number;
}

/** A live index's frozen metadata and current counters, each read once. */
export type ShapesIndexDescription = Omit<CompiledShapesDescription, "metadata"> & {
  metadata: ShapesMetadata;
};

/** The entries one key holds and the points they span. */
export type ShapesKeyStats = CompiledShapesKeyStats;

/** Where a plan's last call stopped. */
export type ShapesPhaseProgress = CompiledPhaseProgress;

/** One refine continued across calls; a receiver's `refinePlanFinish` closes it. While it lives the
 *  index admits no insertion and no save. */
export class ShapesRefinePlan {
  /** The memory the library writes the plan into. */
  readonly state: ArrayBuffer = compiled.CompiledIndex.plan();

  /** Where the last call stopped, "none" before the first and "done" once its round ended. */
  get phase(): string {
    return this.progress.phase;
  }

  /** The phase the last call stopped in, with its position and its units. */
  get progress(): ShapesPhaseProgress {
    return compiled.CompiledIndex.planProgress(this.state);
  }
}

/** One reshard continued across calls; a receiver's `reshardPlanFinish` closes it. Every index it
 *  names admits no insertion and no save while it lives. */
export class ShapesReshardPlan {
  /** The memory the library writes the plan into. */
  readonly state: ArrayBuffer = compiled.CompiledIndex.plan();

  /** Where the last call stopped, "none" before the first and "done" once its rounds ended. */
  get phase(): string {
    return this.progress.phase;
  }

  /** The phase the last call stopped in, with its position and its units. */
  get progress(): ShapesPhaseProgress {
    return compiled.CompiledIndex.planProgress(this.state);
  }
}

/** Optional per-call settings of a refinement round. */
export type ShapesRefineConfig = Omit<DenseRefineConfig, "plan"> & {
  /** The plan this call continues; absent runs one round alone. */
  plan?: ShapesRefinePlan;
};

/** Optional settings of a reshard; the basic modalities take no placement. */
export type ShapesReshardConfig = Pick<DenseReshardConfig, "seed" | "expansionFactor" | "memoryLimit" | "stats"> & {
  /** The plan this call continues; absent moves and runs to the end. */
  plan?: ShapesReshardPlan;
};

/** Shapes as a verb takes them: clouds of interleaved `x, y, z`, or a tape already packed. */
export type ShapesLike = Tape | ReadonlyArray<Float64Array | Float32Array | Float16Array>;

function shapesOf(shapes: ShapesLike): Tape {
  return shapes instanceof Tape ? shapes : Tape.ofShapes(shapes);
}

function metadataOf(read: CompiledShapesMetadata): ShapesMetadata {
  return {
    metric: read.metric as ShapesMetric,
    quantization: read.quantization as Scalar,
    points: read.points,
    connectivity: read.connectivity,
  };
}

/** Optional settings of a retrieval. */
export interface ShapesGetConfig {
  /** The scalar kind to export points as, converted from storage; defaults to `f32`. */
  scalar?: Scalar;
}

function shapesReserve(
  index: CompiledShapesIndex,
  executor: Executor | undefined,
  entries: number,
  points?: number,
): DenseCall<void> {
  return denseCall((completion: Completion) => index.reserve(executor, completion, entries, points), answerNothing);
}

function shapesClear(index: CompiledShapesIndex, executor: Executor | undefined): DenseCall<void> {
  return denseCall((completion: Completion) => index.clear(executor, completion), answerNothing);
}

function shapesDescribe(index: CompiledShapesIndex, executor: Executor | undefined): DenseCall<ShapesIndexDescription> {
  return denseCall(
    (completion: Completion) => index.describe(executor, completion),
    (described: CompiledShapesDescription) => ({ ...described, metadata: metadataOf(described.metadata) }),
  );
}

function shapesHealth(index: CompiledShapesIndex, executor: Executor | undefined): DenseCall<DenseReachability> {
  return denseCall(
    (completion: Completion) => index.health(executor, completion),
    (reachability: DenseReachability) => reachability,
  );
}

function shapesCount(
  index: CompiledShapesIndex,
  executor: Executor | undefined,
  key: KeyLike,
): DenseCall<ShapesKeyStats> {
  const normalized = normalizeKey(key);
  return denseCall(
    (completion: Completion) => index.count(executor, completion, normalized),
    (stats: ShapesKeyStats) => stats,
  );
}

function shapesInsert(
  index: CompiledShapesIndex,
  executor: Executor | undefined,
  keys: KeysLike,
  shapes: ShapesLike,
  config: DenseInsertConfig,
): DenseCall<void> {
  const normalized = normalizeKeys(keys);
  const tape = tapeCrossing(shapesOf(shapes));
  return denseCall(
    (completion: Completion) =>
      index.insert(executor, completion, normalized, ...tape, config.expansionFactor, config.stats),
    answerNothing,
  );
}

function shapesRefine(
  index: CompiledShapesIndex,
  executor: Executor | undefined,
  config: ShapesRefineConfig,
): DenseCall<DenseRefineReport> {
  return denseCall(
    (completion: Completion) =>
      index.refine(executor, completion, config.expansionFactor, config.memoryLimit, config.stats, config.plan?.state),
    answerReport,
  );
}

function shapesRefinePlanFinish(
  index: CompiledShapesIndex,
  executor: Executor | undefined,
  plan: ShapesRefinePlan,
): DenseCall<DenseRefineReport> {
  return denseCall((completion: Completion) => index.refinePlanFinish(executor, completion, plan.state), answerReport);
}

function shapesSearch(
  index: CompiledShapesIndex,
  executor: Executor | undefined,
  queries: ShapesLike,
  wanted: number,
  arena: Arena,
  config: DenseSearchConfig,
): DenseCall<Matches> {
  const tape = shapesOf(queries);
  return denseCall(
    (completion: Completion) =>
      index.search(
        executor,
        completion,
        ...tapeCrossing(tape),
        arena,
        wanted,
        config.expansionFactor,
        config.traversal,
        config.stats,
      ),
    (outputs: CompiledMatches) => new Matches(outputs, tape.entries),
  );
}

function shapesGet(
  index: CompiledShapesIndex,
  executor: Executor | undefined,
  key: KeyLike,
  arena: Arena,
  config: ShapesGetConfig,
): DenseCall<Tape> {
  const normalized = normalizeKey(key);
  const scalar = config.scalar ?? Scalar.F32;
  return denseCall(
    (completion: Completion) => index.get(executor, completion, normalized, scalarCodes[scalar], arena),
    (outputs: CompiledTape) => new Tape(outputs.data as Vector, outputs.offsets, scalar, 3),
  );
}

function shapesSample(
  index: CompiledShapesIndex,
  executor: Executor | undefined,
  wanted: number,
  arena: Arena,
  config: DenseSampleConfig,
): DenseCall<DenseDraws> {
  const shown = config.shown === undefined ? undefined : normalizeKeys(config.shown);
  const places =
    config.shownCoordinates === undefined
      ? ([undefined, undefined, undefined, undefined, undefined] as const)
      : crossing(RowsView.of(config.shownCoordinates, 3));
  const focus = config.focus === undefined ? undefined : normalizeKeys(config.focus);
  return denseCall(
    (completion: Completion) =>
      index.sample(executor, completion, arena, wanted, shown, ...places, focus, config.seed, config.expansionFactor),
    (outputs: CompiledDraws) => new DenseDraws(outputs),
  );
}

function shapesReshard(
  executor: Executor | undefined,
  sources: ShapesIndex[],
  targets: ShapesIndex[],
  config: ShapesReshardConfig,
): DenseCall<DenseRefineReport> {
  return denseCall(
    (completion: Completion) =>
      compiled.CompiledShapesIndex.reshard(
        executor,
        completion,
        sources.map((source) => source.compiled),
        targets.map((target) => target.compiled),
        config.seed,
        config.expansionFactor,
        config.memoryLimit,
        config.stats,
        config.plan?.state,
      ),
    answerReport,
  );
}

function shapesReshardPlanFinish(
  executor: Executor | undefined,
  plan: ShapesReshardPlan,
): DenseCall<DenseRefineReport> {
  return denseCall(
    (completion: Completion) => compiled.CompiledShapesIndex.reshardPlanFinish(executor, completion, plan.state),
    answerReport,
  );
}

function shapesSave(index: CompiledShapesIndex, executor: Executor | undefined, path: string): DenseCall<void> {
  return denseCall((completion: Completion) => index.save(executor, completion, path), answerNothing);
}

function shapesSaveBuffer(
  index: CompiledShapesIndex,
  executor: Executor | undefined,
  arena: Arena,
): DenseCall<Monolith> {
  return denseCall(
    (completion: Completion) => index.saveBuffer(executor, completion, arena),
    (bytes: Uint8Array) => new Monolith(bytes),
  );
}

/** The verbs of one shapes index, blocking on one executor: `index.on(executor)`. */
export class ShapesIndexOn {
  readonly #index: CompiledShapesIndex;
  /** The executor every verb runs on. */
  readonly executor: Executor;

  /** Made by `index.on(executor)`, never directly. */
  constructor(index: CompiledShapesIndex, executor: Executor) {
    this.#index = index;
    this.executor = executor;
  }

  /** {@inheritDoc ShapesIndex.reserve} */
  reserve(entries: number, points?: number): void {
    return block(shapesReserve(this.#index, this.executor, entries, points));
  }

  /** {@inheritDoc ShapesIndex.clear} */
  clear(): void {
    return block(shapesClear(this.#index, this.executor));
  }

  /** {@inheritDoc ShapesIndex.describe} */
  describe(): ShapesIndexDescription {
    return block(shapesDescribe(this.#index, this.executor));
  }

  /** {@inheritDoc ShapesIndex.health} */
  health(): DenseReachability {
    return block(shapesHealth(this.#index, this.executor));
  }

  /** {@inheritDoc ShapesIndex.count} */
  count(key: KeyLike): ShapesKeyStats {
    return block(shapesCount(this.#index, this.executor, key));
  }

  /** {@inheritDoc ShapesIndex.contains} */
  contains(key: KeyLike): boolean {
    return this.count(key).entries !== 0;
  }

  /** {@inheritDoc ShapesIndex.insert} */
  insert(keys: KeysLike, shapes: ShapesLike, config: DenseInsertConfig = {}): void {
    return block(shapesInsert(this.#index, this.executor, keys, shapes, config));
  }

  /** {@inheritDoc ShapesIndex.refine} */
  refine(config: ShapesRefineConfig = {}): DenseRefineReport {
    return block(shapesRefine(this.#index, this.executor, config));
  }

  /** {@inheritDoc ShapesIndex.refinePlanFinish} */
  refinePlanFinish(plan: ShapesRefinePlan): DenseRefineReport {
    return block(shapesRefinePlanFinish(this.#index, this.executor, plan));
  }

  /** {@inheritDoc ShapesIndex.search} */
  search(queries: ShapesLike, wanted: number, arena: Arena, config: DenseSearchConfig = {}): Matches {
    return block(shapesSearch(this.#index, this.executor, queries, wanted, arena, config));
  }

  /** {@inheritDoc ShapesIndex.get} */
  get(key: KeyLike, arena: Arena, config: ShapesGetConfig = {}): Tape {
    return block(shapesGet(this.#index, this.executor, key, arena, config));
  }

  /** {@inheritDoc ShapesIndex.reshard} */
  reshard(sources: ShapesIndex[], targets: ShapesIndex[], config: ShapesReshardConfig = {}): DenseRefineReport {
    return block(shapesReshard(this.executor, sources, targets, config));
  }

  /** {@inheritDoc ShapesIndex.reshardPlanFinish} */
  reshardPlanFinish(plan: ShapesReshardPlan): DenseRefineReport {
    return block(shapesReshardPlanFinish(this.executor, plan));
  }

  /** {@inheritDoc ShapesIndex.sample} */
  sample(wanted: number, arena: Arena, config: DenseSampleConfig = {}): DenseDraws {
    return block(shapesSample(this.#index, this.executor, wanted, arena, config));
  }

  /** {@inheritDoc ShapesIndex.save} */
  save(path: string): void {
    return block(shapesSave(this.#index, this.executor, path));
  }

  /** {@inheritDoc ShapesIndex.saveBuffer} */
  saveBuffer(arena: Arena): Monolith {
    return block(shapesSaveBuffer(this.#index, this.executor, arena));
  }

  /** {@inheritDoc ShapesIndex.cursor} */
  cursor(starts?: KeysLike): DenseCursor {
    return new DenseCursor(this.#index.cursor(this.executor, starts === undefined ? undefined : normalizeKeys(starts)));
  }
}

/** The verbs of one shapes index, submitted to one executor and answered as {@link Deferred} calls:
 *  `index.deferred(executor)`. Cursors and `contains` stay blocking. */
export class ShapesIndexDeferredOn {
  readonly #index: CompiledShapesIndex;
  /** The executor every verb is submitted to. */
  readonly executor: Executor;

  /** Made by `index.deferred(executor)`, never directly. */
  constructor(index: CompiledShapesIndex, executor: Executor) {
    this.#index = index;
    this.executor = executor;
  }

  /** {@inheritDoc ShapesIndex.reserve} */
  reserve(entries: number, points?: number): Deferred<void> {
    return defer(shapesReserve(this.#index, this.executor, entries, points));
  }

  /** {@inheritDoc ShapesIndex.clear} */
  clear(): Deferred<void> {
    return defer(shapesClear(this.#index, this.executor));
  }

  /** {@inheritDoc ShapesIndex.describe} */
  describe(): Deferred<ShapesIndexDescription> {
    return defer(shapesDescribe(this.#index, this.executor));
  }

  /** {@inheritDoc ShapesIndex.health} */
  health(): Deferred<DenseReachability> {
    return defer(shapesHealth(this.#index, this.executor));
  }

  /** {@inheritDoc ShapesIndex.count} */
  count(key: KeyLike): Deferred<ShapesKeyStats> {
    return defer(shapesCount(this.#index, this.executor, key));
  }

  /** {@inheritDoc ShapesIndex.insert} */
  insert(keys: KeysLike, shapes: ShapesLike, config: DenseInsertConfig = {}): Deferred<void> {
    return defer(shapesInsert(this.#index, this.executor, keys, shapes, config));
  }

  /** {@inheritDoc ShapesIndex.refine} */
  refine(config: ShapesRefineConfig = {}): Deferred<DenseRefineReport> {
    return defer(shapesRefine(this.#index, this.executor, config));
  }

  /** {@inheritDoc ShapesIndex.refinePlanFinish} */
  refinePlanFinish(plan: ShapesRefinePlan): Deferred<DenseRefineReport> {
    return defer(shapesRefinePlanFinish(this.#index, this.executor, plan));
  }

  /** {@inheritDoc ShapesIndex.search} */
  search(queries: ShapesLike, wanted: number, arena: Arena, config: DenseSearchConfig = {}): Deferred<Matches> {
    return defer(shapesSearch(this.#index, this.executor, queries, wanted, arena, config));
  }

  /** {@inheritDoc ShapesIndex.get} */
  get(key: KeyLike, arena: Arena, config: ShapesGetConfig = {}): Deferred<Tape> {
    return defer(shapesGet(this.#index, this.executor, key, arena, config));
  }

  /** {@inheritDoc ShapesIndex.reshard} */
  reshard(
    sources: ShapesIndex[],
    targets: ShapesIndex[],
    config: ShapesReshardConfig = {},
  ): Deferred<DenseRefineReport> {
    return defer(shapesReshard(this.executor, sources, targets, config));
  }

  /** {@inheritDoc ShapesIndex.reshardPlanFinish} */
  reshardPlanFinish(plan: ShapesReshardPlan): Deferred<DenseRefineReport> {
    return defer(shapesReshardPlanFinish(this.executor, plan));
  }

  /** {@inheritDoc ShapesIndex.sample} */
  sample(wanted: number, arena: Arena, config: DenseSampleConfig = {}): Deferred<DenseDraws> {
    return defer(shapesSample(this.#index, this.executor, wanted, arena, config));
  }

  /** {@inheritDoc ShapesIndex.save} */
  save(path: string): Deferred<void> {
    return defer(shapesSave(this.#index, this.executor, path));
  }

  /** {@inheritDoc ShapesIndex.saveBuffer} */
  saveBuffer(arena: Arena): Deferred<Monolith> {
    return defer(shapesSaveBuffer(this.#index, this.executor, arena));
  }
}

/** The token the lifecycle verbs hold, which is what a direct `new ShapesIndex` cannot produce. */
const indexSeal = Symbol("ShapesIndex");

/** A shapes index over 3D point clouds, running every verb on the calling thread. `on(executor)`
 *  and `deferred(executor)` reach the same verbs on an executor. */
export class ShapesIndex implements Disposable {
  /** The native handle every verb names. @internal */
  readonly compiled: CompiledShapesIndex;

  /** Reached through the `init*`, `load*` and `view*` lifecycle verbs. */
  private constructor(seal: symbol, compiledIndex: CompiledShapesIndex) {
    if (seal !== indexSeal)
      throw fail(Status.InvalidArgument, "A ShapesIndex is opened by its init, load or view verbs.");
    this.compiled = compiledIndex;
  }

  /** The same verbs, blocking on `executor`. Binding-only. */
  on(executor: Executor): ShapesIndexOn {
    return new ShapesIndexOn(this.compiled, executor);
  }

  /** The same verbs, submitted to `executor`, each answered as a {@link Deferred}. Binding-only. */
  deferred(executor: Executor): ShapesIndexDeferredOn {
    return new ShapesIndexDeferredOn(this.compiled, executor);
  }

  /**
   *  Claims room ahead of insertion, a hint, since insertions grow the index on demand.
   *  @param entries - The desired total entry count, the current size included.
   *  @param points - The points those entries hold in all; absent derives.
   */
  reserve(entries: number, points?: number): void {
    return block(shapesReserve(this.compiled, undefined, entries, points));
  }

  /** Erases every entry, keeping the reserved capacity. */
  clear(): void {
    return block(shapesClear(this.compiled, undefined));
  }

  /** Reports the frozen metadata and the current counters in one call. */
  describe(): ShapesIndexDescription {
    return block(shapesDescribe(this.compiled, undefined));
  }

  /** Audits how much of the graph a search reaches from its root. */
  health(): DenseReachability {
    return block(shapesHealth(this.compiled, undefined));
  }

  /**
   *  Counts the entries stored under one key, with the points they hold.
   *  @param key - The identifier to be enumerated.
   */
  count(key: KeyLike): ShapesKeyStats {
    return block(shapesCount(this.compiled, undefined, key));
  }

  /** Whether the index holds any entry under one key. Binding-only sugar over `count`. */
  contains(key: KeyLike): boolean {
    return this.count(key).entries !== 0;
  }

  /**
   *  Adds shapes under their keys, one key per shape, each a cloud of interleaved `x, y, z` of any
   *  point count. A batch is all-or-nothing.
   *  @param keys - One identifier per shape.
   *  @param shapes - Clouds of one array type, or a packed {@link Tape}.
   *  @param config - Per-call breadth and telemetry sink.
   */
  insert(keys: KeysLike, shapes: ShapesLike, config: DenseInsertConfig = {}): void {
    return block(shapesInsert(this.compiled, undefined, keys, shapes, config));
  }

  /**
   *  Re-selects the neighbors of every stored entry against the finished graph; loop while the
   *  report's `verdict` is "improved".
   *  @param config - Per-call breadth, NVMe workspace, telemetry sink and plan.
   */
  refine(config: ShapesRefineConfig = {}): DenseRefineReport {
    return block(shapesRefine(this.compiled, undefined, config));
  }

  /** Closes a refine plan and frees it whatever the call answers, reporting the whole plan. */
  refinePlanFinish(plan: ShapesRefinePlan): DenseRefineReport {
    return block(shapesRefinePlanFinish(this.compiled, undefined, plan));
  }

  /**
   *  Finds the nearest stored shapes of every query cloud, landing them in `arena`.
   *  @param queries - Clouds of one array type, or a packed {@link Tape}.
   *  @param wanted - Matches per query.
   *  @param arena - The memory the matches land in, rewound by the call.
   *  @param config - Per-call breadth, traversal and telemetry sink.
   */
  search(queries: ShapesLike, wanted: number, arena: Arena, config: DenseSearchConfig = {}): Matches {
    return block(shapesSearch(this.compiled, undefined, queries, wanted, arena, config));
  }

  /**
   *  Retrieves every shape stored under one key, in insertion order and pose, as a tape inside
   *  `arena`.
   *  @param key - The identifier whose entries are exported.
   *  @param arena - The memory the tape lands in, rewound by the call.
   *  @param config - The kind to export points as.
   */
  get(key: KeyLike, arena: Arena, config: ShapesGetConfig = {}): Tape {
    return block(shapesGet(this.compiled, undefined, key, arena, config));
  }

  /**
   *  Moves the entries of `sources` into `targets`, keeping their links, then runs a round on every
   *  target that took entries: several sources gather into one target, and one source cuts into
   *  several. A refusal moves nothing.
   *  @param sources - The indexes whose entries move, each named once.
   *  @param targets - The indexes the entries move into, each a source itself or empty.
   *  @param config - The seed, breadth, NVMe workspace, telemetry sink and plan.
   */
  reshard(sources: ShapesIndex[], targets: ShapesIndex[], config: ShapesReshardConfig = {}): DenseRefineReport {
    return block(shapesReshard(undefined, sources, targets, config));
  }

  /** Closes a reshard plan and frees it whatever the call answers, reporting the whole plan. */
  reshardPlanFinish(plan: ShapesReshardPlan): DenseRefineReport {
    return block(shapesReshardPlanFinish(undefined, plan));
  }

  /** Draws keys around `config.focus`, or across the whole index, placed in three dimensions. */
  sample(wanted: number, arena: Arena, config: DenseSampleConfig = {}): DenseDraws {
    return block(shapesSample(this.compiled, undefined, wanted, arena, config));
  }

  /** Saves the index in the layout the path extension picks. */
  save(path: string): void {
    return block(shapesSave(this.compiled, undefined, path));
  }

  /** Writes the index into `arena` as one monolith. */
  saveBuffer(arena: Arena): Monolith {
    return block(shapesSaveBuffer(this.compiled, undefined, arena));
  }

  /** Opens a walk over the current contents; absent `starts` walks every entry as stored. */
  cursor(starts?: KeysLike): DenseCursor {
    return new DenseCursor(this.compiled.cursor(undefined, starts === undefined ? undefined : normalizeKeys(starts)));
  }

  /** Releases the handle now; a freed index refuses later calls, and a second `free` is a no-op. */
  free(): void {
    this.compiled.free();
  }

  [Symbol.dispose](): void {
    this.free();
  }

  /**
   *  Creates an empty, mutable index in host memory, one shard per named NUMA node.
   *  @param options - The geometry.
   *  @param memoryDomains - NUMA node IDs to shard over; absent spreads over every node.
   */
  static initRam(options: ShapesOptions = {}, memoryDomains?: number[]): ShapesIndex {
    return new ShapesIndex(
      indexSeal,
      compiled.CompiledShapesIndex.initRam(
        options.metric ?? ShapesMetric.Chamfer,
        scalarCodes[options.quantization ?? Scalar.F32],
        options.points,
        options.connectivity,
        options.valuesPerKey,
        memoryDomains,
        options.cpuClassCeiling,
      ),
    );
  }

  /**
   *  Creates an empty, mutable index in device memory, one shard per named CUDA ordinal.
   *  @param options - The geometry.
   *  @param devices - CUDA ordinals to shard over; absent takes one free device.
   */
  static initVram(options: ShapesOptions = {}, devices?: number[]): ShapesIndex {
    return new ShapesIndex(
      indexSeal,
      compiled.CompiledShapesIndex.initVram(
        options.metric ?? ShapesMetric.Chamfer,
        scalarCodes[options.quantization ?? Scalar.F32],
        options.points,
        options.connectivity,
        options.valuesPerKey,
        devices,
        options.gpuClassCeiling,
      ),
    );
  }

  /** Copies a checkpoint into a new, mutable index in host memory. */
  static loadRam(path: string, config: ShapesLoadConfig = {}): ShapesIndex {
    return new ShapesIndex(
      indexSeal,
      compiled.CompiledShapesIndex.loadRam(
        path,
        config.executor,
        config.placement,
        config.valuesPerKey,
        config.classCeiling,
      ),
    );
  }

  /** Copies a checkpoint into a new, mutable index in device memory. */
  static loadVram(path: string, config: ShapesLoadConfig = {}): ShapesIndex {
    return new ShapesIndex(
      indexSeal,
      compiled.CompiledShapesIndex.loadVram(
        path,
        config.executor,
        config.placement,
        config.valuesPerKey,
        config.classCeiling,
      ),
    );
  }

  /** Copies a monolith held in memory into a new, mutable index in host memory. */
  static loadRamBuffer(buffer: Uint8Array | Monolith, config: ShapesLoadConfig = {}): ShapesIndex {
    return new ShapesIndex(
      indexSeal,
      compiled.CompiledShapesIndex.loadRamBuffer(
        buffer instanceof Monolith ? buffer.bytes : buffer,
        config.executor,
        config.placement,
        config.valuesPerKey,
        config.classCeiling,
      ),
    );
  }

  /** Copies a monolith held in memory into a new, mutable index in device memory. */
  static loadVramBuffer(buffer: Uint8Array | Monolith, config: ShapesLoadConfig = {}): ShapesIndex {
    return new ShapesIndex(
      indexSeal,
      compiled.CompiledShapesIndex.loadVramBuffer(
        buffer instanceof Monolith ? buffer.bytes : buffer,
        config.executor,
        config.placement,
        config.valuesPerKey,
        config.classCeiling,
      ),
    );
  }

  /** Serves a checkpoint in place through a memory mapping, as a read-only index. */
  static viewRam(path: string, config: { cpuClassCeiling?: string } = {}): ShapesIndex {
    return new ShapesIndex(indexSeal, compiled.CompiledShapesIndex.viewRam(path, config.cpuClassCeiling));
  }

  /** Serves a checkpoint in place from NVMe drives, as a read-only index. */
  static viewNvme(path: string, config: ShapesViewNvmeConfig = {}): ShapesIndex {
    return new ShapesIndex(
      indexSeal,
      compiled.CompiledShapesIndex.viewNvme(path, config.executor, config.paths, config.cpuClassCeiling),
    );
  }

  /** Reads a checkpoint's geometry and metric without loading it. */
  static metadataRead(path: string, config: { executor?: Executor } = {}): ShapesMetadata {
    return metadataOf(compiled.CompiledShapesIndex.metadataRead(path, config.executor));
  }

  /** Reads the geometry and metric of a monolith held in memory. */
  static metadataReadBuffer(buffer: Uint8Array | Monolith, config: { executor?: Executor } = {}): ShapesMetadata {
    const bytes = buffer instanceof Monolith ? buffer.bytes : buffer;
    return metadataOf(compiled.CompiledShapesIndex.metadataReadBuffer(bytes, config.executor));
  }
}
