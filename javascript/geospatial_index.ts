/**
 *  @file javascript/geospatial_index.ts
 *  @author Ash Vardanian
 *  @date November 16, 2026
 *  @brief Search over observations in space and time under nearness, sight and view cones, and the
 *      receivers its verbs run through.
 */
import {
  Completion,
  CompiledPhaseProgress,
  CompiledGeospatialDescription,
  CompiledGeospatialIndex,
  CompiledGeospatialKeyStats,
  CompiledGeospatialMetadata,
  CompiledDraws,
  CompiledProducts,
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
  Status,
  crossing,
  fail,
  normalizeKey,
  normalizeKeys,
} from "./types";
import {
  DenseCall,
  DenseCursor,
  DenseDraws,
  DenseInsertConfig,
  DenseProducts,
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

/** Which entries count as matches of a geospatial query: the values of the C
 *  `usearch_geospatial_relation_t`, which the library names no spelling for.
 *  @enum {number}
 *  @readonly */
export enum GeospatialRelation {
  /** Within the radius, widened by three standard deviations of both positions combined. */
  Near = 1,
  /** Near, with a clear line of sight over the handle's terrain, which no handle here carries. */
  Visible = 2,
  /** Near, and the entry's view cone holds the query. */
  Sees = 3,
  /** Near, and the query's view cone holds the entry. */
  Seen = 4,
}

/** One observation as devices record it, every field in its natural unit; an absent one is NaN,
 *  which only ever widens the tests that read it. */
export interface GeospatialObservation {
  /** Degrees north on WGS-84, south negative. */
  latitudeDegrees: number;
  /** Degrees east on WGS-84, west negative. */
  longitudeDegrees: number;
  /** Metres above the WGS-84 ellipsoid. */
  altitudeMetres?: number;
  /** One standard deviation of the horizontal position, in metres. */
  accuracyMetres?: number;
  /** Degrees clockwise from true north the device faced. */
  headingDegrees?: number;
  /** Degrees above the horizon the device faced. */
  pitchDegrees?: number;
  /** One standard deviation of the heading, in degrees. */
  headingAccuracyDegrees?: number;
  /** The full field of view, in degrees. */
  fieldOfViewDegrees?: number;
  /** How far the device saw, in metres along the slant. */
  rangeMetres?: number;
  /** The first instant covered, in Unix seconds, UTC. */
  beginSeconds?: number;
  /** The last instant covered, in Unix seconds, UTC. */
  endSeconds?: number;
}

/** Fields one observation spans as it crosses, the doubles of the C struct in order. */
const observationFields = 11;

/**
 *  Packs observations into the `Float64Array` of eleven fields apiece every verb takes, NaN
 *  wherever a field is absent. Binding-only.
 *
 *  @param observations - The observations, in order.
 *  @returns One row of eleven doubles per observation.
 */
export function observationsOf(observations: ReadonlyArray<GeospatialObservation>): Float64Array {
  const batch = new Float64Array(observations.length * observationFields).fill(NaN);
  observations.forEach((observation, row) => {
    batch.set(
      [
        observation.latitudeDegrees,
        observation.longitudeDegrees,
        observation.altitudeMetres ?? NaN,
        observation.accuracyMetres ?? NaN,
        observation.headingDegrees ?? NaN,
        observation.pitchDegrees ?? NaN,
        observation.headingAccuracyDegrees ?? NaN,
        observation.fieldOfViewDegrees ?? NaN,
        observation.rangeMetres ?? NaN,
        observation.beginSeconds ?? NaN,
        observation.endSeconds ?? NaN,
      ],
      row * observationFields,
    );
  });
  return batch;
}

/** The graph every geospatial index freezes at initialization, whatever its residency. Handles
 *  carry no terrain, so the `Visible` relation is refused. */
export interface GeospatialOptions {
  /** Neighbors per node; absent derives it. */
  connectivity?: number;
  /** Entries one key may carry; absent is unbounded. */
  valuesPerKey?: number;
  /** The newest CPU class dispatch may pick, by canonical name; absent means no ceiling. */
  cpuClassCeiling?: string;
  /** The newest GPU class dispatch may pick, by canonical name; absent means no ceiling. */
  gpuClassCeiling?: string;
}

/** Optional settings of a ranked geospatial search. */
export interface GeospatialSearchConfig extends DenseSearchConfig {
  /** The radius around each query; absent or zero ranks by distance alone. */
  metres?: number;
  /** Slack on both ends of the query's time interval, in seconds. */
  seconds?: number;
}

/** Optional settings of a containment walk. */
export interface GeospatialWithinConfig extends Omit<DenseSearchConfig, "expansionFactor"> {
  /** Slack on both ends of the query's time interval, in seconds. */
  seconds?: number;
}

/** Optional settings of a load from a path or a buffer. */
export interface GeospatialLoadConfig {
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
export interface GeospatialViewNvmeConfig {
  /** The NVMe executor whose rings read the checkpoint. */
  executor?: Executor;
  /** The drives to serve the parts from; absent serves each where its manifest points. */
  paths?: string[];
  /** The newest CPU class dispatch may pick, by canonical name; absent means no ceiling. */
  cpuClassCeiling?: string;
}

/** The connectivity a checkpoint froze. */
export interface GeospatialMetadata {
  /** Neighbors per node the graph was built with. */
  connectivity: number;
}

/** A live index's frozen metadata and current counters, each read once. */
export type GeospatialIndexDescription = Omit<CompiledGeospatialDescription, "metadata"> & {
  metadata: GeospatialMetadata;
};

/** The entries one key holds. */
export type GeospatialKeyStats = CompiledGeospatialKeyStats;

/** Where a plan's last call stopped. */
export type GeospatialPhaseProgress = CompiledPhaseProgress;

/** One refine continued across calls; a receiver's `refinePlanFinish` closes it. While it lives the
 *  index admits no insertion and no save. */
export class GeospatialRefinePlan {
  /** The memory the library writes the plan into. */
  readonly state: ArrayBuffer = compiled.CompiledIndex.plan();

  /** Where the last call stopped, "none" before the first and "done" once its round ended. */
  get phase(): string {
    return this.progress.phase;
  }

  /** The phase the last call stopped in, with its position and its units. */
  get progress(): GeospatialPhaseProgress {
    return compiled.CompiledIndex.planProgress(this.state);
  }
}

/** One reshard continued across calls; a receiver's `reshardPlanFinish` closes it. Every index it
 *  names admits no insertion and no save while it lives. */
export class GeospatialReshardPlan {
  /** The memory the library writes the plan into. */
  readonly state: ArrayBuffer = compiled.CompiledIndex.plan();

  /** Where the last call stopped, "none" before the first and "done" once its rounds ended. */
  get phase(): string {
    return this.progress.phase;
  }

  /** The phase the last call stopped in, with its position and its units. */
  get progress(): GeospatialPhaseProgress {
    return compiled.CompiledIndex.planProgress(this.state);
  }
}

/** Optional per-call settings of a refinement round. */
export type GeospatialRefineConfig = Omit<DenseRefineConfig, "plan"> & {
  /** The plan this call continues; absent runs one round alone. */
  plan?: GeospatialRefinePlan;
};

/** Optional settings of a reshard; the basic modalities take no placement. */
export type GeospatialReshardConfig = Pick<DenseReshardConfig, "seed" | "expansionFactor" | "memoryLimit" | "stats"> & {
  /** The plan this call continues; absent moves and runs to the end. */
  plan?: GeospatialReshardPlan;
};

function metadataOf(read: CompiledGeospatialMetadata): GeospatialMetadata {
  return { connectivity: read.connectivity };
}

function geospatialReserve(
  index: CompiledGeospatialIndex,
  executor: Executor | undefined,
  entries: number,
): DenseCall<void> {
  return denseCall((completion: Completion) => index.reserve(executor, completion, entries), answerNothing);
}

function geospatialClear(index: CompiledGeospatialIndex, executor: Executor | undefined): DenseCall<void> {
  return denseCall((completion: Completion) => index.clear(executor, completion), answerNothing);
}

function geospatialDescribe(
  index: CompiledGeospatialIndex,
  executor: Executor | undefined,
): DenseCall<GeospatialIndexDescription> {
  return denseCall(
    (completion: Completion) => index.describe(executor, completion),
    (described: CompiledGeospatialDescription) => ({ ...described, metadata: metadataOf(described.metadata) }),
  );
}

function geospatialHealth(
  index: CompiledGeospatialIndex,
  executor: Executor | undefined,
): DenseCall<DenseReachability> {
  return denseCall(
    (completion: Completion) => index.health(executor, completion),
    (reachability: DenseReachability) => reachability,
  );
}

function geospatialCount(
  index: CompiledGeospatialIndex,
  executor: Executor | undefined,
  key: KeyLike,
): DenseCall<GeospatialKeyStats> {
  const normalized = normalizeKey(key);
  return denseCall(
    (completion: Completion) => index.count(executor, completion, normalized),
    (stats: GeospatialKeyStats) => stats,
  );
}

function geospatialInsert(
  index: CompiledGeospatialIndex,
  executor: Executor | undefined,
  keys: KeysLike,
  observations: Float64Array,
  config: DenseInsertConfig,
): DenseCall<void> {
  const normalized = normalizeKeys(keys);
  return denseCall(
    (completion: Completion) =>
      index.insert(executor, completion, normalized, observations, config.expansionFactor, config.stats),
    answerNothing,
  );
}

function geospatialRefine(
  index: CompiledGeospatialIndex,
  executor: Executor | undefined,
  config: GeospatialRefineConfig,
): DenseCall<DenseRefineReport> {
  return denseCall(
    (completion: Completion) =>
      index.refine(executor, completion, config.expansionFactor, config.memoryLimit, config.stats, config.plan?.state),
    answerReport,
  );
}

function geospatialRefinePlanFinish(
  index: CompiledGeospatialIndex,
  executor: Executor | undefined,
  plan: GeospatialRefinePlan,
): DenseCall<DenseRefineReport> {
  return denseCall((completion: Completion) => index.refinePlanFinish(executor, completion, plan.state), answerReport);
}

function geospatialSearch(
  index: CompiledGeospatialIndex,
  executor: Executor | undefined,
  queries: Float64Array,
  relation: GeospatialRelation,
  wanted: number,
  arena: Arena,
  config: GeospatialSearchConfig,
): DenseCall<Matches> {
  return denseCall(
    (completion: Completion) =>
      index.search(
        executor,
        completion,
        queries,
        relation,
        config.metres,
        config.seconds,
        arena,
        wanted,
        config.expansionFactor,
        config.traversal,
        config.stats,
      ),
    (outputs: CompiledMatches) => new Matches(outputs, queries.length / observationFields),
  );
}

function geospatialWithin(
  index: CompiledGeospatialIndex,
  executor: Executor | undefined,
  queries: Float64Array,
  relation: GeospatialRelation,
  metres: number,
  arena: Arena,
  config: GeospatialWithinConfig,
): DenseCall<DenseProducts> {
  return denseCall(
    (completion: Completion) =>
      index.within(
        executor,
        completion,
        queries,
        relation,
        metres,
        config.seconds,
        arena,
        config.traversal,
        config.stats,
      ),
    (outputs: CompiledProducts) => new DenseProducts(outputs),
  );
}

function geospatialGet(
  index: CompiledGeospatialIndex,
  executor: Executor | undefined,
  key: KeyLike,
  arena: Arena,
): DenseCall<Float64Array> {
  const normalized = normalizeKey(key);
  return denseCall(
    (completion: Completion) => index.get(executor, completion, normalized, arena),
    (observations: Float64Array) => observations,
  );
}

function geospatialSample(
  index: CompiledGeospatialIndex,
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

function geospatialReshard(
  executor: Executor | undefined,
  sources: GeospatialIndex[],
  targets: GeospatialIndex[],
  config: GeospatialReshardConfig,
): DenseCall<DenseRefineReport> {
  return denseCall(
    (completion: Completion) =>
      compiled.CompiledGeospatialIndex.reshard(
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

function geospatialReshardPlanFinish(
  executor: Executor | undefined,
  plan: GeospatialReshardPlan,
): DenseCall<DenseRefineReport> {
  return denseCall(
    (completion: Completion) => compiled.CompiledGeospatialIndex.reshardPlanFinish(executor, completion, plan.state),
    answerReport,
  );
}

function geospatialSave(index: CompiledGeospatialIndex, executor: Executor | undefined, path: string): DenseCall<void> {
  return denseCall((completion: Completion) => index.save(executor, completion, path), answerNothing);
}

function geospatialSaveBuffer(
  index: CompiledGeospatialIndex,
  executor: Executor | undefined,
  arena: Arena,
): DenseCall<Monolith> {
  return denseCall(
    (completion: Completion) => index.saveBuffer(executor, completion, arena),
    (bytes: Uint8Array) => new Monolith(bytes),
  );
}

/** The verbs of one geospatial index, blocking on one executor: `index.on(executor)`. */
export class GeospatialIndexOn {
  readonly #index: CompiledGeospatialIndex;
  /** The executor every verb runs on. */
  readonly executor: Executor;

  /** Made by `index.on(executor)`, never directly. */
  constructor(index: CompiledGeospatialIndex, executor: Executor) {
    this.#index = index;
    this.executor = executor;
  }

  /** {@inheritDoc GeospatialIndex.reserve} */
  reserve(entries: number): void {
    return block(geospatialReserve(this.#index, this.executor, entries));
  }

  /** {@inheritDoc GeospatialIndex.clear} */
  clear(): void {
    return block(geospatialClear(this.#index, this.executor));
  }

  /** {@inheritDoc GeospatialIndex.describe} */
  describe(): GeospatialIndexDescription {
    return block(geospatialDescribe(this.#index, this.executor));
  }

  /** {@inheritDoc GeospatialIndex.health} */
  health(): DenseReachability {
    return block(geospatialHealth(this.#index, this.executor));
  }

  /** {@inheritDoc GeospatialIndex.count} */
  count(key: KeyLike): GeospatialKeyStats {
    return block(geospatialCount(this.#index, this.executor, key));
  }

  /** {@inheritDoc GeospatialIndex.contains} */
  contains(key: KeyLike): boolean {
    return this.count(key).entries !== 0;
  }

  /** {@inheritDoc GeospatialIndex.insert} */
  insert(keys: KeysLike, observations: Float64Array, config: DenseInsertConfig = {}): void {
    return block(geospatialInsert(this.#index, this.executor, keys, observations, config));
  }

  /** {@inheritDoc GeospatialIndex.refine} */
  refine(config: GeospatialRefineConfig = {}): DenseRefineReport {
    return block(geospatialRefine(this.#index, this.executor, config));
  }

  /** {@inheritDoc GeospatialIndex.refinePlanFinish} */
  refinePlanFinish(plan: GeospatialRefinePlan): DenseRefineReport {
    return block(geospatialRefinePlanFinish(this.#index, this.executor, plan));
  }

  /** {@inheritDoc GeospatialIndex.search} */
  search(
    queries: Float64Array,
    relation: GeospatialRelation,
    wanted: number,
    arena: Arena,
    config: GeospatialSearchConfig = {},
  ): Matches {
    return block(geospatialSearch(this.#index, this.executor, queries, relation, wanted, arena, config));
  }

  /** {@inheritDoc GeospatialIndex.within} */
  within(
    queries: Float64Array,
    relation: GeospatialRelation,
    metres: number,
    arena: Arena,
    config: GeospatialWithinConfig = {},
  ): DenseProducts {
    return block(geospatialWithin(this.#index, this.executor, queries, relation, metres, arena, config));
  }

  /** {@inheritDoc GeospatialIndex.get} */
  get(key: KeyLike, arena: Arena): Float64Array {
    return block(geospatialGet(this.#index, this.executor, key, arena));
  }

  /** {@inheritDoc GeospatialIndex.reshard} */
  reshard(
    sources: GeospatialIndex[],
    targets: GeospatialIndex[],
    config: GeospatialReshardConfig = {},
  ): DenseRefineReport {
    return block(geospatialReshard(this.executor, sources, targets, config));
  }

  /** {@inheritDoc GeospatialIndex.reshardPlanFinish} */
  reshardPlanFinish(plan: GeospatialReshardPlan): DenseRefineReport {
    return block(geospatialReshardPlanFinish(this.executor, plan));
  }

  /** {@inheritDoc GeospatialIndex.sample} */
  sample(wanted: number, arena: Arena, config: DenseSampleConfig = {}): DenseDraws {
    return block(geospatialSample(this.#index, this.executor, wanted, arena, config));
  }

  /** {@inheritDoc GeospatialIndex.save} */
  save(path: string): void {
    return block(geospatialSave(this.#index, this.executor, path));
  }

  /** {@inheritDoc GeospatialIndex.saveBuffer} */
  saveBuffer(arena: Arena): Monolith {
    return block(geospatialSaveBuffer(this.#index, this.executor, arena));
  }

  /** {@inheritDoc GeospatialIndex.cursor} */
  cursor(starts?: KeysLike): DenseCursor {
    return new DenseCursor(this.#index.cursor(this.executor, starts === undefined ? undefined : normalizeKeys(starts)));
  }
}

/** The verbs of one geospatial index, submitted to one executor and answered as {@link Deferred}
 *  calls: `index.deferred(executor)`. Cursors and `contains` stay blocking. */
export class GeospatialIndexDeferredOn {
  readonly #index: CompiledGeospatialIndex;
  /** The executor every verb is submitted to. */
  readonly executor: Executor;

  /** Made by `index.deferred(executor)`, never directly. */
  constructor(index: CompiledGeospatialIndex, executor: Executor) {
    this.#index = index;
    this.executor = executor;
  }

  /** {@inheritDoc GeospatialIndex.reserve} */
  reserve(entries: number): Deferred<void> {
    return defer(geospatialReserve(this.#index, this.executor, entries));
  }

  /** {@inheritDoc GeospatialIndex.clear} */
  clear(): Deferred<void> {
    return defer(geospatialClear(this.#index, this.executor));
  }

  /** {@inheritDoc GeospatialIndex.describe} */
  describe(): Deferred<GeospatialIndexDescription> {
    return defer(geospatialDescribe(this.#index, this.executor));
  }

  /** {@inheritDoc GeospatialIndex.health} */
  health(): Deferred<DenseReachability> {
    return defer(geospatialHealth(this.#index, this.executor));
  }

  /** {@inheritDoc GeospatialIndex.count} */
  count(key: KeyLike): Deferred<GeospatialKeyStats> {
    return defer(geospatialCount(this.#index, this.executor, key));
  }

  /** {@inheritDoc GeospatialIndex.insert} */
  insert(keys: KeysLike, observations: Float64Array, config: DenseInsertConfig = {}): Deferred<void> {
    return defer(geospatialInsert(this.#index, this.executor, keys, observations, config));
  }

  /** {@inheritDoc GeospatialIndex.refine} */
  refine(config: GeospatialRefineConfig = {}): Deferred<DenseRefineReport> {
    return defer(geospatialRefine(this.#index, this.executor, config));
  }

  /** {@inheritDoc GeospatialIndex.refinePlanFinish} */
  refinePlanFinish(plan: GeospatialRefinePlan): Deferred<DenseRefineReport> {
    return defer(geospatialRefinePlanFinish(this.#index, this.executor, plan));
  }

  /** {@inheritDoc GeospatialIndex.search} */
  search(
    queries: Float64Array,
    relation: GeospatialRelation,
    wanted: number,
    arena: Arena,
    config: GeospatialSearchConfig = {},
  ): Deferred<Matches> {
    return defer(geospatialSearch(this.#index, this.executor, queries, relation, wanted, arena, config));
  }

  /** {@inheritDoc GeospatialIndex.within} */
  within(
    queries: Float64Array,
    relation: GeospatialRelation,
    metres: number,
    arena: Arena,
    config: GeospatialWithinConfig = {},
  ): Deferred<DenseProducts> {
    return defer(geospatialWithin(this.#index, this.executor, queries, relation, metres, arena, config));
  }

  /** {@inheritDoc GeospatialIndex.get} */
  get(key: KeyLike, arena: Arena): Deferred<Float64Array> {
    return defer(geospatialGet(this.#index, this.executor, key, arena));
  }

  /** {@inheritDoc GeospatialIndex.reshard} */
  reshard(
    sources: GeospatialIndex[],
    targets: GeospatialIndex[],
    config: GeospatialReshardConfig = {},
  ): Deferred<DenseRefineReport> {
    return defer(geospatialReshard(this.executor, sources, targets, config));
  }

  /** {@inheritDoc GeospatialIndex.reshardPlanFinish} */
  reshardPlanFinish(plan: GeospatialReshardPlan): Deferred<DenseRefineReport> {
    return defer(geospatialReshardPlanFinish(this.executor, plan));
  }

  /** {@inheritDoc GeospatialIndex.sample} */
  sample(wanted: number, arena: Arena, config: DenseSampleConfig = {}): Deferred<DenseDraws> {
    return defer(geospatialSample(this.#index, this.executor, wanted, arena, config));
  }

  /** {@inheritDoc GeospatialIndex.save} */
  save(path: string): Deferred<void> {
    return defer(geospatialSave(this.#index, this.executor, path));
  }

  /** {@inheritDoc GeospatialIndex.saveBuffer} */
  saveBuffer(arena: Arena): Deferred<Monolith> {
    return defer(geospatialSaveBuffer(this.#index, this.executor, arena));
  }
}

/** The token the lifecycle verbs hold, which a direct `new GeospatialIndex` cannot produce. */
const indexSeal = Symbol("GeospatialIndex");

/** A geospatial index over observations in space and time, running every verb on the calling
 *  thread. `on(executor)` and `deferred(executor)` reach the same verbs on an executor. */
export class GeospatialIndex implements Disposable {
  /** The native handle every verb names. @internal */
  readonly compiled: CompiledGeospatialIndex;

  /** Reached through the `init*`, `load*` and `view*` lifecycle verbs. */
  private constructor(seal: symbol, compiledIndex: CompiledGeospatialIndex) {
    if (seal !== indexSeal)
      throw fail(Status.InvalidArgument, "A GeospatialIndex is opened by its init, load or view verbs.");
    this.compiled = compiledIndex;
  }

  /** The same verbs, blocking on `executor`. Binding-only. */
  on(executor: Executor): GeospatialIndexOn {
    return new GeospatialIndexOn(this.compiled, executor);
  }

  /** The same verbs, submitted to `executor`, each answered as a {@link Deferred}. Binding-only. */
  deferred(executor: Executor): GeospatialIndexDeferredOn {
    return new GeospatialIndexDeferredOn(this.compiled, executor);
  }

  /**
   *  Claims room ahead of insertion, a hint, since insertions grow the index on demand.
   *  @param entries - The desired total entry count, the current size included.
   */
  reserve(entries: number): void {
    return block(geospatialReserve(this.compiled, undefined, entries));
  }

  /** Erases every entry, keeping the reserved capacity. */
  clear(): void {
    return block(geospatialClear(this.compiled, undefined));
  }

  /** Reports the frozen metadata and the current counters in one call. */
  describe(): GeospatialIndexDescription {
    return block(geospatialDescribe(this.compiled, undefined));
  }

  /** Audits how much of the graph a search reaches from its root. */
  health(): DenseReachability {
    return block(geospatialHealth(this.compiled, undefined));
  }

  /**
   *  Counts the entries stored under one key.
   *  @param key - The identifier to be enumerated.
   */
  count(key: KeyLike): GeospatialKeyStats {
    return block(geospatialCount(this.compiled, undefined, key));
  }

  /** Whether the index holds any entry under one key. Binding-only sugar over `count`. */
  contains(key: KeyLike): boolean {
    return this.count(key).entries !== 0;
  }

  /**
   *  Adds observations under their keys, one key per observation. A batch is all-or-nothing.
   *  @param keys - One identifier per observation.
   *  @param observations - Eleven doubles per observation, as {@link observationsOf} packs them.
   *  @param config - Per-call breadth and telemetry sink.
   */
  insert(keys: KeysLike, observations: Float64Array, config: DenseInsertConfig = {}): void {
    return block(geospatialInsert(this.compiled, undefined, keys, observations, config));
  }

  /**
   *  Re-selects the neighbors of every stored entry against the finished graph; loop while the
   *  report's `verdict` is "improved".
   *  @param config - Per-call breadth, NVMe workspace, telemetry sink and plan.
   */
  refine(config: GeospatialRefineConfig = {}): DenseRefineReport {
    return block(geospatialRefine(this.compiled, undefined, config));
  }

  /** Closes a refine plan and frees it whatever the call answers, reporting the whole plan. */
  refinePlanFinish(plan: GeospatialRefinePlan): DenseRefineReport {
    return block(geospatialRefinePlanFinish(this.compiled, undefined, plan));
  }

  /**
   *  Finds the nearest entries of each query observation that `relation` admits, scored in chord
   *  metres, landing them in `arena`; a selective relation may leave a query fewer than `wanted`.
   *  @param queries - Eleven doubles per observation, as {@link observationsOf} packs them.
   *  @param relation - Which entries count as matches.
   *  @param wanted - Matches per query.
   *  @param arena - The memory the matches land in, rewound by the call.
   *  @param config - The radius, time slack, breadth, traversal and telemetry sink.
   */
  search(
    queries: Float64Array,
    relation: GeospatialRelation,
    wanted: number,
    arena: Arena,
    config: GeospatialSearchConfig = {},
  ): Matches {
    return block(geospatialSearch(this.compiled, undefined, queries, relation, wanted, arena, config));
  }

  /**
   *  Reports every entry within `metres` of each query observation that `relation` admits, with its
   *  chord metres, as one ragged run inside `arena`.
   *  @param queries - Eleven doubles per observation, as {@link observationsOf} packs them.
   *  @param relation - Which entries count as matches.
   *  @param metres - The radius around each query, which containment needs.
   *  @param arena - The memory the matches land in, rewound by the call.
   *  @param config - The time slack, traversal and telemetry sink.
   */
  within(
    queries: Float64Array,
    relation: GeospatialRelation,
    metres: number,
    arena: Arena,
    config: GeospatialWithinConfig = {},
  ): DenseProducts {
    return block(geospatialWithin(this.compiled, undefined, queries, relation, metres, arena, config));
  }

  /**
   *  Retrieves every observation stored under one key, in insertion order, within the rounding the
   *  index stores them at, as eleven doubles apiece inside `arena`.
   *  @param key - The identifier whose entries are exported.
   *  @param arena - The memory the observations land in, rewound by the call.
   */
  get(key: KeyLike, arena: Arena): Float64Array {
    return block(geospatialGet(this.compiled, undefined, key, arena));
  }

  /**
   *  Moves the entries of `sources` into `targets`, keeping their links, then runs a round on every
   *  target that took entries: several sources gather into one target, and one source cuts into
   *  several. A refusal moves nothing.
   *  @param sources - The indexes whose entries move, each named once.
   *  @param targets - The indexes the entries move into, each a source itself or empty.
   *  @param config - The seed, breadth, NVMe workspace, telemetry sink and plan.
   */
  reshard(
    sources: GeospatialIndex[],
    targets: GeospatialIndex[],
    config: GeospatialReshardConfig = {},
  ): DenseRefineReport {
    return block(geospatialReshard(undefined, sources, targets, config));
  }

  /** Closes a reshard plan and frees it whatever the call answers, reporting the whole plan. */
  reshardPlanFinish(plan: GeospatialReshardPlan): DenseRefineReport {
    return block(geospatialReshardPlanFinish(undefined, plan));
  }

  /** Draws keys around `config.focus`, or across the whole index, placed in three dimensions. */
  sample(wanted: number, arena: Arena, config: DenseSampleConfig = {}): DenseDraws {
    return block(geospatialSample(this.compiled, undefined, wanted, arena, config));
  }

  /** Saves the index in the layout the path extension picks. */
  save(path: string): void {
    return block(geospatialSave(this.compiled, undefined, path));
  }

  /** Writes the index into `arena` as one monolith. */
  saveBuffer(arena: Arena): Monolith {
    return block(geospatialSaveBuffer(this.compiled, undefined, arena));
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
   *  @param options - The graph.
   *  @param memoryDomains - NUMA node IDs to shard over; absent spreads over every node.
   */
  static initRam(options: GeospatialOptions = {}, memoryDomains?: number[]): GeospatialIndex {
    return new GeospatialIndex(
      indexSeal,
      compiled.CompiledGeospatialIndex.initRam(
        options.connectivity,
        options.valuesPerKey,
        memoryDomains,
        options.cpuClassCeiling,
      ),
    );
  }

  /**
   *  Creates an empty, mutable index in device memory, one shard per named CUDA ordinal.
   *  @param options - The graph.
   *  @param devices - CUDA ordinals to shard over; absent takes one free device.
   */
  static initVram(options: GeospatialOptions = {}, devices?: number[]): GeospatialIndex {
    return new GeospatialIndex(
      indexSeal,
      compiled.CompiledGeospatialIndex.initVram(
        options.connectivity,
        options.valuesPerKey,
        devices,
        options.gpuClassCeiling,
      ),
    );
  }

  /** Copies a checkpoint into a new, mutable index in host memory. */
  static loadRam(path: string, config: GeospatialLoadConfig = {}): GeospatialIndex {
    return new GeospatialIndex(
      indexSeal,
      compiled.CompiledGeospatialIndex.loadRam(
        path,
        config.executor,
        config.placement,
        config.valuesPerKey,
        config.classCeiling,
      ),
    );
  }

  /** Copies a checkpoint into a new, mutable index in device memory. */
  static loadVram(path: string, config: GeospatialLoadConfig = {}): GeospatialIndex {
    return new GeospatialIndex(
      indexSeal,
      compiled.CompiledGeospatialIndex.loadVram(
        path,
        config.executor,
        config.placement,
        config.valuesPerKey,
        config.classCeiling,
      ),
    );
  }

  /** Copies a monolith held in memory into a new, mutable index in host memory. */
  static loadRamBuffer(buffer: Uint8Array | Monolith, config: GeospatialLoadConfig = {}): GeospatialIndex {
    return new GeospatialIndex(
      indexSeal,
      compiled.CompiledGeospatialIndex.loadRamBuffer(
        buffer instanceof Monolith ? buffer.bytes : buffer,
        config.executor,
        config.placement,
        config.valuesPerKey,
        config.classCeiling,
      ),
    );
  }

  /** Copies a monolith held in memory into a new, mutable index in device memory. */
  static loadVramBuffer(buffer: Uint8Array | Monolith, config: GeospatialLoadConfig = {}): GeospatialIndex {
    return new GeospatialIndex(
      indexSeal,
      compiled.CompiledGeospatialIndex.loadVramBuffer(
        buffer instanceof Monolith ? buffer.bytes : buffer,
        config.executor,
        config.placement,
        config.valuesPerKey,
        config.classCeiling,
      ),
    );
  }

  /** Serves a checkpoint in place through a memory mapping, as a read-only index. */
  static viewRam(path: string, config: { cpuClassCeiling?: string } = {}): GeospatialIndex {
    return new GeospatialIndex(indexSeal, compiled.CompiledGeospatialIndex.viewRam(path, config.cpuClassCeiling));
  }

  /** Serves a checkpoint in place from NVMe drives, as a read-only index. */
  static viewNvme(path: string, config: GeospatialViewNvmeConfig = {}): GeospatialIndex {
    return new GeospatialIndex(
      indexSeal,
      compiled.CompiledGeospatialIndex.viewNvme(path, config.executor, config.paths, config.cpuClassCeiling),
    );
  }

  /** Reads a checkpoint's connectivity without loading it. */
  static metadataRead(path: string, config: { executor?: Executor } = {}): GeospatialMetadata {
    return metadataOf(compiled.CompiledGeospatialIndex.metadataRead(path, config.executor));
  }

  /** Reads the connectivity of a monolith held in memory. */
  static metadataReadBuffer(buffer: Uint8Array | Monolith, config: { executor?: Executor } = {}): GeospatialMetadata {
    const bytes = buffer instanceof Monolith ? buffer.bytes : buffer;
    return metadataOf(compiled.CompiledGeospatialIndex.metadataReadBuffer(bytes, config.executor));
  }
}
