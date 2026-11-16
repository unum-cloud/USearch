/**
 *  @file javascript/sequence_index.ts
 *  @author Ash Vardanian
 *  @date November 16, 2026
 *  @brief Similarity search over byte sequences under edit distances and alignments, and the
 *      receivers its verbs run through.
 */
import {
  Completion,
  CompiledPhaseProgress,
  CompiledSequenceDescription,
  CompiledSequenceIndex,
  CompiledSequenceKeyStats,
  CompiledSequenceMetadata,
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
  crossing,
  fail,
  normalizeKey,
  normalizeKeys,
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

/** How a sequence index scores two sequences, spelled as the library names each kind.
 *  @enum {string}
 *  @readonly */
export enum SequenceMetric {
  /** Single-byte insertions, deletions and substitutions, every edit costing one. */
  Levenshtein = "levenshtein",
  /** The same edits over decoded UTF-8 code points. */
  LevenshteinUtf8 = "levenshtein-utf8",
  /** Global alignment under a substitution matrix and a flat gap cost. */
  NeedlemanWunsch = "needleman-wunsch",
  /** The share of the larger window set the other side misses. */
  Overlap = "overlap",
}

/** The scoring every sequence index freezes at initialization, whatever its residency. */
export interface SequenceOptions {
  /** How sequences are scored; defaults to `levenshtein`. */
  metric?: SequenceMetric;
  /** Row-major 256 × 256 signed scores over bytes, which Needleman-Wunsch alone takes; borrowed
   *  for the call. */
  substitutionMatrix?: Int8Array;
  /** Cost of one unaligned symbol, beside the substitution matrix. */
  gap?: number;
  /** Window widths in bytes, which the overlap kind alone takes. */
  windowWidths?: number[];
  /** Neighbors per node; absent derives it. */
  connectivity?: number;
  /** Entries one key may carry; absent is unbounded. */
  valuesPerKey?: number;
  /** The newest CPU class dispatch may pick, by canonical name; absent means no ceiling. */
  cpuClassCeiling?: string;
  /** The newest GPU class dispatch may pick, by canonical name; absent means no ceiling. */
  gpuClassCeiling?: string;
}

/** The weights a loaded or viewed Needleman-Wunsch checkpoint scores with, never recorded in it. */
export interface SequenceWeights {
  /** Row-major 256 × 256 signed scores over bytes. */
  substitutionMatrix?: Int8Array;
  /** Cost of one unaligned symbol. */
  gap?: number;
}

/** Optional settings of a load from a path or a buffer. */
export interface SequenceLoadConfig extends SequenceWeights {
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
export interface SequenceViewNvmeConfig extends SequenceWeights {
  /** The NVMe executor whose rings read the checkpoint. */
  executor?: Executor;
  /** The drives to serve the parts from; absent serves each where its manifest points. */
  paths?: string[];
  /** The newest CPU class dispatch may pick, by canonical name; absent means no ceiling. */
  cpuClassCeiling?: string;
}

/** The metric and connectivity a checkpoint froze. */
export interface SequenceMetadata {
  /** How the checkpoint scores sequences. */
  metric: SequenceMetric;
  /** Neighbors per node the graph was built with. */
  connectivity: number;
}

/** A live index's frozen metadata and current counters, each read once. */
export type SequenceIndexDescription = Omit<CompiledSequenceDescription, "metadata"> & {
  metadata: SequenceMetadata;
};

/** The entries one key holds and the bytes they span. */
export type SequenceKeyStats = CompiledSequenceKeyStats;

/** Where a plan's last call stopped. */
export type SequencePhaseProgress = CompiledPhaseProgress;

/** One refine continued across calls; a receiver's `refinePlanFinish` closes it. While it lives the
 *  index admits no insertion and no save. */
export class SequenceRefinePlan {
  /** The memory the library writes the plan into. */
  readonly state: ArrayBuffer = compiled.CompiledIndex.plan();

  /** Where the last call stopped, "none" before the first and "done" once its round ended. */
  get phase(): string {
    return this.progress.phase;
  }

  /** The phase the last call stopped in, with its position and its units. */
  get progress(): SequencePhaseProgress {
    return compiled.CompiledIndex.planProgress(this.state);
  }
}

/** One reshard continued across calls; a receiver's `reshardPlanFinish` closes it. Every index it
 *  names admits no insertion and no save while it lives. */
export class SequenceReshardPlan {
  /** The memory the library writes the plan into. */
  readonly state: ArrayBuffer = compiled.CompiledIndex.plan();

  /** Where the last call stopped, "none" before the first and "done" once its rounds ended. */
  get phase(): string {
    return this.progress.phase;
  }

  /** The phase the last call stopped in, with its position and its units. */
  get progress(): SequencePhaseProgress {
    return compiled.CompiledIndex.planProgress(this.state);
  }
}

/** Optional per-call settings of a refinement round. */
export type SequenceRefineConfig = Omit<DenseRefineConfig, "plan"> & {
  /** The plan this call continues; absent runs one round alone. */
  plan?: SequenceRefinePlan;
};

/** Optional settings of a reshard; the basic modalities take no placement. */
export type SequenceReshardConfig = Pick<DenseReshardConfig, "seed" | "expansionFactor" | "memoryLimit" | "stats"> & {
  /** The plan this call continues; absent moves and runs to the end. */
  plan?: SequenceReshardPlan;
};

/** Sequences as a verb takes them: text, bytes, or a tape already packed. */
export type SequencesLike = Tape | ReadonlyArray<string | Uint8Array>;

function sequencesOf(sequences: SequencesLike): Tape {
  return sequences instanceof Tape ? sequences : Tape.ofSequences(sequences);
}

function weightsOf(weights: SequenceWeights): [Int8Array | undefined, number | undefined] {
  return [weights.substitutionMatrix, weights.gap];
}

function metadataOf(read: CompiledSequenceMetadata): SequenceMetadata {
  return { metric: read.metric as SequenceMetric, connectivity: read.connectivity };
}

const answerTape = (outputs: CompiledTape): Tape => new Tape(outputs.data as Uint8Array, outputs.offsets, Scalar.U8);

function sequenceReserve(
  index: CompiledSequenceIndex,
  executor: Executor | undefined,
  entries: number,
  payloadBytes?: number,
): DenseCall<void> {
  return denseCall(
    (completion: Completion) => index.reserve(executor, completion, entries, payloadBytes),
    answerNothing,
  );
}

function sequenceClear(index: CompiledSequenceIndex, executor: Executor | undefined): DenseCall<void> {
  return denseCall((completion: Completion) => index.clear(executor, completion), answerNothing);
}

function sequenceDescribe(
  index: CompiledSequenceIndex,
  executor: Executor | undefined,
): DenseCall<SequenceIndexDescription> {
  return denseCall(
    (completion: Completion) => index.describe(executor, completion),
    (described: CompiledSequenceDescription) => ({ ...described, metadata: metadataOf(described.metadata) }),
  );
}

function sequenceHealth(index: CompiledSequenceIndex, executor: Executor | undefined): DenseCall<DenseReachability> {
  return denseCall(
    (completion: Completion) => index.health(executor, completion),
    (reachability: DenseReachability) => reachability,
  );
}

function sequenceCount(
  index: CompiledSequenceIndex,
  executor: Executor | undefined,
  key: KeyLike,
): DenseCall<SequenceKeyStats> {
  const normalized = normalizeKey(key);
  return denseCall(
    (completion: Completion) => index.count(executor, completion, normalized),
    (stats: SequenceKeyStats) => stats,
  );
}

function sequenceInsert(
  index: CompiledSequenceIndex,
  executor: Executor | undefined,
  keys: KeysLike,
  sequences: SequencesLike,
  config: DenseInsertConfig,
): DenseCall<void> {
  const normalized = normalizeKeys(keys);
  const tape = tapeCrossing(sequencesOf(sequences));
  return denseCall(
    (completion: Completion) =>
      index.insert(executor, completion, normalized, ...tape, config.expansionFactor, config.stats),
    answerNothing,
  );
}

function sequenceRefine(
  index: CompiledSequenceIndex,
  executor: Executor | undefined,
  config: SequenceRefineConfig,
): DenseCall<DenseRefineReport> {
  return denseCall(
    (completion: Completion) =>
      index.refine(executor, completion, config.expansionFactor, config.memoryLimit, config.stats, config.plan?.state),
    answerReport,
  );
}

function sequenceRefinePlanFinish(
  index: CompiledSequenceIndex,
  executor: Executor | undefined,
  plan: SequenceRefinePlan,
): DenseCall<DenseRefineReport> {
  return denseCall((completion: Completion) => index.refinePlanFinish(executor, completion, plan.state), answerReport);
}

function sequenceSearch(
  index: CompiledSequenceIndex,
  executor: Executor | undefined,
  queries: SequencesLike,
  wanted: number,
  arena: Arena,
  config: DenseSearchConfig,
): DenseCall<Matches> {
  const tape = sequencesOf(queries);
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

function sequenceGet(
  index: CompiledSequenceIndex,
  executor: Executor | undefined,
  key: KeyLike,
  arena: Arena,
): DenseCall<Tape> {
  const normalized = normalizeKey(key);
  return denseCall((completion: Completion) => index.get(executor, completion, normalized, arena), answerTape);
}

function sequenceSample(
  index: CompiledSequenceIndex,
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

function sequenceReshard(
  executor: Executor | undefined,
  sources: SequenceIndex[],
  targets: SequenceIndex[],
  config: SequenceReshardConfig,
): DenseCall<DenseRefineReport> {
  return denseCall(
    (completion: Completion) =>
      compiled.CompiledSequenceIndex.reshard(
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

function sequenceReshardPlanFinish(
  executor: Executor | undefined,
  plan: SequenceReshardPlan,
): DenseCall<DenseRefineReport> {
  return denseCall(
    (completion: Completion) => compiled.CompiledSequenceIndex.reshardPlanFinish(executor, completion, plan.state),
    answerReport,
  );
}

function sequenceSave(index: CompiledSequenceIndex, executor: Executor | undefined, path: string): DenseCall<void> {
  return denseCall((completion: Completion) => index.save(executor, completion, path), answerNothing);
}

function sequenceSaveBuffer(
  index: CompiledSequenceIndex,
  executor: Executor | undefined,
  arena: Arena,
): DenseCall<Monolith> {
  return denseCall(
    (completion: Completion) => index.saveBuffer(executor, completion, arena),
    (bytes: Uint8Array) => new Monolith(bytes),
  );
}

/** The verbs of one sequence index, blocking on one executor: `index.on(executor)`. */
export class SequenceIndexOn {
  readonly #index: CompiledSequenceIndex;
  /** The executor every verb runs on. */
  readonly executor: Executor;

  /** Made by `index.on(executor)`, never directly. */
  constructor(index: CompiledSequenceIndex, executor: Executor) {
    this.#index = index;
    this.executor = executor;
  }

  /** {@inheritDoc SequenceIndex.reserve} */
  reserve(entries: number, payloadBytes?: number): void {
    return block(sequenceReserve(this.#index, this.executor, entries, payloadBytes));
  }

  /** {@inheritDoc SequenceIndex.clear} */
  clear(): void {
    return block(sequenceClear(this.#index, this.executor));
  }

  /** {@inheritDoc SequenceIndex.describe} */
  describe(): SequenceIndexDescription {
    return block(sequenceDescribe(this.#index, this.executor));
  }

  /** {@inheritDoc SequenceIndex.health} */
  health(): DenseReachability {
    return block(sequenceHealth(this.#index, this.executor));
  }

  /** {@inheritDoc SequenceIndex.count} */
  count(key: KeyLike): SequenceKeyStats {
    return block(sequenceCount(this.#index, this.executor, key));
  }

  /** {@inheritDoc SequenceIndex.contains} */
  contains(key: KeyLike): boolean {
    return this.count(key).entries !== 0;
  }

  /** {@inheritDoc SequenceIndex.insert} */
  insert(keys: KeysLike, sequences: SequencesLike, config: DenseInsertConfig = {}): void {
    return block(sequenceInsert(this.#index, this.executor, keys, sequences, config));
  }

  /** {@inheritDoc SequenceIndex.refine} */
  refine(config: SequenceRefineConfig = {}): DenseRefineReport {
    return block(sequenceRefine(this.#index, this.executor, config));
  }

  /** {@inheritDoc SequenceIndex.refinePlanFinish} */
  refinePlanFinish(plan: SequenceRefinePlan): DenseRefineReport {
    return block(sequenceRefinePlanFinish(this.#index, this.executor, plan));
  }

  /** {@inheritDoc SequenceIndex.search} */
  search(queries: SequencesLike, wanted: number, arena: Arena, config: DenseSearchConfig = {}): Matches {
    return block(sequenceSearch(this.#index, this.executor, queries, wanted, arena, config));
  }

  /** {@inheritDoc SequenceIndex.get} */
  get(key: KeyLike, arena: Arena): Tape {
    return block(sequenceGet(this.#index, this.executor, key, arena));
  }

  /** {@inheritDoc SequenceIndex.reshard} */
  reshard(sources: SequenceIndex[], targets: SequenceIndex[], config: SequenceReshardConfig = {}): DenseRefineReport {
    return block(sequenceReshard(this.executor, sources, targets, config));
  }

  /** {@inheritDoc SequenceIndex.reshardPlanFinish} */
  reshardPlanFinish(plan: SequenceReshardPlan): DenseRefineReport {
    return block(sequenceReshardPlanFinish(this.executor, plan));
  }

  /** {@inheritDoc SequenceIndex.sample} */
  sample(wanted: number, arena: Arena, config: DenseSampleConfig = {}): DenseDraws {
    return block(sequenceSample(this.#index, this.executor, wanted, arena, config));
  }

  /** {@inheritDoc SequenceIndex.save} */
  save(path: string): void {
    return block(sequenceSave(this.#index, this.executor, path));
  }

  /** {@inheritDoc SequenceIndex.saveBuffer} */
  saveBuffer(arena: Arena): Monolith {
    return block(sequenceSaveBuffer(this.#index, this.executor, arena));
  }

  /** {@inheritDoc SequenceIndex.cursor} */
  cursor(starts?: KeysLike): DenseCursor {
    return new DenseCursor(this.#index.cursor(this.executor, starts === undefined ? undefined : normalizeKeys(starts)));
  }
}

/** The verbs of one sequence index, submitted to one executor and answered as {@link Deferred}
 *  calls: `index.deferred(executor)`. Cursors and `contains` stay blocking. */
export class SequenceIndexDeferredOn {
  readonly #index: CompiledSequenceIndex;
  /** The executor every verb is submitted to. */
  readonly executor: Executor;

  /** Made by `index.deferred(executor)`, never directly. */
  constructor(index: CompiledSequenceIndex, executor: Executor) {
    this.#index = index;
    this.executor = executor;
  }

  /** {@inheritDoc SequenceIndex.reserve} */
  reserve(entries: number, payloadBytes?: number): Deferred<void> {
    return defer(sequenceReserve(this.#index, this.executor, entries, payloadBytes));
  }

  /** {@inheritDoc SequenceIndex.clear} */
  clear(): Deferred<void> {
    return defer(sequenceClear(this.#index, this.executor));
  }

  /** {@inheritDoc SequenceIndex.describe} */
  describe(): Deferred<SequenceIndexDescription> {
    return defer(sequenceDescribe(this.#index, this.executor));
  }

  /** {@inheritDoc SequenceIndex.health} */
  health(): Deferred<DenseReachability> {
    return defer(sequenceHealth(this.#index, this.executor));
  }

  /** {@inheritDoc SequenceIndex.count} */
  count(key: KeyLike): Deferred<SequenceKeyStats> {
    return defer(sequenceCount(this.#index, this.executor, key));
  }

  /** {@inheritDoc SequenceIndex.insert} */
  insert(keys: KeysLike, sequences: SequencesLike, config: DenseInsertConfig = {}): Deferred<void> {
    return defer(sequenceInsert(this.#index, this.executor, keys, sequences, config));
  }

  /** {@inheritDoc SequenceIndex.refine} */
  refine(config: SequenceRefineConfig = {}): Deferred<DenseRefineReport> {
    return defer(sequenceRefine(this.#index, this.executor, config));
  }

  /** {@inheritDoc SequenceIndex.refinePlanFinish} */
  refinePlanFinish(plan: SequenceRefinePlan): Deferred<DenseRefineReport> {
    return defer(sequenceRefinePlanFinish(this.#index, this.executor, plan));
  }

  /** {@inheritDoc SequenceIndex.search} */
  search(queries: SequencesLike, wanted: number, arena: Arena, config: DenseSearchConfig = {}): Deferred<Matches> {
    return defer(sequenceSearch(this.#index, this.executor, queries, wanted, arena, config));
  }

  /** {@inheritDoc SequenceIndex.get} */
  get(key: KeyLike, arena: Arena): Deferred<Tape> {
    return defer(sequenceGet(this.#index, this.executor, key, arena));
  }

  /** {@inheritDoc SequenceIndex.reshard} */
  reshard(
    sources: SequenceIndex[],
    targets: SequenceIndex[],
    config: SequenceReshardConfig = {},
  ): Deferred<DenseRefineReport> {
    return defer(sequenceReshard(this.executor, sources, targets, config));
  }

  /** {@inheritDoc SequenceIndex.reshardPlanFinish} */
  reshardPlanFinish(plan: SequenceReshardPlan): Deferred<DenseRefineReport> {
    return defer(sequenceReshardPlanFinish(this.executor, plan));
  }

  /** {@inheritDoc SequenceIndex.sample} */
  sample(wanted: number, arena: Arena, config: DenseSampleConfig = {}): Deferred<DenseDraws> {
    return defer(sequenceSample(this.#index, this.executor, wanted, arena, config));
  }

  /** {@inheritDoc SequenceIndex.save} */
  save(path: string): Deferred<void> {
    return defer(sequenceSave(this.#index, this.executor, path));
  }

  /** {@inheritDoc SequenceIndex.saveBuffer} */
  saveBuffer(arena: Arena): Deferred<Monolith> {
    return defer(sequenceSaveBuffer(this.#index, this.executor, arena));
  }
}

/** The token the lifecycle verbs hold, which a direct `new SequenceIndex` cannot produce. */
const indexSeal = Symbol("SequenceIndex");

/** A sequence index over byte strings, running every verb on the calling thread. `on(executor)` and
 *  `deferred(executor)` reach the same verbs on an executor. */
export class SequenceIndex implements Disposable {
  /** The native handle every verb names. @internal */
  readonly compiled: CompiledSequenceIndex;

  /** Reached through the `init*`, `load*` and `view*` lifecycle verbs. */
  private constructor(seal: symbol, compiledIndex: CompiledSequenceIndex) {
    if (seal !== indexSeal)
      throw fail(Status.InvalidArgument, "A SequenceIndex is opened by its init, load or view verbs.");
    this.compiled = compiledIndex;
  }

  /** The same verbs, blocking on `executor`. Binding-only. */
  on(executor: Executor): SequenceIndexOn {
    return new SequenceIndexOn(this.compiled, executor);
  }

  /** The same verbs, submitted to `executor`, each answered as a {@link Deferred}. Binding-only. */
  deferred(executor: Executor): SequenceIndexDeferredOn {
    return new SequenceIndexDeferredOn(this.compiled, executor);
  }

  /**
   *  Claims room ahead of insertion, a hint, since insertions grow the index on demand.
   *  @param entries - The desired total entry count, the current size included.
   *  @param payloadBytes - The bytes those entries span in all; absent derives.
   */
  reserve(entries: number, payloadBytes?: number): void {
    return block(sequenceReserve(this.compiled, undefined, entries, payloadBytes));
  }

  /** Erases every entry, keeping the reserved capacity. */
  clear(): void {
    return block(sequenceClear(this.compiled, undefined));
  }

  /** Reports the frozen metadata and the current counters in one call. */
  describe(): SequenceIndexDescription {
    return block(sequenceDescribe(this.compiled, undefined));
  }

  /** Audits how much of the graph a search reaches from its root. */
  health(): DenseReachability {
    return block(sequenceHealth(this.compiled, undefined));
  }

  /**
   *  Counts the entries stored under one key, with the bytes they span.
   *  @param key - The identifier to be enumerated.
   */
  count(key: KeyLike): SequenceKeyStats {
    return block(sequenceCount(this.compiled, undefined, key));
  }

  /** Whether the index holds any entry under one key. Binding-only sugar over `count`. */
  contains(key: KeyLike): boolean {
    return this.count(key).entries !== 0;
  }

  /**
   *  Adds sequences under their keys, one key per sequence; text crosses as UTF-8. A batch is
   *  all-or-nothing.
   *  @param keys - One identifier per sequence.
   *  @param sequences - Text, bytes, or a packed {@link Tape}.
   *  @param config - Per-call breadth and telemetry sink.
   */
  insert(keys: KeysLike, sequences: SequencesLike, config: DenseInsertConfig = {}): void {
    return block(sequenceInsert(this.compiled, undefined, keys, sequences, config));
  }

  /**
   *  Re-selects the neighbors of every stored entry against the finished graph; loop while the
   *  report's `verdict` is "improved".
   *  @param config - Per-call breadth, NVMe workspace, telemetry sink and plan.
   */
  refine(config: SequenceRefineConfig = {}): DenseRefineReport {
    return block(sequenceRefine(this.compiled, undefined, config));
  }

  /** Closes a refine plan and frees it whatever the call answers, reporting the whole plan. */
  refinePlanFinish(plan: SequenceRefinePlan): DenseRefineReport {
    return block(sequenceRefinePlanFinish(this.compiled, undefined, plan));
  }

  /**
   *  Finds the nearest stored sequences of every query, landing them in `arena`.
   *  @param queries - Text, bytes, or a packed {@link Tape}.
   *  @param wanted - Matches per query.
   *  @param arena - The memory the matches land in, rewound by the call.
   *  @param config - Per-call breadth, traversal and telemetry sink.
   */
  search(queries: SequencesLike, wanted: number, arena: Arena, config: DenseSearchConfig = {}): Matches {
    return block(sequenceSearch(this.compiled, undefined, queries, wanted, arena, config));
  }

  /**
   *  Retrieves every sequence stored under one key, in insertion order, as a tape inside `arena`.
   *  @param key - The identifier whose entries are exported.
   *  @param arena - The memory the tape lands in, rewound by the call.
   */
  get(key: KeyLike, arena: Arena): Tape {
    return block(sequenceGet(this.compiled, undefined, key, arena));
  }

  /**
   *  Moves the entries of `sources` into `targets`, keeping their links, then runs a round on every
   *  target that took entries: several sources gather into one target, and one source cuts into
   *  several. A refusal moves nothing.
   *  @param sources - The indexes whose entries move, each named once.
   *  @param targets - The indexes the entries move into, each a source itself or empty.
   *  @param config - The seed, breadth, NVMe workspace, telemetry sink and plan.
   */
  reshard(sources: SequenceIndex[], targets: SequenceIndex[], config: SequenceReshardConfig = {}): DenseRefineReport {
    return block(sequenceReshard(undefined, sources, targets, config));
  }

  /** Closes a reshard plan and frees it whatever the call answers, reporting the whole plan. */
  reshardPlanFinish(plan: SequenceReshardPlan): DenseRefineReport {
    return block(sequenceReshardPlanFinish(undefined, plan));
  }

  /** Draws keys around `config.focus`, or across the whole index, placed in three dimensions. */
  sample(wanted: number, arena: Arena, config: DenseSampleConfig = {}): DenseDraws {
    return block(sequenceSample(this.compiled, undefined, wanted, arena, config));
  }

  /** Saves the index in the layout the path extension picks. */
  save(path: string): void {
    return block(sequenceSave(this.compiled, undefined, path));
  }

  /** Writes the index into `arena` as one monolith. */
  saveBuffer(arena: Arena): Monolith {
    return block(sequenceSaveBuffer(this.compiled, undefined, arena));
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
   *  @param options - The scoring.
   *  @param memoryDomains - NUMA node IDs to shard over; absent spreads over every node.
   */
  static initRam(options: SequenceOptions = {}, memoryDomains?: number[]): SequenceIndex {
    return new SequenceIndex(
      indexSeal,
      compiled.CompiledSequenceIndex.initRam(
        options.metric ?? SequenceMetric.Levenshtein,
        options.substitutionMatrix,
        options.gap,
        options.windowWidths,
        options.connectivity,
        options.valuesPerKey,
        memoryDomains,
        options.cpuClassCeiling,
      ),
    );
  }

  /**
   *  Creates an empty, mutable index in device memory, one shard per named CUDA ordinal.
   *  @param options - The scoring.
   *  @param devices - CUDA ordinals to shard over; absent takes one free device.
   */
  static initVram(options: SequenceOptions = {}, devices?: number[]): SequenceIndex {
    return new SequenceIndex(
      indexSeal,
      compiled.CompiledSequenceIndex.initVram(
        options.metric ?? SequenceMetric.Levenshtein,
        options.substitutionMatrix,
        options.gap,
        options.windowWidths,
        options.connectivity,
        options.valuesPerKey,
        devices,
        options.gpuClassCeiling,
      ),
    );
  }

  /** Copies a checkpoint into a new, mutable index in host memory. */
  static loadRam(path: string, config: SequenceLoadConfig = {}): SequenceIndex {
    return new SequenceIndex(
      indexSeal,
      compiled.CompiledSequenceIndex.loadRam(
        path,
        config.executor,
        config.placement,
        config.valuesPerKey,
        config.classCeiling,
        ...weightsOf(config),
      ),
    );
  }

  /** Copies a checkpoint into a new, mutable index in device memory. */
  static loadVram(path: string, config: SequenceLoadConfig = {}): SequenceIndex {
    return new SequenceIndex(
      indexSeal,
      compiled.CompiledSequenceIndex.loadVram(
        path,
        config.executor,
        config.placement,
        config.valuesPerKey,
        config.classCeiling,
        ...weightsOf(config),
      ),
    );
  }

  /** Copies a monolith held in memory into a new, mutable index in host memory. */
  static loadRamBuffer(buffer: Uint8Array | Monolith, config: SequenceLoadConfig = {}): SequenceIndex {
    return new SequenceIndex(
      indexSeal,
      compiled.CompiledSequenceIndex.loadRamBuffer(
        buffer instanceof Monolith ? buffer.bytes : buffer,
        config.executor,
        config.placement,
        config.valuesPerKey,
        config.classCeiling,
        ...weightsOf(config),
      ),
    );
  }

  /** Copies a monolith held in memory into a new, mutable index in device memory. */
  static loadVramBuffer(buffer: Uint8Array | Monolith, config: SequenceLoadConfig = {}): SequenceIndex {
    return new SequenceIndex(
      indexSeal,
      compiled.CompiledSequenceIndex.loadVramBuffer(
        buffer instanceof Monolith ? buffer.bytes : buffer,
        config.executor,
        config.placement,
        config.valuesPerKey,
        config.classCeiling,
        ...weightsOf(config),
      ),
    );
  }

  /** Serves a checkpoint in place through a memory mapping, as a read-only index. */
  static viewRam(path: string, weights: SequenceWeights & { cpuClassCeiling?: string } = {}): SequenceIndex {
    return new SequenceIndex(
      indexSeal,
      compiled.CompiledSequenceIndex.viewRam(path, weights.cpuClassCeiling, ...weightsOf(weights)),
    );
  }

  /** Serves a checkpoint in place from NVMe drives, as a read-only index. */
  static viewNvme(path: string, config: SequenceViewNvmeConfig = {}): SequenceIndex {
    return new SequenceIndex(
      indexSeal,
      compiled.CompiledSequenceIndex.viewNvme(
        path,
        config.executor,
        config.paths,
        config.cpuClassCeiling,
        ...weightsOf(config),
      ),
    );
  }

  /** Reads a checkpoint's metric and connectivity without loading it. */
  static metadataRead(path: string, config: { executor?: Executor } = {}): SequenceMetadata {
    return metadataOf(compiled.CompiledSequenceIndex.metadataRead(path, config.executor));
  }

  /** Reads the metric and connectivity of a monolith held in memory. */
  static metadataReadBuffer(buffer: Uint8Array | Monolith, config: { executor?: Executor } = {}): SequenceMetadata {
    const bytes = buffer instanceof Monolith ? buffer.bytes : buffer;
    return metadataOf(compiled.CompiledSequenceIndex.metadataReadBuffer(bytes, config.executor));
  }
}
