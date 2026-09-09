/**
 * Vector search over equidimensional dense embeddings, and the snapshot cursor and resumable
 * finder one index opens.
 */
import {
  CompiledCursor,
  CompiledFinder,
  CompiledIndex,
  CompiledPlacement,
  compiled,
} from "./library";
import {
  Executor,
  KeyLike,
  KeysLike,
  MatchesRow,
  MatchesSpan,
  Matrix,
  MetricKind,
  Residency,
  ScalarKind,
  Vector,
  VectorOrMatrix,
  isNonNegativeInteger,
  isOneKey,
  isPositiveInteger,
  keyBytes,
  normalizeKey,
  normalizeKeys,
  normalizeVectors,
  refuseForeignArms,
  rowBytes,
  scalarOf,
} from "./types";

/**
 * Traversal telemetry, accumulated across the calls it is passed to.
 *
 * One accumulator handed to several calls sums them, so a sweep reads its totals at the end
 * rather than adding the readings up itself.
 */
export class Stats {
  /** Entries whose neighborhoods the walk expanded. */
  visitedMembers = 0;
  /** Distance evaluations the walk paid for, diversity re-selections included. */
  computedDistances = 0;
  /** Times the walk fell back to a fresh entry point after the frontier drained early. */
  restarts = 0;

  /** Folds another reading into this one. */
  accumulate(other: Stats): void {
    this.visitedMembers += other.visitedMembers;
    this.computedDistances += other.computedDistances;
    this.restarts += other.restarts;
  }
}

/** Optional per-call settings of an insertion, and of the refinement pass that follows it. */
export interface InsertionOptions {
  /** Candidate-pool breadth the build walk explores; absent lets the engine derive it. */
  expansionFactor?: number;
  /** The scalar kind the rows hold, for the packed and mini-float kinds no TypedArray spells. */
  scalarKind?: ScalarKind;
  /** The accumulator this call folds its telemetry into; absent skips reporting. */
  stats?: Stats;
}

/** Optional per-call settings of a read-only walk. */
export interface SearchOptions {
  /** Candidate-pool breadth the walk explores; absent lets the engine derive it. */
  expansionFactor?: number;
  /** The scalar kind the rows hold, for the packed and mini-float kinds no TypedArray spells. */
  scalarKind?: ScalarKind;
  /** The accumulator this call folds its telemetry into; absent skips reporting. */
  stats?: Stats;
}

/**
 * Optional per-call settings of attention.
 *
 * The two floating knobs shape the softmax rather than the walk: `scoreScale` divides the
 * logits before they are exponentiated, and `massTolerance` stops a walk once the residual
 * probability mass falls below it.
 */
export interface AttentionOptions {
  /** Candidate-pool breadth the walk explores; absent lets the engine derive it. */
  expansionFactor?: number;
  /** Divisor applied to every logit before the softmax; absent derives it from the geometry. */
  scoreScale?: number;
  /** Residual-mass threshold that ends a walk early; absent derives it. */
  massTolerance?: number;
  /** The scalar kind the query rows hold, for the kinds no TypedArray spells. */
  scalarKind?: ScalarKind;
  /** The scalar kind the value rows hold; absent reads it off the array. */
  valuesScalarKind?: ScalarKind;
  /**
   * Two floats per query row - the running maximum and the partition taken against it.
   *
   * Supplying one folds this branch into another over the same queries by partial-softmax
   * rescaling; a maximum without its partition is a number nothing can rescale, which is why
   * they share a row rather than travelling as two arrays.
   */
  carry?: Float32Array;
  /** The accumulator this call folds its telemetry into; absent skips reporting. */
  stats?: Stats;
}

/**
 * Optional per-call settings of a resumable stream.
 *
 * A finder widens as it is pulled, so it takes no breadth, and the ABI reports no telemetry
 * for one either.
 */
export interface FinderOptions {
  /** The scalar kind the rows hold, for the packed and mini-float kinds no TypedArray spells. */
  scalarKind?: ScalarKind;
}

/** The geometry and metric a checkpoint froze; where it was written is never recorded. */
export interface Metadata {
  dimensions: number;
  metric: MetricKind;
  quantization: ScalarKind;
  connectivity: number;
  valuesPerKey: number;
}

/** The geometry every dense index freezes at construction, whatever its residency. */
export interface DenseIndexOptions {
  /** Scalars per row, fixed for the life of the index. */
  dimensions: number;
  /** How dissimilarity is scored; defaults to `angular`. */
  metric?: MetricKind;
  /** The scalar kind rows are stored in; defaults to `f32`. */
  quantization?: ScalarKind;
  /** Neighbors per node; defaults to 0, which derives from the geometry. */
  connectivity?: number;
  /** Entries one key may carry; defaults to 0, which derives to unique keys. */
  valuesPerKey?: number;
}

/** A dense index kept in host memory. */
export interface RamIndexOptions extends DenseIndexOptions {
  /** NUMA node ids to shard over; absent spreads the shards over every node. */
  memoryDomains?: number[];
}

/** A dense index kept in device memory. */
export interface VramIndexOptions extends DenseIndexOptions {
  /** CUDA ordinals to shard over; absent takes every visible device. */
  devices?: number[];
}

/** A dense index kept on disk. */
export interface NvmeIndexOptions extends DenseIndexOptions {
  /** Device or directory paths to shard over; absent derives them. */
  paths?: string[];
}

/** One batch of a snapshot walk: identities and the rows they name. */
export interface CursorBatch {
  keys: Uint8Array;
  vectors: Float32Array;
}

/**
 * A consistent snapshot walk over an index, taken at the moment it was opened, so entries
 * inserted afterwards never appear mid-iteration.
 *
 * Holds engine state until `free` is called; a walk left to the garbage collector releases it
 * whenever the collector gets to it.
 */
export class Cursor {
  /** Wraps the native cursor; opened by {@link DenseIndex.cursor} rather than constructed. */
  constructor(private readonly compiled: CompiledCursor) { }

  /**
   * Pulls the next rows of the snapshot.
   *
   * @param {number} batch - Rows to pull at most.
   * @return {CursorBatch|null} The rows pulled, or null once the snapshot is exhausted.
   */
  next(batch: number): CursorBatch | null {
    const pulled = this.compiled.next(batch);
    if (!pulled) return null;
    return { keys: pulled[0], vectors: pulled[1] };
  }

  /** Releases the snapshot; a second call does nothing. */
  free(): void {
    this.compiled.free();
  }
}

/**
 * A resumable nearest-neighbor stream over one query, pulled from the caller's own thread.
 *
 * This is the filtered path: pull matches, drop the ones a JavaScript predicate rejects, and
 * pull again until satisfied - rather than guessing a `k` wide enough to survive the filter.
 * A live finder pins one of the executor's threads, so free it the moment pulling stops.
 */
export class Finder {
  /** Wraps the native finder; opened by {@link DenseIndex.find} rather than constructed. */
  constructor(private readonly compiled: CompiledFinder) { }

  /**
   * Pulls the next-nearest matches from the stream.
   *
   * @param {number} wanted - Matches to pull at most.
   * @return {MatchesRow|null} The matches pulled, or null once the traversal is drained.
   */
  next(wanted: number): MatchesRow | null {
    const pulled = this.compiled.next(wanted);
    if (!pulled) return null;
    return new MatchesRow(pulled[0], pulled[1]);
  }

  /** Releases the stream and the executor thread it pinned; a second call does nothing. */
  free(): void {
    this.compiled.free();
  }
}

/** Repeats one identity across `rows` entries, which is how a single key broadcasts. */
function repeatKeys(
  keys: Uint8Array | BigUint64Array,
  rows: number
): Uint8Array | BigUint64Array {
  if (keys instanceof BigUint64Array)
    return BigUint64Array.from({ length: rows }, () => keys[0]);
  const repeated = new Uint8Array(rows * keyBytes);
  for (let position = 0; position < rows; position++)
    repeated.set(keys, position * keyBytes);
  return repeated;
}

/** The identities a normalized run carries. */
function countKeys(keys: Uint8Array | BigUint64Array): number {
  return keys instanceof BigUint64Array ? keys.length : keys.length / keyBytes;
}

/** The slots the addon overwrites with one call's telemetry, in the order `Stats` reads them. */
const statsSlots = 3;

/** The sink one call reports into, allocated only when an accumulator asked for a reading. */
function statsSink(stats: Stats | undefined): Float64Array | undefined {
  return stats ? new Float64Array(statsSlots) : undefined;
}

/** Folds a filled sink into the accumulator that asked for it. */
function reportStats(stats: Stats | undefined, sink: Float64Array | undefined): void {
  if (!stats || !sink) return;
  stats.visitedMembers += sink[0];
  stats.computedDistances += sink[1];
  stats.restarts += sink[2];
}

/** The token the three factories hold, which is what a direct `new DenseIndex` cannot produce. */
const indexSeal = Symbol("DenseIndex");

export class DenseIndex {
  #compiledIndex: CompiledIndex;

  /** Reached through {@link DenseIndex.forRam}, {@link DenseIndex.forVram} or {@link DenseIndex.forNvme}. */
  private constructor(seal: symbol, compiledIndex: CompiledIndex) {
    if (seal !== indexSeal)
      throw new Error(
        "A DenseIndex is opened by DenseIndex.forRam, DenseIndex.forVram or DenseIndex.forNvme."
      );
    this.#compiledIndex = compiledIndex;
  }

  /** Checks the shared geometry and opens the native index of one residency. */
  static #opened(
    options: DenseIndexOptions,
    residency: Residency,
    placement: CompiledPlacement
  ): DenseIndex {
    const metric = options.metric ?? MetricKind.Angular;
    const quantization = options.quantization ?? ScalarKind.F32;
    const connectivity = options.connectivity ?? 0;
    const valuesPerKey = options.valuesPerKey ?? 0;

    if (!isPositiveInteger(options.dimensions))
      throw new Error("`dimensions` must be a positive integer.");
    if (!isNonNegativeInteger(connectivity) || !isNonNegativeInteger(valuesPerKey))
      throw new Error(
        "`connectivity` and `valuesPerKey` must be non-negative integers, 0 deriving each."
      );
    if (!Object.values(MetricKind).includes(metric))
      throw new Error(
        `Invalid metric: ${metric}. It must be one of: ${Object.values(
          MetricKind
        ).join(", ")}`
      );
    if (!Object.values(ScalarKind).includes(quantization))
      throw new Error(
        `Invalid quantization: ${quantization}. It must be one of: ${Object.values(
          ScalarKind
        ).join(", ")}`
      );

    return new DenseIndex(
      indexSeal,
      new compiled.CompiledIndex(
        options.dimensions,
        metric,
        quantization,
        connectivity,
        valuesPerKey,
        residency,
        placement
      )
    );
  }

  /**
   * Places the graph in host memory, one shard per named NUMA node.
   *
   * @param {RamIndexOptions} options - The geometry, and the memory domains to shard over.
   * @return {DenseIndex} An index a RAM executor's verbs run on.
   * @throws Will throw an error if any option is of incorrect type or invalid value.
   */
  static forRam(options: RamIndexOptions): DenseIndex {
    refuseForeignArms(options, "memoryDomains", "DenseIndex.forRam");
    return DenseIndex.#opened(options, Residency.Ram, options.memoryDomains);
  }

  /**
   * Places the graph on CUDA devices, one shard per named ordinal.
   *
   * @param {VramIndexOptions} options - The geometry, and the devices to shard over.
   * @return {DenseIndex} An index a VRAM executor's verbs run on.
   * @throws Will throw an error if any option is of incorrect type or invalid value.
   */
  static forVram(options: VramIndexOptions): DenseIndex {
    refuseForeignArms(options, "devices", "DenseIndex.forVram");
    return DenseIndex.#opened(options, Residency.Vram, options.devices);
  }

  /**
   * Places the graph on disk, one shard per named path.
   *
   * @param {NvmeIndexOptions} options - The geometry, and the paths to shard over.
   * @return {DenseIndex} An index an NVMe executor's verbs run on.
   * @throws Will throw an error if any option is of incorrect type or invalid value.
   */
  static forNvme(options: NvmeIndexOptions): DenseIndex {
    refuseForeignArms(options, "paths", "DenseIndex.forNvme");
    return DenseIndex.#opened(options, Residency.Nvme, options.paths);
  }

  /**
   * Reads a checkpoint's geometry and metric without loading it.
   *
   * @param {string} path - The manifest of a sharded checkpoint, or a single monolith.
   * @return {Metadata} The geometry and metric the checkpoint froze.
   */
  static metadata(path: string): Metadata {
    const read = compiled.CompiledIndex.metadata(path);
    return {
      dimensions: read.dimensions,
      metric: read.metric as MetricKind,
      quantization: read.quantization as ScalarKind,
      connectivity: read.connectivity,
      valuesPerKey: read.valuesPerKey,
    };
  }

  /**
   * Reads the geometry and metric of a monolith held in memory.
   *
   * @param {Uint8Array} buffer - The bytes {@link DenseIndex.saveBuffer} produced.
   * @return {Metadata} The geometry and metric the monolith froze.
   */
  static metadataBuffer(buffer: Uint8Array): Metadata {
    const read = compiled.CompiledIndex.metadataBuffer(buffer);
    return {
      dimensions: read.dimensions,
      metric: read.metric as MetricKind,
      quantization: read.quantization as ScalarKind,
      connectivity: read.connectivity,
      valuesPerKey: read.valuesPerKey,
    };
  }

  /**
   * Unwraps the executor a fanned-out verb runs on. Required rather than derived: a pool
   * spawned behind the caller's back claims most of the machine's cores invisibly, and
   * nothing here could ever free it again.
   */
  #executorOf(executor: Executor): Executor {
    if (!(executor instanceof Executor))
      throw new Error("`executor` must be an Executor.");
    return executor;
  }

  /** Rows a flattened batch splits into, at the index's frozen width and the named kind. */
  #rowsOf(vectors: Vector, kind: ScalarKind): number {
    const width = rowBytes(this.#compiledIndex.dimensions(), kind);
    if (!width || vectors.byteLength % width !== 0)
      throw new Error(
        "The size of the flattened vectors must be a whole multiple of one row."
      );
    return vectors.byteLength / width;
  }

  /**
   * Add vectors to the index.
   *
   * Vectors arrive as a flat typed array of `n * d` scalars, one row per key, or as an array
   * of rows this method lays end to end. A single key broadcasts across every row.
   *
   * Rows keep whatever type they arrived in and the engine converts them into the index's own
   * quantization, so the two are independent - a packed `u1x8` or a `bf16` row travels as
   * bytes with `scalarKind` naming what they mean.
   *
   * @param {KeysLike} keys - Input identifiers for every vector.
   * @param {VectorOrMatrix} vectors - Input matrix of size n * d.
   * @param {Executor} executor - The workers to fan the batch out over.
   * @param {InsertionOptions} [options] - Per-call breadth, scalar kind and telemetry sink.
   * @return {number} The rows published, which is every row unless the batch was stopped.
   * @throws Will throw an error if the length of keys doesn't match the number of vectors
   *         or if it's not a single key.
   */
  add(
    keys: KeysLike,
    vectors: VectorOrMatrix | number[][],
    executor: Executor,
    options: InsertionOptions = {}
  ): number {
    const normalizedVectors = normalizeVectors(vectors);
    const scalarKind = options.scalarKind ?? scalarOf(normalizedVectors);
    const countVectors = this.#rowsOf(normalizedVectors, scalarKind);

    let normalizedKeys = normalizeKeys(keys);
    if (countKeys(normalizedKeys) === 1 && countVectors > 1)
      normalizedKeys = repeatKeys(normalizedKeys, countVectors);
    else if (countKeys(normalizedKeys) !== countVectors)
      throw new Error(
        `The length of keys (${countKeys(
          normalizedKeys
        )}) must match the number of vectors (${countVectors}) or be a single key.`
      );

    const sink = statsSink(options.stats);
    const published = this.#compiledIndex.add(
      normalizedKeys,
      normalizedVectors,
      this.#executorOf(executor).compiled,
      options.expansionFactor ?? 0,
      options.scalarKind,
      sink
    );
    reportStats(options.stats, sink);
    return published;
  }

  /**
   * Re-selects the neighbors of every stored entry against the finished graph.
   *
   * A cheap insertion pass followed by one refinement reaches a better graph, in less total
   * time, than a single expensive insertion pass - so a wide breadth belongs here.
   *
   * @param {Executor} executor - The workers to fan the sweep out over.
   * @param {InsertionOptions} [options] - Per-call breadth and telemetry sink.
   * @return {number} Nodes whose edges changed; zero means the graph has settled.
   */
  refine(executor: Executor, options: InsertionOptions = {}): number {
    const sink = statsSink(options.stats);
    const changed = this.#compiledIndex.refine(
      this.#executorOf(executor).compiled,
      options.expansionFactor ?? 0,
      sink
    );
    reportStats(options.stats, sink);
    return changed;
  }

  /**
   * Searches the index for the nearest neighbours of every query row.
   *
   * Writes into the run the caller lends and answers with nothing, so a recall sweep allocates
   * once and reuses one run at every point. The matches wanted per query follow from the run's
   * own shape rather than being passed alongside it.
   *
   * @param {VectorOrMatrix} vectors - One row per query, at the index's own width.
   * @param {MatchesSpan} matches - The run to fill; its `wanted` fixes the width.
   * @param {Executor} executor - The workers to fan the batch out over.
   * @param {SearchOptions} [options] - Per-call breadth, scalar kind and telemetry sink.
   * @throws Will throw when the queries do not split into whole rows of the index geometry.
   */
  search(
    vectors: VectorOrMatrix | number[][],
    matches: MatchesSpan,
    executor: Executor,
    options: SearchOptions = {}
  ): void {
    const normalizedVectors = normalizeVectors(vectors);
    const sink = statsSink(options.stats);
    this.#compiledIndex.search(
      normalizedVectors,
      matches.keys,
      matches.distances,
      matches.counts,
      this.#executorOf(executor).compiled,
      options.expansionFactor ?? 0,
      options.scalarKind,
      sink
    );
    reportStats(options.stats, sink);
  }

  /**
   * Multiplies a matrix against the index, sparsely.
   *
   * Every left row emits up to `wanted` pairs rather than one ranked match, and the scores are
   * unnormalized logits, so the handle must hold the inner-product kind.
   *
   * @param {VectorOrMatrix} left - The rows to multiply, at the index's own width.
   * @param {MatchesSpan} products - The run to fill; its keys name each pair's indexed row.
   * @param {Executor} executor - The workers to fan the batch out over.
   * @param {SearchOptions} [options] - Per-call breadth, scalar kind and telemetry sink.
   * @throws Will throw when the left rows do not span whole rows of the index width.
   */
  multiply(
    left: VectorOrMatrix | number[][],
    products: MatchesSpan,
    executor: Executor,
    options: SearchOptions = {}
  ): void {
    const normalizedLeft = normalizeVectors(left);
    const sink = statsSink(options.stats);
    this.#compiledIndex.multiply(
      normalizedLeft,
      products.keys,
      products.distances,
      products.counts,
      this.#executorOf(executor).compiled,
      options.expansionFactor ?? 0,
      options.scalarKind,
      sink
    );
    reportStats(options.stats, sink);
  }

  /**
   * Attends over the index, writing into the buffer the caller lends.
   *
   * Answers with nothing: the outputs are the caller's array, which is what lets a sweep
   * allocate once. The value rows carry their own width, which need not be the index's.
   *
   * @param {VectorOrMatrix} queries - One row per query, at the index's own width.
   * @param {VectorOrMatrix} values - The value table, at `valueDimensions` scalars per row.
   * @param {number} valueDimensions - Scalars per value row.
   * @param {ArrayBufferView} outputs - One row of `valueDimensions` per query, written in place.
   * @param {Executor} executor - The workers to fan the batch out over.
   * @param {AttentionOptions} [options] - Softmax shape, the carry, and the telemetry sink.
   */
  attend(
    queries: VectorOrMatrix | number[][],
    values: VectorOrMatrix | number[][],
    valueDimensions: number,
    outputs: ArrayBufferView,
    executor: Executor,
    options: AttentionOptions = {}
  ): void {
    if (!isPositiveInteger(valueDimensions))
      throw new Error("`valueDimensions` must be a positive integer.");

    const normalizedQueries = normalizeVectors(queries);
    const normalizedValues = normalizeVectors(values);
    const sink = statsSink(options.stats);
    this.#compiledIndex.attend(
      normalizedQueries,
      normalizedValues,
      valueDimensions,
      outputs,
      options.carry,
      this.#executorOf(executor).compiled,
      options.expansionFactor ?? 0,
      options.scoreScale ?? 0,
      options.massTolerance ?? 0,
      options.scalarKind,
      options.valuesScalarKind,
      sink
    );
    reportStats(options.stats, sink);
  }

  /**
   * Late-interaction scoring of multi-row queries: every query row takes its best match within
   * a key, and those maxima sum to that key's score.
   *
   * The run is over keys rather than rows, which is what an index built with `valuesPerKey`
   * above one exists for. Runs on the inner-product kinds alone.
   *
   * @param {VectorOrMatrix} vectors - The token rows of every query, laid end to end.
   * @param {number} rowsPerQuery - Rows in every query; uniform, so padded queries apply as is.
   * @param {MatchesSpan} matches - The run to fill; one row per query, not per token row.
   * @param {Executor} executor - The workers to fan the batch out over.
   * @param {SearchOptions} [options] - Per-call breadth, scalar kind and telemetry sink.
   * @throws Will throw when the token rows do not split into whole queries.
   */
  maxsim(
    vectors: VectorOrMatrix | number[][],
    rowsPerQuery: number,
    matches: MatchesSpan,
    executor: Executor,
    options: SearchOptions = {}
  ): void {
    if (!isPositiveInteger(rowsPerQuery))
      throw new Error("`rowsPerQuery` must be a positive integer.");

    const normalizedVectors = normalizeVectors(vectors);
    const sink = statsSink(options.stats);
    this.#compiledIndex.maxsim(
      normalizedVectors,
      rowsPerQuery,
      matches.keys,
      matches.distances,
      matches.counts,
      this.#executorOf(executor).compiled,
      options.expansionFactor ?? 0,
      options.scalarKind,
      sink
    );
    reportStats(options.stats, sink);
  }

  /**
   * Verifies the presence of one or more keys in the index.
   *
   * A key spelled on its own answers with a boolean; a run of keys answers with one boolean
   * apiece, in the order they were given.
   *
   * @param {KeysLike} keys - The identifier(s) to be checked for presence in the index.
   * @return {boolean|boolean[]} - Whether each key is present.
   * @throws Will throw an error if a key is spelled in none of the accepted forms.
   */
  contains(key: bigint | number | string): boolean;
  contains(keys: Uint8Array | BigUint64Array | Array<bigint | number | string | Uint8Array>): boolean[];
  contains(keys: KeysLike): boolean | boolean[] {
    const found = this.#compiledIndex.contains(normalizeKeys(keys));
    return isOneKey(keys) ? found[0] : found;
  }

  /**
   * Counts the entries stored under one or more keys.
   *
   * A key spelled on its own answers with a number; a run of keys answers with one number
   * apiece, in the order they were given.
   *
   * @param {KeysLike} keys - The identifier(s) to be enumerated.
   * @return {number|number[]} - How many entries each key holds.
   * @throws Will throw an error if a key is spelled in none of the accepted forms.
   */
  count(key: bigint | number | string): number;
  count(keys: Uint8Array | BigUint64Array | Array<bigint | number | string | Uint8Array>): number[];
  count(keys: KeysLike): number | number[] {
    const counted = this.#compiledIndex.count(normalizeKeys(keys));
    return isOneKey(keys) ? counted[0] : counted;
  }

  /**
   * Retrieves the vectors stored under one key, one row per entry, in insertion order.
   *
   * Rows come back as `f32` whatever the index quantized them into.
   *
   * @param {KeyLike} key - The identifier whose entries are exported.
   * @return {Float32Array} - The rows, laid end to end; empty when the key is absent.
   */
  get(key: KeyLike): Float32Array {
    return this.#compiledIndex.get(normalizeKey(key));
  }

  /**
   * Opens a consistent snapshot walk over the entries published so far.
   * @return {Cursor} - The walk, which must be freed once pulling stops.
   */
  cursor(): Cursor {
    return new Cursor(this.#compiledIndex.cursor());
  }

  /**
   * Opens a resumable nearest-neighbor stream over one query row.
   *
   * @param {Vector} query - A single query row of the index geometry.
   * @param {Executor} executor - The pool one thread is pinned from until the finder is freed.
   * @param {FinderOptions} [options] - The scalar kind, when the TypedArray spells none.
   * @return {Finder} - The stream, which must be freed the moment pulling stops.
   */
  find(query: Vector, executor: Executor, options: FinderOptions = {}): Finder {
    return new Finder(
      this.#compiledIndex.find(
        query,
        this.#executorOf(executor).compiled,
        options.scalarKind
      )
    );
  }

  /**
   * Claims room for a total entry count ahead of insertion.
   * @param {number} capacity - The desired total capacity, the current size included.
   */
  reserve(capacity: number): void {
    this.#compiledIndex.reserve(capacity);
  }

  /** Erases every entry, keeping the reserved capacity and the frozen geometry. */
  clear(): void {
    this.#compiledIndex.clear();
  }

  /**
   * Returns the dimensionality of vectors.
   * @return {number} The dimensionality of vectors.
   */
  dimensions(): number {
    return this.#compiledIndex.dimensions();
  }

  /**
   * Returns connectivity.
   * @return {number} The connectivity of index.
   */
  connectivity(): number {
    return this.#compiledIndex.connectivity();
  }

  /**
   * Returns the number of vectors currently indexed.
   * @return {number} The number of vectors currently indexed.
   */
  size(): number {
    return this.#compiledIndex.size();
  }

  /**
   * Returns index capacity.
   * @return {number} The capacity of index.
   */
  capacity(): number {
    return this.#compiledIndex.capacity();
  }

  /**
   * Returns the memory the index occupies.
   * @return {number} The memory usage of the index, in bytes.
   */
  memoryUsage(): number {
    return this.#compiledIndex.memoryUsage();
  }

  /**
   * Returns the bytes one monolith of this index takes, which is what {@link saveBuffer} needs.
   * @return {number} The size of a serialized monolith, in bytes.
   */
  storageRequirements(): number {
    return this.#compiledIndex.storageRequirements();
  }

  /**
   * Returns the capability codename this index dispatched to on this machine.
   * @return {string} The codename of the kernel chain in use.
   */
  capabilities(): string {
    return this.#compiledIndex.capabilities();
  }

  /**
   * Write index to a file.
   * @param {string} path File path to write.
   * @param {Executor} executor - The workers the write fans out over.
   * @throws Will throw an error if `path` is not a string.
   */
  save(path: string, executor: Executor): void {
    if (typeof path !== "string")
      throw new Error(
        "`path` must be a string representing the file path to write."
      );
    this.#compiledIndex.save(path, this.#executorOf(executor).compiled);
  }

  /**
   * Load index from a file.
   * @param {string} path File path to read.
   * @param {Executor} executor - The workers the read fans out over.
   * @throws Will throw an error if `path` is not a string.
   */
  load(path: string, executor: Executor): void {
    if (typeof path !== "string")
      throw new Error(
        "`path` must be a string representing the file path to read."
      );
    this.#compiledIndex.load(path, this.#executorOf(executor).compiled);
  }

  /**
   * Writes the index into memory as one monolith, sized for the caller.
   *
   * The bytes are what a checkpoint holds, so a service that keeps its indexes in an object
   * store never touches the filesystem.
   *
   * @param {Executor} executor - The workers the write fans out over.
   * @return {Uint8Array} - One serialized monolith.
   */
  saveBuffer(executor: Executor): Uint8Array {
    return this.#compiledIndex.saveBuffer(this.#executorOf(executor).compiled);
  }

  /**
   * Materializes a monolith held in memory into this handle.
   *
   * @param {Uint8Array} buffer - The bytes {@link saveBuffer} produced.
   * @param {Executor} executor - The workers the read fans out over.
   */
  loadBuffer(buffer: Uint8Array, executor: Executor): void {
    this.#compiledIndex.loadBuffer(buffer, this.#executorOf(executor).compiled);
  }
}
