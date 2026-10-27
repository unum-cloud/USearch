/**
 * Vector search over equidimensional dense embeddings, the receivers its verbs run through, and
 * the cursor and resumable search stream one index opens.
 */
import {
  Completion,
  CompiledBatch,
  CompiledCursor,
  CompiledDeferred,
  CompiledIndex,
  CompiledLinks,
  CompiledMatches,
  CompiledMetadata,
  CompiledProducts,
  CompiledSearchStream,
  CompiledVectors,
  compiled,
} from "./library";
import {
  Arena,
  ArenaView,
  Deferred,
  Executor,
  Key,
  KeyLike,
  KeysLike,
  Matches,
  MatchesRow,
  Monolith,
  RowsView,
  Scalar,
  Status,
  Traversal,
  Vector,
  crossing,
  fail,
  isPositiveInteger,
  keyBytes,
  normalizeKey,
  normalizeKeys,
  rowsOf,
  scalarCodes,
} from "./types";

/**
 * Enumeration representing the metric kinds used to score the dissimilarity of two vectors.
 * @enum {string}
 * @readonly
 */
export enum DenseMetric {
  Inner = "inner",
  Angular = "angular",
  Euclidean = "euclidean",
  Hamming = "hamming",
  Jaccard = "jaccard",
}

/**
 * Traversal telemetry from one call, a pure output the call fully overwrites. Mirrors
 * `usearch_traversal_stats_t`.
 *
 * One reading handed to several calls holds the last of them, so a sweep that wants totals folds
 * them with `accumulate` rather than expecting the calls to sum themselves.
 */
export class DenseTraversalStats {
  /** Entries whose neighborhoods the walk expanded. */
  visitedMembers = 0;
  /** Distance evaluations the walk paid for, diversity re-selections included. */
  computedDistances = 0;
  /** Times the walk fell back to a fresh entry point after the frontier drained early. */
  restarts = 0;

  /** Folds another reading into this one, which is how a sweep gathers totals. Binding-only. */
  accumulate(other: DenseTraversalStats): void {
    this.visitedMembers += other.visitedMembers;
    this.computedDistances += other.computedDistances;
    this.restarts += other.restarts;
  }
}

/** Optional per-call settings of an insertion. Mirrors `usearch_dense_insert_t`. */
export interface DenseInsertConfig {
  /** Breadth of the candidates the build walk explores; absent lets the engine derive it. */
  expansionFactor?: number;
  /** The reading this call overwrites with its own telemetry; absent skips reporting. */
  stats?: DenseTraversalStats;
}

/** Optional per-call settings of the refinement pass that follows an insertion. Mirrors `usearch_dense_refine_t`. */
export interface DenseRefineConfig {
  /** Breadth of the candidates the re-selection walk explores; absent lets the engine derive it. */
  expansionFactor?: number;
  /** The reading this call overwrites with its own telemetry; absent skips reporting. */
  stats?: DenseTraversalStats;
}

/** Optional per-call settings of a search. Mirrors `usearch_dense_search_t`. */
export interface DenseSearchConfig {
  /** Breadth of the candidates the walk explores; absent lets the engine derive it, and a scan ignores it. */
  expansionFactor?: number;
  /** Walks the graph when absent; `Traversal.Scan` scores every row exactly, on RAM and NVMe handles. */
  traversal?: Traversal;
  /** The reading this call overwrites with its own telemetry; absent skips reporting. */
  stats?: DenseTraversalStats;
}

/** Optional per-call settings of late-interaction scoring. Mirrors `usearch_dense_maxsim_t`. */
export interface DenseMaxsimConfig {
  /** Breadth of the candidates the walk explores; absent lets the engine derive it. */
  expansionFactor?: number;
  /** The reading this call overwrites with its own telemetry; absent skips reporting. */
  stats?: DenseTraversalStats;
}

/** Optional settings of a retrieval. Mirrors `usearch_dense_get_t`. */
export interface DenseGetConfig {
  /** The scalar kind to export as, converted from storage; defaults to `f32`. */
  scalar?: Scalar;
}

/** Optional per-call settings of a sparse product. Mirrors `usearch_dense_multiply_t`. */
export interface DenseMultiplyConfig {
  /** Pairs kept per row, the first ones its walk scored; absent keeps every pair. */
  wanted?: number;
  /** Breadth of the candidates, which widens the emitted pattern; absent lets the engine derive it. */
  expansionFactor?: number;
  /** The reading this call overwrites with its own telemetry; absent skips reporting. */
  stats?: DenseTraversalStats;
}

/** Optional settings of one cursor pull. Mirrors `usearch_dense_cursor_next_t`. */
export interface DenseCursorNextConfig {
  /** Whether to export the keys each entry links to beside it; absent pulls the keys alone. */
  links?: boolean;
}

/** Optional per-call settings of attention. Mirrors `usearch_dense_attend_t`. */
export interface DenseAttendConfig {
  /** Query heads sharing the index's key-value head, and with it one walk per token; absent is one. */
  headCount?: number;
  /** The scalar kind the attended rows land as; absent takes the queries' own. */
  scalar?: Scalar;
  /** Breadth of the candidates the walk explores; absent lets the engine derive it. Wider candidates fold more of the tail. */
  expansionFactor?: number;
  /** The logit scale, conventionally the inverse square root of the head dimensionality; absent derives it. */
  scoreScale?: number;
  /**
   * Two floats per query row - the largest logit and the partition taken against it.
   *
   * Supplying one folds this branch into another over the same queries by partial-softmax
   * rescaling; a maximum without its partition is a number nothing can rescale, which is why
   * they share a row rather than travelling as two arrays.
   */
  carry?: Float32Array;
  /** The reading this call overwrites with its own telemetry; absent skips reporting. */
  stats?: DenseTraversalStats;
}

/** Optional settings of a resumable search stream. Mirrors `usearch_dense_search_stream_init_t`. */
export interface DenseSearchStreamInitConfig {
  /** Breadth of the candidates the walk explores; absent lets the engine derive it. */
  expansionFactor?: number;
}

/** Where a checkpoint loaded into host memory lands. Mirrors `usearch_dense_load_ram_t`. */
export interface DenseLoadRamConfig {
  /** The workers the read fans out over, one shard per thread; absent reads on the calling thread. */
  executor?: Executor;
  /** NUMA node IDs to place the parts on; absent spreads them over every node. */
  memoryDomains?: number[];
  /** Entries one key may carry, which no checkpoint records; absent derives to unique keys. */
  valuesPerKey?: number;
}

/** Where a checkpoint loaded into device memory lands. Mirrors `usearch_dense_load_vram_t`. */
export interface DenseLoadVramConfig {
  /** The executor the upload runs on, one stream per device; absent uploads on the default stream. */
  executor?: Executor;
  /** CUDA ordinals to place the parts on; absent takes one free device. */
  devices?: number[];
  /** Entries one key may carry, which no checkpoint records; absent derives to unique keys. */
  valuesPerKey?: number;
}

/** Where a monolith held in memory lands in host memory. Mirrors `usearch_dense_load_ram_buffer_t`. */
export interface DenseLoadRamBufferConfig {
  /** The workers the copy fans out over; absent copies on the calling thread. */
  executor?: Executor;
  /** NUMA node IDs to place the shards on; the monolith lands whole on the first. */
  memoryDomains?: number[];
  /** Entries one key may carry, which no checkpoint records; absent derives to unique keys. */
  valuesPerKey?: number;
}

/** Where a monolith held in memory lands in device memory. Mirrors `usearch_dense_load_vram_buffer_t`. */
export interface DenseLoadVramBufferConfig {
  /** The executor the upload runs on, one stream per device; absent uploads on the default stream. */
  executor?: Executor;
  /** CUDA ordinals to place the shards on; the monolith lands whole on the first. */
  devices?: number[];
  /** Entries one key may carry, which no checkpoint records; absent derives to unique keys. */
  valuesPerKey?: number;
}

/** Where a checkpoint served from NVMe drives is read from. Mirrors `usearch_dense_view_nvme_t`. */
export interface DenseViewNvmeConfig {
  /** The NVMe executor whose rings read the checkpoint's headers. */
  executor?: Executor;
  /** Devices or directories to serve the parts from; absent leaves each where its manifest points. */
  paths?: string[];
}

/** The workers a checkpoint's metadata is read on. Mirrors `usearch_dense_metadata_read_t`. */
export interface DenseMetadataReadConfig {
  /** The executor the read runs on; absent reads on the calling thread. */
  executor?: Executor;
}

/** The workers a monolith's metadata is read on. Mirrors `usearch_dense_metadata_read_buffer_t`. */
export interface DenseMetadataReadBufferConfig {
  /** The executor the read runs on; absent reads on the calling thread. */
  executor?: Executor;
}

/**
 * The geometry and metric a checkpoint froze. Mirrors `usearch_dense_metadata_t`.
 *
 * Where it was written is never recorded, and neither is `valuesPerKey`, which bounds a live
 * handle rather than the file - both stay the caller's to choose when loading.
 */
export interface DenseMetadata {
  /** The metric the graph was built for. */
  metric: DenseMetric;
  /** The scalar type its rows are stored as. */
  quantization: Scalar;
  /** Scalars per stored row. */
  dimensions: number;
  /** Connections per node the graph was built with. */
  connectivity: number;
}

/** A live index's frozen metadata and current counters, each read once. Mirrors `usearch_dense_index_description_t`. */
export interface DenseIndexDescription {
  /** The frozen metric, quantization, dimensions and connectivity, as a checkpoint reports them. */
  metadata: DenseMetadata;
  /** Stored entries. */
  size: number;
  /** Entries hosted without growing again. */
  capacity: number;
  /** Bytes the handle holds; traversal scratch is the executor's. */
  memoryUsage: number;
  /** The CPU class dispatch picked on this machine, as `usearch_cpu_class_name` spells it. */
  cpuClass: string;
  /** The GPU class dispatch picked on the index's devices; "unknown" for a host index. */
  gpuClass: string;
  /** Live shards. */
  shardsCount: number;
}

/** One reachability audit: the population it judged, and how much of it a search reaches. Mirrors `usearch_reachability_t`. */
export interface DenseReachability {
  /** Entries published when the audit started; compare `reachable` against this, never a later size. */
  audited: number;
  /** Of those, the entries a search reaches from the landmarks; the rest stay unfound until a `refine`. */
  reachable: number;
  /** Levels the sweep took to converge. */
  sweeps: number;
  /** Levels the sweep back toward the entry point took, where an audit sweeps both ways. */
  backSweeps: number;
}

/** The geometry every dense index freezes at initialization, whatever its residency. */
export interface DenseOptions {
  /** Scalars per row, fixed for the life of the index. */
  dimensions: number;
  /** How dissimilarity is scored; defaults to `angular`. */
  metric?: DenseMetric;
  /** The scalar kind rows are stored in; defaults to `bf16`. */
  quantization?: Scalar;
  /** Neighbors per node; absent derives it from the geometry. */
  connectivity?: number;
  /** Entries one key may carry; absent derives to unique keys. */
  valuesPerKey?: number;
}

/**
 * Resolves one scored key to its row of the values `attend` borrows, on the calling thread. Backs
 * `usearch_value_lookup_t`.
 *
 * Anything but the index of a row the values hold refuses the call with {@link Status.ShapeMismatch}.
 */
export type ValueRowOf = (key: Key) => number;

/** Lifts the canonical spellings the addon reports into the binding's enumerations. */
function metadataOf(read: CompiledMetadata): DenseMetadata {
  return {
    metric: read.metric as DenseMetric,
    quantization: read.quantization as Scalar,
    dimensions: read.dimensions,
    connectivity: read.connectivity,
  };
}

/**
 * Rows a call landed in its arena: the entries `get` exported, or the rows `attend` attended. Mirrors a `usearch_rows_view_t` output.
 *
 * `data` is a `Float64Array`, `Float32Array`, `Float16Array` or `Int8Array` for the kinds
 * JavaScript spells, a `Uint16Array` for `bf16` and for `f16` on a runtime without `Float16Array`,
 * and the raw bytes in a `Uint8Array` for the rest.
 */
export class DenseVectors extends ArenaView {
  /** Rows written; zero when `get` found no entry under the key. */
  readonly rows: number;
  /** Values in each row, as the call reported them. */
  readonly columns: number;
  /** Bytes between consecutive rows. */
  readonly stride: number;
  /** The scalar kind the rows were exported as. */
  readonly scalar: Scalar;
  readonly #data: Vector | null;

  /** Made by a verb, never directly. */
  constructor(landed: CompiledVectors, scalar: Scalar) {
    super();
    this.#data = landed.data as Vector | null;
    this.rows = landed.rows;
    this.columns = landed.columns;
    this.stride = landed.stride;
    this.scalar = scalar;
  }

  /** Every row, laid `stride` bytes apart. */
  get data(): Vector {
    return this.live(this.#data);
  }
}

/**
 * Every pair a sparse product scored, as a ragged run with Apache Arrow offsets: row `i` spans
 * pairs `offsets[i]` up to `offsets[i + 1]`, in traversal order rather than sorted. `maxsim` lands
 * its documents alike, one run per query, largest first. Mirrors `usearch_matches_tape_view_t`.
 */
export class DenseProducts extends ArenaView {
  /** Left rows the product covers. */
  readonly rows: number;
  /** Pairs scored across every row, or documents across every query, before `wanted` cut any. */
  readonly required: number;
  readonly #landed: CompiledProducts;

  /** Made by `multiply` and `maxsim`, never directly. */
  constructor(landed: CompiledProducts, rows: number) {
    super();
    this.#landed = landed;
    this.rows = rows;
    this.required = landed.required;
  }

  /** One more entry than there are left rows, non-decreasing. */
  get offsets(): BigUint64Array {
    return this.live(this.#landed.offsets);
  }

  /** Sixteen bytes per pair, naming each pair's indexed row. */
  get keys(): Uint8Array {
    return this.live(this.#landed.keys);
  }

  /** One product per pair, positionally paired with `keys`. */
  get scores(): Float64Array {
    return this.live(this.#landed.scores);
  }

  /**
   * Borrows the pairs of one left row, without copying. Binding-only.
   *
   * @param {number} row - Which left row to read.
   * @returns {MatchesRow} - The keys and products that row's walk scored.
   */
  get(row: number): MatchesRow {
    if (!Number.isInteger(row) || row < 0 || row >= this.rows)
      throw fail(Status.InvalidArgument, `No row ${row} in a product of ${this.rows}.`);
    const begin = Number(this.offsets[row]);
    const end = Number(this.offsets[row + 1]);
    return new MatchesRow(this.keys.subarray(begin * keyBytes, end * keyBytes), this.scores.subarray(begin, end));
  }
}

/**
 * The keys each entry of a cursor pull links to, as a ragged run with Apache Arrow offsets: entry
 * `i` links to keys `offsets[i]` up to `offsets[i + 1]`. Mirrors `usearch_keys_tape_view_t`.
 */
export class DenseCursorLinks extends ArenaView {
  /** Entries the pull covers. */
  readonly rows: number;
  readonly #landed: CompiledLinks;

  /** Made by `DenseCursor.next`, never directly. */
  constructor(landed: CompiledLinks, rows: number) {
    super();
    this.#landed = landed;
    this.rows = rows;
  }

  /** One more entry than there are pulled entries, non-decreasing. */
  get offsets(): BigUint64Array {
    return this.live(this.#landed.offsets);
  }

  /** Sixteen bytes per link, end to end. */
  get keys(): Uint8Array {
    return this.live(this.#landed.keys);
  }

  /**
   * Borrows the keys one entry links to, without copying. Binding-only.
   *
   * @param {number} row - Which pulled entry to read.
   * @returns {Uint8Array} - Sixteen bytes per key the entry links to.
   */
  get(row: number): Uint8Array {
    if (!Number.isInteger(row) || row < 0 || row >= this.rows)
      throw fail(Status.InvalidArgument, `No entry ${row} in a pull of ${this.rows}.`);
    const begin = Number(this.offsets[row]);
    const end = Number(this.offsets[row + 1]);
    return this.keys.subarray(begin * keyBytes, end * keyBytes);
  }
}

/** One cursor pull, viewed inside its arena: the keys, and the keys they link to when those were asked for. */
export class DenseCursorBatch extends ArenaView {
  /** Entries pulled; zero once the walk is exhausted. */
  readonly rows: number;
  /** The keys each entry links to; absent when the pull asked for keys alone. */
  readonly links: DenseCursorLinks | undefined;
  readonly #keys: Uint8Array | null;

  /** Made by `DenseCursor.next`, never directly. */
  constructor(landed: CompiledBatch) {
    super();
    this.#keys = landed.keys;
    this.rows = landed.rows;
    this.links = landed.links ? new DenseCursorLinks(landed.links, landed.rows) : undefined;
  }

  /** Sixteen bytes per entry pulled. */
  get keys(): Uint8Array {
    return this.live(this.#keys);
  }
}

/**
 * A walk over an index. Each shard's publication watermark is read as the walk enters it, so
 * a row once passed is never revisited. Mirrors `usearch_dense_cursor_t`.
 *
 * Holds engine state until `free` is called; a walk left to the garbage collector releases it
 * whenever the collector gets to it. It keeps its index alive, and pulls on the executor of the
 * receiver that opened it.
 */
export class DenseCursor implements Disposable {
  readonly #compiled: CompiledCursor;

  /** Made by `cursor`, never directly. */
  constructor(compiled: CompiledCursor) {
    this.#compiled = compiled;
  }

  /**
   * Pulls the next entries of the walk into `arena`; the rows themselves come from `get`.
   * Calls `usearch_dense_cursor_next`.
   *
   * @param {number} wanted - The upper bound on entries this pull lands.
   * @param {Arena} arena - The memory the batch lands in, rewound by the pull.
   * @param {DenseCursorNextConfig} [config] - Whether to export each entry's links; absent pulls keys alone.
   * @return {DenseCursorBatch} The batch; zero rows once the walk is exhausted.
   */
  next(wanted: number, arena: Arena, config: DenseCursorNextConfig = {}): DenseCursorBatch {
    return new DenseCursorBatch(this.#compiled.next(wanted, arena, config.links));
  }

  /** Releases the scan; a second call does nothing. Calls `usearch_dense_cursor_free`. */
  free(): void {
    this.#compiled.free();
  }

  [Symbol.dispose](): void {
    this.free();
  }
}

/**
 * A resumable nearest-neighbor walk over one query, pulled from the caller's own thread. Mirrors
 * `usearch_dense_search_stream_t`.
 *
 * This is the filtered path: pull matches, drop the ones a JavaScript predicate rejects, and
 * pull again until satisfied - rather than guessing a `k` wide enough to survive the filter.
 * The stream keeps walk state rather than a thread, and every pull is an ordinary call on the
 * executor of the receiver that opened it.
 */
export class DenseSearchStream implements Disposable {
  readonly #compiled: CompiledSearchStream;

  /** Made by `searchStream`, never directly. */
  constructor(compiled: CompiledSearchStream) {
    this.#compiled = compiled;
  }

  /**
   * Pulls the next-nearest matches from the stream into `arena`. Calls `usearch_dense_search_stream_next`.
   *
   * @param {number} wanted - The upper bound on matches this pull lands.
   * @param {Arena} arena - The memory the matches land in, rewound by the pull.
   * @return {Matches} One ranked row, whose count is zero once the walk is drained.
   */
  next(wanted: number, arena: Arena): Matches {
    return new Matches(this.#compiled.next(wanted, arena), 1);
  }

  /** Releases the walk state; a second call does nothing. Calls `usearch_dense_search_stream_free`. */
  free(): void {
    this.#compiled.free();
  }

  [Symbol.dispose](): void {
    this.free();
  }
}

/** How a receiver hands back what its verbs answer: as is, or as a {@link Deferred} of it. */
type Answer<Receiving, Answered> = Receiving extends "deferred" ? Deferred<Answered> : Answered;

/**
 * The verbs every receiver of a dense index shares, written once.
 *
 * The receiver chooses where a verb runs and how it completes, and the verb stays the same: a
 * {@link DenseIndex} runs it on the calling thread, `index.on(executor)` blocks on an executor, and
 * `index.deferred(executor)` submits it there and answers a {@link Deferred}.
 */
export abstract class DenseVerbs<Receiving extends "blocking" | "deferred"> {
  /** The native handle every verb is called on; shared by every receiver of one index. */
  protected readonly compiled: CompiledIndex;
  /** The executor the verbs run on; absent runs them on the calling thread. */
  readonly executor: Executor | undefined;

  protected constructor(compiled: CompiledIndex, executor: Executor | undefined) {
    this.compiled = compiled;
    this.executor = executor;
  }

  /** Submits one native call with this receiver's completion, and hands its answer back as this receiver does. */
  protected abstract submit<Raw, Answered>(
    call: (completion: Completion) => unknown,
    answer: (raw: Raw) => Answered,
  ): Answer<Receiving, Answered>;

  /**
   * Claims room for a total entry count ahead of insertion, a hint, since insertions grow the index
   * on demand. Calls `usearch_dense_reserve`.
   * @param {number} capacity - The desired total capacity, the current size included.
   */
  reserve(capacity: number): Answer<Receiving, void> {
    return this.submit(
      (completion) => this.compiled.reserve(this.executor, completion, capacity),
      () => undefined,
    );
  }

  /** Erases every entry, keeping the reserved capacity and the frozen geometry. Calls `usearch_dense_clear`. */
  clear(): Answer<Receiving, void> {
    return this.submit(
      (completion) => this.compiled.clear(this.executor, completion),
      () => undefined,
    );
  }

  /**
   * Reports the frozen metadata and the current counters in one call. Calls `usearch_dense_describe`.
   * @return {DenseIndexDescription} The geometry, the size, the capacity, the memory and the kernels in use.
   */
  describe(): Answer<Receiving, DenseIndexDescription> {
    return this.submit(
      (completion) => this.compiled.describe(this.executor, completion),
      (described: { metadata: CompiledMetadata } & Omit<DenseIndexDescription, "metadata">) => ({
        ...described,
        metadata: metadataOf(described.metadata),
      }),
    );
  }

  /**
   * Audits how much of the graph a search reaches from its landmarks. Calls `usearch_dense_health`.
   * @return {DenseReachability} The population audited, and how much of it is reachable.
   */
  health(): Answer<Receiving, DenseReachability> {
    return this.submit(
      (completion) => this.compiled.health(this.executor, completion),
      (reachability: DenseReachability) => reachability,
    );
  }

  /**
   * Counts the entries stored under one key. Calls `usearch_dense_count`.
   *
   * @param {KeyLike} key - The identifier to be enumerated.
   * @return {number} - How many entries the key holds.
   * @throws Will throw an error if the key is spelled in none of the accepted forms.
   */
  count(key: KeyLike): Answer<Receiving, number> {
    const normalized = normalizeKey(key);
    return this.submit(
      (completion) => this.compiled.count(this.executor, completion, normalized),
      (entries: number) => entries,
    );
  }

  /**
   * Add vectors to the index. Calls `usearch_dense_insert`.
   *
   * A typed array is one row, and a {@link RowsView} is a batch borrowed in place. A single key
   * broadcasts across every row. A batch is all-or-nothing: a rejected one leaves the index
   * untouched.
   *
   * Rows keep whatever type they arrived in and the engine converts them into the index's own
   * quantization, so the two are independent - a packed `u1x8` or a `bf16` row travels as
   * bytes with its `RowsView` naming what they mean.
   *
   * @param {KeysLike} keys - Input identifiers for every vector.
   * @param {RowsView|Vector} vectors - One row, or a run of rows of the index width.
   * @param {DenseInsertConfig} [config] - Per-call breadth and telemetry sink.
   * @throws Will throw an error if the length of keys doesn't match the number of vectors
   *         or if it's not a single key.
   */
  insert(keys: KeysLike, vectors: RowsView | Vector, config: DenseInsertConfig = {}): Answer<Receiving, void> {
    const normalized = normalizeKeys(keys);
    const rows = crossing(vectors);
    return this.submit(
      (completion) =>
        this.compiled.insert(this.executor, completion, normalized, ...rows, config.expansionFactor, config.stats),
      () => undefined,
    );
  }

  /**
   * Re-selects the neighbors of every stored entry against the finished graph. Calls `usearch_dense_refine`.
   *
   * A cheap insertion pass followed by one refinement reaches a better graph, in less total
   * time, than a single expensive insertion pass - so a wide breadth belongs here.
   *
   * @param {DenseRefineConfig} [config] - Per-call breadth and telemetry sink.
   * @return {number} Nodes whose edges changed; zero means the graph has settled.
   */
  refine(config: DenseRefineConfig = {}): Answer<Receiving, number> {
    return this.submit(
      (completion) => this.compiled.refine(this.executor, completion, config.expansionFactor, config.stats),
      (changed: number) => changed,
    );
  }

  /**
   * Searches the index for the nearest neighbours of every query row. Calls `usearch_dense_search`.
   *
   * The matches land in `arena` and come back as a view into it, so a recall sweep reusing one
   * arena allocates once.
   *
   * @param {RowsView|Vector} queries - One query row, or a run of them, at the index's own width.
   * @param {number} wanted - Matches per query, the "k" in "kANN".
   * @param {Arena} arena - The memory the matches land in, rewound by the call.
   * @param {DenseSearchConfig} [config] - Per-call breadth and telemetry sink.
   * @return {Matches} The matches, viewed inside `arena`.
   * @throws Will throw when the queries do not split into whole rows of the index geometry.
   */
  search(
    queries: RowsView | Vector,
    wanted: number,
    arena: Arena,
    config: DenseSearchConfig = {},
  ): Answer<Receiving, Matches> {
    const rows = crossing(queries);
    return this.submit(
      (completion) =>
        this.compiled.search(
          this.executor,
          completion,
          ...rows,
          arena,
          wanted,
          config.expansionFactor,
          config.traversal,
          config.stats,
        ),
      (landed: CompiledMatches) => new Matches(landed, rows[1]),
    );
  }

  /**
   * Late-interaction scoring of multi-row queries: every query row takes its best match within
   * a key, and those maxima sum to that key's score. Calls `usearch_dense_maxsim`.
   *
   * The run is over keys rather than rows, which is what an index built with `valuesPerKey`
   * above one exists for. Runs on the inner-product kinds alone.
   *
   * @param {RowsView|Vector} queries - The token rows of every query, end to end.
   * @param {number} rowsPerQuery - Rows in every query; uniform, so padded queries apply as is.
   * @param {number} wanted - Documents per query, not per token row; zero keeps every document a query scored.
   * @param {Arena} arena - The memory the scored keys land in, rewound by the call.
   * @param {DenseMaxsimConfig} [config] - Per-call breadth and telemetry sink.
   * @return {DenseProducts} The scored keys, one run per query, largest first, viewed inside `arena`.
   * @throws Will throw when the token rows do not split into whole queries.
   */
  maxsim(
    queries: RowsView | Vector,
    rowsPerQuery: number,
    wanted: number,
    arena: Arena,
    config: DenseMaxsimConfig = {},
  ): Answer<Receiving, DenseProducts> {
    const rows = crossing(queries);
    const queriesCount = rows[1] / rowsPerQuery;
    if (!isPositiveInteger(queriesCount)) throw fail(Status.ShapeMismatch, "The rows do not split into whole queries.");
    return this.submit(
      (completion) =>
        this.compiled.maxsim(
          this.executor,
          completion,
          ...rows,
          rowsPerQuery,
          arena,
          wanted,
          config.expansionFactor,
          config.stats,
        ),
      (landed: CompiledProducts) => new DenseProducts(landed, queriesCount),
    );
  }

  /**
   * Retrieves the vectors stored under one key, one row per entry, in insertion order. Calls `usearch_dense_get`.
   *
   * Rows land in `arena` and come back as a view into it, converted into `scalar`, shaped by the
   * width and stride the call reports.
   *
   * @param {KeyLike} key - The identifier whose entries are exported.
   * @param {Arena} arena - The memory the rows land in, rewound by the call.
   * @param {DenseGetConfig} [config] - The kind to export as.
   * @return {DenseVectors} - The rows, laid end to end; empty when the key is absent.
   */
  get(key: KeyLike, arena: Arena, config: DenseGetConfig = {}): Answer<Receiving, DenseVectors> {
    const normalized = normalizeKey(key);
    const scalar = config.scalar ?? Scalar.F32;
    return this.submit(
      (completion) => this.compiled.get(this.executor, completion, normalized, scalarCodes[scalar], arena),
      (landed: CompiledVectors) => new DenseVectors(landed, scalar),
    );
  }

  /**
   * Multiplies a matrix against the index, sparsely - every pair the walk scored, not only the
   * nearest. Calls `usearch_dense_multiply`.
   *
   * How many pairs a walk scores is known only once it ends, so the products land in `arena` and
   * come back as views into it. The scores are unnormalized logits, so the handle must hold the
   * inner-product kind.
   *
   * @param {RowsView|Vector} left - One row, or a run of rows, at the index's own width.
   * @param {Arena} arena - The memory the pairs and their offsets land in, rewound by the call.
   * @param {DenseMultiplyConfig} [config] - Per-call pair bound, breadth and telemetry sink.
   * @return {DenseProducts} - The pairs every left row's walk scored.
   */
  multiply(left: RowsView | Vector, arena: Arena, config: DenseMultiplyConfig = {}): Answer<Receiving, DenseProducts> {
    const rows = crossing(left);
    return this.submit(
      (completion) =>
        this.compiled.multiply(
          this.executor,
          completion,
          ...rows,
          arena,
          config.wanted,
          config.expansionFactor,
          config.stats,
        ),
      (landed: CompiledProducts) => new DenseProducts(landed, rows[1]),
    );
  }

  /**
   * Absorbs every donor into this index, one after another, keeping their graphs rather than
   * re-inserting their vectors and linking each in across the appended part. Calls `usearch_dense_absorb`.
   *
   * A donor the library consumed refuses every later verb with {@link Status.Uninitialized}; one it
   * did not absorb, like a donor of another scalar refused with {@link Status.ConfigMismatch}, stays
   * a live index the caller still owns, though its rows may already have landed here. Until a
   * deferred absorb is final, every donor it names refuses every verb, and no other absorb can name it.
   *
   * @param {DenseIndex[]} donors - The indexes to absorb.
   */
  absorb(donors: DenseIndex[]): Answer<Receiving, void> {
    const handles = donors.map((donor) => donor.compiled);
    return this.submit(
      (completion) => this.compiled.absorb(this.executor, completion, handles),
      () => undefined,
    );
  }

  /**
   * Saves the index in the layout the path extension picks - one `.safegraphs` monolith, or a
   * `.safegraphs.json` manifest beside its shards. Calls `usearch_dense_save`.
   *
   * @param {string} path File path to write.
   * @throws Will throw an error if `path` is not a string.
   */
  save(path: string): Answer<Receiving, void> {
    return this.submit(
      (completion) => this.compiled.save(this.executor, completion, path),
      () => undefined,
    );
  }

  /**
   * Writes the index into memory as one monolith. Calls `usearch_dense_save_buffer`.
   *
   * The bytes are what a checkpoint holds, so a service that keeps its indexes in an object
   * store never touches the filesystem. They land in `arena` and come back as a view into it.
   *
   * @param {Arena} arena - The memory the monolith lands in, rewound by the call.
   * @return {Monolith} - One serialized monolith.
   */
  saveBuffer(arena: Arena): Answer<Receiving, Monolith> {
    return this.submit(
      (completion) => this.compiled.saveBuffer(this.executor, completion, arena),
      (bytes: Uint8Array | null) => new Monolith(bytes),
    );
  }
}

/**
 * The verbs of one index, blocking on one executor: `index.on(executor)`. Every call starts after
 * every earlier submission to that executor, and returns with its outputs written.
 */
export class DenseIndexOn extends DenseVerbs<"blocking"> {
  /** Made by `index.on(executor)`, never directly. */
  constructor(compiled: CompiledIndex, executor: Executor | undefined) {
    super(compiled, executor);
  }

  protected submit<Raw, Answered>(call: (completion: Completion) => unknown, answer: (raw: Raw) => Answered): Answered {
    return answer(call(0) as Raw);
  }

  /**
   * Verifies the presence of one key in the index. Binding-only sugar over `count`.
   *
   * @param {KeyLike} key - The identifier to be checked for presence in the index.
   * @return {boolean} - Whether the key is present.
   */
  contains(key: KeyLike): boolean {
    return this.count(key) !== 0;
  }

  /**
   * Opens a walk over the current contents, every pull on this receiver's executor. Without
   * `starts` it yields every row as stored; with them, the start keys' own rows first in order,
   * then the rows those link to breadth-first, each once. Calls `usearch_dense_cursor_init`.
   * @param {KeysLike} [starts] - The keys to walk from; absent walks every row.
   * @return {DenseCursor} - The walk, which should be freed once pulling stops.
   */
  cursor(starts?: KeysLike): DenseCursor {
    return new DenseCursor(
      this.compiled.cursor(this.executor, starts === undefined ? undefined : normalizeKeys(starts)),
    );
  }

  /**
   * Opens a resumable nearest-neighbor walk over one query row, every pull on this receiver's
   * executor. Calls `usearch_dense_search_stream_init`.
   *
   * @param {RowsView|Vector} query - A single query row of the index geometry.
   * @param {DenseSearchStreamInitConfig} [config] - The breadth.
   * @return {DenseSearchStream} - The stream, whose walk state lives until it is freed.
   */
  searchStream(query: RowsView | Vector, config: DenseSearchStreamInitConfig = {}): DenseSearchStream {
    return new DenseSearchStream(this.compiled.searchStream(this.executor, ...crossing(query), config.expansionFactor));
  }
}

/**
 * The verbs of one index, submitted to one executor and answered as {@link Deferred} calls:
 * `index.deferred(executor)`. Cursors, streams and `contains` stay blocking, so this receiver
 * offers none of them.
 *
 * A verb with outputs borrows its arena as the blocking verb does, and waits into the same view.
 */
export class DenseIndexDeferredOn extends DenseVerbs<"deferred"> {
  /** Made by `index.deferred(executor)`, never directly. */
  constructor(compiled: CompiledIndex, executor: Executor) {
    super(compiled, executor);
  }

  protected submit<Raw, Answered>(
    call: (completion: Completion) => unknown,
    answer: (raw: Raw) => Answered,
  ): Deferred<Answered> {
    return new Deferred(call(1) as CompiledDeferred, answer);
  }
}

/** The token the lifecycle verbs hold, which is what a direct `new DenseIndex` cannot produce. */
const indexSeal = Symbol("DenseIndex");

/**
 * A dense vector index, running every verb on the calling thread. `on(executor)` and
 * `deferred(executor)` reach the same verbs on an executor, all but `attend`, whose `rowOf` runs here.
 */
export class DenseIndex extends DenseIndexOn implements Disposable {
  /** Reached through the `init*`, `load*` and `view*` lifecycle verbs. */
  private constructor(seal: symbol, compiledIndex: CompiledIndex) {
    if (seal !== indexSeal)
      throw fail(
        Status.InvalidArgument,
        "A DenseIndex is opened by DenseIndex.initRam, DenseIndex.initVram, or a DenseIndex.load or view verb.",
      );
    super(compiledIndex, undefined);
  }

  /**
   * The same verbs, each blocking on `executor`, after every earlier submission to it. Binding-only.
   * @param {Executor} executor - The workers to run on; its residency must match the index's.
   */
  on(executor: Executor): DenseIndexOn {
    return new DenseIndexOn(this.compiled, executor);
  }

  /**
   * The same verbs, each submitted to `executor` and answered as a {@link Deferred}. Binding-only.
   * @param {Executor} executor - The workers to run on; its residency must match the index's.
   */
  deferred(executor: Executor): DenseIndexDeferredOn {
    return new DenseIndexDeferredOn(this.compiled, executor);
  }

  /**
   * Softmax attention of every query row over every key its graph walk scores, folded as Flash
   * Attention folds key tiles, landing one attended row per query in `arena`. Calls `usearch_dense_attend`.
   *
   * The index stores no values: every key a walk scores resolves through `rowOf` to a row of
   * `values`, borrowed for the call, so only the keys a walk scores are ever resolved. The values
   * must be as wide as the keys and stored in the index's own scalar kind, and a key resolving
   * outside them refuses the call with `ShapeMismatch`, since a missing row is a wrong softmax
   * rather than a smaller one. Row `t * headCount + h` of the queries, outputs and carry is head
   * `h` of token `t`, every head of a token sharing one walk.
   *
   * `rowOf` runs on the calling thread, which no executor's thread is, so only the index itself
   * offers `attend`. The first exception it throws stops every later lookup and is rethrown once
   * the call returns. Being a host function, it serves RAM handles alone: a VRAM walk resolves rows
   * through a device function, which this binding does not build, so a `Vram` arena refuses with
   * {@link Status.Device}.
   *
   * @param {RowsView|Vector} queries - One query row, or a run of them, at the index's own width.
   * @param {RowsView|Vector} values - The value rows the scored keys resolve into, never copied.
   * @param {ValueRowOf} rowOf - Answers one 16-byte key with the index of its row of `values`.
   * @param {Arena} arena - The memory the attended rows land in, rewound by the call.
   * @param {DenseAttendConfig} [config] - Head count, output kind, softmax shape, the carry, and the telemetry sink.
   * @return {DenseVectors} One key-width row per query, viewed inside `arena`.
   */
  attend(
    queries: RowsView | Vector,
    values: RowsView | Vector,
    rowOf: ValueRowOf,
    arena: Arena,
    config: DenseAttendConfig = {},
  ): DenseVectors {
    const landed = this.compiled.attend(
      ...crossing(queries),
      ...crossing(values),
      rowOf,
      arena,
      config.scalar === undefined ? undefined : scalarCodes[config.scalar],
      config.carry,
      config.headCount,
      config.expansionFactor,
      config.scoreScale,
      config.stats,
    );
    return new DenseVectors(landed, config.scalar ?? rowsOf(queries).scalar);
  }

  /** Releases the handle now, waiting for every deferred call naming it; a freed index refuses every later call, and a second `free` does nothing. Calls `usearch_dense_free`. */
  free(): void {
    this.compiled.free();
  }

  [Symbol.dispose](): void {
    this.free();
  }

  /**
   * Creates an empty, mutable index in host memory, one shard per named NUMA node. Calls `usearch_dense_init_ram`.
   *
   * @param {DenseOptions} options - The geometry.
   * @param {number[]} [memoryDomains] - NUMA node IDs to shard over; absent spreads the shards over every node.
   * @return {DenseIndex} An index a RAM executor's verbs run on.
   * @throws Will throw an error if any option is of incorrect type or invalid value.
   */
  static initRam(options: DenseOptions, memoryDomains?: number[]): DenseIndex {
    return new DenseIndex(
      indexSeal,
      compiled.CompiledIndex.initRam(
        options.dimensions,
        options.metric ?? DenseMetric.Angular,
        scalarCodes[options.quantization ?? Scalar.BF16],
        options.connectivity,
        options.valuesPerKey,
        memoryDomains,
      ),
    );
  }

  /**
   * Creates an empty, mutable index in device memory, one shard per named CUDA ordinal. Calls `usearch_dense_init_vram`.
   *
   * @param {DenseOptions} options - The geometry.
   * @param {number[]} [devices] - CUDA ordinals to shard over; absent takes one free device.
   * @return {DenseIndex} An index a VRAM executor's verbs run on.
   * @throws Will throw an error if any option is of incorrect type or invalid value.
   */
  static initVram(options: DenseOptions, devices?: number[]): DenseIndex {
    return new DenseIndex(
      indexSeal,
      compiled.CompiledIndex.initVram(
        options.dimensions,
        options.metric ?? DenseMetric.Angular,
        scalarCodes[options.quantization ?? Scalar.BF16],
        options.connectivity,
        options.valuesPerKey,
        devices,
      ),
    );
  }

  /**
   * Copies a checkpoint into a new, mutable index in host memory, part `p` landing on shard `p`
   * modulo the shard count. The geometry and metric come from the checkpoint itself.
   * Calls `usearch_dense_load_ram`.
   *
   * @param {string} path - The manifest of a sharded checkpoint, or a single monolith.
   * @param {DenseLoadRamConfig} [config] - The executor, the memory domains, and `valuesPerKey`.
   * @return {DenseIndex} The loaded index.
   */
  static loadRam(path: string, config: DenseLoadRamConfig = {}): DenseIndex {
    return new DenseIndex(
      indexSeal,
      compiled.CompiledIndex.loadRam(path, config.executor, config.memoryDomains, config.valuesPerKey),
    );
  }

  /**
   * Copies a checkpoint into a new, mutable index in device memory, with the placement rules of
   * {@link DenseIndex.loadRam}. Calls `usearch_dense_load_vram`.
   *
   * @param {string} path - The manifest of a sharded checkpoint, or a single monolith.
   * @param {DenseLoadVramConfig} [config] - The executor, the devices, and `valuesPerKey`.
   * @return {DenseIndex} The loaded index.
   */
  static loadVram(path: string, config: DenseLoadVramConfig = {}): DenseIndex {
    return new DenseIndex(
      indexSeal,
      compiled.CompiledIndex.loadVram(path, config.executor, config.devices, config.valuesPerKey),
    );
  }

  /**
   * Copies a monolith held in memory into a new, mutable index in host memory. Calls `usearch_dense_load_ram_buffer`.
   *
   * @param {Uint8Array|Monolith} buffer - The bytes {@link DenseVerbs.saveBuffer} produced.
   * @param {DenseLoadRamBufferConfig} [config] - The executor, the memory domains, and `valuesPerKey`.
   * @return {DenseIndex} The loaded index.
   */
  static loadRamBuffer(buffer: Uint8Array | Monolith, config: DenseLoadRamBufferConfig = {}): DenseIndex {
    const bytes = buffer instanceof Monolith ? buffer.bytes : buffer;
    return new DenseIndex(
      indexSeal,
      compiled.CompiledIndex.loadRamBuffer(bytes, config.executor, config.memoryDomains, config.valuesPerKey),
    );
  }

  /**
   * Copies a monolith held in memory into a new, mutable index in device memory. Calls `usearch_dense_load_vram_buffer`.
   *
   * @param {Uint8Array|Monolith} buffer - The bytes {@link DenseVerbs.saveBuffer} produced.
   * @param {DenseLoadVramBufferConfig} [config] - The executor, the devices, and `valuesPerKey`.
   * @return {DenseIndex} The loaded index.
   */
  static loadVramBuffer(buffer: Uint8Array | Monolith, config: DenseLoadVramBufferConfig = {}): DenseIndex {
    const bytes = buffer instanceof Monolith ? buffer.bytes : buffer;
    return new DenseIndex(
      indexSeal,
      compiled.CompiledIndex.loadVramBuffer(bytes, config.executor, config.devices, config.valuesPerKey),
    );
  }

  /**
   * Serves a checkpoint in place through a memory mapping, as a read-only index whose mutating
   * verbs throw with {@link Status.ReadOnly}; the files must stay unmodified while it lives.
   * Calls `usearch_dense_view_ram`.
   *
   * @param {string} path - The manifest of a sharded checkpoint, or a single monolith.
   * @return {DenseIndex} The read-only index.
   */
  static viewRam(path: string): DenseIndex {
    return new DenseIndex(indexSeal, compiled.CompiledIndex.viewRam(path));
  }

  /**
   * Serves a checkpoint in place from NVMe drives, read through the executor's submission rings,
   * as a read-only index. Calls `usearch_dense_view_nvme`.
   *
   * @param {string} path - The manifest of a sharded checkpoint, or a single monolith.
   * @param {DenseViewNvmeConfig} [config] - The NVMe executor, and the paths to serve the parts from.
   * @return {DenseIndex} The read-only index.
   */
  static viewNvme(path: string, config: DenseViewNvmeConfig = {}): DenseIndex {
    return new DenseIndex(indexSeal, compiled.CompiledIndex.viewNvme(path, config.executor, config.paths));
  }

  /**
   * Reads a checkpoint's geometry and metric without loading it. Calls `usearch_dense_metadata_read`.
   *
   * @param {string} path - The manifest of a sharded checkpoint, or a single monolith.
   * @param {DenseMetadataReadConfig} [config] - The workers the read runs on.
   * @return {DenseMetadata} The geometry and metric the checkpoint froze.
   */
  static metadataRead(path: string, config: DenseMetadataReadConfig = {}): DenseMetadata {
    return metadataOf(compiled.CompiledIndex.metadataRead(path, config.executor));
  }

  /**
   * Reads the geometry and metric of a monolith held in memory. Calls `usearch_dense_metadata_read_buffer`.
   *
   * @param {Uint8Array|Monolith} buffer - The bytes {@link DenseVerbs.saveBuffer} produced.
   * @param {DenseMetadataReadBufferConfig} [config] - The workers the read runs on.
   * @return {DenseMetadata} The geometry and metric the monolith froze.
   */
  static metadataReadBuffer(buffer: Uint8Array | Monolith, config: DenseMetadataReadBufferConfig = {}): DenseMetadata {
    const bytes = buffer instanceof Monolith ? buffer.bytes : buffer;
    return metadataOf(compiled.CompiledIndex.metadataReadBuffer(bytes, config.executor));
  }
}
