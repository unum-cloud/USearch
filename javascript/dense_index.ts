/**
 *  @file javascript/dense_index.ts
 *  @author Ash Vardanian
 *  @date September 9, 2026
 *  @brief Vector search over equidimensional dense embeddings, the receivers its verbs run through, and
 *      the cursor and resumable search stream one index opens.
 */
import {
  Completion,
  CompiledBatch,
  CompiledCursor,
  CompiledDeferred,
  CompiledDraws,
  CompiledIndex,
  CompiledLinks,
  CompiledMatches,
  CompiledMetadata,
  CompiledPhaseProgress,
  CompiledProducts,
  CompiledRefineReport,
  CompiledSearchStream,
  CompiledVectors,
  compiled,
} from "./library";
import {
  Arena,
  Deferred,
  Executor,
  Key,
  KeyLike,
  KeysLike,
  Matches,
  QueryMatches,
  Monolith,
  RowsView,
  Scalar,
  Status,
  Traversal,
  Vector,
  crossing,
  fail,
  keyBytes,
  normalizeKey,
  normalizeKeys,
  rowsOf,
  scalarCodes,
} from "./types";

/**
 *  Enumeration representing the metric kinds used to score the dissimilarity of two vectors.
 *  @enum {string}
 *  @readonly
 */
export enum DenseMetric {
  /** The inner product. */
  Inner = "inner",
  /** The cosine distance. */
  Angular = "angular",
  /** The L2 distance. */
  Euclidean = "euclidean",
  /** The count of differing bits. */
  Hamming = "hamming",
  /** The bit-level Jaccard distance. */
  Jaccard = "jaccard",
}

/** Traversal telemetry from one call, a pure output the call fully overwrites.
 *
 *  One reading handed to several calls holds the last of them, so a sweep that wants totals folds
 *  them with `accumulate` rather than expecting the calls to sum themselves. */
export class DenseTraversalStats {
  /** Entries whose neighborhoods the walk expanded. */
  visitedMembers = 0;
  /** Distance evaluations the walk paid for, diversity re-selections included. */
  computedDistances = 0;
  /** Times the walk fell back to a fresh entry point after the frontier drained early. */
  restarts = 0;
  /** Hint keys no shard held an entry under; zero unless a search was handed hints. */
  missingHints = 0;

  /** Folds another reading into this one, which is how a sweep gathers totals. Binding-only. */
  accumulate(other: DenseTraversalStats): void {
    this.visitedMembers += other.visitedMembers;
    this.computedDistances += other.computedDistances;
    this.restarts += other.restarts;
    this.missingHints += other.missingHints;
  }
}

/** Optional per-call settings of an insertion. */
export interface DenseInsertConfig {
  /** Breadth of the candidates the build walk explores; absent lets the engine derive it. */
  expansionFactor?: number;
  /** The reading this call overwrites with its own telemetry; absent skips reporting. */
  stats?: DenseTraversalStats;
}

/** Optional per-call settings of the refinement pass that follows an insertion. */
export interface DenseRefineConfig {
  /** Breadth of any walk beside the round; absent lets the engine derive it. */
  expansionFactor?: number;
  /** Bytes a loaded NVMe index holds for the one part it rewrites: absent rounds that part in RAM,
   *  and a smaller one streams the part's rounds from the drives. */
  memoryLimit?: number;
  /** The reading this call overwrites with its own telemetry; absent skips reporting. */
  stats?: DenseTraversalStats;
  /** The plan this call continues, running its next round from the same root; absent runs one
   *  round alone. */
  plan?: DenseRefinePlan;
}

/** Optional placements and settings of a reshard; an absent placement keeps the handle's own. */
export interface DenseReshardConfig {
  /** NUMA node IDs of a RAM placement, one shard per node at least; empty derives them. */
  memoryDomains?: number[];
  /** Device ordinals of a VRAM placement, one shard per device at least. */
  devices?: number[];
  /** Drives of an NVMe placement, one shard per drive at least. */
  paths?: string[];
  /** Entries per shard of the RAM or VRAM placement; absent keeps the handle's own bound. */
  shardCapacity?: number;
  /** Picks the part roots of a source cut among several targets: equal seeds over an unchanged
   *  source cut equally. */
  seed?: number | bigint;
  /** Breadth of any walk beside the closing rounds; absent lets the engine derive it. */
  expansionFactor?: number;
  /** Bytes a loaded NVMe index holds for the one part it rewrites: absent rounds that part in RAM,
   *  and a smaller one streams the part's rounds from the drives. */
  memoryLimit?: number;
  /** The reading this call overwrites with its own telemetry; absent skips reporting. */
  stats?: DenseTraversalStats;
  /** The plan this call continues; absent moves and runs to the end. */
  plan?: DenseReshardPlan;
}

/** Optional per-call settings of a sample; absent ones draw across the whole index
 *  with nothing shown. */
export interface DenseSampleConfig {
  /** The keys already shown: never drawn again, and the paths reach toward them; read in place,
   *  never copied. */
  shown?: KeysLike;
  /** Where each `shown` key sits, three values apiece, end to end; read in place. */
  shownCoordinates?: Float32Array;
  /** The keys to look around; absent looks around the whole index. */
  focus?: KeysLike;
  /** Equal seeds, shown keys and focus over an unchanged index draw equally. */
  seed?: number | bigint;
  /** Breadth of the candidates of the walks behind the draws; absent lets the engine derive it. */
  expansionFactor?: number;
}

/** Optional per-call settings of a search. */
export interface DenseSearchConfig {
  /** Breadth of the candidates the walk explores; absent lets the engine derive it, and a scan ignores it. */
  expansionFactor?: number;
  /** Walks the graph when absent; `Traversal.Scan` scores every entry exactly, on RAM and NVMe handles. */
  traversal?: Traversal;
  /** The reading this call overwrites with its own telemetry; absent skips reporting. */
  stats?: DenseTraversalStats;
}

/** Optional per-call settings of late-interaction scoring. */
export interface DenseMaxsimConfig {
  /** Breadth of the candidates the walk explores; absent lets the engine derive it. */
  expansionFactor?: number;
  /** The reading this call overwrites with its own telemetry; absent skips reporting. */
  stats?: DenseTraversalStats;
}

/** Optional settings of a retrieval. */
export interface DenseGetConfig {
  /** The scalar kind to export as, converted from storage; defaults to `f32`. */
  scalar?: Scalar;
}

/** Optional per-call settings of a sparse product. */
export interface DenseMultiplyConfig {
  /** Pairs kept per left row, the first ones its walk scored; absent keeps every pair. */
  wanted?: number;
  /** Breadth of the candidates, which widens the emitted pattern; absent lets the engine derive it. */
  expansionFactor?: number;
  /** The reading this call overwrites with its own telemetry; absent skips reporting. */
  stats?: DenseTraversalStats;
}

/** Optional settings of one cursor pull. */
export interface DenseCursorNextConfig {
  /** Whether to export the keys each entry links to beside it; absent pulls the keys alone. */
  links?: boolean;
}

/** Optional per-call settings of attention. */
export interface DenseAttendConfig {
  /** Query heads sharing the index's key-value head, and with it one walk per token; absent is one. */
  headCount?: number;
  /** The scalar kind the attended rows land as; absent takes the queries' own. */
  scalar?: Scalar;
  /** Breadth of the candidates the walk explores; absent lets the engine derive it. Wider candidates fold more of the tail. */
  expansionFactor?: number;
  /** The logit scale, conventionally the inverse square root of the head dimensionality; absent derives it. */
  scoreScale?: number;
  /** Two floats per query row - the largest logit and the partition taken against it.
   *
   *  Supplying one folds this branch into another over the same queries by partial-softmax
   *  rescaling; a maximum without its partition is a number nothing can rescale, which is why
   *  they share a row rather than travelling as two arrays. */
  carry?: Float32Array;
  /** The reading this call overwrites with its own telemetry; absent skips reporting. */
  stats?: DenseTraversalStats;
}

/** Optional settings of a resumable search stream. */
export interface DenseSearchStreamInitConfig {
  /** Breadth of the candidates the walk explores; absent lets the engine derive it. */
  expansionFactor?: number;
}

/** Where a checkpoint loaded into host memory lands. */
export interface DenseLoadRamConfig {
  /** The workers the read fans out over, one shard per thread; absent reads on the calling thread. */
  executor?: Executor;
  /** NUMA node IDs to place the parts on; absent spreads them over every node. */
  memoryDomains?: number[];
  /** The CPU class to dispatch as by its canonical name; absent dispatches as the machine. */
  cpuClass?: string;
  /** Entries one key may carry, which no checkpoint records; absent is unbounded. */
  valuesPerKey?: number;
}

/** Where a checkpoint loaded into device memory lands. */
export interface DenseLoadVramConfig {
  /** The executor the upload runs on, one stream per device; absent uploads on the default stream. */
  executor?: Executor;
  /** CUDA ordinals to place the parts on; absent takes one free device. */
  devices?: number[];
  /** The GPU class to dispatch as by its canonical name; absent dispatches as the devices. */
  gpuClass?: string;
  /** Entries one key may carry, which no checkpoint records; absent is unbounded. */
  valuesPerKey?: number;
}

/** Where a monolith held in memory lands in host memory. */
export interface DenseLoadRamBufferConfig {
  /** The workers the copy fans out over; absent copies on the calling thread. */
  executor?: Executor;
  /** NUMA node IDs to place the shards on; the monolith lands whole on the first. */
  memoryDomains?: number[];
  /** The CPU class to dispatch as by its canonical name; absent dispatches as the machine. */
  cpuClass?: string;
  /** Entries one key may carry, which no checkpoint records; absent is unbounded. */
  valuesPerKey?: number;
}

/** Where a monolith held in memory lands in device memory. */
export interface DenseLoadVramBufferConfig {
  /** The executor the upload runs on, one stream per device; absent uploads on the default stream. */
  executor?: Executor;
  /** CUDA ordinals to place the shards on; the monolith lands whole on the first. */
  devices?: number[];
  /** The GPU class to dispatch as by its canonical name; absent dispatches as the devices. */
  gpuClass?: string;
  /** Entries one key may carry, which no checkpoint records; absent is unbounded. */
  valuesPerKey?: number;
}

/** Where a checkpoint served from NVMe drives is read from. */
export interface DenseViewNvmeConfig {
  /** The NVMe executor whose rings read the checkpoint's headers. */
  executor?: Executor;
  /** Devices or directories to serve the parts from; absent leaves each where its manifest points. */
  paths?: string[];
  /** The CPU class to dispatch as by its canonical name; absent dispatches as the machine. */
  cpuClass?: string;
  /** NUMA node IDs of a key table held in RAM, which `count`, `get`, `sample`, `maxsim` and hinted
   *  searches read; empty derives them, and absent holds none. */
  keyTableDomains?: number[];
}

/** The workers a checkpoint's metadata is read on. */
export interface DenseMetadataReadConfig {
  /** The executor the read runs on; absent reads on the calling thread. */
  executor?: Executor;
}

/** The workers a monolith's metadata is read on. */
export interface DenseMetadataReadBufferConfig {
  /** The executor the read runs on; absent reads on the calling thread. */
  executor?: Executor;
}

/** The geometry and metric a checkpoint froze.
 *
 *  Where it was written is never recorded, and neither is `valuesPerKey`, which bounds a live
 *  handle rather than the file - both stay the caller's to choose when loading. */
export interface DenseMetadata {
  /** The metric the graph was built for. */
  metric: DenseMetric;
  /** The scalar type its vectors are stored as. */
  quantization: Scalar;
  /** Scalars per stored vector. */
  dimensions: number;
  /** Connections per node the graph was built with. */
  connectivity: number;
}

/** A live index's frozen metadata and current counters, each read once. */
export interface DenseIndexDescription {
  /** The frozen metric, quantization, dimensions and connectivity, as a checkpoint reports them. */
  metadata: DenseMetadata;
  /** Stored entries. */
  size: number;
  /** Entries hosted without growing again. */
  capacity: number;
  /** Bytes the handle holds; traversal scratch is the executor's. */
  memoryUsage: number;
  /** The CPU class dispatch picked on this machine by its canonical name. */
  cpuClass: string;
  /** The GPU class dispatch picked on the index's devices; "unknown" for a host index. */
  gpuClass: string;
  /** Live shards. */
  shardsCount: number;
  /** How the last load from a path read the parts: "none", "host-copy" or "cufile". */
  loadPath: string;
}

/** One reachability audit: the population it judged, and how much of it a search reaches. */
export interface DenseReachability {
  /** Entries published when the audit started; compare `reachable` against this, never a later size. */
  audited: number;
  /** Of those, the entries a search reaches from the landmarks; the rest stay unfound until a `refine`. */
  reachable: number;
  /** Depths the walk from the root took to converge. */
  depthFromRoot: number;
  /** Depths the walk back toward the root took, where a health check walks both ways. */
  depthToRoot: number;
  /** Entries no reached entry links to. */
  unlinked: number;
  /** Entries linked to by fewer entries than they link to. */
  underlinked: number;
  /** Entries deeper than the depth holding the most entries. */
  beyondMostPopulatedDepth: number;
  /** Islands no entry outside them links into, so no search from the root reaches them. */
  unreachedIslands: number;
  /** Islands that link to no entry outside them, so no search from inside them leaves. */
  strandedIslands: number;
  /** Components when direction is ignored. */
  components: number;
}

/** What a refine round concluded, as the library spells it, so no caller picks a change rate:
 *  "none" when no round filled the report yet, "improved" when another round can help, "converged"
 *  when it changed no link set, "stalled" when full entries leave no slot, and "stopped" when the
 *  caller stopped it. */
export type RefineVerdict = "none" | "improved" | "converged" | "stalled" | "stopped";

/** What one `refine` or `reshard` did to the graph, or a whole plan did once finished. */
export interface DenseRefineReport {
  /** Entries whose neighbors were selected again, the islands of every heal round included. */
  reselected: number;
  /** Of those, the entries whose neighbor set changed, any link added or dropped. */
  changed: number;
  /** Entries the first audit found not reached both ways. */
  islandsFound: number;
  /** Links written between an island and an entry the audit reached both ways. */
  islandsReconnected: number;
  /** Neighborhoods the round re-stitched. */
  neighborhoods: number;
  /** Entries of the widest neighborhood. */
  widestNeighborhood: number;
  /** Links the round added. */
  addedLinks: number;
  /** Links the round dropped. */
  droppedLinks: number;
  /** What the round concluded. */
  verdict: RefineVerdict;
  /** The health check opening the round. */
  opening: DenseReachability;
  /** The health check closing the heal, as a fresh `health` right after would take it. */
  reachability: DenseReachability;
}

/** The geometry every dense index freezes at initialization, whatever its residency. */
export interface DenseOptions {
  /** Scalars per row, fixed for the life of the index. */
  dimensions: number;
  /** How dissimilarity is scored; defaults to `angular`. */
  metric?: DenseMetric;
  /** The scalar kind vectors are stored in; defaults to `bf16`. */
  quantization?: Scalar;
  /** Neighbors per node; absent derives it from the geometry. */
  connectivity?: number;
  /** Entries one key may carry; absent is unbounded. */
  valuesPerKey?: number;
  /** The CPU class a host index dispatches as by its canonical name; absent dispatches
   *  as the machine. */
  cpuClass?: string;
  /** The GPU class a device index dispatches as by its canonical name; absent dispatches
   *  as the devices. */
  gpuClass?: string;
}

/** Resolves one scored key to its row of the values `attend` borrows, on the calling thread.
 *
 *  Anything but the index of a row the values hold refuses the call with
 *  {@link Status.ShapeMismatch}; an exception it throws goes to the process's `uncaughtException`
 *  handlers, and its key resolves to no row. */
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

/** Rows a call wrote into its arena: the entries `get` exported, or the rows `attend` attended.
 *
 *  `data` is a `Float64Array`, `Float32Array`, `Float16Array` or `Int8Array` for the kinds
 *  JavaScript spells, a `Uint16Array` for `bf16` and for `f16` on a runtime without `Float16Array`,
 *  and the raw bytes in a `Uint8Array` for the rest. */
export class DenseVectors {
  /** Every row, laid `stride` bytes apart. */
  readonly data: Vector;
  /** Rows written; zero when `get` found no entry under the key. */
  readonly rows: number;
  /** Values in each row, as the call reported them. */
  readonly columns: number;
  /** Bytes between consecutive rows. */
  readonly stride: number;
  /** The scalar kind the rows were exported as. */
  readonly scalar: Scalar;

  /** Made by a verb, never directly. */
  constructor(outputs: CompiledVectors, scalar: Scalar) {
    this.data = outputs.data as Vector;
    this.rows = outputs.rows;
    this.columns = outputs.columns;
    this.stride = outputs.stride;
    this.scalar = scalar;
  }
}

/** Every pair a sparse product scored, as a ragged run with Apache Arrow offsets: left row `i`
 *  spans pairs `offsets[i]` up to `offsets[i + 1]`, in traversal order rather than sorted. `maxsim`
 *  lands its documents alike, one run per query, largest first. */
export class DenseProducts {
  /** Left rows the product covers, or queries `maxsim` scored. */
  readonly queries: number;
  /** Pairs scored across every left row, or documents across every query, before `wanted` cut any. */
  readonly required: number;
  /** One more offset than there are queries, non-decreasing. */
  readonly offsets: BigUint64Array;
  /** Sixteen bytes per pair, naming each pair's indexed entry. */
  readonly keys: Uint8Array;
  /** One product per pair, positionally paired with `keys`. */
  readonly scores: Float64Array;

  /** Made by `multiply` and `maxsim`, never directly. */
  constructor(outputs: CompiledProducts) {
    this.queries = outputs.offsets.length - 1;
    this.required = outputs.required;
    this.offsets = outputs.offsets;
    this.keys = outputs.keys;
    this.scores = outputs.scores;
  }

  /**
   *  Borrows the pairs of one left row or the documents of one query, without copying. Binding-only.
   *
   *  @param query - Which left row or query to read.
   *  @returns The keys and products that query's walk scored.
   */
  get(query: number): QueryMatches {
    if (!Number.isInteger(query) || query < 0 || query >= this.queries)
      throw fail(Status.InvalidArgument, `No query ${query} in a product of ${this.queries}.`);
    const begin = Number(this.offsets[query]);
    const end = Number(this.offsets[query + 1]);
    return new QueryMatches(this.keys.subarray(begin * keyBytes, end * keyBytes), this.scores.subarray(begin, end));
  }
}

/** The keys each entry of a cursor pull links to, as a ragged run with Apache Arrow offsets: entry
 *  `i` links to keys `offsets[i]` up to `offsets[i + 1]`. */
export class DenseCursorLinks {
  /** Entries the pull covers. */
  readonly entries: number;
  /** One more entry than there are pulled entries, non-decreasing. */
  readonly offsets: BigUint64Array;
  /** Sixteen bytes per link, end to end. */
  readonly keys: Uint8Array;

  /** Made by `DenseCursor.next`, never directly. */
  constructor(outputs: CompiledLinks, entries: number) {
    this.entries = entries;
    this.offsets = outputs.offsets;
    this.keys = outputs.keys;
  }

  /**
   *  Borrows the keys one entry links to, without copying. Binding-only.
   *
   *  @param entry - Which pulled entry to read.
   *  @returns Sixteen bytes per key the entry links to.
   */
  get(entry: number): Uint8Array {
    if (!Number.isInteger(entry) || entry < 0 || entry >= this.entries)
      throw fail(Status.InvalidArgument, `No entry ${entry} in a pull of ${this.entries}.`);
    const begin = Number(this.offsets[entry]);
    const end = Number(this.offsets[entry + 1]);
    return this.keys.subarray(begin * keyBytes, end * keyBytes);
  }
}

/** The draws a sample wrote, viewed inside its arena: each draw's key, which of that key's entries
 *  it is, the shown or earlier draw it hangs from, and where it is placed. */
export class DenseDraws {
  /** Draws written; fewer than wanted only once the region is exhausted. */
  readonly draws: number;
  /** Each draw's parent and its distance, one column per draw, a root holding none. */
  readonly parents: Matches;
  /** Where each draw is placed, one row of three `f32` values apiece. */
  readonly coordinates: DenseVectors;
  /** Sixteen bytes per draw. */
  readonly keys: Uint8Array;
  /** Which of its key's entries each draw is, in the order `get` writes them. */
  readonly entryOrdinals: BigUint64Array;

  /** Made by `sample`, never directly. */
  constructor(outputs: CompiledDraws) {
    this.draws = outputs.draws;
    this.parents = new Matches(outputs.parents, outputs.draws);
    this.coordinates = new DenseVectors(outputs.coordinates, Scalar.F32);
    this.keys = outputs.keys;
    this.entryOrdinals = outputs.entryOrdinals;
  }
}

/** One cursor pull, viewed inside its arena: the keys, and the keys they link to when those were asked for. */
export class DenseCursorBatch {
  /** Entries pulled; zero once the walk is exhausted. */
  readonly entries: number;
  /** Sixteen bytes per entry pulled. */
  readonly keys: Uint8Array;
  /** The keys each entry links to; absent when the pull asked for keys alone. */
  readonly links: DenseCursorLinks | undefined;

  /** Made by `DenseCursor.next`, never directly. */
  constructor(outputs: CompiledBatch) {
    this.entries = outputs.entries;
    this.keys = outputs.keys;
    this.links = outputs.links ? new DenseCursorLinks(outputs.links, outputs.entries) : undefined;
  }
}

/** A walk over an index. Each shard's publication watermark is read as the walk enters it, so an
 *  entry once passed is never revisited.
 *
 *  Holds engine state until `free` is called; a walk left to the garbage collector releases it
 *  whenever the collector gets to it. It keeps its index alive, and pulls on the executor of the
 *  receiver that opened it. */
export class DenseCursor implements Disposable {
  readonly #compiled: CompiledCursor;

  /** Made by `cursor`, never directly. */
  constructor(compiled: CompiledCursor) {
    this.#compiled = compiled;
  }

  /**
   *  Pulls the next entries of the walk into `arena`; the vectors themselves come from `get`.
   *
   *  @param wanted - The upper bound on entries this pull lands.
   *  @param arena - The memory the batch lands in, rewound by the pull.
   *  @param config - Whether to export each entry's links; absent pulls keys alone.
   *  @returns The batch; zero entries once the walk is exhausted.
   */
  next(wanted: number, arena: Arena, config: DenseCursorNextConfig = {}): DenseCursorBatch {
    return new DenseCursorBatch(this.#compiled.next(wanted, arena, config.links));
  }

  /** Releases the scan; a second call does nothing. */
  free(): void {
    this.#compiled.free();
  }

  [Symbol.dispose](): void {
    this.free();
  }
}

/** A resumable nearest-neighbor walk over one query, pulled from the caller's own thread.
 *
 *  This is the filtered path: pull matches, drop the ones a JavaScript predicate rejects, and
 *  pull again until satisfied - rather than guessing a `k` wide enough to survive the filter.
 *  The stream keeps walk state rather than a thread, and every pull is an ordinary call on the
 *  executor of the receiver that opened it. */
export class DenseSearchStream implements Disposable {
  readonly #compiled: CompiledSearchStream;

  /** Made by `searchStream`, never directly. */
  constructor(compiled: CompiledSearchStream) {
    this.#compiled = compiled;
  }

  /**
   *  Pulls the next-nearest matches from the stream into `arena`.
   *
   *  @param wanted - The upper bound on matches this pull lands.
   *  @param arena - The memory the matches land in, rewound by the pull.
   *  @returns One query's ranked matches, whose count is zero once the walk is drained.
   */
  next(wanted: number, arena: Arena): Matches {
    return new Matches(this.#compiled.next(wanted, arena), 1);
  }

  /** Releases the walk state; a second call does nothing. */
  free(): void {
    this.#compiled.free();
  }

  [Symbol.dispose](): void {
    this.free();
  }
}

/** Where a plan's last call stopped: the phase by its canonical name, its position among the phases
 *  the plan runs, and that phase's units. */
export type DensePhaseProgress = CompiledPhaseProgress;

/** One refine continued across calls: each `refine` naming it runs the next round from the root
 *  the first one drew, and a receiver's `refinePlanFinish` closes it. The index admits no insertion
 *  and no save while the plan lives, and every call names the index and executor that opened it;
 *  a plan dropped unfinished leaks and keeps the index closed. */
export class DenseRefinePlan {
  /** The memory the library writes the plan into. */
  readonly state: ArrayBuffer = compiled.CompiledIndex.plan();

  /** Where the last call stopped, "none" before the first and "done" once its round ended. */
  get phase(): string {
    return this.progress.phase;
  }

  /** The phase the last call stopped in, with its position and its units. */
  get progress(): DensePhaseProgress {
    return compiled.CompiledIndex.planProgress(this.state);
  }
}

/** One reshard continued across calls: each `reshard` naming it moves what is still pending and
 *  continues the rounds, and a receiver's `reshardPlanFinish` closes it. Every index it names
 *  admits no insertion and no save while the plan lives, and every call names the executor that
 *  opened it; a plan dropped unfinished leaks and keeps them closed. */
export class DenseReshardPlan {
  /** The memory the library writes the plan into. */
  readonly state: ArrayBuffer = compiled.CompiledIndex.plan();

  /** Where the last call stopped, "none" before the first and "done" once its rounds ended. */
  get phase(): string {
    return this.progress.phase;
  }

  /** The phase the last call stopped in, with its position and its units. */
  get progress(): DensePhaseProgress {
    return compiled.CompiledIndex.planProgress(this.state);
  }
}

/** One verb's native call, given the completion its receiver picks, and how its answer lifts. */
interface DenseCall<Answered> {
  readonly run: (completion: Completion) => unknown;
  readonly answer: (raw: never) => Answered;
}

/** Runs a call to its end on the receiver's thread or executor, answering its lifted result. */
function block<Answered>(call: DenseCall<Answered>): Answered {
  return (call.answer as (raw: unknown) => Answered)(call.run(0));
}

/** Submits a call to the receiver's executor, answering the {@link Deferred} that owns it. */
function defer<Answered>(call: DenseCall<Answered>): Deferred<Answered> {
  return new Deferred(call.run(1) as CompiledDeferred, call.answer);
}

function denseCall<Raw, Answered>(
  run: (completion: Completion) => unknown,
  answer: (raw: Raw) => Answered,
): DenseCall<Answered> {
  return { run, answer };
}

const answerNothing = (): void => undefined;
const answerReport = (report: CompiledRefineReport): DenseRefineReport => report as DenseRefineReport;

function denseReserve(index: CompiledIndex, executor: Executor | undefined, capacity: number): DenseCall<void> {
  return denseCall((completion) => index.reserve(executor, completion, capacity), answerNothing);
}

function denseClear(index: CompiledIndex, executor: Executor | undefined): DenseCall<void> {
  return denseCall((completion) => index.clear(executor, completion), answerNothing);
}

function denseDescribe(index: CompiledIndex, executor: Executor | undefined): DenseCall<DenseIndexDescription> {
  return denseCall(
    (completion) => index.describe(executor, completion),
    (described: { metadata: CompiledMetadata } & Omit<DenseIndexDescription, "metadata">) => ({
      ...described,
      metadata: metadataOf(described.metadata),
    }),
  );
}

function denseHealth(index: CompiledIndex, executor: Executor | undefined): DenseCall<DenseReachability> {
  return denseCall(
    (completion) => index.health(executor, completion),
    (reachability: DenseReachability) => reachability,
  );
}

function denseCount(index: CompiledIndex, executor: Executor | undefined, key: KeyLike): DenseCall<number> {
  const normalized = normalizeKey(key);
  return denseCall(
    (completion) => index.count(executor, completion, normalized),
    (entries: number) => entries,
  );
}

function denseInsert(
  index: CompiledIndex,
  executor: Executor | undefined,
  keys: KeysLike,
  vectors: RowsView | Vector,
  config: DenseInsertConfig,
): DenseCall<void> {
  const normalized = normalizeKeys(keys);
  const rows = crossing(vectors);
  return denseCall(
    (completion) => index.insert(executor, completion, normalized, ...rows, config.expansionFactor, config.stats),
    answerNothing,
  );
}

function denseRefine(
  index: CompiledIndex,
  executor: Executor | undefined,
  config: DenseRefineConfig,
): DenseCall<DenseRefineReport> {
  return denseCall(
    (completion) =>
      index.refine(executor, completion, config.expansionFactor, config.memoryLimit, config.stats, config.plan?.state),
    answerReport,
  );
}

function denseRefinePlanFinish(
  index: CompiledIndex,
  executor: Executor | undefined,
  plan: DenseRefinePlan,
): DenseCall<DenseRefineReport> {
  return denseCall((completion) => index.refinePlanFinish(executor, completion, plan.state), answerReport);
}

function denseSearch(
  index: CompiledIndex,
  executor: Executor | undefined,
  queries: RowsView | Vector,
  wanted: number,
  arena: Arena,
  config: DenseSearchConfig,
): DenseCall<Matches> {
  const rows = crossing(queries);
  return denseCall(
    (completion) =>
      index.search(
        executor,
        completion,
        ...rows,
        arena,
        wanted,
        config.expansionFactor,
        config.traversal,
        config.stats,
      ),
    (outputs: CompiledMatches) => new Matches(outputs, rows[1]),
  );
}

function denseMaxsim(
  index: CompiledIndex,
  executor: Executor | undefined,
  queries: RowsView | Vector,
  rowsPerQuery: number,
  wanted: number,
  arena: Arena,
  config: DenseMaxsimConfig,
): DenseCall<DenseProducts> {
  const rows = crossing(queries);
  return denseCall(
    (completion) =>
      index.maxsim(executor, completion, ...rows, rowsPerQuery, arena, wanted, config.expansionFactor, config.stats),
    (outputs: CompiledProducts) => new DenseProducts(outputs),
  );
}

function denseGet(
  index: CompiledIndex,
  executor: Executor | undefined,
  key: KeyLike,
  arena: Arena,
  config: DenseGetConfig,
): DenseCall<DenseVectors> {
  const normalized = normalizeKey(key);
  const scalar = config.scalar ?? Scalar.F32;
  return denseCall(
    (completion) => index.get(executor, completion, normalized, scalarCodes[scalar], arena),
    (outputs: CompiledVectors) => new DenseVectors(outputs, scalar),
  );
}

function denseMultiply(
  index: CompiledIndex,
  executor: Executor | undefined,
  left: RowsView | Vector,
  arena: Arena,
  config: DenseMultiplyConfig,
): DenseCall<DenseProducts> {
  const rows = crossing(left);
  return denseCall(
    (completion) =>
      index.multiply(executor, completion, ...rows, arena, config.wanted, config.expansionFactor, config.stats),
    (outputs: CompiledProducts) => new DenseProducts(outputs),
  );
}

function denseReshard(
  executor: Executor | undefined,
  sources: DenseIndex[],
  targets: DenseIndex[],
  config: DenseReshardConfig,
): DenseCall<DenseRefineReport> {
  return denseCall(
    (completion) =>
      compiled.CompiledIndex.reshard(
        executor,
        completion,
        sources.map((source) => source.compiled),
        targets.map((target) => target.compiled),
        config.memoryDomains,
        config.devices,
        config.paths,
        config.shardCapacity,
        config.seed,
        config.expansionFactor,
        config.memoryLimit,
        config.stats,
        config.plan?.state,
      ),
    answerReport,
  );
}

function denseReshardPlanFinish(executor: Executor | undefined, plan: DenseReshardPlan): DenseCall<DenseRefineReport> {
  return denseCall(
    (completion) => compiled.CompiledIndex.reshardPlanFinish(executor, completion, plan.state),
    answerReport,
  );
}

function denseSample(
  index: CompiledIndex,
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
    (completion) =>
      index.sample(executor, completion, arena, wanted, shown, ...places, focus, config.seed, config.expansionFactor),
    (outputs: CompiledDraws) => new DenseDraws(outputs),
  );
}

function denseSave(index: CompiledIndex, executor: Executor | undefined, path: string): DenseCall<void> {
  return denseCall((completion) => index.save(executor, completion, path), answerNothing);
}

function denseSaveBuffer(index: CompiledIndex, executor: Executor | undefined, arena: Arena): DenseCall<Monolith> {
  return denseCall(
    (completion) => index.saveBuffer(executor, completion, arena),
    (bytes: Uint8Array) => new Monolith(bytes),
  );
}

/** The verbs of one index, blocking on one executor: `index.on(executor)`. Every call returns with
 *  its outputs written, or contended while another call holds that executor. */
export class DenseIndexOn {
  readonly #index: CompiledIndex;
  /** The executor every verb runs on. */
  readonly executor: Executor;

  /** Made by `index.on(executor)`, never directly. */
  constructor(index: CompiledIndex, executor: Executor) {
    this.#index = index;
    this.executor = executor;
  }

  /** {@inheritDoc DenseIndex.reserve} */
  reserve(capacity: number): void {
    return block(denseReserve(this.#index, this.executor, capacity));
  }

  /** {@inheritDoc DenseIndex.clear} */
  clear(): void {
    return block(denseClear(this.#index, this.executor));
  }

  /** {@inheritDoc DenseIndex.describe} */
  describe(): DenseIndexDescription {
    return block(denseDescribe(this.#index, this.executor));
  }

  /** {@inheritDoc DenseIndex.health} */
  health(): DenseReachability {
    return block(denseHealth(this.#index, this.executor));
  }

  /** {@inheritDoc DenseIndex.count} */
  count(key: KeyLike): number {
    return block(denseCount(this.#index, this.executor, key));
  }

  /** {@inheritDoc DenseIndex.contains} */
  contains(key: KeyLike): boolean {
    return this.count(key) !== 0;
  }

  /** {@inheritDoc DenseIndex.insert} */
  insert(keys: KeysLike, vectors: RowsView | Vector, config: DenseInsertConfig = {}): void {
    return block(denseInsert(this.#index, this.executor, keys, vectors, config));
  }

  /** {@inheritDoc DenseIndex.refine} */
  refine(config: DenseRefineConfig = {}): DenseRefineReport {
    return block(denseRefine(this.#index, this.executor, config));
  }

  /** {@inheritDoc DenseIndex.refinePlanFinish} */
  refinePlanFinish(plan: DenseRefinePlan): DenseRefineReport {
    return block(denseRefinePlanFinish(this.#index, this.executor, plan));
  }

  /** {@inheritDoc DenseIndex.search} */
  search(queries: RowsView | Vector, wanted: number, arena: Arena, config: DenseSearchConfig = {}): Matches {
    return block(denseSearch(this.#index, this.executor, queries, wanted, arena, config));
  }

  /** {@inheritDoc DenseIndex.maxsim} */
  maxsim(
    queries: RowsView | Vector,
    rowsPerQuery: number,
    wanted: number,
    arena: Arena,
    config: DenseMaxsimConfig = {},
  ): DenseProducts {
    return block(denseMaxsim(this.#index, this.executor, queries, rowsPerQuery, wanted, arena, config));
  }

  /** {@inheritDoc DenseIndex.get} */
  get(key: KeyLike, arena: Arena, config: DenseGetConfig = {}): DenseVectors {
    return block(denseGet(this.#index, this.executor, key, arena, config));
  }

  /** {@inheritDoc DenseIndex.multiply} */
  multiply(left: RowsView | Vector, arena: Arena, config: DenseMultiplyConfig = {}): DenseProducts {
    return block(denseMultiply(this.#index, this.executor, left, arena, config));
  }

  /** {@inheritDoc DenseIndex.reshard} */
  reshard(sources: DenseIndex[], targets: DenseIndex[], config: DenseReshardConfig = {}): DenseRefineReport {
    return block(denseReshard(this.executor, sources, targets, config));
  }

  /** {@inheritDoc DenseIndex.reshardPlanFinish} */
  reshardPlanFinish(plan: DenseReshardPlan): DenseRefineReport {
    return block(denseReshardPlanFinish(this.executor, plan));
  }

  /** {@inheritDoc DenseIndex.sample} */
  sample(wanted: number, arena: Arena, config: DenseSampleConfig = {}): DenseDraws {
    return block(denseSample(this.#index, this.executor, wanted, arena, config));
  }

  /** {@inheritDoc DenseIndex.save} */
  save(path: string): void {
    return block(denseSave(this.#index, this.executor, path));
  }

  /** {@inheritDoc DenseIndex.saveBuffer} */
  saveBuffer(arena: Arena): Monolith {
    return block(denseSaveBuffer(this.#index, this.executor, arena));
  }

  /** {@inheritDoc DenseIndex.cursor} */
  cursor(starts?: KeysLike): DenseCursor {
    return new DenseCursor(this.#index.cursor(this.executor, starts === undefined ? undefined : normalizeKeys(starts)));
  }

  /** {@inheritDoc DenseIndex.searchStream} */
  searchStream(query: RowsView | Vector, config: DenseSearchStreamInitConfig = {}): DenseSearchStream {
    return new DenseSearchStream(this.#index.searchStream(this.executor, ...crossing(query), config.expansionFactor));
  }
}

/** The verbs of one index, submitted to one executor and answered as {@link Deferred} calls:
 *  `index.deferred(executor)`. Cursors, streams and `contains` stay blocking, so this receiver
 *  offers none of them.
 *
 *  A verb with outputs borrows its arena as the blocking one does and waits into the same view. */
export class DenseIndexDeferredOn {
  readonly #index: CompiledIndex;
  /** The executor every verb is submitted to. */
  readonly executor: Executor;

  /** Made by `index.deferred(executor)`, never directly. */
  constructor(index: CompiledIndex, executor: Executor) {
    this.#index = index;
    this.executor = executor;
  }

  /** {@inheritDoc DenseIndex.reserve} */
  reserve(capacity: number): Deferred<void> {
    return defer(denseReserve(this.#index, this.executor, capacity));
  }

  /** {@inheritDoc DenseIndex.clear} */
  clear(): Deferred<void> {
    return defer(denseClear(this.#index, this.executor));
  }

  /** {@inheritDoc DenseIndex.describe} */
  describe(): Deferred<DenseIndexDescription> {
    return defer(denseDescribe(this.#index, this.executor));
  }

  /** {@inheritDoc DenseIndex.health} */
  health(): Deferred<DenseReachability> {
    return defer(denseHealth(this.#index, this.executor));
  }

  /** {@inheritDoc DenseIndex.count} */
  count(key: KeyLike): Deferred<number> {
    return defer(denseCount(this.#index, this.executor, key));
  }

  /** {@inheritDoc DenseIndex.insert} */
  insert(keys: KeysLike, vectors: RowsView | Vector, config: DenseInsertConfig = {}): Deferred<void> {
    return defer(denseInsert(this.#index, this.executor, keys, vectors, config));
  }

  /** {@inheritDoc DenseIndex.refine} */
  refine(config: DenseRefineConfig = {}): Deferred<DenseRefineReport> {
    return defer(denseRefine(this.#index, this.executor, config));
  }

  /** {@inheritDoc DenseIndex.refinePlanFinish} */
  refinePlanFinish(plan: DenseRefinePlan): Deferred<DenseRefineReport> {
    return defer(denseRefinePlanFinish(this.#index, this.executor, plan));
  }

  /** {@inheritDoc DenseIndex.search} */
  search(queries: RowsView | Vector, wanted: number, arena: Arena, config: DenseSearchConfig = {}): Deferred<Matches> {
    return defer(denseSearch(this.#index, this.executor, queries, wanted, arena, config));
  }

  /** {@inheritDoc DenseIndex.maxsim} */
  maxsim(
    queries: RowsView | Vector,
    rowsPerQuery: number,
    wanted: number,
    arena: Arena,
    config: DenseMaxsimConfig = {},
  ): Deferred<DenseProducts> {
    return defer(denseMaxsim(this.#index, this.executor, queries, rowsPerQuery, wanted, arena, config));
  }

  /** {@inheritDoc DenseIndex.get} */
  get(key: KeyLike, arena: Arena, config: DenseGetConfig = {}): Deferred<DenseVectors> {
    return defer(denseGet(this.#index, this.executor, key, arena, config));
  }

  /** {@inheritDoc DenseIndex.multiply} */
  multiply(left: RowsView | Vector, arena: Arena, config: DenseMultiplyConfig = {}): Deferred<DenseProducts> {
    return defer(denseMultiply(this.#index, this.executor, left, arena, config));
  }

  /** {@inheritDoc DenseIndex.reshard} */
  reshard(sources: DenseIndex[], targets: DenseIndex[], config: DenseReshardConfig = {}): Deferred<DenseRefineReport> {
    return defer(denseReshard(this.executor, sources, targets, config));
  }

  /** {@inheritDoc DenseIndex.reshardPlanFinish} */
  reshardPlanFinish(plan: DenseReshardPlan): Deferred<DenseRefineReport> {
    return defer(denseReshardPlanFinish(this.executor, plan));
  }

  /** {@inheritDoc DenseIndex.sample} */
  sample(wanted: number, arena: Arena, config: DenseSampleConfig = {}): Deferred<DenseDraws> {
    return defer(denseSample(this.#index, this.executor, wanted, arena, config));
  }

  /** {@inheritDoc DenseIndex.save} */
  save(path: string): Deferred<void> {
    return defer(denseSave(this.#index, this.executor, path));
  }

  /** {@inheritDoc DenseIndex.saveBuffer} */
  saveBuffer(arena: Arena): Deferred<Monolith> {
    return defer(denseSaveBuffer(this.#index, this.executor, arena));
  }
}

/** The token the lifecycle verbs hold, which is what a direct `new DenseIndex` cannot produce. */
const indexSeal = Symbol("DenseIndex");

/** A dense vector index, running every verb on the calling thread. `on(executor)` and
 *  `deferred(executor)` reach the same verbs on an executor, all but `attend`, whose `rowOf`
 *  runs here. */
export class DenseIndex implements Disposable {
  /** The native handle every verb names. @internal */
  readonly compiled: CompiledIndex;

  /** Reached through the `init*`, `load*` and `view*` lifecycle verbs. */
  private constructor(seal: symbol, compiledIndex: CompiledIndex) {
    if (seal !== indexSeal)
      throw fail(
        Status.InvalidArgument,
        "A DenseIndex is opened by DenseIndex.initRam, DenseIndex.initVram, or a DenseIndex.load or view verb.",
      );
    this.compiled = compiledIndex;
  }

  /**
   *  The same verbs, blocking on `executor` and refused while another call holds it. Binding-only.
   *  @param executor - The workers to run on; its residency must match the index's.
   */
  on(executor: Executor): DenseIndexOn {
    return new DenseIndexOn(this.compiled, executor);
  }

  /**
   *  The same verbs, each submitted to `executor` and answered as a {@link Deferred}. Binding-only.
   *  @param executor - The workers to run on; its residency must match the index's.
   */
  deferred(executor: Executor): DenseIndexDeferredOn {
    return new DenseIndexDeferredOn(this.compiled, executor);
  }

  /**
   *  Claims room for a total entry count ahead of insertion, a hint, since insertions grow the
   *  index on demand.
   *  @param capacity - The desired total capacity, the current size included.
   */
  reserve(capacity: number): void {
    return block(denseReserve(this.compiled, undefined, capacity));
  }

  /** Erases every entry, keeping the reserved capacity and the frozen geometry. */
  clear(): void {
    return block(denseClear(this.compiled, undefined));
  }

  /**
   *  Reports the frozen metadata and the current counters in one call.
   *  @returns The geometry, the size, the capacity, the memory and the kernels in use.
   */
  describe(): DenseIndexDescription {
    return block(denseDescribe(this.compiled, undefined));
  }

  /**
   *  Audits how much of the graph a search reaches from its landmarks.
   *  @returns The population audited, and how much of it is reachable.
   */
  health(): DenseReachability {
    return block(denseHealth(this.compiled, undefined));
  }

  /**
   *  Counts the entries stored under one key.
   *
   *  @param key - The identifier to be enumerated.
   *  @returns How many entries the key holds.
   *  @throws Will throw an error if the key is spelled in none of the accepted forms.
   */
  count(key: KeyLike): number {
    return block(denseCount(this.compiled, undefined, key));
  }

  /**
   *  Verifies the presence of one key in the index. Binding-only sugar over `count`.
   *
   *  @param key - The identifier to be checked for presence in the index.
   *  @returns Whether the key is present.
   */
  contains(key: KeyLike): boolean {
    return this.count(key) !== 0;
  }

  /**
   *  Add vectors to the index.
   *
   *  A typed array is one row, and a {@link RowsView} is a batch borrowed in place. A single key
   *  broadcasts across every row. A batch is all-or-nothing: a rejected one leaves the index
   *  untouched.
   *
   *  Rows keep whatever type they arrived in and the engine converts them into the index's own
   *  quantization, so the two are independent - a packed `u1x8` or a `bf16` row travels as
   *  bytes with its `RowsView` naming what they mean.
   *
   *  @param keys - Input identifiers for every vector.
   *  @param vectors - One row, or a run of rows of the index width.
   *  @param config - Per-call breadth and telemetry sink.
   *  @throws Will throw an error if the length of keys doesn't match the number of vectors
   *          or if it's not a single key.
   */
  insert(keys: KeysLike, vectors: RowsView | Vector, config: DenseInsertConfig = {}): void {
    return block(denseInsert(this.compiled, undefined, keys, vectors, config));
  }

  /**
   *  Re-selects the neighbors of every stored entry against the finished graph.
   *
   *  A cheap insertion pass followed by one refinement reaches a better graph, in less total
   *  time, than a single expensive insertion pass - so a wide breadth belongs here.
   *
   *  @param config - Per-call breadth, NVMe workspace, telemetry sink and plan.
   *  @returns The entries re-selected and changed, the islands healed, the links added and
   *      dropped, and the `verdict`; loop while it is "improved".
   */
  refine(config: DenseRefineConfig = {}): DenseRefineReport {
    return block(denseRefine(this.compiled, undefined, config));
  }

  /**
   *  Closes a refine plan on the index and executor it was opened with, those of this receiver,
   *  and frees it whatever the call answers; a round a stop left open closes as a stop would.
   *  @param plan - The plan to close.
   *  @returns What the whole plan did to the graph.
   */
  refinePlanFinish(plan: DenseRefinePlan): DenseRefineReport {
    return block(denseRefinePlanFinish(this.compiled, undefined, plan));
  }

  /**
   *  Searches the index for the nearest neighbours of every query row.
   *
   *  The matches land in `arena` and come back as a view into it, so a recall sweep reusing one
   *  arena allocates once.
   *
   *  @param queries - One query row, or a run of them, at the index's own width.
   *  @param wanted - Matches per query, the "k" in "kANN".
   *  @param arena - The memory the matches land in, rewound by the call.
   *  @param config - Per-call breadth and telemetry sink.
   *  @returns The matches, viewed inside `arena`.
   *  @throws Will throw when the queries do not split into whole rows of the index geometry.
   */
  search(queries: RowsView | Vector, wanted: number, arena: Arena, config: DenseSearchConfig = {}): Matches {
    return block(denseSearch(this.compiled, undefined, queries, wanted, arena, config));
  }

  /**
   *  Late-interaction scoring of multi-row queries: every query row takes its best match within a
   *  key, and those maxima sum to that key's score.
   *
   *  The run is over keys rather than rows, which is what an index built with `valuesPerKey`
   *  above one exists for. Runs on the inner-product kinds alone.
   *
   *  @param queries - The token rows of every query, end to end.
   *  @param rowsPerQuery - Rows in every query; uniform, so padded queries apply as is.
   *  @param wanted - Documents per query, not per token row; zero keeps every document a query scored.
   *  @param arena - The memory the scored keys land in, rewound by the call.
   *  @param config - Per-call breadth and telemetry sink.
   *  @returns The scored keys, one run per query, largest first, viewed inside `arena`.
   */
  maxsim(
    queries: RowsView | Vector,
    rowsPerQuery: number,
    wanted: number,
    arena: Arena,
    config: DenseMaxsimConfig = {},
  ): DenseProducts {
    return block(denseMaxsim(this.compiled, undefined, queries, rowsPerQuery, wanted, arena, config));
  }

  /**
   *  Retrieves the vectors stored under one key, one row per entry, in insertion order.
   *
   *  Rows land in `arena` and come back as a view into it, converted into `scalar`, shaped by the
   *  width and stride the call reports.
   *
   *  @param key - The identifier whose entries are exported.
   *  @param arena - The memory the rows land in, rewound by the call.
   *  @param config - The kind to export as.
   *  @returns The rows, laid end to end; empty when the key is absent.
   */
  get(key: KeyLike, arena: Arena, config: DenseGetConfig = {}): DenseVectors {
    return block(denseGet(this.compiled, undefined, key, arena, config));
  }

  /**
   *  Multiplies a matrix against the index, sparsely - every pair the walk scored, not
   *  only the nearest.
   *
   *  How many pairs a walk scores is known only once it ends, so the products land in `arena` and
   *  come back as views into it. The scores are unnormalized logits, so the handle must hold the
   *  inner-product kind.
   *
   *  @param left - One row, or a run of rows, at the index's own width.
   *  @param arena - The memory the pairs and their offsets land in, rewound by the call.
   *  @param config - Per-call pair bound, breadth and telemetry sink.
   *  @returns The pairs every left row's walk scored.
   */
  multiply(left: RowsView | Vector, arena: Arena, config: DenseMultiplyConfig = {}): DenseProducts {
    return block(denseMultiply(this.compiled, undefined, left, arena, config));
  }

  /**
   *  Moves the entries of `sources` into `targets` on this receiver's executor, keeping their links
   *  rather than re-inserting their vectors, then runs a round on every target that took entries.
   *
   *  Several sources into one target, itself among them, gather into it; one source into several
   *  empty targets cuts it into graph-local parts, one per target; one index as its only source and
   *  target recuts its own shards in place into as few as fit the placement's shard capacity. Every
   *  index stays the caller's: a source the targets do not name ends empty and live. Every handle
   *  is checked before the first entry moves, so a refusal moves nothing.
   *
   *  @param sources - The indexes whose entries move, each named once.
   *  @param targets - The indexes the entries move into, each a source itself or empty.
   *  @param config - The placement, seed, breadth, NVMe workspace, telemetry sink and plan.
   *  @returns What the closing rounds did to the graph.
   */
  reshard(sources: DenseIndex[], targets: DenseIndex[], config: DenseReshardConfig = {}): DenseRefineReport {
    return block(denseReshard(undefined, sources, targets, config));
  }

  /**
   *  Closes a reshard plan on the executor it was opened with, this receiver's, and frees it
   *  whatever the call answers, reopening every index it names.
   *  @param plan - The plan to close.
   *  @returns What the whole plan did to the graph.
   */
  reshardPlanFinish(plan: DenseReshardPlan): DenseRefineReport {
    return block(denseReshardPlanFinish(undefined, plan));
  }

  /**
   *  Draws keys around `config.focus`, or across the whole index, each hanging from a shown or
   *  earlier draw and placed in three dimensions beside the shown ones.
   *
   *  A larger `wanted` draws a superset, so a view grows one connected sample as it zooms. Only RAM
   *  samples yet.
   *
   *  @param wanted - The upper bound on draws.
   *  @param arena - The memory the draws land in, rewound by the call.
   *  @param config - The shown keys and their coordinates, the focus, the seed and the breadth.
   *  @returns The draws, viewed inside `arena`.
   */
  sample(wanted: number, arena: Arena, config: DenseSampleConfig = {}): DenseDraws {
    return block(denseSample(this.compiled, undefined, wanted, arena, config));
  }

  /**
   *  Saves the index in the layout the path extension picks - one `.safegraphs` monolith, or a
   *  `.safegraphs.json` manifest beside its shards.
   *
   *  @param path - File path to write.
   *  @throws Will throw an error if `path` is not a string.
   */
  save(path: string): void {
    return block(denseSave(this.compiled, undefined, path));
  }

  /**
   *  Writes the index into memory as one monolith.
   *
   *  The bytes are what a checkpoint holds, so a service that keeps its indexes in an object
   *  store never touches the filesystem. They land in `arena` and come back as a view into it.
   *
   *  @param arena - The memory the monolith lands in, rewound by the call.
   *  @returns One serialized monolith.
   */
  saveBuffer(arena: Arena): Monolith {
    return block(denseSaveBuffer(this.compiled, undefined, arena));
  }

  /**
   *  Opens a walk over the current contents, every pull on this receiver's executor. Without
   *  `starts` it yields every entry as stored; with them, the start keys' own entries first in
   *  order, then the entries those link to breadth-first, each once.
   *  @param starts - The keys to walk from; absent walks every entry.
   *  @returns The walk, which should be freed once pulling stops.
   */
  cursor(starts?: KeysLike): DenseCursor {
    return new DenseCursor(this.compiled.cursor(undefined, starts === undefined ? undefined : normalizeKeys(starts)));
  }

  /**
   *  Opens a resumable nearest-neighbor walk over one query row, every pull on
   *  this receiver's executor.
   *
   *  @param query - A single query row of the index geometry.
   *  @param config - The breadth.
   *  @returns The stream, whose walk state lives until it is freed.
   */
  searchStream(query: RowsView | Vector, config: DenseSearchStreamInitConfig = {}): DenseSearchStream {
    return new DenseSearchStream(this.compiled.searchStream(undefined, ...crossing(query), config.expansionFactor));
  }

  /**
   *  Softmax attention of every query row over every key its graph walk scores, folded as Flash
   *  Attention folds key tiles, writing one attended row per query in `arena`.
   *
   *  The index stores no values: every key a walk scores resolves through `rowOf` to a row of
   *  `values`, borrowed for the call, so only the keys a walk scores are ever resolved. The values
   *  must be as wide as the keys and stored in the index's own scalar kind, and a key resolving
   *  outside them refuses the call with `ShapeMismatch`, since a missing row is a wrong softmax
   *  rather than a smaller one. Row `t * headCount + h` of the queries, outputs and carry is head
   *  `h` of token `t`, every head of a token sharing one walk.
   *
   *  `rowOf` runs on the calling thread, which no executor's thread is, so only the index itself
   *  offers `attend`. An exception it throws never unwinds through the library: it reaches the
   *  process's `uncaughtException` handlers, and its key resolves to no row, which refuses the
   *  call. Being a host function, it serves RAM handles alone: a VRAM walk resolves rows through a
   *  device function, which this binding does not build, so a `Vram` arena refuses with
   *  {@link Status.Device}.
   *
   *  @param queries - One query row, or a run of them, at the index's own width.
   *  @param values - The value rows the scored keys resolve into, never copied.
   *  @param rowOf - Answers one 16-byte key with the index of its row of `values`.
   *  @param arena - The memory the attended rows land in, rewound by the call.
   *  @param config - Head count, output kind, softmax shape, the carry, and the telemetry sink.
   *  @returns One key-width row per query, viewed inside `arena`.
   */
  attend(
    queries: RowsView | Vector,
    values: RowsView | Vector,
    rowOf: ValueRowOf,
    arena: Arena,
    config: DenseAttendConfig = {},
  ): DenseVectors {
    const outputs = this.compiled.attend(
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
    return new DenseVectors(outputs, config.scalar ?? rowsOf(queries).scalar);
  }

  /** Releases the handle now, waiting for every deferred call naming it; a freed index refuses
   *  every later call, and a second `free` does nothing. */
  free(): void {
    this.compiled.free();
  }

  [Symbol.dispose](): void {
    this.free();
  }

  /**
   *  Creates an empty, mutable index in host memory, one shard per named NUMA node.
   *
   *  @param options - The geometry.
   *  @param memoryDomains - NUMA node IDs to shard over; absent spreads the shards over every node.
   *  @returns An index a RAM executor's verbs run on.
   *  @throws Will throw an error if any option is of incorrect type or invalid value.
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
        options.cpuClass,
      ),
    );
  }

  /**
   *  Creates an empty, mutable index in device memory, one shard per named CUDA ordinal.
   *
   *  @param options - The geometry.
   *  @param devices - CUDA ordinals to shard over; absent takes one free device.
   *  @returns An index a VRAM executor's verbs run on.
   *  @throws Will throw an error if any option is of incorrect type or invalid value.
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
        options.gpuClass,
      ),
    );
  }

  /**
   *  Copies a checkpoint into a new, mutable index in host memory, part `p` loading into shard `p`
   *  modulo the shard count. The geometry and metric come from the checkpoint itself.
   *
   *  @param path - The manifest of a sharded checkpoint, or a single monolith.
   *  @param config - The executor, the memory domains, and `valuesPerKey`.
   *  @returns The loaded index.
   */
  static loadRam(path: string, config: DenseLoadRamConfig = {}): DenseIndex {
    return new DenseIndex(
      indexSeal,
      compiled.CompiledIndex.loadRam(path, config.executor, config.memoryDomains, config.valuesPerKey, config.cpuClass),
    );
  }

  /**
   *  Copies a checkpoint into a new, mutable index in device memory, with the placement rules
   *  of {@link DenseIndex.loadRam}.
   *
   *  @param path - The manifest of a sharded checkpoint, or a single monolith.
   *  @param config - The executor, the devices, and `valuesPerKey`.
   *  @returns The loaded index.
   */
  static loadVram(path: string, config: DenseLoadVramConfig = {}): DenseIndex {
    return new DenseIndex(
      indexSeal,
      compiled.CompiledIndex.loadVram(path, config.executor, config.devices, config.valuesPerKey, config.gpuClass),
    );
  }

  /**
   *  Copies a monolith held in memory into a new, mutable index in host memory.
   *
   *  @param buffer - The bytes {@link DenseIndex.saveBuffer} produced.
   *  @param config - The executor, the memory domains, and `valuesPerKey`.
   *  @returns The loaded index.
   */
  static loadRamBuffer(buffer: Uint8Array | Monolith, config: DenseLoadRamBufferConfig = {}): DenseIndex {
    const bytes = buffer instanceof Monolith ? buffer.bytes : buffer;
    return new DenseIndex(
      indexSeal,
      compiled.CompiledIndex.loadRamBuffer(
        bytes,
        config.executor,
        config.memoryDomains,
        config.valuesPerKey,
        config.cpuClass,
      ),
    );
  }

  /**
   *  Copies a monolith held in memory into a new, mutable index in device memory.
   *
   *  @param buffer - The bytes {@link DenseIndex.saveBuffer} produced.
   *  @param config - The executor, the devices, and `valuesPerKey`.
   *  @returns The loaded index.
   */
  static loadVramBuffer(buffer: Uint8Array | Monolith, config: DenseLoadVramBufferConfig = {}): DenseIndex {
    const bytes = buffer instanceof Monolith ? buffer.bytes : buffer;
    return new DenseIndex(
      indexSeal,
      compiled.CompiledIndex.loadVramBuffer(
        bytes,
        config.executor,
        config.devices,
        config.valuesPerKey,
        config.gpuClass,
      ),
    );
  }

  /**
   *  Serves a checkpoint in place through a memory mapping, as a read-only index whose mutating
   *  verbs throw with {@link Status.ReadOnly}; the files must stay unmodified while it lives.
   *
   *  @param path - The manifest of a sharded checkpoint, or a single monolith.
   *  @returns The read-only index.
   */
  static viewRam(path: string): DenseIndex {
    return new DenseIndex(indexSeal, compiled.CompiledIndex.viewRam(path));
  }

  /**
   *  Serves a checkpoint in place from NVMe drives, read through the executor's submission rings,
   *  as a read-only index.
   *
   *  @param path - The manifest of a sharded checkpoint, or a single monolith.
   *  @param config - The NVMe executor, the paths to serve the parts from, and the key table.
   *  @returns The read-only index.
   */
  static viewNvme(path: string, config: DenseViewNvmeConfig = {}): DenseIndex {
    return new DenseIndex(
      indexSeal,
      compiled.CompiledIndex.viewNvme(path, config.executor, config.paths, config.cpuClass, config.keyTableDomains),
    );
  }

  /**
   *  Opens a checkpoint's manifest from NVMe drives as a writable index: it answers every read a
   *  view answers, and `refine` and `reshard` rewrite its parts one at a time. A lone monolith has
   *  no manifest to replace and refuses with {@link Status.Unsupported}.
   *
   *  @param path - The manifest of a sharded checkpoint.
   *  @param config - The NVMe executor and the key table; `paths` must name none.
   *  @returns The writable index.
   */
  static loadNvme(path: string, config: DenseViewNvmeConfig = {}): DenseIndex {
    return new DenseIndex(
      indexSeal,
      compiled.CompiledIndex.loadNvme(path, config.executor, config.paths, config.cpuClass, config.keyTableDomains),
    );
  }

  /**
   *  Reads a checkpoint's geometry and metric without loading it.
   *
   *  @param path - The manifest of a sharded checkpoint, or a single monolith.
   *  @param config - The workers the read runs on.
   *  @returns The geometry and metric the checkpoint froze.
   */
  static metadataRead(path: string, config: DenseMetadataReadConfig = {}): DenseMetadata {
    return metadataOf(compiled.CompiledIndex.metadataRead(path, config.executor));
  }

  /**
   *  Reads the geometry and metric of a monolith held in memory.
   *
   *  @param buffer - The bytes {@link DenseIndex.saveBuffer} produced.
   *  @param config - The workers the read runs on.
   *  @returns The geometry and metric the monolith froze.
   */
  static metadataReadBuffer(buffer: Uint8Array | Monolith, config: DenseMetadataReadBufferConfig = {}): DenseMetadata {
    const bytes = buffer instanceof Monolith ? buffer.bytes : buffer;
    return metadataOf(compiled.CompiledIndex.metadataReadBuffer(bytes, config.executor));
  }
}
