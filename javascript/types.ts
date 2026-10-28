/**
 *  @file javascript/types.ts
 *  @author Ash Vardanian
 *  @date September 9, 2026
 *  @brief The vocabulary every modality shares: closed-set names, identities, vector rows, the
 *      executor whose threads a fanned-out verb runs on, the arena every scaling output lands in,
 *      the ranked run viewed inside it, and the deferred call a deferred receiver answers.
 */
import { CompiledDeferred, CompiledMatches, CompiledRows, compiled } from "./library";

/**
 *  What went wrong, one member per `usearch_status_*_k` of the C interface and spelled as its
 *  canonical name. Every error the library or this binding throws carries one as its `code`, so a
 *  caller branches on the identity of the fault rather than on which side of the boundary saw it.
 *  A {@link Deferred} also reports `Success`, and `Deferred` until its call is final.
 *  @enum {string}
 *  @readonly
 */
export enum Status {
  Success = "success",
  Deferred = "deferred",
  OutOfMemory = "out-of-memory",
  ShapeMismatch = "shape-mismatch",
  UnsupportedLayout = "unsupported-layout",
  UnknownKind = "unknown-kind",
  ReadOnly = "read-only",
  Io = "io",
  Device = "device",
  Malformed = "malformed",
  Internal = "internal",
  Overflow = "overflow",
  Uninitialized = "uninitialized",
  ConfigMismatch = "config-mismatch",
  Contended = "contended",
  Cancelled = "cancelled",
  Unsupported = "unsupported",
  InvalidArgument = "invalid-argument",
}

/** An error thrown by the library or this binding, its `code` naming the failure. */
export interface USearchError extends Error {
  code: Status;
}

/** Builds the error this binding throws when it refuses a call before it crosses into the addon. */
export function fail(status: Status, message: string): USearchError {
  return Object.assign(new Error(message), { code: status });
}

/**
 *  What an executor does with its threads or streams once it goes idle.
 *  @enum {string}
 *  @readonly
 */
export enum Mood {
  /** Idle threads sleep; the next call pays a wake-up. */
  Parked = "parked",
  /** Idle threads stay hot, trading power for the next call's latency. */
  Spinning = "spinning",
}

/**
 *  Where an executor computes, and so where the arenas it makes keep their memory.
 *  @enum {string}
 *  @readonly
 */
export enum Residency {
  /** Host memory traversed by CPU threads. */
  Ram = "ram",
  /** Accelerator memory traversed by GPU streams, which no host view reads. */
  Vram = "vram",
  /** Persisted memory read through submission rings; its arenas are host pages. */
  Nvme = "nvme",
}

/**
 *  How a search reaches the rows it scores.
 *  @enum {string}
 *  @readonly
 */
export enum Traversal {
  /** Walks the graph, scoring the rows the expansion reaches - approximate, the default. */
  Walk = "walk",
  /** Scores every row published when the query's scan starts - exact, on RAM and NVMe handles. */
  Scan = "scan",
}

/**
 *  Enumeration representing the scalar kinds vectors are stored and exchanged in.
 *  @enum {string}
 *  @readonly
 */
export enum Scalar {
  F64 = "f64",
  F32 = "f32",
  BF16 = "bf16",
  F16 = "f16",
  E5M2 = "e5m2",
  E4M3 = "e4m3",
  E3M2 = "e3m2",
  E2M3 = "e2m3",
  I8 = "i8",
  I4x2 = "i4x2",
  /** The coordinates of geospatial points, which vectors refuse. */
  I32 = "i32",
  U8 = "u8",
  U4x2 = "u4x2",
  U1x8 = "u1x8",
}

/**
 *  One row of scalars, in whatever numeric TypedArray the caller already holds.
 *
 *  The engine converts on ingest, so this is independent of the index's own quantization: an
 *  `Int8Array` reaches an `f16` index unchanged. The kinds no TypedArray spells on its own -
 *  `bf16`, the mini-floats, the packed nibbles and bits - travel as bytes in a `Uint8Array` or
 *  `Uint16Array`, inside a {@link RowsView} naming their `Scalar`.
 */
export type Vector = Float64Array | Float32Array | Float16Array | Int8Array | Uint8Array | Uint16Array;

/** Bytes one identity occupies, on the wire and in the engine alike. */
export const keyBytes = 16;

/**
 *  Sixteen opaque bytes, most significant first - one identity, losslessly.
 *
 *  A run of several identities is the same type with a length that is a multiple of
 *  {@link keyBytes}, which is what lets one batch of keys cross as one array.
 */
export type Key = Uint8Array;

/** Every spelling one identity is accepted in. */
export type KeyLike = bigint | number | string | Uint8Array;

/** Every spelling a run of identities is accepted in. */
export type KeysLike = KeyLike | BigUint64Array | Array<bigint | number | string | Uint8Array>;

/**
 *  Parks an integer identity in the trailing eight bytes, most significant first. Binding-only.
 *
 *  @param value - A non-negative integer below 2^64.
 *  @returns The sixteen identity bytes, the leading eight zeroed.
 */
export function keyFromInteger(value: bigint | number): Key {
  const integer = BigInt(value);
  if (integer < 0n || integer > 0xffffffffffffffffn)
    throw fail(Status.InvalidArgument, "An integer identity must be a non-negative integer below 2^64.");
  const key = new Uint8Array(keyBytes);
  for (let position = 0; position < 8; position++)
    key[15 - position] = Number((integer >> BigInt(position * 8)) & 0xffn);
  return key;
}

/**
 *  Reports the integer identity in the trailing eight bytes. Binding-only.
 *
 *  @param key - Sixteen identity bytes.
 *  @returns The integer the trailing eight bytes spell.
 *  @throws Will throw an error if the leading eight bytes carry anything, since no integer
 *          spells that identity.
 */
export function integerFromKey(key: Uint8Array): bigint {
  if (key.length !== keyBytes) throw fail(Status.Malformed, "A key is sixteen bytes, most significant first.");
  for (let position = 0; position < 8; position++)
    if (key[position] !== 0) throw fail(Status.Malformed, `The key ${formatKey(key)} carries no integer identity.`);
  let integer = 0n;
  for (let position = 8; position < keyBytes; position++) integer = (integer << 8n) | BigInt(key[position]);
  return integer;
}

/**
 *  Renders one identity as a lowercase dashed UUID, which is how every SDK prints it. Binding-only.
 *
 *  @param key - Sixteen identity bytes.
 *  @returns Thirty-six characters, dashed after the 4th, 6th, 8th and 10th byte.
 */
export function formatKey(key: Uint8Array): string {
  if (key.length !== keyBytes) throw fail(Status.Malformed, "A key is sixteen bytes, most significant first.");
  let rendered = "";
  for (let position = 0; position < keyBytes; position++) {
    if (position === 4 || position === 6 || position === 8 || position === 10) rendered += "-";
    rendered += key[position].toString(16).padStart(2, "0");
  }
  return rendered;
}

/**
 *  Reads the rendering {@link formatKey} produces. Dashes may sit anywhere, or nowhere. Binding-only.
 *
 *  @param text - Thirty-two hexadecimal digits, dashes ignored.
 *  @returns The sixteen identity bytes.
 */
export function parseKey(text: string): Key {
  const malformed = "The key is not thirty-two hexadecimal digits.";
  const digits = text.replace(/-/g, "");
  if (digits.length !== 32 || !/^[0-9a-fA-F]{32}$/.test(digits)) throw fail(Status.Malformed, malformed);
  const key = new Uint8Array(keyBytes);
  for (let position = 0; position < keyBytes; position++) key[position] = parseInt(digits.substr(position * 2, 2), 16);
  return key;
}

/**
 *  Borrows the identity at `position` of a run, as a view rather than a copy. Binding-only.
 *
 *  @param keys - A run of sixteen-byte identities.
 *  @param position - Which identity to borrow.
 *  @returns A view over those sixteen bytes, backed by the run's own storage.
 */
export function keyAt(keys: Uint8Array, position: number): Key {
  return keys.subarray(position * keyBytes, (position + 1) * keyBytes);
}

/** Whether the value is a number above zero and free of a fractional part. */
export function isPositiveInteger(value: unknown): value is number {
  return typeof value === "number" && Number.isInteger(value) && value > 0;
}

/** Reads one identity out of any spelling it is accepted in. */
export function normalizeKey(key: KeyLike): Key {
  if (key instanceof Uint8Array) {
    if (key.length !== keyBytes) throw fail(Status.Malformed, "A key is sixteen bytes, most significant first.");
    return key;
  }
  if (typeof key === "string") return parseKey(key);
  if (typeof key === "bigint") return keyFromInteger(key);
  if (typeof key === "number" && Number.isInteger(key) && key >= 0) return keyFromInteger(key);
  throw fail(
    Status.InvalidArgument,
    "A key must be a non-negative integer, a bigint, a dashed UUID, or sixteen bytes.",
  );
}

/**
 *  Gathers any spelling of a run of identities into the two forms the addon reads.
 *
 *  A `BigUint64Array` and a whole run of sixteen-byte identities pass straight through, so a
 *  caller that already holds either pays for no copy.
 */
export function normalizeKeys(keys: KeysLike): Uint8Array | BigUint64Array {
  if (keys instanceof BigUint64Array) return keys;
  if (keys instanceof Uint8Array) {
    if (keys.length % keyBytes !== 0)
      throw fail(Status.ShapeMismatch, "A run of keys is a whole number of sixteen-byte identities.");
    return keys;
  }
  if (Array.isArray(keys)) {
    const gathered = new Uint8Array(keys.length * keyBytes);
    for (let position = 0; position < keys.length; position++)
      gathered.set(normalizeKey(keys[position]), position * keyBytes);
    return gathered;
  }
  return normalizeKey(keys);
}

/** The code each scalar kind crosses the addon as, mapped once from the C name tables. */
export const scalarCodes = compiled.scalars as Readonly<Record<Scalar, number>>;

/** Bits one value of each kind occupies, which is how a row's bytes become its columns. */
const scalarBits: Record<Scalar, number> = {
  [Scalar.F64]: 64,
  [Scalar.F32]: 32,
  [Scalar.I32]: 32,
  [Scalar.BF16]: 16,
  [Scalar.F16]: 16,
  [Scalar.E5M2]: 8,
  [Scalar.E4M3]: 8,
  [Scalar.E3M2]: 8,
  [Scalar.E2M3]: 8,
  [Scalar.I8]: 8,
  [Scalar.U8]: 8,
  [Scalar.I4x2]: 4,
  [Scalar.U4x2]: 4,
  [Scalar.U1x8]: 1,
};

/** The scalar kind a TypedArray spells on its own; a `Uint16Array` spells none and is refused. */
export function scalarOf(data: Vector): Scalar {
  if (data instanceof Float64Array) return Scalar.F64;
  if (data instanceof Float32Array) return Scalar.F32;
  if (typeof Float16Array !== "undefined" && data instanceof Float16Array) return Scalar.F16;
  if (data instanceof Int8Array) return Scalar.I8;
  if (data instanceof Uint8Array) return Scalar.U8;
  throw fail(Status.UnknownKind, "The rows spell no scalar kind on their own; name one in a RowsView.");
}

/**
 *  A borrowed run of rows of one width - what every batch argument is, and what a single row is
 *  when it travels as a bare TypedArray. Mirrors `usearch_rows_view_t`, with the row count the
 *  call's `count` takes beside it.
 */
export class RowsView {
  /**
   *  Borrows `rows` rows of `columns` values of `scalar`, `stride` bytes apart; a zero stride repeats the first row.
   *
   *  @param data - The array holding every row; never copied.
   *  @param rows - Rows the run holds, which the call reads as its batch.
   *  @param columns - Values per row, as the index counts its dimensions.
   *  @param stride - Bytes between the starts of consecutive rows.
   *  @param scalar - What the bytes hold, converted by the engine where it differs from storage.
   */
  constructor(
    readonly data: Vector,
    readonly rows: number,
    readonly columns: number,
    readonly stride: number,
    readonly scalar: Scalar,
  ) {}

  /**
   *  Borrows a packed run, every row `columns` values wide and laid end to end. Binding-only sugar.
   *
   *  @param data - The rows, end to end.
   *  @param columns - Values per row.
   *  @param scalar - What the bytes hold; defaults to the kind the array spells.
   *  @returns As many rows as `data` holds.
   *  @throws Will throw when the array does not split into whole rows.
   */
  static of(data: Vector, columns: number, scalar: Scalar = scalarOf(data)): RowsView {
    const rowBytes = Math.ceil((columns * scalarBits[scalar]) / 8);
    if (!isPositiveInteger(columns) || data.byteLength % rowBytes !== 0)
      throw fail(Status.ShapeMismatch, `The array does not split into rows of ${columns} ${scalar} values.`);
    return new RowsView(data, data.byteLength / rowBytes, columns, rowBytes, scalar);
  }
}

/** Reads a bare TypedArray as one row of the kind it spells, and a {@link RowsView} as itself. */
export function rowsOf(rows: RowsView | Vector): RowsView {
  if (rows instanceof RowsView) return rows;
  const scalar = scalarOf(rows);
  return new RowsView(rows, 1, (rows.byteLength * 8) / scalarBits[scalar], rows.byteLength, scalar);
}

/** The five values a run of rows crosses the boundary as. */
export function crossing(rows: RowsView | Vector): CompiledRows {
  const view = rowsOf(rows);
  return [view.data, view.rows, view.columns, view.stride, scalarCodes[view.scalar]];
}

/** Where the host threads of a RAM executor are placed, how many there are, and how many deferred calls it tracks. */
export interface ExecutorForRamConfig {
  /** NUMA node IDs the workers are spread over; absent spreads them over every node. */
  memoryDomains?: number[];
  /** Workers to spawn; absent takes 7/8 of the cores the domains carry. */
  threads?: number;
  /** Deferred calls tracked at once; absent takes a default. */
  deferredLimit?: number;
  /** The CPU class to dispatch as, as `usearch_cpu_class_name` spells it; absent dispatches as the machine. */
  cpuClass?: string;
}

/** Where the borrowed pool of a RAM executor is placed, and how many deferred calls it tracks. */
export interface ExecutorOnPoolConfig {
  /** NUMA node IDs of the indexes it will drive; absent derives every node. */
  memoryDomains?: number[];
  /** Deferred calls tracked at once; absent takes a default. */
  deferredLimit?: number;
  /** The CPU class to dispatch as, as `usearch_cpu_class_name` spells it; absent dispatches as the machine. */
  cpuClass?: string;
}

/** Which devices a VRAM executor drives, and on which streams. */
export interface ExecutorForVramConfig {
  /** CUDA ordinals to drive; absent takes one free device. */
  devices?: number[];
  /** Borrowed CUDA stream addresses, one per device in `devices` order; absent creates its own. */
  streams?: bigint[];
  /** Ceiling on scratch bytes per device; absent lets scratch grow as geometry asks. */
  scratchLimit?: number;
  /** Deferred calls tracked at once; absent takes a default. */
  deferredLimit?: number;
  /** The GPU class to dispatch as, as `usearch_gpu_class_name` spells it; absent dispatches as the devices. */
  gpuClass?: string;
}

/** Where an NVMe executor submits its reads. */
export interface ExecutorForNvmeConfig {
  /** Device or directory paths; absent derives them. */
  paths?: string[];
  /** Workers to spawn, one submission ring apiece; absent derives from the devices' queue capabilities. */
  threads?: number;
  /** Deferred calls tracked at once; absent takes a default. */
  deferredLimit?: number;
  /** The CPU class to dispatch as, as `usearch_cpu_class_name` spells it; absent dispatches as the machine. */
  cpuClass?: string;
}

/**
 *  The workers a fanned-out verb runs on, shared across indexes.
 *
 *  An executor owns host threads or device streams; an index owns rows. Keeping them apart is
 *  what lets one pool serve several indexes, and what makes the cost of spawning visible at the
 *  line that spawns it. The collector frees it eventually; `free` joins the workers now.
 */
export interface Executor extends Disposable {
  /**
   *  Waits for every call submitted to this executor so far, then idles in `mood`. Calls `usearch_executor_drain`.
   *
   *  @param [mood=Mood.Parked] - What the threads or streams do once they go idle.
   */
  drain(mood?: Mood): void;

  /** Runs every deferred call and joins the workers now; a freed executor refuses every later call, and a second `free` does nothing. Refuses with {@link Status.Contended} while a GPU arena made on it is live. Calls `usearch_executor_free`. */
  free(): void;
}

/** The four factories an {@link Executor} is spawned by; a direct `new Executor()` is refused. */
export interface ExecutorConstructor {
  /**
   *  Spawns host worker threads for a RAM-resident index. Calls `usearch_executor_for_ram`.
   *
   *  @param config - The NUMA nodes, the thread count and the deferred limit.
   *  @returns The pool, which must outlive every call it runs.
   */
  forRam(config?: ExecutorForRamConfig): Executor;

  /**
   *  Borrows a ForkUnion `fu_pool_t` for a RAM-resident index, keeping its lender's mood. Calls `usearch_executor_on_pool`.
   *
   *  @param pool - The pool's address; it must outlive the executor.
   *  @param config - The NUMA nodes and the deferred limit.
   *  @returns The executor over the borrowed pool.
   */
  onPool(pool: bigint, config?: ExecutorOnPoolConfig): Executor;

  /**
   *  Opens one stream per device for a VRAM-resident index, driven from a single host thread. Calls `usearch_executor_for_vram`.
   *
   *  @param config - The devices, borrowed streams, scratch ceiling and deferred limit.
   *  @returns The streams, which must outlive every call they run.
   */
  forVram(config?: ExecutorForVramConfig): Executor;

  /**
   *  Opens one submission ring per worker for an NVMe-resident index. Calls `usearch_executor_for_nvme`.
   *
   *  @param config - The device or directory paths, the thread count and the deferred limit.
   *  @returns The pool, which must outlive every call it runs.
   */
  forNvme(config?: ExecutorForNvmeConfig): Executor;

  readonly prototype: Executor;
}

/** The native executor class itself, so `instanceof Executor` holds for every one the factories spawn. */
export const Executor = compiled.Executor as ExecutorConstructor;

/**
 *  Memory the library writes every output whose length scales with a call into, reused across calls.
 *
 *  Every verb with such an output takes one and answers a view into it rather than a copy, so a
 *  steady loop allocates once. Each call rewinds the arena it is handed and detaches every buffer
 *  over the previous call's outputs: a view refuses its reads with {@link Status.Uninitialized} from
 *  then on, and every array already taken out of one reads empty rather than another call's memory.
 *  To keep one result past the next call, give that call another arena. A deferred call borrows it
 *  as a blocking one does: another call naming it before the first is final finishes
 *  {@link Status.Contended} and leaves the first call's outputs alone, and `free` waits for it. The
 *  collector frees the arena once nothing views it; `free` releases it now.
 *
 *  A `Vram` arena grows, fills and frees on its executor's first stream, so it serves only calls on
 *  that executor and keeps it alive until freed; host pages may outlive their executor.
 */
export interface Arena extends Disposable {
  /** The residency of the executor that made it; a `Vram` arena's views refuse host reads with {@link Status.Device}. */
  readonly residency: Residency;

  /** Releases the memory now, once every deferred call writing it is final, detaching every view into it; a second call does nothing. Calls `usearch_arena_free`. */
  free(): void;
}

/** Makes an arena where `executor` computes, or host pages for calls on the calling thread when absent. Calls `usearch_arena_init`. */
export interface ArenaConstructor {
  new (executor?: Executor): Arena;
  readonly prototype: Arena;
}

/** The native arena class: `new Arena()` or `new Arena(executor)`. */
export const Arena = compiled.Arena as unknown as ArenaConstructor;

/**
 *  Outputs a call wrote into an arena, readable until the arena's next call rewinds it.
 *
 *  The rewind detaches the buffer under every array, so each accessor refuses once the arena moved
 *  on, with {@link Status.Uninitialized}, and refuses device memory with {@link Status.Device}, so a
 *  view never reads memory its call did not write.
 */
export abstract class ArenaView {
  /** Answers `view` while its buffer still holds this call's outputs in host memory, and refuses otherwise. */
  protected live<View extends ArrayBufferView>(view: View | null): View {
    if (view === null) throw fail(Status.Device, "The outputs are device memory, which no host view reads.");
    if ((view.buffer as ArrayBuffer).detached)
      throw fail(Status.Uninitialized, "The arena was rewound or freed since this view was written.");
    return view;
  }
}

/**
 *  The matches of one query, borrowed out of a run rather than copied.
 *
 *  Both arrays are subarrays of the run's own storage, so reading a row costs nothing.
 */
export class MatchesRow {
  constructor(
    /** Sixteen bytes per match found for this query. */
    public readonly keys: Uint8Array,
    /** The score of every match found, positionally paired with `keys`. */
    public readonly scores: Float64Array,
  ) {}

  /** Matches this query found, which may be fewer than the run asked for. */
  get count(): number {
    return this.scores.length;
  }
}

/**
 *  The ranked rows a search or a stream pull wrote into its arena: `queries` rows of `columns`
 *  cells, row `i` holding `counts[i]` matches. Mirrors `usearch_matches_view_t`.
 */
export class Matches extends ArenaView {
  /** Rows the call resolved, one per query. */
  readonly queries: number;
  /** Columns per row the call wrote, at most its `wanted` - the "k" in "kANN". */
  readonly columns: number;
  readonly #landed: CompiledMatches;

  /** Made by a ranked verb, never directly. */
  constructor(landed: CompiledMatches, queries: number) {
    super();
    this.#landed = landed;
    this.queries = queries;
    this.columns = landed.columns;
  }

  /** The `queries × columns` identity matrix, rows contiguous, sixteen bytes per key. */
  get keys(): Uint8Array {
    return this.live(this.#landed.keys);
  }

  /** The `queries × columns` distance matrix, rows contiguous. */
  get scores(): Float64Array {
    return this.live(this.#landed.scores);
  }

  /** Matches written per query, one entry per query. */
  get counts(): BigUint64Array {
    return this.live(this.#landed.counts);
  }

  /**
   *  Borrows the matches of one query, without copying. Binding-only.
   *
   *  @param queryIndex - Which query of the run to read.
   *  @returns The keys and scores that query found.
   */
  get(queryIndex: number): MatchesRow {
    if (!Number.isInteger(queryIndex) || queryIndex < 0 || queryIndex >= this.queries)
      throw fail(Status.InvalidArgument, `No query ${queryIndex} in a run of ${this.queries}.`);
    const offset = queryIndex * this.columns;
    const count = Number(this.counts[queryIndex]);
    return new MatchesRow(
      this.keys.subarray(offset * keyBytes, (offset + count) * keyBytes),
      this.scores.subarray(offset, offset + count),
    );
  }
}

/**
 *  One serialized monolith, viewed inside the arena `saveBuffer` landed it in.
 *
 *  The bytes are what a checkpoint holds, so a service that keeps its indexes in an object store
 *  never touches the filesystem; `loadRamBuffer` and `metadataReadBuffer` read them back.
 */
export class Monolith extends ArenaView {
  readonly #bytes: Uint8Array | null;

  /** Made by `saveBuffer`, never directly. */
  constructor(bytes: Uint8Array | null) {
    super();
    this.#bytes = bytes;
  }

  /** The monolith, valid until the arena's next call. */
  get bytes(): Uint8Array {
    return this.live(this.#bytes);
  }
}

/** Deferred calls dropped before they are final, each handed back to the collector until it is, so none is freed while the library still writes into it. */
const unfinished = new FinalizationRegistry<CompiledDeferred>((call) => {
  if (call.status() === Status.Deferred) unfinished.register({}, call);
});

/**
 *  A call submitted by a deferred receiver, final once its executor finishes it.
 *
 *  It keeps everything the library reads or writes alive until then - the outcome, the inputs,
 *  the outputs and every handle it names - so none of them is collected under the call, and it
 *  borrows its arena as a blocking call does. A result read after the arena was rewound or freed
 *  throws {@link Status.Uninitialized}. `status` polls without
 *  blocking, `wait` blocks and answers, and `await` runs that wait on the libuv pool; both answer
 *  the same cached result, or throw the same failure, however often they are asked.
 *
 *  ```js
 *  const matches = await call; // or: call.wait()
 *  ```
 */
export class Deferred<T> implements PromiseLike<T>, Disposable {
  readonly #call: CompiledDeferred;
  readonly #answer: (raw: never) => T;
  #outcome: { value: T } | { error: unknown } | undefined;
  #settling: Promise<T> | undefined;

  /** Made by a verb of a deferred receiver, never directly. */
  constructor(call: CompiledDeferred, answer: (raw: never) => T) {
    this.#call = call;
    this.#answer = answer;
    unfinished.register(this, call, this);
  }

  /** `Status.Deferred` until the call is final, then `Status.Success` or its failure. Reads the outcome without blocking. */
  get status(): Status {
    return this.#call.status() as Status;
  }

  /**
   *  Blocks the calling thread until the call is final, then answers its result or throws its failure.
   *  Calls `usearch_outcome_wait`.
   */
  wait(): T {
    if (!this.#outcome) {
      this.#call.wait();
      this.#settle();
    }
    return this.#unwrap();
  }

  /** Waits on the libuv pool rather than the calling thread, which is what `await` reaches for. Calls `usearch_outcome_wait` there. */
  then<Fulfilled = T, Rejected = never>(
    onFulfilled?: ((value: T) => Fulfilled | PromiseLike<Fulfilled>) | null,
    onRejected?: ((reason: unknown) => Rejected | PromiseLike<Rejected>) | null,
  ): Promise<Fulfilled | Rejected> {
    this.#settling ??= this.#outcome
      ? Promise.resolve().then(() => this.#unwrap())
      : this.#call.settled().then(() => {
          this.#settle();
          return this.#unwrap();
        });
    return this.#settling.then(onFulfilled, onRejected);
  }

  /** Waits until the call is final, so `using` never leaves one running past its block. */
  [Symbol.dispose](): void {
    if (this.#outcome) return;
    this.#call.wait();
    this.#settle();
  }

  #settle(): void {
    if (this.#outcome) return;
    try {
      this.#outcome = { value: this.#answer(this.#call.result() as never) };
    } catch (error) {
      this.#outcome = { error };
    }
    unfinished.unregister(this);
  }

  #unwrap(): T {
    const outcome = this.#outcome as { value: T } | { error: unknown };
    if ("error" in outcome) throw outcome.error;
    return outcome.value;
  }
}
