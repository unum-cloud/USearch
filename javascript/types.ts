/**
 *  @file javascript/types.ts
 *  @author Ash Vardanian
 *  @date September 9, 2026
 *  @brief The vocabulary every modality shares: closed-set names, identities, vector rows, the
 *      executor whose threads a fanned-out verb runs on, the arena every scaling output lands in,
 *      the ranked run viewed inside it, and the deferred call a deferred receiver answers.
 */
import { CompiledDeferred, CompiledMatches, CompiledRows, CompiledTapeRows, compiled } from "./library";

/** What went wrong, one member per failure the library reports, spelled as its canonical name.
 *  Every error the library or this binding throws holds one as its `code`, so a caller branches
 *  on the identity of the fault rather than on which side of the boundary saw it. A
 *  {@link Deferred} also reports `Success`, and `Deferred` until its call is final.
 *  @enum {string}
 *  @readonly */
export enum Status {
  /** The call succeeded. */
  Success = "success",
  /** The call was submitted deferred and is not final yet; the one non-final status. */
  Deferred = "deferred",
  /** An allocation or reservation could not be satisfied. */
  OutOfMemory = "out-of-memory",
  /** Dimensionality or row counts disagree. */
  ShapeMismatch = "shape-mismatch",
  /** A stride or packing the engine cannot traverse. */
  UnsupportedLayout = "unsupported-layout",
  /** A scalar, metric, residency, traversal or relation this build does not know. */
  UnknownKind = "unknown-kind",
  /** A mutating verb on an index that serves its checkpoint in place. */
  ReadOnly = "read-only",
  /** A file could not be opened, read, written or parsed. */
  Io = "io",
  /** A device could not be selected, or its memory could not be claimed. */
  Device = "device",
  /** A caller-supplied representation could not be decoded, like a truncated checkpoint. */
  Malformed = "malformed",
  /** The library reported neither success nor a failure, which names a defect, not a state. */
  Internal = "internal",
  /** A hard design ceiling was reached, like the 32-bit ID space; a smaller request cannot help. */
  Overflow = "overflow",
  /** The verb needs a prior successful reservation that never happened. */
  Uninitialized = "uninitialized",
  /** The request contradicts a frozen configuration, like re-reserving with another geometry. */
  ConfigMismatch = "config-mismatch",
  /** The call was refused rather than waited for, like a submission to a held executor. */
  Contended = "contended",
  /** A progress handler asked the operation to stop; it published nothing. */
  Cancelled = "cancelled",
  /** This residency, traversal, predicate or build does not implement the request; pick another. */
  Unsupported = "unsupported",
  /** An argument breaks the verb's contract, like a missing input or a name no table spells. */
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
  /** Accelerator memory traversed by GPU streams, which a host must not read. */
  Vram = "vram",
  /** Persisted memory read through submission rings; its arenas are host pages. */
  Nvme = "nvme",
}

/** The vendor whose runtime drives a GPU, as `usearch_gpu_vendor_t` names it.
 *  @enum {string}
 *  @readonly */
export enum GpuVendor {
  /** No vendor, as host pages report. */
  Unknown = "unknown",
  /** NVIDIA, through the CUDA driver. */
  Nvidia = "nvidia",
  /** AMD, through the HIP runtime. */
  Amd = "amd",
  /** Apple, through Metal. */
  Apple = "apple",
}

/** Whether host code may read an arena's outputs, as `usearch_host_access_t` names it.
 *  @enum {string}
 *  @readonly */
export enum HostAccess {
  /** Device memory: every verb that would view the outputs throws {@link Status.Device}. */
  Refused = "refused",
  /** Host, pinned, managed and Apple's shared memory, viewed in place. */
  Readable = "readable",
}

/**
 *  How a search reaches the entries it scores.
 *  @enum {string}
 *  @readonly
 */
export enum Traversal {
  /** Walks the graph, scoring the entries the expansion reaches - approximate, the default. */
  Walk = "walk",
  /** Scores every entry published when the query's scan starts - exact, on RAM and NVMe handles. */
  Scan = "scan",
}

/**
 *  Enumeration representing the scalar kinds vectors are stored and exchanged in.
 *  @enum {string}
 *  @readonly
 */
export enum Scalar {
  /** IEEE-754 binary64: ±1.80e308, epsilon 2.22e-16, with infinities and NaNs. */
  F64 = "f64",
  /** IEEE-754 binary32: ±3.40e38, epsilon 1.19e-7, with infinities and NaNs. */
  F32 = "f32",
  /** Truncated `f32` with a 7-bit mantissa: `f32`'s range at epsilon 7.81e-3. */
  BF16 = "bf16",
  /** IEEE-754 binary16: ±65504, epsilon 9.77e-4, with infinities and NaNs. */
  F16 = "f16",
  /** FP8 with a 5-bit exponent: ±57344, epsilon 0.25, with infinities and NaNs. */
  E5M2 = "e5m2",
  /** FP8 with a 4-bit exponent: ±448, epsilon 0.125, no infinities. */
  E4M3 = "e4m3",
  /** FP6 with a 3-bit exponent in one byte: ±28, every code a number. */
  E3M2 = "e3m2",
  /** FP6 with a 2-bit exponent in one byte: ±7.5, every code a number. */
  E2M3 = "e2m3",
  /** 8-bit signed integer spanning -128 to 127. */
  I8 = "i8",
  /** Two signed nibbles per byte spanning -8 to 7, even dimensions in the low nibble. */
  I4x2 = "i4x2",
  /** The coordinates of geospatial points, which vectors refuse. */
  I32 = "i32",
  /** 8-bit unsigned integer spanning 0 to 255. */
  U8 = "u8",
  /** Two unsigned nibbles per byte spanning 0 to 15, even dimensions in the low nibble. */
  U4x2 = "u4x2",
  /** Eight single-bit dimensions per byte, the form the Hamming and Jaccard metrics score. */
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
 *  @throws Will throw an error if the leading eight bytes hold anything, since no integer
 *          spells that identity.
 */
export function integerFromKey(key: Uint8Array): bigint {
  if (key.length !== keyBytes) throw fail(Status.Malformed, "A key is sixteen bytes, most significant first.");
  for (let position = 0; position < 8; position++)
    if (key[position] !== 0) throw fail(Status.Malformed, `The key ${formatKey(key)} holds no integer identity.`);
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
 *  Views the identity at `position` of a run, as a view rather than a copy. Binding-only.
 *
 *  @param keys - A run of sixteen-byte identities.
 *  @param position - Which identity to view.
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

/** Gathers any spelling of a run of identities into the two forms the addon reads.
 *
 *  A `BigUint64Array` and a whole run of sixteen-byte identities pass straight through, so a
 *  caller that already holds either pays for no copy. */
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

/** A viewed run of rows of one width - what every batch argument is, and what a single row is
 *  when it travels as a bare TypedArray. Its row count travels beside it as the call's `count`. */
export class RowsView {
  /**
   *  Views `rows` rows of `columns` values of `scalar`, `stride` bytes apart; a zero stride repeats
   *  the first row.
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
   *  Views a packed run, every row `columns` values wide and laid end to end. Binding-only sugar.
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

/** Where the host threads of a RAM executor are placed, and how many there are. */
export interface ExecutorForRamConfig {
  /** NUMA node IDs the workers are spread over; absent spreads them over every node. */
  memoryDomains?: number[];
  /** Workers to spawn; absent takes 7/8 of the cores the domains hold. */
  threads?: number;
  /** The newest CPU class dispatch may pick, by canonical name, intersected with what the machine
   *  runs, so a class it lacks narrows rather than refuses; absent means no ceiling. */
  cpuClassCeiling?: string;
}

/** Which devices a VRAM executor drives, and on which streams. */
export interface ExecutorForVramConfig {
  /** Device ordinals to drive; absent drives every listed device of the first vendor with a known
   *  class. */
  devices?: number[];
  /** The caller's stream addresses, one per device in `devices` order. A CUDA or HIP stream
   *  becomes the executor's, destroyed when it closes; a Metal queue is retained, the executor
   *  releasing its own reference while the caller keeps theirs. A failed creation leaves every
   *  stream the caller's; absent opens the executor's own stream per device. */
  streams?: bigint[];
  /** Ceiling on scratch bytes per device; absent lets scratch grow as geometry asks. */
  scratchLimit?: number;
  /** The newest GPU class dispatch may pick, by canonical name, capped below the best class every
   *  named device loads; a ceiling no named device loads is refused as unsupported; absent means
   *  no ceiling. */
  gpuClassCeiling?: string;
}

/** Where an NVMe executor submits its reads. */
export interface ExecutorForNvmeConfig {
  /** Device or directory paths; absent derives them. */
  paths?: string[];
  /** Workers to spawn, one submission ring apiece; absent derives from the devices' queue capabilities. */
  threads?: number;
  /** The newest CPU class dispatch may pick, by canonical name, intersected with what the machine
   *  runs, so a class it lacks narrows rather than refuses; absent means no ceiling. */
  cpuClassCeiling?: string;
}

/** The workers a fanned-out verb runs on, shared across indexes.
 *
 *  An executor owns host threads or device streams; an index owns entries. Keeping them apart is
 *  what lets one pool serve several indexes, and what makes the cost of spawning visible at the
 *  line that spawns it. The collector frees it eventually; `free` joins the workers now. */
export interface Executor extends Disposable {
  /**
   *  Waits until no call holds this executor, then idles in `mood`.
   *
   *  @param [mood=Mood.Parked] - What the threads or streams do once they go idle.
   */
  drain(mood?: Mood): void;

  /** Waits out the call holding it and joins the workers now; a freed executor refuses every later
   *  call, and a second `free` does nothing. Refuses with {@link Status.Contended} while a GPU
   *  arena made on it is live. */
  free(): void;
}

/** The four factories an {@link Executor} is spawned by; a direct `new Executor()` is refused. */
export interface ExecutorConstructor {
  /**
   *  Spawns host worker threads for a RAM-resident index.
   *
   *  @param config - The NUMA nodes and the thread count.
   *  @returns The pool, which must outlive every call it runs.
   */
  forRam(config?: ExecutorForRamConfig): Executor;

  /**
   *  Opens one non-blocking stream per device for a VRAM-resident index, driven from a single host
   *  thread and created and destroyed with the executor.
   *
   *  @param config - The devices, the caller's streams and the scratch ceiling.
   *  @returns The executor, which must outlive every call it runs.
   */
  forVram(config?: ExecutorForVramConfig): Executor;

  /**
   *  Opens one submission ring per worker for an NVMe-resident index.
   *
   *  @param config - The device or directory paths and the thread count.
   *  @returns The pool, which must outlive every call it runs.
   */
  forNvme(config?: ExecutorForNvmeConfig): Executor;

  readonly prototype: Executor;
}

/** The native executor class itself, so `instanceof Executor` holds for every one the factories spawn. */
export const Executor = compiled.Executor as ExecutorConstructor;

/** Memory the library writes every output whose length scales with a call into, reused across
 *  calls.
 *
 *  Every verb with such an output takes one and answers a view into it rather than a copy, so a
 *  steady loop allocates once. Each call rewinds the arena it is handed, which ends every view of
 *  the previous call's outputs: their arrays read whatever the arena holds from then on, and
 *  nothing once it is freed. To keep one result past the next call, give that call another arena.
 *  A deferred call views it as a blocking one does: another call naming it before the first is
 *  final finishes {@link Status.Contended} and leaves the first call's outputs alone, and `free`
 *  waits for it. The collector frees the arena once nothing views it; `free` releases it now, and
 *  every later call naming it is refused with {@link Status.InvalidArgument}.
 *
 *  A `Vram` arena grows, fills and frees on its executor's first stream, so it serves only calls on
 *  that executor and keeps it alive until freed; every arena is freed before its executor. */
export interface Arena extends Disposable {
  /** Where its outputs live, as the executor that made it decided; a verb whose outputs land in an
   *  arena the host cannot read throws {@link Status.Device} rather than view device memory. */
  describe(): ArenaDescription;

  /** Releases the memory now, once every deferred call writing it is final; a second call does
   *  nothing. */
  free(): void;
}

/** Where an arena's outputs live. */
export interface ArenaDescription {
  /** `Ram` for host pages, `Vram` for one device's memory. */
  residency: Residency;
  /** The device's position in the GPU listing; `null` for host pages. */
  device: number | null;
  /** The vendor whose memory it is, as that listing reports; `GpuVendor.Unknown` for host pages. */
  vendor: GpuVendor;
  /** The vendor's own device number, which Arrow and DLPack name; `null` for host pages. */
  vendorOrdinal: number | null;
  /** Whether host code may read the outputs; when refused, verbs throw rather than view them. */
  hostAccess: HostAccess;
}

/** Makes an arena where `executor` computes, or host pages for calls on the calling
 *  thread when absent. */
export interface ArenaConstructor {
  new (executor?: Executor): Arena;
  readonly prototype: Arena;
}

/** The native arena class: `new Arena()` or `new Arena(executor)`. */
export const Arena = compiled.Arena as unknown as ArenaConstructor;

/** The matches of one query, viewed out of a run rather than copied.
 *
 *  Both arrays are subarrays of the run's own storage, so reading a query costs nothing. */
export class QueryMatches {
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

/** The ranked matches a search or a stream pull wrote into its arena: `queries` rows of `columns`
 *  cells, query `i` holding `counts[i]` matches, readable until the arena's next call. */
export class Matches {
  /** Queries the call resolved. */
  readonly queries: number;
  /** Columns per query the call wrote, at most its `wanted` - the "k" in "kANN". */
  readonly columns: number;
  /** The `queries × columns` identity matrix, rows contiguous, sixteen bytes per key. */
  readonly keys: Uint8Array;
  /** The `queries × columns` distance matrix, rows contiguous. */
  readonly scores: Float64Array;
  /** Matches written per query, one count per query. */
  readonly counts: BigUint64Array;

  /** Made by a ranked verb, never directly. */
  constructor(outputs: CompiledMatches, queries: number) {
    this.queries = queries;
    this.columns = outputs.columns;
    this.keys = outputs.keys;
    this.scores = outputs.scores;
    this.counts = outputs.counts;
  }

  /**
   *  Views the matches of one query, without copying. Binding-only.
   *
   *  @param queryIndex - Which query of the run to read.
   *  @returns The keys and scores that query found.
   */
  get(queryIndex: number): QueryMatches {
    if (!Number.isInteger(queryIndex) || queryIndex < 0 || queryIndex >= this.queries)
      throw fail(Status.InvalidArgument, `No query ${queryIndex} in a run of ${this.queries}.`);
    const offset = queryIndex * this.columns;
    const count = Number(this.counts[queryIndex]);
    return new QueryMatches(
      this.keys.subarray(offset * keyBytes, (offset + count) * keyBytes),
      this.scores.subarray(offset, offset + count),
    );
  }
}

/** One serialized monolith, viewed inside the arena `saveBuffer` wrote it into.
 *
 *  The bytes are what a checkpoint holds, so a service that keeps its indexes in an object store
 *  never touches the filesystem; `loadRamBuffer` and `metadataReadBuffer` read them back. */
export class Monolith {
  /** The monolith, valid until the arena's next call. */
  readonly bytes: Uint8Array;

  /** Made by `saveBuffer`, never directly. */
  constructor(bytes: Uint8Array) {
    this.bytes = bytes;
  }
}

/** Deferred calls dropped before they are final, each handed back to the collector until it is, so none is freed while the library still writes into it. */
const unfinished = new FinalizationRegistry<CompiledDeferred>((call) => {
  if (call.status() === Status.Deferred) unfinished.register({}, call);
});

/** Polls a deferred call makes on the event loop's next turns before falling back to a timer. */
const immediatePolls = 16;

/**
 *  A call submitted by a deferred receiver, final once its executor finishes it.
 *
 *  It keeps everything the library reads or writes alive until then - the outcome, the inputs,
 *  the outputs and every handle it names - so none of them is collected under the call, and it
 *  uses its arena as a blocking call does. `status` polls without blocking, `wait` blocks and
 *  answers, and `await` polls the status between turns of the event loop; both answer the same
 *  cached result, or throw the same failure, however often they are asked.
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

  /** Blocks the calling thread until the call is final, then answers its result or
   *  throws its failure. */
  wait(): T {
    if (!this.#outcome) {
      this.#call.wait();
      this.#settle();
    }
    return this.#unwrap();
  }

  /** Polls the status between turns of the event loop, then every millisecond, so `await` parks
   *  no thread on the call. */
  then<Fulfilled = T, Rejected = never>(
    onFulfilled?: ((value: T) => Fulfilled | PromiseLike<Fulfilled>) | null,
    onRejected?: ((reason: unknown) => Rejected | PromiseLike<Rejected>) | null,
  ): Promise<Fulfilled | Rejected> {
    this.#settling ??= this.#outcome
      ? Promise.resolve().then(() => this.#unwrap())
      : this.#polled().then(() => {
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

  /** Resolves once the call is final: the first polls ride `setImmediate`, later ones a 1 ms timer,
   *  the finest a timer waits. */
  async #polled(): Promise<void> {
    for (let polls = 0; this.#call.status() === Status.Deferred; polls++)
      await new Promise<void>((resume) => (polls < immediatePolls ? setImmediate(resume) : setTimeout(resume, 1)));
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

/** A ragged batch as Apache Arrow lays one out: elements end to end, and one more offset than
 *  entries around them, entry `i` spanning elements `offsets[i]` up to `offsets[i + 1]`.
 *
 *  An element is a byte of a sequence or a point of a shape, three coordinates wide. As an input
 *  it is viewed in place for the call; as a retrieval's output it is viewed inside its arena. */
export class Tape {
  /** Every element's scalars, end to end. */
  readonly data: Vector;
  /** One more offset than there are entries, non-decreasing, counted in elements. */
  readonly offsets: Uint32Array | BigUint64Array;
  /** The scalar kind of `data`. */
  readonly scalar: Scalar;
  /** Scalars per element: one for a sequence byte, three for a point. */
  readonly width: number;

  /**
   *  Wraps elements and their offsets without copying.
   *
   *  @param data - Every element's scalars, end to end.
   *  @param offsets - The Arrow offsets, 32- or 64-bit.
   *  @param scalar - The kind `data` holds; absent derives it from the array's element type.
   *  @param width - Scalars per element; absent is one.
   */
  constructor(data: Vector, offsets: Uint32Array | BigUint64Array, scalar?: Scalar, width = 1) {
    this.data = data;
    this.offsets = offsets;
    this.scalar = scalar ?? scalarOf(data);
    this.width = width;
  }

  /** Entries the tape holds. */
  get entries(): number {
    return Math.max(this.offsets.length - 1, 0);
  }

  /**
   *  Views one entry's scalars, without copying. Binding-only.
   *
   *  @param entry - Which entry to read.
   *  @returns The scalars of that entry's elements, end to end.
   */
  get(entry: number): Vector {
    if (!Number.isInteger(entry) || entry < 0 || entry >= this.entries)
      throw fail(Status.InvalidArgument, `No entry ${entry} in a tape of ${this.entries}.`);
    const begin = Number(this.offsets[entry]) * this.width;
    const end = Number(this.offsets[entry + 1]) * this.width;
    return this.data.subarray(begin, end) as Vector;
  }

  /**
   *  Packs byte strings end to end, encoding text as UTF-8. Binding-only.
   *
   *  @param sequences - The sequences, each text or bytes.
   *  @returns One tape holding them all, with 64-bit offsets.
   */
  static ofSequences(sequences: ReadonlyArray<string | Uint8Array>): Tape {
    const encoder = new TextEncoder();
    const parts = sequences.map((sequence) => (typeof sequence === "string" ? encoder.encode(sequence) : sequence));
    const offsets = new BigUint64Array(parts.length + 1);
    parts.forEach((part, position) => (offsets[position + 1] = offsets[position] + BigInt(part.length)));
    const data = new Uint8Array(Number(offsets[parts.length]));
    parts.forEach((part, position) => data.set(part, Number(offsets[position])));
    return new Tape(data, offsets, Scalar.U8, 1);
  }

  /**
   *  Packs point clouds end to end, three coordinates per point, every cloud of one array type.
   *  Binding-only.
   *
   *  @param shapes - The clouds, each `x, y, z` interleaved.
   *  @returns One tape holding them all, with 64-bit offsets counting points.
   */
  static ofShapes(shapes: ReadonlyArray<Float64Array | Float32Array | Float16Array>): Tape {
    if (shapes.length === 0) return new Tape(new Float32Array(0), new BigUint64Array(1), Scalar.F32, 3);
    const offsets = new BigUint64Array(shapes.length + 1);
    shapes.forEach((shape, position) => {
      if (shape.length % 3 !== 0 || shape.constructor !== shapes[0].constructor)
        throw fail(Status.ShapeMismatch, `Shape ${position} is not whole points of the first shape's type.`);
      offsets[position + 1] = offsets[position] + BigInt(shape.length / 3);
    });
    const Typed = shapes[0].constructor as new (length: number) => Float64Array | Float32Array | Float16Array;
    const data = new Typed(Number(offsets[shapes.length]) * 3);
    shapes.forEach((shape, position) => data.set(shape as never, Number(offsets[position]) * 3));
    return new Tape(data, offsets, scalarOf(shapes[0]), 3);
  }
}

/** One tape as it crosses into a verb. */
export function tapeCrossing(tape: Tape): CompiledTapeRows {
  return [tape.data, tape.offsets, scalarCodes[tape.scalar]];
}
