/**
 * The vocabulary every modality shares: closed-set names, identities, vector rows, the
 * executor whose threads a fanned-out verb runs on, and where a ranked run lands.
 */
import { CompiledExecutor, compiled } from "./library";

/**
 * Enumeration representing the metric kinds used to score the dissimilarity of two vectors.
 * @enum {string}
 * @readonly
 */
export enum MetricKind {
  Unknown = "unknown",
  Inner = "inner",
  Angular = "angular",
  Euclidean = "euclidean",
  Hamming = "hamming",
  Jaccard = "jaccard",
}

/**
 * Enumeration representing the scalar kinds vectors are stored and exchanged in.
 * @enum {string}
 * @readonly
 */
export enum ScalarKind {
  Unknown = "unknown",
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
  U8 = "u8",
  U4x2 = "u4x2",
  U1x8 = "u1x8",
}

/**
 * Enumeration representing where an index keeps its payload, and which engine traverses it.
 * @enum {string}
 * @readonly
 */
export enum Residency {
  Ram = "ram",
  Vram = "vram",
  Nvme = "nvme",
}

/**
 * A row of scalars, in whatever numeric TypedArray the caller already holds.
 *
 * The engine converts on ingest, so this is independent of the index's own quantization: an
 * `Int8Array` reaches an `f16` index unchanged. The kinds no TypedArray spells on its own -
 * `bf16`, `f16`, the mini-floats, the packed nibbles and bits - travel as bytes with a
 * `ScalarKind` named beside them.
 */
export type Vector =
  | Float64Array
  | Float32Array
  | Int32Array
  | Int16Array
  | Int8Array
  | Uint32Array
  | Uint16Array
  | Uint8Array
  | Uint8ClampedArray;

/** Several rows, each its own array, laid end to end before they cross. */
export type Matrix = Vector[];

/** Either shape a batch of rows arrives in. */
export type VectorOrMatrix = Vector | Matrix;

/** Bytes one identity occupies, on the wire and in the engine alike. */
export const keyBytes = 16;

/**
 * Sixteen opaque bytes, most significant first - one identity, losslessly.
 *
 * A run of several identities is the same type with a length that is a multiple of
 * {@link keyBytes}, which is what lets one batch of keys cross as one array.
 */
export type Key = Uint8Array;

/** Every spelling one identity is accepted in. */
export type KeyLike = bigint | number | string | Uint8Array;

/** Every spelling a run of identities is accepted in. */
export type KeysLike =
  | KeyLike
  | BigUint64Array
  | Array<bigint | number | string | Uint8Array>;

/**
 * Parks an integer identity in the trailing eight bytes, most significant first.
 *
 * @param {bigint|number} value - A non-negative integer below 2^64.
 * @return {Key} The sixteen identity bytes, the leading eight zeroed.
 */
export function keyFromInteger(value: bigint | number): Key {
  const integer = BigInt(value);
  if (integer < 0n || integer > 0xffffffffffffffffn)
    throw new Error("An integer identity must be a non-negative integer below 2^64.");
  const key = new Uint8Array(keyBytes);
  for (let position = 0; position < 8; position++)
    key[15 - position] = Number((integer >> BigInt(position * 8)) & 0xffn);
  return key;
}

/**
 * Reports the integer identity in the trailing eight bytes.
 *
 * @param {Uint8Array} key - Sixteen identity bytes.
 * @return {bigint} The integer the trailing eight bytes spell.
 * @throws Will throw an error if the leading eight bytes carry anything, since no integer
 *         spells that identity.
 */
export function integerFromKey(key: Uint8Array): bigint {
  if (key.length !== keyBytes)
    throw new Error("A key is sixteen bytes, most significant first.");
  for (let position = 0; position < 8; position++)
    if (key[position] !== 0)
      throw new Error(`The key ${formatKey(key)} carries no integer identity.`);
  let integer = 0n;
  for (let position = 8; position < keyBytes; position++)
    integer = (integer << 8n) | BigInt(key[position]);
  return integer;
}

/**
 * Renders one identity as a lowercase dashed UUID, which is how every SDK prints it.
 *
 * @param {Uint8Array} key - Sixteen identity bytes.
 * @return {string} Thirty-six characters, dashed after the 4th, 6th, 8th and 10th byte.
 */
export function formatKey(key: Uint8Array): string {
  if (key.length !== keyBytes)
    throw new Error("A key is sixteen bytes, most significant first.");
  let rendered = "";
  for (let position = 0; position < keyBytes; position++) {
    if (position === 4 || position === 6 || position === 8 || position === 10)
      rendered += "-";
    rendered += key[position].toString(16).padStart(2, "0");
  }
  return rendered;
}

/**
 * Reads the rendering {@link formatKey} produces. Dashes may sit anywhere, or nowhere.
 *
 * @param {string} text - Thirty-two hexadecimal digits, dashes ignored.
 * @return {Key} The sixteen identity bytes.
 */
export function parseKey(text: string): Key {
  const malformed = "The key is not thirty-two hexadecimal digits.";
  const digits = text.replace(/-/g, "");
  if (digits.length !== 32 || !/^[0-9a-fA-F]{32}$/.test(digits))
    throw new Error(malformed);
  const key = new Uint8Array(keyBytes);
  for (let position = 0; position < keyBytes; position++)
    key[position] = parseInt(digits.substr(position * 2, 2), 16);
  return key;
}

/**
 * Borrows the identity at `position` of a run, as a view rather than a copy.
 *
 * @param {Uint8Array} keys - A run of sixteen-byte identities.
 * @param {number} position - Which identity to borrow.
 * @return {Key} A view over those sixteen bytes, backed by the run's own storage.
 */
export function keyAt(keys: Uint8Array, position: number): Key {
  return keys.subarray(position * keyBytes, (position + 1) * keyBytes);
}

/**
 * Refuses a residency arm that belongs to another factory.
 *
 * TypeScript catches this at compile time, but a JavaScript caller reaches the factory with no
 * such check, and a silently ignored `devices` reads as a placement that was honoured.
 */
export function refuseForeignArms(
  options: object,
  own: string,
  subject: string
): void {
  const named = options as Record<string, unknown>;
  for (const arm of ["memoryDomains", "devices", "paths"]) {
    if (arm !== own && named[arm] !== undefined)
      throw new Error(
        `\`${arm}\` belongs to another residency; ${subject} places with \`${own}\`.`
      );
  }
}

/** Whether the value is a number above zero and free of a fractional part. */
export function isPositiveInteger(value: unknown): value is number {
  return typeof value === "number" && Number.isInteger(value) && value > 0;
}

/** Whether the value is a number at or above zero and free of a fractional part. */
export function isNonNegativeInteger(value: unknown): value is number {
  return typeof value === "number" && Number.isInteger(value) && value >= 0;
}

/** Whether the argument names one identity rather than a run of them. */
export function isOneKey(keys: KeysLike): boolean {
  return (
    typeof keys === "bigint" ||
    typeof keys === "string" ||
    (typeof keys === "number" && !Number.isNaN(keys))
  );
}

/** Reads one identity out of any spelling it is accepted in. */
export function normalizeKey(key: KeyLike): Key {
  if (key instanceof Uint8Array) {
    if (key.length !== keyBytes)
      throw new Error("A key is sixteen bytes, most significant first.");
    return key;
  }
  if (typeof key === "string") return parseKey(key);
  if (typeof key === "bigint") return keyFromInteger(key);
  if (typeof key === "number" && Number.isInteger(key) && key >= 0)
    return keyFromInteger(key);
  throw new Error(
    "A key must be a non-negative integer, a bigint, a dashed UUID, or sixteen bytes."
  );
}

/**
 * Gathers any spelling of a run of identities into the two forms the addon reads.
 *
 * A `BigUint64Array` and a whole run of sixteen-byte identities pass straight through, so a
 * caller that already holds either pays for no copy.
 */
export function normalizeKeys(keys: KeysLike): Uint8Array | BigUint64Array {
  if (keys instanceof BigUint64Array) return keys;
  if (keys instanceof Uint8Array) {
    if (keys.length % keyBytes !== 0)
      throw new Error("A run of keys is a whole number of sixteen-byte identities.");
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

/** Bytes one row of `dimensions` scalars occupies in `kind`; zero when the kind is unknown. */
export function rowBytes(dimensions: number, kind: ScalarKind): number {
  switch (kind) {
    case ScalarKind.F64:
      return dimensions * 8;
    case ScalarKind.F32:
      return dimensions * 4;
    case ScalarKind.BF16:
    case ScalarKind.F16:
      return dimensions * 2;
    case ScalarKind.E5M2:
    case ScalarKind.E4M3:
    case ScalarKind.E3M2:
    case ScalarKind.E2M3:
    case ScalarKind.I8:
    case ScalarKind.U8:
      return dimensions;
    case ScalarKind.I4x2:
    case ScalarKind.U4x2:
      return Math.ceil(dimensions / 2);
    case ScalarKind.U1x8:
      return Math.ceil(dimensions / 8);
    default:
      return 0;
  }
}

/** The scalar kind a TypedArray spells on its own; unknown when its elements name none. */
export function scalarOf(vectors: Vector): ScalarKind {
  if (vectors instanceof Float64Array) return ScalarKind.F64;
  if (vectors instanceof Float32Array) return ScalarKind.F32;
  if (vectors instanceof Int8Array) return ScalarKind.I8;
  if (vectors instanceof Uint8Array || vectors instanceof Uint8ClampedArray)
    return ScalarKind.U8;
  return ScalarKind.Unknown;
}

/** Whether the argument is one of the TypedArrays a row may arrive in. */
export function isVector(vectors: unknown): vectors is Vector {
  return (
    vectors instanceof Float64Array ||
    vectors instanceof Float32Array ||
    vectors instanceof Int32Array ||
    vectors instanceof Int16Array ||
    vectors instanceof Int8Array ||
    vectors instanceof Uint32Array ||
    vectors instanceof Uint16Array ||
    vectors instanceof Uint8Array ||
    vectors instanceof Uint8ClampedArray
  );
}

/**
 * Lays several rows end to end, leaving a TypedArray the caller already flattened untouched.
 *
 * Rows keep the type they arrived in, since the engine converts on ingest and a copy through
 * `Float32Array` would only discard what a packed or wider row carries. Plain number arrays
 * have no type of their own, so those land in a `Float32Array`.
 */
export function normalizeVectors(vectors: VectorOrMatrix | number[][]): Vector {
  if (isVector(vectors)) return vectors;
  if (!Array.isArray(vectors))
    throw new Error("Vectors must be a TypedArray or an array of arrays.");

  const first = vectors[0];
  const target = isVector(first)
    ? (first.constructor as Float32ArrayConstructor)
    : Float32Array;
  let totalLength = 0;
  for (const row of vectors) totalLength += row.length;
  const flattened = new target(totalLength);
  let offset = 0;
  for (const row of vectors) {
    flattened.set(row as ArrayLike<number>, offset);
    offset += row.length;
  }
  return flattened;
}

/** Where the host threads of a RAM executor are placed. */
export interface RamExecutorOptions {
  /** NUMA node ids the workers are spread over; absent spreads them over every node. */
  memoryDomains?: number[];
}

/** Which devices a VRAM executor drives. */
export interface VramExecutorOptions {
  /** CUDA ordinals to open one stream apiece on; absent takes every visible device. */
  devices?: number[];
}

/** Where an NVMe executor submits its reads. */
export interface NvmeExecutorOptions {
  /** Device or directory paths, one submission ring apiece; absent derives them. */
  paths?: string[];
}

/** The token the three factories hold, which is what a direct `new Executor` cannot produce. */
const executorSeal = Symbol("Executor");

/** Returns a worker count that is a non-negative integer, and throws on anything else. */
function checkedThreads(threads: number): number {
  if (!isNonNegativeInteger(threads))
    throw new Error(
      "`threads` must be a non-negative integer; 0 derives it from the hardware."
    );
  return threads;
}

/**
 * The workers a fanned-out verb runs on, shared across indexes.
 *
 * An executor owns host threads or device streams; an index owns rows. Keeping them apart is
 * what lets one pool serve several indexes, and what makes the cost of spawning visible at the
 * line that spawns it.
 */
export class Executor {
  /** The native executor the verbs are handed; not part of the public contract. */
  readonly compiled: CompiledExecutor;

  /** Reached through {@link Executor.forRam}, {@link Executor.forVram} or {@link Executor.forNvme}. */
  private constructor(seal: symbol, compiledExecutor: CompiledExecutor) {
    if (seal !== executorSeal)
      throw new Error(
        "An Executor is spawned by Executor.forRam, Executor.forVram or Executor.forNvme."
      );
    this.compiled = compiledExecutor;
  }

  /**
   * Spawns host worker threads for a RAM-resident index.
   *
   * @param {number} [threads=0] - Workers to spawn; 0 derives from the hardware.
   * @param {RamExecutorOptions} [options] - The NUMA nodes the workers are spread over.
   * @return {Executor} The pool, which must outlive every index it drives.
   */
  static forRam(threads: number = 0, options: RamExecutorOptions = {}): Executor {
    refuseForeignArms(options, "memoryDomains", "Executor.forRam");
    return new Executor(
      executorSeal,
      new compiled.Executor(
        Residency.Ram,
        checkedThreads(threads),
        options.memoryDomains
      )
    );
  }

  /**
   * Opens one stream per device for a VRAM-resident index, driven from a single host thread.
   *
   * @param {VramExecutorOptions} [options] - The CUDA ordinals to drive.
   * @return {Executor} The streams, which must outlive every index they drive.
   */
  static forVram(options: VramExecutorOptions = {}): Executor {
    refuseForeignArms(options, "devices", "Executor.forVram");
    return new Executor(
      executorSeal,
      new compiled.Executor(Residency.Vram, 0, options.devices)
    );
  }

  /**
   * Opens one submission ring per worker for an NVMe-resident index.
   *
   * @param {number} [threads=0] - Workers to spawn; 0 derives from the hardware.
   * @param {NvmeExecutorOptions} [options] - The device or directory paths to submit to.
   * @return {Executor} The pool, which must outlive every index it drives.
   */
  static forNvme(threads: number = 0, options: NvmeExecutorOptions = {}): Executor {
    refuseForeignArms(options, "paths", "Executor.forNvme");
    return new Executor(
      executorSeal,
      new compiled.Executor(
        Residency.Nvme,
        checkedThreads(threads),
        options.paths
      )
    );
  }

  /** Waits for every deferred call on this executor to finish. */
  drain(): void {
    this.compiled.drain();
  }
}

/**
 * The matches of one query, borrowed out of a run rather than copied.
 *
 * Both arrays are subarrays of the run's own storage, so reading a row costs nothing.
 */
export class MatchesRow {
  constructor(
    /** Sixteen bytes per match found for this query. */
    public readonly keys: Uint8Array,
    /** The distance of every match found, positionally paired with `keys`. */
    public readonly distances: Float64Array
  ) { }

  /** Matches this query found, which may be fewer than the run asked for. */
  get count(): number {
    return this.distances.length;
  }
}

/**
 * Three caller-owned buffers a ranked verb fills, and the row width their lengths fix.
 *
 * Every ranked verb writes into one of these and answers with telemetry alone, so a sweep
 * allocates once and reuses the same run at every point. The matches wanted per query is derived
 * from the lengths rather than asked for, which is what keeps the space reserved and the width
 * asked for from disagreeing.
 */
export class MatchesSpan {
  /** The `queries × wanted` identity matrix, rows contiguous, sixteen bytes per key. */
  public readonly keys: Uint8Array;
  /** The `queries × wanted` distance matrix, rows contiguous. */
  public readonly distances: Float64Array;
  /** Matches written per query, one entry per query. */
  public readonly counts: BigUint64Array;
  /** Columns per row, and the upper bound on matches - the "k" in "kANN". */
  public readonly wanted: number;

  /**
   * Lends three buffers already sized to `counts.length` rows.
   *
   * @param {Uint8Array} keys - Sixteen bytes per match, `counts.length × wanted` of them.
   * @param {Float64Array} distances - One score per match, the same shape as `keys`.
   * @param {BigUint64Array} counts - One entry per query.
   * @throws Will throw when the three lengths cannot be read as one shape.
   */
  constructor(keys: Uint8Array, distances: Float64Array, counts: BigUint64Array) {
    if (counts.length === 0)
      throw new Error("A run must hold at least one query.");
    if (keys.length % keyBytes !== 0)
      throw new Error("A run of keys is a whole number of sixteen-byte identities.");
    const cells = keys.length / keyBytes;
    if (cells !== distances.length)
      throw new Error("The key and distance buffers are not the same shape.");
    if (cells % counts.length !== 0 || cells / counts.length === 0)
      throw new Error("The buffers do not divide into one row per query.");
    this.keys = keys;
    this.distances = distances;
    this.counts = counts;
    this.wanted = cells / counts.length;
  }

  /** Queries the run covers. */
  get queries(): number {
    return this.counts.length;
  }

  /**
   * Borrows the matches of one query, without copying.
   *
   * @param {number} queryIndex - Which query of the run to read.
   * @returns {MatchesRow} - The keys and distances that query found.
   */
  get(queryIndex: number): MatchesRow {
    if (!Number.isInteger(queryIndex) || queryIndex < 0 || queryIndex >= this.counts.length)
      throw new Error(`No query ${queryIndex} in a run of ${this.counts.length}.`);
    const offset = queryIndex * this.wanted;
    const count = Number(this.counts[queryIndex]);
    return new MatchesRow(
      this.keys.subarray(offset * keyBytes, (offset + count) * keyBytes),
      this.distances.subarray(offset, offset + count)
    );
  }

  /** Allocates a run wide enough for `queries` rows of `wanted` matches apiece. */
  static reserve(queries: number, wanted: number): MatchesSpan {
    if (!isPositiveInteger(queries))
      throw new Error("`queries` must be a positive integer.");
    if (!isPositiveInteger(wanted))
      throw new Error("`wanted` must be a positive integer.");
    return new MatchesSpan(
      new Uint8Array(queries * wanted * keyBytes),
      new Float64Array(queries * wanted),
      new BigUint64Array(queries)
    );
  }
}
