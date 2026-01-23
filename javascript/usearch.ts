import build from "node-gyp-build";
import * as path from "path";
import { existsSync } from "fs";
import { getFileName, getRoot } from "bindings";

const compiled: Compiled = build(getBuildDir(getDirName()));

type Vector = Float32Array | Float64Array | Int8Array;
type Matrix = Vector[];
type VectorOrMatrix = Vector | Matrix;

type CompiledSearchResult = [
  keys: BigUint64Array,
  distances: Float32Array,
  counts: BigUint64Array
];

interface CompiledIndex {
  add(keys: BigUint64Array, vectors: Vector, threads: number): void;
  search(vectors: VectorOrMatrix, k: number, threads: number): CompiledSearchResult;
  contains(keys: BigUint64Array): boolean[];
  count(keys: BigUint64Array): number | number[];
  remove(keys: BigUint64Array): number[];
  dimensions(): number;
  connectivity(): number;
  size(): number;
  capacity(): number;
  save(path: string): void;
  load(path: string): void;
  view(path: string): void;
}

interface Compiled {
  CompiledIndex: CompiledIndex;
  exactSearch(
    dataset: VectorOrMatrix,
    queries: VectorOrMatrix,
    dimensions: number,
    count: number,
    metric: MetricKind,
    threads: number
  ): CompiledSearchResult;
}

/**
 * Enumeration representing the various metric kinds used to measure the distance between vectors in the index.
 * @enum {string}
 * @readonly
 */
export enum MetricKind {
  Unknown = "unknown",
  Cos = "cos",
  IP = "ip",
  L2sq = "l2sq",
  Haversine = "haversine",
  Divergence = "divergence",
  Pearson = "pearson",
  Jaccard = "jaccard",
  Hamming = "hamming",
  Tanimoto = "tanimoto",
  Sorensen = "sorensen",
}

/**
 * Enumeration representing the various scalar kinds used to define the type of scalar values in vectors.
 * @enum {string}
 * @readonly
 */
export enum ScalarKind {
  Unknown = "unknown",
  F32 = "f32",
  F64 = "f64",
  F16 = "f16",
  BF16 = "bf16",
  I8 = "i8",
  B1 = "b1",
}

/**
 * Represents a set of search results.
 */
export class Matches {
  /**
   * Constructs a Matches object.
   *
   * @param {BigUint64Array} keys - The keys of the nearest neighbors found.
   * @param {Float32Array} distances - The distances of the nearest neighbors found.
   */
  constructor(public keys: BigUint64Array, public distances: Float32Array) { }
}

/**
 * Represents a set of batched search results.
 */
export class BatchMatches {
  /**
   * Constructs a BatchMatches object.
   *
   * @param {BigUint64Array} keys - The keys of the nearest neighbors found in the batch.
   * @param {Float32Array} distances - The distances of the nearest neighbors found in the batch.
   * @param {BigUint64Array} counts - The number of neighbors found for each query in the batch.
   * @param {number} k - The limit for search results per query in the batch.
   */
  constructor(
    public keys: BigUint64Array,
    public distances: Float32Array,
    public counts: BigUint64Array,
    public k: number
  ) { }

  /**
   * Retrieves a Matches object at the specified index in the batch.
   *
   * @param {number} i - The index at which to retrieve the Matches object.
   * @returns {Matches} - A Matches object representing the search results at the specified index in the batch.
   */
  get(i: number): Matches {
    const index = Number(i) * Number(this.k);
    const count = Number(this.counts[i]);
    const keysSlice = this.keys.slice(index, index + count);
    const distancesSlice = this.distances.slice(index, index + count);
    return new Matches(keysSlice, distancesSlice);
  }
}

function isOneKey(keys: number | bigint | BigUint64Array | bigint[]): boolean {
  return (
    (!Number.isNaN(keys) && typeof keys === "number") ||
    typeof keys === "bigint"
  );
}

function normalizeKeys(keys: unknown): BigUint64Array {
  if (keys instanceof BigUint64Array) {
    return keys;
  }
  let normalizedKeys: BigUint64Array;
  if (
    (typeof keys === "number" && !Number.isNaN(keys)) ||
    typeof keys === "bigint"
  ) {
    normalizedKeys = BigUint64Array.of(BigInt(keys));
  } else if (Array.isArray(keys)) {
    const bigintkeys = keys.map((key) => {
      if (typeof key === "bigint") {
        return key;
      } else if (
        typeof key === "number" &&
        !Number.isNaN(key) &&
        Number.isInteger(key) &&
        key >= 0
      ) {
        return BigInt(key);
      }
      throw new Error("All keys must be positive integers or bigints.");
    });
    normalizedKeys = BigUint64Array.from(bigintkeys);
  } else {
    throw new Error(
      "Keys must be a number, bigint, an array of numbers or bigints, or a BigUint64Array."
    );
  }
  return normalizedKeys;
}

function isVector(vectors: unknown) {
  return (
    vectors instanceof Float32Array ||
    vectors instanceof Float64Array ||
    vectors instanceof Int8Array
  );
}

function normalizeVectors(
  vectors: VectorOrMatrix,
  dimensions: number,
  targetType: NumberArrayConstructor = Float32Array
): Vector {
  let flattenedVectors: Vector;
  if (isVector(vectors)) {
    flattenedVectors =
      vectors.constructor === targetType
        ? vectors
        : new targetType(vectors as Vector);
  } else if (Array.isArray(vectors)) {
    let totalLength = 0;
    for (const vec of vectors) totalLength += vec.length;

    flattenedVectors = new targetType(totalLength);
    let offset = 0;
    for (const vec of vectors) {
      flattenedVectors.set(vec, offset);
      offset += vec.length;
    }
  } else {
    throw new Error("Vectors must be a TypedArray or an array of arrays.");
  }

  if (flattenedVectors.length % dimensions !== 0)
    throw new Error(
      "The size of the flattened vectors must be a multiple of the dimension of the vectors."
    );

  return flattenedVectors;
}

export interface IndexConfig {
  dimensions: number;
  metric: MetricKind;
  quantization: ScalarKind;
  connectivity: number;
  expansion_add: number;
  expansion_search: number;
  multi: boolean;
}

export class Index {
  /**
   * Constructs a new index.
   *
   * @param {(number | {dimensions: number, metric: MetricKind = MetricKind.Cos, quantization: ScalarKind = ScalarKind.F32, connectivity: number = 0, expansion_add: number = 0, expansion_search: number = 0, multi: boolean = false})} dimensionsOrConfigs
   * @param {MetricKind} [metric=MetricKind.Cos] - Optional, default is 'cos'.
   * @param {ScalarKind} [quantization=ScalarKind.F32] - Optional, default is 'f32'.
   * @param {number} [connectivity=0] - Optional, default is 0.
   * @param {number} [expansion_add=0] - Optional, default is 0.
   * @param {number} [expansion_search=0] - Optional, default is 0.
   * @param {boolean} [multi=false] - Optional, default is false.
   * @throws Will throw an error if any of the parameters are of incorrect type or invalid value.
   */
  constructor(
    dimensionsOrConfigs: number | bigint | IndexConfig,
    metric: MetricKind = MetricKind.Cos,
    quantization: ScalarKind = ScalarKind.F32,
    connectivity: number = 0,
    expansion_add: number = 0,
    expansion_search: number = 0,
    multi: boolean = false
  ) {
    let dimensions: number | bigint;
    if (
      (typeof dimensionsOrConfigs === "number" &&
        !Number.isNaN(dimensionsOrConfigs)) ||
      typeof dimensionsOrConfigs === "bigint"
    ) {
      // Parameters are provided as individual arguments
      dimensions = dimensionsOrConfigs;
    } else if (
      typeof dimensionsOrConfigs === "object" &&
      dimensionsOrConfigs !== null
    ) {
      // Parameters are provided as an object
      ({
        dimensions,
        metric = MetricKind.Cos,
        quantization = ScalarKind.F32,
        connectivity = 0,
        expansion_add = 0,
        expansion_search = 0,
        multi = false,
      } = dimensionsOrConfigs);
    } else {
      throw new Error(
        "Invalid arguments. Expected either individual arguments or a single object argument."
      );
    }

    if (
      (typeof dimensions !== 'bigint' && (!Number.isInteger(dimensions) || dimensions <= 0)) ||
      (typeof connectivity !== 'bigint' && (!Number.isInteger(connectivity) || connectivity < 0)) ||
      (typeof expansion_add !== 'bigint' && (!Number.isInteger(expansion_add) || expansion_add < 0)) ||
      (typeof expansion_search !== 'bigint' && (!Number.isInteger(expansion_search) || expansion_search < 0))) {
      throw new Error(
        "`dimensions`, `connectivity`, `expansion_add`, and `expansion_search` must be non-negative integers, with `dimensions` being positive."
      );
    }

    if (typeof multi !== "boolean") {
      throw new Error("`multi` must be a boolean value.");
    }

    if (!Object.values(MetricKind).includes(metric)) {
      throw new Error(
        `Invalid metric: ${metric}. It must be one of: ${Object.values(
          MetricKind
        ).join(", ")}`
      );
    }

    if (!Object.values(ScalarKind).includes(quantization)) {
      throw new Error(
        `Invalid quantization: ${quantization}. It must be one of: ${Object.values(
          ScalarKind
        ).join(", ")}`
      );
    }

    // @ts-expect-error
    this.#compiledIndex = new compiled.CompiledIndex(
      dimensions,
      metric,
      quantization,
      connectivity,
      expansion_add,
      expansion_search,
      multi
    );
  }

  #compiledIndex: CompiledIndex;

  /**
   * Add vectors to the index.
   *
   * This method accepts vectors and their corresponding keys for indexing.
   * Each key should correspond to a vector. If a single key is provided,
   * it is broadcasted to match the number of provided vectors.
   *
   * Vectors should be provided as a flat typed array representing a matrix
   * where each row is a vector to be indexed. The matrix should have a size
   * of n * d, where n is the number of vectors, and d is the dimensionality
   * of the vectors.
   *
   * Keys should be provided as a BigInt or an array-like object of BigInts
   * representing the unique identifier for each vector.
   *
   * @param {bigint|bigint[]|BigUint64Array} keys - Input identifiers for every vector.
   *        If a single key is provided, it is associated with all provided vectors.
   * @param {Float32Array|Float64Array|Int8Array} vectors - Input matrix representing vectors,
   *        matrix of size n * d, where n is the number of vectors, and d is their dimensionality.
   * @param {number} [threads=0] - Optional, default is 0. Number of threads to use for indexing.
   *        If set to 0, the number of threads is determined automatically.
   * @throws Will throw an error if the length of keys doesn't match the number of vectors
   *         or if it's not a single key.
   */
  add(keys: bigint | bigint[] | BigUint64Array, vectors: VectorOrMatrix, threads: number = 0): void {
    let normalizedKeys = normalizeKeys(keys);
    let normalizedVectors = normalizeVectors(
      vectors,
      this.#compiledIndex.dimensions()
    );
    let countVectors =
      normalizedVectors.length / this.#compiledIndex.dimensions();

    // If a single key is provided but there are multiple vectors,
    // broadcast the single key value to match the number of vectors
    if (normalizedKeys.length === 1 && countVectors > 1) {
      normalizedKeys = BigUint64Array.from(
        { length: countVectors },
        () => normalizedKeys[0]
      );
    } else if (normalizedKeys.length !== countVectors) {
      throw new Error(
        `The length of keys (${normalizedKeys.length}) must match the number of vectors (${countVectors}) or be a single key.`
      );
    }

    if ((!Number.isNaN(threads) && typeof threads !== "number") || threads < 0) {
      throw new Error(
        "`threads` must be a non-negative integer representing the number of threads to use for searching."
      );
    }

    // Call the compiled method
    this.#compiledIndex.add(normalizedKeys, normalizedVectors, threads);
  }

  /**
   * Perform a k-nearest neighbor search on the index.
   *
   * This method accepts a matrix of query vectors and returns the closest vectors
   * from the index for each query. The method returns an object containing the keys,
   * distances, and counts of the matches found.
   *
   * Vectors should be provided as a flat typed array representing a matrix where
   * each row is a vector. The matrix should be of size n * d, where n is the
   * number of query vectors, and d is their dimensionality.
   *
   * The parameter `k` specifies the number of nearest neighbors to return for each
   * query vector. If there are not enough results for a query, the result array is
   * padded with -1s.
   *
   * @param {Float32Array|Float64Array|Int8Array|Array<Array<number>>} vectors - Input matrix representing query vectors, can be a TypedArray or an array of TypedArray.
   * @param {number} k - The number of nearest neighbors to search for each query vector.
   * @param {number} [threads=0] - Optional, default is 0. Number of threads to use for searching. If set to 0, the number of threads is determined automatically.
   * @return {Matches|BatchMatches} - Search results for one or more queries, containing keys, distances, and counts of the matches found.
   * @throws Will throw an error if `k` is not a positive integer or if the size of the vectors is not a multiple of dimensions.
   * @throws Will throw an error if `vectors` is not a valid input type (TypedArray or an array of TypedArray) or if its flattened size is not a multiple of dimensions.
   */
  search(vectors: Vector, k: number, threads: number): Matches;
  search(vectors: Matrix, k: number, threads: number): BatchMatches;
  search(vectors: VectorOrMatrix, k: number, threads: number = 0): Matches | BatchMatches {
    if ((!Number.isNaN(k) && typeof k !== "number") || k <= 0) {
      throw new Error(
        "`k` must be a positive integer representing the number of nearest neighbors to search for."
      );
    }
    if ((!Number.isNaN(threads) && typeof threads !== "number") || threads < 0) {
      throw new Error(
        "`threads` must be a non-negative integer representing the number of threads to use for searching."
      );
    }

    const normalizedVectors = normalizeVectors(
      vectors,
      this.#compiledIndex.dimensions()
    );

    // Call the compiled method and create Matches or BatchMatches object with the result
    const result = this.#compiledIndex.search(normalizedVectors, k, threads);
    const countInQueries =
      normalizedVectors.length / Number(this.#compiledIndex.dimensions());
    const batchMatches = new BatchMatches(...result, k);

    if (countInQueries === 1) {
      return batchMatches.get(0);
    } else {
      return batchMatches;
    }
  }

  /**
   * Verifies the presence of one or more keys in the index.
   *
   * This method accepts one or multiple keys as input and returns a boolean or
   * an array of booleans indicating whether each key is present in the index.
   *
   * @param {bigint|bigint[]|BigUint64Array} keys - The identifier(s) of the vector(s) to be checked for presence in the index.
   * @return {boolean|boolean[]} - Returns true if a single key is contained in the index, false otherwise. Returns an array of booleans corresponding to the presence of each key in the index when multiple keys are provided.
   * @throws Will throw an error if keys are not integers.
   */
  contains(keys: bigint | bigint[] | BigUint64Array): boolean | boolean[] {
    let normalizedKeys = normalizeKeys(keys);
    let normalizedResults = this.#compiledIndex.contains(normalizedKeys);
    if (isOneKey(keys)) return normalizedResults[0];
    else return normalizedResults;
  }

  /**
   * Counts the number of times keys shows up in the index.
   *
   * @param {bigint|bigint[]|BigUint64Array} keys - The identifier(s) of the vector(s) to be enumerated.
   * @return {number|number[]} - Returns the number of vectors found when a single key is provided. Returns an array of big integers corresponding to the number of vectors found for each key when multiple keys are provided.
   * @throws Will throw an error if keys are not integers.
   */
  count(keys: bigint | bigint[] | BigUint64Array): number | number[] {
    let normalizedKeys = normalizeKeys(keys);
    let normalizedResults = this.#compiledIndex.count(normalizedKeys);
    if (isOneKey(keys)) return (normalizedResults as unknown as number[])[0];
    else return normalizedResults;
  }

  /**
   * Removes one or multiple vectors from the index.
   *
   * This method accepts one or multiple keys as input and removes the corresponding vectors from the index.
   * It returns the number of vectors actually removed for each key provided.
   *
   * @param {bigint|bigint[]|BigUint64Array} keys - The identifier(s) of the vector(s) to be removed.
   * @return {number|number[]} - Returns the number of vectors deleted when a single key is provided. Returns an array of big integers corresponding to the number of vectors deleted for each key when multiple keys are provided.
   * @throws Will throw an error if keys are not integers.
   */
  remove(keys: bigint | bigint[] | BigUint64Array): number | number[] {
    let normalizedKeys = normalizeKeys(keys);
    let normalizedResults = this.#compiledIndex.remove(normalizedKeys);
    if (isOneKey(keys)) return normalizedResults[0];
    else return normalizedResults;
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
   * Write index to a file.
   * @param {string} path File path to write.
   * @throws Will throw an error if `path` is not a string.
   */
  save(path: string) {
    if (typeof path !== "string")
      throw new Error(
        "`path` must be a string representing the file path to write."
      );
    this.#compiledIndex.save(path);
  }

  /**
   * Load index from a file.
   * @param {string} path File path to read.
   * @throws Will throw an error if `path` is not a string.
   */
  load(path: string) {
    if (typeof path !== "string")
      throw new Error(
        "`path` must be a string representing the file path to read."
      );
    this.#compiledIndex.load(path);
  }

  /**
   * View index from a file, without loading into RAM.
   * @param {string} path File path to read.
   * @throws Will throw an error if `path` is not a string.
   */
  view(path: string) {
    if (typeof path !== "string")
      throw new Error(
        "`path` must be a string representing the file path to read."
      );
    this.#compiledIndex.view(path);
  }
}

type NumberArrayConstructor =
  | Float64ArrayConstructor
  | Float32ArrayConstructor
  | Int8ArrayConstructor;
/**
 * Performs an exact search on the given dataset to find the best matching vectors for each query.
 *
 * @param {Float32Array|Float64Array|Int8Array|Array<Array<number>>} dataset - The dataset containing vectors to be searched. It can be a TypedArray or an array of arrays.
 * @param {Float32Array|Float64Array|Int8Array|Array<Array<number>>} queries - The queries containing vectors to search for in the dataset. It can be a TypedArray or an array of arrays.
 * @param {number} dimensions - The dimensionality of the vectors in both the dataset and the queries. It defines the number of elements in each vector.
 * @param {number} count - The number of nearest neighbors to return for each query. If the dataset contains fewer vectors than the specified count, the result will contain only the available vectors.
 * @param {MetricKind} metric - The distance metric to be used for the search.
 * @param {number} [threads=0] - Optional, default is 0. The number of threads to use for the search. If set to 0, the number of threads is determined automatically.
 * @return {Matches|BatchMatches} - Returns a `Matches` or `BatchMatches` object containing the results of the search.
 * @throws Will throw an error if `dimensions` and `count` are not positive integers.
 * @throws Will throw an error if `metric` is not a valid MetricKind.
 * @throws Will throw an error if `dataset` and `queries` are not valid input types (TypedArray or an array of arrays).
 * @throws Will throw an error if the sizes of the flattened `dataset` and `queries` are not multiples of `dimensions`.
 * @throws Will throw an error if `count` is greater than the number of vectors in the `dataset`.
 *
 * @example
 * const dataset = [[1.0, 2.0], [3.0, 4.0]]; // Two vectors: [1.0, 2.0] and [3.0, 4.0]
 * const queries = [[1.5, 2.5]]; // One vector: [1.5, 2.5]
 * const dimensions = 2; // The number of elements in each vector.
 * const count = 1; // The number of nearest neighbors to return for each query.
 * const metric = MetricKind.IP; // Using the Inner Product distance metric.
 * const threads = 0; // How many threads to use to perform the search.
 *
 * const result = exactSearch(dataset, queries, dimensions, count, metric, threads);
 * // result might be:
 * // {
 * //    keys: BigUint64Array [ 1n ],
 * //    distances: Float32Array [ some_value ],
 * // }
 */
export function exactSearch(
  dataset: VectorOrMatrix,
  queries: VectorOrMatrix,
  dimensions: number,
  count: number,
  metric: MetricKind,
  threads: number = 0
): Matches | BatchMatches {
  // Validate and normalize the dimensions and count
  dimensions = Number(dimensions);
  count = Number(count);
  if (count <= 0 || dimensions <= 0) {
    throw new Error("Dimensions and count must be positive integers.");
  }
  if ((!Number.isNaN(threads) && typeof threads !== "number") || threads < 0) {
    throw new Error(
      "`threads` must be a non-negative integer representing the number of threads to use for searching."
    );
  }

  // Validate metric
  if (!Object.values(MetricKind).includes(metric)) {
    throw new Error(
      `Invalid metric: ${metric}. It must be one of: ${Object.values(
        MetricKind
      ).join(", ")}`
    );
  }

  // Flatten and normalize dataset and queries if they are arrays of arrays
  let targetType: NumberArrayConstructor;
  if (dataset instanceof Float64Array) targetType = Float64Array;
  else if (dataset instanceof Int8Array) targetType = Int8Array;
  else targetType = Float32Array; // default to Float32Array if dataset is not Float64Array or Int8Array

  dataset = normalizeVectors(dataset, dimensions, targetType);
  queries = normalizeVectors(queries, dimensions, targetType);
  const countInDataset = dataset.length / dimensions;
  const countInQueries = queries.length / dimensions;
  if (count > countInDataset) {
    throw new Error(
      "Count must be equal or smaller than the number of vectors in the dataset."
    );
  }

  // Call the compiled function with the normalized input
  const result = compiled.exactSearch(
    dataset,
    queries,
    dimensions,
    count,
    metric,
    threads
  );

  // Create and return a Matches or BatchMatches object with the result
  if (countInQueries == 1) {
    return new Matches(result[0], result[1]);
  } else {
    return new BatchMatches(...result, count);
  }
}

const usearch = {
  Index,
  MetricKind,
  ScalarKind,
  Matches,
  BatchMatches,
  exactSearch,
};
export default usearch;

// utility functions to help find native builds

function getBuildDir(dir: string) {
  if (existsSync(path.join(dir, "build"))) return dir;
  if (existsSync(path.join(dir, "prebuilds"))) return dir;
  if (path.basename(dir) === ".next") {
    // special case for next.js on custom node (not vercel)
    const sideways = path.join(dir, "..", "node_modules", "usearch");
    if (existsSync(sideways)) return getBuildDir(sideways);
  }
  if (dir === "/") throw new Error("Could not find native build for usearch");
  return getBuildDir(path.join(dir, ".."));
}

function getDirName() {
  try {
    if (__dirname) return __dirname;
  } catch (e) { }
  return getRoot(getFileName());
}

// dummy code for ncc to include the native module
if (process.uptime() < 0) {
  require(__dirname + "/../../../prebuilds/darwin-arm64+x64/usearch.node");
  require(__dirname + "/../../../prebuilds/linux-arm64/usearch.node");
  require(__dirname + "/../../../prebuilds/linux-x64/usearch.node");
  require(__dirname + "/../../../prebuilds/win32-ia32/usearch.node");
  require(__dirname + "/../../../prebuilds/win32-x64/usearch.node");
  require(__dirname + "/../../../build/Release/usearch.node");
}
