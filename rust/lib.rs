//! # USearch Crate for Rust
//!
//! `usearch` is a high-performance library for Approximate Nearest Neighbor (ANN) search in high-dimensional spaces.
//! It offers efficient and scalable solutions for indexing and querying dense vector spaces with support for multiple distance metrics and vector types.
//!
//! This crate wraps the native functionalities of USearch, providing Rust-friendly interfaces and integration capabilities.
//! It is designed to facilitate rapid development and deployment of applications requiring fast and accurate vector search functionalities, such as recommendation systems, image retrieval systems, and natural language processing tasks.
//!
//! ## Features
//!
//! - SIMD-accelerated distance calculations for various metrics.
//! - Support for `f32`, `f64`, `i8`, custom `f16`, and binary (`b1x8`) vector types.
//! - Extensible with custom distance metrics and filtering predicates.
//! - Efficient serialization and deserialization for persistence and network transfers.
//!
//! ## Quick Start
//!
//! Refer to the `Index` struct for detailed usage examples.

/// Returns the version of the USearch crate.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// The key type used to identify vectors in the index.
/// It is a 64-bit unsigned integer.
pub type Key = u64;

/// The distance type used to represent the similarity between vectors.
/// It is a 32-bit floating-point number.
pub type Distance = f32;

/// Callback signature for custom metric functions, defined in the Rust layer and used in the C++ layer.
pub type StatefulMetric = unsafe extern "C" fn(
    *const std::ffi::c_void,
    *const std::ffi::c_void,
    *mut std::ffi::c_void,
) -> Distance;

/// Callback signature for custom predicate functions, defined in the Rust layer and used in the C++ layer.
pub type StatefulPredicate = unsafe extern "C" fn(Key, *mut std::ffi::c_void) -> bool;

/// Represents errors that can occur when addressing bits.
#[derive(Debug)]
pub enum BitAddressableError {
    /// Error indicating the specified index is out of the allowable range.
    IndexOutOfRange,
}

impl std::fmt::Display for BitAddressableError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match *self {
            BitAddressableError::IndexOutOfRange => write!(f, "Index out of range"),
        }
    }
}

impl std::error::Error for BitAddressableError {}

/// Trait for types that can be addressed at the bit level.
/// Provides methods to set and get individual bits within the implementing type.
pub trait BitAddressable {
    /// Sets a bit at the specified index.
    /// Returns an error if the index is out of range.
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the bit to set.
    /// * `value` - The value to set the bit to (`true` for 1, `false` for 0).
    fn set_bit(&mut self, index: usize, value: bool) -> Result<(), BitAddressableError>;

    /// Gets the value of a bit at the specified index.
    /// Returns an error if the index is out of range.
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the bit to retrieve.
    fn get_bit(&self, index: usize) -> Result<bool, BitAddressableError>;
}

/// A byte-wide bit vector type that provides low-level control over individual bits.
///
/// This struct represents a single byte (8 bits) and enables manipulation and
/// interpretation of individual bits via various utility functions.
#[repr(transparent)]
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Eq, PartialEq)]
pub struct b1x8(pub u8);

impl b1x8 {
    /// Casts a slice of `u8` bytes to a slice of `b1x8`, allowing bit-level operations on byte slices.
    pub fn from_u8s(slice: &[u8]) -> &[Self] {
        unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const Self, slice.len()) }
    }

    /// Casts a mutable slice of `u8` bytes to a mutable slice of `b1x8`, enabling mutable
    /// bit-level operations on byte slices.
    pub fn from_mut_u8s(slice: &mut [u8]) -> &mut [Self] {
        unsafe { std::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut Self, slice.len()) }
    }

    /// Converts a slice of `b1x8` back to a slice of `u8`, useful for reading bit-level manipulations
    /// in byte-oriented contexts.
    pub fn to_u8s(slice: &[Self]) -> &[u8] {
        unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, slice.len()) }
    }

    /// Converts a mutable slice of `b1x8` back to a mutable slice of `u8`, enabling further
    /// modifications on the original byte data after bit-level manipulations.
    pub fn to_mut_u8s(slice: &mut [Self]) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut u8, slice.len()) }
    }
}

/// A struct representing a half-precision floating-point number based on the IEEE 754 standard.
///
/// This struct uses an `i16` to store the half-precision floating-point data, which includes
/// 1 sign bit, 5 exponent bits, and 10 mantissa bits.
#[repr(transparent)]
#[allow(non_camel_case_types)]
#[derive(Clone, Copy)]
pub struct f16(i16);

impl f16 {
    /// Casts a slice of `i16` integers to a slice of `f16`, allowing operations on half-precision
    /// floating-point data stored in standard 16-bit integer arrays.
    pub fn from_i16s(slice: &[i16]) -> &[Self] {
        unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const Self, slice.len()) }
    }

    /// Casts a mutable slice of `i16` integers to a mutable slice of `f16`, enabling mutable operations
    /// on half-precision floating-point data.
    pub fn from_mut_i16s(slice: &mut [i16]) -> &mut [Self] {
        unsafe { std::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut Self, slice.len()) }
    }

    /// Converts a slice of `f16` back to a slice of `i16`, useful for storage or manipulation in formats
    /// that require standard integer types.
    pub fn to_i16s(slice: &[Self]) -> &[i16] {
        unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const i16, slice.len()) }
    }

    /// Converts a mutable slice of `f16` back to a mutable slice of `i16`, enabling further
    /// modifications on the original integer data after operations involving half-precision
    /// floating-point numbers.
    pub fn to_mut_i16s(slice: &mut [Self]) -> &mut [i16] {
        unsafe { std::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut i16, slice.len()) }
    }
}

impl BitAddressable for b1x8 {
    /// Sets a bit at a specific index within the byte.
    ///
    /// # Arguments
    ///
    /// * `index` - The 0-based index of the bit to set, ranging from 0 to 7.
    /// * `value` - The boolean value to assign to the bit (`true` for 1, `false` for 0).
    ///
    /// # Returns
    ///
    /// This method returns `Ok(())` if the bit was successfully set, or an `Err(BitAddressableError::IndexOutOfRange)`
    /// if the provided index is outside the valid range.
    fn set_bit(&mut self, index: usize, value: bool) -> Result<(), BitAddressableError> {
        if index >= 8 {
            Err(BitAddressableError::IndexOutOfRange)
        } else {
            if value {
                self.0 |= 1 << index;
            } else {
                self.0 &= !(1 << index);
            }
            Ok(())
        }
    }

    /// Retrieves the value of a bit at a specific index within the byte.
    ///
    /// # Arguments
    ///
    /// * `index` - The 0-based index of the bit to retrieve, ranging from 0 to 7.
    ///
    /// # Returns
    ///
    /// Returns `Ok(true)` if the bit is set (1), `Ok(false)` if the bit is not set (0),
    /// or an `Err(BitAddressableError::IndexOutOfRange)` if the provided index is outside
    /// the valid range.
    fn get_bit(&self, index: usize) -> Result<bool, BitAddressableError> {
        if index >= 8 {
            Err(BitAddressableError::IndexOutOfRange)
        } else {
            Ok(((self.0 >> index) & 1) == 1)
        }
    }
}

impl BitAddressable for [b1x8] {
    /// Sets a bit at a specific index across the slice of `b1x8`.
    fn set_bit(&mut self, index: usize, value: bool) -> Result<(), BitAddressableError> {
        let byte_index = index / 8;
        let bit_index = index % 8;
        if byte_index >= self.len() {
            Err(BitAddressableError::IndexOutOfRange)
        } else {
            self[byte_index].set_bit(bit_index, value)
        }
    }

    /// Gets a bit at a specific index across the slice of `b1x8`.
    fn get_bit(&self, index: usize) -> Result<bool, BitAddressableError> {
        let byte_index = index / 8;
        let bit_index = index % 8;
        if byte_index >= self.len() {
            Err(BitAddressableError::IndexOutOfRange)
        } else {
            self[byte_index].get_bit(bit_index)
        }
    }
}

impl PartialEq for f16 {
    fn eq(&self, other: &Self) -> bool {
        // Check for NaN values first (exponent all ones and non-zero mantissa)
        let nan_self = (self.0 & 0x7C00) == 0x7C00 && (self.0 & 0x03FF) != 0;
        let nan_other = (other.0 & 0x7C00) == 0x7C00 && (other.0 & 0x03FF) != 0;
        if nan_self || nan_other {
            return false;
        }

        self.0 == other.0
    }
}

impl std::fmt::Debug for b1x8 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:08b}", self.0)
    }
}

impl std::fmt::Debug for f16 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let bits = self.0;
        let sign = (bits >> 15) & 1;
        let exponent = (bits >> 10) & 0x1F;
        let mantissa = bits & 0x3FF;
        write!(f, "{}|{:05b}|{:010b}", sign, exponent, mantissa)
    }
}

#[cxx::bridge]
pub mod ffi {

    /// The metric kind used to differentiate built-in distance functions.
    #[derive(Debug)]
    #[repr(i32)]
    enum MetricKind {
        Unknown,
        /// The Inner Product metric, defined as `IP = 1 - sum(a[i] * b[i])`.
        IP,
        /// The squared Euclidean Distance metric, defined as `L2 = sum((a[i] - b[i])^2)`.
        L2sq,
        /// The Cosine Similarity metric, defined as `Cos = 1 - sum(a[i] * b[i]) / (sqrt(sum(a[i]^2) * sqrt(sum(b[i]^2)))`.
        Cos,
        /// The Pearson Correlation metric.
        Pearson,
        /// The Haversine (Great Circle) Distance metric.
        Haversine,
        /// The Jensen Shannon Divergence metric.
        Divergence,
        /// The bit-level Hamming Distance metric, defined as the number of differing bits.
        Hamming,
        /// The bit-level Tanimoto (Jaccard) metric, defined as the number of intersecting bits divided by the number of union bits.
        Tanimoto,
        /// The bit-level Sorensen metric.
        Sorensen,
    }

    /// The scalar kind used to differentiate built-in vector element types.
    #[derive(Debug)]
    #[repr(i32)]
    enum ScalarKind {
        Unknown,
        /// 64-bit double-precision IEEE 754 floating-point number.
        F64,
        /// 32-bit single-precision IEEE 754 floating-point number.
        F32,
        /// 16-bit half-precision IEEE 754 floating-point number (different from `bf16`).
        F16,
        /// 16-bit brain floating-point number.
        BF16,
        /// 8-bit signed integer.
        I8,
        /// 1-bit binary value, packed 8 per byte.
        B1,
    }

    /// The resulting matches from a search operation.
    /// It contains the keys and distances of the closest vectors.
    #[derive(Debug)]
    struct Matches {
        keys: Vec<u64>,
        distances: Vec<f32>,
    }

    /// The index options used to configure the dense index during creation.
    /// It contains the number of dimensions, the metric kind, the scalar kind,
    /// the connectivity, the expansion values, and the multi-flag.
    #[derive(Debug, PartialEq)]
    struct IndexOptions {
        dimensions: usize,
        metric: MetricKind,
        quantization: ScalarKind,
        connectivity: usize,
        expansion_add: usize,
        expansion_search: usize,
        multi: bool,
    }

    // C++ types and signatures exposed to Rust.
    unsafe extern "C++" {
        include!("lib.hpp");

        /// Low-level C++ interface that is further wrapped into the high-level `Index`
        type NativeIndex;

        pub fn expansion_add(self: &NativeIndex) -> usize;
        pub fn expansion_search(self: &NativeIndex) -> usize;
        pub fn change_expansion_add(self: &NativeIndex, n: usize);
        pub fn change_expansion_search(self: &NativeIndex, n: usize);
        pub fn change_metric_kind(self: &NativeIndex, metric: MetricKind);

        /// Changes the metric function used to calculate the distance between vectors.
        /// Avoids the `std::ffi::c_void` type and the `StatefulMetric` type, that the FFI
        /// does not support, replacing them with basic pointer-sized integer types.
        /// The first two arguments are the pointers to the vectors to compare, and the third
        /// argument is the `metric_state` propagated from the Rust layer.
        pub fn change_metric(self: &NativeIndex, metric: usize, metric_state: usize);

        pub fn new_native_index(options: &IndexOptions) -> Result<UniquePtr<NativeIndex>>;
        pub fn reserve(self: &NativeIndex, capacity: usize) -> Result<()>;
        pub fn reserve_capacity_and_threads(
            self: &NativeIndex,
            capacity: usize,
            threads: usize,
        ) -> Result<()>;

        pub fn dimensions(self: &NativeIndex) -> usize;
        pub fn connectivity(self: &NativeIndex) -> usize;
        pub fn size(self: &NativeIndex) -> usize;
        pub fn capacity(self: &NativeIndex) -> usize;
        pub fn serialized_length(self: &NativeIndex) -> usize;

        pub fn add_b1x8(self: &NativeIndex, key: u64, vector: &[u8]) -> Result<()>;
        pub fn add_i8(self: &NativeIndex, key: u64, vector: &[i8]) -> Result<()>;
        pub fn add_f16(self: &NativeIndex, key: u64, vector: &[i16]) -> Result<()>;
        pub fn add_f32(self: &NativeIndex, key: u64, vector: &[f32]) -> Result<()>;
        pub fn add_f64(self: &NativeIndex, key: u64, vector: &[f64]) -> Result<()>;

        pub fn search_b1x8(self: &NativeIndex, query: &[u8], count: usize) -> Result<Matches>;
        pub fn search_i8(self: &NativeIndex, query: &[i8], count: usize) -> Result<Matches>;
        pub fn search_f16(self: &NativeIndex, query: &[i16], count: usize) -> Result<Matches>;
        pub fn search_f32(self: &NativeIndex, query: &[f32], count: usize) -> Result<Matches>;
        pub fn search_f64(self: &NativeIndex, query: &[f64], count: usize) -> Result<Matches>;

        pub fn exact_search_b1x8(self: &NativeIndex, query: &[u8], count: usize)
            -> Result<Matches>;
        pub fn exact_search_i8(self: &NativeIndex, query: &[i8], count: usize) -> Result<Matches>;
        pub fn exact_search_f16(self: &NativeIndex, query: &[i16], count: usize)
            -> Result<Matches>;
        pub fn exact_search_f32(self: &NativeIndex, query: &[f32], count: usize)
            -> Result<Matches>;
        pub fn exact_search_f64(self: &NativeIndex, query: &[f64], count: usize)
            -> Result<Matches>;

        pub fn filtered_search_b1x8(
            self: &NativeIndex,
            query: &[u8],
            count: usize,
            filter: usize,
            filter_state: usize,
        ) -> Result<Matches>;
        pub fn filtered_search_i8(
            self: &NativeIndex,
            query: &[i8],
            count: usize,
            filter: usize,
            filter_state: usize,
        ) -> Result<Matches>;
        pub fn filtered_search_f16(
            self: &NativeIndex,
            query: &[i16],
            count: usize,
            filter: usize,
            filter_state: usize,
        ) -> Result<Matches>;
        pub fn filtered_search_f32(
            self: &NativeIndex,
            query: &[f32],
            count: usize,
            filter: usize,
            filter_state: usize,
        ) -> Result<Matches>;
        pub fn filtered_search_f64(
            self: &NativeIndex,
            query: &[f64],
            count: usize,
            filter: usize,
            filter_state: usize,
        ) -> Result<Matches>;

        pub fn get_b1x8(self: &NativeIndex, key: u64, buffer: &mut [u8]) -> Result<usize>;
        pub fn get_i8(self: &NativeIndex, key: u64, buffer: &mut [i8]) -> Result<usize>;
        pub fn get_f16(self: &NativeIndex, key: u64, buffer: &mut [i16]) -> Result<usize>;
        pub fn get_f32(self: &NativeIndex, key: u64, buffer: &mut [f32]) -> Result<usize>;
        pub fn get_f64(self: &NativeIndex, key: u64, buffer: &mut [f64]) -> Result<usize>;

        pub fn remove(self: &NativeIndex, key: u64) -> Result<usize>;
        pub fn rename(self: &NativeIndex, from: u64, to: u64) -> Result<usize>;
        pub fn contains(self: &NativeIndex, key: u64) -> bool;
        pub fn count(self: &NativeIndex, key: u64) -> usize;

        pub fn save(self: &NativeIndex, path: &str) -> Result<()>;
        pub fn load(self: &NativeIndex, path: &str) -> Result<()>;
        pub fn view(self: &NativeIndex, path: &str) -> Result<()>;
        pub fn reset(self: &NativeIndex) -> Result<()>;
        pub fn memory_usage(self: &NativeIndex) -> usize;
        pub fn hardware_acceleration(self: &NativeIndex) -> *const c_char;

        pub fn save_to_buffer(self: &NativeIndex, buffer: &mut [u8]) -> Result<()>;
        pub fn load_from_buffer(self: &NativeIndex, buffer: &[u8]) -> Result<()>;
        pub fn view_from_buffer(self: &NativeIndex, buffer: &[u8]) -> Result<()>;
    }
}

// Re-export the FFI structs and enums at the crate root for easy access
pub use ffi::{IndexOptions, MetricKind, ScalarKind};

/// Represents custom metric functions for calculating distances between vectors in various formats.
///
/// This enum allows the encapsulation of custom distance calculation logic for vectors of different
/// data types, facilitating the use of custom metrics in vector space operations. Each variant of this
/// enum holds a boxed function pointer (`std::boxed::Box<dyn Fn(...) -> Distance + Send + Sync>`) that defines
/// the distance calculation between two vectors of a specific type. The function returns a `Distance`, which
/// is typically a floating-point value representing the calculated distance between the two vectors.
///
/// # Variants
///
/// - `B1X8Metric`: A metric function for binary vectors packed in `u8` containers, represented here by `b1x8`.
/// - `I8Metric`: A metric function for vectors of 8-bit signed integers (`i8`).
/// - `F16Metric`: A metric function for vectors of 16-bit half-precision floating-point numbers (`f16`).
/// - `F32Metric`: A metric function for vectors of 32-bit floating-point numbers (`f32`).
/// - `F64Metric`: A metric function for vectors of 64-bit floating-point numbers (`f64`).
///
/// Each metric function takes two pointers to the vectors of the respective type and returns a `Distance`.
///
/// # Usage
///
/// Custom metric functions can be used to define how distances are calculated between vectors, enabling
/// the implementation of various distance metrics such as Euclidean distance, Manhattan distance, or
/// Cosine similarity, depending on the specific requirements of the application.
///
/// # Safety
///
/// Since these functions operate on raw pointers, care must be taken to ensure that the pointers are valid
/// and that the lifetime of the referenced data extends at least as long as the lifetime of the metric
/// function's use. Improper use of these functions can lead to undefined behavior.
///
/// # Examples
///
/// ```
/// use usearch::{Distance, f16, b1x8};
///
/// let euclidean_fn = Box::new(|a: *const f32, b: *const f32| -> f32 {
///     let dimensions = 256;
///     let a = unsafe { std::slice::from_raw_parts(a, dimensions) };
///     let b = unsafe { std::slice::from_raw_parts(b, dimensions) };
///     a.iter().zip(b.iter())
///         .map(|(a, b)| (*a - *b).powi(2))
///         .sum::<f32>()
///         .sqrt()
/// });
/// ```
///
/// In this example, `dimensions` should be defined and valid for the vectors `a` and `b`.
pub enum MetricFunction {
    B1X8Metric(*mut std::boxed::Box<dyn Fn(*const b1x8, *const b1x8) -> Distance + Send + Sync>),
    I8Metric(*mut std::boxed::Box<dyn Fn(*const i8, *const i8) -> Distance + Send + Sync>),
    F16Metric(*mut std::boxed::Box<dyn Fn(*const f16, *const f16) -> Distance + Send + Sync>),
    F32Metric(*mut std::boxed::Box<dyn Fn(*const f32, *const f32) -> Distance + Send + Sync>),
    F64Metric(*mut std::boxed::Box<dyn Fn(*const f64, *const f64) -> Distance + Send + Sync>),
}

/// Approximate Nearest Neighbors search index for dense vectors.
///
/// The `Index` struct provides an abstraction over a dense vector space, allowing
/// for efficient addition, search, and management of high-dimensional vectors.
/// It supports various distance metrics and vector types through generic interfaces.
///
/// # Examples
///
/// Basic usage:
///
/// ```rust
/// use usearch::{Index, IndexOptions, MetricKind, ScalarKind};
///
/// let mut options = IndexOptions::default();
/// options.dimensions = 4; // Set the number of dimensions for vectors
/// options.metric = MetricKind::Cos; // Use cosine similarity for distance measurement
/// options.quantization = ScalarKind::F32; // Use 32-bit floating point numbers
///
/// let index = Index::new(&options).expect("Failed to create index.");
/// index.reserve(1000).expect("Failed to reserve capacity.");
///
/// // Add vectors to the index
/// let vector1: Vec<f32> = vec![0.0, 1.0, 0.0, 1.0];
/// let vector2: Vec<f32> = vec![1.0, 0.0, 1.0, 0.0];
/// index.add(1, &vector1).expect("Failed to add vector1.");
/// index.add(2, &vector2).expect("Failed to add vector2.");
///
/// // Search for the nearest neighbors to a query vector
/// let query: Vec<f32> = vec![0.5, 0.5, 0.5, 0.5];
/// let results = index.search(&query, 5).expect("Search failed.");
/// for (key, distance) in results.keys.iter().zip(results.distances.iter()) {
///     println!("Key: {}, Distance: {}", key, distance);
/// }
/// ```
/// For more examples, including how to add vectors to the index and perform searches,
/// refer to the individual method documentation.
pub struct Index {
    inner: cxx::UniquePtr<ffi::NativeIndex>,
    metric_fn: Option<MetricFunction>,
}

unsafe impl Send for Index {}
unsafe impl Sync for Index {}

impl Drop for Index {
    fn drop(&mut self) {
        if let Some(metric) = &self.metric_fn {
            match metric {
                MetricFunction::B1X8Metric(pointer) => unsafe {
                    drop(Box::from_raw(*pointer));
                },
                MetricFunction::I8Metric(pointer) => unsafe {
                    drop(Box::from_raw(*pointer));
                },
                MetricFunction::F16Metric(pointer) => unsafe {
                    drop(Box::from_raw(*pointer));
                },
                MetricFunction::F32Metric(pointer) => unsafe {
                    drop(Box::from_raw(*pointer));
                },
                MetricFunction::F64Metric(pointer) => unsafe {
                    drop(Box::from_raw(*pointer));
                },
            }
        }
    }
}

impl Default for ffi::IndexOptions {
    fn default() -> Self {
        Self {
            dimensions: 256,
            metric: MetricKind::Cos,
            quantization: ScalarKind::BF16,
            connectivity: 0,
            expansion_add: 0,
            expansion_search: 0,
            multi: false,
        }
    }
}

impl Clone for ffi::IndexOptions {
    fn clone(&self) -> Self {
        ffi::IndexOptions {
            dimensions: (self.dimensions),
            metric: (self.metric),
            quantization: (self.quantization),
            connectivity: (self.connectivity),
            expansion_add: (self.expansion_add),
            expansion_search: (self.expansion_search),
            multi: (self.multi),
        }
    }
}

/// The `VectorType` trait defines operations for managing and querying vectors
/// in an index. It supports generic operations on vectors of different types,
/// allowing for the addition, retrieval, and search of vectors within an index.
pub trait VectorType {
    /// Adds a vector to the index under the specified key.
    ///
    /// # Parameters
    /// - `index`: A reference to the `Index` where the vector is to be added.
    /// - `key`: The key under which the vector should be stored.
    /// - `vector`: A slice representing the vector to be added.
    ///
    /// # Returns
    /// - `Ok(())` if the vector was successfully added to the index.
    /// - `Err(cxx::Exception)` if an error occurred during the operation.
    fn add(index: &Index, key: Key, vector: &[Self]) -> Result<(), cxx::Exception>
    where
        Self: Sized;

    /// Retrieves a vector from the index by its key.
    ///
    /// # Parameters
    /// - `index`: A reference to the `Index` from which the vector is to be retrieved.
    /// - `key`: The key of the vector to retrieve.
    /// - `buffer`: A mutable slice where the retrieved vector will be stored. The size of the
    ///   buffer determines the maximum number of elements that can be retrieved.
    ///
    /// # Returns
    /// - `Ok(usize)` indicating the number of elements actually written into the `buffer`.
    /// - `Err(cxx::Exception)` if an error occurred during the operation.
    fn get(index: &Index, key: Key, buffer: &mut [Self]) -> Result<usize, cxx::Exception>
    where
        Self: Sized;

    /// Performs a search in the index using the given query vector, returning
    /// up to `count` closest matches.
    ///
    /// # Parameters
    /// - `index`: A reference to the `Index` where the search is to be performed.
    /// - `query`: A slice representing the query vector.
    /// - `count`: The maximum number of matches to return.
    ///
    /// # Returns
    /// - `Ok(ffi::Matches)` containing the matches found.
    /// - `Err(cxx::Exception)` if an error occurred during the search operation.
    fn search(index: &Index, query: &[Self], count: usize) -> Result<ffi::Matches, cxx::Exception>
    where
        Self: Sized;

    /// Performs an exact (brute force) search in the index using the given query vector, returning
    /// up to `count` closest matches. This search checks all vectors in the index, guaranteeing to find
    /// the true nearest neighbors, but will be slower especially for large indices.
    ///
    /// # Parameters
    /// - `index`: A reference to the `Index` where the search is to be performed.
    /// - `query`: A slice representing the query vector.
    /// - `count`: The maximum number of matches to return.
    ///
    /// # Returns
    /// - `Ok(ffi::Matches)` containing the matches found.
    /// - `Err(cxx::Exception)` if an error occurred during the search operation.
    fn exact_search(
        index: &Index,
        query: &[Self],
        count: usize,
    ) -> Result<ffi::Matches, cxx::Exception>
    where
        Self: Sized;

    /// Performs a filtered search in the index using a query vector and a custom
    /// filter function, returning up to `count` matches that satisfy the filter.
    ///
    /// # Parameters
    /// - `index`: A reference to the `Index` where the search is to be performed.
    /// - `query`: A slice representing the query vector.
    /// - `count`: The maximum number of matches to return.
    /// - `filter`: A closure that takes a `Key` and returns `true` if the corresponding
    ///   vector should be included in the search results, or `false` otherwise.
    ///
    /// # Returns
    /// - `Ok(ffi::Matches)` containing the matches that satisfy the filter.
    /// - `Err(cxx::Exception)` if an error occurred during the filtered search operation.
    fn filtered_search<F>(
        index: &Index,
        query: &[Self],
        count: usize,
        filter: F,
    ) -> Result<ffi::Matches, cxx::Exception>
    where
        Self: Sized,
        F: Fn(Key) -> bool;

    /// Changes the metric used for distance calculations within the index.
    ///
    /// # Parameters
    /// - `index`: A mutable reference to the `Index` for which the metric is to be changed.
    /// - `metric`: A boxed closure that defines the new metric for distance calculation. The
    ///   closure must take two pointers to elements of type `Self` and return a `Distance`.
    ///
    /// # Returns
    /// - `Ok(())` if the metric was successfully changed.
    /// - `Err(cxx::Exception)` if an error occurred during the operation.
    fn change_metric(
        index: &mut Index,
        metric: std::boxed::Box<dyn Fn(*const Self, *const Self) -> Distance + Send + Sync>,
    ) -> Result<(), cxx::Exception>
    where
        Self: Sized;
}

impl VectorType for f32 {
    fn search(index: &Index, query: &[Self], count: usize) -> Result<ffi::Matches, cxx::Exception> {
        index.inner.search_f32(query, count)
    }

    fn exact_search(
        index: &Index,
        query: &[Self],
        count: usize,
    ) -> Result<ffi::Matches, cxx::Exception> {
        index.inner.exact_search_f32(query, count)
    }

    fn get(index: &Index, key: Key, vector: &mut [Self]) -> Result<usize, cxx::Exception> {
        index.inner.get_f32(key, vector)
    }

    fn add(index: &Index, key: Key, vector: &[Self]) -> Result<(), cxx::Exception> {
        index.inner.add_f32(key, vector)
    }

    fn filtered_search<F>(
        index: &Index,
        query: &[Self],
        count: usize,
        filter: F,
    ) -> Result<ffi::Matches, cxx::Exception>
    where
        Self: Sized,
        F: Fn(Key) -> bool,
    {
        // Trampoline is the function that knows how to call the Rust closure.
        extern "C" fn trampoline<F: Fn(u64) -> bool>(key: u64, closure_address: usize) -> bool {
            let closure = closure_address as *const F;
            unsafe { (*closure)(key) }
        }

        // Temporarily cast the closure to a raw pointer for passing.
        let trampoline_fn: usize = trampoline::<F> as *const () as usize;
        let closure_address: usize = &filter as *const F as usize;
        index
            .inner
            .filtered_search_f32(query, count, trampoline_fn, closure_address)
    }

    fn change_metric(
        index: &mut Index,
        metric: std::boxed::Box<dyn Fn(*const Self, *const Self) -> Distance + Send + Sync>,
    ) -> Result<(), cxx::Exception> {
        // Store the metric function in the Index.
        type MetricFn = Box<dyn Fn(*const f32, *const f32) -> Distance>;
        index.metric_fn = Some(MetricFunction::F32Metric(Box::into_raw(Box::new(metric))));

        // Trampoline is the function that knows how to call the Rust closure.
        // The `first` is a pointer to the first vector, `second` is a pointer to the second vector,
        // and `index_wrapper` is a pointer to the `index` itself, from which we can infer the metric function
        // and the number of dimensions.
        extern "C" fn trampoline(first: usize, second: usize, closure_address: usize) -> Distance {
            let first_ptr = first as *const f32;
            let second_ptr = second as *const f32;
            let closure: *mut MetricFn = closure_address as *mut MetricFn;
            unsafe { (*closure)(first_ptr, second_ptr) }
        }

        let trampoline_fn: usize = trampoline as *const () as usize;
        let closure_address = match index.metric_fn {
            Some(MetricFunction::F32Metric(metric)) => metric as *mut () as usize,
            _ => panic!("Expected F32Metric"),
        };
        index.inner.change_metric(trampoline_fn, closure_address);

        Ok(())
    }
}

impl VectorType for i8 {
    fn search(index: &Index, query: &[Self], count: usize) -> Result<ffi::Matches, cxx::Exception> {
        index.inner.search_i8(query, count)
    }

    fn exact_search(
        index: &Index,
        query: &[Self],
        count: usize,
    ) -> Result<ffi::Matches, cxx::Exception> {
        index.inner.exact_search_i8(query, count)
    }

    fn get(index: &Index, key: Key, vector: &mut [Self]) -> Result<usize, cxx::Exception> {
        index.inner.get_i8(key, vector)
    }

    fn add(index: &Index, key: Key, vector: &[Self]) -> Result<(), cxx::Exception> {
        index.inner.add_i8(key, vector)
    }

    fn filtered_search<F>(
        index: &Index,
        query: &[Self],
        count: usize,
        filter: F,
    ) -> Result<ffi::Matches, cxx::Exception>
    where
        Self: Sized,
        F: Fn(Key) -> bool,
    {
        // Trampoline is the function that knows how to call the Rust closure.
        extern "C" fn trampoline<F: Fn(u64) -> bool>(key: u64, closure_address: usize) -> bool {
            let closure = closure_address as *const F;
            unsafe { (*closure)(key) }
        }

        // Temporarily cast the closure to a raw pointer for passing.
        let trampoline_fn: usize = trampoline::<F> as *const () as usize;
        let closure_address: usize = &filter as *const F as usize;
        index
            .inner
            .filtered_search_i8(query, count, trampoline_fn, closure_address)
    }
    fn change_metric(
        index: &mut Index,
        metric: std::boxed::Box<dyn Fn(*const Self, *const Self) -> Distance + Send + Sync>,
    ) -> Result<(), cxx::Exception> {
        // Store the metric function in the Index.
        type MetricFn = Box<dyn Fn(*const i8, *const i8) -> Distance>;
        index.metric_fn = Some(MetricFunction::I8Metric(Box::into_raw(Box::new(metric))));

        // Trampoline is the function that knows how to call the Rust closure.
        // The `first` is a pointer to the first vector, `second` is a pointer to the second vector,
        // and `index_wrapper` is a pointer to the `index` itself, from which we can infer the metric function
        // and the number of dimensions.
        extern "C" fn trampoline(first: usize, second: usize, closure_address: usize) -> Distance {
            let first_ptr = first as *const i8;
            let second_ptr = second as *const i8;
            let closure: *mut MetricFn = closure_address as *mut MetricFn;
            unsafe { (*closure)(first_ptr, second_ptr) }
        }

        let trampoline_fn: usize = trampoline as *const () as usize;
        let closure_address = match index.metric_fn {
            Some(MetricFunction::I8Metric(metric)) => metric as *mut () as usize,
            _ => panic!("Expected I8Metric"),
        };
        index.inner.change_metric(trampoline_fn, closure_address);

        Ok(())
    }
}

impl VectorType for f64 {
    fn search(index: &Index, query: &[Self], count: usize) -> Result<ffi::Matches, cxx::Exception> {
        index.inner.search_f64(query, count)
    }

    fn exact_search(
        index: &Index,
        query: &[Self],
        count: usize,
    ) -> Result<ffi::Matches, cxx::Exception> {
        index.inner.exact_search_f64(query, count)
    }

    fn get(index: &Index, key: Key, vector: &mut [Self]) -> Result<usize, cxx::Exception> {
        index.inner.get_f64(key, vector)
    }

    fn add(index: &Index, key: Key, vector: &[Self]) -> Result<(), cxx::Exception> {
        index.inner.add_f64(key, vector)
    }

    fn filtered_search<F>(
        index: &Index,
        query: &[Self],
        count: usize,
        filter: F,
    ) -> Result<ffi::Matches, cxx::Exception>
    where
        Self: Sized,
        F: Fn(Key) -> bool,
    {
        // Trampoline is the function that knows how to call the Rust closure.
        extern "C" fn trampoline<F: Fn(u64) -> bool>(key: u64, closure_address: usize) -> bool {
            let closure = closure_address as *const F;
            unsafe { (*closure)(key) }
        }

        // Temporarily cast the closure to a raw pointer for passing.
        let trampoline_fn: usize = trampoline::<F> as *const () as usize;
        let closure_address: usize = &filter as *const F as usize;
        index
            .inner
            .filtered_search_f64(query, count, trampoline_fn, closure_address)
    }
    fn change_metric(
        index: &mut Index,
        metric: std::boxed::Box<dyn Fn(*const Self, *const Self) -> Distance + Send + Sync>,
    ) -> Result<(), cxx::Exception> {
        // Store the metric function in the Index.
        type MetricFn = Box<dyn Fn(*const f64, *const f64) -> Distance>;
        index.metric_fn = Some(MetricFunction::F64Metric(Box::into_raw(Box::new(metric))));

        // Trampoline is the function that knows how to call the Rust closure.
        // The `first` is a pointer to the first vector, `second` is a pointer to the second vector,
        // and `index_wrapper` is a pointer to the `index` itself, from which we can infer the metric function
        // and the number of dimensions.
        extern "C" fn trampoline(first: usize, second: usize, closure_address: usize) -> Distance {
            let first_ptr = first as *const f64;
            let second_ptr = second as *const f64;
            let closure: *mut MetricFn = closure_address as *mut MetricFn;
            unsafe { (*closure)(first_ptr, second_ptr) }
        }

        let trampoline_fn: usize = trampoline as *const () as usize;
        let closure_address = match index.metric_fn {
            Some(MetricFunction::F64Metric(metric)) => metric as *mut () as usize,
            _ => panic!("Expected F64Metric"),
        };
        index.inner.change_metric(trampoline_fn, closure_address);

        Ok(())
    }
}

impl VectorType for f16 {
    fn search(index: &Index, query: &[Self], count: usize) -> Result<ffi::Matches, cxx::Exception> {
        index.inner.search_f16(f16::to_i16s(query), count)
    }

    fn exact_search(
        index: &Index,
        query: &[Self],
        count: usize,
    ) -> Result<ffi::Matches, cxx::Exception> {
        index.inner.exact_search_f16(f16::to_i16s(query), count)
    }

    fn get(index: &Index, key: Key, vector: &mut [Self]) -> Result<usize, cxx::Exception> {
        index.inner.get_f16(key, f16::to_mut_i16s(vector))
    }

    fn add(index: &Index, key: Key, vector: &[Self]) -> Result<(), cxx::Exception> {
        index.inner.add_f16(key, f16::to_i16s(vector))
    }

    fn filtered_search<F>(
        index: &Index,
        query: &[Self],
        count: usize,
        filter: F,
    ) -> Result<ffi::Matches, cxx::Exception>
    where
        Self: Sized,
        F: Fn(Key) -> bool,
    {
        // Trampoline is the function that knows how to call the Rust closure.
        extern "C" fn trampoline<F: Fn(u64) -> bool>(key: u64, closure_address: usize) -> bool {
            let closure = closure_address as *const F;
            unsafe { (*closure)(key) }
        }

        // Temporarily cast the closure to a raw pointer for passing.
        let trampoline_fn: usize = trampoline::<F> as *const () as usize;
        let closure_address: usize = &filter as *const F as usize;
        index.inner.filtered_search_f16(
            f16::to_i16s(query),
            count,
            trampoline_fn,
            closure_address,
        )
    }

    fn change_metric(
        index: &mut Index,
        metric: std::boxed::Box<dyn Fn(*const Self, *const Self) -> Distance + Send + Sync>,
    ) -> Result<(), cxx::Exception> {
        // Store the metric function in the Index.
        type MetricFn = Box<dyn Fn(*const f16, *const f16) -> Distance>;
        index.metric_fn = Some(MetricFunction::F16Metric(Box::into_raw(Box::new(metric))));

        // Trampoline is the function that knows how to call the Rust closure.
        // The `first` is a pointer to the first vector, `second` is a pointer to the second vector,
        // and `index_wrapper` is a pointer to the `index` itself, from which we can infer the metric function
        // and the number of dimensions.
        extern "C" fn trampoline(first: usize, second: usize, closure_address: usize) -> Distance {
            let first_ptr = first as *const f16;
            let second_ptr = second as *const f16;
            let closure: *mut MetricFn = closure_address as *mut MetricFn;
            unsafe { (*closure)(first_ptr, second_ptr) }
        }

        let trampoline_fn: usize = trampoline as *const () as usize;
        let closure_address = match index.metric_fn {
            Some(MetricFunction::F16Metric(metric)) => metric as *mut () as usize,
            _ => panic!("Expected F16Metric"),
        };
        index.inner.change_metric(trampoline_fn, closure_address);

        Ok(())
    }
}

impl VectorType for b1x8 {
    fn search(index: &Index, query: &[Self], count: usize) -> Result<ffi::Matches, cxx::Exception> {
        index.inner.search_b1x8(b1x8::to_u8s(query), count)
    }

    fn exact_search(
        index: &Index,
        query: &[Self],
        count: usize,
    ) -> Result<ffi::Matches, cxx::Exception> {
        index.inner.exact_search_b1x8(b1x8::to_u8s(query), count)
    }

    fn get(index: &Index, key: Key, vector: &mut [Self]) -> Result<usize, cxx::Exception> {
        index.inner.get_b1x8(key, b1x8::to_mut_u8s(vector))
    }

    fn add(index: &Index, key: Key, vector: &[Self]) -> Result<(), cxx::Exception> {
        index.inner.add_b1x8(key, b1x8::to_u8s(vector))
    }

    fn filtered_search<F>(
        index: &Index,
        query: &[Self],
        count: usize,
        filter: F,
    ) -> Result<ffi::Matches, cxx::Exception>
    where
        Self: Sized,
        F: Fn(Key) -> bool,
    {
        // Trampoline is the function that knows how to call the Rust closure.
        extern "C" fn trampoline<F: Fn(u64) -> bool>(key: u64, closure_address: usize) -> bool {
            let closure = closure_address as *const F;
            unsafe { (*closure)(key) }
        }

        // Temporarily cast the closure to a raw pointer for passing.
        let trampoline_fn: usize = trampoline::<F> as *const () as usize;
        let closure_address: usize = &filter as *const F as usize;
        index.inner.filtered_search_b1x8(
            b1x8::to_u8s(query),
            count,
            trampoline_fn,
            closure_address,
        )
    }

    fn change_metric(
        index: &mut Index,
        metric: std::boxed::Box<dyn Fn(*const Self, *const Self) -> Distance + Send + Sync>,
    ) -> Result<(), cxx::Exception> {
        // Store the metric function in the Index.
        type MetricFn = Box<dyn Fn(*const b1x8, *const b1x8) -> Distance>;
        index.metric_fn = Some(MetricFunction::B1X8Metric(Box::into_raw(Box::new(metric))));

        // Trampoline is the function that knows how to call the Rust closure.
        // The `first` is a pointer to the first vector, `second` is a pointer to the second vector,
        // and `index_wrapper` is a pointer to the `index` itself, from which we can infer the metric function
        // and the number of dimensions.
        extern "C" fn trampoline(first: usize, second: usize, closure_address: usize) -> Distance {
            let first_ptr = first as *const b1x8;
            let second_ptr = second as *const b1x8;
            let closure: *mut MetricFn = closure_address as *mut MetricFn;
            unsafe { (*closure)(first_ptr, second_ptr) }
        }

        let trampoline_fn: usize = trampoline as *const () as usize;
        let closure_address = match index.metric_fn {
            Some(MetricFunction::B1X8Metric(metric)) => metric as *mut () as usize,
            _ => panic!("Expected F1X8Metric"),
        };
        index.inner.change_metric(trampoline_fn, closure_address);

        Ok(())
    }
}

impl Index {
    pub fn new(options: &ffi::IndexOptions) -> Result<Self, cxx::Exception> {
        match ffi::new_native_index(options) {
            Ok(inner) => Result::Ok(Self {
                inner,
                metric_fn: None,
            }),
            Err(err) => Err(err),
        }
    }

    /// Retrieves the expansion value used during index creation.
    pub fn expansion_add(self: &Index) -> usize {
        self.inner.expansion_add()
    }

    /// Retrieves the expansion value used during search.
    pub fn expansion_search(self: &Index) -> usize {
        self.inner.expansion_search()
    }

    /// Updates the expansion value used during index creation. Rarely used.
    pub fn change_expansion_add(self: &Index, n: usize) {
        self.inner.change_expansion_add(n)
    }

    /// Updates the expansion value used during search operations.
    pub fn change_expansion_search(self: &Index, n: usize) {
        self.inner.change_expansion_search(n)
    }

    /// Changes the metric kind used to calculate the distance between vectors.
    pub fn change_metric_kind(self: &Index, metric: ffi::MetricKind) {
        self.inner.change_metric_kind(metric)
    }

    /// Overrides the metric function used to calculate the distance between vectors.
    pub fn change_metric<T: VectorType>(
        self: &mut Index,
        metric: std::boxed::Box<dyn Fn(*const T, *const T) -> Distance + Send + Sync>,
    ) {
        T::change_metric(self, metric).unwrap();
    }

    /// Retrieves the hardware acceleration information.
    pub fn hardware_acceleration(&self) -> String {
        use core::ffi::CStr;
        unsafe {
            let c_str = CStr::from_ptr(self.inner.hardware_acceleration());
            c_str.to_string_lossy().into_owned()
        }
    }

    /// Performs k-Approximate Nearest Neighbors (kANN) Search for closest vectors to the provided query.
    ///
    /// # Arguments
    ///
    /// * `query` - A slice containing the query vector data.
    /// * `count` - The maximum number of neighbors to search for.
    ///
    /// # Returns
    ///
    /// A `Result` containing the matches found.
    pub fn search<T: VectorType>(
        self: &Index,
        query: &[T],
        count: usize,
    ) -> Result<ffi::Matches, cxx::Exception> {
        T::search(self, query, count)
    }

    /// Performs exact (brute force) Nearest Neighbors Search for closest vectors to the provided query.
    /// This search checks all vectors in the index, guaranteeing to find the true nearest neighbors,
    /// but may be slower for large indices.
    ///
    /// # Arguments
    ///
    /// * `query` - A slice containing the query vector data.
    /// * `count` - The maximum number of neighbors to search for.
    ///
    /// # Returns
    ///
    /// A `Result` containing the matches found.
    pub fn exact_search<T: VectorType>(
        self: &Index,
        query: &[T],
        count: usize,
    ) -> Result<ffi::Matches, cxx::Exception> {
        T::exact_search(self, query, count)
    }

    /// Performs k-Approximate Nearest Neighbors (kANN) Search for closest vectors to the provided query
    /// satisfying a custom filter function.
    ///
    /// # Arguments
    ///
    /// * `query` - A slice containing the query vector data.
    /// * `count` - The maximum number of neighbors to search for.
    /// * `filter` - A closure that takes a `Key` and returns `true` if the corresponding vector should be included in the search results, or `false` otherwise.
    ///
    /// # Returns
    ///
    /// A `Result` containing the matches found.
    pub fn filtered_search<T: VectorType, F>(
        self: &Index,
        query: &[T],
        count: usize,
        filter: F,
    ) -> Result<ffi::Matches, cxx::Exception>
    where
        F: Fn(Key) -> bool,
    {
        T::filtered_search(self, query, count, filter)
    }

    /// Adds a vector with a specified key to the index.
    ///
    /// # Arguments
    ///
    /// * `key` - The key associated with the vector.
    /// * `vector` - A slice containing the vector data.
    pub fn add<T: VectorType>(self: &Index, key: Key, vector: &[T]) -> Result<(), cxx::Exception> {
        T::add(self, key, vector)
    }

    /// Extracts one or more vectors matching the specified key.
    /// The `vector` slice must be a multiple of the number of dimensions in the index.
    /// After the execution, return the number `X` of vectors found.
    /// The vector slice's first `X * dimensions` elements will be filled.
    ///
    /// If you are a novice user, consider `export`.
    ///
    /// # Arguments
    ///
    /// * `key` - The key associated with the vector.
    /// * `vector` - A slice containing the vector data.
    pub fn get<T: VectorType>(
        self: &Index,
        key: Key,
        vector: &mut [T],
    ) -> Result<usize, cxx::Exception> {
        T::get(self, key, vector)
    }

    /// Extracts one or more vectors matching specified key into supplied resizable vector.
    /// The `vector` is resized to a multiple of the number of dimensions in the index.
    ///
    /// # Arguments
    ///
    /// * `key` - The key associated with the vector.
    /// * `vector` - A mutable vector containing the vector data.
    pub fn export<T: VectorType + Default + Clone>(
        self: &Index,
        key: Key,
        vector: &mut Vec<T>,
    ) -> Result<usize, cxx::Exception> {
        let dim = self.dimensions();
        let max_matches = self.count(key);
        vector.resize(dim * max_matches, T::default());
        let matches = T::get(self, key, &mut vector[..])?;
        vector.resize(dim * matches, T::default());
        Ok(matches)
    }

    /// Reserves memory for a specified number of incoming vectors.
    ///
    /// # Arguments
    ///
    /// * `capacity` - The desired total capacity, including the current size.
    pub fn reserve(self: &Index, capacity: usize) -> Result<(), cxx::Exception> {
        self.inner.reserve(capacity)
    }

    /// Reserves memory for a specified number of incoming vectors & active threads.
    ///
    /// # Arguments
    ///
    /// * `capacity` - The desired total capacity, including the current size.
    /// * `threads` - The number of threads to use for the operation.
    pub fn reserve_capacity_and_threads(
        self: &Index,
        capacity: usize,
        threads: usize,
    ) -> Result<(), cxx::Exception> {
        self.inner.reserve_capacity_and_threads(capacity, threads)
    }

    /// Retrieves the number of dimensions in the vectors indexed.
    pub fn dimensions(self: &Index) -> usize {
        self.inner.dimensions()
    }

    /// Retrieves the connectivity parameter that limits connections-per-node in the graph.
    pub fn connectivity(self: &Index) -> usize {
        self.inner.connectivity()
    }

    /// Retrieves the current number of vectors in the index.
    pub fn size(self: &Index) -> usize {
        self.inner.size()
    }

    /// Retrieves the total capacity of the index, including reserved space.
    pub fn capacity(self: &Index) -> usize {
        self.inner.capacity()
    }

    /// Reports expected file size after serialization.
    pub fn serialized_length(self: &Index) -> usize {
        self.inner.serialized_length()
    }

    /// Removes the vector associated with the given key from the index.
    ///
    /// # Arguments
    ///
    /// * `key` - The key of the vector to be removed.
    ///
    /// # Returns
    ///
    /// `true` if the vector is successfully removed, `false` otherwise.
    pub fn remove(self: &Index, key: Key) -> Result<usize, cxx::Exception> {
        self.inner.remove(key)
    }

    /// Renames the vector under a specific key.
    ///
    /// # Arguments
    ///
    /// * `from` - The key of the vector to be renamed.
    /// * `to` - The new name.
    ///
    /// # Returns
    ///
    /// `true` if the vector is renamed, `false` otherwise.
    pub fn rename(self: &Index, from: Key, to: Key) -> Result<usize, cxx::Exception> {
        self.inner.rename(from, to)
    }

    /// Checks if the index contains a vector with a specified key.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to be checked.
    ///
    /// # Returns
    ///
    /// `true` if the index contains the vector with the given key, `false` otherwise.
    pub fn contains(self: &Index, key: Key) -> bool {
        self.inner.contains(key)
    }

    /// Count the count of vectors with the same specified key.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to be checked.
    ///
    /// # Returns
    ///
    /// Number of vectors found.
    pub fn count(self: &Index, key: Key) -> usize {
        self.inner.count(key)
    }

    /// Saves the index to a specified file.
    ///
    /// # Arguments
    ///
    /// * `path` - The file path where the index will be saved.
    pub fn save(self: &Index, path: &str) -> Result<(), cxx::Exception> {
        self.inner.save(path)
    }

    /// Loads the index from a specified file.
    ///
    /// # Arguments
    ///
    /// * `path` - The file path from where the index will be loaded.
    pub fn load(self: &Index, path: &str) -> Result<(), cxx::Exception> {
        self.inner.load(path)
    }

    /// Creates a view of the index from a file without loading it into memory.
    ///
    /// # Arguments
    ///
    /// * `path` - The file path from where the view will be created.
    pub fn view(self: &Index, path: &str) -> Result<(), cxx::Exception> {
        self.inner.view(path)
    }

    /// Erases all members from the index, closes files, and returns RAM to OS.
    pub fn reset(self: &Index) -> Result<(), cxx::Exception> {
        self.inner.reset()
    }

    /// A relatively accurate lower bound on the amount of memory consumed by the system.
    /// In practice, its error will be below 10%.
    pub fn memory_usage(self: &Index) -> usize {
        self.inner.memory_usage()
    }

    /// Saves the index to a specified file.
    ///
    /// # Arguments
    ///
    /// * `buffer` - The buffer where the index will be saved.
    pub fn save_to_buffer(self: &Index, buffer: &mut [u8]) -> Result<(), cxx::Exception> {
        self.inner.save_to_buffer(buffer)
    }

    /// Loads the index from a specified file.
    ///
    /// # Arguments
    ///
    /// * `buffer` - The buffer from where the index will be loaded.
    pub fn load_from_buffer(self: &Index, buffer: &[u8]) -> Result<(), cxx::Exception> {
        self.inner.load_from_buffer(buffer)
    }

    /// Creates a view of the index from a file without loading it into memory.
    ///
    /// # Arguments
    ///
    /// * `buffer` - The buffer from where the view will be created.
    ///
    /// # Safety
    ///
    /// This function is marked as `unsafe` because it stores a pointer to the input buffer.
    /// The caller must ensure that the buffer outlives the index and is not dropped
    /// or modified for the duration of the index's use. Dereferencing a pointer to a
    /// temporary buffer after it has been dropped can lead to undefined behavior,
    /// which violates Rust's memory safety guarantees.
    ///
    /// Example of misuse:
    ///
    /// ```rust,ignore
    /// let index: usearch::Index = usearch::new_index(&usearch::IndexOptions::default()).unwrap();
    ///
    /// let temporary = vec![0u8; 100];
    /// index.view_from_buffer(&temporary);
    /// std::mem::drop(temporary);
    ///
    /// let query = vec![0.0; 256];
    /// let results = index.search(&query, 5).unwrap();
    /// ```
    ///
    /// The above example would result in use-after-free and undefined behavior.
    pub unsafe fn view_from_buffer(self: &Index, buffer: &[u8]) -> Result<(), cxx::Exception> {
        self.inner.view_from_buffer(buffer)
    }
}

pub fn new_index(options: &ffi::IndexOptions) -> Result<Index, cxx::Exception> {
    Index::new(options)
}

#[cfg(test)]
mod tests {
    use crate::ffi::IndexOptions;
    use crate::ffi::MetricKind;
    use crate::ffi::ScalarKind;

    use crate::b1x8;
    use crate::new_index;
    use crate::Index;
    use crate::Key;

    use std::env;

    #[test]
    fn print_specs() {
        println!("--------------------------------------------------");
        println!("OS: {}", env::consts::OS);
        println!(
            "Rust version: {}",
            env::var("RUST_VERSION").unwrap_or_else(|_| "unknown".into())
        );

        // Create indexes with different configurations
        let f64_index = Index::new(&IndexOptions {
            dimensions: 256,
            metric: MetricKind::Cos,
            quantization: ScalarKind::F64,
            ..Default::default()
        })
        .unwrap();

        let f32_index = Index::new(&IndexOptions {
            dimensions: 256,
            metric: MetricKind::Cos,
            quantization: ScalarKind::F32,
            ..Default::default()
        })
        .unwrap();

        let f16_index = Index::new(&IndexOptions {
            dimensions: 256,
            metric: MetricKind::Cos,
            quantization: ScalarKind::F16,
            ..Default::default()
        })
        .unwrap();

        let i8_index = Index::new(&IndexOptions {
            dimensions: 256,
            metric: MetricKind::Cos,
            quantization: ScalarKind::I8,
            ..Default::default()
        })
        .unwrap();

        let b1_index = Index::new(&IndexOptions {
            dimensions: 256,
            metric: MetricKind::Hamming,
            quantization: ScalarKind::B1,
            ..Default::default()
        })
        .unwrap();

        println!(
            "f64 hardware acceleration: {}",
            f64_index.hardware_acceleration()
        );
        println!(
            "f32 hardware acceleration: {}",
            f32_index.hardware_acceleration()
        );
        println!(
            "f16 hardware acceleration: {}",
            f16_index.hardware_acceleration()
        );
        println!(
            "i8 hardware acceleration: {}",
            i8_index.hardware_acceleration()
        );
        println!(
            "b1 hardware acceleration: {}",
            b1_index.hardware_acceleration()
        );
        println!("--------------------------------------------------");
    }

    #[test]
    fn test_add_get_vector() {
        let options = IndexOptions {
            dimensions: 5,
            quantization: ScalarKind::F32,
            ..Default::default()
        };
        let index = Index::new(&options).unwrap();
        assert!(index.reserve(10).is_ok());

        let first: [f32; 5] = [0.2, 0.1, 0.2, 0.1, 0.3];
        let second: [f32; 5] = [0.3, 0.2, 0.4, 0.0, 0.1];
        let too_long: [f32; 6] = [0.3, 0.2, 0.4, 0.0, 0.1, 0.1];
        let too_short: [f32; 4] = [0.3, 0.2, 0.4, 0.0];
        assert!(index.add(1, &first).is_ok());
        assert!(index.add(2, &second).is_ok());
        assert!(index.add(3, &too_long).is_err());
        assert!(index.add(4, &too_short).is_err());
        assert_eq!(index.size(), 2);

        // Test using Vec<T>
        let mut found_vec: Vec<f32> = Vec::new();
        assert_eq!(index.export(1, &mut found_vec).unwrap(), 1);
        assert_eq!(found_vec.len(), 5);
        assert_eq!(found_vec, first.to_vec());

        // Test using slice
        let mut found_slice = [0.0f32; 5];
        assert_eq!(index.get(1, &mut found_slice).unwrap(), 1);
        assert_eq!(found_slice, first);

        // Create a slice with incorrect size
        let mut found = [0.0f32; 6]; // This isn't a multiple of the index's dimensions.
        let result = index.get(1, &mut found);
        assert!(result.is_err());
    }
    #[test]
    fn test_search_vector() {
        let options = IndexOptions {
            dimensions: 5,
            quantization: ScalarKind::F32,
            ..Default::default()
        };
        let index = Index::new(&options).unwrap();
        assert!(index.reserve(10).is_ok());

        let first: [f32; 5] = [0.2, 0.1, 0.2, 0.1, 0.3];
        let second: [f32; 5] = [0.3, 0.2, 0.4, 0.0, 0.1];
        let too_long: [f32; 6] = [0.3, 0.2, 0.4, 0.0, 0.1, 0.1];
        let too_short: [f32; 4] = [0.3, 0.2, 0.4, 0.0];
        assert!(index.add(1, &first).is_ok());
        assert!(index.add(2, &second).is_ok());
        assert_eq!(index.size(), 2);
        //assert!(index.add(3, &too_long).is_err());
        //assert!(index.add(4, &too_short).is_err());

        assert!(index.search(&too_long, 1).is_err());
        assert!(index.search(&too_short, 1).is_err());
    }

    #[test]
    fn test_add_remove_vector() {
        let options = IndexOptions {
            dimensions: 4,
            metric: MetricKind::IP,
            quantization: ScalarKind::F64,
            connectivity: 10,
            expansion_add: 128,
            expansion_search: 3,
            ..Default::default()
        };
        let index = Index::new(&options).unwrap();
        assert!(index.reserve(10).is_ok());
        assert!(index.capacity() >= 10);

        let first: [f32; 4] = [0.2, 0.1, 0.2, 0.1];
        let second: [f32; 4] = [0.3, 0.2, 0.4, 0.0];

        // IDs until 18446744073709551615 should be fine:
        let id1 = 483367403120493160;
        let id2 = 483367403120558696;
        let id3 = 483367403120624232;
        let id4 = 483367403120624233;

        assert!(index.add(id1, &first).is_ok());
        let mut found_slice = [0.0f32; 4];
        assert_eq!(index.get(id1, &mut found_slice).unwrap(), 1);
        assert!(index.remove(id1).is_ok());

        assert!(index.add(id2, &second).is_ok());
        let mut found_slice = [0.0f32; 4];
        assert_eq!(index.get(id2, &mut found_slice).unwrap(), 1);
        assert!(index.remove(id2).is_ok());

        assert!(index.add(id3, &second).is_ok());
        let mut found_slice = [0.0f32; 4];
        assert_eq!(index.get(id3, &mut found_slice).unwrap(), 1);
        assert!(index.remove(id3).is_ok());

        assert!(index.add(id4, &second).is_ok());
        let mut found_slice = [0.0f32; 4];
        assert_eq!(index.get(id4, &mut found_slice).unwrap(), 1);
        assert!(index.remove(id4).is_ok());

        assert_eq!(index.size(), 0);
    }

    #[test]
    fn integration() {
        let mut options = IndexOptions {
            dimensions: 5,
            ..Default::default()
        };

        let index = Index::new(&options).unwrap();

        assert!(index.expansion_add() > 0);
        assert!(index.expansion_search() > 0);

        assert!(index.reserve(10).is_ok());
        assert!(index.capacity() >= 10);
        assert!(index.connectivity() != 0);
        assert_eq!(index.dimensions(), 5);
        assert_eq!(index.size(), 0);

        let first: [f32; 5] = [0.2, 0.1, 0.2, 0.1, 0.3];
        let second: [f32; 5] = [0.3, 0.2, 0.4, 0.0, 0.1];

        println!("--------------------------------------------------");
        println!(
            "before add, memory_usage: {} \
            cap: {} \
            ",
            index.memory_usage(),
            index.capacity(),
        );
        index.change_expansion_add(10);
        assert_eq!(index.expansion_add(), 10);
        assert!(index.add(42, &first).is_ok());
        index.change_expansion_add(12);
        assert_eq!(index.expansion_add(), 12);
        assert!(index.add(43, &second).is_ok());
        assert_eq!(index.size(), 2);
        println!(
            "after add, memory_usage: {} \
            cap: {} \
            ",
            index.memory_usage(),
            index.capacity(),
        );

        index.change_expansion_search(10);
        assert_eq!(index.expansion_search(), 10);
        // Read back the tags
        let results = index.search(&first, 10).unwrap();
        println!("{:?}", results);
        assert_eq!(results.keys.len(), 2);

        index.change_expansion_search(12);
        assert_eq!(index.expansion_search(), 12);
        let results = index.search(&first, 10).unwrap();
        println!("{:?}", results);
        assert_eq!(results.keys.len(), 2);
        println!("--------------------------------------------------");

        // Validate serialization
        assert!(index.save("index.rust.usearch").is_ok());
        assert!(index.load("index.rust.usearch").is_ok());
        assert!(index.view("index.rust.usearch").is_ok());

        // Make sure every function is called at least once
        assert!(new_index(&options).is_ok());
        options.metric = MetricKind::L2sq;
        assert!(new_index(&options).is_ok());
        options.metric = MetricKind::Cos;
        assert!(new_index(&options).is_ok());
        options.metric = MetricKind::Haversine;
        options.quantization = ScalarKind::F32;
        options.dimensions = 2;
        assert!(new_index(&options).is_ok());

        let mut serialization_buffer = vec![0; index.serialized_length()];
        assert!(index.save_to_buffer(&mut serialization_buffer).is_ok());

        let deserialized_index = new_index(&options).unwrap();
        assert!(deserialized_index
            .load_from_buffer(&serialization_buffer)
            .is_ok());
        assert_eq!(index.size(), deserialized_index.size());

        // reset
        assert_ne!(index.memory_usage(), 0);
        assert!(index.reset().is_ok());
        assert_eq!(index.size(), 0);
        assert_eq!(index.memory_usage(), 0);

        // clone
        options.metric = MetricKind::Haversine;
        let mut opts = options.clone();
        assert_eq!(opts.metric, options.metric);
        assert_eq!(opts.quantization, options.quantization);
        assert_eq!(opts, options);
        opts.metric = MetricKind::Cos;
        assert_ne!(opts.metric, options.metric);
        assert!(new_index(&opts).is_ok());
    }

    #[test]
    fn test_search_with_stateless_filter() {
        let options = IndexOptions {
            dimensions: 5,
            ..Default::default()
        };
        let index = Index::new(&options).unwrap();
        index.reserve(10).unwrap();

        // Adding sample vectors to the index
        let first: [f32; 5] = [0.2, 0.1, 0.2, 0.1, 0.3];
        let second: [f32; 5] = [0.3, 0.2, 0.4, 0.0, 0.1];
        index.add(1, &first).unwrap();
        index.add(2, &second).unwrap();

        // Stateless filter: checks if the key is odd
        let is_odd = |key: Key| key % 2 == 1;
        let query = vec![0.2, 0.1, 0.2, 0.1, 0.3]; // Example query vector
        let results = index.filtered_search(&query, 10, is_odd).unwrap();
        assert!(
            results.keys.iter().all(|&key| key % 2 == 1),
            "All keys must be odd"
        );
    }

    #[test]
    fn test_search_with_stateful_filter() {
        use std::collections::HashSet;

        let options = IndexOptions {
            dimensions: 5,
            ..Default::default()
        };
        let index = Index::new(&options).unwrap();
        index.reserve(10).unwrap();

        // Adding sample vectors to the index
        let first: [f32; 5] = [0.2, 0.1, 0.2, 0.1, 0.3];
        index.add(1, &first).unwrap();
        index.add(2, &first).unwrap();

        let allowed_keys = vec![1, 2, 3].into_iter().collect::<HashSet<Key>>();
        // Clone `allowed_keys` for use in the closure
        let filter_keys = allowed_keys.clone();
        let stateful_filter = move |key: Key| filter_keys.contains(&key);

        let query = vec![0.2, 0.1, 0.2, 0.1, 0.3]; // Example query vector
        let results = index.filtered_search(&query, 10, stateful_filter).unwrap();

        // Use the original `allowed_keys` for assertion
        assert!(
            results.keys.iter().all(|&key| allowed_keys.contains(&key)),
            "All keys must be in the allowed set"
        );
    }

    #[test]
    fn test_zero_distances() {
        let options = IndexOptions {
            dimensions: 8,
            metric: MetricKind::L2sq,
            quantization: ScalarKind::F16,
            ..Default::default()
        };

        let index = new_index(&options).unwrap();
        index.reserve(10).unwrap();
        index
            .add(0, &[0.4, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
            .unwrap();
        index
            .add(1, &[0.5, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
            .unwrap();
        index
            .add(2, &[0.6, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
            .unwrap();

        // Make sure non of the distances are zeros
        let matches = index
            .search(&[0.05, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0], 2)
            .unwrap();
        for distance in matches.distances.iter() {
            assert_ne!(*distance, 0.0);
        }
    }

    #[test]
    fn test_exact_search() {
        use std::collections::HashSet;

        // Create an index with many vectors
        let options = IndexOptions {
            dimensions: 4,
            metric: MetricKind::L2sq,
            quantization: ScalarKind::F32,
            ..Default::default()
        };
        let index = new_index(&options).unwrap();
        index.reserve(100).unwrap();
        // Add 100 vectors to the index
        for i in 0..100 {
            let vec = vec![
                i as f32 * 0.1,
                (i as f32 * 0.05).sin(),
                (i as f32 * 0.05).cos(),
                0.0,
            ];
            index.add(i, &vec).unwrap();
        }
        // Query vector
        let query = vec![4.5, 0.0, 1.0, 0.0];
        // Compare approximate and exact search results
        let approx_matches = index.search(&query, 10).unwrap();
        let exact_matches = index.exact_search(&query, 10).unwrap();
        // Collect the keys from both result sets
        let approx_keys: HashSet<Key> = approx_matches.keys.iter().cloned().collect();
        let exact_keys: HashSet<Key> = exact_matches.keys.iter().cloned().collect();
        // Check that both methods return 10 results
        assert_eq!(approx_matches.keys.len(), 10);
        assert_eq!(exact_matches.keys.len(), 10);

        // The exact search should find the true nearest neighbors
        // Verify that the minimum distance in exact results is <= minimum distance in approximate results
        assert!(exact_matches.distances[0] <= approx_matches.distances[0]);
        // The nearest neighbor according to exact search might be different from approximate search
        println!(
            "Approximate search first match: key={}, distance={}",
            approx_matches.keys[0], approx_matches.distances[0]
        );
        println!(
            "Exact search first match: key={}, distance={}",
            exact_matches.keys[0], exact_matches.distances[0]
        );
        // Results from both should be mostly similar, but may differ due to approximation
        let intersection: HashSet<_> = approx_keys.intersection(&exact_keys).collect();
        println!(
            "Number of common results between approximate and exact search: {}",
            intersection.len()
        );
    }

    #[test]
    fn test_change_distance_function() {
        let options = IndexOptions {
            dimensions: 2, // Adjusted for simplicity in creating test vectors
            ..Default::default()
        };
        let mut index = Index::new(&options).unwrap();
        index.reserve(10).unwrap();

        // Adding a simple vector to test the distance function changes
        let vector: [f32; 2] = [1.0, 0.0];
        index.add(1, &vector).unwrap();

        // Stateful distance function with adjustments for pointer to slice conversion
        let first_factor: f32 = 2.0;
        let second_factor: f32 = 0.7;
        let stateful_distance = Box::new(move |a: *const f32, b: *const f32| unsafe {
            let a_slice = std::slice::from_raw_parts(a, 2);
            let b_slice = std::slice::from_raw_parts(b, 2);
            (a_slice[0] - b_slice[0]).abs() * first_factor
                + (a_slice[1] - b_slice[1]).abs() * second_factor
        });
        index.change_metric(stateful_distance);

        let another_vector: [f32; 2] = [0.0, 1.0];
        index.add(2, &another_vector).unwrap();
    }

    #[test]
    fn test_binary_vectors_and_hamming_distance() {
        let index = Index::new(&IndexOptions {
            dimensions: 8,
            metric: MetricKind::Hamming,
            quantization: ScalarKind::B1,
            ..Default::default()
        })
        .unwrap();

        // Binary vectors represented as `b1x8` slices
        let vector42: Vec<b1x8> = vec![b1x8(0b00001111)];
        let vector43: Vec<b1x8> = vec![b1x8(0b11110000)];
        let query: Vec<b1x8> = vec![b1x8(0b01111000)];

        // Adding binary vectors to the index
        index.reserve(10).unwrap();
        index.add(42, &vector42).unwrap();
        index.add(43, &vector43).unwrap();

        let results = index.search(&query, 5).unwrap();

        // Validate the search results based on Hamming distance
        assert_eq!(results.keys.len(), 2);
        assert_eq!(results.keys[0], 43);
        assert_eq!(results.distances[0], 2.0);
        assert_eq!(results.keys[1], 42);
        assert_eq!(results.distances[1], 6.0);
    }

    #[test]
    fn test_concurrency() {
        use fork_union as fu;
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha8Rng;
        use rand_distr::Uniform;
        use std::sync::Arc;

        const DIMENSIONS: usize = 128;
        const VECTOR_COUNT: usize = 1000;
        const THREAD_COUNT: usize = 4;

        let options = IndexOptions {
            dimensions: DIMENSIONS,
            metric: MetricKind::Cos,
            quantization: ScalarKind::F32,
            ..Default::default()
        };

        let index = Arc::new(Index::new(&options).unwrap());
        index
            .reserve_capacity_and_threads(VECTOR_COUNT, THREAD_COUNT)
            .unwrap();

        // Generate deterministic vectors using rand crate for reproducible testing
        let seed = 42; // Fixed seed for reproducibility
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let uniform = Uniform::new(-1.0f32, 1.0f32).unwrap();

        // Store reference vectors for validation
        let mut reference_vectors: Vec<[f32; DIMENSIONS]> = Vec::with_capacity(VECTOR_COUNT);
        for _ in 0..VECTOR_COUNT {
            let mut vector = [0.0f32; DIMENSIONS];
            // Fill with random values in [-1, 1]
            for item in vector.iter_mut().take(DIMENSIONS) {
                *item = rng.sample(uniform);
            }
            reference_vectors.push(vector);
        }

        let mut pool = fu::spawn(THREAD_COUNT);

        // Concurrent indexing
        pool.for_n(VECTOR_COUNT, |prong| {
            let index_clone = Arc::clone(&index);
            let i = prong.task_index;
            let vector = reference_vectors[i];
            index_clone.add(i as u64, &vector).unwrap();
        });

        assert_eq!(index.size(), VECTOR_COUNT);

        // Concurrent retrieval and validation
        let mut pool = fu::spawn(THREAD_COUNT);
        let validation_results = Arc::new(std::sync::Mutex::new(Vec::new()));

        pool.for_n(VECTOR_COUNT, |prong| {
            let index_clone = Arc::clone(&index);
            let results_clone = Arc::clone(&validation_results);
            let i = prong.task_index;
            let expected_vector = &reference_vectors[i];

            let mut retrieved_vector = [0.0f32; DIMENSIONS];
            let count = index_clone.get(i as u64, &mut retrieved_vector).unwrap();
            assert_eq!(count, 1);

            // Validate retrieved vector matches expected
            let matches = retrieved_vector
                .iter()
                .zip(expected_vector.iter())
                .all(|(a, b)| (a - b).abs() < 1e-6);

            let mut results = results_clone.lock().unwrap();
            results.push(matches);
        });

        let validation_results = validation_results.lock().unwrap();
        assert_eq!(validation_results.len(), VECTOR_COUNT);
        assert!(
            validation_results.iter().all(|&x| x),
            "All retrieved vectors should match the original ones"
        );

        // Concurrent search testing
        let mut pool = fu::spawn(THREAD_COUNT);
        let search_results = Arc::new(std::sync::Mutex::new(Vec::new()));

        pool.for_n(100, |prong| {
            // Test 100 searches
            let index_clone = Arc::clone(&index);
            let results_clone = Arc::clone(&search_results);
            let query_idx = prong.task_index % VECTOR_COUNT;
            let query_vector = &reference_vectors[query_idx];

            let matches = index_clone.exact_search(query_vector, 10).unwrap();

            // The first result should be the exact match with distance ~0
            let exact_match_found = !matches.keys.is_empty()
                && matches.keys[0] == query_idx as u64
                && matches.distances[0] < 1e-6;

            let mut results = results_clone.lock().unwrap();
            results.push(exact_match_found);
        });

        let search_results = search_results.lock().unwrap();
        assert_eq!(search_results.len(), 100);
        assert!(
            search_results.iter().all(|&x| x),
            "All searches should find exact matches"
        );
    }
}
