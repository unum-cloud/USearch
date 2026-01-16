// Package usearch provides Go bindings for the USearch library, a high-performance
// approximate nearest neighbor search implementation.
//
// Basic usage:
//
//	conf := usearch.DefaultConfig(128)  // 128-dimensional vectors
//	index, err := usearch.NewIndex(conf)
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer index.Destroy()
//
//	// Add vectors
//	vec := make([]float32, 128)
//	err = index.Add(42, vec)
//
//	// Search
//	keys, distances, err := index.Search(vec, 10)
package usearch

import (
	"errors"
	"fmt"
	"runtime"
	"unsafe"
)

/*
#cgo LDFLAGS: -L. -L/usr/local/lib -lusearch_c
#include "usearch.h"
#include <stdlib.h>
*/
import "C"

// Key represents a unique identifier for vectors in the index.
// Keys must be unique within an index; adding a vector with an existing
// key will update the associated vector.
type Key = uint64

// Metric defines the distance calculation method used for comparing vectors.
// Different metrics are suitable for different use cases:
//   - Cosine: Normalized dot product, ideal for text embeddings
//   - L2sq: Squared Euclidean distance, for spatial data
//   - InnerProduct: Dot product, for recommendation systems
type Metric uint8

// Different metric kinds supported by the USearch library.
const (
	// InnerProduct computes the dot product between vectors
	InnerProduct Metric = iota
	// Cosine computes cosine similarity (normalized dot product)
	Cosine
	// L2sq computes squared Euclidean distance
	L2sq
	// Haversine computes great-circle distance for geographic coordinates
	Haversine
	// Divergence computes Jensen-Shannon divergence
	Divergence
	// Pearson computes Pearson correlation coefficient
	Pearson
	// Hamming computes Hamming distance for binary data
	Hamming
	// Tanimoto computes Tanimoto/Jaccard coefficient
	Tanimoto
	// Sorensen computes Sørensen-Dice coefficient
	Sorensen
)

// String returns the string representation of the Metric.
func (m Metric) String() string {
	switch m {
	case L2sq:
		return "l2sq"
	case InnerProduct:
		return "ip"
	case Cosine:
		return "cos"
	case Haversine:
		return "haversine"
	case Divergence:
		return "divergence"
	case Pearson:
		return "pearson"
	case Hamming:
		return "hamming"
	case Tanimoto:
		return "tanimoto"
	case Sorensen:
		return "sorensen"
	default:
		panic("unknown metric")
	}
}
func (m Metric) CValue() C.usearch_metric_kind_t {
	switch m {
	case L2sq:
		return C.usearch_metric_l2sq_k
	case InnerProduct:
		return C.usearch_metric_ip_k
	case Cosine:
		return C.usearch_metric_cos_k
	case Haversine:
		return C.usearch_metric_haversine_k
	case Divergence:
		return C.usearch_metric_divergence_k
	case Pearson:
		return C.usearch_metric_pearson_k
	case Hamming:
		return C.usearch_metric_hamming_k
	case Tanimoto:
		return C.usearch_metric_tanimoto_k
	case Sorensen:
		return C.usearch_metric_sorensen_k
	}
	return C.usearch_metric_l2sq_k
}

// Quantization represents the scalar type used for storing vectors in the index.
// Different quantization types offer different trade-offs between memory usage and precision.
type Quantization uint8

// Different quantization kinds supported by the USearch library.
const (
	// F32 uses 32-bit floating point (standard precision)
	F32 Quantization = iota
	// BF16 uses brain floating-point format (16-bit)
	BF16
	// F16 uses half-precision floating point (16-bit)
	F16
	// F64 uses 64-bit double precision floating point
	F64
	// I8 uses 8-bit signed integers (quantized)
	I8
	// B1 uses binary representation (1-bit per dimension)
	B1
)

// String returns the string representation of the Quantization.
func (q Quantization) String() string {
	switch q {
	case BF16:
		return "BF16"
	case F16:
		return "F16"
	case F32:
		return "F32"
	case F64:
		return "F64"
	case I8:
		return "I8"
	case B1:
		return "B1"
	default:
		panic("unknown quantization")
	}
}

func (q Quantization) CValue() C.usearch_scalar_kind_t {
	switch q {
	case F16:
		return C.usearch_scalar_f16_k
	case F32:
		return C.usearch_scalar_f32_k
	case F64:
		return C.usearch_scalar_f64_k
	case I8:
		return C.usearch_scalar_i8_k
	case B1:
		return C.usearch_scalar_b1_k
	case BF16:
		return C.usearch_scalar_bf16_k
	default:
		return C.usearch_scalar_unknown_k
	}
}

// IndexConfig represents the configuration options for initializing a USearch index.
//
// Zero values for optional parameters (Connectivity, ExpansionAdd, ExpansionSearch)
// will be replaced with optimal defaults by the C library.
type IndexConfig struct {
	Quantization    Quantization // The scalar kind used for quantization of vector data during indexing.
	Metric          Metric       // The metric kind used for distance calculation between vectors.
	Dimensions      uint         // The number of dimensions in the vectors to be indexed.
	Connectivity    uint         // The optional connectivity parameter that limits connections-per-node in the graph (0 for default).
	ExpansionAdd    uint         // The optional expansion factor used for index construction when adding vectors (0 for default).
	ExpansionSearch uint         // The optional expansion factor used for index construction during search operations (0 for default).
	Multi           bool         // Indicates whether multiple vectors can map to the same key.
}

// DefaultConfig returns an IndexConfig with default values for the specified number of dimensions.
// Uses Cosine metric and F32 quantization by default.
//
// Example:
//
//	config := usearch.DefaultConfig(128)  // 128-dimensional vectors
//	index, err := usearch.NewIndex(config)
func DefaultConfig(dimensions uint) IndexConfig {
	c := IndexConfig{}
	c.Dimensions = dimensions
	c.Metric = Cosine
	c.Quantization = F32
	// Zeros will be replaced by the underlying C implementation
	c.Connectivity = 0
	c.ExpansionAdd = 0
	c.ExpansionSearch = 0
	c.Multi = false
	return c
}

// FilteredSearchHandler includes the callback function and user data.
type FilteredSearchHandler struct {
	Callback func(key Key, handler *FilteredSearchHandler) int
	Data     any
}

// Index represents a USearch approximate nearest neighbor index.
// It implements io.Closer for idiomatic resource cleanup.
//
// The index must be properly initialized with NewIndex() and destroyed
// with Destroy() or Close() when no longer needed to free resources.
type Index struct {
	handle C.usearch_index_t
	config IndexConfig
}

// NewIndex creates a new approximate nearest neighbor index with the specified configuration.
//
// The index must be destroyed with Destroy() when no longer needed.
//
// Example:
//
//	config := usearch.DefaultConfig(128)
//	config.Metric = usearch.L2sq
//	index, err := usearch.NewIndex(config)
//	if err != nil {
//	    log.Fatal(err)
//	}
//	defer index.Destroy()
func NewIndex(conf IndexConfig) (index *Index, err error) {
	if conf.Dimensions == 0 {
		return nil, errors.New("dimensions must be greater than 0")
	}
	index = &Index{config: conf}

	conf = index.config
	dimensions := C.size_t(conf.Dimensions)
	connectivity := C.size_t(conf.Connectivity)
	expansionAdd := C.size_t(conf.ExpansionAdd)
	expansionSearch := C.size_t(conf.ExpansionSearch)
	multi := C.bool(conf.Multi)

	options := C.struct_usearch_init_options_t{}
	options.dimensions = dimensions
	options.connectivity = connectivity
	options.expansion_add = expansionAdd
	options.expansion_search = expansionSearch
	options.multi = multi
	options.metric_kind = conf.Metric.CValue()

	// Map the quantization method
	options.quantization = conf.Quantization.CValue()

	var errorMessage *C.char
	ptr := C.usearch_init(&options, (*C.usearch_error_t)(&errorMessage))
	if errorMessage != nil {
		return nil, errors.New(C.GoString(errorMessage))
	}

	index.handle = ptr
	return index, nil
}

// Version returns the USearch library version string.
func Version() string {
	return C.GoString(C.usearch_version())
}

// GetHandle returns the C index handel.
func (index *Index) GetHandle() C.usearch_index_t {
	return index.handle
}

// GetConfig returns the index config.
func (index *Index) GetConfig() IndexConfig {
	return index.config
}

// Len returns the number of vectors in the index.
func (index *Index) Len() (len uint, err error) {
	var errorMessage *C.char
	len = uint(C.usearch_size(index.handle, (*C.usearch_error_t)(&errorMessage)))
	if errorMessage != nil {
		err = errors.New(C.GoString(errorMessage))
	}
	return len, err
}

// SerializedLength reports the expected file size after serialization.
func (index *Index) SerializedLength() (len uint, err error) {
	var errorMessage *C.char
	len = uint(C.usearch_serialized_length(index.handle, (*C.usearch_error_t)(&errorMessage)))
	if errorMessage != nil {
		err = errors.New(C.GoString(errorMessage))
	}
	return len, err
}

// MemoryUsage reports the memory usage of the index
func (index *Index) MemoryUsage() (len uint, err error) {
	var errorMessage *C.char
	len = uint(C.usearch_memory_usage(index.handle, (*C.usearch_error_t)(&errorMessage)))
	if errorMessage != nil {
		err = errors.New(C.GoString(errorMessage))
	}
	return len, err
}

// ExpansionAdd returns the expansion value used during index creation
func (index *Index) ExpansionAdd() (val uint, err error) {
	var errorMessage *C.char
	val = uint(C.usearch_expansion_add(index.handle, (*C.usearch_error_t)(&errorMessage)))
	if errorMessage != nil {
		err = errors.New(C.GoString(errorMessage))
	}
	return val, err
}

// ExpansionSearch returns the expansion value used during search
func (index *Index) ExpansionSearch() (val uint, err error) {
	var errorMessage *C.char
	val = uint(C.usearch_expansion_search(index.handle, (*C.usearch_error_t)(&errorMessage)))
	if errorMessage != nil {
		err = errors.New(C.GoString(errorMessage))
	}
	return val, err
}

// ChangeExpansionAdd sets the expansion value used during index creation
func (index *Index) ChangeExpansionAdd(val uint) error {
	var errorMessage *C.char
	C.usearch_change_expansion_add(index.handle, C.size_t(val), (*C.usearch_error_t)(&errorMessage))
	if errorMessage != nil {
		return errors.New(C.GoString(errorMessage))
	}
	return nil
}

// ChangeExpansionSearch sets the expansion value used during search
func (index *Index) ChangeExpansionSearch(val uint) error {
	var errorMessage *C.char
	C.usearch_change_expansion_search(index.handle, C.size_t(val), (*C.usearch_error_t)(&errorMessage))
	if errorMessage != nil {
		return errors.New(C.GoString(errorMessage))
	}
	return nil
}

// ChangeThreadsAdd sets the maximum number of CPU threads used by the index
// during add/build operations. This controls internal parallelism for indexing.
func (index *Index) ChangeThreadsAdd(val uint) error {
	var errorMessage *C.char
	C.usearch_change_threads_add(index.handle, C.size_t(val), (*C.usearch_error_t)(&errorMessage))
	if errorMessage != nil {
		return errors.New(C.GoString(errorMessage))
	}
	return nil
}

// ChangeThreadsSearch sets the maximum number of CPU threads used by the index
// during search operations. This controls internal parallelism for queries.
func (index *Index) ChangeThreadsSearch(val uint) error {
	var errorMessage *C.char
	C.usearch_change_threads_search(index.handle, C.size_t(val), (*C.usearch_error_t)(&errorMessage))
	if errorMessage != nil {
		return errors.New(C.GoString(errorMessage))
	}
	return nil
}

// Connectivity returns the connectivity parameter of the index.
func (index *Index) Connectivity() (con uint, err error) {
	var errorMessage *C.char
	con = uint(C.usearch_connectivity(index.handle, (*C.usearch_error_t)(&errorMessage)))
	if errorMessage != nil {
		err = errors.New(C.GoString(errorMessage))
	}
	return con, err
}

// Dimensions returns the number of dimensions of the vectors in the index.
func (index *Index) Dimensions() (dim uint, err error) {
	var errorMessage *C.char
	dim = uint(C.usearch_dimensions(index.handle, (*C.usearch_error_t)(&errorMessage)))
	if errorMessage != nil {
		err = errors.New(C.GoString(errorMessage))
	}
	return dim, err
}

// Capacity returns the capacity (maximum number of vectors) of the index.
func (index *Index) Capacity() (cap uint, err error) {
	var errorMessage *C.char
	cap = uint(C.usearch_capacity(index.handle, (*C.usearch_error_t)(&errorMessage)))
	if errorMessage != nil {
		err = errors.New(C.GoString(errorMessage))
	}
	return cap, err
}

// HardwareAcceleration returns a string showing the SIMD capability for the index.
func (index *Index) HardwareAcceleration() (string, error) {
	var str *C.char
	var errorMessage *C.char
	str = C.usearch_hardware_acceleration(index.handle, (*C.usearch_error_t)(&errorMessage))
	if errorMessage != nil {
		return "", errors.New(C.GoString(errorMessage))
	}
	return C.GoString(str), nil
}

// Destroy frees the resources associated with the index.
func (index *Index) Destroy() error {
	if index.handle == nil {
		panic("index is uninitialized")
	}

	var errorMessage *C.char
	C.usearch_free(index.handle, (*C.usearch_error_t)(&errorMessage))
	if errorMessage != nil {
		return errors.New(C.GoString(errorMessage))
	}
	index.handle = nil
	index.config = IndexConfig{}
	return nil
}

// Close implements io.Closer interface and calls Destroy() to free resources.
// This provides idiomatic Go resource cleanup that can be used with defer statements.
func (index *Index) Close() error {
	return index.Destroy()
}

// Reserve reserves memory for a specified number of incoming vectors.
func (index *Index) Reserve(capacity uint) error {
	if index.handle == nil {
		panic("index is uninitialized")
	}

	var errorMessage *C.char
	C.usearch_reserve(index.handle, (C.size_t)(capacity), (*C.usearch_error_t)(&errorMessage))
	if errorMessage != nil {
		return errors.New(C.GoString(errorMessage))
	}
	return nil
}

// Add inserts or updates a vector in the index with the specified key.
// The vector must have exactly Dimensions() elements.
// If a vector with this key already exists, it will be replaced.
//
// Returns an error if:
//   - The index is not initialized
//   - The vector is empty or has wrong dimensions
//   - The underlying C library reports an error
func (index *Index) Add(key Key, vec []float32) error {
	if index.handle == nil {
		panic("index is uninitialized")
	}

	if len(vec) == 0 {
		return errors.New("vector cannot be empty")
	}
	if uint(len(vec)) != index.config.Dimensions {
		return fmt.Errorf("vector dimension mismatch: got %d, expected %d", len(vec), index.config.Dimensions)
	}

	var errorMessage *C.char
	C.usearch_add(index.handle, (C.usearch_key_t)(key), unsafe.Pointer(&vec[0]), C.usearch_scalar_f32_k, (*C.usearch_error_t)(&errorMessage))
	runtime.KeepAlive(vec)
	if errorMessage != nil {
		return errors.New(C.GoString(errorMessage))
	}
	return nil
}

// AddUnsafe adds a vector using a raw pointer, bypassing Go's type safety.
//
// SAFETY REQUIREMENTS:
//   - vec must not be nil
//   - Memory at vec must contain exactly Dimensions() scalars
//   - Scalar type must match index.config.Quantization
//   - Memory must remain valid for the duration of the call
//   - Caller is responsible for ensuring correct data layout
//
// Use Add() or AddI8() instead unless you need maximum performance
// and understand the safety implications.
func (index *Index) AddUnsafe(key Key, vec unsafe.Pointer) error {
	if index.handle == nil {
		panic("index is uninitialized")
	}

	if vec == nil {
		return errors.New("vector pointer cannot be nil")
	}

	var errorMessage *C.char
	C.usearch_add(index.handle, (C.usearch_key_t)(key), vec, index.config.Quantization.CValue(), (*C.usearch_error_t)(&errorMessage))
	runtime.KeepAlive(vec)
	if errorMessage != nil {
		return errors.New(C.GoString(errorMessage))
	}
	return nil
}

// Remove removes the vector associated with the given key from the index.
func (index *Index) Remove(key Key) error {
	if index.handle == nil {
		panic("index is uninitialized")
	}

	var errorMessage *C.char
	C.usearch_remove(index.handle, (C.usearch_key_t)(key), (*C.usearch_error_t)(&errorMessage))
	if errorMessage != nil {
		return errors.New(C.GoString(errorMessage))
	}
	return nil
}

// Clear removes all vectors from the index while preserving its structure.
func (index *Index) Clear() error {
	if index.handle == nil {
		panic("index is uninitialized")
	}

	var errorMessage *C.char
	C.usearch_clear(index.handle, (*C.usearch_error_t)(&errorMessage))
	if errorMessage != nil {
		return errors.New(C.GoString(errorMessage))
	}
	return nil
}

// Contains checks if the index contains a vector with a specific key.
func (index *Index) Contains(key Key) (found bool, err error) {
	if index.handle == nil {
		panic("index is uninitialized")
	}

	var errorMessage *C.char
	found = bool(C.usearch_contains(index.handle, (C.usearch_key_t)(key), (*C.usearch_error_t)(&errorMessage)))
	if errorMessage != nil {
		return found, errors.New(C.GoString(errorMessage))
	}
	return found, nil
}

// Count returns the number of vectors stored under the given key.
// Useful for multi-vector indexes where multiple vectors can share a key.
func (index *Index) Count(key Key) (count uint, err error) {
	if index.handle == nil {
		panic("index is uninitialized")
	}

	var errorMessage *C.char
	count = uint(C.usearch_count(index.handle, (C.usearch_key_t)(key), (*C.usearch_error_t)(&errorMessage)))
	if errorMessage != nil {
		return 0, errors.New(C.GoString(errorMessage))
	}
	return count, nil
}

// Get retrieves the vectors associated with the given key from the index.
// Returns nil if the key is not found.
func (index *Index) Get(key Key, maxCount uint) (vectors []float32, err error) {
	if index.handle == nil {
		panic("index is uninitialized")
	}

	if maxCount == 0 {
		return nil, nil
	}

	vectors = make([]float32, index.config.Dimensions*maxCount)
	var errorMessage *C.char
	found := uint(C.usearch_get(index.handle, (C.usearch_key_t)(key), (C.size_t)(maxCount), unsafe.Pointer(&vectors[0]), C.usearch_scalar_f32_k, (*C.usearch_error_t)(&errorMessage)))
	runtime.KeepAlive(vectors)
	if errorMessage != nil {
		return nil, errors.New(C.GoString(errorMessage))
	}
	if found == 0 {
		return nil, nil
	}
	return vectors, nil
}

// Rename changes the key of a vector from one value to another.
func (index *Index) Rename(from Key, to Key) error {
	var errorMessage *C.char
	C.usearch_rename(index.handle, C.usearch_key_t(from), C.usearch_key_t(to), (*C.usearch_error_t)(&errorMessage))
	if errorMessage != nil {
		return errors.New(C.GoString(errorMessage))
	}
	return nil
}

// Distance computes the distance between two float32 vectors using the specified metric.
// Both vectors must have exactly 'dims' elements.
func Distance(vec1 []float32, vec2 []float32, vectorDimensions uint, metric Metric) (float32, error) {
	if len(vec1) == 0 || len(vec2) == 0 {
		return 0, errors.New("vectors cannot be empty")
	}
	if uint(len(vec1)) < vectorDimensions || uint(len(vec2)) < vectorDimensions {
		return 0, fmt.Errorf("vectors too short for specified dimensions: need %d elements", vectorDimensions)
	}

	var errorMessage *C.char
	dist := C.usearch_distance(unsafe.Pointer(&vec1[0]), unsafe.Pointer(&vec2[0]), C.usearch_scalar_f32_k, C.size_t(vectorDimensions), metric.CValue(), (*C.usearch_error_t)(&errorMessage))
	runtime.KeepAlive(vec1)
	runtime.KeepAlive(vec2)
	if errorMessage != nil {
		return 0, errors.New(C.GoString(errorMessage))
	}
	return float32(dist), nil
}

// DistanceUnsafe computes the distance between two vectors using unsafe pointers.
//
// SAFETY REQUIREMENTS:
//   - vec1 and vec2 must not be nil
//   - Memory at both pointers must contain exactly 'dims' scalars
//   - Scalar type must match the specified quantization
//   - Memory must remain valid for the duration of the call
func DistanceUnsafe(vec1 unsafe.Pointer, vec2 unsafe.Pointer, vectorDimensions uint, metric Metric, quantization Quantization) (float32, error) {
	if vec1 == nil || vec2 == nil {
		return 0, errors.New("vector pointers cannot be nil")
	}
	if vectorDimensions == 0 {
		return 0, errors.New("dimensions must be greater than zero")
	}

	var errorMessage *C.char
	dist := C.usearch_distance(vec1, vec2, quantization.CValue(), C.size_t(vectorDimensions), metric.CValue(), (*C.usearch_error_t)(&errorMessage))
	runtime.KeepAlive(vec1)
	runtime.KeepAlive(vec2)
	if errorMessage != nil {
		return 0, errors.New(C.GoString(errorMessage))
	}
	return float32(dist), nil
}

// Search finds the k nearest neighbors to the query vector.
//
// Parameters:
//   - query: Must have exactly Dimensions() elements
//   - limit: Maximum number of results to return
//
// Returns:
//   - keys: IDs of the nearest vectors (up to limit)
//   - distances: Distance to each result (same length as keys)
//   - err: Error if query is invalid or search fails
//
// The actual number of results may be less than limit if the index
// contains fewer vectors.
func (index *Index) Search(query []float32, limit uint) (keys []Key, distances []float32, err error) {
	if index.handle == nil {
		panic("index is uninitialized")
	}

	if len(query) == 0 {
		return nil, nil, errors.New("query vector cannot be empty")
	}
	if uint(len(query)) != index.config.Dimensions {
		return nil, nil, fmt.Errorf("query dimension mismatch: got %d, expected %d", len(query), index.config.Dimensions)
	}
	if limit == 0 {
		return []Key{}, []float32{}, nil
	}

	keys = make([]Key, limit)
	distances = make([]float32, limit)
	var errorMessage *C.char
	resultCount := uint(C.usearch_search(index.handle, unsafe.Pointer(&query[0]), C.usearch_scalar_f32_k, (C.size_t)(limit), (*C.usearch_key_t)(&keys[0]), (*C.usearch_distance_t)(&distances[0]), (*C.usearch_error_t)(&errorMessage)))
	runtime.KeepAlive(query)
	runtime.KeepAlive(keys)
	runtime.KeepAlive(distances)
	if errorMessage != nil {
		return nil, nil, errors.New(C.GoString(errorMessage))
	}

	keys = keys[:resultCount]
	distances = distances[:resultCount]
	return keys, distances, nil
}

// FilteredSearch finds the k nearest neighbors to the query vector, applying a filter.
//
// Parameters:
//   - query: Must have exactly Dimensions() elements
//   - limit: Maximum number of results to return
//   - handler: Filter callback that returns non-zero to accept a candidate, zero to reject
//
// Returns:
//   - keys: IDs of the nearest vectors that passed the filter (up to limit)
//   - distances: Distance to each result (same length as keys)
//   - err: Error if query is invalid, handler is nil, or search fails
//
// The actual number of results may be less than limit if fewer vectors pass the filter.
func (index *Index) FilteredSearch(query []float32, limit uint, handler *FilteredSearchHandler) (keys []Key, distances []float32, err error) {
	if index.handle == nil {
		panic("index is uninitialized")
	}

	if len(query) == 0 {
		return nil, nil, errors.New("query vector cannot be empty")
	}
	if uint(len(query)) != index.config.Dimensions {
		return nil, nil, fmt.Errorf("query dimension mismatch: got %d, expected %d", len(query), index.config.Dimensions)
	}
	if handler == nil {
		return nil, nil, errors.New("filtered search handler cannot be nil")
	}
	if limit == 0 {
		return []Key{}, []float32{}, nil
	}

	keys = make([]Key, limit)
	distances = make([]float32, limit)
	var errorMessage *C.char
	resultCount := uint(C.usearch_filtered_search(index.handle, unsafe.Pointer(&query[0]), C.usearch_scalar_f32_k, (C.size_t)(limit),
		(C.usearch_filtered_search_callback_t)(C.goFilteredSearchCallback), unsafe.Pointer(handler),
		(*C.usearch_key_t)(&keys[0]), (*C.usearch_distance_t)(&distances[0]), (*C.usearch_error_t)(&errorMessage)))
	runtime.KeepAlive(query)
	runtime.KeepAlive(keys)
	runtime.KeepAlive(distances)
	runtime.KeepAlive(handler)
	if errorMessage != nil {
		return nil, nil, errors.New(C.GoString(errorMessage))
	}

	keys = keys[:resultCount]
	distances = distances[:resultCount]
	return keys, distances, nil
}

// SearchUnsafe performs k-Approximate Nearest Neighbors Search using an unsafe pointer.
//
// SAFETY REQUIREMENTS:
//   - query must not be nil
//   - Memory at query must contain exactly Dimensions() scalars
//   - Scalar type must match index.config.Quantization
//   - Memory must remain valid for the duration of the call
//
// Use Search() or SearchI8() instead unless you need maximum performance
// and understand the safety implications.
func (index *Index) SearchUnsafe(query unsafe.Pointer, limit uint) (keys []Key, distances []float32, err error) {
	if index.handle == nil {
		panic("index is uninitialized")
	}

	if query == nil {
		return nil, nil, errors.New("query pointer cannot be nil")
	}
	if limit == 0 {
		return []Key{}, []float32{}, nil
	}

	keys = make([]Key, limit)
	distances = make([]float32, limit)
	var errorMessage *C.char
	resultCount := uint(C.usearch_search(index.handle, query, index.config.Quantization.CValue(), (C.size_t)(limit), (*C.usearch_key_t)(&keys[0]), (*C.usearch_distance_t)(&distances[0]), (*C.usearch_error_t)(&errorMessage)))
	runtime.KeepAlive(query)
	runtime.KeepAlive(keys)
	runtime.KeepAlive(distances)
	if errorMessage != nil {
		return nil, nil, errors.New(C.GoString(errorMessage))
	}

	keys = keys[:resultCount]
	distances = distances[:resultCount]
	return keys, distances, nil
}

//export goFilteredSearchCallback
func goFilteredSearchCallback(key C.usearch_key_t, ptr unsafe.Pointer) C.int {
	handler := (*FilteredSearchHandler)(ptr)
	return C.int(handler.Callback(Key(key), handler))
}

// FilteredSearchUnsafe performs filtered k-ANN search using an unsafe pointer.
//
// SAFETY REQUIREMENTS:
//   - query must not be nil
//   - Memory at query must contain exactly Dimensions() scalars
//   - Scalar type must match index.config.Quantization
//   - Memory must remain valid for the duration of the call
func (index *Index) FilteredSearchUnsafe(query unsafe.Pointer, limit uint, handler *FilteredSearchHandler) (keys []Key, distances []float32, err error) {
	if index.handle == nil {
		panic("index is uninitialized")
	}

	if query == nil {
		return nil, nil, errors.New("query pointer cannot be nil")
	}

	if handler == nil {
		return nil, nil, errors.New("filtered search handler cannot be nil")
	}

	if limit == 0 {
		return []Key{}, []float32{}, nil
	}

	keys = make([]Key, limit)
	distances = make([]float32, limit)
	var errorMessage *C.char
	resultCount := uint(C.usearch_filtered_search(index.handle, query, index.config.Quantization.CValue(), (C.size_t)(limit),
		(C.usearch_filtered_search_callback_t)(C.goFilteredSearchCallback), unsafe.Pointer(handler),
		(*C.usearch_key_t)(&keys[0]), (*C.usearch_distance_t)(&distances[0]), (*C.usearch_error_t)(&errorMessage)))
	runtime.KeepAlive(query)
	runtime.KeepAlive(keys)
	runtime.KeepAlive(distances)
	runtime.KeepAlive(handler)
	if errorMessage != nil {
		return nil, nil, errors.New(C.GoString(errorMessage))
	}

	keys = keys[:resultCount]
	distances = distances[:resultCount]
	return keys, distances, nil
}

// ExactSearch performs multithreaded exact nearest neighbors search.
// Unlike the index-based search, this computes distances to all vectors in the dataset.
//
// Parameters:
//   - dataset: Flattened array of vectors (datasetSize x vectorDimensions)
//   - queries: Flattened array of query vectors (queryCount x vectorDimensions)
//   - datasetSize, queryCount: Number of vectors in dataset and queries
//   - datasetStride, queryStride: Memory stride in bytes between consecutive vectors (use vectorDimensions * sizeof(float32) for contiguous data)
//   - vectorDimensions: Number of dimensions per vector
//   - metric: Distance metric to use
//   - maxResults: Maximum results per query
//   - numThreads: Number of threads to use (0 for auto-detection)
func ExactSearch(dataset []float32, queries []float32, datasetSize uint, queryCount uint,
	datasetStride uint, queryStride uint, vectorDimensions uint, metric Metric,
	maxResults uint, numThreads uint) (keys []Key, distances []float32, err error) {

	if len(dataset) == 0 || len(queries) == 0 {
		return nil, nil, errors.New("dataset and queries cannot be empty")
	}
	if vectorDimensions == 0 {
		return nil, nil, errors.New("dimensions must be greater than zero")
	}
	if (len(dataset) % int(vectorDimensions)) != 0 {
		return nil, nil, errors.New("dataset length must be a multiple of the dimensions")
	}
	if (len(queries) % int(vectorDimensions)) != 0 {
		return nil, nil, errors.New("queries length must be a multiple of the dimensions")
	}
	if maxResults == 0 {
		return nil, nil, errors.New("maxResults must be greater than zero")
	}

	keys = make([]Key, queryCount*maxResults)
	distances = make([]float32, queryCount*maxResults)
	resultKeysStride := uint32(maxResults * 8)      // int64 - 8 bytes
	resultDistancesStride := uint32(maxResults * 4) // float32 - 4 bytes

	var errorMessage *C.char
	C.usearch_exact_search(unsafe.Pointer(&dataset[0]), C.size_t(datasetSize), C.size_t(datasetStride), unsafe.Pointer(&queries[0]), C.size_t(queryCount), C.size_t(queryStride),
		C.usearch_scalar_f32_k, C.size_t(vectorDimensions), metric.CValue(), C.size_t(maxResults), C.size_t(numThreads),
		(*C.usearch_key_t)(&keys[0]), C.size_t(resultKeysStride), (*C.usearch_distance_t)(&distances[0]), C.size_t(resultDistancesStride), (*C.usearch_error_t)(&errorMessage))
	runtime.KeepAlive(dataset)
	runtime.KeepAlive(queries)
	runtime.KeepAlive(keys)
	runtime.KeepAlive(distances)
	if errorMessage != nil {
		return nil, nil, errors.New(C.GoString(errorMessage))
	}

	return keys, distances, nil
}

// ExactSearchUnsafe performs multithreaded exact nearest neighbors search using unsafe pointers.
//
// SAFETY REQUIREMENTS:
//   - dataset and queries must not be nil
//   - Memory must contain contiguous vectors of the specified quantization type
//   - dataset must contain datasetSize vectors of vectorDimensions elements each
//   - queries must contain queryCount vectors of vectorDimensions elements each
//   - Memory must remain valid for the duration of the call
//
// Stride parameters specify memory offset in bytes between consecutive vectors.
// For contiguous data, use vectorDimensions * sizeof(element_type).
func ExactSearchUnsafe(dataset unsafe.Pointer, queries unsafe.Pointer, datasetSize uint, queryCount uint,
	datasetStride uint, queryStride uint, vectorDimensions uint, metric Metric, quantization Quantization,
	maxResults uint, numThreads uint) (keys []Key, distances []float32, err error) {

	if dataset == nil || queries == nil {
		return nil, nil, errors.New("dataset and queries pointers cannot be nil")
	}
	if vectorDimensions == 0 || datasetSize == 0 || queryCount == 0 || maxResults == 0 {
		return nil, nil, errors.New("dimensions, query count, max results and sizes must be greater than zero")
	}

	keys = make([]Key, queryCount*maxResults)
	distances = make([]float32, queryCount*maxResults)
	resultKeysStride := uint32(maxResults * 8)      // int64 - 8 bytes
	resultDistancesStride := uint32(maxResults * 4) // float32 - 4 bytes
	var errorMessage *C.char
	C.usearch_exact_search(dataset, C.size_t(datasetSize), C.size_t(datasetStride), queries, C.size_t(queryCount), C.size_t(queryStride),
		quantization.CValue(), C.size_t(vectorDimensions), metric.CValue(), C.size_t(maxResults), C.size_t(numThreads),
		(*C.usearch_key_t)(&keys[0]), C.size_t(resultKeysStride), (*C.usearch_distance_t)(&distances[0]), C.size_t(resultDistancesStride), (*C.usearch_error_t)(&errorMessage))
	runtime.KeepAlive(keys)
	runtime.KeepAlive(distances)
	if errorMessage != nil {
		return nil, nil, errors.New(C.GoString(errorMessage))
	}

	return keys, distances, nil
}

// Convenience I8 helpers

// AddI8 adds an int8 vector to the index.
// The vector must have exactly Dimensions() elements.
//
// This is a convenience method for indexes using I8 quantization.
func (index *Index) AddI8(key Key, vec []int8) error {
	if index.handle == nil {
		panic("index is uninitialized")
	}
	if len(vec) == 0 {
		return errors.New("vector cannot be empty")
	}
	if uint(len(vec)) != index.config.Dimensions {
		return fmt.Errorf("vector dimension mismatch: got %d, expected %d", len(vec), index.config.Dimensions)
	}
	var errorMessage *C.char
	C.usearch_add(index.handle, (C.usearch_key_t)(key), unsafe.Pointer(&vec[0]), C.usearch_scalar_i8_k, (*C.usearch_error_t)(&errorMessage))
	runtime.KeepAlive(vec)
	if errorMessage != nil {
		return errors.New(C.GoString(errorMessage))
	}
	return nil
}

// SearchI8 searches for nearest neighbors using an int8 query vector.
// The query must have exactly Dimensions() elements.
//
// This is a convenience method for indexes using I8 quantization.
func (index *Index) SearchI8(query []int8, limit uint) (keys []Key, distances []float32, err error) {
	if index.handle == nil {
		panic("index is uninitialized")
	}
	if len(query) == 0 {
		return nil, nil, errors.New("query vector cannot be empty")
	}
	if uint(len(query)) != index.config.Dimensions {
		return nil, nil, fmt.Errorf("query dimension mismatch: got %d, expected %d", len(query), index.config.Dimensions)
	}
	if limit == 0 {
		return []Key{}, []float32{}, nil
	}
	keys = make([]Key, limit)
	distances = make([]float32, limit)
	var errorMessage *C.char
	resultCount := uint(C.usearch_search(index.handle, unsafe.Pointer(&query[0]), C.usearch_scalar_i8_k, (C.size_t)(limit), (*C.usearch_key_t)(&keys[0]), (*C.usearch_distance_t)(&distances[0]), (*C.usearch_error_t)(&errorMessage)))
	runtime.KeepAlive(query)
	runtime.KeepAlive(keys)
	runtime.KeepAlive(distances)
	if errorMessage != nil {
		return nil, nil, errors.New(C.GoString(errorMessage))
	}
	keys = keys[:resultCount]
	distances = distances[:resultCount]
	return keys, distances, nil
}

func (index *Index) FilteredSearchI8(query []int8, limit uint, handler *FilteredSearchHandler) (keys []Key, distances []float32, err error) {
	if index.handle == nil {
		panic("index is uninitialized")
	}

	if len(query) == 0 {
		return nil, nil, errors.New("query vector cannot be empty")
	}
	if uint(len(query)) != index.config.Dimensions {
		return nil, nil, fmt.Errorf("query dimension mismatch: got %d, expected %d", len(query), index.config.Dimensions)
	}
	if handler == nil {
		return nil, nil, errors.New("filtered search handler cannot be nil")
	}
	if limit == 0 {
		return []Key{}, []float32{}, nil
	}

	keys = make([]Key, limit)
	distances = make([]float32, limit)
	var errorMessage *C.char
	resultCount := uint(C.usearch_filtered_search(index.handle, unsafe.Pointer(&query[0]), C.usearch_scalar_i8_k, (C.size_t)(limit),
		(C.usearch_filtered_search_callback_t)(C.goFilteredSearchCallback), unsafe.Pointer(handler),
		(*C.usearch_key_t)(&keys[0]), (*C.usearch_distance_t)(&distances[0]), (*C.usearch_error_t)(&errorMessage)))
	runtime.KeepAlive(query)
	runtime.KeepAlive(keys)
	runtime.KeepAlive(distances)
	runtime.KeepAlive(handler)
	if errorMessage != nil {
		return nil, nil, errors.New(C.GoString(errorMessage))
	}

	keys = keys[:resultCount]
	distances = distances[:resultCount]
	return keys, distances, nil
}

// DistanceI8 computes the distance between two int8 vectors.
//
// Example:
//
//	vec1 := []int8{1, 2, 3, 4}
//	vec2 := []int8{5, 6, 7, 8}
//	dist, err := usearch.DistanceI8(vec1, vec2, 4, usearch.L2sq)
func DistanceI8(vec1 []int8, vec2 []int8, vectorDimensions uint, metric Metric) (float32, error) {
	if len(vec1) == 0 || len(vec2) == 0 {
		return 0, errors.New("vectors cannot be empty")
	}
	if uint(len(vec1)) < vectorDimensions || uint(len(vec2)) < vectorDimensions {
		return 0, fmt.Errorf("vectors too short for specified dimensions: need %d elements", vectorDimensions)
	}
	var errorMessage *C.char
	dist := C.usearch_distance(unsafe.Pointer(&vec1[0]), unsafe.Pointer(&vec2[0]), C.usearch_scalar_i8_k, C.size_t(vectorDimensions), metric.CValue(), (*C.usearch_error_t)(&errorMessage))
	runtime.KeepAlive(vec1)
	runtime.KeepAlive(vec2)
	if errorMessage != nil {
		return 0, errors.New(C.GoString(errorMessage))
	}
	return float32(dist), nil
}

// ExactSearchI8 performs exact nearest neighbors search on int8 vectors.
// This computes distances to all vectors in the dataset without using an index.
//
// Stride parameters specify memory offset in bytes between consecutive vectors.
// For contiguous int8 data, use vectorDimensions * 1 byte.
func ExactSearchI8(dataset []int8, queries []int8, datasetSize uint, queryCount uint,
	datasetStride uint, queryStride uint, vectorDimensions uint, metric Metric,
	maxResults uint, numThreads uint) (keys []Key, distances []float32, err error) {

	if len(dataset) == 0 || len(queries) == 0 {
		return nil, nil, errors.New("dataset and queries cannot be empty")
	}
	if vectorDimensions == 0 {
		return nil, nil, errors.New("dimensions must be greater than zero")
	}
	if maxResults == 0 {
		return nil, nil, errors.New("maxResults must be greater than zero")
	}
	keys = make([]Key, queryCount*maxResults)
	distances = make([]float32, queryCount*maxResults)
	resultKeysStride := uint32(maxResults * 8)      // int64 - 8 bytes
	resultDistancesStride := uint32(maxResults * 4) // float32 - 4 bytes
	var errorMessage *C.char
	C.usearch_exact_search(unsafe.Pointer(&dataset[0]), C.size_t(datasetSize), C.size_t(datasetStride), unsafe.Pointer(&queries[0]), C.size_t(queryCount), C.size_t(queryStride),
		C.usearch_scalar_i8_k, C.size_t(vectorDimensions), metric.CValue(), C.size_t(maxResults), C.size_t(numThreads),
		(*C.usearch_key_t)(&keys[0]), C.size_t(resultKeysStride), (*C.usearch_distance_t)(&distances[0]), C.size_t(resultDistancesStride), (*C.usearch_error_t)(&errorMessage))
	runtime.KeepAlive(dataset)
	runtime.KeepAlive(queries)
	runtime.KeepAlive(keys)
	runtime.KeepAlive(distances)
	if errorMessage != nil {
		return nil, nil, errors.New(C.GoString(errorMessage))
	}
	return keys, distances, nil
}

// SaveBuffer serializes the index into a byte buffer.
// The buffer must be large enough to hold the serialized index.
// Use SerializedLength() to determine the required buffer size.
func (index *Index) SaveBuffer(buf []byte, bufferSize uint) error {
	if index.handle == nil {
		panic("index is uninitialized")
	}

	if len(buf) == 0 {
		return errors.New("buffer cannot be empty")
	}
	if uint(len(buf)) < bufferSize {
		return fmt.Errorf("buffer too small: has %d bytes, need %d", len(buf), bufferSize)
	}

	var errorMessage *C.char
	C.usearch_save_buffer(index.handle, unsafe.Pointer(&buf[0]), C.size_t(bufferSize), (*C.usearch_error_t)(&errorMessage))
	runtime.KeepAlive(buf)
	if errorMessage != nil {
		return errors.New(C.GoString(errorMessage))
	}
	return nil
}

// LoadBuffer loads a serialized index from a byte buffer.
// The buffer must contain a valid serialized index.
func (index *Index) LoadBuffer(buf []byte, bufferSize uint) error {
	if index.handle == nil {
		panic("index is uninitialized")
	}

	if len(buf) == 0 {
		return errors.New("buffer cannot be empty")
	}
	if uint(len(buf)) < bufferSize {
		return fmt.Errorf("buffer too small: has %d bytes, need %d", len(buf), bufferSize)
	}

	var errorMessage *C.char
	C.usearch_load_buffer(index.handle, unsafe.Pointer(&buf[0]), C.size_t(bufferSize), (*C.usearch_error_t)(&errorMessage))
	runtime.KeepAlive(buf)
	if errorMessage != nil {
		return errors.New(C.GoString(errorMessage))
	}
	return nil
}

// ViewBuffer creates a view of a serialized index without copying the data.
// The buffer must remain valid for the lifetime of the index.
// Changes to the buffer will affect the index.
func (index *Index) ViewBuffer(buf []byte, bufferSize uint) error {
	if index.handle == nil {
		panic("index is uninitialized")
	}

	if len(buf) == 0 {
		return errors.New("buffer cannot be empty")
	}
	if uint(len(buf)) < bufferSize {
		return fmt.Errorf("buffer too small: has %d bytes, need %d", len(buf), bufferSize)
	}

	var errorMessage *C.char
	C.usearch_view_buffer(index.handle, unsafe.Pointer(&buf[0]), C.size_t(bufferSize), (*C.usearch_error_t)(&errorMessage))
	runtime.KeepAlive(buf)
	if errorMessage != nil {
		return errors.New(C.GoString(errorMessage))
	}
	return nil
}

// MetadataBuffer extracts index configuration metadata from a serialized buffer.
// This can be used to inspect an index before loading it.
func MetadataBuffer(buf []byte, bufferSize uint) (c IndexConfig, err error) {
	if len(buf) == 0 {
		return c, errors.New("buffer cannot be empty")
	}
	if uint(len(buf)) < bufferSize {
		return c, fmt.Errorf("buffer too small: has %d bytes, need %d", len(buf), bufferSize)
	}
	c = IndexConfig{}

	options := C.struct_usearch_init_options_t{}

	var errorMessage *C.char
	C.usearch_metadata_buffer(unsafe.Pointer(&buf[0]), C.size_t(bufferSize), &options, (*C.usearch_error_t)(&errorMessage))
	runtime.KeepAlive(buf)
	if errorMessage != nil {
		return c, errors.New(C.GoString(errorMessage))
	}

	c.Dimensions = uint(options.dimensions)
	c.Connectivity = uint(options.connectivity)
	c.ExpansionAdd = uint(options.expansion_add)
	c.ExpansionSearch = uint(options.expansion_search)
	c.Multi = bool(options.multi)

	// Map the metric kind
	switch options.metric_kind {
	case C.usearch_metric_l2sq_k:
		c.Metric = L2sq
	case C.usearch_metric_ip_k:
		c.Metric = InnerProduct
	case C.usearch_metric_cos_k:
		c.Metric = Cosine
	case C.usearch_metric_haversine_k:
		c.Metric = Haversine
	case C.usearch_metric_pearson_k:
		c.Metric = Pearson
	case C.usearch_metric_hamming_k:
		c.Metric = Hamming
	case C.usearch_metric_tanimoto_k:
		c.Metric = Tanimoto
	case C.usearch_metric_sorensen_k:
		c.Metric = Sorensen
	}

	// Map the quantization method
	switch options.quantization {
	case C.usearch_scalar_f16_k:
		c.Quantization = F16
	case C.usearch_scalar_f32_k:
		c.Quantization = F32
	case C.usearch_scalar_f64_k:
		c.Quantization = F64
	case C.usearch_scalar_i8_k:
		c.Quantization = I8
	case C.usearch_scalar_b1_k:
		c.Quantization = B1
	case C.usearch_scalar_bf16_k:
		c.Quantization = BF16
	}

	return c, nil
}

// Metadata loads the index configuration metadata from a file.
// This can be used to inspect an index file before loading it.
func Metadata(path string) (c IndexConfig, err error) {
	if path == "" {
		return c, errors.New("path cannot be empty")
	}

	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	options := C.struct_usearch_init_options_t{}

	var errorMessage *C.char
	C.usearch_metadata(cPath, &options, (*C.usearch_error_t)(&errorMessage))
	if errorMessage != nil {
		return c, errors.New(C.GoString(errorMessage))
	}

	c.Dimensions = uint(options.dimensions)
	c.Connectivity = uint(options.connectivity)
	c.ExpansionAdd = uint(options.expansion_add)
	c.ExpansionSearch = uint(options.expansion_search)
	c.Multi = bool(options.multi)

	// Map the metric kind
	switch options.metric_kind {
	case C.usearch_metric_l2sq_k:
		c.Metric = L2sq
	case C.usearch_metric_ip_k:
		c.Metric = InnerProduct
	case C.usearch_metric_cos_k:
		c.Metric = Cosine
	case C.usearch_metric_haversine_k:
		c.Metric = Haversine
	case C.usearch_metric_pearson_k:
		c.Metric = Pearson
	case C.usearch_metric_hamming_k:
		c.Metric = Hamming
	case C.usearch_metric_tanimoto_k:
		c.Metric = Tanimoto
	case C.usearch_metric_sorensen_k:
		c.Metric = Sorensen
	}

	// Map the quantization method
	switch options.quantization {
	case C.usearch_scalar_f16_k:
		c.Quantization = F16
	case C.usearch_scalar_f32_k:
		c.Quantization = F32
	case C.usearch_scalar_f64_k:
		c.Quantization = F64
	case C.usearch_scalar_i8_k:
		c.Quantization = I8
	case C.usearch_scalar_b1_k:
		c.Quantization = B1
	case C.usearch_scalar_bf16_k:
		c.Quantization = BF16
	}

	return c, nil
}

// Save saves the index to a specified file.
func (index *Index) Save(path string) error {
	if index.handle == nil {
		panic("index is uninitialized")
	}

	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	var errorMessage *C.char
	C.usearch_save(index.handle, cPath, (*C.usearch_error_t)(&errorMessage))
	if errorMessage != nil {
		return errors.New(C.GoString(errorMessage))
	}
	return nil
}

// Load loads the index from a specified file.
func (index *Index) Load(path string) error {
	if index.handle == nil {
		panic("index is uninitialized")
	}

	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	var errorMessage *C.char
	C.usearch_load(index.handle, cPath, (*C.usearch_error_t)(&errorMessage))
	if errorMessage != nil {
		return errors.New(C.GoString(errorMessage))
	}
	return nil
}

// View creates a view of the index from a specified file without loading it into memory.
func (index *Index) View(path string) error {
	if index.handle == nil {
		panic("index is uninitialized")
	}

	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	var errorMessage *C.char
	C.usearch_view(index.handle, cPath, (*C.usearch_error_t)(&errorMessage))
	if errorMessage != nil {
		return errors.New(C.GoString(errorMessage))
	}
	return nil
}
