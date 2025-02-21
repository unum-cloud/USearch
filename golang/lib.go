package usearch

import (
	"errors"
	"unsafe"
)

/*
#cgo LDFLAGS: -L. -L/usr/local/lib -lusearch_c
#include "usearch.h"
#include <stdlib.h>
*/
import "C"

// Key represents the type for keys used in the USearch index.
type Key = uint64

// Metric represents the type for different metrics used in distance calculations.
type Metric uint8

// Different metric kinds supported by the USearch library.
const (
	InnerProduct Metric = iota
	Cosine
	L2sq
	Haversine
	Divergence
	Pearson
	Hamming
	Tanimoto
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
		panic("Unknown metric")
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

// Quantization represents the type for different scalar kinds used in quantization.
type Quantization uint8

// Different quantization kinds supported by the USearch library.
const (
	F32 Quantization = iota
	BF16
	F16
	F64
	I8
	B1
)

// String returns the string representation of the Quantization.
func (a Quantization) String() string {
	switch a {
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
		panic("Unknown quantization")
	}
}

// IndexConfig represents the configuration options for initializing a USearch index.
type IndexConfig struct {
	Quantization    Quantization // The scalar kind used for quantization of vector data during indexing.
	Metric          Metric       // The metric kind used for distance calculation between vectors.
	Dimensions      uint         // The number of dimensions in the vectors to be indexed.
	Connectivity    uint         // The optional connectivity parameter that limits connections-per-node in the graph.
	ExpansionAdd    uint         // The optional expansion factor used for index construction when adding vectors.
	ExpansionSearch uint         // The optional expansion factor used for index construction during search operations.
	Multi           bool         // Indicates whether multiple vectors can map to the same key.
}

// DefaultConfig returns an IndexConfig with default values for the specified number of dimensions.
func DefaultConfig(dimensions uint) IndexConfig {
	c := IndexConfig{}
	c.Dimensions = dimensions
	c.Metric = Cosine
	// Zeros will be replaced by the underlying C implementation
	c.Connectivity = 0
	c.ExpansionAdd = 0
	c.ExpansionSearch = 0
	c.Multi = false
	return c
}

// Index represents a USearch index.
type Index struct {
	opaque_handle *C.void
	config        IndexConfig
}

// NewIndex initializes a new instance of the index with the specified configuration.
func NewIndex(conf IndexConfig) (index *Index, err error) {
	index = &Index{config: conf}

	conf = index.config
	dimensions := C.size_t(conf.Dimensions)
	connectivity := C.size_t(conf.Connectivity)
	expansion_add := C.size_t(conf.ExpansionAdd)
	expansion_search := C.size_t(conf.ExpansionSearch)
	multi := C.bool(conf.Multi)

	options := C.struct_usearch_init_options_t{}
	options.dimensions = dimensions
	options.connectivity = connectivity
	options.expansion_add = expansion_add
	options.expansion_search = expansion_search
	options.multi = multi
	options.metric_kind = conf.Metric.CValue()

	// Map the quantization method
	switch conf.Quantization {
	case F16:
		options.quantization = C.usearch_scalar_f16_k
	case F32:
		options.quantization = C.usearch_scalar_f32_k
	case F64:
		options.quantization = C.usearch_scalar_f64_k
	case I8:
		options.quantization = C.usearch_scalar_i8_k
	case B1:
		options.quantization = C.usearch_scalar_b1_k
	default:
		options.quantization = C.usearch_scalar_unknown_k
	}

	var errorMessage *C.char
	ptr := C.usearch_init(&options, (*C.usearch_error_t)(&errorMessage))
	if errorMessage != nil {
		return nil, errors.New(C.GoString(errorMessage))
	}

	index.opaque_handle = (*C.void)(unsafe.Pointer(ptr))
	return index, nil
}

// Len returns the number of vectors in the index.
func (index *Index) Len() (len uint, err error) {
	var errorMessage *C.char
	len = uint(C.usearch_size((C.usearch_index_t)(unsafe.Pointer(index.opaque_handle)), (*C.usearch_error_t)(&errorMessage)))
	if errorMessage != nil {
		err = errors.New(C.GoString(errorMessage))
	}
	return len, err
}

// SerializedLength reports the expected file size after serialization.
func (index *Index) SerializedLength() (len uint, err error) {
	var errorMessage *C.char
	len = uint(C.usearch_serialized_length((C.usearch_index_t)(unsafe.Pointer(index.opaque_handle)), (*C.usearch_error_t)(&errorMessage)))
	if errorMessage != nil {
		err = errors.New(C.GoString(errorMessage))
	}
	return len, err
}

// MemoryUsage reports the memory usage of the index
func (index *Index) MemoryUsage() (len uint, err error) {
	var errorMessage *C.char
	len = uint(C.usearch_memory_usage((C.usearch_index_t)(unsafe.Pointer(index.opaque_handle)), (*C.usearch_error_t)(&errorMessage)))
	if errorMessage != nil {
		err = errors.New(C.GoString(errorMessage))
	}
	return len, err
}

// ExpansionAdd returns the expansion value used during index creation
func (index *Index) ExpansionAdd() (val uint, err error) {
	var errorMessage *C.char
	val = uint(C.usearch_expansion_add((C.usearch_index_t)(unsafe.Pointer(index.opaque_handle)), (*C.usearch_error_t)(&errorMessage)))
	if errorMessage != nil {
		err = errors.New(C.GoString(errorMessage))
	}
	return val, err
}

// ExpansionSearch returns the expansion value used during search
func (index *Index) ExpansionSearch() (val uint, err error) {
	var errorMessage *C.char
	val = uint(C.usearch_expansion_search((C.usearch_index_t)(unsafe.Pointer(index.opaque_handle)), (*C.usearch_error_t)(&errorMessage)))
	if errorMessage != nil {
		err = errors.New(C.GoString(errorMessage))
	}
	return val, err
}

// ChangeExpansionAdd sets the expansion value used during index creation
func (index *Index) ChangeExpansionAdd(val uint) error {
	var errorMessage *C.char
	C.usearch_change_expansion_add((C.usearch_index_t)(unsafe.Pointer(index.opaque_handle)), C.size_t(val), (*C.usearch_error_t)(&errorMessage))
	if errorMessage != nil {
		return errors.New(C.GoString(errorMessage))
	}
	return nil
}

// ChangeExpansionSearch sets the expansion value used during search
func (index *Index) ChangeExpansionSearch(val uint) error {
	var errorMessage *C.char
	C.usearch_change_expansion_search((C.usearch_index_t)(unsafe.Pointer(index.opaque_handle)), C.size_t(val), (*C.usearch_error_t)(&errorMessage))
	if errorMessage != nil {
		return errors.New(C.GoString(errorMessage))
	}
	return nil
}

// ChangeThreadsAdd sets the threads limit for add
func (index *Index) ChangeThreadsAdd(val uint) error {
	var errorMessage *C.char
	C.usearch_change_threads_add((C.usearch_index_t)(unsafe.Pointer(index.opaque_handle)), C.size_t(val), (*C.usearch_error_t)(&errorMessage))
	if errorMessage != nil {
		return errors.New(C.GoString(errorMessage))
	}
	return nil
}

// ChangeThreadsSearch sets the threads limit for search
func (index *Index) ChangeThreadsSearch(val uint) error {
	var errorMessage *C.char
	C.usearch_change_threads_search((C.usearch_index_t)(unsafe.Pointer(index.opaque_handle)), C.size_t(val), (*C.usearch_error_t)(&errorMessage))
	if errorMessage != nil {
		return errors.New(C.GoString(errorMessage))
	}
	return nil
}

// Connectivity returns the connectivity parameter of the index.
func (index *Index) Connectivity() (con uint, err error) {
	var errorMessage *C.char
	con = uint(C.usearch_connectivity((C.usearch_index_t)(unsafe.Pointer(index.opaque_handle)), (*C.usearch_error_t)(&errorMessage)))
	if errorMessage != nil {
		err = errors.New(C.GoString(errorMessage))
	}
	return con, err
}

// Dimensions returns the number of dimensions of the vectors in the index.
func (index *Index) Dimensions() (dim uint, err error) {
	var errorMessage *C.char
	dim = uint(C.usearch_dimensions((C.usearch_index_t)(unsafe.Pointer(index.opaque_handle)), (*C.usearch_error_t)(&errorMessage)))
	if errorMessage != nil {
		err = errors.New(C.GoString(errorMessage))
	}
	return dim, err
}

// Capacity returns the capacity (maximum number of vectors) of the index.
func (index *Index) Capacity() (cap uint, err error) {
	var errorMessage *C.char
	cap = uint(C.usearch_capacity((C.usearch_index_t)(unsafe.Pointer(index.opaque_handle)), (*C.usearch_error_t)(&errorMessage)))
	if errorMessage != nil {
		err = errors.New(C.GoString(errorMessage))
	}
	return cap, err
}

// HardwareAcceleration returns a string showing the SIMD capability for the index
func (index *Index) HardwareAcceleration() (string, error) {
	var str *C.char
	var errorMessage *C.char
	str = C.usearch_hardware_acceleration((C.usearch_index_t)(unsafe.Pointer(index.opaque_handle)), (*C.usearch_error_t)(&errorMessage))
	if errorMessage != nil {
		return C.GoString(nil), errors.New(C.GoString(errorMessage))
	}
	return C.GoString(str), nil
}

// Destroy frees the resources associated with the index.
func (index *Index) Destroy() error {
	if index.opaque_handle == nil {
		panic("Index is uninitialized")
	}

	var errorMessage *C.char
	C.usearch_free((C.usearch_index_t)(unsafe.Pointer(index.opaque_handle)), (*C.usearch_error_t)(&errorMessage))
	if errorMessage != nil {
		return errors.New(C.GoString(errorMessage))
	}
	index.opaque_handle = nil
	index.config = IndexConfig{}
	return nil
}

// Reserve reserves memory for a specified number of incoming vectors.
func (index *Index) Reserve(capacity uint) error {
	if index.opaque_handle == nil {
		panic("Index is uninitialized")
	}

	var errorMessage *C.char
	C.usearch_reserve((C.usearch_index_t)(unsafe.Pointer(index.opaque_handle)), (C.size_t)(capacity), (*C.usearch_error_t)(&errorMessage))
	if errorMessage != nil {
		return errors.New(C.GoString(errorMessage))
	}
	return nil
}

// Add adds a vector with a specified key to the index.
func (index *Index) Add(key Key, vec []float32) error {
	if index.opaque_handle == nil {
		panic("Index is uninitialized")
	}

	var errorMessage *C.char
	C.usearch_add((C.usearch_index_t)(unsafe.Pointer(index.opaque_handle)), (C.usearch_key_t)(key), unsafe.Pointer(&vec[0]), C.usearch_scalar_f32_k, (*C.usearch_error_t)(&errorMessage))
	if errorMessage != nil {
		return errors.New(C.GoString(errorMessage))
	}
	return nil
}

// Remove removes the vector associated with the given key from the index.
func (index *Index) Remove(key Key) error {
	if index.opaque_handle == nil {
		panic("Index is uninitialized")
	}

	var errorMessage *C.char
	C.usearch_remove((C.usearch_index_t)(unsafe.Pointer(index.opaque_handle)), (C.usearch_key_t)(key), (*C.usearch_error_t)(&errorMessage))
	if errorMessage != nil {
		return errors.New(C.GoString(errorMessage))
	}
	return nil
}

// Contains checks if the index contains a vector with a specific key.
func (index *Index) Contains(key Key) (found bool, err error) {
	if index.opaque_handle == nil {
		panic("Index is uninitialized")
	}

	var errorMessage *C.char
	found = bool(C.usearch_contains((C.usearch_index_t)(unsafe.Pointer(index.opaque_handle)), (C.usearch_key_t)(key), (*C.usearch_error_t)(&errorMessage)))
	if errorMessage != nil {
		return found, errors.New(C.GoString(errorMessage))
	}
	return found, nil
}

// Get retrieves the vectors associated with the given key from the index.
func (index *Index) Get(key Key, count uint) (vectors []float32, err error) {
	if index.opaque_handle == nil {
		panic("Index is uninitialized")
	}

	vectors = make([]float32, index.config.Dimensions*count)
	var errorMessage *C.char
	found := uint(C.usearch_get((C.usearch_index_t)(unsafe.Pointer(index.opaque_handle)), (C.usearch_key_t)(key), (C.size_t)(count), unsafe.Pointer(&vectors[0]), C.usearch_scalar_f32_k, (*C.usearch_error_t)(&errorMessage)))
	if errorMessage != nil {
		return nil, errors.New(C.GoString(errorMessage))
	}
	if found == 0 {
		return nil, nil
	}
	return vectors, nil
}

// Rename the vector at key from to key to
func (index *Index) Rename(from Key, to Key) error {
	var errorMessage *C.char
	C.usearch_rename((C.usearch_index_t)(unsafe.Pointer(index.opaque_handle)), C.usearch_key_t(from), C.usearch_key_t(to), (*C.usearch_error_t)(&errorMessage))
	if errorMessage != nil {
		return errors.New(C.GoString(errorMessage))
	}
	return nil
}

// Distance computes the distance between two vectors
func Distance(vec1 []float32, vec2 []float32, dims uint, metric Metric) (float32, error) {

	var errorMessage *C.char
	dist := C.usearch_distance(unsafe.Pointer(&vec1[0]), unsafe.Pointer(&vec2[0]), C.usearch_scalar_f32_k, C.size_t(dims), metric.CValue(), (*C.usearch_error_t)(&errorMessage))
	if errorMessage != nil {
		return 0, errors.New(C.GoString(errorMessage))
	}
	return float32(dist), nil
}

// Search performs k-Approximate Nearest Neighbors Search for the closest vectors to the query vector.
func (index *Index) Search(query []float32, limit uint) (keys []Key, distances []float32, err error) {
	if index.opaque_handle == nil {
		panic("Index is uninitialized")
	}
	if len(query) != int(index.config.Dimensions) {
		return nil, nil, errors.New("Number of dimensions doesn't match!")
	}

	keys = make([]Key, limit)
	distances = make([]float32, limit)
	var errorMessage *C.char
	count := uint(C.usearch_search((C.usearch_index_t)(unsafe.Pointer(index.opaque_handle)), unsafe.Pointer(&query[0]), C.usearch_scalar_f32_k, (C.size_t)(limit), (*C.usearch_key_t)(&keys[0]), (*C.usearch_distance_t)(&distances[0]), (*C.usearch_error_t)(&errorMessage)))
	if errorMessage != nil {
		return nil, nil, errors.New(C.GoString(errorMessage))
	}

	keys = keys[:count]
	distances = distances[:count]
	return keys, distances, nil
}

// ExactSearch is a multithreaded exact nearest neighbors search
func ExactSearch(dataset []float32, queries []float32, dataset_size uint, queries_size uint,
	dataset_stride uint, queries_stride uint, dims uint, metric Metric,
	count uint, threads uint, keys_stride uint, distances_stride uint) (keys []Key, distances []float32, err error) {
	if (len(dataset) % int(dims)) != 0 {
		return nil, nil, errors.New("Dataset length must be a multiple of the dimensions")
	}
	if (len(queries) % int(dims)) != 0 {
		return nil, nil, errors.New("Queries length must be a multiple of the dimensions")
	}

	keys = make([]Key, count)
	distances = make([]float32, count)
	var errorMessage *C.char
	C.usearch_exact_search(unsafe.Pointer(&dataset[0]), C.size_t(dataset_size), C.size_t(dataset_stride), unsafe.Pointer(&queries[0]), C.size_t(queries_size), C.size_t(queries_stride),
		C.usearch_scalar_f32_k, C.size_t(dims), metric.CValue(), C.size_t(count), C.size_t(threads),
		(*C.usearch_key_t)(&keys[0]), C.size_t(keys_stride), (*C.usearch_distance_t)(&distances[0]), C.size_t(distances_stride), (*C.usearch_error_t)(&errorMessage))
	if errorMessage != nil {
		return nil, nil, errors.New(C.GoString(errorMessage))
	}

	keys = keys[:count]
	distances = distances[:count]
	return keys, distances, nil
}

// Save saves the index to a specified buffer.
func (index *Index) SaveBuffer(buf []byte, buffer_size uint) error {
	if index.opaque_handle == nil {
		panic("Index is uninitialized")
	}

	var errorMessage *C.char
	C.usearch_save_buffer((C.usearch_index_t)(unsafe.Pointer(index.opaque_handle)), unsafe.Pointer(&buf[0]), C.size_t(buffer_size), (*C.usearch_error_t)(&errorMessage))
	if errorMessage != nil {
		return errors.New(C.GoString(errorMessage))
	}
	return nil
}

// Loads the index from a specified buffer.
func (index *Index) LoadBuffer(buf []byte, buffer_size uint) error {
	if index.opaque_handle == nil {
		panic("Index is uninitialized")
	}

	var errorMessage *C.char
	C.usearch_load_buffer((C.usearch_index_t)(unsafe.Pointer(index.opaque_handle)), unsafe.Pointer(&buf[0]), C.size_t(buffer_size), (*C.usearch_error_t)(&errorMessage))
	if errorMessage != nil {
		return errors.New(C.GoString(errorMessage))
	}
	return nil
}

// Loads the index from a specified buffer without copying the data.
func (index *Index) ViewBuffer(buf []byte, buffer_size uint) error {
	if index.opaque_handle == nil {
		panic("Index is uninitialized")
	}

	var errorMessage *C.char
	C.usearch_view_buffer((C.usearch_index_t)(unsafe.Pointer(index.opaque_handle)), unsafe.Pointer(&buf[0]), C.size_t(buffer_size), (*C.usearch_error_t)(&errorMessage))
	if errorMessage != nil {
		return errors.New(C.GoString(errorMessage))
	}
	return nil
}

// Loads the metadata from a specified buffer.
func MetadataBuffer(buf []byte, buffer_size uint) (c IndexConfig, err error) {
	if buf == nil {
		panic("Buffer is uninitialized")
	}
	c = IndexConfig{}

	options := C.struct_usearch_init_options_t{}

	var errorMessage *C.char
	C.usearch_metadata_buffer(unsafe.Pointer(&buf[0]), C.size_t(buffer_size), &options, (*C.usearch_error_t)(&errorMessage))
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
	}

	return c, nil
}

// Metadata loads the metadata from a specified file.
func Metadata(path string) (c IndexConfig, err error) {

	c_path := C.CString(path)
	defer C.free(unsafe.Pointer(c_path))

	options := C.struct_usearch_init_options_t{}

	var errorMessage *C.char
	C.usearch_metadata(c_path, &options, (*C.usearch_error_t)(&errorMessage))
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
	}

	return c, nil
}

// Save saves the index to a specified file.
func (index *Index) Save(path string) error {
	if index.opaque_handle == nil {
		panic("Index is uninitialized")
	}

	c_path := C.CString(path)
	defer C.free(unsafe.Pointer(c_path))

	var errorMessage *C.char
	C.usearch_save((C.usearch_index_t)(unsafe.Pointer(index.opaque_handle)), c_path, (*C.usearch_error_t)(&errorMessage))
	if errorMessage != nil {
		return errors.New(C.GoString(errorMessage))
	}
	return nil
}

// Load loads the index from a specified file.
func (index *Index) Load(path string) error {
	if index.opaque_handle == nil {
		panic("Index is uninitialized")
	}

	c_path := C.CString(path)
	defer C.free(unsafe.Pointer(c_path))

	var errorMessage *C.char
	C.usearch_load((C.usearch_index_t)(unsafe.Pointer(index.opaque_handle)), c_path, (*C.usearch_error_t)(&errorMessage))
	if errorMessage != nil {
		return errors.New(C.GoString(errorMessage))
	}
	return nil
}

// View creates a view of the index from a specified file without loading it into memory.
func (index *Index) View(path string) error {
	if index.opaque_handle == nil {
		panic("Index is uninitialized")
	}

	c_path := C.CString(path)
	defer C.free(unsafe.Pointer(c_path))

	var errorMessage *C.char
	C.usearch_view((C.usearch_index_t)(unsafe.Pointer(index.opaque_handle)), c_path, (*C.usearch_error_t)(&errorMessage))
	if errorMessage != nil {
		return errors.New(C.GoString(errorMessage))
	}
	return nil
}
