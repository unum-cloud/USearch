package usearch

import (
	"fmt"
	"io"
	"math"
	"runtime"
	"sync"
	"testing"
	"unsafe"
)

// Test constants
const (
	defaultTestDimensions = 128
	distanceTolerance     = 1e-2
	bufferSize            = 1024 * 1024
)

// Helper functions to reduce code duplication

func createTestIndex(t *testing.T, dimensions uint, quantization Quantization) *Index {
	conf := DefaultConfig(dimensions)
	conf.Quantization = quantization
	index, err := NewIndex(conf)
	if err != nil {
		t.Fatalf("Failed to create test index: %v", err)
	}
	return index
}

func generateTestVector(dimensions uint) []float32 {
	vector := make([]float32, dimensions)
	for i := uint(0); i < dimensions; i++ {
		vector[i] = float32(i) + 0.1
	}
	return vector
}

func generateTestVectorI8(dimensions uint) []int8 {
	vector := make([]int8, dimensions)
	for i := uint(0); i < dimensions; i++ {
		vector[i] = int8((i % 127) + 1)
	}
	return vector
}

func populateIndex(t *testing.T, index *Index, vectorCount int) [][]float32 {
	vectors := make([][]float32, vectorCount)
	err := index.Reserve(uint(vectorCount))
	if err != nil {
		t.Fatalf("Failed to reserve capacity: %v", err)
	}

	dimensions, err := index.Dimensions()
	if err != nil {
		t.Fatalf("Failed to get dimensions: %v", err)
	}

	for i := 0; i < vectorCount; i++ {
		vector := generateTestVector(dimensions)
		vector[0] = float32(i) // Make each vector unique
		vectors[i] = vector

		err = index.Add(Key(i), vector)
		if err != nil {
			t.Fatalf("Failed to add vector %d: %v", i, err)
		}
	}
	return vectors
}

// Core functionality tests (improved versions of existing)

func TestIndexLifecycle(t *testing.T) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	t.Run("Index creation and configuration", func(t *testing.T) {
		dimensions := uint(64)
		index := createTestIndex(t, dimensions, F32)
		defer func() {
			if err := index.Destroy(); err != nil {
				t.Errorf("Failed to destroy index: %v", err)
			}
		}()

		// Verify dimensions
		actualDimensions, err := index.Dimensions()
		if err != nil {
			t.Fatalf("Failed to retrieve dimensions: %v", err)
		}
		if actualDimensions != dimensions {
			t.Fatalf("Expected %d dimensions, got %d", dimensions, actualDimensions)
		}

		// Verify empty index
		size, err := index.Len()
		if err != nil {
			t.Fatalf("Failed to retrieve size: %v", err)
		}
		if size != 0 {
			t.Fatalf("Expected empty index, got size %d", size)
		}

		// Capacity may be zero before any reservation; ensure Reserve works
		if err := index.Reserve(10); err != nil {
			t.Fatalf("Failed to reserve capacity: %v", err)
		}
		capacity, err := index.Capacity()
		if err != nil {
			t.Fatalf("Failed to retrieve capacity: %v", err)
		}
		if capacity < 10 {
			t.Fatalf("Expected capacity >= 10 after reserve, got %d", capacity)
		}

		// Verify memory usage
		memUsage, err := index.MemoryUsage()
		if err != nil {
			t.Fatalf("Failed to retrieve memory usage: %v", err)
		}
		if memUsage == 0 {
			t.Fatalf("Expected positive memory usage")
		}

		// Verify hardware acceleration info
		hwAccel, err := index.HardwareAcceleration()
		if err != nil {
			t.Fatalf("Failed to retrieve hardware acceleration: %v", err)
		}
		if hwAccel == "" {
			t.Fatalf("Expected non-empty hardware acceleration string")
		}
	})

	t.Run("Index configuration validation", func(t *testing.T) {
		// Test different configurations
		configs := []struct {
			name         string
			dimensions   uint
			quantization Quantization
			metric       Metric
		}{
			{"F32-Cosine", 128, F32, Cosine},
			{"F64-L2sq", 64, F64, L2sq},
			{"I8-InnerProduct", 32, I8, InnerProduct},
		}

		for _, config := range configs {
			t.Run(config.name, func(t *testing.T) {
				conf := DefaultConfig(config.dimensions)
				conf.Quantization = config.quantization
				conf.Metric = config.metric

				index, err := NewIndex(conf)
				if err != nil {
					t.Fatalf("Failed to create index with config %s: %v", config.name, err)
				}
				defer func() {
					if err := index.Destroy(); err != nil {
						t.Errorf("Failed to destroy index: %v", err)
					}
				}()

				actualDims, err := index.Dimensions()
				if err != nil || actualDims != config.dimensions {
					t.Fatalf("Configuration mismatch for %s", config.name)
				}
			})
		}
	})
}

func TestBasicOperations(t *testing.T) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	t.Run("Add and retrieve", func(t *testing.T) {
		index := createTestIndex(t, defaultTestDimensions, F32)
		defer func() {
			if err := index.Destroy(); err != nil {
				t.Errorf("Failed to destroy index: %v", err)
			}
		}()

		// Ensure capacity before first add
		if err := index.Reserve(1); err != nil {
			t.Fatalf("Failed to reserve capacity: %v", err)
		}

		// Add a vector
		vector := generateTestVector(defaultTestDimensions)
		vector[0] = 42.0
		vector[1] = 24.0

		err := index.Add(100, vector)
		if err != nil {
			t.Fatalf("Failed to add vector: %v", err)
		}

		// Verify index size
		size, err := index.Len()
		if err != nil {
			t.Fatalf("Failed to get index size: %v", err)
		}
		if size != 1 {
			t.Fatalf("Expected size 1, got %d", size)
		}

		// Test Contains
		found, err := index.Contains(100)
		if err != nil {
			t.Fatalf("Contains check failed: %v", err)
		}
		if !found {
			t.Fatalf("Expected to find key 100")
		}

		// Test Get
		retrieved, err := index.Get(100, 1)
		if err != nil {
			t.Fatalf("Failed to retrieve vector: %v", err)
		}
		if retrieved == nil || len(retrieved) != int(defaultTestDimensions) {
			t.Fatalf("Retrieved vector has wrong dimensions")
		}
	})

	t.Run("Search functionality", func(t *testing.T) {
		index := createTestIndex(t, defaultTestDimensions, F32)
		defer func() {
			if err := index.Destroy(); err != nil {
				t.Errorf("Failed to destroy index: %v", err)
			}
		}()

		// Add test data
		testVectors := populateIndex(t, index, 10)

		// Search with first vector (should find itself)
		keys, distances, err := index.Search(testVectors[0], 5)
		if err != nil {
			t.Fatalf("Search failed: %v", err)
		}

		if len(keys) == 0 || len(distances) == 0 {
			t.Fatalf("Search returned no results")
		}

		// First result should be the exact match with near-zero distance
		if keys[0] != 0 {
			t.Fatalf("Expected first result to be key 0, got %d", keys[0])
		}

		if math.Abs(float64(distances[0])) > distanceTolerance {
			t.Fatalf("Expected near-zero distance for exact match, got %f", distances[0])
		}
	})

	t.Run("Remove operations", func(t *testing.T) {
		index := createTestIndex(t, defaultTestDimensions, F32)
		defer func() {
			if err := index.Destroy(); err != nil {
				t.Errorf("Failed to destroy index: %v", err)
			}
		}()

		// Add vectors
		populateIndex(t, index, 5)

		// Remove one vector
		err := index.Remove(2)
		if err != nil {
			t.Fatalf("Failed to remove vector: %v", err)
		}

		// Verify it's gone
		found, err := index.Contains(2)
		if err != nil {
			t.Fatalf("Contains check failed after removal: %v", err)
		}
		if found {
			t.Fatalf("Key 2 should have been removed")
		}

		// Verify size decreased
		size, err := index.Len()
		if err != nil {
			t.Fatalf("Failed to get size after removal: %v", err)
		}
		if size != 4 {
			t.Fatalf("Expected size 4 after removal, got %d", size)
		}
	})
}

func TestIOCloser(t *testing.T) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	t.Run("io.Closer interface compliance", func(t *testing.T) {
		index := createTestIndex(t, 32, F32)

		// Verify that Index can be used as io.Closer
		var closer io.Closer = index

		// Test Close method works like Destroy
		err := closer.Close()
		if err != nil {
			t.Fatalf("Close failed: %v", err)
		}
	})
}

func TestSerialization(t *testing.T) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	t.Run("Buffer save/load/view operations", func(t *testing.T) {
		// Create and populate original index
		originalIndex := createTestIndex(t, defaultTestDimensions, F32)
		defer func() {
			if err := originalIndex.Destroy(); err != nil {
				t.Errorf("Failed to destroy original index: %v", err)
			}
		}()

		testVectors := populateIndex(t, originalIndex, 50)

		originalSize, err := originalIndex.Len()
		if err != nil {
			t.Fatalf("Failed to get original index size: %v", err)
		}

		// Save to buffer
		buf := make([]byte, bufferSize)
		err = originalIndex.SaveBuffer(buf, bufferSize)
		if err != nil {
			t.Fatalf("Failed to save index to buffer: %v", err)
		}

		// Test metadata extraction
		metadata, err := MetadataBuffer(buf, bufferSize)
		if err != nil {
			t.Fatalf("Failed to extract metadata: %v", err)
		}

		if metadata.Dimensions != defaultTestDimensions {
			t.Fatalf("Metadata dimensions mismatch: expected %d, got %d",
				defaultTestDimensions, metadata.Dimensions)
		}

		// Test LoadBuffer
		loadedIndex := createTestIndex(t, defaultTestDimensions, F32)
		defer func() {
			if err := loadedIndex.Destroy(); err != nil {
				t.Errorf("Failed to destroy loaded index: %v", err)
			}
		}()

		err = loadedIndex.LoadBuffer(buf, bufferSize)
		if err != nil {
			t.Fatalf("Failed to load index from buffer: %v", err)
		}

		loadedSize, err := loadedIndex.Len()
		if err != nil {
			t.Fatalf("Failed to get loaded index size: %v", err)
		}

		if loadedSize != originalSize {
			t.Fatalf("Loaded index size mismatch: expected %d, got %d",
				originalSize, loadedSize)
		}

		// Verify search results are consistent
		keys, distances, err := loadedIndex.Search(testVectors[0], 3)
		if err != nil {
			t.Fatalf("Search failed on loaded index: %v", err)
		}

		if len(keys) == 0 || keys[0] != 0 {
			t.Fatalf("Loaded index search results inconsistent")
		}

		// Verify distance is near zero for exact match
		if math.Abs(float64(distances[0])) > distanceTolerance {
			t.Fatalf("Expected near-zero distance for exact match, got %f", distances[0])
		}

		// Test ViewBuffer
		viewIndex := createTestIndex(t, defaultTestDimensions, F32)
		defer func() {
			if err := viewIndex.Destroy(); err != nil {
				t.Errorf("Failed to destroy view index: %v", err)
			}
		}()

		err = viewIndex.ViewBuffer(buf, bufferSize)
		if err != nil {
			t.Fatalf("Failed to create view from buffer: %v", err)
		}

		viewSize, err := viewIndex.Len()
		if err != nil {
			t.Fatalf("Failed to get view index size: %v", err)
		}

		if viewSize != originalSize {
			t.Fatalf("View index size mismatch: expected %d, got %d",
				originalSize, viewSize)
		}
	})
}

func TestInputValidation(t *testing.T) {
	t.Run("Zero dimensions", func(t *testing.T) {
		conf := DefaultConfig(0)
		_, err := NewIndex(conf)
		if err == nil {
			t.Fatalf("Expected error for zero dimensions")
		}
	})

	t.Run("Empty vectors", func(t *testing.T) {
		index := createTestIndex(t, 64, F32)
		defer func() {
			if err := index.Destroy(); err != nil {
				t.Errorf("Failed to destroy index: %v", err)
			}
		}()

		// Test Add with empty vector
		err := index.Add(1, []float32{})
		if err == nil {
			t.Fatalf("Expected error for empty vector in Add")
		}

		// Test Search with empty vector
		_, _, err = index.Search([]float32{}, 10)
		if err == nil {
			t.Fatalf("Expected error for empty vector in Search")
		}
	})

	t.Run("Dimension mismatches", func(t *testing.T) {
		index := createTestIndex(t, 64, F32)
		defer func() {
			if err := index.Destroy(); err != nil {
				t.Errorf("Failed to destroy index: %v", err)
			}
		}()

		// Test Add with wrong dimensions
		wrongVec := make([]float32, 32) // Should be 64
		err := index.Add(1, wrongVec)
		if err == nil {
			t.Fatalf("Expected error for dimension mismatch in Add")
		}

		// Test Search with wrong dimensions
		_, _, err = index.Search(wrongVec, 10)
		if err == nil {
			t.Fatalf("Expected error for dimension mismatch in Search")
		}
	})

	t.Run("Nil pointers", func(t *testing.T) {
		index := createTestIndex(t, 64, F32)
		defer func() {
			if err := index.Destroy(); err != nil {
				t.Errorf("Failed to destroy index: %v", err)
			}
		}()

		// Test AddUnsafe with nil pointer
		err := index.AddUnsafe(1, nil)
		if err == nil {
			t.Fatalf("Expected error for nil pointer in AddUnsafe")
		}

		// Test SearchUnsafe with nil pointer
		_, _, err = index.SearchUnsafe(nil, 10)
		if err == nil {
			t.Fatalf("Expected error for nil pointer in SearchUnsafe")
		}
	})

	t.Run("Buffer validation", func(t *testing.T) {
		index := createTestIndex(t, 64, F32)
		defer func() {
			if err := index.Destroy(); err != nil {
				t.Errorf("Failed to destroy index: %v", err)
			}
		}()

		// Test SaveBuffer with empty buffer
		err := index.SaveBuffer([]byte{}, 100)
		if err == nil {
			t.Fatalf("Expected error for empty buffer in SaveBuffer")
		}

		// Test LoadBuffer with empty buffer
		err = index.LoadBuffer([]byte{}, 100)
		if err == nil {
			t.Fatalf("Expected error for empty buffer in LoadBuffer")
		}
	})
}

func TestQuantizationTypes(t *testing.T) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	t.Run("F32 operations", func(t *testing.T) {
		index := createTestIndex(t, 32, F32)
		defer func() {
			if err := index.Destroy(); err != nil {
				t.Errorf("Failed to destroy index: %v", err)
			}
		}()

		if err := index.Reserve(1); err != nil {
			t.Fatalf("Failed to reserve capacity: %v", err)
		}
		vector := generateTestVector(32)
		err := index.Add(1, vector)
		if err != nil {
			t.Fatalf("F32 Add failed: %v", err)
		}

		keys, _, err := index.Search(vector, 1)
		if err != nil {
			t.Fatalf("F32 Search failed: %v", err)
		}

		if len(keys) == 0 || keys[0] != 1 {
			t.Fatalf("F32 search results incorrect")
		}
	})

	t.Run("F64 operations", func(t *testing.T) {
		index := createTestIndex(t, 32, F64)
		defer func() {
			if err := index.Destroy(); err != nil {
				t.Errorf("Failed to destroy index: %v", err)
			}
		}()

		if err := index.Reserve(1); err != nil {
			t.Fatalf("Failed to reserve capacity: %v", err)
		}
		vector := make([]float64, 32)
		for i := range vector {
			vector[i] = float64(i) + 0.5
		}

		err := index.AddUnsafe(1, unsafe.Pointer(&vector[0]))
		if err != nil {
			t.Fatalf("F64 AddUnsafe failed: %v", err)
		}

		keys, _, err := index.SearchUnsafe(unsafe.Pointer(&vector[0]), 1)
		if err != nil {
			t.Fatalf("F64 SearchUnsafe failed: %v", err)
		}

		if len(keys) == 0 || keys[0] != 1 {
			t.Fatalf("F64 search results incorrect")
		}
	})

	t.Run("I8 operations", func(t *testing.T) {
		index := createTestIndex(t, 32, I8)
		defer func() {
			if err := index.Destroy(); err != nil {
				t.Errorf("Failed to destroy index: %v", err)
			}
		}()

		if err := index.Reserve(1); err != nil {
			t.Fatalf("Failed to reserve capacity: %v", err)
		}
		vector := generateTestVectorI8(32)
		err := index.AddI8(1, vector)
		if err != nil {
			t.Fatalf("I8 Add failed: %v", err)
		}

		keys, _, err := index.SearchI8(vector, 1)
		if err != nil {
			t.Fatalf("I8 Search failed: %v", err)
		}

		if len(keys) == 0 || keys[0] != 1 {
			t.Fatalf("I8 search results incorrect")
		}
	})
}

func TestUnsafeOperations(t *testing.T) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	t.Run("Unsafe pointer operations", func(t *testing.T) {
		index := createTestIndex(t, 64, F32)
		defer func() {
			if err := index.Destroy(); err != nil {
				t.Errorf("Failed to destroy index: %v", err)
			}
		}()

		if err := index.Reserve(1); err != nil {
			t.Fatalf("Failed to reserve capacity: %v", err)
		}
		vector := generateTestVector(64)
		ptr := unsafe.Pointer(&vector[0])

		// Test AddUnsafe
		err := index.AddUnsafe(100, ptr)
		if err != nil {
			t.Fatalf("AddUnsafe failed: %v", err)
		}

		// Verify vector was added
		size, err := index.Len()
		if err != nil {
			t.Fatalf("Failed to get size after AddUnsafe: %v", err)
		}
		if size != 1 {
			t.Fatalf("Expected size 1 after AddUnsafe, got %d", size)
		}

		// Test SearchUnsafe
		keys, distances, err := index.SearchUnsafe(ptr, 5)
		if err != nil {
			t.Fatalf("SearchUnsafe failed: %v", err)
		}

		if len(keys) == 0 || keys[0] != 100 {
			t.Fatalf("SearchUnsafe returned incorrect results")
		}

		if math.Abs(float64(distances[0])) > distanceTolerance {
			t.Fatalf("Expected near-zero distance for exact match, got %f", distances[0])
		}

		// Test FilteredSearchUnsafe
		handler := &FilteredSearchHandler{
			f: func(key Key, ptr unsafe.Pointer) int {
				if key%2 == 0 {
					return 0
				}
				return 1
			},
			data: int64(1),
		}

		keys, distances, err = index.FilteredSearchUnsafe(ptr, 5, handler)
		if err != nil {
			t.Fatalf("FilteredSearchUnsafe failed: %v", err)
		}

		if len(keys) > 0 {
			t.Fatalf("FilteredSearchUnsafe returned incorrect results")
		}
	})
}

func TestConcurrentInsertions(t *testing.T) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	t.Run("Parallelized insertions via internal threads", func(t *testing.T) {
		index := createTestIndex(t, 64, F32)
		defer func() {
			if err := index.Destroy(); err != nil {
				t.Errorf("Failed to destroy index: %v", err)
			}
		}()

		const totalVectors = 1000

		err := index.Reserve(totalVectors)
		if err != nil {
			t.Fatalf("Failed to reserve capacity: %v", err)
		}

		// Let the library parallelize inserts internally
		_ = index.ChangeThreadsAdd(uint(runtime.NumCPU()))

		for i := 0; i < totalVectors; i++ {
			vector := generateTestVector(64)
			vector[0] = float32(i)
			if err := index.Add(Key(i), vector); err != nil {
				t.Fatalf("Insertion failed at %d: %v", i, err)
			}
		}

		// Verify final count
		finalSize, err := index.Len()
		if err != nil {
			t.Fatalf("Failed to get final size: %v", err)
		}

		if finalSize != totalVectors {
			t.Fatalf("Expected %d vectors after concurrent insertions, got %d",
				totalVectors, finalSize)
		}
	})
}

func TestConcurrentSearches(t *testing.T) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	t.Run("Multiple concurrent searches", func(t *testing.T) {
		index := createTestIndex(t, 64, F32)
		defer func() {
			if err := index.Destroy(); err != nil {
				t.Errorf("Failed to destroy index: %v", err)
			}
		}()

		// Pre-populate with data
		testVectors := populateIndex(t, index, 200)

		// Let the library parallelize search internally as well
		_ = index.ChangeThreadsSearch(uint(runtime.NumCPU()))

		const numGoroutines = 30
		const searchesPerGoroutine = 50

		var wg sync.WaitGroup
		errorChan := make(chan error, numGoroutines)

		// Only concurrent searches - no mixed operations
		for i := 0; i < numGoroutines; i++ {
			wg.Add(1)
			go func(goroutineID int) {
				defer wg.Done()

				for j := 0; j < searchesPerGoroutine; j++ {
					// Use different query vectors
					queryIndex := (goroutineID*searchesPerGoroutine + j) % len(testVectors)
					query := testVectors[queryIndex]

					keys, distances, err := index.Search(query, 10)
					if err != nil {
						errorChan <- err
						return
					}

					// Basic validation - should find at least the exact match
					if len(keys) == 0 || len(distances) == 0 {
						errorChan <- fmt.Errorf("search returned empty results")
						return
					}

					// First result should be the exact match
					if keys[0] != Key(queryIndex) || math.Abs(float64(distances[0])) > distanceTolerance {
						errorChan <- fmt.Errorf("search results inconsistent: expected key %d, got %d", queryIndex, keys[0])
						return
					}
				}
			}(i)
		}

		wg.Wait()
		close(errorChan)

		// Check for any errors
		for err := range errorChan {
			t.Fatalf("Concurrent search failed: %v", err)
		}
	})
}

func TestExactSearch(t *testing.T) {
	t.Run("Float32 exact search", func(t *testing.T) {
		// Create dataset and queries
		const datasetSize = 100
		const querySize = 10
		const vectorDims = 32

		dataset := make([]float32, datasetSize*vectorDims)
		queries := make([]float32, querySize*vectorDims)

		// Fill with test data
		for i := 0; i < len(dataset); i++ {
			dataset[i] = float32(i%100) + 0.1
		}

		for i := 0; i < len(queries); i++ {
			queries[i] = float32(i%50) + 0.1
		}

		keys, distances, err := ExactSearch(
			dataset, queries,
			datasetSize, querySize,
			vectorDims*4, vectorDims*4, // Stride in bytes for float32
			vectorDims, Cosine,
			5, 0, // maxResults=5, numThreads=0 (auto)
			8, 4, // resultKeysStride, resultDistancesStride
		)

		if err != nil {
			t.Fatalf("ExactSearch failed: %v", err)
		}

		if len(keys) != 5 || len(distances) != 5 {
			t.Fatalf("Expected 5 results from ExactSearch, got %d keys and %d distances",
				len(keys), len(distances))
		}
	})

	t.Run("I8 exact search", func(t *testing.T) {
		const datasetSize = 50
		const querySize = 5
		const vectorDims = 16

		dataset := make([]int8, datasetSize*vectorDims)
		queries := make([]int8, querySize*vectorDims)

		// Fill with test data
		for i := 0; i < len(dataset); i++ {
			dataset[i] = int8((i % 100) + 1)
		}

		for i := 0; i < len(queries); i++ {
			queries[i] = int8((i % 50) + 1)
		}

		keys, distances, err := ExactSearchI8(
			dataset, queries,
			datasetSize, querySize,
			vectorDims, vectorDims, // Stride in bytes for int8
			vectorDims, L2sq,
			3, 0, // maxResults=3, numThreads=0 (auto)
			8, 4, // resultKeysStride, resultDistancesStride
		)

		if err != nil {
			t.Fatalf("ExactSearchI8 failed: %v", err)
		}

		if len(keys) != 3 || len(distances) != 3 {
			t.Fatalf("Expected 3 results from ExactSearchI8, got %d keys and %d distances",
				len(keys), len(distances))
		}
	})
}

func TestDistanceCalculations(t *testing.T) {
	t.Run("Float32 distance calculations", func(t *testing.T) {
		vec1 := []float32{1.0, 0.0, 0.0}
		vec2 := []float32{0.0, 1.0, 0.0}

		// Test different metrics
		metrics := []struct {
			metric    Metric
			expected  float32
			tolerance float32
		}{
			{Cosine, 1.0, 0.01}, // Perpendicular vectors
			{L2sq, 2.0, 0.01},   // Squared Euclidean distance
		}

		for _, test := range metrics {
			distance, err := Distance(vec1, vec2, 3, test.metric)
			if err != nil {
				t.Fatalf("Distance calculation failed for %v: %v", test.metric, err)
			}

			if math.Abs(float64(distance-test.expected)) > float64(test.tolerance) {
				t.Fatalf("Distance mismatch for %v: expected %f, got %f",
					test.metric, test.expected, distance)
			}
		}
	})

	t.Run("I8 distance calculations", func(t *testing.T) {
		vec1 := []int8{10, 0, 0}
		vec2 := []int8{0, 10, 0}

		distance, err := DistanceI8(vec1, vec2, 3, L2sq)
		if err != nil {
			t.Fatalf("DistanceI8 failed: %v", err)
		}

		expected := float32(200.0) // 10^2 + 10^2 = 200
		if math.Abs(float64(distance-expected)) > 0.1 {
			t.Fatalf("I8 distance mismatch: expected %f, got %f", expected, distance)
		}
	})
}
