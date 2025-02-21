package usearch

import (
	"math"
	"runtime"
	"testing"
)

func TestUSearch(t *testing.T) {
	runtime.LockOSThread()

	t.Run("Test Index Initialization", func(t *testing.T) {
		dim := uint(128)
		conf := DefaultConfig(dim)
		ind, err := NewIndex(conf)
		if err != nil {
			t.Fatalf("Failed to construct the index: %s", err)
		}
		defer ind.Destroy()

		found_dims, err := ind.Dimensions()
		if err != nil {
			t.Fatalf("Failed to retrieve dimensions: %s", err)
		}
		if found_dims != dim {
			t.Fatalf("Expected %d dimensions, got %d", dim, found_dims)
		}

		found_len, err := ind.Len()
		if err != nil {
			t.Fatalf("Failed to retrieve size: %s", err)
		}
		if found_len != 0 {
			t.Fatalf("Expected size to be 0, got %d", found_len)
		}

		found_len, err = ind.SerializedLength()
		if err != nil {
			t.Fatalf("Failed to retrieve serialized length: %s", err)
		}
		if found_len != 112 {
			t.Fatalf("Expected serialized length to be 112, got %d", found_len)
		}

		err = ind.Reserve(100)
		if err != nil {
			t.Fatalf("Failed to reserve capacity: %s", err)
		}

		mem, err := ind.MemoryUsage()
		if err != nil {
			t.Fatalf("Failed to retrieve serialized length: %s", err)
		}
		if mem == 0 {
			t.Fatalf("Expected the empty index memory usage to be positive, got zero")
		}

		s, err := ind.HardwareAcceleration()
		if err != nil {
			t.Fatalf("Failed to retrieve hardware acceleration: %s", err)
		}
		if s == "" {
			t.Fatalf("An empty string was returned from HardwareAcceleration")
		}

	})

	t.Run("Test Insertion", func(t *testing.T) {
		dim := uint(128)
		conf := DefaultConfig(dim)
		ind, err := NewIndex(conf)
		if err != nil {
			t.Fatalf("Failed to construct the index: %s", err)
		}
		defer ind.Destroy()

		err = ind.Reserve(100)
		if err != nil {
			t.Fatalf("Failed to reserve capacity: %s", err)
		}

		err = ind.ChangeThreadsAdd(10)
		if err != nil {
			t.Fatalf("Failed to change threads add: %s", err)
		}

		vec := make([]float32, dim)
		vec[0] = 40.0
		vec[1] = 2.0

		err = ind.Add(42, vec)
		if err != nil {
			t.Fatalf("Failed to insert: %s", err)
		}

		found_len, err := ind.Len()
		if err != nil {
			t.Fatalf("Failed to retrieve size after insertion: %s", err)
		}
		if found_len != 1 {
			t.Fatalf("Expected size to be 1, got %d", found_len)
		}
	})

	t.Run("Test Search", func(t *testing.T) {
		dim := uint(128)
		conf := DefaultConfig(dim)
		ind, err := NewIndex(conf)
		if err != nil {
			t.Fatalf("Failed to construct the index: %s", err)
		}
		defer ind.Destroy()

		err = ind.Reserve(100)
		if err != nil {
			t.Fatalf("Failed to reserve capacity: %s", err)
		}

		err = ind.ChangeThreadsSearch(10)
		if err != nil {
			t.Fatalf("Failed to change threads search: %s", err)
		}

		vec := make([]float32, dim)
		vec[0] = 40.0
		vec[1] = 2.0

		err = ind.Add(42, vec)
		if err != nil {
			t.Fatalf("Failed to insert: %s", err)
		}

		keys, distances, err := ind.Search(vec, 10)
		if err != nil {
			t.Fatalf("Failed to search: %s", err)
		}

		const tolerance = 1e-2 // For example, this sets the tolerance to 0.01
		if keys[0] != 42 || math.Abs(float64(distances[0])) > tolerance {
			t.Fatalf("Expected result 42 with distance 0, got key %d with distance %f", keys[0], distances[0])
		}

		// TODO: Add exact search
	})

	t.Run("Test Save and Load", func(t *testing.T) {
		dim := uint(128)
		conf := DefaultConfig(dim)
		ind, err := NewIndex(conf)
		if err != nil {
			t.Fatalf("Failed to construct the index: %s", err)
		}
		defer ind.Destroy()
		ind2, err := NewIndex(conf)
		if err != nil {
			t.Fatalf("Failed to construct the index: %s", err)
		}
		defer ind2.Destroy()
		indView, err := NewIndex(conf)
		if err != nil {
			t.Fatalf("Failed to construct the index: %s", err)
		}
		defer indView.Destroy()

		err = ind.Reserve(100)
		if err != nil {
			t.Fatalf("Failed to reserve capacity: %s", err)
		}

		vec := make([]float32, dim)
		for i := uint(0); i < dim; i++ {
			vec[i] = float32(i) + 0.2
			err = ind.Add(uint64(i), vec)
			if err != nil {
				t.Fatalf("Failed to insert: %s", err)
			}
		}

		ind_length, err := ind.Len()
		if err != nil {
			t.Fatalf("Failed to retrieve size: %s", err)
		}

		// TODO: Add invalid save and loads?
		buffer_size := uint(1 * 1024 * 1024)
		buf := make([]byte, buffer_size)
		err = ind.SaveBuffer(buf, buffer_size)
		if err != nil {
			t.Fatalf("Failed to save the index to a buffer: %s", err)
		}

		err = ind2.LoadBuffer(buf, buffer_size)
		if err != nil {
			t.Fatalf("Failed to load the index from a buffer: %s", err)
		}

		ind2_length, err := ind2.Len()
		if err != nil {
			t.Fatalf("Failed to retrieve size: %s", err)
		}
		if ind_length != ind2_length {
			t.Fatalf("Loaded index length %d doesn't match original of %d ", ind2_length, ind_length)
		}
		// TODO: Check some values

		err = indView.ViewBuffer(buf, buffer_size)
		if err != nil {
			t.Fatalf("Failed to load the view from a buffer: %s", err)
		}

		indView_length, err := indView.Len()
		if err != nil {
			t.Fatalf("Failed to retrieve size: %s", err)
		}
		if ind_length != indView_length {
			t.Fatalf("Loaded view length %d doesn't match original of %d ", indView_length, ind_length)
		}

		conf, err = MetadataBuffer(buf, buffer_size)
		if err != nil {
			t.Fatalf("Failed to load the metadata from a buffer: %s", err)
		}
		if conf != ind.config {
			t.Fatalf("Loaded metadata doesn't match the index metadata")
		}

		// TODO: Check file save/load/metadata
	})
}
