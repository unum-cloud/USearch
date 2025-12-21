# USearch for Go

## Installation

### Linux

Download and install the Debian package from the latest release.
Substitute `<release_tag>`, `<arch>`, and `<usearch_version>` with your settings.

```sh
wget https://github.com/unum-cloud/usearch/releases/download/<release_tag>/usearch_linux_<arch>_<usearch_version>.deb
dpkg -i usearch_linux_<arch>_<usearch_version>.deb
```

### Windows

Run the `winlibinstaller.bat` script from the main repository in the folder where you will run `go run`.
This will install the USearch library and include it in the same folder where the script was run.

```sh
.\usearch\winlibinstaller.bat
```

### macOS

Download and unpack the zip archive from the latest release.
Move the USearch library and the include file to their respective folders.

```sh
wget https://github.com/unum-cloud/usearch/releases/download/<release_tag>/usearch_macos_<arch>_<usearch_version>.zip
unzip usearch_macos_<arch>_<usearch_version>.zip
sudo mv libusearch_c.dylib /usr/local/lib && sudo mv usearch.h /usr/local/include
```

## Quickstart

1. Create a `go.mod` file:

```
module usearch_example

go <go_version>
```

2. Create an `example.go`:

```go
package main

import (
	"fmt"
	usearch "github.com/unum-cloud/usearch/golang"
)

func main() {

   	// Create Index
   	vectorSize := 3
   	vectorsCount := 100
   	conf := usearch.DefaultConfig(uint(vectorSize))
   	index, err := usearch.NewIndex(conf)
   	if err != nil {
   		panic("Failed to create Index")
   	}
   	defer index.Destroy()

   	// Reserve capacity and configure internal threading
   	err = index.Reserve(uint(vectorsCount))
   	_ = index.ChangeThreadsAdd(uint(runtime.NumCPU()))
   	_ = index.ChangeThreadsSearch(uint(runtime.NumCPU()))
   	for i := 0; i < vectorsCount; i++ {
   		err = index.Add(usearch.Key(i), []float32{float32(i), float32(i + 1), float32(i + 2)})
      	if err != nil {
      		panic("Failed to add")
      	}
   	}

   	// Search
   	keys, distances, err := index.Search([]float32{0.0, 1.0, 2.0}, 3)
   	if err != nil {
    	panic("Failed to search")
   	}
   	fmt.Println(keys, distances)
}
```

Always call `Reserve(capacity)` before the first write.

3. Get USearch:

```sh
go get github.com/unum-cloud/usearch/golang
```

4. Run:

```sh
go run example.go
```

## Serialization

To save and load the index from disk, use the following methods:

```go
err := index.Save("index.usearch")
if err != nil {
    panic("Failed to save index")
}

err = index.Load("index.usearch")
if err != nil {
    panic("Failed to load index")
}

err = index.View("index.usearch")
if err != nil {
    panic("Failed to view index")
}
```

## Index Introspection

Inspect and interact with the index:

```go
dimensions, _ := index.Dimensions()     // Get the number of dimensions
size, _ := index.Len()                  // Get the number of vectors
capacity, _ := index.Capacity()         // Get the capacity
containsKey, _ := index.Contains(42)    // Check if a key is in the index
count, _ := index.Count(42)             // Get the count of vectors for a key (multi-vector indexes)
version := usearch.Version()            // Get the library version string
```

## Modifying the Index

```go
// Remove a vector by key
err := index.Remove(42)

// Clear all vectors while preserving the index structure
err = index.Clear()

// Rename a key
err = index.Rename(oldKey, newKey)
```

## Filtered Search

Perform searches with custom filtering:

```go
// Define a filter callback
handler := &usearch.FilteredSearchHandler{
    Callback: func(key usearch.Key, handler *usearch.FilteredSearchHandler) int {
        // Return non-zero to accept, zero to reject
        if key % 2 == 0 {
            return 1 // Accept even keys
        }
        return 0 // Reject odd keys
    },
    Data: nil, // Optional user data
}

// Perform filtered search
keys, distances, err := index.FilteredSearch(queryVector, 10, handler)
```

## Exact Search

For smaller datasets, perform brute-force exact search without building an index:

```go
dataset := []float32{...}  // Flattened vectors
queries := []float32{...}  // Flattened query vectors

keys, distances, err := usearch.ExactSearch(
    dataset, queries,
    datasetSize, queryCount,
    vectorDims*4, vectorDims*4,  // Strides in bytes
    vectorDims, usearch.Cosine,
    maxResults, 0,  // 0 threads = auto-detect
)
```

## Concurrency

USearch supports concurrent operations from multiple goroutines. Use `ChangeThreadsAdd` and `ChangeThreadsSearch` to configure the number of concurrent operations allowed:

```go
err := index.ChangeThreadsAdd(8)	// Allow up to 8 concurrent additions
err = index.ChangeThreadsSearch(16)	// Allow up to 16 concurrent searches
```

When using multiple goroutines, reserve at least as many threads as the number of concurrent callers:

```go
const numWorkers = 10

// Reserve threads for concurrent searches
_ = index.ChangeThreadsSearch(numWorkers)

var wg sync.WaitGroup
for i := 0; i < numWorkers; i++ {
    wg.Add(1)
    go func() {
        defer wg.Done()
        keys, distances, err := index.Search(queryVector, 10)
        // ...
    }()
}
wg.Wait()
```
