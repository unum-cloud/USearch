# USearch Zig

High-performance Zig bindings for [USearch](https://github.com/unum-cloud/usearch) - a smaller, faster, and more scalable vector search library for approximate nearest neighbor (ANN) search.

## Installation

### Add as a dependency

Add USearch Zig to your project using the Zig package manager:

```bash
zig fetch --save git+https://github.com/pacifio/usearch-zig#main
```

### Configure your `build.zig`

Add the dependency to your executable or library:

```zig
const usearch_dep = b.dependency("usearch_zig", .{
    .target = target,
    .optimize = optimize,
});

exe.root_module.addImport("usearch_zig", usearch_dep.module("usearch_zig"));

exe.root_module.link_libc = true;
exe.root_module.link_libcpp = true;
```

## Quick Start

```zig
const std = @import("std");
const usearch = @import("usearch_zig");

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    // Create an index for 3-dimensional vectors
    const config = usearch.IndexConfig.default(3);
    var index = try usearch.Index.init(allocator, config);
    defer index.deinit();

    // Add some vectors
    var vec1 = [_]f32{ 1.0, 2.0, 3.0 };
    var vec2 = [_]f32{ 4.0, 5.0, 6.0 };

    try index.add(1, &vec1);
    try index.add(2, &vec2);

    // Search for similar vectors
    const results = try index.search(&vec1, 2);
    defer allocator.free(results);

    for (results) |result| {
        std.debug.print("Key: {}, Distance: {}\n", .{
            result.key,
            result.distance,
        });
    }
}
```

## Usage Examples

### Basic Index Operations

Create an index, add vectors, and perform similarity searches:

```zig
const std = @import("std");
const usearch = @import("usearch_zig");

pub fn basicExample() !void {
    const allocator = std.heap.page_allocator;

    // Create an index for 3-dimensional vectors
    const config = usearch.IndexConfig.default(3);
    var index = try usearch.Index.init(allocator, config);
    defer index.deinit();

    // Check initial size
    const size_initial = try index.len();
    std.debug.print("Initial size: {}\n", .{size_initial});

    // Add vectors with unique keys
    var vec1 = [_]f32{ 1.0, 2.0, 3.0 };
    var vec2 = [_]f32{ 4.0, 5.0, 6.0 };
    var vec3 = [_]f32{ 1.1, 2.1, 3.1 };

    try index.add(1, &vec1);
    try index.add(2, &vec2);
    try index.add(3, &vec3);

    const size_after = try index.len();
    std.debug.print("Size after adding: {}\n", .{size_after});

    // Search for the 2 nearest neighbors
    const results = try index.search(&vec1, 2);
    defer allocator.free(results);

    std.debug.print("Found {} results\n", .{results.len});
    for (results) |result| {
        std.debug.print("  Key: {}, Distance: {d:.6}\n", .{
            result.key,
            result.distance,
        });
    }
}
```

### Custom Distance Metrics

Configure the index to use different distance metrics:

```zig
pub fn customMetricExample() !void {
    const allocator = std.heap.page_allocator;

    // Create an index with L2 squared distance
    var config = usearch.IndexConfig.default(3);
    config.metric = .l2sq;  // Options: .cosine, .l2sq, .inner_product, etc.

    var index = try usearch.Index.init(allocator, config);
    defer index.deinit();

    var vec1 = [_]f32{ 1.0, 0.0, 0.0 };
    var vec2 = [_]f32{ 0.0, 1.0, 0.0 };

    try index.add(1, &vec1);
    try index.add(2, &vec2);

    const results = try index.search(&vec1, 1);
    defer allocator.free(results);

    std.debug.print("Nearest neighbor: key={}\n", .{results[0].key});
}
```

### Remove and Contains Operations

Manage vectors in the index dynamically:

```zig
pub fn removeAndContainsExample() !void {
    const allocator = std.heap.page_allocator;

    const config = usearch.IndexConfig.default(3);
    var index = try usearch.Index.init(allocator, config);
    defer index.deinit();

    var vec = [_]f32{ 1.0, 2.0, 3.0 };
    try index.add(42, &vec);

    // Check if vector exists
    const exists_before = try index.contains(42);
    std.debug.print("Vector 42 exists: {}\n", .{exists_before});

    // Remove the vector
    try index.remove(42);

    const exists_after = try index.contains(42);
    std.debug.print("Vector 42 exists after removal: {}\n", .{exists_after});
}
```

### Direct Distance Calculation

Calculate distances between vectors without building an index:

```zig
pub fn distanceExample() !void {
    var vec1 = [_]f32{ 1.0, 0.0, 0.0 };
    var vec2 = [_]f32{ 0.0, 1.0, 0.0 };

    const dist = try usearch.distance(&vec1, &vec2, 3, .l2sq);
    std.debug.print("L2 squared distance: {}\n", .{dist});
}
```

### Quantized Vectors (int8)

Use int8 quantization for memory-efficient storage:

```zig
pub fn int8VectorsExample() !void {
    const allocator = std.heap.page_allocator;

    var config = usearch.IndexConfig.default(4);
    config.quantization = .i8;  // Options: .f32, .f16, .bf16, .f64, .i8, .b1

    var index = try usearch.Index.init(allocator, config);
    defer index.deinit();

    var vec1 = [_]i8{ 1, 2, 3, 4 };
    var vec2 = [_]i8{ 5, 6, 7, 8 };

    try index.addI8(1, &vec1);
    try index.addI8(2, &vec2);

    const results = try index.searchI8(&vec1, 1);
    defer allocator.free(results);

    std.debug.print("Nearest neighbor: key={}\n", .{results[0].key});
}
```

### Reserve Capacity

Pre-allocate space for better performance:

```zig
pub fn reserveCapacityExample() !void {
    const allocator = std.heap.page_allocator;

    const config = usearch.IndexConfig.default(3);
    var index = try usearch.Index.init(allocator, config);
    defer index.deinit();

    // Reserve space for 5000 vectors upfront
    try index.reserve(5000);

    const cap = try index.capacity();
    std.debug.print("Reserved capacity: {}\n", .{cap});
}
```

### Query Index Properties

Inspect index configuration and statistics:

```zig
pub fn queryPropertiesExample() !void {
    const allocator = std.heap.page_allocator;

    const config = usearch.IndexConfig.default(128);
    var index = try usearch.Index.init(allocator, config);
    defer index.deinit();

    const dims = try index.dimensions();
    std.debug.print("Dimensions: {}\n", .{dims});

    const cap = try index.capacity();
    std.debug.print("Capacity: {}\n", .{cap});

    const mem = try index.memoryUsage();
    std.debug.print("Memory usage: {} bytes\n", .{mem});
}
```

### Working with Larger Datasets

Efficiently handle large collections of vectors:

```zig
pub fn largerDatasetExample() !void {
    const allocator = std.heap.page_allocator;

    var config = usearch.IndexConfig.default(8);
    config.initial_capacity = 100;

    var index = try usearch.Index.init(allocator, config);
    defer index.deinit();

    // Add 50 vectors
    var i: usize = 0;
    while (i < 50) : (i += 1) {
        var vec = [_]f32{
            @as(f32, @floatFromInt(i)),
            @as(f32, @floatFromInt(i + 1)),
            @as(f32, @floatFromInt(i + 2)),
            @as(f32, @floatFromInt(i + 3)),
            @as(f32, @floatFromInt(i + 4)),
            @as(f32, @floatFromInt(i + 5)),
            @as(f32, @floatFromInt(i + 6)),
            @as(f32, @floatFromInt(i + 7)),
        };
        try index.add(i, &vec);
    }

    const final_size = try index.len();
    std.debug.print("Final index size: {}\n", .{final_size});

    // Search for similar vectors
    var query = [_]f32{ 10, 11, 12, 13, 14, 15, 16, 17 };
    const results = try index.search(&query, 5);
    defer allocator.free(results);

    std.debug.print("Top 5 results:\n", .{});
    for (results) |result| {
        std.debug.print("  Key: {}, Distance: {d:.6}\n", .{
            result.key,
            result.distance,
        });
    }
}
```

### Persistence: Save and Load

Save indices to disk and load them later:

```zig
pub fn persistenceExample() !void {
    const allocator = std.heap.page_allocator;

    // Create and populate an index
    var config = usearch.IndexConfig.default(3);
    var index = try usearch.Index.init(allocator, config);
    defer index.deinit();

    var vec1 = [_]f32{ 1.0, 2.0, 3.0 };
    var vec2 = [_]f32{ 4.0, 5.0, 6.0 };

    try index.add(1, &vec1);
    try index.add(2, &vec2);

    // Save to disk
    try index.save("my_index.usearch");
    std.debug.print("Index saved\n", .{});

    // Load from disk
    var loaded_index = try usearch.Index.init(allocator, config);
    defer loaded_index.deinit();

    try loaded_index.load("my_index.usearch");

    const size = try loaded_index.len();
    std.debug.print("Loaded index with {} vectors\n", .{size});
}
```

### Advanced Configuration

Fine-tune performance parameters:

```zig
pub fn advancedConfigExample() !void {
    const allocator = std.heap.page_allocator;

    var config = usearch.IndexConfig{
        .dimensions = 128,
        .metric = .cosine,
        .quantization = .f32,
        .connectivity = 16,           // Higher = more accurate but slower
        .expansion_add = 128,         // Controls indexing quality
        .expansion_search = 64,       // Controls search quality
        .multi = false,               // Allow multiple vectors per key
        .initial_capacity = 10000,
    };

    var index = try usearch.Index.init(allocator, config);
    defer index.deinit();

    // Dynamically adjust parameters
    try index.setExpansionSearch(128);
    try index.setThreadsSearch(4);

    std.debug.print("Advanced index configured\n", .{});
}
```

## API Reference

### Types

#### `Key`

```zig
pub const Key = u64;
```

Unique identifier for vectors in the index.

#### `Metric`

```zig
pub const Metric = enum(u8) {
    inner_product,
    cosine,
    l2sq,
    haversine,
    divergence,
    pearson,
    hamming,
    tanimoto,
    sorensen,
};
```

Distance metrics for comparing vectors.

#### `Quantization`

```zig
pub const Quantization = enum(u8) {
    f32,
    bf16,
    f16,
    f64,
    i8,
    b1,
};
```

Scalar quantization types for vector storage.

#### `IndexConfig`

```zig
pub const IndexConfig = struct {
    quantization: Quantization = .f32,
    metric: Metric = .cosine,
    dimensions: usize,
    connectivity: usize = 0,
    expansion_add: usize = 0,
    expansion_search: usize = 0,
    multi: bool = false,
    initial_capacity: usize = 1000,
};
```

Configuration options for creating an index.

#### `SearchResult`

```zig
pub const SearchResult = struct {
    key: Key,
    distance: f32,
};
```

Result containing a key and its distance from the query.

### Index Methods

#### Core Operations

- **`init(allocator, config) !Index`** - Create a new index
- **`deinit()`** - Free index resources
- **`add(key, vector) !void`** - Add a float32 vector
- **`addI8(key, vector) !void`** - Add an int8 vector
- **`search(query, k) ![]SearchResult`** - Find k nearest neighbors (returns owned slice)
- **`searchI8(query, k) ![]SearchResult`** - Search using int8 query
- **`remove(key) !void`** - Remove a vector by key
- **`contains(key) !bool`** - Check if a key exists
- **`get(key, max_count) !?[]f32`** - Retrieve vector by key (returns owned slice or null)
- **`rename(from, to) !void`** - Rename a vector key

#### Queries

- **`len() !usize`** - Get number of vectors in index
- **`capacity() !usize`** - Get current capacity
- **`dimensions() !usize`** - Get vector dimensionality
- **`memoryUsage() !usize`** - Get memory usage in bytes
- **`reserve(capacity) !void`** - Reserve space for vectors

#### Persistence

- **`save(path) !void`** - Save index to file
- **`load(path) !void`** - Load index from file
- **`view(path) !void`** - Memory-map index from file (zero-copy)

#### Configuration

- **`setExpansionAdd(expansion) !void`** - Set expansion factor for adding
- **`setExpansionSearch(expansion) !void`** - Set expansion factor for search
- **`setThreadsAdd(threads) !void`** - Set number of threads for indexing
- **`setThreadsSearch(threads) !void`** - Set number of threads for search

### Utility Functions

- **`distance(vec1, vec2, dimensions, metric) !f32`** - Compute distance between float32 vectors
- **`distanceI8(vec1, vec2, dimensions, metric) !f32`** - Compute distance between int8 vectors
- **`loadMetadata(allocator, path) !IndexConfig`** - Load metadata from a saved index file

## Performance Tips

1. **Reserve capacity upfront** - Use `reserve()` or set `initial_capacity` to avoid reallocations
2. **Choose the right metric** - Cosine similarity is good for normalized vectors, L2 for geometric distances
3. **Tune connectivity** - Higher connectivity (16-32) improves accuracy but increases memory usage
4. **Use quantization** - int8 or f16 can significantly reduce memory usage with minimal accuracy loss
5. **Parallel operations** - Use `setThreadsAdd()` and `setThreadsSearch()` for multi-threaded performance
6. **Adjust expansion factors** - Higher values improve accuracy but slow down operations

## Building from Source

```bash
git clone https://github.com/pacifio/usearch-zig
cd usearch-zig
zig build test
```

## Requirements

- Zig 0.15.1 or later
- C++ compiler with C++17 support
