const std = @import("std");
const c = @cImport({
    @cInclude("usearch.h");
});

/// Unique identifier for vectors in the index
pub const Key = u64;

/// Distance metric for comparing vectors
pub const Metric = enum(u8) {
    inner_product = 0,
    cosine = 1,
    l2sq = 2,
    haversine = 3,
    divergence = 4,
    pearson = 5,
    hamming = 6,
    tanimoto = 7,
    sorensen = 8,

    pub fn toCValue(self: Metric) c.usearch_metric_kind_t {
        return switch (self) {
            .l2sq => c.usearch_metric_l2sq_k,
            .inner_product => c.usearch_metric_ip_k,
            .cosine => c.usearch_metric_cos_k,
            .haversine => c.usearch_metric_haversine_k,
            .divergence => c.usearch_metric_divergence_k,
            .pearson => c.usearch_metric_pearson_k,
            .hamming => c.usearch_metric_hamming_k,
            .tanimoto => c.usearch_metric_tanimoto_k,
            .sorensen => c.usearch_metric_sorensen_k,
        };
    }

    pub fn toString(self: Metric) []const u8 {
        return switch (self) {
            .l2sq => "l2sq",
            .inner_product => "ip",
            .cosine => "cos",
            .haversine => "haversine",
            .divergence => "divergence",
            .pearson => "pearson",
            .hamming => "hamming",
            .tanimoto => "tanimoto",
            .sorensen => "sorensen",
        };
    }
};

/// Scalar quantization type for vector storage
pub const Quantization = enum(u8) {
    f32 = 0,
    bf16 = 1,
    f16 = 2,
    f64 = 3,
    i8 = 4,
    b1 = 5,

    pub fn toCValue(self: Quantization) c.usearch_scalar_kind_t {
        return switch (self) {
            .f16 => c.usearch_scalar_f16_k,
            .f32 => c.usearch_scalar_f32_k,
            .f64 => c.usearch_scalar_f64_k,
            .i8 => c.usearch_scalar_i8_k,
            .b1 => c.usearch_scalar_b1_k,
            .bf16 => c.usearch_scalar_bf16_k,
        };
    }

    pub fn toString(self: Quantization) []const u8 {
        return switch (self) {
            .bf16 => "BF16",
            .f16 => "F16",
            .f32 => "F32",
            .f64 => "F64",
            .i8 => "I8",
            .b1 => "B1",
        };
    }
};

/// Configuration for creating a new USearch index
pub const IndexConfig = struct {
    quantization: Quantization = .f32,
    metric: Metric = .cosine,
    dimensions: usize,
    connectivity: usize = 0,
    expansion_add: usize = 0,
    expansion_search: usize = 0,
    multi: bool = false,
    initial_capacity: usize = 1000,

    /// Create default configuration for given dimensions
    pub fn default(dimensions: usize) IndexConfig {
        return .{
            .dimensions = dimensions,
            .metric = .cosine,
            .quantization = .f32,
        };
    }
};

/// Errors that can occur during USearch operations
pub const Error = error{
    IndexUninitialized,
    DimensionMismatch,
    EmptyVector,
    BufferTooSmall,
    InvalidPath,
    UsearchError,
    OutOfMemory,
};

/// Search result containing a key and its distance
pub const SearchResult = struct {
    key: Key,
    distance: f32,
};

/// Main USearch index for approximate nearest neighbor search
pub const Index = struct {
    handle: c.usearch_index_t,
    config: IndexConfig,
    allocator: std.mem.Allocator,

    /// Create a new USearch index with the given configuration
    pub fn init(allocator: std.mem.Allocator, config: IndexConfig) Error!Index {
        if (config.dimensions == 0) {
            return Error.DimensionMismatch;
        }

        var options = std.mem.zeroes(c.usearch_init_options_t);
        options.metric_kind = config.metric.toCValue();
        options.quantization = config.quantization.toCValue();
        options.dimensions = config.dimensions;
        options.connectivity = config.connectivity;
        options.expansion_add = config.expansion_add;
        options.expansion_search = config.expansion_search;
        options.multi = config.multi;

        var error_msg: c.usearch_error_t = null;
        const handle = c.usearch_init(&options, &error_msg);

        if (error_msg != null) {
            return Error.UsearchError;
        }

        if (handle == null) {
            return Error.UsearchError;
        }

        const index = Index{
            .handle = handle,
            .config = config,
            .allocator = allocator,
        };

        var reserve_err: c.usearch_error_t = null;
        c.usearch_reserve(index.handle, config.initial_capacity, &reserve_err);

        if (reserve_err != null) {
            c.usearch_free(index.handle, &reserve_err);
            return Error.UsearchError;
        }

        return index;
    }

    /// Free resources associated with the index
    pub fn deinit(self: *Index) void {
        if (self.handle) |handle| {
            var error_msg: c.usearch_error_t = null;
            c.usearch_free(handle, &error_msg);
            if (error_msg != null) {
                std.debug.print("Warning: error during usearch_free: {s}\n", .{error_msg});
            }
        }
        self.handle = null;
    }

    /// Get the number of vectors in the index
    pub fn len(self: *const Index) Error!usize {
        if (self.handle == null) return Error.IndexUninitialized;

        var error_msg: c.usearch_error_t = null;
        const size = c.usearch_size(self.handle, &error_msg);

        if (error_msg != null) return Error.UsearchError;
        return size;
    }

    /// Get the capacity of the index
    pub fn capacity(self: *const Index) Error!usize {
        if (self.handle == null) return Error.IndexUninitialized;

        var error_msg: c.usearch_error_t = null;
        const cap = c.usearch_capacity(self.handle, &error_msg);

        if (error_msg != null) return Error.UsearchError;
        return cap;
    }

    /// Get the dimensions of vectors in the index
    pub fn dimensions(self: *const Index) Error!usize {
        if (self.handle == null) return Error.IndexUninitialized;

        var error_msg: c.usearch_error_t = null;
        const dims = c.usearch_dimensions(self.handle, &error_msg);

        if (error_msg != null) return Error.UsearchError;
        return dims;
    }

    /// Get memory usage in bytes
    pub fn memoryUsage(self: *const Index) Error!usize {
        if (self.handle == null) return Error.IndexUninitialized;

        var error_msg: c.usearch_error_t = null;
        const usage = c.usearch_memory_usage(self.handle, &error_msg);

        if (error_msg != null) return Error.UsearchError;
        return usage;
    }

    /// Reserve capacity for a number of vectors
    pub fn reserve(self: *Index, cap: usize) Error!void {
        if (self.handle == null) return Error.IndexUninitialized;

        var error_msg: c.usearch_error_t = null;
        c.usearch_reserve(self.handle, cap, &error_msg);

        if (error_msg != null) return Error.UsearchError;
    }

    /// Add a float32 vector to the index
    pub fn add(self: *Index, key: Key, vector: []const f32) Error!void {
        if (self.handle == null) return Error.IndexUninitialized;
        if (vector.len == 0) return Error.EmptyVector;
        if (vector.len != self.config.dimensions) return Error.DimensionMismatch;

        const vec_ptr: ?*const anyopaque = @ptrCast(vector.ptr);

        var error_msg: c.usearch_error_t = null;
        c.usearch_add(
            self.handle,
            @as(c.usearch_key_t, key),
            vec_ptr,
            c.usearch_scalar_f32_k,
            @as([*c]c.usearch_error_t, @ptrCast(&error_msg)),
        );

        if (error_msg != null) return Error.UsearchError;
    }

    /// Add an int8 vector to the index
    pub fn addI8(self: *Index, key: Key, vector: []const i8) Error!void {
        if (self.handle == null) return Error.IndexUninitialized;
        if (vector.len == 0) return Error.EmptyVector;
        if (vector.len != self.config.dimensions) return Error.DimensionMismatch;

        var error_msg: c.usearch_error_t = null;
        c.usearch_add(
            self.handle,
            key,
            @as(?*const anyopaque, @ptrCast(vector.ptr)),
            c.usearch_scalar_i8_k,
            &error_msg,
        );

        if (error_msg != null) return Error.UsearchError;
    }

    /// Remove a vector by key
    pub fn remove(self: *Index, key: Key) Error!void {
        if (self.handle == null) return Error.IndexUninitialized;

        var error_msg: c.usearch_error_t = null;
        _ = c.usearch_remove(self.handle, key, &error_msg);

        if (error_msg != null) return Error.UsearchError;
    }

    /// Check if a key exists in the index
    pub fn contains(self: *const Index, key: Key) Error!bool {
        if (self.handle == null) return Error.IndexUninitialized;

        var error_msg: c.usearch_error_t = null;
        const found = c.usearch_contains(self.handle, key, &error_msg);

        if (error_msg != null) return Error.UsearchError;
        return found;
    }

    /// Get a vector by key. Returns owned slice, caller must free.
    pub fn get(self: *const Index, key: Key, max_count: usize) Error!?[]f32 {
        if (self.handle == null) return Error.IndexUninitialized;
        if (max_count == 0) return null;

        const buffer = try self.allocator.alloc(f32, self.config.dimensions * max_count);
        errdefer self.allocator.free(buffer);

        var error_msg: c.usearch_error_t = null;
        const found = c.usearch_get(
            self.handle,
            key,
            max_count,
            @as(?*anyopaque, @ptrCast(buffer.ptr)),
            c.usearch_scalar_f32_k,
            &error_msg,
        );

        if (error_msg != null) {
            self.allocator.free(buffer);
            return Error.UsearchError;
        }

        if (found == 0) {
            self.allocator.free(buffer);
            return null;
        }

        return buffer;
    }

    /// Rename a vector from one key to another
    pub fn rename(self: *Index, from: Key, to: Key) Error!void {
        if (self.handle == null) return Error.IndexUninitialized;

        var error_msg: c.usearch_error_t = null;
        _ = c.usearch_rename(self.handle, from, to, &error_msg);

        if (error_msg != null) return Error.UsearchError;
    }

    /// Search for nearest neighbors. Returns owned slice, caller must free.
    pub fn search(
        self: *const Index,
        query: []const f32,
        limit: usize,
    ) Error![]SearchResult {
        if (self.handle == null) return Error.IndexUninitialized;
        if (query.len == 0) return Error.EmptyVector;
        if (query.len != self.config.dimensions) return Error.DimensionMismatch;
        if (limit == 0) return &[_]SearchResult{};

        const keys = try self.allocator.alloc(Key, limit);
        defer self.allocator.free(keys);

        const distances = try self.allocator.alloc(f32, limit);
        defer self.allocator.free(distances);

        var error_msg: c.usearch_error_t = null;
        const result_count = c.usearch_search(
            self.handle,
            @as(?*const anyopaque, @ptrCast(query.ptr)),
            c.usearch_scalar_f32_k,
            limit,
            @as([*c]c.usearch_key_t, @ptrCast(keys.ptr)),
            @as([*c]c.usearch_distance_t, @ptrCast(distances.ptr)),
            &error_msg,
        );

        if (error_msg != null) return Error.UsearchError;

        const results = try self.allocator.alloc(SearchResult, result_count);
        for (0..result_count) |i| {
            results[i] = .{
                .key = keys[i],
                .distance = distances[i],
            };
        }

        return results;
    }

    /// Search using int8 query vector. Returns owned slice, caller must free.
    pub fn searchI8(
        self: *const Index,
        query: []const i8,
        limit: usize,
    ) Error![]SearchResult {
        if (self.handle == null) return Error.IndexUninitialized;
        if (query.len == 0) return Error.EmptyVector;
        if (query.len != self.config.dimensions) return Error.DimensionMismatch;
        if (limit == 0) return &[_]SearchResult{};

        const keys = try self.allocator.alloc(Key, limit);
        defer self.allocator.free(keys);

        const distances = try self.allocator.alloc(f32, limit);
        defer self.allocator.free(distances);

        var error_msg: c.usearch_error_t = null;
        const result_count = c.usearch_search(
            self.handle,
            @as(?*const anyopaque, @ptrCast(query.ptr)),
            c.usearch_scalar_i8_k,
            limit,
            @as([*c]c.usearch_key_t, @ptrCast(keys.ptr)),
            @as([*c]c.usearch_distance_t, @ptrCast(distances.ptr)),
            &error_msg,
        );

        if (error_msg != null) return Error.UsearchError;

        const results = try self.allocator.alloc(SearchResult, result_count);
        for (0..result_count) |i| {
            results[i] = .{
                .key = keys[i],
                .distance = distances[i],
            };
        }

        return results;
    }

    /// Save index to file
    pub fn save(self: *const Index, path: []const u8) Error!void {
        if (self.handle == null) return Error.IndexUninitialized;
        if (path.len == 0) return Error.InvalidPath;

        const c_path = try self.allocator.dupeZ(u8, path);
        defer self.allocator.free(c_path);

        var error_msg: c.usearch_error_t = null;
        c.usearch_save(self.handle, c_path.ptr, &error_msg);

        if (error_msg != null) return Error.UsearchError;
    }

    /// Load index from file
    pub fn load(self: *Index, path: []const u8) Error!void {
        if (self.handle == null) return Error.IndexUninitialized;
        if (path.len == 0) return Error.InvalidPath;

        const c_path = try self.allocator.dupeZ(u8, path);
        defer self.allocator.free(c_path);

        var error_msg: c.usearch_error_t = null;
        c.usearch_load(self.handle, c_path.ptr, &error_msg);

        if (error_msg != null) return Error.UsearchError;
    }

    /// View index from file without loading into memory
    pub fn view(self: *Index, path: []const u8) Error!void {
        if (self.handle == null) return Error.IndexUninitialized;
        if (path.len == 0) return Error.InvalidPath;

        const c_path = try self.allocator.dupeZ(u8, path);
        defer self.allocator.free(c_path);

        var error_msg: c.usearch_error_t = null;
        c.usearch_view(self.handle, c_path.ptr, &error_msg);

        if (error_msg != null) return Error.UsearchError;
    }

    /// Change the expansion factor for adding vectors
    pub fn setExpansionAdd(self: *Index, expansion: usize) Error!void {
        if (self.handle == null) return Error.IndexUninitialized;

        var error_msg: c.usearch_error_t = null;
        c.usearch_change_expansion_add(self.handle, expansion, &error_msg);

        if (error_msg != null) return Error.UsearchError;
    }

    /// Change the expansion factor for search
    pub fn setExpansionSearch(self: *Index, expansion: usize) Error!void {
        if (self.handle == null) return Error.IndexUninitialized;

        var error_msg: c.usearch_error_t = null;
        c.usearch_change_expansion_search(self.handle, expansion, &error_msg);

        if (error_msg != null) return Error.UsearchError;
    }

    /// Set number of threads for adding vectors
    pub fn setThreadsAdd(self: *Index, threads: usize) Error!void {
        if (self.handle == null) return Error.IndexUninitialized;

        var error_msg: c.usearch_error_t = null;
        c.usearch_change_threads_add(self.handle, threads, &error_msg);

        if (error_msg != null) return Error.UsearchError;
    }

    /// Set number of threads for search
    pub fn setThreadsSearch(self: *Index, threads: usize) Error!void {
        if (self.handle == null) return Error.IndexUninitialized;

        var error_msg: c.usearch_error_t = null;
        c.usearch_change_threads_search(self.handle, threads, &error_msg);

        if (error_msg != null) return Error.UsearchError;
    }
};

/// Compute distance between two float32 vectors
pub fn distance(
    vec1: []const f32,
    vec2: []const f32,
    dimensions: usize,
    metric: Metric,
) Error!f32 {
    if (vec1.len == 0 or vec2.len == 0) return Error.EmptyVector;
    if (vec1.len < dimensions or vec2.len < dimensions) return Error.DimensionMismatch;

    var error_msg: c.usearch_error_t = null;
    const dist = c.usearch_distance(
        @as(?*const anyopaque, @ptrCast(vec1.ptr)),
        @as(?*const anyopaque, @ptrCast(vec2.ptr)),
        c.usearch_scalar_f32_k,
        dimensions,
        metric.toCValue(),
        &error_msg,
    );

    if (error_msg != null) return Error.UsearchError;
    return dist;
}

/// Compute distance between two int8 vectors
pub fn distanceI8(
    vec1: []const i8,
    vec2: []const i8,
    dimensions: usize,
    metric: Metric,
) Error!f32 {
    if (vec1.len == 0 or vec2.len == 0) return Error.EmptyVector;
    if (vec1.len < dimensions or vec2.len < dimensions) return Error.DimensionMismatch;

    var error_msg: c.usearch_error_t = null;
    const dist = c.usearch_distance(
        @as(?*const anyopaque, @ptrCast(vec1.ptr)),
        @as(?*const anyopaque, @ptrCast(vec2.ptr)),
        c.usearch_scalar_i8_k,
        dimensions,
        metric.toCValue(),
        &error_msg,
    );

    if (error_msg != null) return Error.UsearchError;
    return dist;
}

/// Load metadata from a saved index file
pub fn loadMetadata(allocator: std.mem.Allocator, path: []const u8) Error!IndexConfig {
    if (path.len == 0) return Error.InvalidPath;

    const c_path = try allocator.dupeZ(u8, path);
    defer allocator.free(c_path);

    var options = std.mem.zeroes(c.usearch_init_options_t);
    var error_msg: c.usearch_error_t = null;

    c.usearch_metadata(c_path.ptr, &options, &error_msg);
    if (error_msg != null) return Error.UsearchError;

    var config = IndexConfig{
        .dimensions = options.dimensions,
        .connectivity = options.connectivity,
        .expansion_add = options.expansion_add,
        .expansion_search = options.expansion_search,
        .multi = options.multi,
        .metric = .cosine,
        .quantization = .f32,
    };

    config.metric = switch (options.metric_kind) {
        c.usearch_metric_l2sq_k => .l2sq,
        c.usearch_metric_ip_k => .inner_product,
        c.usearch_metric_cos_k => .cosine,
        c.usearch_metric_haversine_k => .haversine,
        c.usearch_metric_divergence_k => .divergence,
        c.usearch_metric_pearson_k => .pearson,
        c.usearch_metric_hamming_k => .hamming,
        c.usearch_metric_tanimoto_k => .tanimoto,
        c.usearch_metric_sorensen_k => .sorensen,
        else => .cosine,
    };

    config.quantization = switch (options.quantization) {
        c.usearch_scalar_f16_k => .f16,
        c.usearch_scalar_f32_k => .f32,
        c.usearch_scalar_f64_k => .f64,
        c.usearch_scalar_i8_k => .i8,
        c.usearch_scalar_b1_k => .b1,
        c.usearch_scalar_bf16_k => .bf16,
        else => .f32,
    };

    return config;
}

test "basic index operations" {
    const allocator = std.testing.allocator;

    const config = IndexConfig.default(3);
    var index = try Index.init(allocator, config);
    defer index.deinit();

    const size_initial = try index.len();
    try std.testing.expectEqual(@as(usize, 0), size_initial);

    var vec1 = [_]f32{ 1.0, 2.0, 3.0 };
    var vec2 = [_]f32{ 4.0, 5.0, 6.0 };
    var vec3 = [_]f32{ 1.1, 2.1, 3.1 };

    try index.add(1, &vec1);
    try index.add(2, &vec2);
    try index.add(3, &vec3);

    const size_after = try index.len();
    try std.testing.expectEqual(@as(usize, 3), size_after);

    const results = try index.search(&vec1, 2);
    defer allocator.free(results);

    try std.testing.expect(results.len > 0);
    try std.testing.expectEqual(@as(Key, 1), results[0].key);
    try std.testing.expect(results[0].distance < 0.01);

    if (results.len > 1) {
        try std.testing.expectEqual(@as(Key, 3), results[1].key);
    }
}

test "index with custom metric" {
    const allocator = std.testing.allocator;

    var config = IndexConfig.default(3);
    config.metric = .l2sq;

    var index = try Index.init(allocator, config);
    defer index.deinit();

    var vec1 = [_]f32{ 1.0, 0.0, 0.0 };
    var vec2 = [_]f32{ 0.0, 1.0, 0.0 };

    try index.add(1, &vec1);
    try index.add(2, &vec2);

    const results = try index.search(&vec1, 1);
    defer allocator.free(results);

    try std.testing.expectEqual(@as(Key, 1), results[0].key);
}

test "remove and contains" {
    const allocator = std.testing.allocator;

    const config = IndexConfig.default(3);
    var index = try Index.init(allocator, config);
    defer index.deinit();

    var vec = [_]f32{ 1.0, 2.0, 3.0 };
    try index.add(42, &vec);

    const exists_before = try index.contains(42);
    try std.testing.expect(exists_before);

    try index.remove(42);

    const exists_after = try index.contains(42);
    try std.testing.expect(!exists_after);
}

test "distance calculation" {
    var vec1 = [_]f32{ 1.0, 0.0, 0.0 };
    var vec2 = [_]f32{ 0.0, 1.0, 0.0 };

    const dist = try distance(&vec1, &vec2, 3, .l2sq);
    try std.testing.expectEqual(@as(f32, 2.0), dist);
}

test "int8 vectors" {
    const allocator = std.testing.allocator;

    var config = IndexConfig.default(4);
    config.quantization = .i8;
    var index = try Index.init(allocator, config);
    defer index.deinit();

    var vec1 = [_]i8{ 1, 2, 3, 4 };
    var vec2 = [_]i8{ 5, 6, 7, 8 };

    try index.addI8(1, &vec1);
    try index.addI8(2, &vec2);

    const results = try index.searchI8(&vec1, 1);
    defer allocator.free(results);

    try std.testing.expectEqual(@as(Key, 1), results[0].key);
}

test "reserve capacity" {
    const allocator = std.testing.allocator;

    const config = IndexConfig.default(3);
    var index = try Index.init(allocator, config);
    defer index.deinit();

    try index.reserve(5000);

    const cap = try index.capacity();
    try std.testing.expect(cap >= 5000);
}

test "dimensions and capacity queries" {
    const allocator = std.testing.allocator;

    const config = IndexConfig.default(128);
    var index = try Index.init(allocator, config);
    defer index.deinit();

    const dims = try index.dimensions();
    try std.testing.expectEqual(@as(usize, 128), dims);

    const cap = try index.capacity();
    try std.testing.expect(cap > 0);
}

test "index with larger dataset" {
    const allocator = std.testing.allocator;

    var config = IndexConfig.default(8);
    config.initial_capacity = 100;
    var index = try Index.init(allocator, config);
    defer index.deinit();

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
    try std.testing.expectEqual(@as(usize, 50), final_size);

    var query = [_]f32{ 10, 11, 12, 13, 14, 15, 16, 17 };
    const results = try index.search(&query, 5);
    defer allocator.free(results);

    try std.testing.expect(results.len > 0);
    try std.testing.expectEqual(@as(Key, 10), results[0].key);
}
