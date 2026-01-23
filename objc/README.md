# USearch for Objective-C

This documentation covers the USearch Objective-C SDK, which provides functionality for building and managing vector search indices.
Below, you will find detailed information on installation, usage, and advanced features like serialization and custom metrics.

## Installation

USearch is available through the Swift Package Manager.
To install it, simply add the following line to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/unum-cloud/usearch", .upToNextMajor(from: "2.0.0"))
]
```

Alternatively, manually copy the following files to your project:

- `objc/include/USearchIndex.h`
- `objc/USearchIndex.m`
- `include/usearch/index.hpp`
- `include/usearch/index_dense.hpp`
- `include/usearch/index_plugins.hpp`

## Quickstart

Here's how to quickly set up a vector search index in Objective-C:

```objc
#import "USearchIndex.h"

// Creating an index with specific parameters
USearchIndex *index = [USearchIndex make:USearchMetricIP 
                              dimensions:3 
                            connectivity:10 
                            quantization:USearchScalarF32];

// Reserving space for vectors
[index reserve:10];


// Adding a double-precision vector (will be cast to float32)
double doubleVector[3] = {0.1, 0.2, 0.3};
[index addDouble:44 vector:doubleVector];

// Searching with an integer vector (will be cast to float32)
int intQueryVector[3] = {1, 2, 3};
UInt32 count = 5;
USearchKey keys[count];
float distances[count];
[index searchSingle:(Float32 const *)intQueryVector count:count keys:keys distances:distances];

// Adding a half-precision vector (requires casting to specified quantization type)
void *halfVector = ...; // Assume half-precision data
[index addHalf:45 vector:halfVector];
```

## Serialization

USearch supports saving and loading indices from disk, which allows for persistence across sessions.

```objc
// Saving an index to disk
[index save:@"path_to_save_index.usearch"];

// Loading an index from disk
[index load:@"path_to_load_index.usearch"];

// Viewing an index directly from disk (does not load into RAM)
[index view:@"path_to_view_index.usearch"];
```

## Metrics

USearch supports a variety of metrics for calculating distances or similarities between vectors, such as:

- `USearchMetricIP` - Inner Product
- `USearchMetricL2sq` - Squared Euclidean Distance
- `USearchMetricCos` - Cosine Similarity
- `USearchMetricPearson` - Pearson Correlation
- `USearchMetricHaversine` - Haversine (Great Circle) Distance
- `USearchMetricJaccard` - Jaccard Similarity

You can specify the metric when creating the index to ensure the most appropriate comparisons for your data.

## Advanced Index Management

Here are additional methods for managing data within the index:

```objc
// Checking if a key exists in the index
Boolean exists = [index contains:42];

// Counting the number of vectors associated with a key
UInt32 count = [index count:42];

// Removing a vector from the index
[index remove:42];

// Renaming a key within the index
[index rename:42 to:44];

// Clearing all vectors from the index
[index clear];
```

## Performance Considerations

Tune the `connectivity`, `quantization`, and other parameters according to the size and dimensionality of your dataset.
These settings can significantly affect both the performance and accuracy of your searches.
The choice of metric and scalar type (e.g., `USearchScalarF32`, `USearchScalarF16`, `USearchScalarI8`) also influences the memory usage and computation speed.
