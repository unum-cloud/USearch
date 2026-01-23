# USearch for Swift

## Installation

USearch is available through the Swift Package Manager.
To install it, simply add the following line to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/unum-cloud/usearch", .upToNextMajor(from: "2.0.0"))
]
```

## Quickstart

Here’s a basic example:

```swift
let index = USearchIndex.make(metric: .Cos, dimensions: 3, connectivity: 8)
let vectorA: [Float32] = [0.3, 0.5, 1.2] // `Float32` and `Float64` are always supported
let vectorB: [Float32] = [0.4, 0.2, 1.2] // `Float16` support depends on the OS and hardware
index.add(key: 42, vector: vectorA) // Pass full arrays or slices
index.add(key: 43, vector: vectorB)

let results = index.search(vector: vectorA, count: 10)
assert(results.0[0] == 42)
let retrieved: [[Float32]]? = index.get(key: 42)
assert(retrieved![0] == vectorA)
```

If using in a SwiftUI application, make sure to ignore return values where appropriate:

```swift
import SwiftUI
import USearch

@main
struct USearchMobileApp: App {
    var body: some Scene {
        WindowGroup {
            let index = USearchIndex.make(metric: .IP, dimensions: 2, connectivity: 16, quantization: .F32)
            let _ = index.reserve(10)
            let coordinates: Array<Float32> = [40.177200, 44.503490]
            let _ = index.add(key: 10, vector: coordinates)            
            VStack {
                Text("USearch index contains \(index.count) vectors")
                Spacer()
            }

        }
    }
}
```

You can find a working example of using USearch with SwiftUI maps at [ashvardanian/SwiftSemanticSearch](https://github.com/ashvardanian/SwiftSemanticSearch).

[![](https://media.githubusercontent.com/media/ashvardanian/SwiftSemanticSearch/main/USearch%2BSwiftUI.gif)](https://github.com/ashvardanian/SwiftSemanticSearch)

## Serialization

Save and load your indices for efficient reuse:

```swift
try index.save("path/to/save/index")
try index.load("path/to/load/index")
try index.view("path/to/view/index")
```

## Updates and Metadata

Just like in Objective-C, you can manipulate the index and check its metadata:

```swift
// Retrieve structural properties of the index
let dimensions = index.dimensions
let connectivity = index.connectivity
let expansionAdd = index.expansionAdd
let expansionSearch = index.expansionSearch

// Check the number of vectors and if the index is empty
let numberOfVectors = index.count
let isEmpty = index.isEmpty

// Remove a vector by key
index.remove(key: 42)

// Rename a vector key
index.rename(from: 42, to: 52)

// Check if a specific key exists in the index
let exists = index.contains(key: 52)

// Clear all vectors from the index while preserving settings
index.clear()
```
