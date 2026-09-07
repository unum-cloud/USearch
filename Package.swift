// swift-tools-version:6.1

import PackageDescription

// The C surface is consumed as headers, not as sources: `src/dense_index.cpp` pulls the whole
// NumKong, ForkUnion, and StringZilla stack, has a CUDA sibling SwiftPM cannot drive, and is
// built by CMake into `libusearch_c` for every other binding already. So `USearchC` is the
// module map over `include/`, and the Swift target links the library CMake produced. Point the
// linker at it with `swift build -Xlinker -L<build directory>`, or install it system-wide.
//
// NumKong is a source dependency rather than a header one: its matrix views are the batch
// currency of every verb here, and its low-precision types are how a caller builds an `E4M3` row
// in the first place.
let package = Package(
    name: "USearch",
    platforms: [
        .macOS(.v12),
        .iOS(.v15),
        .watchOS(.v8),
        .tvOS(.v15),
    ],
    products: [
        .library(
            name: "USearch",
            targets: ["USearch"]
        )
    ],
    dependencies: [
        .package(path: "numkong")
    ],
    targets: [
        .target(
            name: "USearchC",
            path: ".",
            sources: ["swift/shim.c"],
            publicHeadersPath: "include",
            cSettings: [.headerSearchPath("include")]
        ),
        .target(
            name: "USearch",
            dependencies: ["USearchC", .product(name: "NumKong", package: "numkong")],
            path: "swift",
            exclude: ["README.md", "Test.swift", "shim.c"],
            sources: ["Types.swift", "DenseIndex.swift"],
            linkerSettings: [.linkedLibrary("usearch_c")]
        ),
        .testTarget(
            name: "USearchTestsSwift",
            dependencies: ["USearch", .product(name: "NumKong", package: "numkong")],
            path: "swift",
            exclude: ["README.md", "shim.c", "Types.swift", "DenseIndex.swift"],
            sources: ["Test.swift"]
        ),
    ]
)
