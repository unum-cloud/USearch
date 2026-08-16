// swift-tools-version:6.0

import PackageDescription

// The C surface is consumed as headers, not as sources: `src/dense_index.cpp` pulls the whole
// NumKong, ForkUnion, and StringZilla stack, has a CUDA sibling SwiftPM cannot drive, and is
// built by CMake into `libusearch_c` for every other binding already. So `USearchC` is the
// module map over `include/`, and the Swift target links the library CMake produced. Point the
// linker at it with `swift build -Xlinker -L<build directory>`, or install it system-wide.
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
    targets: [
        .target(
            name: "USearchC",
            path: ".",
            sources: ["swift/USearchC.c"],
            publicHeadersPath: "include",
            cSettings: [.headerSearchPath("include")]
        ),
        .target(
            name: "USearch",
            dependencies: ["USearchC"],
            path: "swift",
            exclude: ["README.md", "Test.swift", "CUSearch.c"],
            sources: ["Types.swift", "DenseIndex.swift"],
            linkerSettings: [.linkedLibrary("usearch_c")]
        ),
        .testTarget(
            name: "USearchTestsSwift",
            dependencies: ["USearch"],
            path: "swift",
            exclude: ["README.md", "CUSearch.c", "Types.swift", "DenseIndex.swift"],
            sources: ["Test.swift"]
        ),
    ]
)
