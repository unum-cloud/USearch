// swift-tools-version:6.4

import PackageDescription

// The C surface is consumed as headers, not as sources: `src/dense_index.cpp` pulls the whole
// NumKong, ForkUnion, and StringZilla stack, has a CUDA sibling SwiftPM cannot drive, and is
// built by CMake into `libusearch_static` for every other binding already. So `USearchC` is the
// module map over `c/`, and the Swift target links the library CMake produced. Point the
// linker at it with `swift build -Xlinker -L<build directory>`, or install it system-wide.
//
// NumKong is a source dependency rather than a header one: its matrix views are the batch
// currency of every verb here, and its low-precision types are how a caller builds an `E4M3` row
// in the first place.
let package = Package(
    name: "USearch",
    platforms: [
        .macOS(.v15),
        .iOS(.v18),
        .tvOS(.v18),
        .watchOS(.v11),
        .visionOS(.v2),
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
            // The NumKong and StringZilla checkouts sit under this path, and their Metal kernels would read
            // as resources.
            exclude: ["numkong", "stringzilla"],
            sources: ["swift/shim.c"],
            publicHeadersPath: "c",
            cSettings: [.headerSearchPath("c")]
        ),
        .target(
            name: "USearch",
            dependencies: ["USearchC", .product(name: "NumKong", package: "numkong")],
            path: "swift",
            exclude: ["README.md", "Test.swift", "shim.c"],
            sources: ["Types.swift", "DenseIndex.swift"],
            linkerSettings: [
                .linkedLibrary("usearch_static"),
                // The engine is C++, and Apple SDKs ship only libc++ - there is no libstdc++ to name.
                .linkedLibrary("c++", .when(platforms: [.macOS, .iOS, .tvOS, .watchOS, .visionOS])),
                .linkedLibrary("stdc++", .when(platforms: [.linux])),
            ]
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
