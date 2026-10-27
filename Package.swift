// swift-tools-version:6.2

import PackageDescription

// The C surface is consumed as a library, not as sources: `src/dense_index.cpp` pulls the whole
// NumKong, ForkUnion, and StringZilla stack, has a CUDA sibling SwiftPM cannot drive, and is
// built by CMake into `libusearch_static` for every other binding already. SwiftPM links what
// CMake produced: an XCFramework on Apple platforms and an artifact bundle elsewhere. A checkout
// points `USEARCH_SWIFT_ARTIFACT` at the one `cmake --build --preset swift` wrote, relative to
// this directory; everyone else downloads the release's, whose checksums the release workflow
// writes.
//
// NumKong is a source dependency rather than a header one: its matrix views are the batch
// currency of every verb here, and its low-precision types are how a caller builds an `E4M3` row
// in the first place.
let release = "https://github.com/unum-cloud/USearch/releases/download/v3.0.0-rc.1"
let appleChecksum = "0000000000000000000000000000000000000000000000000000000000000000"
let portableChecksum = "0000000000000000000000000000000000000000000000000000000000000000"

let library: [Target]
let libraryDependencies: [Target.Dependency]
if let artifact = Context.environment["USEARCH_SWIFT_ARTIFACT"] {
    library = [.binaryTarget(name: "USearchC", path: artifact)]
    libraryDependencies = ["USearchC"]
}
else {
    library = [
        .binaryTarget(name: "USearchCApple", url: "\(release)/USearchC.xcframework.zip", checksum: appleChecksum),
        .binaryTarget(
            name: "USearchCPortable",
            url: "\(release)/USearchC.artifactbundle.zip",
            checksum: portableChecksum
        ),
    ]
    libraryDependencies = [
        .target(
            name: "USearchCApple",
            condition: .when(platforms: [.macOS, .macCatalyst, .iOS, .tvOS, .watchOS, .visionOS])
        ),
        .target(name: "USearchCPortable", condition: .when(platforms: [.linux, .android, .windows, .wasi])),
    ]
}

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
    targets: library + [
        .target(
            name: "USearch",
            dependencies: libraryDependencies + [.product(name: "NumKong", package: "numkong")],
            path: "swift",
            exclude: ["README.md", "Test.swift"],
            sources: ["Types.swift", "DenseIndex.swift"],
            linkerSettings: [
                // The engine is C++, and Apple SDKs ship only libc++ - there is no libstdc++ to name.
                .linkedLibrary("c++", .when(platforms: [.macOS, .iOS, .tvOS, .watchOS, .visionOS])),
                .linkedLibrary("stdc++", .when(platforms: [.linux])),
            ]
        ),
        .testTarget(
            name: "USearchTestsSwift",
            dependencies: ["USearch", .product(name: "NumKong", package: "numkong")],
            path: "swift",
            exclude: ["README.md", "Types.swift", "DenseIndex.swift"],
            sources: ["Test.swift"]
        ),
    ]
)
