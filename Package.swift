// swift-tools-version:6.0

import PackageDescription

let cxxSettings: [CXXSetting] = [
    .headerSearchPath("../include/"),
    .headerSearchPath("../fp16/include/"),
    .headerSearchPath("../simsimd/include/"),
    .define("USEARCH_USE_FP16LIB", to: "1"),
    .define("USEARCH_USE_SIMSIMD", to: "1"),
]

var targets: [Target] = []

// Conditionally build the Objective-C target only on non-Linux platforms.
#if !os(Linux)
    targets.append(
        .target(
            name: "USearchObjectiveC",
            path: "objc",
            sources: ["USearchObjective.mm", "../simsimd/c/lib.c"],
            cxxSettings: cxxSettings
        )
    )
#endif

// Always build the C and Swift targets.
targets += [
    .target(
        name: "USearchC",
        path: "c",
        sources: ["usearch.h", "lib.cpp"],
        publicHeadersPath: ".",
        cxxSettings: cxxSettings
    ),
    .target(
        name: "USearch",
        dependencies: ["USearchC"],
        path: "swift",
        exclude: ["README.md", "Test.swift"],
        sources: ["USearchIndex.swift", "USearchIndex+Sugar.swift", "Util.swift"],
        cxxSettings: cxxSettings
    ),
    .testTarget(
        name: "USearchTestsSwift",
        dependencies: ["USearch"],
        path: "swift",
        sources: ["Test.swift"]
    ),
]

// Configure products similarly.
var products: [Product] = []

#if !os(Linux)
    products.append(
        .library(
            name: "USearchObjectiveC",
            targets: ["USearchObjectiveC"]
        )
    )
#endif

products.append(
    .library(
        name: "USearch",
        targets: ["USearch"]
    )
)

let package = Package(
    name: "USearch",
    products: products,
    dependencies: [],
    targets: targets,
    cxxLanguageStandard: CXXLanguageStandard.cxx11
)
