// swift-tools-version:6.0

import PackageDescription

let cxxSettings: [CXXSetting] = [
    .headerSearchPath("../include/"),
    .define("USEARCH_USE_NUMKONG", to: "1"),
    .define("NK_DYNAMIC_DISPATCH", to: "1"),
    .define("NK_NATIVE_F16", to: "0"),
    .define("NK_NATIVE_BF16", to: "0"),
]

var targets: [Target] = []

// Conditionally build the Objective-C target only on non-Linux platforms.
#if !os(Linux)
    targets.append(
        .target(
            name: "USearchObjectiveC",
            dependencies: [
                .product(name: "CNumKongDispatch", package: "NumKong"),
            ],
            path: "objc",
            sources: ["USearchObjective.mm"],
            cxxSettings: cxxSettings
        )
    )
#endif

// Always build the C and Swift targets.
targets += [
    .target(
        name: "USearchC",
        dependencies: [
            .product(name: "CNumKongDispatch", package: "NumKong"),
        ],
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
    platforms: [
        .macOS(.v12),
        .iOS(.v15),
        .watchOS(.v8),
        .tvOS(.v15),
    ],
    products: products,
    dependencies: [
        .package(url: "https://github.com/ashvardanian/NumKong", from: "7.5.0"),
    ],
    targets: targets,
    cxxLanguageStandard: CXXLanguageStandard.cxx11
)
