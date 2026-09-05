//! Builds the pre-instantiated C library the crate links.
//!
//! There is no C++ in the crate itself: `src/dense_index.cpp` is the one type-erasure boundary,
//! and the `src/` shards carry the instantiation grid, each in its own translation unit so the
//! LLVM backends run in parallel. Setting `USEARCH_LIB_DIR` skips the compile entirely and links
//! a library CMake already built, which is what CI wants.

use std::env;
use std::error::Error;

/// The type-erasure boundary plus the per-dtype shards, in the order `src/CMakeLists.txt` lists
/// them - add a shard here when one joins the library.
const SOURCES: &[&str] = &[
    "src/types.cpp",
    "src/dense_index.cpp",
    "src/dense/ram_index_bf16.cpp",
    "src/dense/ram_index_i8.cpp",
    "src/dense/ram_index_i4.cpp",
    "src/dense/ram_index_u8.cpp",
    "src/sequence.cpp",
];

fn main() -> Result<(), Box<dyn Error>> {
    println!("cargo:rerun-if-env-changed=USEARCH_LIB_DIR");
    if let Ok(directory) = env::var("USEARCH_LIB_DIR") {
        println!("cargo:rustc-link-search=native={directory}");
        println!("cargo:rustc-link-lib=static=usearch_indexes");
        println!("cargo:rustc-link-lib=static=numkong");
        return Ok(());
    }

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .warnings(false)
        .flag_if_supported("-Wno-unknown-pragmas");
    for source in SOURCES {
        build.file(source);
        println!("cargo:rerun-if-changed={source}");
    }

    // Sibling headers come from the submodules, the same three roots every other binding uses,
    // so a crate built from a non-recursive clone won't compile. The `numkong` crate stays a
    // dependency for its object code alone: it compiles the SIMD kernels with dynamic dispatch
    // and ISA fallback, and Cargo does not re-emit `-lnumkong` for downstream native code.
    build
        .include("include")
        .include("src")
        .include("numkong/include")
        .include("forkunion/include")
        .include("stringzilla/include")
        .define("USEARCH_USE_NUMKONG", "1")
        .define("USEARCH_USE_FORKUNION", "1")
        .define("USEARCH_USE_STRINGZILLA", "1")
        .define("USEARCH_BUILD_CUDA", "0")
        .define("NK_DYNAMIC_DISPATCH", "1")
        .define("NK_NATIVE_BF16", "0")
        .define("NK_NATIVE_F16", "0");
    println!("cargo:rustc-link-lib=static=numkong");

    match env::var("CARGO_CFG_TARGET_OS")?.as_str() {
        "windows" => {
            build
                .flag_if_supported("/std:c++20")
                .flag_if_supported("/O2")
                .flag_if_supported("/fp:fast")
                .flag_if_supported("/W1")
                .flag_if_supported("/EHsc")
                .flag_if_supported("/permissive-");
        }
        target => {
            if target == "macos" {
                build.flag_if_supported("-mmacosx-version-min=10.15");
            }
            build
                .flag_if_supported("-std=c++20")
                .flag_if_supported("-O3")
                .flag_if_supported("-ffast-math")
                .flag_if_supported("-g1"); // Simplify debugging
        }
    }

    build.try_compile("usearch_indexes")?;
    println!("cargo:rerun-if-changed=include");
    Ok(())
}
