use std::error::Error;

fn main() {
    build_usearch().expect("Failed to build USearch");
}

fn build_usearch() -> Result<(), Box<dyn Error>> {
    let mut build = cxx_build::bridge("rust/lib.rs");

    build
        .file("rust/lib.cpp")
        .flag_if_supported("-Wno-unknown-pragmas")
        .warnings(false)
        .include("include")
        .include("rust");

    // Check for optional features
    if cfg!(feature = "openmp") {
        build.define("USEARCH_USE_OPENMP", "1");
    } else {
        build.define("USEARCH_USE_OPENMP", "0");
    }

    // When the `numkong` feature is enabled, the `numkong` crate compiles the SIMD
    // kernels itself (with dynamic dispatch and fallback across SIMD backends).
    // We only need its include path for the C++ headers.
    if cfg!(feature = "numkong") {
        let numkong_include = std::env::var("DEP_NUMKONG_INCLUDE")
            .map_err(|_| "numkong crate must set DEP_NUMKONG_INCLUDE via `links` metadata")?;
        build
            .include(&numkong_include)
            .define("USEARCH_USE_NUMKONG", "1")
            .define("NK_DYNAMIC_DISPATCH", "1")
            .define("NK_NATIVE_BF16", "0")
            .define("NK_NATIVE_F16", "0");

        // Link the NumKong static library compiled by the numkong crate. Cargo propagates
        // the library search path via `links` metadata, but doesn't re-emit `-lnumkong`
        // for downstream native code. Our C++ libusearch.a references NumKong symbols
        // (nk_find_kernel_punned, nk_capabilities), so we must link explicitly.
        println!("cargo:rustc-link-lib=static=numkong");
    } else {
        build.define("USEARCH_USE_NUMKONG", "0");
    }

    let target_os = std::env::var("CARGO_CFG_TARGET_OS")?;
    // Conditional compilation depending on the target operating system.
    if target_os == "linux" || target_os == "android" {
        build
            .flag_if_supported("-std=c++17")
            .flag_if_supported("-O3")
            .flag_if_supported("-ffast-math")
            .flag_if_supported("-fdiagnostics-color=always")
            .flag_if_supported("-g1"); // Simplify debugging
    } else if target_os == "macos" {
        build
            .flag_if_supported("-mmacosx-version-min=10.15")
            .flag_if_supported("-std=c++17")
            .flag_if_supported("-O3")
            .flag_if_supported("-ffast-math")
            .flag_if_supported("-fcolor-diagnostics")
            .flag_if_supported("-g1"); // Simplify debugging
    } else if target_os == "windows" {
        build
            .flag_if_supported("/std:c++17")
            .flag_if_supported("/O2")
            .flag_if_supported("/fp:fast")
            .flag_if_supported("/W1") // Reduce warnings verbosity
            .flag_if_supported("/EHsc")
            .flag_if_supported("/MD")
            .flag_if_supported("/permissive-")
            .flag_if_supported("/sdl-")
            .define("_ALLOW_RUNTIME_LIBRARY_MISMATCH", None)
            .define("_ALLOW_POINTER_TO_CONST_MISMATCH", None);
    }

    build.try_compile("usearch")?;

    println!("cargo:rerun-if-changed=rust/lib.rs");
    println!("cargo:rerun-if-changed=rust/lib.cpp");
    println!("cargo:rerun-if-changed=rust/lib.hpp");
    println!("cargo:rerun-if-changed=include/index_plugins.hpp");
    println!("cargo:rerun-if-changed=include/index_dense.hpp");
    println!("cargo:rerun-if-changed=include/usearch/index.hpp");
    Ok(())
}
