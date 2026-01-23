fn main() {
    let mut build = cxx_build::bridge("rust/lib.rs");

    build
        .file("rust/lib.cpp")
        .flag_if_supported("-Wno-unknown-pragmas")
        .warnings(false)
        .include("include")
        .include("rust")
        .include("fp16/include")
        .include("simsimd/include");

    // Check for optional features
    if cfg!(feature = "openmp") {
        build.define("USEARCH_USE_OPENMP", "1");
    } else {
        build.define("USEARCH_USE_OPENMP", "0");
    }

    if cfg!(feature = "fp16lib") {
        build.define("USEARCH_USE_FP16LIB", "1");
    } else {
        build.define("USEARCH_USE_FP16LIB", "0");
    }

    // Define all possible SIMD targets as 1
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();

    let mut flags_to_try;
    if cfg!(feature = "simsimd") {
        build
            .file("simsimd/c/lib.c")
            .define("USEARCH_USE_SIMSIMD", "1")
            .define("SIMSIMD_DYNAMIC_DISPATCH", "1")
            .define("SIMSIMD_NATIVE_BF16", "0")
            .define("SIMSIMD_NATIVE_F16", "0");
        flags_to_try = match target_arch.as_str() {
            "arm" | "aarch64" => vec![
                "SIMSIMD_TARGET_NEON",
                "SIMSIMD_TARGET_NEON_I8",
                "SIMSIMD_TARGET_NEON_F16",
                "SIMSIMD_TARGET_NEON_BF16",
                "SIMSIMD_TARGET_SVE",
                "SIMSIMD_TARGET_SVE_I8",
                "SIMSIMD_TARGET_SVE_F16",
                "SIMSIMD_TARGET_SVE_BF16",
            ],
            _ => vec![
                "SIMSIMD_TARGET_HASWELL",
                "SIMSIMD_TARGET_SKYLAKE",
                "SIMSIMD_TARGET_ICE",
                "SIMSIMD_TARGET_GENOA",
                "SIMSIMD_TARGET_SAPPHIRE",
            ],
        };
    } else {
        build.define("USEARCH_USE_SIMSIMD", "0");
        flags_to_try = vec![];
    }

    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap();
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

    let base_build = build.clone();

    let mut pop_flag = None;
    loop {
        let mut sub_build = base_build.clone();
        for flag in &flags_to_try {
            sub_build.define(flag, "1");
        }
        let result = sub_build.try_compile("usearch");
        if result.is_err() {
            if let Some(flag) = pop_flag {
                println!(
                    "cargo:warning=Failed to compile after disabling {:?}, trying next configuration...",
                    flag
                );
            } else if !flags_to_try.is_empty() {
                print!("cargo:warning=Failed to compile with all SIMD backends...");
            }

            pop_flag = flags_to_try.pop();
            if pop_flag.is_none() {
                result.unwrap();
            }
        } else {
            break;
        }
    }

    println!("cargo:rerun-if-changed=rust/lib.rs");
    println!("cargo:rerun-if-changed=rust/lib.cpp");
    println!("cargo:rerun-if-changed=rust/lib.hpp");
    println!("cargo:rerun-if-changed=include/index_plugins.hpp");
    println!("cargo:rerun-if-changed=include/index_dense.hpp");
    println!("cargo:rerun-if-changed=include/usearch/index.hpp");
}
