//! Builds the engine through CMake and links what it produced.
//!
//! No C or C++ is compiled here directly: `src/CMakeLists.txt` owns the instantiation grid, the
//! per-target compiler flags, and the CUDA sibling, and duplicating any of that in Rust would be
//! a second source of truth that drifts. So this drives CMake and consumes its output, which is
//! what makes `cargo build` work against a clean checkout rather than against whatever a previous
//! in-tree build happened to leave behind.
//!
//! One implementation of the C interface is linked, never two. Each names a CMake target and the
//! directory it lands in; all of them answer the same verbs, and the engine and the stand-in even
//! carry the same library name, so choosing between them is a search path rather than a different
//! link line.

use std::env;
use std::error::Error;
use std::path::{Path, PathBuf};

/// Every CMake option this crate sets, so an implementation is the subset it switches on and
/// nothing is left to whatever CMake would otherwise detect on the host.
const KNOWN_OPTIONS: &[&str] = &[
    "USEARCH_BUILD_LIB_C",
    "USEARCH_BUILD_LIB_C_STUB",
    "USEARCH_BUILD_CUDA",
    "USEARCH_BUILD_TEST",
    "USEARCH_BUILD_BENCH",
    "USEARCH_BUILD_LIB_WOLFRAM",
];

/// One implementation of the C interface, and what CMake must be told to produce it.
#[derive(Clone, Copy)]
struct Implementation {
    /// The feature that selects it, and the name reported when two are asked for at once.
    feature: &'static str,
    /// The options switched on for it; every other one this crate knows is switched off.
    options: &'static [&'static str],
    /// The target CMake is asked to build.
    target: &'static str,
    /// Where the generator leaves the archive, relative to the build tree; empty names its root.
    directory: &'static str,
    /// The archive's name, without the platform's prefix or extension.
    library: &'static str,
    /// The directories whose contents rebuild it, beside the interface every one of them reads.
    sources: &'static [&'static str],
}

/// The engine, linked as a self-contained archive - what a caller gets by naming nothing.
const ENGINE: Implementation = Implementation {
    feature: "static",
    options: &["USEARCH_BUILD_LIB_C"],
    target: "usearch_static",
    directory: "",
    library: "usearch_static",
    sources: &["src", "c"],
};

/// The engine with its CUDA backends compiled in.
const ENGINE_CUDA: Implementation = Implementation {
    feature: "cuda",
    options: &["USEARCH_BUILD_LIB_C", "USEARCH_BUILD_CUDA"],
    ..ENGINE
};

/// The brute-force stand-in, which answers the same verbs over the standard library alone.
const STUB: Implementation = Implementation {
    feature: "stub",
    options: &["USEARCH_BUILD_LIB_C_STUB"],
    target: "usearch_static_stub",
    directory: "stub",
    library: "usearch_static",
    sources: &["src_stub"],
};

/// Answers the one implementation the features select, refusing a pair rather than choosing.
///
/// Naming none is the engine, which is why `static` is not a feature: Cargo unions features across
/// the whole dependency graph and never subtracts, so a named default could only be turned off by
/// every consumer at once, and a `static` that no code reads would be a name that does nothing.
fn selected() -> Result<Implementation, String> {
    let named: Vec<Implementation> = [(cfg!(feature = "cuda"), ENGINE_CUDA), (cfg!(feature = "stub"), STUB)]
        .into_iter()
        .filter_map(|(enabled, implementation)| enabled.then_some(implementation))
        .collect();

    match named.len() {
        0 => Ok(ENGINE),
        1 => Ok(named[0]),
        _ => Err(format!(
            "the {} features are mutually exclusive, and both were enabled - each is a whole \
             implementation of the C interface, so only one can answer a call. Cargo unions \
             features across the dependency graph, so this usually means a second crate asked \
             for the other one.",
            named
                .iter()
                .map(|implementation| implementation.feature)
                .collect::<Vec<_>>()
                .join(" and ")
        )),
    }
}

/// Rebuilds when a directory's contents change, recursing because Cargo watches only what it is
/// told and a nested source would otherwise go unnoticed.
fn watch(directory: &Path) {
    println!("cargo:rerun-if-changed={}", directory.display());
    for entry in std::fs::read_dir(directory).into_iter().flatten().flatten() {
        let path = entry.path();
        match path.is_dir() {
            true => watch(&path),
            false => println!("cargo:rerun-if-changed={}", path.display()),
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
    let chosen = selected()?;

    let mut engine = cmake::Config::new(&manifest);
    // The crate reconfigures on every invocation by default, which on this tree is a fresh ISA
    // probe and submodule walk before a build that has nothing to do.
    // CMake's Debug configuration instruments for sanitizers whose runtimes rustc never links.
    engine
        .profile("RelWithDebInfo")
        .define("USEARCH_INSTALL", "OFF")
        .build_target(chosen.target)
        .always_configure(false);
    for option in KNOWN_OPTIONS {
        let wanted = chosen.options.contains(option);
        engine.define(option, if wanted { "ON" } else { "OFF" });
    }

    // `cmake-rs` has no WASI system name to hand CMake, so the checked-in toolchain names the SDK and the
    // sysroot; the one thing a user sets is `WASI_SDK_PATH`.
    let target_os = env::var("CARGO_CFG_TARGET_OS")?;
    if target_os == "wasi" {
        println!("cargo:rerun-if-env-changed=WASI_SDK_PATH");
        println!("cargo:rerun-if-env-changed=CARGO_CFG_TARGET_FEATURE");
        engine.define(
            "CMAKE_TOOLCHAIN_FILE",
            manifest.join("cmake/toolchain-wasm32-wasi.cmake"),
        );
        // An engine validates a module whole, so the SIMD tier is the one the Rust target declares; the flags
        // `cmake-rs` passes replace the toolchain's initial ones, so the tier and `-fno-exceptions` ride with them.
        let target_features = env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default();
        let declares = |feature: &str| target_features.split(',').any(|declared| declared == feature);
        if !declares("simd128") {
            return Err("USearch on WebAssembly needs `-C target-feature=+simd128`".into());
        }
        // Declared relaxed SIMD narrows to the strict class, since its wasm64 class needs threads no WASI target has.
        engine
            .cflag("-msimd128")
            .cxxflag("-msimd128")
            .cxxflag("-fno-exceptions");
    }

    // The architecture list is CUDA's alone, and says nothing unless its kernels are compiled.
    if chosen.options.contains(&"USEARCH_BUILD_CUDA") {
        println!("cargo:rerun-if-env-changed=USEARCH_CUDA_ARCHITECTURES");
        if let Ok(architectures) = env::var("USEARCH_CUDA_ARCHITECTURES") {
            engine.define("USEARCH_CUDA_ARCHITECTURES", architectures);
        }
    }

    // `build_target` stops short of installing, so the archive sits where the generator left it:
    // directly under the target's own directory, or one config deeper on the multi-config ones.
    let build = engine.build().join("build").join(chosen.directory);
    println!("cargo:rustc-link-search=native={}", build.display());
    println!("cargo:rustc-link-search=native={}", build.join("Release").display());
    println!("cargo:rustc-link-lib=static={}", chosen.library);

    // The archive is C++, so the standard library it was compiled against has to come along; the
    // `numkong` crate emits its own directives for the SIMD kernels.
    match target_os.as_str() {
        "windows" => {}
        "macos" | "ios" => println!("cargo:rustc-link-lib=dylib=c++"),
        // wasi-sdk ships libc++ and libc++abi as archives under its sysroot, in the `noeh` multilib the
        // toolchain file compiles against.
        "wasi" => {
            let sysroot = PathBuf::from(env::var("WASI_SDK_PATH")?).join("share/wasi-sysroot/lib/wasm32-wasip1/noeh");
            println!("cargo:rustc-link-search=native={}", sysroot.display());
            println!("cargo:rustc-link-lib=static=c++");
            println!("cargo:rustc-link-lib=static=c++abi");
        }
        _ => println!("cargo:rustc-link-lib=dylib=stdc++"),
    }

    // Only what the chosen implementation actually compiles, so a stand-in build ignores the
    // engine's sources and an engine build ignores the stand-in's.
    println!("cargo:rerun-if-changed={}", manifest.join("CMakeLists.txt").display());
    watch(&manifest.join("include"));
    for source in chosen.sources {
        watch(&manifest.join(source));
    }
    Ok(())
}
