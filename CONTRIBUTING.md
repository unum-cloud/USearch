# Contribution Guide

Thank you for coming here!
It's always nice to have third-party contributors 🤗

---

To keep the quality of the code high, we have a set of guidelines common to [all Unum projects](https://github.com/unum-cloud).

- [What's the procedure?](https://github.com/unum-cloud/awesome/blob/main/Workflow.md#organizing-software-development)
- [How to organize branches?](https://github.com/unum-cloud/awesome/blob/main/Workflow.md#branches)
- [How to style commits?](https://github.com/unum-cloud/awesome/blob/main/Workflow.md#commits)

## Before you start

Before building the first time, please pull `git` submodules.
That's how we bring in NumKong and other optional dependencies to test all of the available functionality.

```sh
git submodule update --init --recursive
```

The repository carries its own hooks, gating each staged file against the formatter its language uses and skipping the rest:

```sh
git config core.hooksPath .githooks
```

## C++ 11 and C 99

Our primary C++ implementation uses CMake for builds.
If this is your first experience with CMake, use the following commands to get started:

```sh
sudo apt-get update && sudo apt-get install cmake build-essential libjemalloc-dev # Ubuntu
brew install libomp llvm # macOS
```

Using modern syntax, this is how you build and run the test suite:

```sh
cmake -D USEARCH_BUILD_TEST=1 -D CMAKE_BUILD_TYPE=Debug -B build_debug
cmake --build build_debug --config Debug
build_debug/usearch_test
```

If the build mode is not specified, the default is `Release`.

```sh
cmake -D USEARCH_BUILD_TEST=1 -B build_release
cmake --build build_release --config Release
build_release/usearch_test
```

For development purposes, you may want to include symbols information in the build:

```sh
cmake -D USEARCH_BUILD_TEST=1 -D CMAKE_BUILD_TYPE=RelWithDebInfo -B build_relwithdebinfo
cmake --build build_relwithdebinfo --config RelWithDebInfo
build_relwithdebinfo/usearch_test
```

`usearch_test` and `usearch_test_c` open by naming the build and the machine it runs on:

```text
USearch 3.0.0
- Compiled for: serial,neon,neonsdot,neonbfdot,neonfhm,neonfp8
- This machine: serial,neon,neonsdot,neonbfdot,neonfhm
```

Both lists come from the C interface, the same strings the bindings report.
"Compiled for" is `usearch_capabilities_compiled()`, the NumKong capability levels this build carries kernels for, and "This machine" is `usearch_capabilities_available()`, those of them this processor can also run.
`usearch_test_cuda` opens with `- CUDA: <device> sm_<major><minor>` for the first visible device.
Without one it prints `- CUDA: no device` and exits 0, skipping every GPU test.

Those three configurations, and the benchmarking one from [bench/README.md](bench/README.md), are also packaged as CMake presets, so the flag soup below is optional:

```sh
cmake --list-presets              # the ones this host can build, from `debug` to `wasm32_wasi` and `msvc_x64`
cmake --preset release            # configures into `build_release`
cmake --build --preset release    # builds it
ctest --preset release            # runs the suites
```

Every preset writes into `build_<preset name>` and exports a `compile_commands.json` for the language server.
The development ones build with LLVM through Ninja and link with `lld`; `tsan` uses GCC, and CI rows override the compiler with `-D CMAKE_CXX_COMPILER`.
The rest match CI: `release_lib` for the C library alone, `linux_aarch64`, `android_arm64`, `android_armv7`, the three WASM toolchains, and `msvc_x64` / `msvc_arm64`.
Two of them are about GPUs: `dev` compiles the CUDA suites for the host GPU alone, which is the fastest way to iterate and the tree the repo's `.clangd` reads, while `release_cuda` emits a fat binary spanning every architecture from the Ampere floor through both Blackwell lines.
`trace` adds `-ftime-trace`, dropping a profile beside every object for when a translation unit starts dominating the build.
Machine-specific overrides - a different compiler, `ccache`, a CUDA toolkit outside the default prefix - belong in an untracked `CMakeUserPresets.json` beside it:

```json
{
    "version": 6,
    "configurePresets": [
        {
            "name": "dev-local",
            "inherits": "dev",
            "binaryDir": "${sourceDir}/build_dev",
            "cacheVariables": {
                "CMAKE_CUDA_HOST_COMPILER": "/usr/bin/g++-14",
                "CMAKE_CXX_COMPILER_LAUNCHER": "ccache"
            }
        }
    ]
}
```

The CUDA presets need CMake 3.31 or newer - the release that learned the Blackwell architectures - while the CPU-only build stays on the 3.28 floor.
Ubuntu's own package predates the former, so [Kitware's repository](https://apt.kitware.com) is the way to get it.

The CMakeLists.txt file has a number of options you can pass:

- What to build:
  - `USEARCH_BUILD_TEST` - build the C++ test suite, and the C one alongside `USEARCH_BUILD_LIB_C`
  - `USEARCH_BUILD_BENCH` - build the C++ benchmark suite
  - `USEARCH_BUILD_LIB_C` - build the C library
  - `USEARCH_BUILD_LIB_C_STUB` - a brute-force stand-in answering the same verbs over the standard library alone, for isolating a binding from the engine while bisecting one.
    It carries the same library names in a `stub/` subdirectory, so pointing a binding at it is a search path rather than a different link line.
  - `USEARCH_BUILD_CUDA` - build the CUDA test & benchmark suites
- Which GPUs to target:
  - `USEARCH_CUDA_ARCHITECTURES` - the architectures the CUDA tests, benchmarks and the C interface's dispatch unit compile for, `80;90a;100f;120f` by default: the Ampere floor, then Hopper, architecture-specific, and both Blackwell lines, family-specific so they carry the fp6 and fp4 conversions.
    The library itself ignores the list and compiles one unit per GPU class, each for its own code alone - `80` with its PTX for every GPU outside the other classes, `89`, `90a`, `100f`, which B300 runs as well, and `120f` - and picks the unit by the device's generation at runtime.
    A NumKong GPU tier turns on in those targets only when every listed architecture runs it, so the default list exercises the Ampere kits alone; iterate on one GPU with its family code, like `100f`, to test that generation's kits.
- Which other dependencies to use:
  - `USEARCH_USE_JEMALLOC` - use Jemalloc for memory management

The sibling libraries carry no switches.
NumKong, ForkUnion, StringZilla and SmashTable are what the headers are written against, so every build acquires all four - from the submodule when a recursive clone provides one, and from the pinned commit otherwise.
The language bindings take their headers from `numkong/include`, `forkunion/include`, and `stringzilla/include` in the checkout, so `pyproject.toml`, `package.json`, `build.rs`, and `Package.swift` each reach the same roots, and none of them builds from a non-recursive clone.

Putting all of this together, compiling all targets on most platforms should work with the following snippet:

```sh
cmake -D CMAKE_BUILD_TYPE=Release -D USEARCH_USE_JEMALLOC=1 -D USEARCH_BUILD_TEST=1 -D USEARCH_BUILD_BENCH=1 -D USEARCH_BUILD_LIB_C=1 -B build_release

cmake --build build_release --config Release
build_release/usearch_test
build_release/usearch_test_c
```

Similarly, to use the most recent Clang compiler version from Homebrew on macOS:

```sh
brew install clang++ clang cmake
cmake \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_C_COMPILER="$(brew --prefix llvm)/bin/clang" \
    -D CMAKE_CXX_COMPILER="$(brew --prefix llvm)/bin/clang++" \
    -D USEARCH_USE_JEMALLOC=1 \
    -D USEARCH_BUILD_TEST=1 \
    -D USEARCH_BUILD_BENCH=1 \
    -D USEARCH_BUILD_LIB_C=1 \
    -B build_release

cmake --build build_release --config Release
build_release/usearch_test
build_release/usearch_test_c
```

Linting:

```sh
cppcheck --enable=all --force --suppress=cstyleCast --suppress=unusedFunction \
    include/usearch/utilities/types.hpp \
    include/usearch/basic/ram_index.hpp \
    include/usearch/dense/ram_index.hpp \
    include/usearch/dense/ram_metric.hpp
```

I'd recommend putting the following breakpoints when debugging the code in GDB:

- `__asan::ReportGenericError` - to detect illegal memory accesses.
- `__ubsan::ScopedReport::~ScopedReport` - to catch undefined behavior.
- `__GI_exit` - to stop at exit points - the end of running any executable.
- `__builtin_unreachable` - to catch all the places where the code is expected to be unreachable.
- `usearch_raise_runtime_error` - for USearch-specific assertions.

### Cross Compilation

Unlike GCC, LLVM handles cross compilation very easily.
You just need to pass the right `TARGET_ARCH` and `BUILD_ARCH` to CMake.
The [list includes](https://packages.ubuntu.com/search?keywords=crossbuild-essential&searchon=names):

- `crossbuild-essential-amd64` for 64-bit x86
- `crossbuild-essential-arm64` for 64-bit Arm
- `crossbuild-essential-armhf` for 32-bit ARM hard-float
- `crossbuild-essential-armel` for 32-bit ARM soft-float (emulates `float`)
- `crossbuild-essential-riscv64` for RISC-V
- `crossbuild-essential-powerpc` for PowerPC
- `crossbuild-essential-s390x` for IBM Z
- `crossbuild-essential-mips` for MIPS
- `crossbuild-essential-ppc64el` for PowerPC 64-bit little-endian

Here is an example for cross-compiling for Arm64 on an x86_64 machine:

```sh
sudo apt-get update
sudo apt-get install -y clang lld make crossbuild-essential-arm64 crossbuild-essential-armhf
export CC="clang"
export CXX="clang++"
export AR="llvm-ar"
export NM="llvm-nm"
export RANLIB="llvm-ranlib"
export TARGET_ARCH="aarch64-linux-gnu" # Or "x86_64-linux-gnu"
export BUILD_ARCH="arm64" # Or "amd64"

cmake -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_C_COMPILER_TARGET=${TARGET_ARCH} \
    -D CMAKE_CXX_COMPILER_TARGET=${TARGET_ARCH} \
    -D CMAKE_SYSTEM_NAME=Linux \
    -D CMAKE_SYSTEM_PROCESSOR=${BUILD_ARCH} \
    -B build_artifacts
cmake --build build_artifacts --config Release
```

For Android development, you can cross-compile for ARM architectures without requiring the full Android NDK setup.
Here's an example targeting 32-bit ARM (`armeabi-v7a`):

```sh
sudo apt-get update
sudo apt-get install -y clang lld crossbuild-essential-armhf

# Cross-compile for 32-bit ARM (Android compatible)
CMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY \
cmake -B build_artifacts \
  -D CMAKE_C_COMPILER=clang \
  -D CMAKE_CXX_COMPILER=clang++ \
  -D CMAKE_C_COMPILER_TARGET=armv7-linux-gnueabihf \
  -D CMAKE_CXX_COMPILER_TARGET=armv7-linux-gnueabihf \
  -D CMAKE_SYSTEM_NAME=Linux \
  -D CMAKE_SYSTEM_PROCESSOR=armv7 \
  -D CMAKE_C_FLAGS="--target=armv7-linux-gnueabihf -march=armv7-a" \
  -D CMAKE_CXX_FLAGS="--target=armv7-linux-gnueabihf -march=armv7-a" \
  -D CMAKE_BUILD_TYPE=RelWithDebInfo \
  -D USEARCH_BUILD_LIB_C=1 \
  -D USEARCH_BUILD_TEST=0 \
  -D USEARCH_BUILD_BENCH=0

cmake --build build_artifacts --config RelWithDebInfo
file build_artifacts/libusearch_shared.so # Verify the output
# Should show: ELF 32-bit LSB shared object, ARM, EABI5 version 1 (SYSV), dynamically linked, ...
```

The resulting `libusearch_shared.so` can be used in Android projects by placing it in `src/main/jniLibs/armeabi-v7a/` for 32-bit ARM or `arm64-v8a/` for 64-bit ARM.

## Python 3

Python bindings are a CPython extension available on [PyPI](https://pypi.org/project/usearch/).
`scikit-build-core` drives the project's own CMake from `pyproject.toml`, so the extension is a CMake target rather than a second native build described in a `setup.py`.
To install USearch locally using `uv`:

```sh
uv venv --python 3.11                   # or your preferred Python version
source .venv/bin/activate               # to activate the virtual environment
uv pip install -e . --force-reinstall   # to build locally from source
```

Or using `pip` directly:

```sh
pip install -e . --force-reinstall
```

For testing USearch uses PyTest, which is pre-configured in `pyproject.toml`.
Following options are enabled:

- The `-s` option will disable capturing the logs.
- The `-x` option will exit after first failure to simplify debugging.
- The `-p no:warnings` option will suppress and allow warnings.

```sh
uv pip install -e . --group tests             # for repeated fuzzy tests
python -m pytest                                      # if you trust the default settings
python -m pytest python/scripts/ -s -x -p no:warnings # to overwrite the default settings
```

The `seed` fixture hands a test 42, unless `USEARCH_SEED` names another seed or is `random`, which draws a fresh one per run.
pytest prints the seed in its header as `seed: <n>, pin with USEARCH_SEED`, so a failure under a drawn seed reruns with that number.
Under pytest-repeat, the fixture adds the repeat step to the seed, so repetitions differ and each one still reproduces.
A value that is neither an integer nor `random` stops the run with `USEARCH_SEED="<value>" does not parse`.

```sh
USEARCH_SEED=random python -m pytest test/               # draws a seed and prints it in the header
USEARCH_SEED=1528003704 python -m pytest test/ --count 3 # replays that seed, then the next two
```

Linting:

```sh
uv pip install -e . --group lint
ruff --format=github python
```

Before merging your changes you may want to test your changes against the entire matrix of Python versions USearch supports.
For that you need the `cibuildwheel`, which is tricky to use on macOS and Windows, as it would target just the local environment.
Still, if you have Docker running on any desktop OS, you can use it to build and test the Python bindings for all Python versions for Linux:

```sh
pip install cibuildwheel
cibuildwheel
cibuildwheel --platform linux                   # works on any OS and builds all Linux backends
cibuildwheel --platform linux --archs x86_64    # 64-bit x86, the most common on desktop and servers
cibuildwheel --platform linux --archs aarch64   # 64-bit Arm for mobile devices, Apple M-series, and AWS Graviton
cibuildwheel --platform macos                   # works only on macOS
cibuildwheel --platform windows                 # works only on Windows
```

You may need root privileges for multi-architecture builds:

```sh
sudo $(which cibuildwheel) --platform linux
```

On Windows and macOS, to avoid frequent path resolution issues, you may want to use:

```sh
python -m cibuildwheel --platform windows
```

## JavaScript

USearch provides N-API bindings for NodeJS available on [NPM](https://www.npmjs.com/package/usearch).
`cmake-js` drives the project's own CMake from `package.json`, so the addon is a CMake target rather than a second native build described in a `binding.gyp`.
If you don't have NPM installed, first the Node Version Manager:

```sh
wget -qO- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
nvm install 20
```

Testing:

```sh
npm install -g typescript   # Install TypeScript globally
npm install                 # Compile `javascript/lib.cpp`
npm run build-js            # Generate JS from TS
npm test                    # Run the test suite
```

To compile for AWS Lambda you'd need to recompile the binding.
You can test the setup locally, overriding some of the compilation variables in Docker image:

```Dockerfile
FROM public.ecr.aws/lambda/nodejs:18-x86_64
RUN npm init -y
RUN yum install tar git python3 cmake gcc-c++ -y && yum groupinstall "Development Tools" -y

# Assuming AWS Linux 2 uses old compilers:
ENV NUMKONG_TARGET_HASWELL 1
ENV NUMKONG_TARGET_SKYLAKE 0
ENV NUMKONG_TARGET_ICELAKE 0
ENV NUMKONG_TARGET_SAPPHIRE 0
ENV NUMKONG_TARGET_NEON 1
ENV NUMKONG_TARGET_SVE 0

# For specific PR:
# RUN npm install --build-from-source unum-cloud/USearch#pull/302/head
# For specific version:
# RUN npm install --build-from-source usearch@2.8.8
RUN npm install --build-from-source usearch
```

To compile to WebAssembly make sure you have `emscripten` installed and pick one of the checked-in toolchains.
There is no WebAssembly target of its own — the C library is what crosses, and a module carries one SIMD tier, width and threads choice, so each toolchain file fixes all three:

- `cmake/toolchain-wasm32-emscripten.cmake` builds the `wasm32_v128_k` class — 32-bit, single-threaded, the fixed-width `simd128` tier every engine loads.
- `cmake/toolchain-wasm64-emscripten.cmake` builds the `wasm64_extended_k` class — relaxed SIMD, shared memory for real thread pools, and 64-bit addressing.
- `cmake/toolchain-wasm32-wasi.cmake` builds `wasm32_v128_k` against the WASI SDK, for Wasmtime, Wasmer and the Rust crate.

```sh
emcmake cmake -B build_wasm32_emscripten -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-wasm32-emscripten.cmake -DUSEARCH_BUILD_LIB_C=1 -DUSEARCH_BUILD_TEST=1
cmake --build build_wasm32_emscripten --target usearch_static usearch_test_c
ctest --test-dir build_wasm32_emscripten -R '^usearch_test_c$'
```

The toolchain file must be passed as the single argument `-DCMAKE_TOOLCHAIN_FILE=…`; written with a space, `emcmake` appends its own file, which then wins.
If you don't yet have `emcmake` installed:

```sh
git clone https://github.com/emscripten-core/emsdk.git && ./emsdk/emsdk install latest && ./emsdk/emsdk activate latest && source ./emsdk/emsdk_env.sh
```

## Rust

USearch provides Rust bindings available on [Crates.io](https://crates.io/crates/usearch).
`build.rs` compiles no C or C++ of its own — it drives the project's CMake and links what it produced.

Naming no feature links the engine, and `cuda` links it with the CUDA backends compiled in.
Each feature is a whole implementation of the C interface rather than a knob on one, so `build.rs` refuses a pair — never `--all-features`.

```sh
cargo test -- --nocapture --test-threads=1
cargo clippy --all-targets
cargo fmt --check
```

Publishing the crate is a bit more complicated than normally.
If you simply pull the repository with submodules and run the following command it will list fewer files than expected:

```sh
cargo package --list --allow-dirty
```

The reason for that is the heuristic that Cargo uses to determine the files to include in the package.

> Regardless of whether exclude or include is specified, the following files are always excluded:
> Any sub-packages will be skipped (any subdirectory that contains a Cargo.toml file).

Assuming both NumKong and StringZilla contain their own `Cargo.toml` files, we need to temporarily exclude them from the package.

```sh
mv numkong/Cargo.toml numkong/Cargo.toml.bak
mv stringzilla/Cargo.toml stringzilla/Cargo.toml.bak
cargo package --list --allow-dirty
cargo publish

# Revert back
mv numkong/Cargo.toml.bak numkong/Cargo.toml
mv stringzilla/Cargo.toml.bak stringzilla/Cargo.toml
```

## Zig

USearch provides Zig bindings under `zig/`, driven by `build.zig` and needing Zig 0.16 or newer.
No C or C++ is compiled there: the engine is CMake's to build, and `-Dlib-dir` names a directory holding it, repeatable.
It takes the shared library rather than the archive, since Zig answers `linkSystemLibrary("stdc++")` with its own libc++, which leaves a GCC-built archive short of symbols.

```sh
cmake -B build_release -D USEARCH_BUILD_LIB_C=1
cmake --build build_release --target usearch_shared

zig build                                     # compile check, resolving no symbols
zig build test -Dlib-dir=build_release        # links, so every `extern fn` must resolve
zig fmt --check zig build.zig
```

`build.zig.zon` pins its version against the tracked `VERSION` file at comptime, so bumping one and forgetting the other fails the build.

## Mojo

The Mojo binding lives under `mojo/` and `dlopen`s the engine rather than linking it, so a consumer needs no linker flags of its own.
Pixi drives both halves: the root `pixi.toml` declares the `usearch` package the tree's own CMake builds, and `mojo/pixi.toml` the Mojo package depending on it.
The engine lands in the environment prefix, which is the runpath `mojo build` bakes.

```sh
pixi run test
pixi run build
pixi run mojo format -l 120 mojo/test.mojo
```

Mojo 1.0 has no test discovery, so every check is registered in `main` by hand and one that is not silently never runs.

## Swift

USearch provides Swift bindings through the [Swift Package Manager](https://swiftpackageindex.com/unum-cloud/USearch).
`Package.swift` controls the compilation settings and is independent of the CMake used for C and C++ builds.
Swift 6.3 or newer is required: the streaming and checkpoint verbs take `MutableSpan` and `RawSpan`, and `Key` stores its bytes in an `InlineArray`.

The package links a library CMake built, so the engine is compiled first and the linker is pointed at it:

```sh
cmake -B build_release -D USEARCH_BUILD_LIB_C=1
cmake --build build_release --target usearch_static

swift build -Xlinker -Lbuild_release
swift test -Xlinker -Lbuild_release
```

The suite is Swift Testing: tests are `@Test` functions grouped into `@Suite` types, and assertions read as `#expect(...)`.
It covers what the binding itself decides — key byte order, options marshalling through the residency union, matrix lowering, handle lifetime and the span borrows — since that vector search returns near neighbors is the engine's contract and is tested where the engine lives.

Formatting is enforced with Apple's `swift-format`, whose style lives in the `.swift-format` file at the repository root:

```sh
swift format --in-place --recursive swift Package.swift
swift format lint --strict --recursive swift Package.swift
```

Neither the build nor the suite needs an Apple device — both run on Linux, where [`swift.org/install`](https://www.swift.org/install) carries the toolchains.
A toolchain built for an older Ubuntu than the host may want that release's `libxml2` and `libicu` alongside it, since it links their older sonames.

## Go

USearch provides Go bindings, that depend on the C library that must be installed beforehand.
So one should first compile the C library, link it with Go, and only then run tests.

```sh
cmake -B build_release -D USEARCH_BUILD_LIB_C=1
cmake --build build_release --config Release -j

cp build_release/libusearch_static.a golang/         # the archive the cgo directives link
cp build_release/numkong/libnumkong_static.a golang/ # the punned kernels it calls
cp -r include/usearch golang/                        # to make the headers available to Go

cd golang && go test -race -v ./... ; cd ..
```

The race detector matters here: the tests drive one shared handle from many goroutines, and it is the only thing that catches a missing `runtime.KeepAlive` around a matrix crossing into C.

Compilation and the static checks need only the headers, not the library, so they run before it links:

```sh
cp -r include/usearch golang/

cd golang
go build ./...      # compiles the cgo preamble and shim.c against the real headers
go vet ./...        # catches the cgo pointer-rule violations the headers cannot
gofmt -l .          # must print nothing
staticcheck ./...   # if installed
golangci-lint run   # if installed
```

## Java, Kotlin and Scala

The Java SDK reaches the C interface through the FFM API, so it compiles no native boundary of its own — there is no JNI shim and no `usearch_java` target.
Java 25 is the floor, and the runtime needs `--enable-native-access`, which `build.gradle` already passes to every test task.

Gradle drives CMake for the engine, so a plain build needs nothing else:

```sh
gradle clean build --warning-mode=all   # build, engine included
gradle test --rerun-tasks               # the Java suite
gradle spotlessApply                    # format Java, Kotlin and Scala
```

`-PusearchPrebuilt=<directory>` points every suite at a library that is already built and skips the CMake tasks, which is how CI builds the engine once and tests it from three languages.
```sh
cmake -B build_release -D USEARCH_BUILD_LIB_C=1
cmake --build build_release --target usearch_static usearch_shared
gradle -PusearchPrebuilt=$PWD/build_release test
```

Kotlin and Scala consume the same published Java surface rather than a binding of their own, and each has a suite under `kotlin/` and `scala/` covering what only another JVM language can observe — that `@NullMarked` reaches a Kotlin caller as real non-null types, and that the array-based batch verbs take Scala arrays unconverted.
`check` runs all three.

```sh
gradle kotlinInteropTest scalaInteropTest
```

Adding a fourth JVM language is one entry in the `interopFrameworks` map in `build.gradle`, which derives its source set, its configurations and its test task.

## C#

Setup the .NET environment:

```sh
dotnet nuget add source https://api.nuget.org/v3/index.json -n nuget.org
```

USearch provides CSharp bindings on .NET 10, reaching the C interface through source-generated `LibraryImport`.
The project drives no CMake of its own, so the shared library is built first and dropped where the runtime probes for it.
```sh
cmake -B build_artifacts -D USEARCH_BUILD_LIB_C=1
cmake --build build_artifacts --config Release -j
```

Then, on Windows, copy the library to the CSharp project and run the tests.
The binding imports `libusearch_shared`, which .NET probes for verbatim while MSVC emits no `lib` prefix, so the destination keeps the name the import expects:

```sh
mkdir -p ".\csharp\lib\runtimes\win-x64\native"
cp ".\build_artifacts\usearch_shared.dll" ".\csharp\lib\runtimes\win-x64\native\libusearch_shared.dll"
cd csharp
dotnet test -c Debug --logger "console;verbosity=detailed"
dotnet test -c Release
```

On Linux, the process is similar:

```sh
mkdir -p "csharp/lib/runtimes/linux-x64/native" # for x86
cp "build_artifacts/libusearch_shared.so" "csharp/lib/runtimes/linux-x64/native" # for x86
mkdir -p "csharp/lib/runtimes/linux-arm64/native" # for ARM
cp "build_artifacts/libusearch_shared.so" "csharp/lib/runtimes/linux-arm64/native" # for ARM
cd csharp
dotnet test -c Debug --logger "console;verbosity=detailed"
dotnet test -c Release
```

On macOS with Arm-based chips:

```sh
mkdir -p "csharp/lib/runtimes/osx-arm64/native"
cp "build_artifacts/libusearch_shared.dylib" "csharp/lib/runtimes/osx-arm64/native"
cd csharp
dotnet test -c Debug --logger "console;verbosity=detailed"
dotnet test -c Release
```

## Wolfram

USearch reaches the Wolfram Language through LibraryLink, whose boundary is `wolfram/lib.c`.
It builds from the repository root, and `WOLFRAM_PATH` names the Mathematica installation when it is not in the default place.

```sh
brew install --cask wolfram-engine

cmake -B build_release -D USEARCH_BUILD_LIB_C=1 -D USEARCH_BUILD_LIB_WOLFRAM=1 \
    -D WOLFRAM_PATH=/usr/local/Wolfram/Mathematica
cmake --build build_release --config Release --target usearch_wolfram

wolframscript -file wolfram/test.wls
```

It links the C interface as an archive, so what Wolfram loads is one file, and it is the only target here without a `lib` prefix — Wolfram finds a library in `LibraryResources/$SystemID` by its bare name.

## Docker

```sh
docker build -t unum/usearch . && docker run unum/usearch
```

For multi-architecture builds and publications:

```sh
version=$(cat VERSION)
docker buildx create --use &&
    docker login &&
    docker buildx build \
        --platform "linux/amd64,linux/arm64" \
        --build-arg version=$version \
        --file Dockerfile \
        --tag unum/usearch:$version \
        --tag unum/usearch:latest \
        --push .
```

## WebAssembly

The WASI build uses the same module shape as the Emscripten `wasm32_v128_k` build, but for standalone runtimes and for the Rust crate's `wasm32-wasip1` target.
Unpack a [WASI SDK release](https://github.com/WebAssembly/wasi-sdk/releases) and point `WASI_SDK_PATH` at it; `cmake/toolchain-wasm32-wasi.cmake` names the compilers, the sysroot and the SIMD tier from there:

```sh
export WASI_SDK_PATH=/opt/wasi-sdk-34.0-x86_64-linux
cmake -B build_wasm32_wasi -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-wasm32-wasi.cmake -DUSEARCH_BUILD_LIB_C=1 -DUSEARCH_BUILD_TEST=1
cmake --build build_wasm32_wasi --target usearch_static usearch_test_c
ctest --test-dir build_wasm32_wasi -R '^usearch_test_c$'
```

`ctest` runs the module under Wasmtime when it is on the path.
The Rust crate hands CMake the same toolchain file on its own, so `cargo build --target wasm32-wasip1` needs only `WASI_SDK_PATH`.

## Working on Sub-Modules

Extending metrics in NumKong:

```sh
git push --set-upstream https://github.com/ashvardanian/numkong.git HEAD:main
```
