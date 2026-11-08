#!/usr/bin/env bash
# Writes the symbol bundle unum.cloud renders: each binding's own tool output, or its sources where no tool emits one.
#
#     .github/scripts/build-docs.sh        # → build/usearch_docs_<version>.tar.gz
#
# Swift typechecks against NumKong's module, taken from the `numkong` submodule or FETCHCONTENT_SOURCE_DIR_NUMKONG.
set -euo pipefail
version=$(cat VERSION)
name=usearch_docs_$version out=build/usearch_docs_$version scratch=build/docs-scratch
numkong=${FETCHCONTENT_SOURCE_DIR_NUMKONG:-numkong}
# Rustdoc's JSON is unstable, and `rust-version` needs 1.100, which reaches stable on 2026-11-12.
toolchain=${USEARCH_RUST_TOOLCHAIN:-beta-2026-10-18}
rm -rf "$out" "$scratch" && mkdir -p "$out"/{doxygen,doxygen-java,doxygen-csharp,headers,python,rust,swift} \
    "$out"/{javascript,go,zig,wolfram,mojo} "$scratch"/{library,java,csharp}

rm -rf build/xml && (cd docs && doxygen conf.dox) && cp -r build/xml/. "$out/doxygen"
# The two managed bindings reuse the C pass's settings, each with its own sources and output; Doxygen cannot
# resolve their links into the JDK or .NET, so those passes keep quiet instead of failing.
(cd docs && { cat conf.dox; printf '%s\n' 'INPUT = ../java' 'EXCLUDE =' 'FILE_PATTERNS = *.java' \
    'EXCLUDE_PATTERNS = */test/*' 'EXTRACT_PACKAGE = NO' 'WARNINGS = NO' 'WARN_AS_ERROR = NO' \
    "OUTPUT_DIRECTORY = ../$scratch/java"; } | doxygen -)
cp -r "$scratch/java/xml/." "$out/doxygen-java"
(cd docs && { cat conf.dox; printf '%s\n' 'INPUT = ../csharp/src/Cloud.Unum.USearch' 'EXCLUDE =' \
    'FILE_PATTERNS = *.cs' 'EXCLUDE_PATTERNS = */obj/* */bin/*' 'WARNINGS = NO' 'WARN_AS_ERROR = NO' \
    "OUTPUT_DIRECTORY = ../$scratch/csharp"; } | doxygen -)
cp -r "$scratch/csharp/xml/." "$out/doxygen-csharp"
git ls-files ':(glob)c/usearch/*.h' ':(glob)cpp/usearch/*.hpp' ':(glob)java/cloud/**/*.java' \
    ':(glob)csharp/src/Cloud.Unum.USearch/*.cs' | xargs cp --parents -t "$out/headers"

griffe dump usearch --search python --no-inspection --docstyle google --full --output "$out/python/usearch.json"
cp python/usearch/lib.pyi "$out/python/"

# Rustdoc links nothing, so every crate with a native library is pointed at an empty directory instead of building it.
USEARCH_LIBRARY_DIR=$PWD/$scratch/library NUMKONG_LIBRARY_DIR=$PWD/$scratch/library \
    STRINGZILLA_LIBRARY_DIR=$PWD/$scratch/library RUSTC_BOOTSTRAP=1 \
    cargo +"$toolchain" rustdoc --lib -- -Z unstable-options --output-format json
cp "${CARGO_TARGET_DIR:-target}/doc/usearch.json" "$out/rust/"

swiftc -emit-module -emit-module-path "$scratch/NumKong.swiftmodule" -module-name NumKong -package-name NumKong \
    -parse-as-library -I "$numkong/c" -I "$numkong/include" "$numkong"/swift/*.swift
# The module map sits beside `usearch/*.h`, as the release's artifact bundle installs them.
mkdir -p "$scratch/usearch-c"
cp include/module.modulemap "$scratch/usearch-c/" && cp -r c/usearch "$scratch/usearch-c/"
swiftc -emit-module -emit-module-path "$scratch/USearch.swiftmodule" -module-name USearch -package-name USearch \
    -parse-as-library -I "$scratch" -I "$numkong/c" -I "$scratch/usearch-c" \
    -emit-symbol-graph -emit-symbol-graph-dir "$out/swift" swift/Types.swift swift/DenseIndex.swift

cp javascript/*.ts "$out/javascript/"
cp golang/*.go golang/go.mod "$out/go/" && rm -f "$out"/go/*_test.go
cp -r zig/usearch.zig zig/usearch "$out/zig/"
cp wolfram/USearch.wl "$out/wolfram/"
mojo doc -o "$out/mojo/usearch.json" mojo/usearch
cp mojo/usearch/__init__.mojo "$out/mojo/"

# griffe, rustdoc, Mojo and Doxygen's body locations hold absolute paths, which would leak the build machine's
# directories.
sed -i "s|$PWD/||g; s|$PWD|.|g; s|$HOME|~|g" "$out/python/usearch.json" "$out/rust/usearch.json" \
    "$out/mojo/usearch.json" "$out"/doxygen*/*.xml "$out"/swift/*.json
printf '{"project":"usearch","version":"%s","commit":"%s"}\n' "$version" "$(git rev-parse HEAD)" > "$out/manifest.json"
tar -czf "build/$name.tar.gz" -C build "$name"
