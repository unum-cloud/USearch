#!/usr/bin/env bash
# Checks that HEAD is the release commit for a tag and that every version site carries its version.
#
#     .github/scripts/check-release.sh v2.26.7
set -euo pipefail
tag=$1 version=${1#v}
core=${version%%-*}
subject=$(git log -1 --format=%s)
if [[ ! $tag =~ ^v[0-9]+\.[0-9]+\.[0-9]+(-rc\.[0-9]+)?$ ]]; then
    echo "check-release: a release tag is exactly 'vX.Y.Z' or 'vX.Y.Z-rc.N', got '$tag'"
    exit 1
fi
if [ "$subject" != "Release: $tag" ]; then
    echo "check-release: $tag must sit on 'Release: $tag', but HEAD is '$subject'"
    exit 1
fi
stale=""
[ "$(cat VERSION)" = "$version" ] || stale="$stale VERSION"
grep -qxF "    \"version\": \"$version\"," package.json || stale="$stale package.json"
grep -qxF "version: $version" CITATION.cff || stale="$stale CITATION.cff"
grep -qxF "version = \"$version\"" Cargo.toml || stale="$stale Cargo.toml"
grep -qxF "version = \"$version\"" wasmer.toml || stale="$stale wasmer.toml"
grep -qxF "    version = \"$version\"" conanfile.py || stale="$stale conanfile.py"
grep -qF "def usearchVersion = '$version'" java/README.md || stale="$stale java/README.md"
grep -qF "implementation name: 'usearch', version: '$version'," java/README.md || stale="$stale java/README.md"
grep -qxF "version = {$version}," README.md || stale="$stale README.md"
grep -qF "'\$(Version)' == ''\">$version</Version>" csharp/nuget/nuget-package.props || stale="$stale csharp/nuget/nuget-package.props"
# CMake's `project(VERSION)` and the C++ macros take numbers only, so a candidate keeps its core there.
grep -qE "^[[:space:]]+VERSION ${core//./\\.}$" CMakeLists.txt || stale="$stale CMakeLists.txt"
header=$(sed -nE 's/^#define USEARCH_VERSION_(MAJOR|MINOR|PATCH) ([0-9]+)$/\2/p' include/usearch/index.hpp | paste -sd. -)
[ "$header" = "$core" ] || stale="$stale include/usearch/index.hpp"
if [ -n "$stale" ]; then
    echo "check-release: these files don't carry $version:$stale"
    exit 1
fi
