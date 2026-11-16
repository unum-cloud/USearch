#!/usr/bin/env bash
# Fails when a GPU class bundle holds a NumKong or StringZilla kernel or device table: a class unit compiles the
# libraries' building blocks with their GPU floors off, so every entry and global it ships is USearch's own.
#
#     test/bundles.sh <bundles> <cuobjdump> <clang-offload-bundler> <llvm-readelf> [processor...]
#
# NVIDIA bundles are read with `cuobjdump`, AMD offload bundles are split per listed processor and read with
# `llvm-readelf`; an empty tool skips its vendor.

set -euo pipefail
bundles=$1 cuobjdump=$2 bundler=$3 readelf=$4
shift 4
object=$(mktemp)
trap 'rm -f "$object"' EXIT
checked=0 leaks=""
while IFS= read -r -d '' bundle; do
    [ -s "$bundle" ] || continue
    names=""
    if [ "$(head -c 24 "$bundle" | tr -d '\0')" = "__CLANG_OFFLOAD_BUNDLE__" ]; then
        [ -n "$bundler" ] || continue
        split=0
        for processor in "$@"; do
            : > "$object"
            "$bundler" --unbundle --type=o --allow-missing-bundles --input="$bundle" \
                --targets="hipv4-amdgcn-amd-amdhsa--$processor" --output="$object"
            [ -s "$object" ] || continue
            split=$((split + 1))
            names+=$("$readelf" --symbols --wide "$object" | awk '$4 == "FUNC" || $4 == "OBJECT" { print $8 }')$'\n'
        done
        [ "$split" != 0 ] || { echo "$bundle: no code object for processors $*" >&2; exit 1; }
    else
        [ -n "$cuobjdump" ] || continue
        names=$("$cuobjdump" -elf "$bundle" | awk '
            /^\.section/ { symbols = ($0 ~ /^\.section \.symtab/); next }
            symbols && NF > 4 && $4 ~ /^0x(1|2|11|12)$/ { print $NF }')
    fi
    checked=$((checked + 1))
    leaks+=$(printf '%s\n' "$names" | grep -E '^\$?(_Z[0-9]+)?(nk|sz)_' | sed "s|^|${bundle#"$bundles"/}: |" || true)
done < <(find "$bundles" -name '*.bundle' -print0)

if [ -n "$leaks" ]; then
    echo "$(printf '%s\n' "$leaks" | grep -c .) library symbols in USearch's GPU class bundles, first ones:" >&2
    printf '%s\n' "$leaks" | head -n 20 >&2
    exit 1
fi
[ "$checked" != 0 ] || { echo "No GPU class bundle found under $bundles" >&2; exit 1; }
echo "$checked GPU class bundles hold USearch's kernels alone"
