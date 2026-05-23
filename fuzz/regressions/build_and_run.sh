#!/usr/bin/env bash
# Build and run a single regression test against the in-tree library.
# Usage: ./build_and_run.sh test_load_buffer_bounds.c
set -u
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
SRC=${1:?usage: $0 <test.c>}
NAME=$(basename "$SRC" .c)
OUT=$(mktemp -d)/$NAME

clang++ -std=c++20 -O1 -g -fno-omit-frame-pointer \
    -fsanitize=address,undefined \
    -I"$ROOT/include" -I"$ROOT/c" \
    -x c++ "$ROOT/c/lib.cpp" "$ROOT/fuzz/regressions/$SRC" \
    -DUSEARCH_USE_OPENMP=0 -DUSEARCH_USE_NUMKONG=0 -DNDEBUG \
    -pthread -o "$OUT" 2>&1 || { echo "BUILD FAILED"; exit 2; }

ASAN_OPTIONS="allocator_may_return_null=1:detect_leaks=0:abort_on_error=1:halt_on_error=1" \
UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1:abort_on_error=1" \
    timeout 10 "$OUT"
RC=$?
rm -rf "$(dirname "$OUT")"
exit $RC
