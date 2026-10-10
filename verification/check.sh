#!/usr/bin/env bash
# Runs every `@verify` line of every Promela scenario and GenMC client and compares each verdict
# with the expected one: `pass` means the search completed with no error, `fail` that an assertion
# was violated, `stuck` that a process waits forever. Needs `spin` and a C compiler; a GenMC on the
# path, or named by `GENMC`, also runs the clients.

set -u
build=$(mktemp -d)
trap 'rm -rf "$build"' EXIT
harness=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd) # `genmc.hpp` lives beside this file, wherever it is run from
# Every verdict runs in its own directory in the background, `VERIFY_JOBS` at a time - four by
# default, since each pan holds a hash table of its own; `finish` prints them in the order they
# were queued, so the output reads like a serial run.
jobs_limit=${VERIFY_JOBS:-4}
queued=0

throttle() { while [ "$(jobs -rp | wc -l)" -ge "$jobs_limit" ]; do sleep 0.1; done; }

# section <title>: a heading, printed in its turn
section() {
    mkdir -p "$build/$queued"
    echo "$1" >"$build/$queued/result"
    queued=$((queued + 1))
}

# verify <model.pml> <expected> <spin defines...>
verify() {
    local model=$1 expected=$2 dir=$build/$queued
    shift 2
    mkdir -p "$dir"
    queued=$((queued + 1))
    throttle
    (
        local verdict
        if ! (cd "$dir" && spin -a "$@" "$OLDPWD/$model" >pan.out 2>&1 &&
            cc -O2 -DSAFETY -DCOLLAPSE -DVECTORSZ=4096 -w -o pan pan.c >>pan.out 2>&1); then verdict=broken
        else
            (cd "$dir" && ./pan -m1000000 -w26 >>pan.out 2>&1)
            # A cut-short search, a knob defined twice, or a write or index past a scenario's shape proves nothing
            if grep -qE "macro redefined|max search depth too small|invalid array index|violated.*history_full" "$dir/pan.out"; then verdict=broken
            elif grep -q "errors: 0" "$dir/pan.out"; then verdict=pass
            elif grep -q "pan:1: assertion violated" "$dir/pan.out"; then verdict=fail
            elif grep -q "pan:1: invalid end state" "$dir/pan.out"; then verdict=stuck
            else verdict=broken; fi
        fi
        report "$verdict" "$expected" "$model $*" "$dir/pan.out" >"$dir/result"
    ) &
}

report() { # report <verdict> <expected> <label> <log>
    if [ "$1" = "$2" ]; then echo "  ok    $3"
    else echo "  WRONG $3 : expected $2, got $1"; sed -n 1,12p "$4"; touch "$(dirname "$4")/wrong"; fi
}

# finish: waits for every verdict, prints them in order, and exits with the count of wrong ones
finish() {
    wait
    local index failed=0
    for ((index = 0; index < queued; index++)); do
        cat "$build/$index/result"
        [ -e "$build/$index/wrong" ] && failed=$((failed + 1))
    done
    exit $failed
}

# GenMC ships a freestanding C library whose headers shadow the platform's - `atomic` without
# `std::atomic_ref`, `stdio.h` without `EOF` - and puts its include directory behind the caller's.
# The clients need the real standard library, so a directory of forwarders for the shadowed
# names goes first. The compiler asked is the one GenMC was built with, `clang++` by default.
genmc_ready() {
    genmc=${GENMC:-$(command -v genmc || true)}
    [ -n "$genmc" ] || { section "  skip  GenMC is not installed; set GENMC to its executable"; return 1; }
    local compiler=${GENMC_CLANG:-clang++} directories libcxx="" libc=""
    directories=$(echo | "$compiler" -x c++ -E -v - 2>&1 | sed -n '/<...> search starts here/,/End of search list/p' | grep '^ /' | awk '{print $1}')
    for directory in $directories; do
        [ -z "$libcxx" ] && [ -f "$directory/atomic" ] && libcxx=$directory && continue
        [ -z "$libc" ] && [ -f "$directory/stdio.h" ] && libc=$directory
    done
    shadow=$build/shadow
    mkdir -p "$shadow"
    local name guard
    for name in atomic thread cassert cstdio cstdlib; do printf '#include "%s/%s"\n' "$libcxx" "$name" >"$shadow/$name"; done
    for name in assert.h pthread.h; do printf '#include "%s/%s"\n' "$libc" "$name" >"$shadow/$name"; done
    for name in stdio.h stdlib.h errno.h; do
        guard=_LIBCPP_$(echo "$name" | tr 'a-z.' 'A-Z_')
        printf '#define %s\n#include "%s/%s"\n' "$guard" "$libc" "$name" >"$shadow/$name"
    done
}

# GenMC treats a weak compare-exchange as one that may fail spuriously, so a read-first retry
# loop never ends for it; `--unroll` gives every loop that many turns.
genmc_unroll=${GENMC_UNROLL:-3}

# Runs a command with a wall-clock limit, since an exhaustive exploration that grows past a few
# minutes is a client to shrink, not a result to wait for.
limited() { # limited <seconds> <command...>
    local seconds=$1
    shift
    "$@" &
    local pid=$!
    (sleep "$seconds" && kill "$pid" 2>/dev/null) &
    local watchdog=$!
    wait "$pid" 2>/dev/null
    local status=$?
    pkill -P "$watchdog" 2>/dev/null
    kill "$watchdog" 2>/dev/null
    return $status
}

# verify_client <client.cpp> <expected> <memory model> <entry>: GenMC exits 0 when it found no
# error and 42 when it found one; `--check-liveness` makes a wait that never ends one of them.
verify_client() {
    local client=$1 expected=$2 memory=$3 entry=$4 dir=$build/$queued unroll=$genmc_unroll
    mkdir -p "$dir"
    queued=$((queued + 1))
    throttle
    (
        local verdict
        limited "${GENMC_SECONDS:-300}" "$genmc" --"$memory" --check-liveness --disable-estimation --unroll="$unroll" \
            --program-entry-function="$entry" -- -std=c++23 -I"$shadow" -I"$harness" -I../include -I"$harness/../include" \
            "$client" >"$dir/genmc.out" 2>&1
        case $? in
        0) if grep -q "complete executions explored: 0" "$dir/genmc.out"; then verdict=broken; else verdict=pass; fi ;;
        42) if grep -q "Liveness violation" "$dir/genmc.out"; then verdict=stuck; else verdict=fail; fi ;;
        *) verdict=broken ;;
        esac
        report "$verdict" "$expected" "$client $memory $entry" "$dir/genmc.out" >"$dir/result"
    ) &
}

# verify_all [file...]: every `@verify` line of the given scenarios and clients, or of every one
# here and one directory down, then the verdicts in order. A line names the expected verdict and
# the memory models, then knob overrides for a scenario or the entry for a client. Every knob the
# files declare must be set on some line, and every knob a line sets must be declared.
verify_all() {
    local files=("$@") file expected memories rest memory declared used unmatched entries field entry
    genmc_ready
    [ ${#files[@]} -gt 0 ] || files=(*.pml */*.pml *.cpp */*.cpp)
    declared=$(sed -n 's/^#ifndef \([a-z0-9_]*\)$/\1/p' "${files[@]}" */protocol.pml *.pml 2>/dev/null | grep -vx memory | sort -u)
    used=$(sed -n 's/^ \*  @verify [a-z]* [a-z0-9,]* \([^:]*\).*/\1/p' "${files[@]}" 2>/dev/null | tr ' ' '\n' | sed -n 's/=.*//p' | sort -u)
    [ $# -gt 0 ] && declared=$(comm -12 <(echo "$declared") <(echo "$used"))
    unmatched=$(comm -3 <(echo "$declared") <(echo "$used") | xargs)
    # A client's knobs are its `knobs_t` fields, each named by some entry, and every entry exists
    for file in "${files[@]}"; do
        [[ $file == *.cpp ]] && grep -qs '^ \*  @verify' "$file" || continue
        entries=$(sed -n 's/^ \*  @verify [a-z]* [a-z0-9,]* \([a-z0-9_]*\).*/\1/p' "$file" | sort -u)
        for field in $(sed -n '/^struct knobs_t {/,/^};/s/^ *[a-z_:]* \([a-z0-9_]*\) = .*;$/\1/p' "$file"); do
            grep -q "_${field}_" <<<"$entries" || unmatched="$unmatched $file:$field"
        done
        for entry in $entries; do grep -q "^extern \"C\" int $entry()" "$file" || unmatched="$unmatched $file:$entry"; done
    done
    mkdir -p "$build/$queued"
    if [ -z "$unmatched" ]; then echo "  ok    every knob is declared and set, every entry exists" >"$build/$queued/result"
    else
        echo "  WRONG knobs or entries declared or set alone: $unmatched" >"$build/$queued/result"
        touch "$build/$queued/wrong"
    fi
    queued=$((queued + 1))
    for file in "${files[@]}"; do
        grep -qs '^ \*  @verify' "$file" || continue
        [[ $file == *.cpp && -z "${genmc:-}" ]] && continue
        section "$file: $(sed -n 's/^ \*  @brief //p' "$file" | head -1)"
        while read -r expected memories rest; do
            for memory in ${memories//,/ }; do
                case $file in
                *.pml) verify "$file" "$expected" -Dmemory="$memory" $(for knob in $rest; do echo "-D$knob"; done) ;;
                *.cpp) verify_client "$file" "$expected" "$memory" "$rest" ;;
                esac
            done
        done < <(sed -n 's/^ \*  @verify \([^:]*\).*/\1/p' "$file")
    done
    finish
}

# Sourced, it only defines the functions above; run directly, it checks the directory it lives in.
[ "${BASH_SOURCE[0]}" = "$0" ] || return 0
cd "$(dirname "$0")"
verify_all "$@"
