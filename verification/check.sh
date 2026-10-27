#!/usr/bin/env bash
# Runs every `@verify` line of the Promela models and GenMC clients here through ForkUnion's runner,
# which also supplies `weak_memory.pml` and `genmc.hpp`. ForkUnion comes from the `forkunion`
# submodule, or from the checkout `FORKUNION_SOURCE` names. Needs `spin` and a C compiler; a GenMC
# on the path, or named by `GENMC`, also runs the clients.

set -u
cd "$(dirname "$0")"
forkunion=${FORKUNION_SOURCE:-../forkunion}
if [ ! -f "$forkunion/verification/check.sh" ]; then
    echo "No ForkUnion runner at $forkunion: run 'git submodule update --init forkunion' or set FORKUNION_SOURCE" >&2
    exit 1
fi
source "$forkunion/verification/check.sh"
verify_all "$@"
