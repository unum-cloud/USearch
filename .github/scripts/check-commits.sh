#!/usr/bin/env bash
# Runs the repository's Git hooks over every commit since a base, as they ran on each commit.
#
#     .github/scripts/check-commits.sh <base>
set -euo pipefail
base=$1
[ ! -x .githooks/selftest ] || .githooks/selftest
head=$(git rev-parse HEAD)
message=$(mktemp)
index=$(mktemp -u)
# Each commit is replayed as it was made: HEAD at its parent, a private index holding its tree.
trap 'git reset --quiet --soft "$head"; rm -f "$message" "$index"' EXIT
for commit in $(git rev-list --reverse --no-merges "$base..$head"); do
    git log -1 --format=%B "$commit" > "$message"
    git reset --quiet --soft "$commit^"
    GIT_INDEX_FILE=$index git read-tree "$commit"
    GIT_INDEX_FILE=$index .githooks/commit-msg "$message"
done
# Staging the whole range lets `pre-commit` judge every added line.
git reset --quiet --soft "$base"
.githooks/pre-commit
