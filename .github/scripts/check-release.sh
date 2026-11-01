#!/usr/bin/env bash
# Checks that HEAD is the release commit for a tag and that its version files agree with it.
#
#     .github/scripts/check-release.sh v3.0.0
set -euo pipefail
tag=$1
subject=$(git log -1 --format=%s)
if [ "$subject" != "Release: $tag" ]; then
    echo "check-release: $tag must sit on 'Release: $tag', but HEAD is '$subject'"
    exit 1
fi
message=$(mktemp)
git log -1 --format=%B > "$message"
.githooks/commit-msg "$message"
