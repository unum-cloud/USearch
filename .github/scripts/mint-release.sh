#!/usr/bin/env bash
# Mints a release: stamps the version everywhere, commits, tags, pushes and publishes the release page.
#
#     .github/scripts/mint-release.sh 3.0.0          # stable
#     .github/scripts/mint-release.sh 3.1.0-rc.1     # candidate, published as a prerelease
set -euo pipefail
version=$1 core=${1%%-*}
branch=$(git branch --show-current)
# The first release has no tag before it, so its notes start at the root commit.
previous=$(git describe --tags --abbrev=0 2>/dev/null || git rev-list --max-parents=0 HEAD)
git diff --quiet HEAD || { echo "mint-release: commit or stash your changes first"; exit 1; }

# The Swift binaries `Package.swift` names by checksum are built from this pushed head, stamped with the version
git push origin "$branch"
gh workflow run build-swift.yml --ref "$branch" -f version="$version"
sleep 5
run=$(gh run list --workflow build-swift.yml --branch "$branch" --event workflow_dispatch --limit 1 --json databaseId --jq '.[0].databaseId')
gh run watch "$run" --exit-status
assets=$(mktemp -d)
gh run download "$run" --name swift-xcframework --name swift-artifactbundle --dir "$assets"

IFS=. read -r major minor patch <<< "$core"
export version core major minor patch
echo "$version" > VERSION
perl -pi -e 's/^(  "version": ")[^"]+/$1$ENV{version}/' package.json
perl -pi -e 's/^(version = ")[^"]+/$1$ENV{version}/' Cargo.toml pixi.toml mojo/pixi.toml
perl -pi -e 's/^(    \.version = ")[^"]+/$1$ENV{version}/' build.zig.zon
perl -pi -e 's/(def usearchVersion = \x27)[^\x27]+\x27/$1$ENV{version}\x27/; s/(implementation name: \x27usearch\x27, version: \x27)[^\x27]+\x27/$1$ENV{version}\x27/' java/README.md
perl -pi -e 's/(<Version Condition=[^>]+>)[^<]+(<\/Version>)/$1$ENV{version}$2/' csharp/nuget/nuget-package.props
perl -pi -e 's/^(    VERSION )[0-9.]+$/$1$ENV{core}/' CMakeLists.txt
perl -pi -e 's/^(#define USEARCH_VERSION_MAJOR )[0-9]+$/$1$ENV{major}/; s/^(#define USEARCH_VERSION_MINOR )[0-9]+$/$1$ENV{minor}/; s/^(#define USEARCH_VERSION_PATCH )[0-9]+$/$1$ENV{patch}/' c/usearch/types.h
perl -pi -e 's/(usearch_version_name\(void\) \{ return ")[^"]+/$1$ENV{core}/' src_stub/types.cpp
perl -pi -e 's#(releases/download/v)[^"]+#$1$ENV{version}#' Package.swift
apple=$(sha256sum "$assets"/*/*.xcframework.zip | cut -c -64) portable=$(sha256sum "$assets"/*/*.artifactbundle.zip | cut -c -64)
perl -pi -e "s/^let appleChecksum = .*/let appleChecksum = \"$apple\"/; s/^let portableChecksum = .*/let portableChecksum = \"$portable\"/" Package.swift
git commit -am "Release: v$version"
.github/scripts/check-release.sh "v$version" # catches a missed site even where `core.hooksPath` is unset
git tag -a "v$version" -m "Release: v$version"
git push --atomic origin "$branch" "v$version"

# Candidates never take "Latest", and neither does a patch to an older line.
flags=()
[[ $version == *-rc.* ]] && flags+=(--prerelease --title "Candidate v$version")
[ "$(git tag -l 'v*' --sort=-v:refname | grep -v -- - | head -1)" = "v$version" ] || flags+=(--latest=false)
gh release create "v$version" --verify-tag --notes "$(git log --no-merges --format='- %s (%h)' "$previous..v$version^")" \
    "${flags[@]}" "$assets"/*/*.zip
