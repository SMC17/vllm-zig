#!/usr/bin/env bash
# Clone private sibling deps into deps/ for clean-clone builds.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p deps
ST_SHA="${SAFETENSORS_SHA:-17328a3f984ab2b657396302f438a572e8a05be9}"
TK_SHA="${TOKENIZERS_SHA:-558bfbcaf29319aacf6641ce81c901e571f14a15}"

clone_dep() {
  local name=$1 sha=$2
  local dest="deps/$name"
  if [ -e "$dest/.git" ] || [ -L "$dest" ]; then
    echo "deps/$name already present"
    return 0
  fi
  if [ -d "../$name/.git" ]; then
    ln -sfn "../$name" "$dest"
    echo "symlinked deps/$name -> ../$name"
    return 0
  fi
  echo "cloning SMC17/$name @$sha"
  gh repo clone "SMC17/$name" "$dest" -- --depth 1
  git -C "$dest" fetch --depth 1 origin "$sha"
  git -C "$dest" checkout "$sha"
}

clone_dep safetensors-zig "$ST_SHA"
clone_dep tokenizers-zig "$TK_SHA"
echo "deps ready"
