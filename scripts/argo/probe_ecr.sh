#!/usr/bin/env bash
# What GPU images exist, and what architecture are they?
#
# Two questions this answers, neither of which can be asked from a Jupyter pod (the registry
# is in account 444488357543, the notebook role in 262301216538, so ECR denies it):
#
#   1. which CUDA versions have been built  -- needed to pick a fallback cuda-tag
#   2. whether the bare tag is a multi-arch index or a single image, and of which arch
#      -- decides whether cube_argo.yaml should use `develop.latest` or `develop.latest-amd64`,
#         and whether pinning nodes to amd64 is right or exactly backwards
#
# Uses skopeo, which needs no ecr:DescribeRepositories -- each repository is addressed by name.
REG=444488357543.dkr.ecr.us-west-2.amazonaws.com
REPO_FMT="easi-workflows-base-nvidia-%s-torch-cudnn-runtime"

# ---------------------------------------------------------------------------
# 1. Architecture of the tag we intend to use. THIS IS THE ONE THAT MATTERS.
# ---------------------------------------------------------------------------
# If .mediaType is an index/manifest.list, the bare tag is multi-arch and the platforms are
# listed -- use the bare tag. If it is a plain manifest, the bare tag is ONE architecture and
# `skopeo inspect` (below) says which; if that turns out to be arm64, cube_argo.yaml must use
# gpu-image-tag=develop.latest-amd64 instead, and the amd64 nodeSelector stays as it is.
arch_of() {   # arch_of <cuda-tag> <image-tag>
  local repo; repo=$(printf "$REPO_FMT" "$1")
  echo "--- $repo:$2"
  skopeo inspect --raw "docker://$REG/$repo:$2" \
    | jq -r '.mediaType, (.manifests[]? | "  platform: \(.platform.os)/\(.platform.architecture)")'
  skopeo inspect "docker://$REG/$repo:$2" | jq -r '"  resolved: \(.Os)/\(.Architecture)"'
}

arch_of 13-2-1 develop.latest
arch_of 13-2-1 develop.latest-amd64

# ---------------------------------------------------------------------------
# 2. Which CUDA versions exist. Only real upstream CUDA releases are worth trying, since the
#    repository naming mirrors NVIDIA's own image tags.
# ---------------------------------------------------------------------------
# Newest-first within 13.x, then the 12.x line. 12.x matters as a fallback: a T4 is compute
# capability 7.5, and each CUDA major drops older architectures, so if gpu-preflight reports
# no sm_75 kernels the fix is an older CUDA.
for c in 13-2-0 13-1-1 13-1-0 13-0-1 13-0-0 12-9-1 12-8-1 12-6-3 12-4-1; do
  repo=$(printf "$REPO_FMT" "$c")
  if tags=$(skopeo list-tags "docker://$REG/$repo" 2>/dev/null); then
    echo "EXISTS $c -> $(echo "$tags" | jq -r '.Tags | join(" ")')"
  fi
done

# The two non-GPU images cube_argo.yaml names, for completeness.
for repo in easi-workflows-base easi-workflows-base-torch-cpu; do
  if tags=$(skopeo list-tags "docker://$REG/$repo" 2>/dev/null); then
    echo "EXISTS $repo -> $(echo "$tags" | jq -r '.Tags | join(" ")')"
  fi
done
