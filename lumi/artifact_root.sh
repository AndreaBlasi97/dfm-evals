#!/usr/bin/env bash

resolve_post_artifact_root() {
  local start="${1:-$PWD}"
  local dir="$start"

  if [[ -n "${POST_ARTIFACT_ROOT:-}" ]]; then
    printf '%s\n' "$POST_ARTIFACT_ROOT"
    return 0
  fi

  if [[ ! -d "$dir" ]]; then
    dir="$(dirname "$dir")"
  fi
  dir="$(cd "$dir" && pwd -P)"

  while true; do
    if [[ "$(basename "$dir")" == "post" ]]; then
      printf '%s\n' "$(dirname "$dir")/artifacts"
      return 0
    fi
    if [[ "$dir" == "/" ]]; then
      break
    fi
    dir="$(dirname "$dir")"
  done

  printf '%s\n' "$start/artifacts"
}
