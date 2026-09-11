#!/usr/bin/env bash
# Copyright 2026 The autonne Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Checks that every floating-point-sensitive entry point has exactly one
# strong definition in each binary handed to it, and that nothing under
# autonne::verify is exported with vague (weak) linkage.
#
# Why this matters: a weak symbol is emitted by every translation unit that
# instantiates it and the linker keeps one copy per binary, chosen without
# regard to the compile flags it was built under. A per-file -ffast-math (or
# -fno-fast-math) then governs which copy a file EMITS and never which copy
# SURVIVES. One strong definition per binary means the flag on that one file is
# the flag that applies.
#
# Usage: tools/check_symbols.sh <archive-or-executable>...
# Set NM to override the nm binary (llvm-nm works).

set -euo pipefail

nm_tool="${NM:-nm}"
if ! command -v "$nm_tool" >/dev/null 2>&1; then
  echo "check_symbols: '$nm_tool' not found" >&2
  exit 2
fi

entry_points=(
  'autonne::svd_thin('
  'autonne::eigh('
  'autonne::verify::check_svd('
  'autonne::verify::check_eigh('
)

status=0
for bin in "$@"; do
  if [ ! -f "$bin" ]; then
    echo "check_symbols: no such file: $bin" >&2
    status=1
    continue
  fi
  # GNU and LLVM nm demangle with -C and filter with --defined-only; Apple's
  # nm spells the filter -U and may lack -C, so fall back to c++filt.
  if ! symbols="$("$nm_tool" -C --defined-only "$bin" 2>/dev/null)"; then
    symbols="$("$nm_tool" -U "$bin" 2>/dev/null | c++filt || true)"
  fi

  # Itanium demangling prints "autonne::svd_thin(...)"; Microsoft demangling
  # prints "bool __cdecl autonne::svd_thin(...)". Match on the qualified name
  # anywhere after the symbol type so both work.
  for sym in "${entry_points[@]}"; do
    strong="$(printf '%s\n' "$symbols" | grep -E ' T ' | grep -F "$sym" | grep -c . || true)"
    weak="$(printf '%s\n' "$symbols" | grep -E ' [WVwv] ' | grep -F "$sym" | grep -c . || true)"
    if [ "$strong" -ne 1 ] || [ "$weak" -ne 0 ]; then
      echo "check_symbols: $bin: $sym has $strong strong and $weak weak definitions (want 1 and 0)"
      status=1
    fi
  done

  # The harness's floating-point helpers live in an anonymous namespace inside
  # verify.cpp. If any of them shows up with vague linkage, something has moved
  # back into a header.
  for helper in within frobenius_sq orthonormality_residual all_finite; do
    weak_helper="$(printf '%s\n' "$symbols" | grep -E ' [WVwv] ' | grep -E "autonne::verify::(\(anonymous namespace\)::)?${helper}[<(]" || true)"
    if [ -n "$weak_helper" ]; then
      echo "check_symbols: $bin: harness helper with vague linkage:"
      printf '%s\n' "$weak_helper"
      status=1
    fi
  done
done

if [ "$status" -eq 0 ]; then
  echo "check_symbols: ok ($# file(s))"
fi
exit "$status"
