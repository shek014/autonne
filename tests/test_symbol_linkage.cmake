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

# Asserts that the harness entry points are strong, single definitions.
#
# check_svd, check_eigh and orthonormality_residual are defined in
# src/verify.cpp precisely so that the linker cannot merge copies compiled
# under different floating-point flags. That guarantee is invisible in the
# source: move any of them back into the header and it becomes a weak symbol
# again, the build still succeeds, every other test still passes, and the
# protection is silently gone. This is the check that notices.
#
# A global definition must be exactly one, and it must be strong (nm type T),
# never weak (W or V). Local clones -- GCC emits .constprop / .isra / .part
# variants at -O3, which nm reports in lower case -- are fine: they have
# internal linkage and cannot participate in cross-translation-unit merging.
#
# Usage:
#   cmake -DAUTONNE_NM=<nm> -DAUTONNE_BINARY=<file> -P test_symbol_linkage.cmake

if(NOT DEFINED AUTONNE_NM OR NOT DEFINED AUTONNE_BINARY)
  message(FATAL_ERROR "AUTONNE_NM and AUTONNE_BINARY must both be set")
endif()

if(NOT EXISTS "${AUTONNE_BINARY}")
  message(FATAL_ERROR "no such binary: ${AUTONNE_BINARY}")
endif()

execute_process(
  COMMAND "${AUTONNE_NM}" --defined-only "${AUTONNE_BINARY}"
  OUTPUT_VARIABLE nm_output
  ERROR_VARIABLE nm_error
  RESULT_VARIABLE nm_result)

if(NOT nm_result EQUAL 0)
  message(FATAL_ERROR "nm failed on ${AUTONNE_BINARY}:\n${nm_error}")
endif()

string(REPLACE "\n" ";" nm_lines "${nm_output}")

set(failures "")

foreach(fn IN ITEMS check_svd check_eigh orthonormality_residual)
  set(global_matches "")
  set(local_count 0)

  foreach(line IN LISTS nm_lines)
    if(NOT line MATCHES "^[0-9a-fA-F]+[ \t]+([A-Za-z])[ \t]+(.+)$")
      continue()
    endif()
    set(sym_type "${CMAKE_MATCH_1}")
    set(sym_name "${CMAKE_MATCH_2}")
    if(NOT sym_name MATCHES "autonne" OR NOT sym_name MATCHES "${fn}")
      continue()
    endif()
    if(sym_type STREQUAL "T" OR sym_type STREQUAL "W" OR sym_type STREQUAL "V"
       OR sym_type STREQUAL "D" OR sym_type STREQUAL "B")
      list(APPEND global_matches "${sym_type} ${sym_name}")
    else()
      math(EXPR local_count "${local_count} + 1")
    endif()
  endforeach()

  list(LENGTH global_matches n_global)

  if(NOT n_global EQUAL 1)
    string(REPLACE ";" "\n      " shown "${global_matches}")
    list(APPEND failures
      "  ${fn}: expected exactly 1 global definition, found ${n_global}\n      ${shown}")
  else()
    list(GET global_matches 0 only)
    if(NOT only MATCHES "^T ")
      list(APPEND failures
        "  ${fn}: definition is not strong (expected nm type T)\n      ${only}")
    else()
      message(STATUS "${fn}: 1 strong definition (T), ${local_count} local clone(s)")
    endif()
  endif()
endforeach()

if(NOT failures STREQUAL "")
  string(REPLACE ";" "\n" failures "${failures}")
  message(FATAL_ERROR
    "harness entry points do not have single strong definitions in\n"
    "${AUTONNE_BINARY}\n${failures}\n"
    "Moving one of these back into verify.hpp gives it vague linkage, which is "
    "exactly what src/verify.cpp exists to prevent.")
endif()
