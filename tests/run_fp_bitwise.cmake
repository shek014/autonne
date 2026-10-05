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

# Runs the strict dump as built for the default target, for x86-64-v3, and for
# x86-64-v3 with vectorisation off, on one corpus matrix, and requires the
# three output files to be identical byte for byte.
#
# Expects: AUTONNE_DUMP_STRICT, AUTONNE_DUMP_FMA, AUTONNE_DUMP_FMA_SCALAR,
#          AUTONNE_INPUT, AUTONNE_KIND (svd|svd_bdc|eigh), AUTONNE_WORK_DIR

get_filename_component(case_name "${AUTONNE_INPUT}" NAME_WE)
file(MAKE_DIRECTORY "${AUTONNE_WORK_DIR}")

foreach(build STRICT FMA FMA_SCALAR)
  string(TOLOWER "${build}" label)
  set(out_${build} "${AUTONNE_WORK_DIR}/${case_name}.${AUTONNE_KIND}.${label}.txt")
  execute_process(
    COMMAND "${AUTONNE_DUMP_${build}}" "${AUTONNE_INPUT}" "${out_${build}}" "${AUTONNE_KIND}"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE output)
  if(NOT result EQUAL 0)
    message(FATAL_ERROR "${label} dump failed (${result}): ${output}")
  endif()
endforeach()

set(failed FALSE)
foreach(build FMA FMA_SCALAR)
  string(TOLOWER "${build}" label)
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E compare_files "${out_STRICT}" "${out_${build}}"
    RESULT_VARIABLE result)
  if(result EQUAL 0)
    message("${case_name}.${AUTONNE_KIND}: strict and ${label} builds agree bit for bit")
  else()
    message("${case_name}.${AUTONNE_KIND}: strict and ${label} builds differ:\n"
            "  ${out_STRICT}\n  ${out_${build}}")
    set(failed TRUE)
  endif()
endforeach()

if(failed)
  message(FATAL_ERROR
    "the strict model gave different bits for ${case_name} (${AUTONNE_KIND}) "
    "depending on the target or on vectorisation; see #20")
endif()
