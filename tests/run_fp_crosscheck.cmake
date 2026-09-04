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

# Runs the strict and the fast-math dump executables on one corpus matrix and
# compares what they wrote.
#
# Expects: AUTONNE_DUMP_STRICT, AUTONNE_DUMP_FASTMATH, AUTONNE_COMPARE,
#          AUTONNE_INPUT, AUTONNE_KIND (svd|eigh), AUTONNE_WORK_DIR

get_filename_component(case_name "${AUTONNE_INPUT}" NAME_WE)
set(strict_out "${AUTONNE_WORK_DIR}/${case_name}.${AUTONNE_KIND}.strict.txt")
set(fast_out "${AUTONNE_WORK_DIR}/${case_name}.${AUTONNE_KIND}.fastmath.txt")

file(MAKE_DIRECTORY "${AUTONNE_WORK_DIR}")

execute_process(
  COMMAND "${AUTONNE_DUMP_STRICT}" "${AUTONNE_INPUT}" "${strict_out}" "${AUTONNE_KIND}"
  RESULT_VARIABLE strict_result
  OUTPUT_VARIABLE strict_output
  ERROR_VARIABLE strict_output)
if(NOT strict_result EQUAL 0)
  message(FATAL_ERROR "strict dump failed (${strict_result}): ${strict_output}")
endif()

execute_process(
  COMMAND "${AUTONNE_DUMP_FASTMATH}" "${AUTONNE_INPUT}" "${fast_out}" "${AUTONNE_KIND}"
  RESULT_VARIABLE fast_result
  OUTPUT_VARIABLE fast_output
  ERROR_VARIABLE fast_output)
if(NOT fast_result EQUAL 0)
  message(FATAL_ERROR "fast-math dump failed (${fast_result}): ${fast_output}")
endif()

execute_process(
  COMMAND "${AUTONNE_COMPARE}" "${strict_out}" "${fast_out}"
          --label "${case_name}.${AUTONNE_KIND}"
  RESULT_VARIABLE compare_result
  OUTPUT_VARIABLE compare_output
  ERROR_VARIABLE compare_output)
message("${compare_output}")
if(NOT compare_result EQUAL 0)
  message(FATAL_ERROR
    "the strict and fast-math builds disagree beyond tolerance on "
    "${case_name} (${AUTONNE_KIND})")
endif()
