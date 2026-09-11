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

# autonne consumed with add_subdirectory, which is what FetchContent does and
# therefore how the project that asked for this kernel will consume it.
#
# What this checks, beyond "it links": that a subproject build adds no test
# targets and fetches no GoogleTest. Both are off by default because
# AUTONNE_BUILD_TESTS follows AUTONNE_IS_TOP_LEVEL, and a regression there
# would make every consumer clone and build a test framework they did not ask
# for -- visible immediately to them and to nobody here.

cmake_minimum_required(VERSION 3.20)
project(autonne_subproject_check LANGUAGES CXX)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../.. autonne_build)

if(TARGET gtest OR TARGET gtest_main)
  message(FATAL_ERROR
    "a subproject build pulled in GoogleTest; AUTONNE_BUILD_TESTS should "
    "default to off when autonne is not the top-level project")
endif()
foreach(test_target autonne_tests_strict autonne_tests_fastmath
                    autonne_tests_strict_nomdspan autonne_dump_strict)
  if(TARGET ${test_target})
    message(FATAL_ERROR
      "a subproject build defined ${test_target}; the test suite should not "
      "be built when autonne is consumed as a subproject")
  endif()
endforeach()

# The same program the install check uses, so the two paths are held to one
# standard: factor, verify, and refuse a NaN written as bits from a
# translation unit built with -ffast-math.
add_executable(autonne_subproject_check ../consumer/main.cpp)
target_link_libraries(autonne_subproject_check PRIVATE autonne::autonne)
if(CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang)$")
  target_compile_options(autonne_subproject_check PRIVATE
    -ffast-math -Wall -Wextra -Werror)
endif()
