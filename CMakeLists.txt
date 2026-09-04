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

cmake_minimum_required(VERSION 3.20)

project(autonne
  VERSION 0.1.0
  DESCRIPTION "Truncated SVD and Hermitian eigendecomposition for small dense complex matrices"
  LANGUAGES CXX)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(AUTONNE_IS_TOP_LEVEL ON)
else()
  set(AUTONNE_IS_TOP_LEVEL OFF)
endif()

option(AUTONNE_BUILD_TESTS "Build the autonne test suite" ${AUTONNE_IS_TOP_LEVEL})

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# autonne depends on no linear algebra library. Eigen will appear later as a
# baseline inside a benchmark target only, and must never become a dependency
# of the library or of the test suite.

set(AUTONNE_SOURCES
  src/svd.cpp
  src/eigh.cpp
  src/verify.cpp)

set(AUTONNE_HEADERS
  include/autonne/autonne.hpp
  include/autonne/verify.hpp
  include/autonne/hexfloat.hpp
  include/autonne/detail/matrix_view.hpp)

# --- shared compile settings -------------------------------------------------

add_library(autonne_warnings INTERFACE)
if(CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang)$")
  target_compile_options(autonne_warnings INTERFACE -Wall -Wextra -Werror)
elseif(MSVC)
  target_compile_options(autonne_warnings INTERFACE /W4 /WX)
endif()

# The two floating-point modes the test suite is built under. Both must pass:
# the finiteness guards exist precisely because -ffast-math lets a compiler
# assume NaN and infinity away, and a guard that only survives strict mode is
# not a guard.
if(CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang)$")
  set(AUTONNE_STRICT_FP_FLAGS -fno-fast-math)
  set(AUTONNE_FAST_FP_FLAGS -ffast-math)
elseif(MSVC)
  set(AUTONNE_STRICT_FP_FLAGS /fp:strict)
  set(AUTONNE_FAST_FP_FLAGS /fp:fast)
else()
  set(AUTONNE_STRICT_FP_FLAGS "")
  set(AUTONNE_FAST_FP_FLAGS "")
  message(WARNING
    "Unrecognised compiler '${CMAKE_CXX_COMPILER_ID}': the strict and "
    "fast-math test variants will be built identically.")
endif()

# --- library -----------------------------------------------------------------

add_library(autonne STATIC ${AUTONNE_SOURCES} ${AUTONNE_HEADERS})
add_library(autonne::autonne ALIAS autonne)

target_include_directories(autonne
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_compile_features(autonne PUBLIC cxx_std_23)
target_link_libraries(autonne PRIVATE autonne_warnings)
target_compile_options(autonne PRIVATE ${AUTONNE_STRICT_FP_FLAGS})

set_target_properties(autonne PROPERTIES
  CXX_EXTENSIONS OFF
  POSITION_INDEPENDENT_CODE ON)

# --- tests -------------------------------------------------------------------

if(AUTONNE_BUILD_TESTS)
  include(FetchContent)

  FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.17.0
    GIT_SHALLOW TRUE)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  # GoogleTest is not built under -Werror; its headers are not ours to fix.
  if(NOT CMAKE_VERSION VERSION_LESS 3.25)
    foreach(gtest_target gtest gtest_main)
      if(TARGET ${gtest_target})
        set_target_properties(${gtest_target} PROPERTIES SYSTEM TRUE)
      endif()
    endforeach()
  endif()

  include(GoogleTest)
  enable_testing()

  set(AUTONNE_TEST_SOURCES
    tests/test_fp_bad.cpp
    tests/test_matrix_view.cpp
    tests/test_verify_svd.cpp
    tests/test_verify_eigh.cpp
    tests/test_hexfloat.cpp
    tests/test_api_stubs.cpp
    tests/test_build_mode.cpp)

  # nm backs the linkage test below. Without it that check is skipped rather
  # than silently passing.
  find_program(AUTONNE_NM_EXECUTABLE NAMES nm llvm-nm)
  if(NOT AUTONNE_NM_EXECUTABLE)
    message(WARNING
      "nm not found: the symbol linkage tests will not be registered, so "
      "nothing will notice if a harness entry point regains vague linkage.")
  endif()

  # Each variant compiles the library sources into its own binary rather than
  # linking the library target. The harness measures floating point, so the
  # flags under test have to reach every translation unit in the binary, not
  # just the test files -- src/verify.cpp above all, which is where check_svd
  # and check_eigh now live.
  function(autonne_add_test_variant variant)
    cmake_parse_arguments(ARG "" "" "FLAGS;DEFINES" ${ARGN})
    set(target autonne_tests_${variant})

    add_executable(${target} ${AUTONNE_TEST_SOURCES} ${AUTONNE_SOURCES})
    target_include_directories(${target} PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/include
      ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    target_link_libraries(${target} PRIVATE GTest::gtest_main autonne_warnings)
    target_compile_options(${target} PRIVATE ${ARG_FLAGS})
    target_compile_definitions(${target} PRIVATE ${ARG_DEFINES})
    set_target_properties(${target} PROPERTIES CXX_EXTENSIONS OFF)

    gtest_discover_tests(${target}
      TEST_PREFIX "${variant}."
      DISCOVERY_TIMEOUT 60)

    # Not a GoogleTest case: it inspects the linked binary from outside, which
    # is the only vantage point from which "exactly one strong definition" is
    # observable at all.
    if(AUTONNE_NM_EXECUTABLE AND CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang)$")
      add_test(NAME ${variant}.SymbolLinkage
        COMMAND ${CMAKE_COMMAND}
          -DAUTONNE_NM=${AUTONNE_NM_EXECUTABLE}
          -DAUTONNE_BINARY=$<TARGET_FILE:${target}>
          -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_symbol_linkage.cmake)
    endif()
  endfunction()

  autonne_add_test_variant(strict
    FLAGS ${AUTONNE_STRICT_FP_FLAGS}
    DEFINES AUTONNE_TEST_FAST_MATH=0)

  autonne_add_test_variant(fastmath
    FLAGS ${AUTONNE_FAST_FP_FLAGS}
    DEFINES AUTONNE_TEST_FAST_MATH=1)
endif()
