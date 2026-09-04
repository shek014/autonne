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
option(AUTONNE_BUILD_BENCHMARKS "Build the benchmark against Eigen (fetches Eigen)" OFF)
set(AUTONNE_SANITIZERS "" CACHE STRING
  "Sanitizers to build everything with, e.g. address,undefined")
option(AUTONNE_INSTALL "Generate install and package-config rules" ${AUTONNE_IS_TOP_LEVEL})

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# autonne depends on no linear algebra library. Eigen is fetched only when
# AUTONNE_BUILD_BENCHMARKS is on, as the baseline inside the benchmark target,
# and must never become a dependency of the library or of the test suite.

# verify.cpp is a source, not a header, so that check_svd and check_eigh have
# exactly one definition per binary, built under this file's floating-point
# flags. See the note at the top of include/autonne/verify.hpp.
set(AUTONNE_SOURCES
  src/svd.cpp
  src/eigh.cpp
  src/verify.cpp)

set(AUTONNE_HEADERS
  include/autonne/autonne.hpp
  include/autonne/verify.hpp
  include/autonne/hexfloat.hpp
  include/autonne/detail/fp_bits.hpp
  include/autonne/detail/matrix_view.hpp)

# Private to the kernels; included relative to src/ and never installed.
set(AUTONNE_PRIVATE_HEADERS
  src/detail/kernel_common.hpp)

# --- shared compile settings -------------------------------------------------

# The warning set, as a variable rather than an interface target: the library
# is exported, and a private link to an interface target would drag that
# target into the export set.
if(CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang)$")
  set(AUTONNE_WARNING_FLAGS -Wall -Wextra -Werror)
elseif(MSVC)
  # /Zc:__cplusplus makes MSVC report the language standard it is actually
  # compiling, which the build-mode test reads.
  set(AUTONNE_WARNING_FLAGS /W4 /WX /Zc:__cplusplus)
else()
  set(AUTONNE_WARNING_FLAGS "")
endif()

add_library(autonne_warnings INTERFACE)
target_compile_options(autonne_warnings INTERFACE ${AUTONNE_WARNING_FLAGS})

# Sanitizers apply to every target, tests included. -fno-sanitize-recover
# makes a report fatal: a kernel that reads out of bounds or shifts by a
# negative amount should fail the run, not annotate it.
if(AUTONNE_SANITIZERS)
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang)$")
    message(FATAL_ERROR "AUTONNE_SANITIZERS needs GCC or Clang")
  endif()
  add_compile_options(-fsanitize=${AUTONNE_SANITIZERS}
    -fno-sanitize-recover=all -fno-omit-frame-pointer -g)
  add_link_options(-fsanitize=${AUTONNE_SANITIZERS})
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

add_library(autonne STATIC ${AUTONNE_SOURCES} ${AUTONNE_HEADERS} ${AUTONNE_PRIVATE_HEADERS})
add_library(autonne::autonne ALIAS autonne)

target_include_directories(autonne
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_compile_features(autonne PUBLIC cxx_std_23)
target_compile_options(autonne PRIVATE ${AUTONNE_WARNING_FLAGS} ${AUTONNE_STRICT_FP_FLAGS})

set_target_properties(autonne PROPERTIES
  CXX_EXTENSIONS OFF
  POSITION_INDEPENDENT_CODE ON)

# --- install -----------------------------------------------------------------

if(AUTONNE_INSTALL)
  include(GNUInstallDirs)
  include(CMakePackageConfigHelpers)

  install(TARGETS autonne EXPORT autonneTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  install(DIRECTORY include/autonne DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

  set(AUTONNE_CMAKE_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/autonne)
  install(EXPORT autonneTargets
    NAMESPACE autonne::
    FILE autonneTargets.cmake
    DESTINATION ${AUTONNE_CMAKE_DIR})

  configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/autonneConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/autonneConfig.cmake
    INSTALL_DESTINATION ${AUTONNE_CMAKE_DIR})
  write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/autonneConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)
  install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/autonneConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/autonneConfigVersion.cmake
    DESTINATION ${AUTONNE_CMAKE_DIR})
endif()

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
    tests/test_api.cpp
    tests/test_svd.cpp
    tests/test_eigh.cpp
    tests/test_corpus.cpp
    tests/test_stress.cpp
    tests/test_build_mode.cpp)

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
    target_compile_definitions(${target} PRIVATE ${ARG_DEFINES}
      AUTONNE_CORPUS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/corpus")
    set_target_properties(${target} PROPERTIES CXX_EXTENSIONS OFF)

    gtest_discover_tests(${target}
      TEST_PREFIX "${variant}."
      DISCOVERY_TIMEOUT 60)
  endfunction()

  autonne_add_test_variant(strict
    FLAGS ${AUTONNE_STRICT_FP_FLAGS}
    DEFINES AUTONNE_TEST_FAST_MATH=0)

  autonne_add_test_variant(fastmath
    FLAGS ${AUTONNE_FAST_FP_FLAGS}
    DEFINES AUTONNE_TEST_FAST_MATH=1)

  # The hand-rolled accessor in matrix_view.hpp is dead code wherever <mdspan>
  # exists, and dead code rots. This variant forces it live; floating-point
  # mode is irrelevant to an index calculation, so strict is enough.
  autonne_add_test_variant(strict_nomdspan
    FLAGS ${AUTONNE_STRICT_FP_FLAGS}
    DEFINES AUTONNE_TEST_FAST_MATH=0 AUTONNE_NO_STD_MDSPAN=1)

  # --- cross-binary floating-point comparison --------------------------------
  #
  # The two floating-point models are compared ACROSS executables, never
  # inside one. Two variants linked into a single binary do not measure two
  # floating-point models: the inline helpers have vague linkage, the linker
  # keeps one copy of each without regard to the flags it was built under, and
  # the comparison would be reading link order. So the kernel is built into
  # one small executable per model, both are run on the same frozen corpus
  # matrix, and their output files are diffed by a third program.

  function(autonne_add_dump_variant variant)
    cmake_parse_arguments(ARG "" "" "FLAGS" ${ARGN})
    set(target autonne_dump_${variant})
    add_executable(${target} tools/dump_factors.cpp ${AUTONNE_SOURCES})
    target_include_directories(${target} PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/include
      ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_compile_options(${target} PRIVATE ${AUTONNE_WARNING_FLAGS} ${ARG_FLAGS})
    set_target_properties(${target} PROPERTIES CXX_EXTENSIONS OFF)
  endfunction()

  autonne_add_dump_variant(strict FLAGS ${AUTONNE_STRICT_FP_FLAGS})
  autonne_add_dump_variant(fastmath FLAGS ${AUTONNE_FAST_FP_FLAGS})

  add_executable(autonne_compare tools/compare_factors.cpp)
  target_include_directories(autonne_compare PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_compile_options(autonne_compare PRIVATE
    ${AUTONNE_WARNING_FLAGS} ${AUTONNE_STRICT_FP_FLAGS})
  set_target_properties(autonne_compare PROPERTIES CXX_EXTENSIONS OFF)

  # Every corpus matrix, through both models. The eigh cases are square by
  # construction; the svd ones need not be.
  file(GLOB AUTONNE_CORPUS_SVD
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/tests/corpus
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/corpus/svd_*.txt)
  file(GLOB AUTONNE_CORPUS_EIGH
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/tests/corpus
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/corpus/eigh_*.txt)

  foreach(kind svd eigh)
    if(kind STREQUAL "svd")
      set(cases ${AUTONNE_CORPUS_SVD})
    else()
      set(cases ${AUTONNE_CORPUS_EIGH})
    endif()
    foreach(case_file ${cases})
      get_filename_component(case_name ${case_file} NAME_WE)
      add_test(NAME fpcrosscheck.${case_name}
        COMMAND ${CMAKE_COMMAND}
          -DAUTONNE_DUMP_STRICT=$<TARGET_FILE:autonne_dump_strict>
          -DAUTONNE_DUMP_FASTMATH=$<TARGET_FILE:autonne_dump_fastmath>
          -DAUTONNE_COMPARE=$<TARGET_FILE:autonne_compare>
          -DAUTONNE_INPUT=${CMAKE_CURRENT_SOURCE_DIR}/tests/corpus/${case_file}
          -DAUTONNE_KIND=${kind}
          -DAUTONNE_WORK_DIR=${CMAKE_CURRENT_BINARY_DIR}/fpcrosscheck
          -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_fp_crosscheck.cmake)
    endforeach()
  endforeach()
endif()

# --- benchmark ---------------------------------------------------------------

if(AUTONNE_BUILD_BENCHMARKS)
  add_subdirectory(bench)
endif()
