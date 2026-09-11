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

# The benchmark is the only place Eigen is allowed to appear. It is fetched
# here, at the pinned release the spec measured against, and linked to this
# one executable.

include(FetchContent)

# Eigen is header-only, and its own CMakeLists.txt is not used: it probes the
# platform in ways that fail on some toolchains (clang targeting MSVC cannot
# "link to the standard math library" by its test) and would also register
# targets this build has no use for. Pointing SOURCE_SUBDIR at a directory
# that does not exist makes FetchContent populate the sources without adding
# them as a subdirectory; the include path is all that is needed.
FetchContent_Declare(eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0
  GIT_SHALLOW TRUE
  SOURCE_SUBDIR cmake-not-used)
FetchContent_MakeAvailable(eigen)

add_executable(autonne_bench bench.cpp)
target_link_libraries(autonne_bench PRIVATE autonne::autonne)
target_include_directories(autonne_bench PRIVATE ${CMAKE_SOURCE_DIR}/tests)
# Eigen's headers are not ours to warn about.
target_include_directories(autonne_bench SYSTEM PRIVATE ${eigen_SOURCE_DIR})
target_compile_options(autonne_bench PRIVATE ${AUTONNE_WARNING_FLAGS} ${AUTONNE_STRICT_FP_FLAGS})
set_target_properties(autonne_bench PROPERTIES CXX_EXTENSIONS OFF)
