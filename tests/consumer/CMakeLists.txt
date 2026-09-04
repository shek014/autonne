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

# A stand-alone consumer of an installed autonne, built by CI against the
# result of `cmake --install`. It exists to prove the package config: that
# find_package(autonne) works, that autonne::autonne carries its include
# directory and C++23 requirement, and that a consumer compiled under
# -ffast-math still gets strict kernels and a strict harness.

cmake_minimum_required(VERSION 3.20)
project(autonne_consumer LANGUAGES CXX)

find_package(autonne CONFIG REQUIRED)

add_executable(autonne_consumer main.cpp)
target_link_libraries(autonne_consumer PRIVATE autonne::autonne)
if(CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang)$")
  target_compile_options(autonne_consumer PRIVATE -ffast-math -Wall -Wextra -Werror)
endif()
