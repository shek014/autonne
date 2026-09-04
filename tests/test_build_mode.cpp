// Copyright 2026 The autonne Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Guards the premise of the whole two-variant arrangement.
//
// The fast-math assertions in test_fp_bad.cpp only mean something if the
// fast-math binary was in fact built with -ffast-math. Without this check a
// misconfigured target would compile both variants strictly, every test would
// pass, and the suite would report success while testing one mode twice.

#include <gtest/gtest.h>

#include "autonne/detail/matrix_view.hpp"

#ifndef AUTONNE_TEST_FAST_MATH
#error "AUTONNE_TEST_FAST_MATH must be defined by the build for each variant"
#endif

namespace {

TEST(BuildMode, VariantMatchesItsFloatingPointFlags) {
#if defined(__GNUC__) || defined(__clang__)
#if defined(__FAST_MATH__)
  constexpr bool compiled_fast = true;
#else
  constexpr bool compiled_fast = false;
#endif
#if AUTONNE_TEST_FAST_MATH
  EXPECT_TRUE(compiled_fast)
      << "the fastmath variant was built without -ffast-math, so the "
         "fp_bad assertions in this binary prove nothing";
#else
  EXPECT_FALSE(compiled_fast)
      << "the strict variant was built with -ffast-math";
#endif
#else
  GTEST_SKIP() << "no portable way to observe the fast-math flag on this "
                  "compiler";
#endif
}

// The library must build as C++23; this is what makes std::bit_cast and
// std::mdspan available to the headers above.
TEST(BuildMode, LanguageStandardIsCpp23) {
#if defined(_MSVC_LANG)
  // MSVC reports the standard in _MSVC_LANG unconditionally and in
  // __cplusplus only under /Zc:__cplusplus, which the build sets; check both.
  EXPECT_GE(_MSVC_LANG, 202100L);
#endif
  EXPECT_GE(__cplusplus, 202100L);
}

// Records which accessor path is compiled in. Both are supposed to behave
// identically, and test_matrix_view.cpp checks that they do, but a failure
// there reads very differently depending on which one is live.
TEST(BuildMode, MdspanAvailabilityIsRecorded) {
  RecordProperty("uses_std_mdspan", AUTONNE_HAS_STD_MDSPAN);
  SUCCEED();
}

// The hand-rolled accessor is compiled in a variant of its own. That variant
// only proves something if the request to bypass std::mdspan was honoured.
TEST(BuildMode, HandRolledAccessorVariantBypassesStdMdspan) {
#if defined(AUTONNE_NO_STD_MDSPAN)
  EXPECT_EQ(AUTONNE_HAS_STD_MDSPAN, 0)
      << "AUTONNE_NO_STD_MDSPAN is defined but the std::mdspan path is live";
#else
  GTEST_SKIP() << "this variant does not request the hand-rolled accessor";
#endif
}

}  // namespace
