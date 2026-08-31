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

// The critical file. These assertions are compiled into both the strict and
// the fast-math test binary; a fp_bad that passes under -fno-fast-math and
// fails under -ffast-math means the guard technique does not work and every
// finiteness check in the harness is decoration.

#include <gtest/gtest.h>

#include <cmath>
#include <complex>

#include "autonne/verify.hpp"
#include "test_support.hpp"

namespace {

using autonne::verify::fp_bad;
using autonne_test::largest_normal;
using autonne_test::negative_inf;
using autonne_test::opaque;
using autonne_test::positive_inf;
using autonne_test::quiet_nan;
using autonne_test::signaling_nan;
using autonne_test::smallest_subnormal;

TEST(FpBad, DetectsNan) {
  EXPECT_TRUE(fp_bad(opaque(quiet_nan())));
  EXPECT_TRUE(fp_bad(opaque(signaling_nan())));
  EXPECT_TRUE(fp_bad(opaque(-quiet_nan())));
}

TEST(FpBad, DetectsInfinities) {
  EXPECT_TRUE(fp_bad(opaque(positive_inf())));
  EXPECT_TRUE(fp_bad(opaque(negative_inf())));
}

TEST(FpBad, AcceptsFiniteValues) {
  EXPECT_FALSE(fp_bad(opaque(0.0)));
  EXPECT_FALSE(fp_bad(opaque(-0.0)));
  EXPECT_FALSE(fp_bad(opaque(1.0)));
  EXPECT_FALSE(fp_bad(opaque(-3.5)));
  EXPECT_FALSE(fp_bad(opaque(smallest_subnormal())));
  EXPECT_FALSE(fp_bad(opaque(largest_normal())));
  EXPECT_FALSE(fp_bad(opaque(-largest_normal())));
}

// The exponent field is all ones for NaN and infinity and nothing else, so the
// boundary is one ulp below the largest finite value.
TEST(FpBad, BoundaryIsExact) {
  EXPECT_FALSE(fp_bad(opaque(autonne_test::bits_to_double(UINT64_C(0x7FEFFFFFFFFFFFFF)))));
  EXPECT_TRUE(fp_bad(opaque(autonne_test::bits_to_double(UINT64_C(0x7FF0000000000000)))));
  EXPECT_FALSE(fp_bad(opaque(autonne_test::bits_to_double(UINT64_C(0xFFEFFFFFFFFFFFFF)))));
  EXPECT_TRUE(fp_bad(opaque(autonne_test::bits_to_double(UINT64_C(0xFFF0000000000000)))));
}

TEST(FpBad, ComplexOverloadChecksBothParts) {
  const std::complex<double> good(1.0, -2.0);
  const std::complex<double> bad_real(opaque(quiet_nan()), -2.0);
  const std::complex<double> bad_imag(1.0, opaque(positive_inf()));
  EXPECT_FALSE(fp_bad(good));
  EXPECT_TRUE(fp_bad(bad_real));
  EXPECT_TRUE(fp_bad(bad_imag));
}

// A NaN that arrives from arithmetic rather than from a literal bit pattern.
//
// Under strict floating point inf - inf is a NaN and the guard must catch it.
// Under -ffast-math the compiler is entitled to fold the subtraction to zero
// before a NaN ever exists, and GCC does exactly that here -- so there is no
// non-finite value for any guard to find. That is a fact about the
// arithmetic, not about fp_bad; the bit-pattern tests above are what pin the
// guard down, and those hold in both builds.
TEST(FpBad, DetectsNanFromArithmetic) {
  const double inf = opaque(positive_inf());
  const double produced = opaque(inf - inf);
  RecordProperty("subtraction_produced_non_finite", fp_bad(produced) ? 1 : 0);
#if AUTONNE_TEST_FAST_MATH
  GTEST_SKIP() << "-ffast-math may fold inf - inf before a NaN is ever "
                  "materialised; nothing reaches the guard to be detected";
#else
  EXPECT_TRUE(fp_bad(produced));
#endif
}

// Records, without asserting, what the standard predicate does in this build.
// Under -ffast-math std::isnan is entitled to answer false for a genuine NaN;
// the point of fp_bad is that it is not.
TEST(FpBad, StdPredicateBehaviourIsRecorded) {
  const double nan_value = opaque(quiet_nan());
  RecordProperty("std_isnan_on_nan", std::isnan(nan_value) ? 1 : 0);
  RecordProperty("fp_bad_on_nan", fp_bad(nan_value) ? 1 : 0);
  EXPECT_TRUE(fp_bad(nan_value));
}

}  // namespace
