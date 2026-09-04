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
//
// Every non-finite value here is written into memory as an integer bit pattern
// and reaches fp_bad by reference. That is not a convenience: under
// -ffinite-math-only a NaN that crosses a function boundary by value is
// assumed away, and Clang 22 folds a guard on such a value to false. The
// guard's contract is therefore "detects a non-finite object representation
// in memory", which is exactly what a kernel scanning a caller's buffer needs.

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <vector>

#include "autonne/verify.hpp"
#include "test_support.hpp"

namespace {

using autonne::verify::fp_bad;
using autonne_test::Complex;
using autonne_test::Slot;
using autonne_test::bits_of;
using autonne_test::kLargestNormalBits;
using autonne_test::kNegativeInfBits;
using autonne_test::kPositiveInfBits;
using autonne_test::kQuietNanBits;
using autonne_test::kSignalingNanBits;
using autonne_test::kSmallestSubnormalBits;
using autonne_test::opaque;
using autonne_test::poke_bits;

TEST(FpBad, DetectsNan) {
  EXPECT_TRUE(fp_bad(Slot(kQuietNanBits).get()));
  EXPECT_TRUE(fp_bad(Slot(kSignalingNanBits).get()));
  EXPECT_TRUE(fp_bad(Slot(kQuietNanBits | UINT64_C(0x8000000000000000)).get()));
}

TEST(FpBad, DetectsInfinities) {
  EXPECT_TRUE(fp_bad(Slot(kPositiveInfBits).get()));
  EXPECT_TRUE(fp_bad(Slot(kNegativeInfBits).get()));
}

TEST(FpBad, AcceptsFiniteValues) {
  EXPECT_FALSE(fp_bad(opaque(0.0)));
  EXPECT_FALSE(fp_bad(opaque(-0.0)));
  EXPECT_FALSE(fp_bad(opaque(1.0)));
  EXPECT_FALSE(fp_bad(opaque(-3.5)));
  EXPECT_FALSE(fp_bad(Slot(kSmallestSubnormalBits).get()));
  EXPECT_FALSE(fp_bad(Slot(kLargestNormalBits).get()));
  EXPECT_FALSE(fp_bad(Slot(kLargestNormalBits | UINT64_C(0x8000000000000000)).get()));
}

// The exponent field is all ones for NaN and infinity and nothing else, so the
// boundary is one ulp below the largest finite value.
TEST(FpBad, BoundaryIsExact) {
  EXPECT_FALSE(fp_bad(Slot(UINT64_C(0x7FEFFFFFFFFFFFFF)).get()));
  EXPECT_TRUE(fp_bad(Slot(UINT64_C(0x7FF0000000000000)).get()));
  EXPECT_FALSE(fp_bad(Slot(UINT64_C(0xFFEFFFFFFFFFFFFF)).get()));
  EXPECT_TRUE(fp_bad(Slot(UINT64_C(0xFFF0000000000000)).get()));
}

// The complex overload must read the object representation directly. Going
// through real() and imag() would hand each part to the guard by value, which
// is the boundary that -ffinite-math-only assumes finite.
TEST(FpBad, ComplexOverloadChecksBothParts) {
  std::vector<Complex> z(3, Complex(1.0, -2.0));
  poke_bits(z[1], kQuietNanBits, bits_of(-2.0));
  poke_bits(z[2], bits_of(1.0), kPositiveInfBits);
  EXPECT_FALSE(fp_bad(z[0]));
  EXPECT_TRUE(fp_bad(z[1]));
  EXPECT_TRUE(fp_bad(z[2]));
}

// A buffer scan, the shape in which the kernel and the harness actually use
// the guard: a caller's array with one bad element somewhere in the middle.
TEST(FpBad, FindsOneBadElementInABuffer) {
  std::vector<double> buffer(64, 0.25);
  poke_bits(buffer[37], kNegativeInfBits);
  int found = 0;
  for (const double& x : buffer) {
    if (fp_bad(x)) ++found;
  }
  EXPECT_EQ(found, 1);

  std::vector<Complex> cbuffer(64, Complex(0.25, -0.75));
  poke_bits(cbuffer[5], bits_of(0.25), kSignalingNanBits);
  found = 0;
  for (const Complex& z : cbuffer) {
    if (fp_bad(z)) ++found;
  }
  EXPECT_EQ(found, 1);
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
  const Slot inf(kPositiveInfBits);
  std::vector<double> produced(1, 0.0);
  produced[0] = inf.get() - inf.get();
  RecordProperty("subtraction_produced_non_finite", fp_bad(produced[0]) ? 1 : 0);
#if AUTONNE_TEST_FAST_MATH
  GTEST_SKIP() << "-ffast-math may fold inf - inf before a NaN is ever "
                  "materialised; nothing reaches the guard to be detected";
#else
  EXPECT_TRUE(fp_bad(produced[0]));
#endif
}

// Records, without asserting, what the standard predicate does in this build.
// Under -ffast-math std::isnan is entitled to answer false for a genuine NaN;
// the point of fp_bad is that it is not.
TEST(FpBad, StdPredicateBehaviourIsRecorded) {
  const Slot nan_slot(kQuietNanBits);
  const double& nan_value = nan_slot.get();
  // The warning group is recent; AppleClang does not have it, and naming an
  // unknown group is itself an error under -Werror.
  #if defined(__clang__) && defined(__has_warning)
  #  if __has_warning("-Wnan-infinity-disabled")
  #    define AUTONNE_SUPPRESS_NAN_USE 1
  #  endif
  #endif
  #if defined(AUTONNE_SUPPRESS_NAN_USE)
  #  pragma clang diagnostic push
  #  pragma clang diagnostic ignored "-Wnan-infinity-disabled"
  #endif
    // Deliberate: this test records what std::isnan does when the compiler is
    // permitted to assume NaN cannot occur. Clang correctly warns that using a
    // NaN here is UB under -ffast-math — that is precisely what is being
    // measured, and the same diagnostic Eigen's deleted guard produces.
    RecordProperty("std_isnan_on_nan", std::isnan(nan_value) ? 1 : 0);
  #if defined(AUTONNE_SUPPRESS_NAN_USE)
  #  pragma clang diagnostic pop
  #  undef AUTONNE_SUPPRESS_NAN_USE
  #endif
  RecordProperty("fp_bad_on_nan", fp_bad(nan_value) ? 1 : 0);
  EXPECT_TRUE(fp_bad(nan_value));
}

// Records what happens when a NaN crosses a function boundary by value before
// reaching the guard. Under strict floating point it must still be detected.
// Under -ffast-math the compiler may assume the returned value is finite, and
// Clang 22 does: the guard then sees a value that is no longer a NaN. This is
// the reason the harness and the kernel only ever inspect memory by reference,
// and the reason this file never builds a NaN any other way.
TEST(FpBad, ByValueBoundaryBehaviourIsRecorded) {
  const Slot nan_slot(kQuietNanBits);
  const double through_boundary = opaque(nan_slot.get());
  RecordProperty("nan_survives_by_value_boundary", fp_bad(through_boundary) ? 1 : 0);
#if AUTONNE_TEST_FAST_MATH
  SUCCEED();
#else
  EXPECT_TRUE(fp_bad(through_boundary));
#endif
}

}  // namespace
