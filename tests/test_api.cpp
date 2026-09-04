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

// The contract around the kernels: failure is reported by return value only,
// nothing throws, and a false return leaves the caller's output buffers
// exactly as they were. The numerics are tested in test_svd.cpp and
// test_eigh.cpp; this file is about the boundary.

#include <gtest/gtest.h>

#include <complex>
#include <type_traits>
#include <vector>

#include "autonne/autonne.hpp"
#include "test_support.hpp"

namespace {

using autonne::MatrixOrder;
using autonne_test::Complex;
using autonne_test::bits_of;
using autonne_test::kNegativeInfBits;
using autonne_test::kQuietNanBits;
using autonne_test::kSentinel;
using autonne_test::kSentinelReal;
using autonne_test::make_eigh_case;
using autonne_test::make_svd_case;
using autonne_test::poke_bits;
using autonne_test::run_eigh;
using autonne_test::run_svd;

// --- success on well-formed input ---------------------------------------------

TEST(Api, SvdThinSucceedsOnValidInput) {
  const auto c = make_svd_case(4, 3, {4.0, 2.0, 1.0}, MatrixOrder::ColMajor, 8);
  EXPECT_TRUE(run_svd(c).ok);
}

TEST(Api, EighSucceedsOnValidInput) {
  const auto c = make_eigh_case(4, {1.0, 2.0, 3.0, 4.0}, MatrixOrder::ColMajor, 9);
  EXPECT_TRUE(run_eigh(c).ok);
}

// The lindblad seam wants the spectrum alone in two of its three eigh call
// sites, so a null eigenvector buffer is a request, not an error.
TEST(Api, EighAcceptsNullEigenvectorBuffer) {
  const auto c = make_eigh_case(4, {1.0, 2.0, 3.0, 4.0}, MatrixOrder::ColMajor, 9);
  std::vector<double> evals(4, kSentinelReal);
  EXPECT_TRUE(autonne::eigh(c.a.data(), c.n, c.order, evals.data(), nullptr));
  for (const double x : evals) EXPECT_NE(x, kSentinelReal);
}

// --- rejection of malformed arguments --------------------------------------

TEST(Api, SvdThinRejectsMalformedArguments) {
  const auto c = make_svd_case(4, 3, {4.0, 2.0, 1.0}, MatrixOrder::ColMajor, 10);
  std::vector<Complex> u(12, kSentinel);
  std::vector<double> s(3, kSentinelReal);
  std::vector<Complex> v(9, kSentinel);

  EXPECT_FALSE(autonne::svd_thin(nullptr, 4, 3, c.order, u.data(), s.data(), v.data()));
  EXPECT_FALSE(autonne::svd_thin(c.m.data(), 0, 3, c.order, u.data(), s.data(), v.data()));
  EXPECT_FALSE(autonne::svd_thin(c.m.data(), 4, 0, c.order, u.data(), s.data(), v.data()));
  EXPECT_FALSE(autonne::svd_thin(c.m.data(), -4, 3, c.order, u.data(), s.data(), v.data()));
  EXPECT_FALSE(autonne::svd_thin(c.m.data(), 4, -3, c.order, u.data(), s.data(), v.data()));
  EXPECT_FALSE(autonne::svd_thin(c.m.data(), 4, 3, c.order, nullptr, s.data(), v.data()));
  EXPECT_FALSE(autonne::svd_thin(c.m.data(), 4, 3, c.order, u.data(), nullptr, v.data()));
  EXPECT_FALSE(autonne::svd_thin(c.m.data(), 4, 3, c.order, u.data(), s.data(), nullptr));

  for (const Complex& z : u) EXPECT_EQ(z, kSentinel);
  for (const double x : s) EXPECT_EQ(x, kSentinelReal);
  for (const Complex& z : v) EXPECT_EQ(z, kSentinel);
}

TEST(Api, EighRejectsMalformedArguments) {
  const auto c = make_eigh_case(4, {1.0, 2.0, 3.0, 4.0}, MatrixOrder::ColMajor, 11);
  std::vector<double> evals(4, kSentinelReal);
  std::vector<Complex> evecs(16, kSentinel);

  EXPECT_FALSE(autonne::eigh(nullptr, 4, c.order, evals.data(), evecs.data()));
  EXPECT_FALSE(autonne::eigh(c.a.data(), 0, c.order, evals.data(), evecs.data()));
  EXPECT_FALSE(autonne::eigh(c.a.data(), -1, c.order, evals.data(), evecs.data()));
  EXPECT_FALSE(autonne::eigh(c.a.data(), 4, c.order, nullptr, evecs.data()));

  for (const double x : evals) EXPECT_EQ(x, kSentinelReal);
  for (const Complex& z : evecs) EXPECT_EQ(z, kSentinel);
}

// A non-finite entry anywhere in the input is a refusal, not a propagated NaN
// in the factors. The bad value is written as bits: see test_support.hpp.
TEST(Api, SvdThinRejectsNonFiniteInputAndLeavesOutputsUntouched) {
  auto c = make_svd_case(5, 4, {4.0, 3.0, 2.0, 1.0}, MatrixOrder::RowMajor, 12);
  poke_bits(c.m[7], kQuietNanBits, bits_of(0.25));
  const auto r = run_svd(c);
  EXPECT_FALSE(r.ok);
  for (const Complex& z : r.u) EXPECT_EQ(z, kSentinel);
  for (const double x : r.s) EXPECT_EQ(x, kSentinelReal);
  for (const Complex& z : r.v) EXPECT_EQ(z, kSentinel);

  auto d = make_svd_case(5, 4, {4.0, 3.0, 2.0, 1.0}, MatrixOrder::ColMajor, 13);
  poke_bits(d.m[19], bits_of(0.5), kNegativeInfBits);
  EXPECT_FALSE(run_svd(d).ok);
}

TEST(Api, EighRejectsNonFiniteInputAndLeavesOutputsUntouched) {
  auto c = make_eigh_case(4, {-3.0, -0.5, 1.0, 4.0}, MatrixOrder::ColMajor, 14);
  poke_bits(c.a[9], kQuietNanBits, bits_of(0.0));
  const auto r = run_eigh(c);
  EXPECT_FALSE(r.ok);
  for (const double x : r.evals) EXPECT_EQ(x, kSentinelReal);
  for (const Complex& z : r.evecs) EXPECT_EQ(z, kSentinel);
}

// Failure is signalled by the return value only. Degenerate shapes are the
// most likely place for an implementation to reach for an exception instead.
TEST(Api, NeverThrows) {
  std::vector<Complex> m(1, Complex(1.0, 0.0));
  std::vector<Complex> u(1, kSentinel);
  std::vector<double> s(1, kSentinelReal);
  std::vector<Complex> v(1, kSentinel);

  EXPECT_NO_THROW({
    (void)autonne::svd_thin(m.data(), 1, 1, MatrixOrder::ColMajor, u.data(),
                            s.data(), v.data());
    (void)autonne::svd_thin(nullptr, 0, 0, MatrixOrder::RowMajor, nullptr,
                            nullptr, nullptr);
    (void)autonne::svd_thin(m.data(), -1, -1, MatrixOrder::ColMajor, u.data(),
                            s.data(), v.data());
    (void)autonne::eigh(m.data(), 1, MatrixOrder::ColMajor, s.data(), u.data());
    (void)autonne::eigh(nullptr, 0, MatrixOrder::RowMajor, nullptr, nullptr);
    (void)autonne::eigh(m.data(), -1, MatrixOrder::ColMajor, s.data(), u.data());
  });
}

// Compile-time surface checks: exact signatures, and no exception escape route
// hidden behind a noexcept mismatch.
static_assert(std::is_same_v<decltype(autonne::svd_thin),
                             bool(const std::complex<double>*, int, int,
                                  MatrixOrder, std::complex<double>*, double*,
                                  std::complex<double>*)>,
              "svd_thin signature drifted from the documented interface");

static_assert(std::is_same_v<decltype(autonne::eigh),
                             bool(const std::complex<double>*, int, MatrixOrder,
                                  double*, std::complex<double>*)>,
              "eigh signature drifted from the documented interface");

}  // namespace
