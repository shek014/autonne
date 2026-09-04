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

// Interface-level contract, independent of which kernels are implemented.
//
// eigh has no implementation yet, so its failure report is exercised directly
// here. svd_thin does, and its rejection paths (null arguments, non-positive
// dimensions, non-finite input) are covered in test_svd_jacobi.cpp alongside
// the factorisations they guard.
//
// What stays interesting at this level for both: failure travels by return
// value and never by exception, and the signatures have not drifted from the
// documented interface.

#include <gtest/gtest.h>

#include <complex>
#include <type_traits>
#include <vector>

#include "autonne/autonne.hpp"
#include "test_support.hpp"

namespace {

using autonne::MatrixOrder;
using autonne_test::Complex;
using autonne_test::make_eigh_case;
using autonne_test::make_svd_case;

constexpr Complex kSentinel(-12345.0, 6789.0);
constexpr double kSentinelReal = -98765.0;

TEST(ApiStubs, EighReportsFailure) {
  const auto c = make_eigh_case(4, {1.0, 2.0, 3.0, 4.0}, MatrixOrder::ColMajor, 9);
  std::vector<double> evals(4, kSentinelReal);
  std::vector<Complex> evecs(16, kSentinel);

  EXPECT_FALSE(
      autonne::eigh(c.a.data(), c.n, c.order, evals.data(), evecs.data()));
}

// On a false return the outputs are unspecified and the caller must not read
// them. Both entry points go further and touch nothing at all, which is what
// makes a sentinel check possible.
//
// This RECORDS that behaviour rather than requiring it. An implementation that
// wrote a partial result before deciding to fail would still honour the
// documented contract, and this test would be the thing to change rather than
// the evidence of a defect.
TEST(ApiStubs, FailureLeavesOutputBuffersUntouched) {
  // A non-finite entry, so this is a real rejection inside the kernel rather
  // than an argument check that returns before looking at the matrix.
  const auto c = make_svd_case(4, 3, {4.0, 2.0, 1.0}, MatrixOrder::ColMajor, 10);
  std::vector<Complex> m = c.m;
  autonne_test::poke_bits(m[3], autonne_test::kQuietNanBits,
                          autonne_test::bits_of(0.0));

  std::vector<Complex> u(12, kSentinel);
  std::vector<double> s(3, kSentinelReal);
  std::vector<Complex> v(9, kSentinel);

  ASSERT_FALSE(autonne::svd_thin(m.data(), c.rows, c.cols, c.order, u.data(),
                                 s.data(), v.data()));

  for (const Complex& z : u) EXPECT_EQ(z, kSentinel);
  for (const double x : s) EXPECT_EQ(x, kSentinelReal);
  for (const Complex& z : v) EXPECT_EQ(z, kSentinel);

  // eigh has no implementation, so every call is the failure path.
  const auto e =
      make_eigh_case(4, {1.0, 2.0, 3.0, 4.0}, MatrixOrder::ColMajor, 11);
  std::vector<double> evals(4, kSentinelReal);
  std::vector<Complex> evecs(16, kSentinel);

  ASSERT_FALSE(
      autonne::eigh(e.a.data(), e.n, e.order, evals.data(), evecs.data()));

  for (const double x : evals) EXPECT_EQ(x, kSentinelReal);
  for (const Complex& z : evecs) EXPECT_EQ(z, kSentinel);
}

// Failure is signalled by the return value only. Degenerate shapes are the
// most likely place for an implementation to reach for an exception instead.
TEST(ApiStubs, NeverThrows) {
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
