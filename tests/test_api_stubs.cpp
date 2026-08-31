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

// The kernels are not written yet, so what is testable is the contract around
// them: they report failure by return value, they do not throw, and they leave
// the caller's output buffers alone. A caller written against this build is
// already on the interface the real kernel will honour.

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

TEST(ApiStubs, SvdThinReportsFailure) {
  const auto c = make_svd_case(4, 3, {4.0, 2.0, 1.0}, MatrixOrder::ColMajor, 8);
  std::vector<Complex> u(12, kSentinel);
  std::vector<double> s(3, kSentinelReal);
  std::vector<Complex> v(9, kSentinel);

  EXPECT_FALSE(autonne::svd_thin(c.m.data(), c.rows, c.cols, c.order, u.data(),
                                 s.data(), v.data()));
}

TEST(ApiStubs, EighReportsFailure) {
  const auto c = make_eigh_case(4, {1.0, 2.0, 3.0, 4.0}, MatrixOrder::ColMajor, 9);
  std::vector<double> evals(4, kSentinelReal);
  std::vector<Complex> evecs(16, kSentinel);

  EXPECT_FALSE(
      autonne::eigh(c.a.data(), c.n, c.order, evals.data(), evecs.data()));
}

// On a false return the outputs are unspecified and the caller must not read
// them. The stubs go further and touch nothing, which is what makes a
// sentinel-based check possible at all; when the kernels land this test
// documents the boundary rather than constraining it.
TEST(ApiStubs, FailureLeavesOutputBuffersUntouched) {
  const auto c = make_svd_case(4, 3, {4.0, 2.0, 1.0}, MatrixOrder::ColMajor, 10);
  std::vector<Complex> u(12, kSentinel);
  std::vector<double> s(3, kSentinelReal);
  std::vector<Complex> v(9, kSentinel);

  ASSERT_FALSE(autonne::svd_thin(c.m.data(), c.rows, c.cols, c.order, u.data(),
                                 s.data(), v.data()));

  for (const Complex& z : u) EXPECT_EQ(z, kSentinel);
  for (const double x : s) EXPECT_EQ(x, kSentinelReal);
  for (const Complex& z : v) EXPECT_EQ(z, kSentinel);
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
