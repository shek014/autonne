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

// Randomised sweeps.
//
// The other test files pin down named properties on chosen matrices. This
// one covers the space between them: every shape from 1 x 1 up, every rank
// from empty to full, spectra that are flat, graded, or clustered, and
// scalings that put the whole matrix near the ends of the exponent range.
// Each case is judged by the harness, which is the contract, and the seeds
// are fixed so a failure is reproducible from the case number alone.
//
// Kept to a few seconds per variant: the point is breadth, and the expensive
// shapes are covered by name in test_svd.cpp and test_eigh.cpp.

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "autonne/autonne.hpp"
#include "autonne/verify.hpp"
#include "report_matchers.hpp"
#include "test_support.hpp"

namespace {

using autonne::MatrixOrder;
using autonne_test::Complex;
using autonne_test::EighAccepted;
using autonne_test::EighResult;
using autonne_test::Lcg;
using autonne_test::SvdAccepted;
using autonne_test::SvdResult;
using autonne_test::make_eigh_case;
using autonne_test::make_svd_case;
using autonne_test::random_hermitian;
using autonne_test::random_matrix;
using autonne_test::run_eigh;
using autonne_test::run_svd;

autonne::verify::SvdReport judge(const std::vector<Complex>& m, int rows, int cols,
                                 MatrixOrder order, const SvdResult& r) {
  const int k = rows < cols ? rows : cols;
  return autonne::verify::check_svd(m.data(), rows, cols, order, r.u.data(),
                                    r.s.data(), r.v.data(), k);
}

// A spectrum of `k` values with `rank` of them nonzero, shaped by `style`.
std::vector<double> shaped_spectrum(int k, int rank, int style, Lcg& rng) {
  std::vector<double> s(static_cast<std::size_t>(k), 0.0);
  for (int i = 0; i < rank; ++i) {
    const std::size_t idx = static_cast<std::size_t>(i);
    switch (style) {
      case 0:  // flat: every value identical, the hardest case for ordering
        s[idx] = 1.0;
        break;
      case 1:  // geometric decay over twelve decades
        s[idx] = std::pow(10.0, -12.0 * static_cast<double>(i) /
                                    static_cast<double>(rank > 1 ? rank - 1 : 1));
        break;
      case 2:  // two clusters, exactly degenerate within each
        s[idx] = (i < rank / 2) ? 4.0 : 0.5;
        break;
      default: {  // random magnitudes over four decades
        const double u = 0.5 * (rng.next_uniform() + 1.0);
        s[idx] = std::pow(10.0, -4.0 * u);
        break;
      }
    }
  }
  return s;
}

TEST(Stress, SvdOverShapesRanksAndSpectra) {
  int cases = 0;
  for (int rows = 1; rows <= 9; ++rows) {
    for (int cols = 1; cols <= 9; ++cols) {
      const int k = rows < cols ? rows : cols;
      for (int rank = 0; rank <= k; ++rank) {
        for (int style = 0; style < 4; ++style) {
          const std::uint64_t seed =
              static_cast<std::uint64_t>(((rows * 16 + cols) * 16 + rank) * 4 + style);
          Lcg rng(seed);
          const std::vector<double> spectrum = shaped_spectrum(k, rank, style, rng);
          const MatrixOrder order =
              (seed % 2 == 0) ? MatrixOrder::ColMajor : MatrixOrder::RowMajor;
          const auto c = make_svd_case(rows, cols, spectrum, order, seed + 7000);
          const SvdResult r = run_svd(c);
          ASSERT_TRUE(r.ok) << "case " << cases << ": " << rows << "x" << cols
                            << " rank " << rank << " style " << style;
          EXPECT_TRUE(SvdAccepted(judge(c.m, rows, cols, order, r)))
              << "case " << cases << ": " << rows << "x" << cols << " rank "
              << rank << " style " << style << " seed " << seed;
          ++cases;
        }
      }
    }
  }
  EXPECT_GT(cases, 500);
}

// The same sweep on matrices pushed to the ends of the exponent range, where
// a squared norm would overflow or underflow if anything were computed the
// naive way.
TEST(Stress, SvdAtTheEndsOfTheExponentRange) {
  for (const int exponent : {-1000, -700, -300, 300, 700, 1000}) {
    const double factor = std::ldexp(1.0, exponent);
    for (int shape = 0; shape < 6; ++shape) {
      const int rows = 2 + shape;
      const int cols = 8 - shape;
      const std::uint64_t seed = static_cast<std::uint64_t>(1000 + exponent + shape);
      std::vector<Complex> m = random_matrix(rows, cols, seed);
      for (Complex& z : m) z *= factor;
      const SvdResult r = run_svd(m.data(), rows, cols, MatrixOrder::ColMajor);
      ASSERT_TRUE(r.ok) << "2^" << exponent << " " << rows << "x" << cols;
      EXPECT_TRUE(SvdAccepted(judge(m, rows, cols, MatrixOrder::ColMajor, r)))
          << "2^" << exponent << " " << rows << "x" << cols;
    }
  }
}

// Matrices with entries at wildly different magnitudes in the same column,
// which is where a naive norm loses the small ones entirely.
TEST(Stress, SvdOnMixedMagnitudeEntries) {
  for (int trial = 0; trial < 40; ++trial) {
    const int n = 2 + (trial % 7);
    Lcg rng(static_cast<std::uint64_t>(9000 + trial));
    std::vector<Complex> m(static_cast<std::size_t>(n) * static_cast<std::size_t>(n));
    for (Complex& z : m) {
      const double u = 0.5 * (rng.next_uniform() + 1.0);
      z = rng.next_complex() * std::pow(10.0, -60.0 * u);
    }
    const SvdResult r = run_svd(m.data(), n, n, MatrixOrder::ColMajor);
    ASSERT_TRUE(r.ok) << "trial " << trial;
    EXPECT_TRUE(SvdAccepted(judge(m, n, n, MatrixOrder::ColMajor, r))) << "trial " << trial;
  }
}

TEST(Stress, EighOverSizesAndSpectra) {
  int cases = 0;
  for (int n = 1; n <= 12; ++n) {
    for (int style = 0; style < 5; ++style) {
      const std::uint64_t seed = static_cast<std::uint64_t>(n * 8 + style);
      Lcg rng(seed);
      std::vector<double> evals(static_cast<std::size_t>(n), 0.0);
      for (int i = 0; i < n; ++i) {
        const std::size_t idx = static_cast<std::size_t>(i);
        switch (style) {
          case 0: evals[idx] = 1.0; break;                       // flat, positive
          case 1: evals[idx] = (i % 2 == 0) ? -2.0 : 2.0; break; // exactly +/- pairs
          case 2: evals[idx] = (i < n / 2) ? 0.0 : 1.0; break;   // half a null space
          case 3: evals[idx] = std::pow(10.0, -3.0 * i); break;  // graded
          default: evals[idx] = rng.next_uniform(); break;       // indefinite, generic
        }
      }
      const MatrixOrder order =
          (seed % 2 == 0) ? MatrixOrder::ColMajor : MatrixOrder::RowMajor;
      const auto c = make_eigh_case(n, evals, order, seed + 4000);
      const EighResult r = run_eigh(c);
      ASSERT_TRUE(r.ok) << "n " << n << " style " << style;
      EXPECT_TRUE(EighAccepted(autonne::verify::check_eigh(
          c.a.data(), n, order, r.evals.data(), r.evecs.data())))
          << "n " << n << " style " << style << " seed " << seed;
      ++cases;
    }
  }
  EXPECT_GT(cases, 50);
}

TEST(Stress, EighOnGenericHermitianAcrossSizes) {
  for (int n = 1; n <= 20; ++n) {
    const std::uint64_t seed = static_cast<std::uint64_t>(n) * 6151u;
    const std::vector<Complex> a = random_hermitian(n, seed);
    const EighResult r = run_eigh(a.data(), n, MatrixOrder::ColMajor);
    ASSERT_TRUE(r.ok) << "n " << n;
    EXPECT_TRUE(EighAccepted(autonne::verify::check_eigh(
        a.data(), n, MatrixOrder::ColMajor, r.evals.data(), r.evecs.data())))
        << "n " << n;
  }
}

// Every entry non-finite, every entry but one non-finite, one entry
// non-finite: all refused, with the caller's buffers untouched.
TEST(Stress, NonFiniteInputIsAlwaysRefused) {
  const int n = 5;
  const std::uint64_t patterns[] = {UINT64_C(0x7FF8000000000000),
                                    UINT64_C(0x7FF0000000000000),
                                    UINT64_C(0xFFF0000000000000),
                                    UINT64_C(0x7FF0000000000001)};
  for (const std::uint64_t bits : patterns) {
    for (int position = 0; position < n * n; ++position) {
      std::vector<Complex> m = random_matrix(n, n, static_cast<std::uint64_t>(position));
      autonne_test::poke_bits(m[static_cast<std::size_t>(position)], bits,
                              autonne_test::bits_of(1.0));
      const SvdResult r = run_svd(m.data(), n, n, MatrixOrder::ColMajor);
      ASSERT_FALSE(r.ok) << "position " << position;
      for (const double s : r.s) EXPECT_EQ(s, autonne_test::kSentinelReal);

      const EighResult e = run_eigh(m.data(), n, MatrixOrder::ColMajor);
      ASSERT_FALSE(e.ok) << "position " << position;
      for (const double x : e.evals) EXPECT_EQ(x, autonne_test::kSentinelReal);
    }
  }
}

}  // namespace
