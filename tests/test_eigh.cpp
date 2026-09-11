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

// The Hermitian eigensolver against the harness and against spectra known by
// construction. Same shape as test_svd.cpp.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "autonne/autonne.hpp"
#include "autonne/verify.hpp"
#include "report_matchers.hpp"
#include "test_support.hpp"

namespace {

using autonne::MatrixOrder;
using autonne::verify::EighReport;
using autonne::verify::check_eigh;
using autonne_test::Complex;
using autonne_test::EighAccepted;
using autonne_test::EighCase;
using autonne_test::EighResult;
using autonne_test::SpectrumClose;
using autonne_test::SpectrumWithinFactor;
using autonne_test::make_eigh_case;
using autonne_test::random_hermitian;
using autonne_test::run_eigh;

constexpr double kEps = std::numeric_limits<double>::epsilon();

EighReport check(const Complex* a, int n, MatrixOrder order, const EighResult& r) {
  return check_eigh(a, n, order, r.evals.data(), r.evecs.data());
}

EighReport check(const EighCase& c, const EighResult& r) {
  return check(c.a.data(), c.n, c.order, r);
}

double spectrum_tol(int n, double lambda_max) {
  return 32.0 * static_cast<double>(n) * kEps * lambda_max;
}

// --- shapes and orders -----------------------------------------------------

TEST(Eigh, DecomposesFixture) {
  const EighCase c = make_eigh_case(4, {-3.0, -0.5, 1.0, 4.0}, MatrixOrder::ColMajor, 5150);
  const EighResult r = run_eigh(c);
  ASSERT_TRUE(r.ok);
  EXPECT_TRUE(EighAccepted(check(c, r)));
  EXPECT_TRUE(SpectrumClose(r.evals, c.evals, spectrum_tol(4, 4.0), false));
}

TEST(Eigh, StorageOrderDoesNotChangeTheResult) {
  const std::vector<double> evals = {-2.0, 0.0, 0.5, 3.0, 7.0};
  const EighCase col = make_eigh_case(5, evals, MatrixOrder::ColMajor, 61);
  std::vector<Complex> row(col.a.size());
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 5; ++j) {
      row[static_cast<std::size_t>(i * 5 + j)] = col.a[static_cast<std::size_t>(j * 5 + i)];
    }
  }
  const EighResult rc = run_eigh(col);
  const EighResult rr = run_eigh(row.data(), 5, MatrixOrder::RowMajor);
  ASSERT_TRUE(rc.ok);
  ASSERT_TRUE(rr.ok);
  EXPECT_TRUE(EighAccepted(check(col, rc)));
  EXPECT_TRUE(EighAccepted(check(row.data(), 5, MatrixOrder::RowMajor, rr)));
  EXPECT_TRUE(SpectrumClose(rc.evals, evals, spectrum_tol(5, 7.0), false));
  EXPECT_EQ(rc.evals, rr.evals);
  EXPECT_EQ(rc.evecs, rr.evecs);
}

TEST(Eigh, IsDeterministic) {
  const EighCase c = make_eigh_case(6, {-1.0, -0.5, 0.0, 0.5, 1.0, 2.0}, MatrixOrder::ColMajor, 62);
  const EighResult a = run_eigh(c);
  const EighResult b = run_eigh(c);
  ASSERT_TRUE(a.ok);
  ASSERT_TRUE(b.ok);
  EXPECT_EQ(a.evals, b.evals);
  EXPECT_EQ(a.evecs, b.evecs);
}

// The eigenvalues-only path must agree with the full path bit for bit: it is
// the same iteration, minus the accumulation of the eigenvectors.
TEST(Eigh, EigenvaluesOnlyMatchesFullDecomposition) {
  const EighCase c = make_eigh_case(6, {-1.0, -0.5, 0.0, 0.5, 1.0, 2.0}, MatrixOrder::RowMajor, 63);
  const EighResult full = run_eigh(c);
  ASSERT_TRUE(full.ok);
  std::vector<double> only(6, 0.0);
  ASSERT_TRUE(autonne::eigh(c.a.data(), c.n, c.order, only.data(), nullptr));
  EXPECT_EQ(only, full.evals);
}

TEST(Eigh, DecomposesSingletonAndTwoByTwo) {
  const std::vector<Complex> one = {Complex(2.5, 0.0)};
  const EighResult r1 = run_eigh(one.data(), 1, MatrixOrder::ColMajor);
  ASSERT_TRUE(r1.ok);
  EXPECT_TRUE(EighAccepted(check(one.data(), 1, MatrixOrder::ColMajor, r1)));
  EXPECT_EQ(r1.evals[0], 2.5);
  EXPECT_EQ(std::abs(r1.evecs[0]), 1.0);

  // [[1, i], [-i, 1]] has eigenvalues 0 and 2.
  const std::vector<Complex> two = {Complex(1.0, 0.0), Complex(0.0, -1.0),
                                    Complex(0.0, 1.0), Complex(1.0, 0.0)};
  const EighResult r2 = run_eigh(two.data(), 2, MatrixOrder::ColMajor);
  ASSERT_TRUE(r2.ok);
  EXPECT_TRUE(EighAccepted(check(two.data(), 2, MatrixOrder::ColMajor, r2)));
  EXPECT_TRUE(SpectrumClose(r2.evals, {0.0, 2.0}, 8.0 * kEps, false));
}

// --- degenerate and singular spectra ---------------------------------------

TEST(Eigh, ResolvesDegenerateAndSingularSpectra) {
  const EighCase a = make_eigh_case(5, {1.0, 1.0, 1.0, 4.0, 4.0}, MatrixOrder::ColMajor, 71);
  const EighResult ra = run_eigh(a);
  ASSERT_TRUE(ra.ok);
  EXPECT_TRUE(EighAccepted(check(a, ra)));
  EXPECT_TRUE(SpectrumClose(ra.evals, a.evals, spectrum_tol(5, 4.0), false));

  const EighCase b = make_eigh_case(4, {0.0, 0.0, 0.0, 2.0}, MatrixOrder::ColMajor, 72);
  const EighResult rb = run_eigh(b);
  ASSERT_TRUE(rb.ok);
  EXPECT_TRUE(EighAccepted(check(b, rb)));
  EXPECT_TRUE(SpectrumClose(rb.evals, b.evals, spectrum_tol(4, 2.0), false));
}

TEST(Eigh, DecomposesTheZeroMatrix) {
  const int n = 4;
  const std::vector<Complex> zero(16, Complex(0.0, 0.0));
  const EighResult r = run_eigh(zero.data(), n, MatrixOrder::ColMajor);
  ASSERT_TRUE(r.ok);
  EXPECT_TRUE(EighAccepted(check(zero.data(), n, MatrixOrder::ColMajor, r)));
  for (const double x : r.evals) EXPECT_EQ(x, 0.0);
}

// A diagonal input needs no rotation at all, so the eigenvectors come out as
// exact canonical basis vectors, permuted into ascending order.
TEST(Eigh, DiagonalInputGivesCanonicalEigenvectors) {
  const int n = 5;
  const std::vector<double> diag = {3.0, -1.0, 0.0, 2.0, -4.0};
  std::vector<Complex> a(25, Complex(0.0, 0.0));
  for (int i = 0; i < n; ++i) a[static_cast<std::size_t>(i + i * n)] = Complex(diag[static_cast<std::size_t>(i)], 0.0);
  const EighResult r = run_eigh(a.data(), n, MatrixOrder::ColMajor);
  ASSERT_TRUE(r.ok);
  EXPECT_TRUE(EighAccepted(check(a.data(), n, MatrixOrder::ColMajor, r)));
  EXPECT_EQ(r.evals, (std::vector<double>{-4.0, -1.0, 0.0, 2.0, 3.0}));
  const std::vector<int> source = {4, 1, 2, 3, 0};
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      const Complex z = r.evecs[static_cast<std::size_t>(i + j * n)];
      if (i == source[static_cast<std::size_t>(j)]) {
        EXPECT_EQ(std::abs(z), 1.0) << "(" << i << ", " << j << ")";
      } else {
        EXPECT_EQ(z, Complex(0.0, 0.0)) << "(" << i << ", " << j << ")";
      }
    }
  }
}

// An exactly zero row (and therefore column) is structural: its eigenvalue
// is exactly zero and its eigenvector is the canonical basis vector.
TEST(Eigh, StructurallyZeroRowsGiveExactZeroEigenvalues) {
  const int n = 6;
  std::vector<Complex> a = random_hermitian(n, 909);
  for (int t = 0; t < n; ++t) {
    a[static_cast<std::size_t>(2 + t * n)] = Complex(0.0, 0.0);
    a[static_cast<std::size_t>(t + 2 * n)] = Complex(0.0, 0.0);
    a[static_cast<std::size_t>(5 + t * n)] = Complex(0.0, 0.0);
    a[static_cast<std::size_t>(t + 5 * n)] = Complex(0.0, 0.0);
  }
  const EighResult r = run_eigh(a.data(), n, MatrixOrder::ColMajor);
  ASSERT_TRUE(r.ok);
  EXPECT_TRUE(EighAccepted(check(a.data(), n, MatrixOrder::ColMajor, r)));

  int exact_zeros = 0;
  std::vector<int> canonical_rows;
  for (int j = 0; j < n; ++j) {
    if (r.evals[static_cast<std::size_t>(j)] != 0.0) continue;
    ++exact_zeros;
    int hits = 0;
    int row = -1;
    for (int i = 0; i < n; ++i) {
      const Complex z = r.evecs[static_cast<std::size_t>(i + j * n)];
      if (z == Complex(0.0, 0.0)) continue;
      ++hits;
      row = i;
      EXPECT_EQ(std::abs(z), 1.0);
    }
    EXPECT_EQ(hits, 1);
    canonical_rows.push_back(row);
  }
  EXPECT_EQ(exact_zeros, 2);
  std::sort(canonical_rows.begin(), canonical_rows.end());
  EXPECT_EQ(canonical_rows, (std::vector<int>{2, 5}));
}

// --- the input contract ----------------------------------------------------

// The kernel decomposes the Hermitian part (A + A^*) / 2 of whatever it is
// given. For Hermitian input that is the input itself, bit for bit; for a
// slightly non-Hermitian input it is the nearest Hermitian matrix, which is
// what a caller holding a rounded product wants.
TEST(Eigh, DecomposesTheHermitianPartOfItsInput) {
  const int n = 5;
  std::vector<Complex> a = random_hermitian(n, 4040);
  const std::vector<Complex> e = autonne_test::random_matrix(n, n, 4041);
  for (std::size_t i = 0; i < a.size(); ++i) a[i] += 1e-3 * e[i];

  std::vector<Complex> h(a.size());
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      h[static_cast<std::size_t>(i + j * n)] =
          0.5 * (a[static_cast<std::size_t>(i + j * n)] + std::conj(a[static_cast<std::size_t>(j + i * n)]));
    }
  }

  const EighResult r = run_eigh(a.data(), n, MatrixOrder::ColMajor);
  ASSERT_TRUE(r.ok);
  EXPECT_TRUE(EighAccepted(check(h.data(), n, MatrixOrder::ColMajor, r)));
  EXPECT_FALSE(check(a.data(), n, MatrixOrder::ColMajor, r).input_hermitian);

  const EighResult rh = run_eigh(h.data(), n, MatrixOrder::ColMajor);
  ASSERT_TRUE(rh.ok);
  EXPECT_EQ(rh.evals, r.evals);
}

// --- accuracy --------------------------------------------------------------

TEST(Eigh, ResolvesIndefiniteSpectrumWithTinyEigenvalues) {
  const EighCase c = make_eigh_case(6, {-1.0, -1e-9, -1e-12, 1e-12, 1e-9, 1.0}, MatrixOrder::ColMajor, 808);
  const EighResult r = run_eigh(c);
  ASSERT_TRUE(r.ok);
  EXPECT_TRUE(EighAccepted(check(c, r)));
  EXPECT_TRUE(SpectrumClose(r.evals, c.evals, spectrum_tol(6, 1.0), false));
}

// A Gram matrix X^* X of a 6 x 4 X has rank 4: two eigenvalues are zero and
// none may come out negative beyond rounding.
TEST(Eigh, PositiveSemidefiniteGramMatrix) {
  const int rows = 6;
  const int n = 4;
  const std::vector<Complex> x = autonne_test::random_matrix(rows, n, 777);
  std::vector<Complex> g(16, Complex(0.0, 0.0));
  double norm_g = 0.0;
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      Complex acc(0.0, 0.0);
      for (int t = 0; t < rows; ++t) {
        acc += std::conj(x[static_cast<std::size_t>(t + i * rows)]) * x[static_cast<std::size_t>(t + j * rows)];
      }
      g[static_cast<std::size_t>(i + j * n)] = acc;
      norm_g += std::norm(acc);
    }
  }
  norm_g = std::sqrt(norm_g);
  const EighResult r = run_eigh(g.data(), n, MatrixOrder::ColMajor);
  ASSERT_TRUE(r.ok);
  EXPECT_TRUE(EighAccepted(check(g.data(), n, MatrixOrder::ColMajor, r)));
  for (const double x_ : r.evals) EXPECT_GE(x_, -32.0 * n * kEps * norm_g);
}

// For H = D B D with every eigenvalue of B in [0.9, 1.1], each eigenvalue of
// H lies in [0.9, 1.1] times the matching entry of D^2 (Ostrowski). A solver
// with only absolute accuracy eps * ||H|| fails this as soon as d_k^2 drops
// below eps; Jacobi keeps it (Demmel & Veselic 1992).
TEST(Eigh, GradedPositiveDefiniteMatrixKeepsRelativeAccuracy) {
  const int n = 8;
  const std::vector<double> d = {1.0, 1e-4, 1e-8, 1e-12, 1e-16, 1e-20, 1e-24, 1e-28};
  std::vector<Complex> h = autonne_test::well_conditioned_hermitian(n, 3003);
  autonne_test::scale_rows(h, n, n, d);
  autonne_test::scale_columns(h, n, n, d);
  std::vector<double> d_sq(d.size());
  for (std::size_t i = 0; i < d.size(); ++i) d_sq[i] = d[i] * d[i];

  const EighResult r = run_eigh(h.data(), n, MatrixOrder::ColMajor);
  ASSERT_TRUE(r.ok);
  EXPECT_TRUE(EighAccepted(check(h.data(), n, MatrixOrder::ColMajor, r)));
  EXPECT_TRUE(SpectrumWithinFactor(r.evals, d_sq, 0.9, 1.1, false));
}

// --- the awkward corners of the exponent range -----------------------------

// Eigenvectors must stay unitary even when the scaling drives an off-diagonal
// entry subnormal. Nothing in this input is subnormal or even unusual: one
// large diagonal entry and one small off-diagonal pair. The kernel scales by
// the largest component, which pushes the off-diagonal below the normal
// range, and a rotation whose phase is computed from a subnormal value is not
// unitary, so the accumulated eigenvector matrix stops being one.
TEST(Eigh, KeepsEigenvectorsUnitaryWhenScalingDrivesEntriesSubnormal) {
  const int n = 3;
  struct Case {
    int big;
    int small;
  };
  for (const Case c : {Case{1020, -50}, Case{1000, -40}, Case{980, -60}}) {
    std::vector<Complex> h(9, Complex(0.0, 0.0));
    h[static_cast<std::size_t>(2 + 2 * n)] = Complex(std::ldexp(1.0, c.big), 0.0);
    const double off = std::ldexp(1.0, c.small);
    h[static_cast<std::size_t>(0 + 1 * n)] = Complex(off, 1.5 * off);
    h[static_cast<std::size_t>(1 + 0 * n)] = Complex(off, -1.5 * off);

    const EighResult r = run_eigh(h.data(), n, MatrixOrder::ColMajor);
    ASSERT_TRUE(r.ok) << c.big << "/" << c.small;
    EXPECT_TRUE(EighAccepted(check(h.data(), n, MatrixOrder::ColMajor, r)))
        << c.big << "/" << c.small;
  }
}

// The same defect reached directly, with literal subnormals in the input.
TEST(Eigh, KeepsEigenvectorsUnitaryWithSubnormalOffDiagonal) {
  const int n = 3;
  const double t = std::ldexp(1.0, -1074);
  std::vector<Complex> h(9, Complex(0.0, 0.0));
  h[static_cast<std::size_t>(2 + 2 * n)] = Complex(1.0, 0.0);
  h[static_cast<std::size_t>(0 + 1 * n)] = Complex(2.0 * t, 3.0 * t);
  h[static_cast<std::size_t>(1 + 0 * n)] = Complex(2.0 * t, -3.0 * t);

  const EighResult r = run_eigh(h.data(), n, MatrixOrder::ColMajor);
  ASSERT_TRUE(r.ok);
  EXPECT_TRUE(EighAccepted(check(h.data(), n, MatrixOrder::ColMajor, r)));
}

// --- size ------------------------------------------------------------------

TEST(Eigh, DecomposesLargeRandomHermitian) {
  for (const int n : {33, 64, 128}) {
    const std::vector<Complex> a = random_hermitian(n, static_cast<std::uint64_t>(n) * 7919u);
    const EighResult r = run_eigh(a.data(), n, MatrixOrder::ColMajor);
    ASSERT_TRUE(r.ok) << n;
    EXPECT_TRUE(EighAccepted(check(a.data(), n, MatrixOrder::ColMajor, r))) << n;
  }
}

TEST(Eigh, DecomposesLargeDegenerateSpectrum) {
  std::vector<double> evals(96, 0.0);
  for (int i = 0; i < 32; ++i) evals[static_cast<std::size_t>(i)] = -1.0;
  for (int i = 64; i < 96; ++i) evals[static_cast<std::size_t>(i)] = 0.5;
  const EighCase c = make_eigh_case(96, evals, MatrixOrder::ColMajor, 9696);
  const EighResult r = run_eigh(c);
  ASSERT_TRUE(r.ok);
  EXPECT_TRUE(EighAccepted(check(c, r)));
  EXPECT_TRUE(SpectrumClose(r.evals, c.evals, spectrum_tol(96, 1.0), false));
}

}  // namespace
