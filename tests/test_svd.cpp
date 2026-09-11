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

// The SVD kernel against the harness and against spectra known by
// construction. Every case is judged by check_svd first -- that is the
// contract -- and then by whatever sharper property the fixture makes
// available: the exact spectrum, exact zeros, relative accuracy, or bitwise
// invariance.
//
// Compiled into every test variant, so each case runs under strict and
// fast-math floating point.

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
using autonne::verify::SvdReport;
using autonne::verify::check_svd;
using autonne_test::Complex;
using autonne_test::SpectrumClose;
using autonne_test::SpectrumWithinFactor;
using autonne_test::SvdAccepted;
using autonne_test::SvdCase;
using autonne_test::SvdResult;
using autonne_test::make_svd_case;
using autonne_test::random_matrix;
using autonne_test::run_svd;

constexpr double kEps = std::numeric_limits<double>::epsilon();

SvdReport check(const Complex* m, int rows, int cols, MatrixOrder order,
                const SvdResult& r) {
  const int k = rows < cols ? rows : cols;
  return check_svd(m, rows, cols, order, r.u.data(), r.s.data(), r.v.data(), k);
}

SvdReport check(const SvdCase& c, const SvdResult& r) {
  return check(c.m.data(), c.rows, c.cols, c.order, r);
}

// Absolute tolerance for a spectrum built by construction: the fixture's own
// rounding perturbs the true singular values by a few ulps of the largest.
double spectrum_tol(int rows, int cols, double s_max) {
  const int dim = rows > cols ? rows : cols;
  return 32.0 * static_cast<double>(dim) * kEps * s_max;
}

// --- shapes and orders -----------------------------------------------------

TEST(Svd, FactorsSquareMatrix) {
  const SvdCase c = make_svd_case(4, 4, {8.0, 4.0, 2.0, 1.0}, MatrixOrder::ColMajor, 12345);
  const SvdResult r = run_svd(c);
  ASSERT_TRUE(r.ok);
  EXPECT_TRUE(SvdAccepted(check(c, r)));
  EXPECT_TRUE(SpectrumClose(r.s, c.s, spectrum_tol(4, 4, 8.0), true));
}

TEST(Svd, FactorsTallAndWideMatrices) {
  const SvdCase tall = make_svd_case(6, 3, {5.0, 2.0, 0.25}, MatrixOrder::ColMajor, 777);
  const SvdResult rt = run_svd(tall);
  ASSERT_TRUE(rt.ok);
  EXPECT_TRUE(SvdAccepted(check(tall, rt)));
  EXPECT_TRUE(SpectrumClose(rt.s, tall.s, spectrum_tol(6, 3, 5.0), true));

  const SvdCase wide = make_svd_case(3, 6, {5.0, 2.0, 0.25}, MatrixOrder::ColMajor, 778);
  const SvdResult rw = run_svd(wide);
  ASSERT_TRUE(rw.ok);
  EXPECT_TRUE(SvdAccepted(check(wide, rw)));
  EXPECT_TRUE(SpectrumClose(rw.s, wide.s, spectrum_tol(3, 6, 5.0), true));
}

TEST(Svd, FactorsVectorsAndScalars) {
  // 1 x 1: the singular value is the modulus and the factors are phases.
  const std::vector<Complex> scalar = {Complex(3.0, -4.0)};
  const SvdResult rs = run_svd(scalar.data(), 1, 1, MatrixOrder::ColMajor);
  ASSERT_TRUE(rs.ok);
  EXPECT_TRUE(SvdAccepted(check(scalar.data(), 1, 1, MatrixOrder::ColMajor, rs)));
  EXPECT_NEAR(rs.s[0], 5.0, 4.0 * kEps * 5.0);

  // n x 1 and 1 x n.
  const std::vector<Complex> column = random_matrix(7, 1, 41);
  double norm_sq = 0.0;
  for (const Complex& z : column) norm_sq += std::norm(z);
  const SvdResult rc = run_svd(column.data(), 7, 1, MatrixOrder::ColMajor);
  ASSERT_TRUE(rc.ok);
  EXPECT_TRUE(SvdAccepted(check(column.data(), 7, 1, MatrixOrder::ColMajor, rc)));
  EXPECT_NEAR(rc.s[0], std::sqrt(norm_sq), 16.0 * kEps * std::sqrt(norm_sq));

  const SvdResult rr = run_svd(column.data(), 1, 7, MatrixOrder::RowMajor);
  ASSERT_TRUE(rr.ok);
  EXPECT_TRUE(SvdAccepted(check(column.data(), 1, 7, MatrixOrder::RowMajor, rr)));
  EXPECT_NEAR(rr.s[0], std::sqrt(norm_sq), 16.0 * kEps * std::sqrt(norm_sq));
}

// The same logical matrix in both storage orders must give the same factors
// bit for bit: the kernel copies into its own layout before any arithmetic.
TEST(Svd, StorageOrderDoesNotChangeTheResult) {
  const std::vector<double> spectrum = {9.0, 3.0, 1.5, 0.5, 0.125};
  const SvdCase col = make_svd_case(5, 3, {9.0, 3.0, 1.5}, MatrixOrder::ColMajor, 4242);
  // Re-express col.m (column-major 5 x 3) as row-major.
  std::vector<Complex> row(col.m.size());
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 3; ++j) {
      row[static_cast<std::size_t>(i * 3 + j)] = col.m[static_cast<std::size_t>(j * 5 + i)];
    }
  }
  const SvdResult rc = run_svd(col);
  const SvdResult rr = run_svd(row.data(), 5, 3, MatrixOrder::RowMajor);
  ASSERT_TRUE(rc.ok);
  ASSERT_TRUE(rr.ok);
  EXPECT_TRUE(SvdAccepted(check(col, rc)));
  EXPECT_TRUE(SvdAccepted(check(row.data(), 5, 3, MatrixOrder::RowMajor, rr)));
  EXPECT_EQ(rc.s, rr.s);
  EXPECT_EQ(rc.u, rr.u);
  EXPECT_EQ(rc.v, rr.v);

  const SvdCase square = make_svd_case(5, 5, spectrum, MatrixOrder::RowMajor, 4243);
  const SvdResult rq = run_svd(square);
  ASSERT_TRUE(rq.ok);
  EXPECT_TRUE(SvdAccepted(check(square, rq)));
  EXPECT_TRUE(SpectrumClose(rq.s, square.s, spectrum_tol(5, 5, 9.0), true));
}

TEST(Svd, IsDeterministic) {
  const SvdCase c = make_svd_case(7, 5, {5.0, 4.0, 3.0, 2.0, 1.0}, MatrixOrder::ColMajor, 99);
  const SvdResult a = run_svd(c);
  const SvdResult b = run_svd(c);
  ASSERT_TRUE(a.ok);
  ASSERT_TRUE(b.ok);
  EXPECT_EQ(a.s, b.s);
  EXPECT_EQ(a.u, b.u);
  EXPECT_EQ(a.v, b.v);
}

// --- degenerate and rank-deficient spectra ---------------------------------

TEST(Svd, ResolvesDegenerateSpectrum) {
  const SvdCase c = make_svd_case(5, 5, {3.0, 3.0, 3.0, 1.0, 1.0}, MatrixOrder::ColMajor, 91);
  const SvdResult r = run_svd(c);
  ASSERT_TRUE(r.ok);
  EXPECT_TRUE(SvdAccepted(check(c, r)));
  EXPECT_TRUE(SpectrumClose(r.s, c.s, spectrum_tol(5, 5, 3.0), true));
}

TEST(Svd, ResolvesRankDeficientSpectrum) {
  const SvdCase c = make_svd_case(5, 4, {2.0, 1.0, 0.0, 0.0}, MatrixOrder::ColMajor, 92);
  const SvdResult r = run_svd(c);
  ASSERT_TRUE(r.ok);
  EXPECT_TRUE(SvdAccepted(check(c, r)));
  EXPECT_TRUE(SpectrumClose(r.s, c.s, spectrum_tol(5, 4, 2.0), true));
}

TEST(Svd, FactorsTheZeroMatrix) {
  for (const auto& shape : {std::pair<int, int>{3, 3}, {5, 2}, {2, 5}, {1, 1}}) {
    const int rows = shape.first;
    const int cols = shape.second;
    const std::vector<Complex> zero(static_cast<std::size_t>(rows * cols), Complex(0.0, 0.0));
    const SvdResult r = run_svd(zero.data(), rows, cols, MatrixOrder::ColMajor);
    ASSERT_TRUE(r.ok) << rows << "x" << cols;
    EXPECT_TRUE(SvdAccepted(check(zero.data(), rows, cols, MatrixOrder::ColMajor, r)))
        << rows << "x" << cols;
    for (const double s : r.s) EXPECT_EQ(s, 0.0);
  }
}

// Every singular value of a unitary is one: the fully degenerate case.
TEST(Svd, UnitaryInputHasFlatSpectrum) {
  const int n = 8;
  const std::vector<Complex> f = autonne_test::dft_unitary(n);
  const SvdResult r = run_svd(f.data(), n, n, MatrixOrder::ColMajor);
  ASSERT_TRUE(r.ok);
  EXPECT_TRUE(SvdAccepted(check(f.data(), n, n, MatrixOrder::ColMajor, r)));
  EXPECT_TRUE(SpectrumClose(r.s, std::vector<double>(8, 1.0), 32.0 * n * kEps, true));
}

TEST(Svd, FactorsRankOneMatrix) {
  const int rows = 6;
  const int cols = 4;
  const std::vector<Complex> x = random_matrix(rows, 1, 501);
  const std::vector<Complex> y = random_matrix(cols, 1, 502);
  std::vector<Complex> m(static_cast<std::size_t>(rows * cols));
  double nx = 0.0;
  double ny = 0.0;
  for (const Complex& z : x) nx += std::norm(z);
  for (const Complex& z : y) ny += std::norm(z);
  for (int j = 0; j < cols; ++j) {
    for (int i = 0; i < rows; ++i) {
      m[static_cast<std::size_t>(i + j * rows)] =
          x[static_cast<std::size_t>(i)] * std::conj(y[static_cast<std::size_t>(j)]);
    }
  }
  const SvdResult r = run_svd(m.data(), rows, cols, MatrixOrder::ColMajor);
  ASSERT_TRUE(r.ok);
  EXPECT_TRUE(SvdAccepted(check(m.data(), rows, cols, MatrixOrder::ColMajor, r)));
  const double sigma = std::sqrt(nx * ny);
  EXPECT_TRUE(SpectrumClose(r.s, {sigma, 0.0, 0.0, 0.0}, spectrum_tol(rows, cols, sigma), true));
}

// Rows and columns that are exactly zero are structural, and the kernel is
// expected to treat them as such: the singular values they account for are
// exactly zero, and the null vectors for them are the canonical basis vectors
// of the missing rows and columns, which are orthonormal to everything else
// with no arithmetic at all.
TEST(Svd, StructurallyZeroRowsAndColumnsGiveExactZeros) {
  const int rows = 7;
  const int cols = 6;
  std::vector<Complex> m = random_matrix(rows, cols, 313);
  for (int j = 0; j < cols; ++j) {
    m[static_cast<std::size_t>(1 + j * rows)] = Complex(0.0, 0.0);
    m[static_cast<std::size_t>(4 + j * rows)] = Complex(0.0, 0.0);
  }
  for (int i = 0; i < rows; ++i) {
    m[static_cast<std::size_t>(i + 0 * rows)] = Complex(0.0, 0.0);
    m[static_cast<std::size_t>(i + 3 * rows)] = Complex(0.0, 0.0);
  }
  const SvdResult r = run_svd(m.data(), rows, cols, MatrixOrder::ColMajor);
  ASSERT_TRUE(r.ok);
  EXPECT_TRUE(SvdAccepted(check(m.data(), rows, cols, MatrixOrder::ColMajor, r)));

  // Rank is at most min(5 live rows, 4 live columns) = 4.
  ASSERT_EQ(r.s.size(), 6u);
  EXPECT_GT(r.s[3], 0.0);
  EXPECT_EQ(r.s[4], 0.0);
  EXPECT_EQ(r.s[5], 0.0);

  // The two null columns of V are e_0 and e_3 in some order; those of U are
  // e_1 and e_4.
  auto is_canonical = [](const std::vector<Complex>& mat, int dim, int col,
                         std::vector<int> allowed) {
    int hits = 0;
    bool ok = true;
    for (int i = 0; i < dim; ++i) {
      const Complex z = mat[static_cast<std::size_t>(i + col * dim)];
      if (z == Complex(0.0, 0.0)) continue;
      ++hits;
      if (std::abs(z) != 1.0) ok = false;
      if (std::find(allowed.begin(), allowed.end(), i) == allowed.end()) ok = false;
    }
    return ok && hits == 1;
  };
  EXPECT_TRUE(is_canonical(r.v, cols, 4, {0, 3}));
  EXPECT_TRUE(is_canonical(r.v, cols, 5, {0, 3}));
  EXPECT_TRUE(is_canonical(r.u, rows, 4, {1, 4}));
  EXPECT_TRUE(is_canonical(r.u, rows, 5, {1, 4}));
}

// --- the two shapes from the spec ------------------------------------------

// Twelve-fold degenerate, rank 12 of 36. Eigen's divide-and-conquer returned
// a spectrum summing to 0.9861 against a norm of 1.0 on the original.
TEST(Svd, SimonCosetMatrixHasExactSpectrum) {
  const int n = 36;
  const std::vector<Complex> m = autonne_test::simon_coset_matrix();
  const SvdResult r = run_svd(m.data(), n, n, MatrixOrder::ColMajor);
  ASSERT_TRUE(r.ok);
  EXPECT_TRUE(SvdAccepted(check(m.data(), n, n, MatrixOrder::ColMajor, r)));

  const double sigma = std::sqrt(3.0) * (1.0 / 6.0);
  std::vector<double> expected(36, 0.0);
  for (int i = 0; i < 12; ++i) expected[static_cast<std::size_t>(i)] = sigma;
  EXPECT_TRUE(SpectrumClose(r.s, expected, 32.0 * n * kEps * sigma, true));
  // The twenty-four zero columns are structural, so their values are exact.
  for (int i = 12; i < 36; ++i) EXPECT_EQ(r.s[static_cast<std::size_t>(i)], 0.0) << i;

  double energy = 0.0;
  for (const double s : r.s) energy += s * s;
  EXPECT_NEAR(energy, 1.0, 64.0 * n * kEps);
}

// The same matrix carrying a rounding residue in one entry, as the original
// did: no longer structurally rank 12, and the tail must stay at the level of
// that residue rather than growing into a phantom singular value.
TEST(Svd, SimonCosetMatrixWithResidueKeepsTinyTail) {
  const int n = 36;
  const std::vector<Complex> m = autonne_test::simon_coset_matrix(3.4e-17);
  const SvdResult r = run_svd(m.data(), n, n, MatrixOrder::ColMajor);
  ASSERT_TRUE(r.ok);
  EXPECT_TRUE(SvdAccepted(check(m.data(), n, n, MatrixOrder::ColMajor, r)));

  const double sigma = std::sqrt(3.0) * (1.0 / 6.0);
  for (int i = 0; i < 12; ++i) {
    EXPECT_NEAR(r.s[static_cast<std::size_t>(i)], sigma, 32.0 * n * kEps * sigma) << i;
  }
  for (int i = 12; i < 36; ++i) EXPECT_LE(r.s[static_cast<std::size_t>(i)], 1e-15) << i;
}

// Four ones over a four-dimensional null space, with entries spanning sixty
// orders of magnitude. Eigen's Jacobi under -ffast-math put a NaN in a
// null-space column of U on the original; the harness's finiteness and
// orthonormality checks are what catch that here.
TEST(Svd, PoisonThetaLikeMatrixHasCleanNullSpace) {
  const int n = 8;
  const std::vector<Complex> m = autonne_test::poison_theta_like();
  const SvdResult r = run_svd(m.data(), n, n, MatrixOrder::ColMajor);
  ASSERT_TRUE(r.ok);
  EXPECT_TRUE(SvdAccepted(check(m.data(), n, n, MatrixOrder::ColMajor, r)));
  for (int i = 0; i < 4; ++i) EXPECT_NEAR(r.s[static_cast<std::size_t>(i)], 1.0, 32.0 * n * kEps) << i;
  for (int i = 4; i < 8; ++i) EXPECT_LE(r.s[static_cast<std::size_t>(i)], 1e-48) << i;
}

// --- scale -----------------------------------------------------------------

TEST(Svd, HandlesTinyAndHugeSpectra) {
  const SvdCase tiny = make_svd_case(4, 4, {1e-12, 1e-13, 1e-14, 1e-15}, MatrixOrder::ColMajor, 55);
  const SvdResult rt = run_svd(tiny);
  ASSERT_TRUE(rt.ok);
  EXPECT_TRUE(SvdAccepted(check(tiny, rt)));
  EXPECT_TRUE(SpectrumClose(rt.s, tiny.s, spectrum_tol(4, 4, 1e-12), true));

  const SvdCase huge = make_svd_case(4, 4, {1e12, 1e11, 1e10, 1e9}, MatrixOrder::ColMajor, 56);
  const SvdResult rh = run_svd(huge);
  ASSERT_TRUE(rh.ok);
  EXPECT_TRUE(SvdAccepted(check(huge, rh)));
  EXPECT_TRUE(SpectrumClose(rh.s, huge.s, spectrum_tol(4, 4, 1e12), true));

  // Near the ends of the range the harness can still measure (its squared
  // norms must not overflow or underflow).
  const SvdCase deep = make_svd_case(3, 3, {1e-150, 1e-151, 1e-152}, MatrixOrder::ColMajor, 57);
  const SvdResult rd = run_svd(deep);
  ASSERT_TRUE(rd.ok);
  EXPECT_TRUE(SvdAccepted(check(deep, rd)));
  EXPECT_TRUE(SpectrumClose(rd.s, deep.s, spectrum_tol(3, 3, 1e-150), true));

  const SvdCase high = make_svd_case(3, 3, {1e150, 1e149, 1e148}, MatrixOrder::ColMajor, 58);
  const SvdResult rg = run_svd(high);
  ASSERT_TRUE(rg.ok);
  EXPECT_TRUE(SvdAccepted(check(high, rg)));
  EXPECT_TRUE(SpectrumClose(rg.s, high.s, spectrum_tol(3, 3, 1e150), true));
}

// The kernel scales its input by a power of two before any arithmetic, so a
// matrix and its power-of-two multiple must produce bit-identical factors and
// singular values that differ by exactly that power.
TEST(Svd, ScalingByAPowerOfTwoIsExact) {
  const SvdCase c = make_svd_case(5, 4, {3.0, 2.0, 1.0, 0.5}, MatrixOrder::ColMajor, 606);
  const SvdResult base = run_svd(c);
  ASSERT_TRUE(base.ok);

  for (const int exponent : {-600, -40, 40, 600}) {
    const double factor = std::ldexp(1.0, exponent);
    std::vector<Complex> scaled(c.m.size());
    for (std::size_t i = 0; i < c.m.size(); ++i) scaled[i] = c.m[i] * factor;
    const SvdResult r = run_svd(scaled.data(), c.rows, c.cols, c.order);
    ASSERT_TRUE(r.ok) << exponent;
    EXPECT_EQ(r.u, base.u) << exponent;
    EXPECT_EQ(r.v, base.v) << exponent;
    ASSERT_EQ(r.s.size(), base.s.size());
    for (std::size_t i = 0; i < r.s.size(); ++i) {
      EXPECT_EQ(r.s[i], base.s[i] * factor) << exponent << " value " << i;
    }
  }
}

// --- relative accuracy on graded input -------------------------------------
//
// For M = B D with every singular value of B in [0.9, 1.1], each singular
// value of M lies in [0.9, 1.1] times the matching entry of D (sorted). That
// bound holds however small D gets, so a kernel that only promises absolute
// accuracy eps * ||M|| fails it as soon as d_k drops below eps. One-sided
// Jacobi preceded by a column-pivoted QR is known to keep it (Demmel &
// Veselic 1992; Drmac & Veselic 2008).

TEST(Svd, ColumnScaledMatrixKeepsRelativeAccuracy) {
  const int n = 8;
  const std::vector<double> d = {1.0, 1e-10, 1e-20, 1e-30, 1e-40, 1e-50, 1e-60, 1e-70};
  std::vector<Complex> m = autonne_test::well_conditioned_matrix(n, 2001);
  autonne_test::scale_columns(m, n, n, d);
  const SvdResult r = run_svd(m.data(), n, n, MatrixOrder::ColMajor);
  ASSERT_TRUE(r.ok);
  EXPECT_TRUE(SvdAccepted(check(m.data(), n, n, MatrixOrder::ColMajor, r)));
  EXPECT_TRUE(SpectrumWithinFactor(r.s, d, 0.9, 1.1, true));
}

TEST(Svd, RowScaledMatrixKeepsRelativeAccuracy) {
  const int n = 8;
  const std::vector<double> d = {1e-70, 1e-60, 1e-50, 1e-40, 1e-30, 1e-20, 1e-10, 1.0};
  std::vector<Complex> m = autonne_test::well_conditioned_matrix(n, 2002);
  autonne_test::scale_rows(m, n, n, d);
  const SvdResult r = run_svd(m.data(), n, n, MatrixOrder::ColMajor);
  ASSERT_TRUE(r.ok);
  EXPECT_TRUE(SvdAccepted(check(m.data(), n, n, MatrixOrder::ColMajor, r)));
  EXPECT_TRUE(SpectrumWithinFactor(r.s, d, 0.9, 1.1, true));
}

// The tail is what a truncating caller sums. Here it is far enough above eps
// to be checked in relative terms: the discarded weight computed from the
// individual values must agree with the constructed one.
TEST(Svd, TailIsReportedAccuratelyEnoughToSum) {
  const SvdCase c = make_svd_case(6, 4, {1.0, 1e-4, 1e-6, 1e-7}, MatrixOrder::ColMajor, 31337);
  const SvdResult r = run_svd(c);
  ASSERT_TRUE(r.ok);
  EXPECT_TRUE(SvdAccepted(check(c, r)));
  const double tail = r.s[2] * r.s[2] + r.s[3] * r.s[3];
  const double expected = 1e-12 + 1e-14;
  EXPECT_NEAR(tail, expected, 1e-8 * expected);
}

// --- the awkward corners of the exponent range -----------------------------

// A subnormal entry beside ordinary ones.
//
// The kernel scales its input so the largest component lands in [0.5, 1), but
// that says nothing about the smallest: an entry near the bottom of the
// subnormal range stays there. The Householder step forms the phase of the
// leading entry of each column, and a phase computed from a subnormal value
// by dividing through its modulus is not unimodular -- the modulus itself is
// rounded onto the subnormal grid. The reflection is then asserted rather
// than computed, so R is wrong while Q stays unitary, and the factorisation
// comes back with a large backward error.
TEST(Svd, FactorsMatricesCarryingASubnormalEntry) {
  const double t = std::ldexp(1.0, -1074);  // the smallest positive double
  std::vector<Complex> m = {
      Complex(8.0 * t, 5.0 * t),
      Complex(0.9346819330656273, -0.52682436833153679),
      Complex(-0.099011449652049763, 0.14929178335087376),
      Complex(8.0 * t, 2.0 * t),
  };
  const SvdResult r = run_svd(m.data(), 2, 2, MatrixOrder::ColMajor);
  ASSERT_TRUE(r.ok);
  EXPECT_TRUE(SvdAccepted(check(m.data(), 2, 2, MatrixOrder::ColMajor, r)));

  // The same thing in a larger shape, with one subnormal among ordinary
  // entries rather than two.
  std::vector<Complex> big = random_matrix(3, 4, 8181);
  big[2] = Complex(8.0 * t, 8.0 * t);
  const SvdResult rb = run_svd(big.data(), 3, 4, MatrixOrder::ColMajor);
  ASSERT_TRUE(rb.ok);
  EXPECT_TRUE(SvdAccepted(check(big.data(), 3, 4, MatrixOrder::ColMajor, rb)));
}

// A pair of columns whose inner product sits just above the rotation
// threshold while the rotation that would fix it underflows: cos comes out
// exactly 1 and the update degenerates to a phase multiply, which flips the
// sign of one column without changing the magnitude of the inner product. The
// pair then cycles forever and the sweep budget runs out, so the kernel
// refuses an ordinary well-conditioned matrix. This 2x2 is one such case.
TEST(Svd, DoesNotStallOnAPairThatCannotBeImproved) {
  const std::vector<Complex> m = {
      Complex(-0.29125996458617709, -0.27330429881289592),
      Complex(0.030033115161182761, 0.19209323043486018),
      Complex(-0.17752614758532445, 0.080546835854100676),
      Complex(0.21093110527014736, -0.14286197298506795),
  };
  const SvdResult r = run_svd(m.data(), 2, 2, MatrixOrder::ColMajor);
  ASSERT_TRUE(r.ok) << "the kernel refused an ordinary 2x2";
  EXPECT_TRUE(SvdAccepted(check(m.data(), 2, 2, MatrixOrder::ColMajor, r)));
}

// The same property over a wide sweep: an ordinary random matrix must not be
// refused. The rate of the stall above is around one in ten thousand at these
// sizes, so a few thousand cases is enough to catch a regression.
TEST(Svd, AcceptsEveryOrdinaryRandomMatrix) {
  int refused = 0;
  for (int trial = 0; trial < 20000; ++trial) {
    const int n = 2 + (trial % 5);
    const std::vector<Complex> m =
        random_matrix(n, n, static_cast<std::uint64_t>(trial) + 500000u);
    const SvdResult r = run_svd(m.data(), n, n, MatrixOrder::ColMajor);
    if (!r.ok) {
      ++refused;
      if (refused <= 3) ADD_FAILURE() << "refused trial " << trial << " at " << n << "x" << n;
    }
  }
  EXPECT_EQ(refused, 0);
}

// A singular value far below the old column floor but comfortably
// representable. The floor exists because the column norm is formed by
// summing squares, which underflows below about 1e-154; a value at 1e-121 is
// nowhere near that, and discarding it contradicts the relative-accuracy
// claim on the most sharply scaled matrix there is.
TEST(Svd, KeepsSingularValuesFarBelowOne) {
  for (const int exponent : {-395, -400, -450, -490}) {
    const double tiny = std::ldexp(1.0, exponent);
    const std::vector<Complex> m = {Complex(1.0, 0.0), Complex(0.0, 0.0),
                                    Complex(0.0, 0.0), Complex(tiny, 0.0)};
    const SvdResult r = run_svd(m.data(), 2, 2, MatrixOrder::ColMajor);
    ASSERT_TRUE(r.ok) << exponent;
    EXPECT_TRUE(SvdAccepted(check(m.data(), 2, 2, MatrixOrder::ColMajor, r))) << exponent;
    EXPECT_EQ(r.s[0], 1.0) << exponent;
    EXPECT_NEAR(r.s[1], tiny, 1e-12 * tiny) << exponent;
  }
}

// A row or column made entirely of subnormal entries is not structurally
// zero, and must not be treated as one differently by different builds. The
// concern is a denormals-are-zero rounding mode, which MSVC's /fp:fast
// selects: the structural-zero test compares entries against 0.0, and if
// subnormal operands read as zero there the kernel would silently take a
// different path. Measured identical across MSVC /fp:strict, MSVC /fp:fast
// and Clang -ffast-math, so this pins it rather than reporting it.
TEST(Svd, TreatsSubnormalRowsAndColumnsConsistently) {
  const double t = std::ldexp(1.0, -1074);

  std::vector<Complex> subnormal_row = {Complex(3 * t, 0.0), Complex(0.5, 0.25),
                                        Complex(5 * t, 0.0), Complex(-0.25, 0.5)};
  const SvdResult rr = run_svd(subnormal_row.data(), 2, 2, MatrixOrder::ColMajor);
  ASSERT_TRUE(rr.ok);
  EXPECT_TRUE(SvdAccepted(check(subnormal_row.data(), 2, 2, MatrixOrder::ColMajor, rr)));

  std::vector<Complex> subnormal_col = {Complex(0.5, 0.25), Complex(-0.25, 0.5),
                                        Complex(3 * t, 0.0), Complex(5 * t, 0.0)};
  const SvdResult rc = run_svd(subnormal_col.data(), 2, 2, MatrixOrder::ColMajor);
  ASSERT_TRUE(rc.ok);
  EXPECT_TRUE(SvdAccepted(check(subnormal_col.data(), 2, 2, MatrixOrder::ColMajor, rc)));

  // A matrix that is entirely subnormal is scaled back into the normal range
  // like any other, so its spectrum comes out at the right magnitude rather
  // than collapsing to zero.
  std::vector<Complex> all_subnormal = {Complex(3 * t, 0.0), Complex(0.0, 0.0),
                                        Complex(0.0, 0.0), Complex(5 * t, 0.0)};
  const SvdResult ra = run_svd(all_subnormal.data(), 2, 2, MatrixOrder::ColMajor);
  ASSERT_TRUE(ra.ok);
  EXPECT_TRUE(SvdAccepted(check(all_subnormal.data(), 2, 2, MatrixOrder::ColMajor, ra)));
  EXPECT_EQ(ra.s[0], 5.0 * t);
  EXPECT_EQ(ra.s[1], 3.0 * t);
}

// --- size ------------------------------------------------------------------

TEST(Svd, FactorsLargeRandomShapes) {
  for (const auto& shape : {std::pair<int, int>{128, 128}, {128, 32}, {32, 128}, {100, 7}, {7, 100}, {64, 64}}) {
    const int rows = shape.first;
    const int cols = shape.second;
    const std::vector<Complex> m = random_matrix(rows, cols, static_cast<std::uint64_t>(rows * 1000 + cols));
    const SvdResult r = run_svd(m.data(), rows, cols, MatrixOrder::ColMajor);
    ASSERT_TRUE(r.ok) << rows << "x" << cols;
    EXPECT_TRUE(SvdAccepted(check(m.data(), rows, cols, MatrixOrder::ColMajor, r)))
        << rows << "x" << cols;
  }
}

// A large matrix with a heavily degenerate, rank-deficient spectrum: the
// regime the kernel exists for, at the top of its size range.
TEST(Svd, FactorsLargeDegenerateRankDeficientMatrix) {
  std::vector<double> spectrum(128, 0.0);
  for (int i = 0; i < 16; ++i) spectrum[static_cast<std::size_t>(i)] = 1.0;
  for (int i = 16; i < 48; ++i) spectrum[static_cast<std::size_t>(i)] = 0.25;
  for (int i = 48; i < 64; ++i) spectrum[static_cast<std::size_t>(i)] = 1e-9;
  const SvdCase c = make_svd_case(128, 128, spectrum, MatrixOrder::ColMajor, 128128);
  const SvdResult r = run_svd(c);
  ASSERT_TRUE(r.ok);
  EXPECT_TRUE(SvdAccepted(check(c, r)));
  EXPECT_TRUE(SpectrumClose(r.s, c.s, spectrum_tol(128, 128, 1.0), true));
}

}  // namespace
