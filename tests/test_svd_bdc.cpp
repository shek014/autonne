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

// The divide-and-conquer SVD kernel against the harness and against spectra
// known by construction. Every case is judged by check_svd first (that is
// the contract) and then by the sharper property the fixture makes
// available: the exact spectrum to an absolute tolerance, exact zeros, or
// bitwise invariance. Nothing here asks for relative accuracy on a graded
// spectrum, because this kernel does not promise it; the test that pins
// what it does promise is ValuesAreAccurateAbsolutely.
//
// Compiled into every test variant, so each case runs under strict and
// fast-math floating point.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
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
using autonne_test::SvdAccepted;
using autonne_test::SvdCase;
using autonne_test::SvdResult;
using autonne_test::make_svd_case;
using autonne_test::random_matrix;
using autonne_test::run_svd_bdc;

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
// rounding perturbs the true singular values by a few ulps of the largest,
// and the kernel adds an error of the same order.
double spectrum_tol(int rows, int cols, double s_max) {
  const int dim = rows > cols ? rows : cols;
  return 32.0 * static_cast<double>(dim) * kEps * s_max;
}

// --- shapes and orders -----------------------------------------------------

TEST(SvdBdc, FactorsSquareMatrix) {
  const SvdCase c = make_svd_case(4, 4, {8.0, 4.0, 2.0, 1.0}, MatrixOrder::ColMajor, 12345);
  const SvdResult r = run_svd_bdc(c);
  ASSERT_TRUE(r.ok);
  EXPECT_TRUE(SvdAccepted(check(c, r)));
  EXPECT_TRUE(SpectrumClose(r.s, c.s, spectrum_tol(4, 4, 8.0), true));
}

// Tall on both sides of the QR crossover (6 x 3 is factored first, 5 x 4 is
// bidiagonalised directly), and the wide transposes of both.
TEST(SvdBdc, FactorsTallAndWideMatrices) {
  for (const auto& shape : {std::pair<int, int>{6, 3}, {5, 4}, {3, 6}, {4, 5}, {16, 10}, {10, 16}}) {
    const int rows = shape.first;
    const int cols = shape.second;
    const int k = rows < cols ? rows : cols;
    std::vector<double> spectrum(static_cast<std::size_t>(k));
    for (int i = 0; i < k; ++i) spectrum[static_cast<std::size_t>(i)] = 5.0 / (1.0 + i);
    const SvdCase c = make_svd_case(rows, cols, spectrum, MatrixOrder::ColMajor,
                                    static_cast<std::uint64_t>(700 + rows * 10 + cols));
    const SvdResult r = run_svd_bdc(c);
    ASSERT_TRUE(r.ok) << rows << "x" << cols;
    EXPECT_TRUE(SvdAccepted(check(c, r))) << rows << "x" << cols;
    EXPECT_TRUE(SpectrumClose(r.s, c.s, spectrum_tol(rows, cols, 5.0), true))
        << rows << "x" << cols;
  }
}

TEST(SvdBdc, FactorsVectorsAndScalars) {
  const std::vector<Complex> scalar = {Complex(3.0, -4.0)};
  const SvdResult rs = run_svd_bdc(scalar.data(), 1, 1, MatrixOrder::ColMajor);
  ASSERT_TRUE(rs.ok);
  EXPECT_TRUE(SvdAccepted(check(scalar.data(), 1, 1, MatrixOrder::ColMajor, rs)));
  EXPECT_NEAR(rs.s[0], 5.0, 4.0 * kEps * 5.0);

  const std::vector<Complex> column = random_matrix(7, 1, 41);
  double norm_sq = 0.0;
  for (const Complex& z : column) norm_sq += std::norm(z);
  const SvdResult rc = run_svd_bdc(column.data(), 7, 1, MatrixOrder::ColMajor);
  ASSERT_TRUE(rc.ok);
  EXPECT_TRUE(SvdAccepted(check(column.data(), 7, 1, MatrixOrder::ColMajor, rc)));
  EXPECT_NEAR(rc.s[0], std::sqrt(norm_sq), 16.0 * kEps * std::sqrt(norm_sq));

  const SvdResult rr = run_svd_bdc(column.data(), 1, 7, MatrixOrder::RowMajor);
  ASSERT_TRUE(rr.ok);
  EXPECT_TRUE(SvdAccepted(check(column.data(), 1, 7, MatrixOrder::RowMajor, rr)));
  EXPECT_NEAR(rr.s[0], std::sqrt(norm_sq), 16.0 * kEps * std::sqrt(norm_sq));
}

// The same logical matrix in both storage orders must give the same factors
// bit for bit: the kernel copies into its own layout before any arithmetic.
TEST(SvdBdc, StorageOrderDoesNotChangeTheResult) {
  const SvdCase col = make_svd_case(5, 3, {9.0, 3.0, 1.5}, MatrixOrder::ColMajor, 4242);
  std::vector<Complex> row(col.m.size());
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 3; ++j) {
      row[static_cast<std::size_t>(i * 3 + j)] = col.m[static_cast<std::size_t>(j * 5 + i)];
    }
  }
  const SvdResult rc = run_svd_bdc(col);
  const SvdResult rr = run_svd_bdc(row.data(), 5, 3, MatrixOrder::RowMajor);
  ASSERT_TRUE(rc.ok);
  ASSERT_TRUE(rr.ok);
  EXPECT_TRUE(SvdAccepted(check(col, rc)));
  EXPECT_TRUE(SvdAccepted(check(row.data(), 5, 3, MatrixOrder::RowMajor, rr)));
  EXPECT_EQ(rc.s, rr.s);
  EXPECT_EQ(rc.u, rr.u);
  EXPECT_EQ(rc.v, rr.v);
}

TEST(SvdBdc, IsDeterministic) {
  const SvdCase c = make_svd_case(40, 33, std::vector<double>(33, 1.0), MatrixOrder::ColMajor, 99);
  const SvdResult a = run_svd_bdc(c);
  const SvdResult b = run_svd_bdc(c);
  ASSERT_TRUE(a.ok);
  ASSERT_TRUE(b.ok);
  EXPECT_EQ(a.s, b.s);
  EXPECT_EQ(a.u, b.u);
  EXPECT_EQ(a.v, b.v);
}

// --- degenerate and rank-deficient spectra ---------------------------------

TEST(SvdBdc, ResolvesDegenerateSpectrum) {
  const SvdCase c = make_svd_case(5, 5, {3.0, 3.0, 3.0, 1.0, 1.0}, MatrixOrder::ColMajor, 91);
  const SvdResult r = run_svd_bdc(c);
  ASSERT_TRUE(r.ok);
  EXPECT_TRUE(SvdAccepted(check(c, r)));
  EXPECT_TRUE(SpectrumClose(r.s, c.s, spectrum_tol(5, 5, 3.0), true));
}

// The zero singular values of a rank-deficient matrix come out at the
// level of the absolute tolerance, not exactly zero: nothing is structural
// about them.
TEST(SvdBdc, ResolvesRankDeficientSpectrum) {
  const SvdCase c = make_svd_case(5, 4, {2.0, 1.0, 0.0, 0.0}, MatrixOrder::ColMajor, 92);
  const SvdResult r = run_svd_bdc(c);
  ASSERT_TRUE(r.ok);
  EXPECT_TRUE(SvdAccepted(check(c, r)));
  EXPECT_TRUE(SpectrumClose(r.s, c.s, spectrum_tol(5, 4, 2.0), true));
}

TEST(SvdBdc, FactorsTheZeroMatrix) {
  for (const auto& shape : {std::pair<int, int>{3, 3}, {5, 2}, {2, 5}, {1, 1}}) {
    const int rows = shape.first;
    const int cols = shape.second;
    const std::vector<Complex> zero(static_cast<std::size_t>(rows * cols), Complex(0.0, 0.0));
    const SvdResult r = run_svd_bdc(zero.data(), rows, cols, MatrixOrder::ColMajor);
    ASSERT_TRUE(r.ok) << rows << "x" << cols;
    EXPECT_TRUE(SvdAccepted(check(zero.data(), rows, cols, MatrixOrder::ColMajor, r)))
        << rows << "x" << cols;
    for (const double s : r.s) EXPECT_EQ(s, 0.0);
  }
}

// Every singular value of a unitary is one: the fully degenerate case, in
// which every merge deflates everything.
TEST(SvdBdc, UnitaryInputHasFlatSpectrum) {
  for (const int n : {8, 40}) {
    const std::vector<Complex> f = autonne_test::dft_unitary(n);
    const SvdResult r = run_svd_bdc(f.data(), n, n, MatrixOrder::ColMajor);
    ASSERT_TRUE(r.ok) << n;
    EXPECT_TRUE(SvdAccepted(check(f.data(), n, n, MatrixOrder::ColMajor, r))) << n;
    EXPECT_TRUE(SpectrumClose(r.s, std::vector<double>(static_cast<std::size_t>(n), 1.0),
                              32.0 * n * kEps, true))
        << n;
  }
}

// Twelve-fold degenerate, rank 12 of 36: the shape that Eigen's
// divide-and-conquer got wrong on the original (a spectrum summing to
// 0.9861 against a norm of 1.0). The zero columns are structural, so
// their values are exact even here.
TEST(SvdBdc, SimonCosetMatrixHasExactSpectrum) {
  const int n = 36;
  const std::vector<Complex> m = autonne_test::simon_coset_matrix();
  const SvdResult r = run_svd_bdc(m.data(), n, n, MatrixOrder::ColMajor);
  ASSERT_TRUE(r.ok);
  EXPECT_TRUE(SvdAccepted(check(m.data(), n, n, MatrixOrder::ColMajor, r)));

  const double sigma = std::sqrt(3.0) * (1.0 / 6.0);
  std::vector<double> expected(36, 0.0);
  for (int i = 0; i < 12; ++i) expected[static_cast<std::size_t>(i)] = sigma;
  EXPECT_TRUE(SpectrumClose(r.s, expected, 32.0 * n * kEps * sigma, true));
  for (int i = 12; i < 36; ++i) EXPECT_EQ(r.s[static_cast<std::size_t>(i)], 0.0) << i;

  double energy = 0.0;
  for (const double s : r.s) energy += s * s;
  EXPECT_NEAR(energy, 1.0, 64.0 * n * kEps);
}

// With a rounding residue in one entry the matrix is no longer structurally
// rank 12, and the tail must stay at the level of that residue.
TEST(SvdBdc, SimonCosetMatrixWithResidueKeepsTinyTail) {
  const int n = 36;
  const std::vector<Complex> m = autonne_test::simon_coset_matrix(3.4e-17);
  const SvdResult r = run_svd_bdc(m.data(), n, n, MatrixOrder::ColMajor);
  ASSERT_TRUE(r.ok);
  EXPECT_TRUE(SvdAccepted(check(m.data(), n, n, MatrixOrder::ColMajor, r)));

  const double sigma = std::sqrt(3.0) * (1.0 / 6.0);
  for (int i = 0; i < 12; ++i) {
    EXPECT_NEAR(r.s[static_cast<std::size_t>(i)], sigma, 32.0 * n * kEps * sigma) << i;
  }
  for (int i = 12; i < 36; ++i) EXPECT_LE(r.s[static_cast<std::size_t>(i)], 1e-15) << i;
}

// A block-diagonal input keeps its structure exactly through the
// bidiagonalisation: a reflector formed on a column that is zero outside
// its block is zero there too, so the superdiagonal entry at the boundary
// is an exact zero and the solver treats the blocks independently. The
// spectrum is the union of the blocks' spectra.
TEST(SvdBdc, BlockDiagonalInputDecouplesExactly) {
  const int na = 20;
  const int nb = 24;
  std::vector<double> sa(static_cast<std::size_t>(na));
  std::vector<double> sb(static_cast<std::size_t>(nb));
  for (int i = 0; i < na; ++i) sa[static_cast<std::size_t>(i)] = 7.0 - 0.3 * i;
  for (int i = 0; i < nb; ++i) sb[static_cast<std::size_t>(i)] = 6.85 - 0.25 * i;
  const SvdCase a = make_svd_case(na, na, sa, MatrixOrder::ColMajor, 71);
  const SvdCase b = make_svd_case(nb, nb, sb, MatrixOrder::ColMajor, 72);
  const int n = na + nb;
  std::vector<Complex> m(static_cast<std::size_t>(n * n), Complex(0.0, 0.0));
  for (int j = 0; j < na; ++j) {
    for (int i = 0; i < na; ++i) {
      m[static_cast<std::size_t>(i + j * n)] = a.m[static_cast<std::size_t>(i + j * na)];
    }
  }
  for (int j = 0; j < nb; ++j) {
    for (int i = 0; i < nb; ++i) {
      m[static_cast<std::size_t>(na + i + (na + j) * n)] = b.m[static_cast<std::size_t>(i + j * nb)];
    }
  }
  const SvdResult r = run_svd_bdc(m.data(), n, n, MatrixOrder::ColMajor);
  ASSERT_TRUE(r.ok);
  EXPECT_TRUE(SvdAccepted(check(m.data(), n, n, MatrixOrder::ColMajor, r)));
  std::vector<double> expected = a.s;
  expected.insert(expected.end(), b.s.begin(), b.s.end());
  EXPECT_TRUE(SpectrumClose(r.s, expected, spectrum_tol(n, n, 7.0), true));
}

// --- structural zeros ------------------------------------------------------

TEST(SvdBdc, StructurallyZeroRowsAndColumnsGiveExactZeros) {
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
  const SvdResult r = run_svd_bdc(m.data(), rows, cols, MatrixOrder::ColMajor);
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

// --- scale -----------------------------------------------------------------

TEST(SvdBdc, ScalingByAPowerOfTwoIsExact) {
  const SvdCase c = make_svd_case(5, 4, {3.0, 2.0, 1.0, 0.5}, MatrixOrder::ColMajor, 606);
  const SvdResult base = run_svd_bdc(c);
  ASSERT_TRUE(base.ok);

  for (const int exponent : {-600, -40, 40, 600}) {
    const double factor = std::ldexp(1.0, exponent);
    std::vector<Complex> scaled(c.m.size());
    for (std::size_t i = 0; i < c.m.size(); ++i) scaled[i] = c.m[i] * factor;
    const SvdResult r = run_svd_bdc(scaled.data(), c.rows, c.cols, c.order);
    ASSERT_TRUE(r.ok) << exponent;
    EXPECT_EQ(r.u, base.u) << exponent;
    EXPECT_EQ(r.v, base.v) << exponent;
    ASSERT_EQ(r.s.size(), base.s.size());
    for (std::size_t i = 0; i < r.s.size(); ++i) {
      EXPECT_EQ(r.s[i], base.s[i] * factor) << exponent << " value " << i;
    }
  }
}

TEST(SvdBdc, HandlesTinyAndHugeSpectra) {
  const SvdCase tiny = make_svd_case(4, 4, {1e-12, 1e-13, 1e-14, 1e-15}, MatrixOrder::ColMajor, 55);
  const SvdResult rt = run_svd_bdc(tiny);
  ASSERT_TRUE(rt.ok);
  EXPECT_TRUE(SvdAccepted(check(tiny, rt)));
  EXPECT_TRUE(SpectrumClose(rt.s, tiny.s, spectrum_tol(4, 4, 1e-12), true));

  const SvdCase huge = make_svd_case(4, 4, {1e12, 1e11, 1e10, 1e9}, MatrixOrder::ColMajor, 56);
  const SvdResult rh = run_svd_bdc(huge);
  ASSERT_TRUE(rh.ok);
  EXPECT_TRUE(SvdAccepted(check(huge, rh)));
  EXPECT_TRUE(SpectrumClose(rh.s, huge.s, spectrum_tol(4, 4, 1e12), true));

  // Near the ends of the range the harness can still measure (its squared
  // norms must not overflow or underflow).
  const SvdCase deep = make_svd_case(3, 3, {1e-150, 1e-151, 1e-152}, MatrixOrder::ColMajor, 57);
  const SvdResult rd = run_svd_bdc(deep);
  ASSERT_TRUE(rd.ok);
  EXPECT_TRUE(SvdAccepted(check(deep, rd)));
  EXPECT_TRUE(SpectrumClose(rd.s, deep.s, spectrum_tol(3, 3, 1e-150), true));

  const SvdCase high = make_svd_case(3, 3, {1e150, 1e149, 1e148}, MatrixOrder::ColMajor, 58);
  const SvdResult rg = run_svd_bdc(high);
  ASSERT_TRUE(rg.ok);
  EXPECT_TRUE(SvdAccepted(check(high, rg)));
  EXPECT_TRUE(SpectrumClose(rg.s, high.s, spectrum_tol(3, 3, 1e150), true));
}

TEST(SvdBdc, FactorsMatricesCarryingASubnormalEntry) {
  const double t = std::ldexp(1.0, -1074);  // the smallest positive double
  std::vector<Complex> m = {
      Complex(8.0 * t, 5.0 * t),
      Complex(0.9346819330656273, -0.52682436833153679),
      Complex(-0.099011449652049763, 0.14929178335087376),
      Complex(8.0 * t, 2.0 * t),
  };
  const SvdResult r = run_svd_bdc(m.data(), 2, 2, MatrixOrder::ColMajor);
  ASSERT_TRUE(r.ok);
  EXPECT_TRUE(SvdAccepted(check(m.data(), 2, 2, MatrixOrder::ColMajor, r)));

  std::vector<Complex> big = random_matrix(3, 4, 8181);
  big[2] = Complex(8.0 * t, 8.0 * t);
  const SvdResult rb = run_svd_bdc(big.data(), 3, 4, MatrixOrder::ColMajor);
  ASSERT_TRUE(rb.ok);
  EXPECT_TRUE(SvdAccepted(check(big.data(), 3, 4, MatrixOrder::ColMajor, rb)));
}

// --- the accuracy the contract states --------------------------------------
//
// A spectrum spanning twenty decades. Every value is returned within
// 64 dim eps s_max of the truth, which is what the contract says, and the
// values under that bound are not claimed to carry anything more: the test
// asserts the absolute bound only, where svd_thin's graded-input tests
// assert a relative one.
TEST(SvdBdc, ValuesAreAccurateAbsolutely) {
  const std::vector<double> spectrum = {1.0, 1e-4, 1e-8, 1e-12, 1e-16, 1e-20};
  const SvdCase c = make_svd_case(8, 6, spectrum, MatrixOrder::ColMajor, 2020);
  const SvdResult r = run_svd_bdc(c);
  ASSERT_TRUE(r.ok);
  EXPECT_TRUE(SvdAccepted(check(c, r)));
  const double bound = 64.0 * 8 * kEps * 1.0;
  for (std::size_t i = 0; i < spectrum.size(); ++i) {
    EXPECT_NEAR(r.s[i], spectrum[i], bound) << i;
  }
}

// --- breadth ---------------------------------------------------------------

// Square matrices of every order from 1 to 40, several of each: below the
// leaf size, at it, and through two merge levels above it.
TEST(SvdBdc, AcceptsEveryOrdinaryRandomMatrix) {
  int refused = 0;
  int rejected = 0;
  for (int trial = 0; trial < 400; ++trial) {
    const int n = 1 + (trial % 40);
    const std::vector<Complex> m =
        random_matrix(n, n, static_cast<std::uint64_t>(trial) + 600000u);
    const SvdResult r = run_svd_bdc(m.data(), n, n, MatrixOrder::ColMajor);
    if (!r.ok) {
      ++refused;
      if (refused <= 3) ADD_FAILURE() << "refused trial " << trial << " at " << n << "x" << n;
      continue;
    }
    const SvdReport report = check(m.data(), n, n, MatrixOrder::ColMajor, r);
    if (!report.ok()) {
      ++rejected;
      if (rejected <= 3) {
        ADD_FAILURE() << "trial " << trial << " at " << n << "x" << n << ": "
                      << SvdAccepted(report).message();
      }
    }
  }
  EXPECT_EQ(refused, 0);
  EXPECT_EQ(rejected, 0);
}

TEST(SvdBdc, FactorsLargeRandomShapes) {
  for (const auto& shape : {std::pair<int, int>{128, 128}, {128, 32}, {32, 128}, {100, 7},
                            {7, 100}, {64, 64}, {96, 64}, {64, 96}, {160, 100}}) {
    const int rows = shape.first;
    const int cols = shape.second;
    const std::vector<Complex> m =
        random_matrix(rows, cols, static_cast<std::uint64_t>(rows * 1000 + cols));
    const SvdResult r = run_svd_bdc(m.data(), rows, cols, MatrixOrder::ColMajor);
    ASSERT_TRUE(r.ok) << rows << "x" << cols;
    EXPECT_TRUE(SvdAccepted(check(m.data(), rows, cols, MatrixOrder::ColMajor, r)))
        << rows << "x" << cols;
  }
}

// A large matrix with a heavily degenerate, rank-deficient spectrum, at the
// top of the size range: the case in which Eigen's divide-and-conquer is
// rejected by the harness on some machines.
TEST(SvdBdc, FactorsLargeDegenerateRankDeficientMatrix) {
  std::vector<double> spectrum(128, 0.0);
  for (int i = 0; i < 16; ++i) spectrum[static_cast<std::size_t>(i)] = 1.0;
  for (int i = 16; i < 48; ++i) spectrum[static_cast<std::size_t>(i)] = 0.25;
  for (int i = 48; i < 64; ++i) spectrum[static_cast<std::size_t>(i)] = 1e-9;
  const SvdCase c = make_svd_case(128, 128, spectrum, MatrixOrder::ColMajor, 128128);
  const SvdResult r = run_svd_bdc(c);
  ASSERT_TRUE(r.ok);
  EXPECT_TRUE(SvdAccepted(check(c, r)));
  EXPECT_TRUE(SpectrumClose(r.s, c.s, spectrum_tol(128, 128, 1.0), true));
}

// Both kernels on the same input agree on the spectrum to the absolute
// tolerance: the one property the two contracts share.
TEST(SvdBdc, AgreesWithTheJacobiKernelOnOrdinaryInput) {
  for (const auto& shape : {std::pair<int, int>{48, 48}, {70, 30}, {30, 70}}) {
    const int rows = shape.first;
    const int cols = shape.second;
    const std::vector<Complex> m =
        random_matrix(rows, cols, static_cast<std::uint64_t>(rows * 77 + cols));
    const SvdResult jacobi = autonne_test::run_svd(m.data(), rows, cols, MatrixOrder::ColMajor);
    const SvdResult bdc = run_svd_bdc(m.data(), rows, cols, MatrixOrder::ColMajor);
    ASSERT_TRUE(jacobi.ok) << rows << "x" << cols;
    ASSERT_TRUE(bdc.ok) << rows << "x" << cols;
    EXPECT_TRUE(SpectrumClose(bdc.s, jacobi.s, spectrum_tol(rows, cols, jacobi.s[0]), true))
        << rows << "x" << cols;
  }
}

}  // namespace
