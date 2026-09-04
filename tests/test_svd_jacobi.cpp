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

// svd_thin, judged two ways.
//
// Against the harness, which asks only whether the factorisation reconstructs
// the matrix it came from, and against a spectrum known independently. The
// second matters because the first cannot catch a systematic error: a
// factorisation can satisfy every residual bound and still report a spectrum
// that belongs to no correct answer, which is exactly the failure mode the
// frozen matrices below were captured for.
//
// Every case runs in each floating-point variant the suite builds. A kernel
// that is correct only under strict arithmetic has not met its contract.

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstddef>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include "autonne/autonne.hpp"
#include "autonne/hexfloat.hpp"
#include "autonne/verify.hpp"
#include "test_support.hpp"

namespace {

using autonne::MatrixOrder;
using autonne::svd_thin;
using autonne::verify::SvdReport;
using autonne::verify::check_svd;
using autonne_test::Complex;
using autonne_test::SvdCase;
using autonne_test::make_svd_case;

constexpr double kEps = std::numeric_limits<double>::epsilon();

int min_dim(int rows, int cols) { return rows < cols ? rows : cols; }

// Everything the interface hands back for one call, sized by the contract.
struct Factorisation {
  bool ok = false;
  std::vector<Complex> u;
  std::vector<double> s;
  std::vector<Complex> v;
};

Factorisation factor(const std::vector<Complex>& m, int rows, int cols,
                     MatrixOrder order) {
  const int k = min_dim(rows, cols);
  Factorisation f;
  f.u.assign(static_cast<std::size_t>(rows) * static_cast<std::size_t>(k),
             Complex(0.0, 0.0));
  f.s.assign(static_cast<std::size_t>(k), 0.0);
  f.v.assign(static_cast<std::size_t>(cols) * static_cast<std::size_t>(k),
             Complex(0.0, 0.0));
  f.ok = svd_thin(m.data(), rows, cols, order, f.u.data(), f.s.data(),
                  f.v.data());
  return f;
}

SvdReport judge(const std::vector<Complex>& m, int rows, int cols,
                MatrixOrder order, const Factorisation& f) {
  return check_svd(m.data(), rows, cols, order, f.u.data(), f.s.data(),
                   f.v.data(), min_dim(rows, cols));
}

// Absolute slack for comparing one singular value against a known one. Scaled
// by the largest value present, because the accuracy of a small singular value
// is governed by the norm of the whole matrix and not by its own size.
double spectrum_slack(int rows, int cols, double scale) {
  const double dim = static_cast<double>(rows > cols ? rows : cols);
  return 64.0 * dim * kEps * scale;
}

// Runs the whole contract for one constructed case: the harness accepts it,
// and the spectrum matches what the case was built from.
void expect_factors_to(int rows, int cols, const std::vector<double>& spectrum,
                       MatrixOrder order, std::uint64_t seed) {
  const SvdCase c = make_svd_case(rows, cols, spectrum, order, seed);
  const Factorisation f = factor(c.m, rows, cols, order);

  ASSERT_TRUE(f.ok) << rows << "x" << cols << " reported failure";

  const SvdReport r = judge(c.m, rows, cols, order, f);
  EXPECT_TRUE(r.ok()) << rows << "x" << cols
                      << ": residual " << r.residual << " against bound "
                      << r.backward_bound << ", U ortho " << r.u_ortho_residual
                      << ", V ortho " << r.v_ortho_residual
                      << ", energy defect " << r.energy_defect;

  double scale = 0.0;
  for (double s : spectrum) {
    if (s > scale) scale = s;
  }
  const double slack = spectrum_slack(rows, cols, scale);

  // make_svd_case takes the spectrum in the order given, and svd_thin returns
  // it descending, so the comparison sorts the expectation rather than
  // assuming the fixture supplied it that way.
  std::vector<double> expected = spectrum;
  for (std::size_t a = 0; a + 1 < expected.size(); ++a) {
    for (std::size_t b = a + 1; b < expected.size(); ++b) {
      if (expected[b] > expected[a]) {
        const double t = expected[a];
        expected[a] = expected[b];
        expected[b] = t;
      }
    }
  }

  ASSERT_EQ(f.s.size(), expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(f.s[i], expected[i], slack)
        << "singular value " << i << " of " << rows << "x" << cols;
  }
}

// ---------------------------------------------------------------------------
// Shape coverage
// ---------------------------------------------------------------------------

TEST(SvdJacobi, SquareFullRank) {
  expect_factors_to(6, 6, {4.0, 3.0, 2.0, 1.5, 1.0, 0.5},
                    MatrixOrder::ColMajor, 0x5eedu);
}

TEST(SvdJacobi, TallFullRank) {
  expect_factors_to(9, 4, {3.0, 2.0, 1.0, 0.25}, MatrixOrder::ColMajor, 0x1234u);
}

// The wide case exercises the adjoint path, where the factors are computed for
// A^* and exchanged on the way out. A transposition error there survives every
// residual check on square input and fails only here.
TEST(SvdJacobi, WideFullRank) {
  expect_factors_to(4, 9, {3.0, 2.0, 1.0, 0.25}, MatrixOrder::ColMajor, 0x1234u);
}

TEST(SvdJacobi, RowMajorInputMatchesColumnMajor) {
  expect_factors_to(7, 5, {2.0, 1.5, 1.0, 0.5, 0.125}, MatrixOrder::RowMajor,
                    0xabcdu);
}

TEST(SvdJacobi, SingleColumn) {
  expect_factors_to(5, 1, {2.5}, MatrixOrder::ColMajor, 0x11u);
}

TEST(SvdJacobi, SingleRow) {
  expect_factors_to(1, 5, {2.5}, MatrixOrder::ColMajor, 0x22u);
}

// ---------------------------------------------------------------------------
// Rank deficiency
// ---------------------------------------------------------------------------

// The columns of U past the rank have no direction to inherit from the working
// matrix, so they are completed rather than normalised. Dividing by a zero
// singular value instead would put a NaN in a null-space column, which is
// invisible to a caller inspecting only the leading block and is the specific
// defect this library exists to not have.
TEST(SvdJacobi, RankDeficientSquareKeepsOrthonormalNullSpace) {
  expect_factors_to(8, 8, {2.0, 1.0, 0.5, 0.25, 0.0, 0.0, 0.0, 0.0},
                    MatrixOrder::ColMajor, 0xdeadu);
}

TEST(SvdJacobi, RankOneOfMany) {
  expect_factors_to(6, 6, {1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                    MatrixOrder::ColMajor, 0xbeefu);
}

TEST(SvdJacobi, ZeroMatrixFactorsWithOrthonormalFactors) {
  const int n = 5;
  const std::vector<Complex> m(static_cast<std::size_t>(n * n),
                               Complex(0.0, 0.0));
  const Factorisation f = factor(m, n, n, MatrixOrder::ColMajor);
  ASSERT_TRUE(f.ok);

  const SvdReport r = judge(m, n, n, MatrixOrder::ColMajor, f);
  EXPECT_TRUE(r.finite);
  EXPECT_TRUE(r.u_orthonormal) << "U ortho residual " << r.u_ortho_residual;
  EXPECT_TRUE(r.v_orthonormal) << "V ortho residual " << r.v_ortho_residual;
  for (double s : f.s) EXPECT_DOUBLE_EQ(s, 0.0);
}

// Exact degeneracy is where a factorisation stops being unique: any rotation
// within the degenerate subspace is a correct answer. An implementation that
// assumes distinct values can return vectors that are individually plausible
// and jointly not orthogonal.
TEST(SvdJacobi, ExactlyDegenerateSpectrum) {
  expect_factors_to(8, 8, {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0},
                    MatrixOrder::ColMajor, 0x99u);
}

TEST(SvdJacobi, DegenerateOverNullSpace) {
  expect_factors_to(8, 6, {0.5, 0.5, 0.5, 0.0, 0.0, 0.0},
                    MatrixOrder::ColMajor, 0x77u);
}

// ---------------------------------------------------------------------------
// Graded spectra
// ---------------------------------------------------------------------------

// These establish that a widely graded input factors and verifies. They do NOT
// establish the high relative accuracy that motivated choosing this method,
// and it would be wrong to read them that way.
//
// Two reasons, both in the fixture rather than the kernel. The matrix is
// formed by multiplying the factors out, which perturbs every singular value
// by around eps * ||M||, so a value entered as 1e-160 is already near 1e-16 by
// the time the kernel sees it: the grading is destroyed before the test
// begins. And the comparison uses an absolute slack scaled by the LARGEST
// singular value, so the small end is unconstrained either way.
//
// Testing relative accuracy needs a fixture whose small singular values are
// exact by construction rather than computed, for instance a product of exact
// diagonal scalings with a well-conditioned factor. That is worth building and
// is not what these are.
TEST(SvdJacobi, WidelyGradedSpectrum) {
  expect_factors_to(6, 6, {1.0, 1e-3, 1e-6, 1e-9, 1e-12, 1e-15},
                    MatrixOrder::ColMajor, 0x314u);
}

TEST(SvdJacobi, SeverelyGradedSpectrum) {
  expect_factors_to(5, 5, {1.0, 1e-40, 1e-80, 1e-120, 1e-160},
                    MatrixOrder::ColMajor, 0x271u);
}

// ---------------------------------------------------------------------------
// Input guards
// ---------------------------------------------------------------------------

TEST(SvdJacobi, RejectsNullPointers) {
  std::vector<Complex> m(4, Complex(1.0, 0.0));
  std::vector<Complex> u(4);
  std::vector<double> s(2);
  std::vector<Complex> v(4);

  EXPECT_FALSE(svd_thin(nullptr, 2, 2, MatrixOrder::ColMajor, u.data(),
                        s.data(), v.data()));
  EXPECT_FALSE(svd_thin(m.data(), 2, 2, MatrixOrder::ColMajor, nullptr,
                        s.data(), v.data()));
  EXPECT_FALSE(svd_thin(m.data(), 2, 2, MatrixOrder::ColMajor, u.data(),
                        nullptr, v.data()));
  EXPECT_FALSE(svd_thin(m.data(), 2, 2, MatrixOrder::ColMajor, u.data(),
                        s.data(), nullptr));
}

TEST(SvdJacobi, RejectsNonPositiveDimensions) {
  std::vector<Complex> m(4, Complex(1.0, 0.0));
  std::vector<Complex> u(4);
  std::vector<double> s(2);
  std::vector<Complex> v(4);

  EXPECT_FALSE(svd_thin(m.data(), 0, 2, MatrixOrder::ColMajor, u.data(),
                        s.data(), v.data()));
  EXPECT_FALSE(svd_thin(m.data(), 2, -1, MatrixOrder::ColMajor, u.data(),
                        s.data(), v.data()));
}

// A NaN anywhere in the input spreads across every column a rotation touches,
// so it is rejected before the first sweep rather than factored around. The
// value is written as an object representation for the reason test_support
// documents: a NaN in a double-typed expression is not reliably a NaN in a
// translation unit built with -ffinite-math-only.
TEST(SvdJacobi, RejectsNonFiniteInput) {
  const int n = 4;
  std::vector<Complex> m(static_cast<std::size_t>(n * n), Complex(0.5, 0.0));
  std::vector<Complex> u(static_cast<std::size_t>(n * n));
  std::vector<double> s(static_cast<std::size_t>(n));
  std::vector<Complex> v(static_cast<std::size_t>(n * n));

  autonne_test::poke_bits(m[5], autonne_test::kQuietNanBits,
                          autonne_test::bits_of(0.0));
  EXPECT_FALSE(svd_thin(m.data(), n, n, MatrixOrder::ColMajor, u.data(),
                        s.data(), v.data()));

  // The infinity goes in the imaginary part, so the two halves of the complex
  // guard are each exercised by one of these cases rather than both by the
  // real part.
  m[5] = Complex(0.5, 0.0);
  autonne_test::poke_bits(m[9], autonne_test::bits_of(0.5),
                          autonne_test::kPositiveInfBits);
  EXPECT_FALSE(svd_thin(m.data(), n, n, MatrixOrder::ColMajor, u.data(),
                        s.data(), v.data()));
}

// ---------------------------------------------------------------------------
// Frozen corpus
// ---------------------------------------------------------------------------

// These two matrices are the reason the project exists. Both are exact
// hex-float literals rather than recipes, because a reproducer rebuilt by
// running a circuit through a library tracks that library: move one amplitude
// by an ulp and the defect stops reproducing. A literal is a fact.

#ifndef AUTONNE_TEST_DATA_DIR
#error "AUTONNE_TEST_DATA_DIR must be defined (see CMakeLists.txt)"
#endif

autonne::hexfloat::Matrix load_corpus(const std::string& name) {
  const std::string path = std::string(AUTONNE_TEST_DATA_DIR) + "/" + name;
  std::ifstream in(path);
  autonne::hexfloat::Matrix m;
  if (!in) return m;
  if (!autonne::hexfloat::read_matrix(in, m)) return autonne::hexfloat::Matrix();
  return m;
}

// Rank 4 over a four-dimensional null space, with magnitudes running from
// 7.07e-01 down to 1.57e-65. Both properties matter: the null space is where a
// normalisation by zero would land, and the dynamic range is where a method
// without high relative accuracy loses the small directions.
TEST(SvdJacobiCorpus, Rank4NullSpace8x8) {
  const autonne::hexfloat::Matrix m = load_corpus("rank4_nullspace_8x8.hexfloat");
  ASSERT_EQ(m.rows, 8);
  ASSERT_EQ(m.cols, 8);

  const Factorisation f = factor(m.data, m.rows, m.cols, m.order);
  ASSERT_TRUE(f.ok);

  const SvdReport r = judge(m.data, m.rows, m.cols, m.order, f);
  EXPECT_TRUE(r.ok()) << "residual " << r.residual << " against bound "
                      << r.backward_bound << ", U ortho " << r.u_ortho_residual;

  // Four ones and four zeros. Stated as a derivation from the matrix rather
  // than transcribed from a run: sum of squares is the Frobenius norm squared,
  // which is 4 here, spread over four unit values.
  const double slack = spectrum_slack(m.rows, m.cols, 1.0);
  for (int i = 0; i < 4; ++i) EXPECT_NEAR(f.s[static_cast<std::size_t>(i)], 1.0, slack);
  for (int i = 4; i < 8; ++i) EXPECT_LE(f.s[static_cast<std::size_t>(i)], slack);
}

// Rank 12, twelve-fold degenerate at 1/(2*sqrt(3)). Eigen 3.4.0 returned a
// spectrum summing to less than the Frobenius norm here, with a value near
// 0.263523 displacing one of the correct entries. That belongs to no correct
// factorisation of this matrix.
TEST(SvdJacobiCorpus, Degenerate12_36x36) {
  const autonne::hexfloat::Matrix m = load_corpus("degenerate12_36x36.hexfloat");
  ASSERT_EQ(m.rows, 36);
  ASSERT_EQ(m.cols, 36);

  const Factorisation f = factor(m.data, m.rows, m.cols, m.order);
  ASSERT_TRUE(f.ok);

  const SvdReport r = judge(m.data, m.rows, m.cols, m.order, f);
  EXPECT_TRUE(r.ok()) << "residual " << r.residual << " against bound "
                      << r.backward_bound << ", U ortho " << r.u_ortho_residual;

  // Twelve groups of three entries of magnitude 1/6 give sqrt(3)/6 each, which
  // is 1/(2*sqrt(3)). Derived, not observed.
  const double expected = 1.0 / (2.0 * std::sqrt(3.0));
  const double slack = spectrum_slack(m.rows, m.cols, expected);
  for (int i = 0; i < 12; ++i) {
    EXPECT_NEAR(f.s[static_cast<std::size_t>(i)], expected, slack)
        << "degenerate singular value " << i;
  }
  for (int i = 12; i < 36; ++i) {
    EXPECT_LE(f.s[static_cast<std::size_t>(i)], slack)
        << "null-space singular value " << i;
  }

  // The sum of squares is the whole point of the Eigen failure: a spectrum
  // that loses weight is one a truncation budget cannot be computed from.
  double energy = 0.0;
  for (double s : f.s) energy += s * s;
  EXPECT_NEAR(energy, 1.0, 64.0 * 36.0 * kEps);
}

}  // namespace
