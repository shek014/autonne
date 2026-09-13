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

// The real bidiagonal divide-and-conquer core, on its own, before any
// complex matrix is bidiagonalised into it. Every case is judged by
// check_svd on the real factors embedded as complex numbers, which is the
// contract, and then by whatever the fixture knows exactly: a closed-form
// spectrum, a decoupled block structure, a zero on the diagonal, or the
// leaf-only solve of the same matrix.
//
// Compiled into every test variant, so each case runs under strict and
// fast-math floating point.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <vector>

#include "autonne/verify.hpp"
#include "detail/bidiag_dc.hpp"
#include "report_matchers.hpp"
#include "test_support.hpp"

namespace {

using autonne::MatrixOrder;
using autonne::detail::bidiag_dc::Index;
using autonne::detail::bidiag_dc::at;
using autonne::detail::bidiag_dc::bidiag_svd;
using autonne::detail::bidiag_dc::secular_roots;
using autonne::detail::bidiag_dc::SecularRoot;
using autonne::verify::SvdReport;
using Complex = std::complex<double>;

constexpr double kEps = std::numeric_limits<double>::epsilon();

struct Bidiagonal {
  std::vector<double> d;
  std::vector<double> e;
  Index n() const { return d.size(); }
};

Bidiagonal random_bidiagonal(Index n, std::uint64_t seed) {
  autonne_test::Lcg rng(seed);
  Bidiagonal b;
  b.d.resize(n);
  b.e.resize(n > 0 ? n - 1 : 0);
  for (double& x : b.d) x = rng.next_uniform();
  for (double& x : b.e) x = rng.next_uniform();
  return b;
}

struct Solved {
  bool ok = false;
  std::vector<double> U;
  std::vector<double> s;
  std::vector<double> V;
};

Solved solve(const Bidiagonal& b, Index leaf) {
  Solved r;
  r.ok = bidiag_svd(b.d.data(), b.e.data(), b.n(), leaf, r.U, r.s, r.V);
  return r;
}

// The harness verdict on B = U diag(s) V^T, with everything embedded as
// complex numbers with zero imaginary parts.
SvdReport judge(const Bidiagonal& b, const Solved& r) {
  const Index n = b.n();
  std::vector<Complex> m(n * n, Complex(0.0, 0.0));
  std::vector<Complex> u(n * n);
  std::vector<Complex> v(n * n);
  for (Index i = 0; i < n; ++i) {
    m[at(i, i, n)] = Complex(b.d[i], 0.0);
    if (i + 1 < n) m[at(i, i + 1, n)] = Complex(b.e[i], 0.0);
  }
  for (Index k = 0; k < n * n; ++k) {
    u[k] = Complex(r.U[k], 0.0);
    v[k] = Complex(r.V[k], 0.0);
  }
  return autonne::verify::check_svd(m.data(), static_cast<int>(n), static_cast<int>(n),
                                    MatrixOrder::ColMajor, u.data(), r.s.data(), v.data(),
                                    static_cast<int>(n));
}

double frobenius(const Bidiagonal& b) {
  double acc = 0.0;
  for (const double x : b.d) acc += x * x;
  for (const double x : b.e) acc += x * x;
  return std::sqrt(acc);
}

// The absolute accuracy the core promises on a singular value.
double value_bound(const Bidiagonal& b) {
  return 64.0 * static_cast<double>(b.n()) * kEps * frobenius(b);
}

// --- the harness on every shape and leaf size ------------------------------

TEST(BidiagDc, RandomBidiagonalsPassTheHarness) {
  const Index sizes[] = {1, 2, 3, 4, 5, 7, 8, 13, 16, 17, 31, 32, 33, 64, 100, 128};
  const Index leaves[] = {2, 4, 16, 32};
  for (const Index n : sizes) {
    const Bidiagonal b = random_bidiagonal(n, 1000u + n);
    for (const Index leaf : leaves) {
      const Solved r = solve(b, leaf);
      ASSERT_TRUE(r.ok) << "n = " << n << ", leaf = " << leaf;
      EXPECT_TRUE(autonne_test::SvdAccepted(judge(b, r))) << "n = " << n << ", leaf = " << leaf;
    }
  }
}

TEST(BidiagDc, LeafSizeDoesNotChangeTheSpectrum) {
  const Bidiagonal b = random_bidiagonal(97, 42);
  const Solved coarse = solve(b, 2);
  const Solved fine = solve(b, 128);  // leaf-only: pivoted QR and Jacobi on the whole block
  ASSERT_TRUE(coarse.ok);
  ASSERT_TRUE(fine.ok);
  const double bound = value_bound(b);
  for (Index i = 0; i < b.n(); ++i) {
    EXPECT_NEAR(coarse.s[i], fine.s[i], bound) << "i = " << i;
  }
}

// --- spectra known exactly -------------------------------------------------

TEST(BidiagDc, OneByOneIsTheAbsoluteValue) {
  for (const double d : {0.75, -0.75, 0.0}) {
    const Bidiagonal b{{d}, {}};
    const Solved r = solve(b, 16);
    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.s[0], std::fabs(d));
    EXPECT_TRUE(autonne_test::SvdAccepted(judge(b, r)));
  }
}

TEST(BidiagDc, TwoByTwoMatchesTheClosedForm) {
  // [a b; 0 c]: sigma^2 = (t +- sqrt(t^2 - 4 a^2 c^2)) / 2, t = a^2 + b^2 + c^2.
  const double a = 0.6;
  const double bb = -0.8;
  const double c = 0.3;
  const Bidiagonal b{{a, c}, {bb}};
  for (const Index leaf : {2u, 16u}) {
    const Solved r = solve(b, leaf);
    ASSERT_TRUE(r.ok);
    const double t = a * a + bb * bb + c * c;
    const double disc = std::sqrt(t * t - 4.0 * a * a * c * c);
    const double big = std::sqrt((t + disc) / 2.0);
    const double small = std::sqrt((t - disc) / 2.0);
    EXPECT_NEAR(r.s[0], big, 8.0 * kEps * big);
    EXPECT_NEAR(r.s[1], small, 8.0 * kEps * big);
    EXPECT_TRUE(autonne_test::SvdAccepted(judge(b, r)));
  }
}

TEST(BidiagDc, DiagonalInputGivesSortedAbsoluteValues) {
  const Bidiagonal b{{0.3, -0.9, 0.0, 0.5, -0.1, 0.7}, {0.0, 0.0, 0.0, 0.0, 0.0}};
  const Solved r = solve(b, 2);
  ASSERT_TRUE(r.ok);
  const double expected[] = {0.9, 0.7, 0.5, 0.3, 0.1, 0.0};
  for (Index i = 0; i < b.n(); ++i) {
    EXPECT_NEAR(r.s[i], expected[i], 8.0 * kEps) << "i = " << i;
  }
  EXPECT_TRUE(autonne_test::SvdAccepted(judge(b, r)));
}

TEST(BidiagDc, ZeroSuperdiagonalsDecoupleTheBlocks) {
  // Three blocks of orders 7, 1 and 12; the spectrum is the union of theirs.
  Bidiagonal b = random_bidiagonal(20, 7);
  b.e[6] = 0.0;
  b.e[7] = 0.0;
  const Solved whole = solve(b, 4);
  ASSERT_TRUE(whole.ok);
  EXPECT_TRUE(autonne_test::SvdAccepted(judge(b, whole)));

  std::vector<double> expected;
  const Index starts[] = {0, 7, 8};
  const Index lengths[] = {7, 1, 12};
  for (int k = 0; k < 3; ++k) {
    Bidiagonal block;
    block.d.assign(b.d.begin() + static_cast<long>(starts[k]),
                   b.d.begin() + static_cast<long>(starts[k] + lengths[k]));
    block.e.assign(b.e.begin() + static_cast<long>(starts[k]),
                   b.e.begin() + static_cast<long>(starts[k] + lengths[k] - 1));
    const Solved part = solve(block, 4);
    ASSERT_TRUE(part.ok);
    expected.insert(expected.end(), part.s.begin(), part.s.end());
  }
  std::sort(expected.begin(), expected.end(), [](double x, double y) { return x > y; });
  const double bound = value_bound(b);
  for (Index i = 0; i < b.n(); ++i) {
    EXPECT_NEAR(whole.s[i], expected[i], bound) << "i = " << i;
  }
}

// --- the cases the merge has to get right ----------------------------------

TEST(BidiagDc, IdenticalHalvesDeflateEveryPair) {
  // Both halves have the same spectrum, so every d in the top merge appears
  // twice and every pair goes through the equal-d rotation.
  const std::vector<double> hd = {0.9, 0.3, 0.7, 0.1, 0.5, 0.8, 0.2, 0.6};
  const std::vector<double> he = {0.2, 0.4, 0.1, 0.3, 0.5, 0.2, 0.1};
  Bidiagonal b;
  b.d = hd;
  b.d.push_back(0.45);
  b.d.insert(b.d.end(), hd.begin(), hd.end());
  b.e = he;
  b.e.push_back(0.33);
  b.e.push_back(0.27);
  b.e.insert(b.e.end(), he.begin(), he.end());
  for (const Index leaf : {2u, 4u, 8u}) {
    const Solved r = solve(b, leaf);
    ASSERT_TRUE(r.ok) << "leaf = " << leaf;
    EXPECT_TRUE(autonne_test::SvdAccepted(judge(b, r))) << "leaf = " << leaf;
  }
}

TEST(BidiagDc, ZerosOnTheDiagonalMakeItSingular) {
  Bidiagonal b = random_bidiagonal(16, 3);
  for (const Index i : {1u, 3u, 6u, 8u, 11u, 14u}) b.d[i] = 0.0;
  for (const Index leaf : {2u, 4u, 16u}) {
    const Solved r = solve(b, leaf);
    ASSERT_TRUE(r.ok) << "leaf = " << leaf;
    EXPECT_TRUE(autonne_test::SvdAccepted(judge(b, r))) << "leaf = " << leaf;
    // det B = prod d_i = 0, so the smallest singular value is zero to the
    // core's absolute accuracy.
    EXPECT_LE(r.s[b.n() - 1], value_bound(b)) << "leaf = " << leaf;
  }
}

TEST(BidiagDc, GradedSpectrumIsAccurateAbsolutely) {
  // Sixteen decades. The leaf-only solve is pivoted QR plus Jacobi, which is
  // accurate relatively; the divide-and-conquer is held to the absolute
  // bound, which is its contract.
  Bidiagonal b;
  b.d.resize(32);
  b.e.resize(31);
  for (Index i = 0; i < 32; ++i) b.d[i] = std::pow(10.0, -16.0 * static_cast<double>(i) / 31.0);
  for (Index i = 0; i < 31; ++i) b.e[i] = 0.3 * b.d[i];
  const Solved reference = solve(b, 32);
  ASSERT_TRUE(reference.ok);
  for (const Index leaf : {2u, 16u}) {
    const Solved r = solve(b, leaf);
    ASSERT_TRUE(r.ok) << "leaf = " << leaf;
    EXPECT_TRUE(autonne_test::SvdAccepted(judge(b, r))) << "leaf = " << leaf;
    const double bound = value_bound(b);
    for (Index i = 0; i < b.n(); ++i) {
      EXPECT_NEAR(r.s[i], reference.s[i], bound) << "leaf = " << leaf << ", i = " << i;
    }
  }
}

// --- the secular solver on its own -----------------------------------------

double secular(const std::vector<double>& d, const std::vector<double>& z, double sigma) {
  double f = 1.0;
  for (Index j = 0; j < d.size(); ++j) f += z[j] * z[j] / ((d[j] - sigma) * (d[j] + sigma));
  return f;
}

TEST(BidiagDc, SecularRootsInterlaceAndSolveTheEquation) {
  const std::vector<double> d = {0.0, 0.2, 0.5, 0.9};
  const std::vector<double> z = {0.3, -0.1, 0.4, 0.2};
  std::vector<SecularRoot> roots;
  ASSERT_TRUE(secular_roots(d.data(), z.data(), 4, roots));
  double z_norm = 0.0;
  for (const double x : z) z_norm += x * x;
  z_norm = std::sqrt(z_norm);
  for (Index i = 0; i < 4; ++i) {
    const double lower = d[i];
    const double upper = (i + 1 < 4) ? d[i + 1] : d[3] + z_norm;
    EXPECT_GT(roots[i].sigma, lower) << "i = " << i;
    EXPECT_LT(roots[i].sigma, upper) << "i = " << i;
    // |f| at the root, against the rounding in evaluating it there.
    double scale = 1.0;
    for (Index j = 0; j < 4; ++j) {
      scale += std::fabs(z[j] * z[j] / ((d[j] - roots[i].sigma) * (d[j] + roots[i].sigma)));
    }
    EXPECT_LE(std::fabs(secular(d, z, roots[i].sigma)), 64.0 * kEps * scale) << "i = " << i;
  }
}

TEST(BidiagDc, SecularRootIsResolvedInsideATinyGap) {
  // Two poles thirty ulps apart, just above the deflation tolerance: the
  // root between them must still satisfy the equation to rounding, which
  // needs its offset from the nearer pole to relative accuracy.
  const double gap = 30.0 * kEps * 0.5;
  const std::vector<double> d = {0.0, 0.25, 0.5, 0.5 + gap, 0.9};
  const std::vector<double> z = {0.3, 0.2, 0.4, 0.35, 0.2};
  std::vector<SecularRoot> roots;
  ASSERT_TRUE(secular_roots(d.data(), z.data(), 5, roots));
  EXPECT_GT(roots[2].sigma, d[2]);
  EXPECT_LT(roots[2].sigma, d[3]);
  // The stored differences carry the root's precision.
  double scale = 1.0;
  double f = 1.0;
  for (Index j = 0; j < 5; ++j) {
    const double term = z[j] * z[j] / (roots[2].delta[j] * roots[2].w[j]);
    scale += std::fabs(term);
    f += term;
  }
  EXPECT_LE(std::fabs(f), 64.0 * kEps * scale);
}

TEST(BidiagDc, SingleTermSecularRootIsInClosedForm) {
  const std::vector<double> d = {0.6};
  const std::vector<double> z = {0.8};
  std::vector<SecularRoot> roots;
  ASSERT_TRUE(secular_roots(d.data(), z.data(), 1, roots));
  EXPECT_NEAR(roots[0].sigma, 1.0, 4.0 * kEps);
}

}  // namespace
