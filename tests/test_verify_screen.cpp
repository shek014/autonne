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

// The screen against the harness it stands in for. Two halves, as for the
// harness itself: the screen accepts what is correct, by construction and
// from both kernels, and it rejects each way of corrupting a factorisation,
// including the one only a residual can see (two singular vectors swapped,
// which leaves the spectrum and both bases intact). The closing tests hold
// the screen to the harness verdict for verdict over a population of cases.

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "autonne/autonne.hpp"
#include "autonne/verify.hpp"
#include "test_support.hpp"

namespace {

using autonne::MatrixOrder;
using autonne::verify::SvdScreenReport;
using autonne::verify::check_svd;
using autonne::verify::screen_svd;
using autonne_test::Complex;
using autonne_test::SvdCase;
using autonne_test::SvdResult;
using autonne_test::bits_of;
using autonne_test::kPositiveInfBits;
using autonne_test::kQuietNanBits;
using autonne_test::make_svd_case;
using autonne_test::poke_bits;
using autonne_test::random_matrix;
using autonne_test::run_svd;
using autonne_test::run_svd_bdc;

SvdScreenReport screen(const SvdCase& c, int probes = 3) {
  return screen_svd(c.m.data(), c.rows, c.cols, c.order, c.u.data(), c.s.data(),
                    c.v.data(), c.k, autonne::verify::Tolerances(), probes);
}

SvdScreenReport screen(const Complex* m, int rows, int cols, MatrixOrder order,
                       const SvdResult& r) {
  const int k = rows < cols ? rows : cols;
  return screen_svd(m, rows, cols, order, r.u.data(), r.s.data(), r.v.data(), k);
}

::testing::AssertionResult ScreenAccepted(const SvdScreenReport& r) {
  if (r.ok()) return ::testing::AssertionSuccess();
  return ::testing::AssertionFailure()
         << "screen_svd rejected a correct factorisation:"
         << "\n  inputs_valid   = " << r.inputs_valid
         << "\n  finite         = " << r.finite
         << "\n  nonnegative    = " << r.nonnegative
         << "\n  descending     = " << r.descending
         << "\n  energy_ok      = " << r.energy_ok << "  (defect "
         << r.energy_defect << " vs bound " << r.energy_bound << ")"
         << "\n  backward_ok    = " << r.backward_ok << "  (estimate "
         << r.residual_estimate << " vs bound " << r.backward_bound << ")"
         << "\n  u_orthonormal  = " << r.u_orthonormal << "  (" << r.u_ortho_estimate
         << " vs " << r.ortho_bound << ")"
         << "\n  v_orthonormal  = " << r.v_orthonormal << "  (" << r.v_ortho_estimate
         << " vs " << r.ortho_bound << ")";
}

// --- acceptance -----------------------------------------------------------

TEST(VerifyScreen, AcceptsExactFactorisations) {
  for (const auto& shape : {std::pair<int, int>{4, 4}, {6, 3}, {3, 6}, {1, 1}, {1, 9}, {9, 1}}) {
    const int rows = shape.first;
    const int cols = shape.second;
    const int k = rows < cols ? rows : cols;
    std::vector<double> spectrum(static_cast<std::size_t>(k));
    for (int i = 0; i < k; ++i) spectrum[static_cast<std::size_t>(i)] = 8.0 / (1 << i);
    for (const MatrixOrder order : {MatrixOrder::ColMajor, MatrixOrder::RowMajor}) {
      const SvdCase c = make_svd_case(rows, cols, spectrum, order,
                                      static_cast<std::uint64_t>(1000 + rows * 10 + cols));
      EXPECT_TRUE(ScreenAccepted(screen(c))) << rows << "x" << cols;
    }
  }
}

// Output of both kernels, at the top of the size range, in both orders.
TEST(VerifyScreen, AcceptsBothKernelsOutput) {
  for (const auto& shape : {std::pair<int, int>{128, 128}, {128, 32}, {32, 128}, {17, 17}}) {
    const int rows = shape.first;
    const int cols = shape.second;
    std::vector<Complex> m = random_matrix(rows, cols, static_cast<std::uint64_t>(rows * 100 + cols));
    for (const MatrixOrder order : {MatrixOrder::ColMajor, MatrixOrder::RowMajor}) {
      // The buffer is reinterpreted in the other order: still some matrix.
      const SvdResult jacobi = run_svd(m.data(), rows, cols, order);
      ASSERT_TRUE(jacobi.ok);
      EXPECT_TRUE(ScreenAccepted(screen(m.data(), rows, cols, order, jacobi)))
          << "jacobi " << rows << "x" << cols;
      const SvdResult bdc = run_svd_bdc(m.data(), rows, cols, order);
      ASSERT_TRUE(bdc.ok);
      EXPECT_TRUE(ScreenAccepted(screen(m.data(), rows, cols, order, bdc)))
          << "bdc " << rows << "x" << cols;
    }
  }
}

TEST(VerifyScreen, AcceptsDegenerateAndRankDeficientSpectra) {
  const SvdCase degenerate = make_svd_case(5, 5, {3.0, 3.0, 3.0, 1.0, 1.0}, MatrixOrder::ColMajor, 91);
  EXPECT_TRUE(ScreenAccepted(screen(degenerate)));
  const SvdCase deficient = make_svd_case(5, 4, {2.0, 1.0, 0.0, 0.0}, MatrixOrder::ColMajor, 92);
  EXPECT_TRUE(ScreenAccepted(screen(deficient)));
}

TEST(VerifyScreen, AcceptsTruncatedFactorisation) {
  const SvdCase full = make_svd_case(6, 4, {8.0, 4.0, 2.0, 1.0}, MatrixOrder::ColMajor, 31337);
  const int k = 2;
  const SvdScreenReport r = screen_svd(full.m.data(), full.rows, full.cols, full.order,
                                       full.u.data(), full.s.data(), full.v.data(), k);
  EXPECT_TRUE(ScreenAccepted(r));
  EXPECT_TRUE(r.truncated);
  // discarded = 2^2 + 1^2 = 5. The probes never see it: M V_2 = U_2 S_2
  // holds to rounding, so the estimate sits at the rounding level while
  // the whole residual is sqrt 5.
  EXPECT_NEAR(r.discarded_energy, 5.0, 1e-9);
  EXPECT_LT(r.residual_estimate, 1e-12);
}

// The same truncation with the kept factors corrupted: a swapped pair of
// kept vectors and a perturbed kept value. Both live inside the kept
// subspace, which is exactly what the probes look at.
TEST(VerifyScreen, RejectsCorruptedTruncatedFactorisation) {
  const SvdCase full = make_svd_case(6, 4, {8.0, 4.0, 2.0, 1.0}, MatrixOrder::ColMajor, 31338);
  const int k = 2;

  SvdCase swapped = full;
  for (int i = 0; i < 6; ++i) {
    std::swap(swapped.u[static_cast<std::size_t>(0 * 6 + i)],
              swapped.u[static_cast<std::size_t>(1 * 6 + i)]);
  }
  const SvdScreenReport rs = screen_svd(swapped.m.data(), 6, 4, swapped.order, swapped.u.data(),
                                        swapped.s.data(), swapped.v.data(), k);
  EXPECT_TRUE(rs.energy_ok);
  EXPECT_FALSE(rs.backward_ok);
  EXPECT_FALSE(rs.ok());

  SvdCase perturbed = full;
  perturbed.s[1] *= 0.999;  // downward, so the one-sided energy check still passes
  const SvdScreenReport rp = screen_svd(perturbed.m.data(), 6, 4, perturbed.order,
                                        perturbed.u.data(), perturbed.s.data(),
                                        perturbed.v.data(), k);
  EXPECT_TRUE(rp.energy_ok);
  EXPECT_FALSE(rp.backward_ok);
  EXPECT_FALSE(rp.ok());
}

// --- rejection ------------------------------------------------------------

TEST(VerifyScreen, RejectsNonFiniteFactors) {
  SvdCase c = make_svd_case(4, 4, {8.0, 4.0, 2.0, 1.0}, MatrixOrder::ColMajor, 500);
  ASSERT_TRUE(screen(c).ok());

  SvdCase bad_u = c;
  poke_bits(bad_u.u[5], kQuietNanBits, bits_of(0.0));
  EXPECT_FALSE(screen(bad_u).finite);
  EXPECT_FALSE(screen(bad_u).ok());

  SvdCase bad_v = c;
  poke_bits(bad_v.v[9], bits_of(0.5), kPositiveInfBits);
  EXPECT_FALSE(screen(bad_v).finite);
  EXPECT_FALSE(screen(bad_v).ok());

  SvdCase bad_s = c;
  poke_bits(bad_s.s[2], kQuietNanBits);
  const SvdScreenReport r = screen(bad_s);
  EXPECT_FALSE(r.finite);
  EXPECT_FALSE(r.nonnegative);
  EXPECT_FALSE(r.descending);
  EXPECT_FALSE(r.ok());
}

TEST(VerifyScreen, RejectsPerturbedSingularValue) {
  SvdCase c = make_svd_case(4, 4, {8.0, 4.0, 2.0, 1.0}, MatrixOrder::ColMajor, 2024);
  ASSERT_TRUE(screen(c).ok());

  c.s[1] *= 1.001;  // 4.0 -> 4.004

  const SvdScreenReport r = screen(c);
  EXPECT_TRUE(r.finite);
  EXPECT_FALSE(r.energy_ok);
  EXPECT_FALSE(r.backward_ok);
  EXPECT_GT(r.residual_estimate, r.backward_bound);
  EXPECT_FALSE(r.ok());
}

// A perturbation far smaller than the value it sits on. The energy identity
// sees 2 s delta = 2e-9 against a bound of 64 * 4 * eps * 85, and the
// residual probe sees delta itself against 64 * 4 * eps * 9.2.
TEST(VerifyScreen, RejectsFinelyPerturbedSingularValue) {
  SvdCase c = make_svd_case(4, 4, {8.0, 4.0, 2.0, 1.0}, MatrixOrder::ColMajor, 2025);
  ASSERT_TRUE(screen(c).ok());

  c.s[3] += 1e-9;

  const SvdScreenReport r = screen(c);
  EXPECT_FALSE(r.energy_ok);
  EXPECT_FALSE(r.backward_ok);
  EXPECT_FALSE(r.ok());
}

// Two columns of U exchanged. The spectrum is untouched, so the energy
// identity holds; U is still orthonormal, so both orthonormality probes
// pass; only the reconstruction is wrong, and only the residual probe can
// see that. This is the case the screen must not be without.
TEST(VerifyScreen, RejectsSwappedSingularVectors) {
  SvdCase c = make_svd_case(6, 4, {8.0, 4.0, 2.0, 1.0}, MatrixOrder::ColMajor, 2026);
  ASSERT_TRUE(screen(c).ok());

  for (int i = 0; i < 6; ++i) {
    std::swap(c.u[static_cast<std::size_t>(0 * 6 + i)], c.u[static_cast<std::size_t>(1 * 6 + i)]);
  }

  const SvdScreenReport r = screen(c);
  EXPECT_TRUE(r.energy_ok);
  EXPECT_TRUE(r.u_orthonormal);
  EXPECT_TRUE(r.v_orthonormal);
  EXPECT_FALSE(r.backward_ok);
  EXPECT_FALSE(r.ok());
}

TEST(VerifyScreen, RejectsNonOrthogonalU) {
  SvdCase c = make_svd_case(4, 4, {8.0, 4.0, 2.0, 1.0}, MatrixOrder::ColMajor, 606);
  ASSERT_TRUE(screen(c).ok());

  for (int i = 0; i < 4; ++i) c.u[static_cast<std::size_t>(1 * 4 + i)] *= 1.5;

  const SvdScreenReport r = screen(c);
  EXPECT_TRUE(r.finite);
  EXPECT_FALSE(r.u_orthonormal);
  EXPECT_GT(r.u_ortho_estimate, r.ortho_bound);
  EXPECT_TRUE(r.v_orthonormal);
  EXPECT_FALSE(r.ok());
}

// Column norms intact, two columns of U made parallel: only the off-diagonal
// of U^* U gives it away, and the probe sees it through U^* (U y) - y.
TEST(VerifyScreen, RejectsUWithDependentColumns) {
  SvdCase c = make_svd_case(4, 4, {8.0, 4.0, 2.0, 1.0}, MatrixOrder::ColMajor, 607);
  ASSERT_TRUE(screen(c).ok());

  for (int i = 0; i < 4; ++i) {
    c.u[static_cast<std::size_t>(1 * 4 + i)] = c.u[static_cast<std::size_t>(0 * 4 + i)];
  }

  const SvdScreenReport r = screen(c);
  EXPECT_FALSE(r.u_orthonormal);
  EXPECT_FALSE(r.ok());
}

TEST(VerifyScreen, RejectsNonOrthogonalV) {
  SvdCase c = make_svd_case(4, 4, {8.0, 4.0, 2.0, 1.0}, MatrixOrder::ColMajor, 608);
  ASSERT_TRUE(screen(c).ok());

  for (int i = 0; i < 4; ++i) c.v[static_cast<std::size_t>(2 * 4 + i)] *= 0.5;

  const SvdScreenReport r = screen(c);
  EXPECT_FALSE(r.v_orthonormal);
  EXPECT_TRUE(r.u_orthonormal);
  EXPECT_FALSE(r.ok());
}

// Built ascending with the vectors permuted to match, so that every other
// check passes and the ordering verdict is isolated.
TEST(VerifyScreen, RejectsSpectrumOutOfDescendingOrder) {
  const SvdCase c = make_svd_case(4, 4, {1.0, 2.0, 4.0, 8.0}, MatrixOrder::ColMajor, 609);
  const SvdScreenReport r = screen(c);
  EXPECT_TRUE(r.energy_ok);
  EXPECT_TRUE(r.backward_ok);
  EXPECT_TRUE(r.u_orthonormal);
  EXPECT_TRUE(r.v_orthonormal);
  EXPECT_FALSE(r.descending);
  EXPECT_FALSE(r.ok());
}

TEST(VerifyScreen, RejectsNegativeSingularValue) {
  SvdCase c = make_svd_case(4, 4, {8.0, 4.0, 2.0, 1.0}, MatrixOrder::ColMajor, 610);
  ASSERT_TRUE(screen(c).ok());
  c.s[3] = -1.0;
  const SvdScreenReport r = screen(c);
  EXPECT_FALSE(r.nonnegative);
  EXPECT_FALSE(r.ok());
}

// A factorisation of a different matrix altogether: values right by
// construction (both spectra have the same energy), vectors unrelated.
TEST(VerifyScreen, RejectsFactorsOfAnotherMatrix) {
  const SvdCase c = make_svd_case(8, 8, {4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0},
                                  MatrixOrder::ColMajor, 611);
  const SvdCase other = make_svd_case(8, 8, {4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0},
                                      MatrixOrder::ColMajor, 612);
  const SvdScreenReport r = screen_svd(c.m.data(), c.rows, c.cols, c.order, other.u.data(),
                                       other.s.data(), other.v.data(), c.k);
  EXPECT_TRUE(r.energy_ok);
  EXPECT_TRUE(r.u_orthonormal);
  EXPECT_TRUE(r.v_orthonormal);
  EXPECT_FALSE(r.backward_ok);
  EXPECT_FALSE(r.ok());
}

// --- scale ----------------------------------------------------------------
//
// The screen measures on the input scaled by the harness's exponent, and
// these are the harness's own scale tests: the tiny direction is the one in
// which an unscaled screen would accept anything.

TEST(VerifyScreen, AcceptsACorrectFactorisationOfAHugeMatrix) {
  for (const int exponent : {700, 900, 1000}) {
    const double factor = std::ldexp(1.0, exponent);
    SvdCase c = make_svd_case(5, 4, {8.0, 4.0, 2.0, 1.0}, MatrixOrder::ColMajor, 3001);
    for (Complex& z : c.m) z *= factor;
    for (double& x : c.s) x *= factor;
    EXPECT_TRUE(ScreenAccepted(screen(c))) << "2^" << exponent;
  }
}

TEST(VerifyScreen, RejectsAWrongFactorisationOfATinyMatrix) {
  for (const int exponent : {-700, -900, -1000}) {
    const double factor = std::ldexp(1.0, exponent);
    SvdCase c = make_svd_case(5, 4, {8.0, 4.0, 2.0, 1.0}, MatrixOrder::ColMajor, 3002);
    for (Complex& z : c.m) z *= factor;
    for (double& x : c.s) x *= factor;
    ASSERT_TRUE(screen(c).ok()) << "2^" << exponent;

    SvdCase wrong = c;
    wrong.s[0] *= 2.0;
    const SvdScreenReport r = screen(wrong);
    EXPECT_FALSE(r.backward_ok) << "2^" << exponent;
    EXPECT_FALSE(r.ok()) << "2^" << exponent;

    SvdCase other = make_svd_case(5, 4, {1.0, 1.0, 1.0, 1.0}, MatrixOrder::ColMajor, 3003);
    for (double& x : other.s) x *= factor;
    const SvdScreenReport r2 = screen_svd(c.m.data(), c.rows, c.cols, c.order, other.u.data(),
                                          other.s.data(), other.v.data(), c.k);
    EXPECT_FALSE(r2.ok()) << "2^" << exponent;
  }
}

TEST(VerifyScreen, MeasuresMatricesBelowTheNormalRange) {
  for (const int exponent : {-1024, -1025, -1030, -1060}) {
    const double tiny = std::ldexp(1.0, exponent);
    const std::vector<Complex> m = {Complex(tiny, 0.0)};
    const std::vector<Complex> u = {Complex(1.0, 0.0)};
    const std::vector<double> sv = {tiny};
    const std::vector<Complex> v = {Complex(1.0, 0.0)};
    const SvdScreenReport r = screen_svd(m.data(), 1, 1, MatrixOrder::ColMajor, u.data(),
                                         sv.data(), v.data(), 1);
    EXPECT_TRUE(ScreenAccepted(r)) << "2^" << exponent;
  }
}

TEST(VerifyScreen, RejectsAWrongFactorisationBelowTheNormalRange) {
  for (const int exponent : {-1025, -1030, -1060}) {
    const double tiny = std::ldexp(1.0, exponent);
    const std::vector<Complex> m = {Complex(tiny, 0.0), Complex(0.0, 0.0),
                                    Complex(0.0, 0.0), Complex(tiny, 0.0)};
    const std::vector<Complex> u = {Complex(1.0, 0.0), Complex(0.0, 0.0),
                                    Complex(0.0, 0.0), Complex(1.0, 0.0)};
    const std::vector<double> sv = {3.0 * tiny, 0.25 * tiny};
    const std::vector<Complex> v = u;
    const SvdScreenReport r = screen_svd(m.data(), 2, 2, MatrixOrder::ColMajor, u.data(),
                                         sv.data(), v.data(), 2);
    EXPECT_FALSE(r.ok()) << "2^" << exponent;
    EXPECT_FALSE(r.backward_ok) << "2^" << exponent;
  }
}

// --- input validation -----------------------------------------------------

TEST(VerifyScreen, RejectsMalformedArguments) {
  const SvdCase c = make_svd_case(4, 4, {8.0, 4.0, 2.0, 1.0}, MatrixOrder::ColMajor, 1);
  const autonne::verify::Tolerances tol;
  auto call = [&](const Complex* m, int rows, int cols, const Complex* u, const double* s,
                  const Complex* v, int k, int probes) {
    return screen_svd(m, rows, cols, c.order, u, s, v, k, tol, probes);
  };
  EXPECT_FALSE(call(nullptr, 4, 4, c.u.data(), c.s.data(), c.v.data(), 4, 3).inputs_valid);
  EXPECT_FALSE(call(c.m.data(), 0, 4, c.u.data(), c.s.data(), c.v.data(), 4, 3).inputs_valid);
  EXPECT_FALSE(call(c.m.data(), 4, -1, c.u.data(), c.s.data(), c.v.data(), 4, 3).inputs_valid);
  EXPECT_FALSE(call(c.m.data(), 4, 4, nullptr, c.s.data(), c.v.data(), 4, 3).inputs_valid);
  EXPECT_FALSE(call(c.m.data(), 4, 4, c.u.data(), nullptr, c.v.data(), 4, 3).inputs_valid);
  EXPECT_FALSE(call(c.m.data(), 4, 4, c.u.data(), c.s.data(), nullptr, 4, 3).inputs_valid);
  EXPECT_FALSE(call(c.m.data(), 4, 4, c.u.data(), c.s.data(), c.v.data(), 0, 3).inputs_valid);
  EXPECT_FALSE(call(c.m.data(), 4, 4, c.u.data(), c.s.data(), c.v.data(), 5, 3).inputs_valid);
  EXPECT_FALSE(call(c.m.data(), 4, 4, c.u.data(), c.s.data(), c.v.data(), 4, 0).inputs_valid);
  EXPECT_TRUE(call(c.m.data(), 4, 4, c.u.data(), c.s.data(), c.v.data(), 4, 3).ok());
  EXPECT_TRUE(call(c.m.data(), 4, 4, c.u.data(), c.s.data(), c.v.data(), 4, 1).ok());
}

// --- repeatability and probe count ----------------------------------------

TEST(VerifyScreen, IsRepeatableBitForBit) {
  const std::vector<Complex> m = random_matrix(40, 30, 4040);
  const SvdResult r = run_svd_bdc(m.data(), 40, 30, MatrixOrder::ColMajor);
  ASSERT_TRUE(r.ok);
  const SvdScreenReport a = screen(m.data(), 40, 30, MatrixOrder::ColMajor, r);
  const SvdScreenReport b = screen(m.data(), 40, 30, MatrixOrder::ColMajor, r);
  EXPECT_TRUE(a.ok());
  EXPECT_EQ(a.residual_estimate, b.residual_estimate);
  EXPECT_EQ(a.u_ortho_estimate, b.u_ortho_estimate);
  EXPECT_EQ(a.v_ortho_estimate, b.v_ortho_estimate);
  EXPECT_EQ(a.energy_defect, b.energy_defect);
}

// The gross corruptions above are caught by a single probe: the estimate
// would have to land ten orders of magnitude under the truth to miss.
TEST(VerifyScreen, OneProbeCatchesGrossDefects) {
  SvdCase c = make_svd_case(6, 5, {8.0, 4.0, 2.0, 1.0, 0.5}, MatrixOrder::ColMajor, 777);
  ASSERT_TRUE(screen(c, 1).ok());

  SvdCase swapped = c;
  for (int i = 0; i < 6; ++i) {
    std::swap(swapped.u[static_cast<std::size_t>(0 * 6 + i)],
              swapped.u[static_cast<std::size_t>(1 * 6 + i)]);
  }
  EXPECT_FALSE(screen(swapped, 1).backward_ok);

  SvdCase stretched = c;
  for (int i = 0; i < 6; ++i) stretched.u[static_cast<std::size_t>(2 * 6 + i)] *= 1.5;
  EXPECT_FALSE(screen(stretched, 1).u_orthonormal);

  SvdCase shrunk = c;
  for (int i = 0; i < 5; ++i) shrunk.v[static_cast<std::size_t>(3 * 5 + i)] *= 0.5;
  EXPECT_FALSE(screen(shrunk, 1).v_orthonormal);
}

// --- agreement with the harness --------------------------------------------

// Over a population of shapes and seeds, the screen and the harness give
// the same verdict on correct factorisations from both kernels and on each
// corruption. The corruptions are gross by the harness's standard (a factor
// on a value or a column, a swap), which is the regime a screen is for; a
// defect sitting at the bound is the harness's to judge.
TEST(VerifyScreen, AgreesWithTheHarnessOverAPopulation) {
  int disagreements = 0;
  int cases = 0;
  auto compare = [&](const Complex* m, int rows, int cols, MatrixOrder order, const Complex* u,
                     const double* s, const Complex* v, int k, const char* what) {
    const bool harness = check_svd(m, rows, cols, order, u, s, v, k).ok();
    const bool screened = screen_svd(m, rows, cols, order, u, s, v, k).ok();
    ++cases;
    if (harness != screened) {
      ++disagreements;
      if (disagreements <= 5) {
        ADD_FAILURE() << what << " at " << rows << "x" << cols << ": harness " << harness
                      << ", screen " << screened;
      }
    }
  };
  for (int trial = 0; trial < 60; ++trial) {
    const int rows = 2 + (trial * 7) % 23;
    const int cols = 2 + (trial * 11) % 19;
    const int k = rows < cols ? rows : cols;
    const MatrixOrder order = (trial % 2 == 0) ? MatrixOrder::ColMajor : MatrixOrder::RowMajor;
    const std::vector<Complex> m = random_matrix(rows, cols, static_cast<std::uint64_t>(9000 + trial));
    for (int which = 0; which < 2; ++which) {
      SvdResult r = which == 0 ? run_svd(m.data(), rows, cols, order)
                               : run_svd_bdc(m.data(), rows, cols, order);
      ASSERT_TRUE(r.ok) << trial;
      compare(m.data(), rows, cols, order, r.u.data(), r.s.data(), r.v.data(), k, "correct");

      SvdResult value = r;
      value.s[static_cast<std::size_t>(k - 1)] += 1e-6 * value.s[0];
      compare(m.data(), rows, cols, order, value.u.data(), value.s.data(), value.v.data(), k,
              "perturbed value");

      if (k >= 2) {
        SvdResult swapped = r;
        for (int i = 0; i < rows; ++i) {
          std::swap(swapped.u[static_cast<std::size_t>(0 * rows + i)],
                    swapped.u[static_cast<std::size_t>(1 * rows + i)]);
        }
        compare(m.data(), rows, cols, order, swapped.u.data(), swapped.s.data(), swapped.v.data(),
                k, "swapped vectors");
      }

      SvdResult stretched = r;
      for (int i = 0; i < cols; ++i) {
        stretched.v[static_cast<std::size_t>((k - 1) * cols + i)] *= 1.0 + 1e-6;
      }
      compare(m.data(), rows, cols, order, stretched.u.data(), stretched.s.data(),
              stretched.v.data(), k, "stretched column");
    }
  }
  EXPECT_EQ(disagreements, 0) << "over " << cases << " cases";
}

}  // namespace
