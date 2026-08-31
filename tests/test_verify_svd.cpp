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

// Two halves: the harness accepts a factorisation that is correct by
// construction, and it rejects each way of corrupting one. The second half is
// what makes the first half mean anything.

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "autonne/verify.hpp"
#include "test_support.hpp"

namespace {

using autonne::MatrixOrder;
using autonne::verify::SvdReport;
using autonne::verify::check_svd;
using autonne_test::Complex;
using autonne_test::SvdCase;
using autonne_test::make_svd_case;
using autonne_test::opaque;
using autonne_test::quiet_nan;

SvdReport check(const SvdCase& c) {
  return check_svd(c.m.data(), c.rows, c.cols, c.order, c.u.data(), c.s.data(),
                   c.v.data(), c.k);
}

// Fails the whole report at once, with the measurements attached, so a
// regression says which bound moved rather than just "expected true".
::testing::AssertionResult ReportAccepted(const SvdReport& r) {
  if (r.ok()) return ::testing::AssertionSuccess();
  return ::testing::AssertionFailure()
         << "check_svd rejected a correct factorisation:"
         << "\n  inputs_valid   = " << r.inputs_valid
         << "\n  finite         = " << r.finite
         << "\n  nonnegative    = " << r.nonnegative
         << "\n  descending     = " << r.descending
         << "\n  backward_ok    = " << r.backward_ok << "  (residual "
         << r.residual << " vs bound " << r.backward_bound << ")"
         << "\n  u_orthonormal  = " << r.u_orthonormal << "  (" << r.u_ortho_residual
         << " vs " << r.ortho_bound << ")"
         << "\n  v_orthonormal  = " << r.v_orthonormal << "  (" << r.v_ortho_residual
         << " vs " << r.ortho_bound << ")"
         << "\n  energy_ok      = " << r.energy_ok << "  (defect "
         << r.energy_defect << " vs bound " << r.energy_bound << ")";
}

// --- acceptance -----------------------------------------------------------

TEST(VerifySvd, AcceptsExactFactorisationSquare) {
  const SvdCase c = make_svd_case(4, 4, {8.0, 4.0, 2.0, 1.0},
                                  MatrixOrder::ColMajor, 12345);
  EXPECT_TRUE(ReportAccepted(check(c)));
}

TEST(VerifySvd, AcceptsExactFactorisationTallAndWide) {
  const SvdCase tall = make_svd_case(6, 3, {5.0, 2.0, 0.25},
                                     MatrixOrder::ColMajor, 777);
  EXPECT_TRUE(ReportAccepted(check(tall)));
  EXPECT_EQ(tall.k, 3);

  const SvdCase wide = make_svd_case(3, 6, {5.0, 2.0, 0.25},
                                     MatrixOrder::ColMajor, 778);
  EXPECT_TRUE(ReportAccepted(check(wide)));
}

TEST(VerifySvd, AcceptsBothStorageOrders) {
  const std::vector<double> spectrum = {9.0, 3.0, 1.5, 0.5, 0.125};
  EXPECT_TRUE(ReportAccepted(
      check(make_svd_case(5, 5, spectrum, MatrixOrder::RowMajor, 4242))));
  EXPECT_TRUE(ReportAccepted(
      check(make_svd_case(5, 5, spectrum, MatrixOrder::ColMajor, 4242))));
}

// Degenerate and rank-deficient spectra are the design target, not the edge
// case, so they belong in the acceptance set from the start.
TEST(VerifySvd, AcceptsDegenerateAndRankDeficientSpectra) {
  EXPECT_TRUE(ReportAccepted(check(
      make_svd_case(5, 5, {3.0, 3.0, 3.0, 1.0, 1.0}, MatrixOrder::ColMajor, 91))));
  EXPECT_TRUE(ReportAccepted(check(
      make_svd_case(5, 4, {2.0, 1.0, 0.0, 0.0}, MatrixOrder::ColMajor, 92))));
  EXPECT_TRUE(ReportAccepted(check(
      make_svd_case(3, 3, {0.0, 0.0, 0.0}, MatrixOrder::ColMajor, 93))));
}

TEST(VerifySvd, AcceptsScaledSpectra) {
  EXPECT_TRUE(ReportAccepted(check(make_svd_case(
      4, 4, {1e-12, 1e-13, 1e-14, 1e-15}, MatrixOrder::ColMajor, 55))));
  EXPECT_TRUE(ReportAccepted(check(make_svd_case(
      4, 4, {1e12, 1e11, 1e10, 1e9}, MatrixOrder::ColMajor, 56))));
}

// A genuinely truncated factorisation: keep two of four directions. The
// residual is then sqrt(discarded), which the amplitude-form bound admits and
// a squared-form bound would not.
TEST(VerifySvd, AcceptsTruncatedFactorisation) {
  const SvdCase full = make_svd_case(6, 4, {8.0, 4.0, 2.0, 1.0},
                                     MatrixOrder::ColMajor, 31337);
  const int k = 2;
  const SvdReport r = check_svd(full.m.data(), full.rows, full.cols, full.order,
                                full.u.data(), full.s.data(), full.v.data(), k);
  EXPECT_TRUE(ReportAccepted(r));
  EXPECT_TRUE(r.truncated);
  // discarded = 2^2 + 1^2 = 5, so the residual sits right at sqrt(5).
  EXPECT_NEAR(r.discarded_energy, 5.0, 1e-9);
  EXPECT_NEAR(r.residual, 2.2360679774997896, 1e-9);
}

// --- rejection ------------------------------------------------------------

TEST(VerifySvd, RejectsNanInUColumn) {
  SvdCase c = make_svd_case(4, 4, {8.0, 4.0, 2.0, 1.0},
                            MatrixOrder::ColMajor, 12345);
  ASSERT_TRUE(check(c).ok());

  // Second column of U, third row.
  c.u[static_cast<std::size_t>(1) * 4 + 2] =
      Complex(opaque(quiet_nan()), 0.0);

  const SvdReport r = check(c);
  EXPECT_FALSE(r.finite);
  EXPECT_FALSE(r.ok());
}

TEST(VerifySvd, RejectsInfInVColumn) {
  SvdCase c = make_svd_case(4, 4, {8.0, 4.0, 2.0, 1.0},
                            MatrixOrder::ColMajor, 12345);
  ASSERT_TRUE(check(c).ok());

  c.v[0] = Complex(0.5, opaque(autonne_test::positive_inf()));

  const SvdReport r = check(c);
  EXPECT_FALSE(r.finite);
  EXPECT_FALSE(r.ok());
}

TEST(VerifySvd, RejectsPerturbedSingularValue) {
  SvdCase c = make_svd_case(4, 4, {8.0, 4.0, 2.0, 1.0},
                            MatrixOrder::ColMajor, 2024);
  ASSERT_TRUE(check(c).ok());

  c.s[1] *= 1.001;  // 4.0 -> 4.004

  const SvdReport r = check(c);
  EXPECT_TRUE(r.finite);
  EXPECT_FALSE(r.backward_ok);
  EXPECT_GT(r.residual, r.backward_bound);
  EXPECT_FALSE(r.energy_ok);
  EXPECT_FALSE(r.ok());
}

// A perturbation far smaller than the value it sits on, to show the bound is
// not merely catching gross damage.
TEST(VerifySvd, RejectsFinelyPerturbedSingularValue) {
  SvdCase c = make_svd_case(4, 4, {8.0, 4.0, 2.0, 1.0},
                            MatrixOrder::ColMajor, 2025);
  ASSERT_TRUE(check(c).ok());

  c.s[3] += 1e-9;

  const SvdReport r = check(c);
  EXPECT_FALSE(r.backward_ok);
  EXPECT_FALSE(r.ok());
}

TEST(VerifySvd, RejectsNonOrthogonalU) {
  SvdCase c = make_svd_case(4, 4, {8.0, 4.0, 2.0, 1.0},
                            MatrixOrder::ColMajor, 606);
  ASSERT_TRUE(check(c).ok());

  // Stretch the second column of U: its norm is no longer one, so U^* U - I
  // picks up a diagonal entry of 1.25.
  for (int i = 0; i < 4; ++i) {
    c.u[static_cast<std::size_t>(1) * 4 + static_cast<std::size_t>(i)] *= 1.5;
  }

  const SvdReport r = check(c);
  EXPECT_TRUE(r.finite);
  EXPECT_FALSE(r.u_orthonormal);
  EXPECT_GT(r.u_ortho_residual, r.ortho_bound);
  EXPECT_TRUE(r.v_orthonormal);
  EXPECT_FALSE(r.ok());
}

// Column norms left intact, but two columns of U made parallel. The diagonal
// of U^* U is still exactly one; only the off-diagonal gives it away.
TEST(VerifySvd, RejectsUWithDependentColumns) {
  SvdCase c = make_svd_case(4, 4, {8.0, 4.0, 2.0, 1.0},
                            MatrixOrder::ColMajor, 607);
  ASSERT_TRUE(check(c).ok());

  for (int i = 0; i < 4; ++i) {
    c.u[static_cast<std::size_t>(1) * 4 + static_cast<std::size_t>(i)] =
        c.u[static_cast<std::size_t>(0) * 4 + static_cast<std::size_t>(i)];
  }

  const SvdReport r = check(c);
  EXPECT_FALSE(r.u_orthonormal);
  EXPECT_FALSE(r.ok());
}

TEST(VerifySvd, RejectsNonOrthogonalV) {
  SvdCase c = make_svd_case(4, 4, {8.0, 4.0, 2.0, 1.0},
                            MatrixOrder::ColMajor, 608);
  ASSERT_TRUE(check(c).ok());

  for (int i = 0; i < 4; ++i) {
    c.v[static_cast<std::size_t>(2) * 4 + static_cast<std::size_t>(i)] *= 0.5;
  }

  const SvdReport r = check(c);
  EXPECT_FALSE(r.v_orthonormal);
  EXPECT_TRUE(r.u_orthonormal);
  EXPECT_FALSE(r.ok());
}

// Built ascending, with U and V columns permuted to match. Every other check
// passes -- the reconstruction is exact and both bases are orthonormal -- so
// this isolates the ordering test rather than tripping the backward error as
// a bare swap of two singular values would.
TEST(VerifySvd, RejectsSpectrumOutOfDescendingOrder) {
  const SvdCase c = make_svd_case(4, 4, {1.0, 2.0, 4.0, 8.0},
                                  MatrixOrder::ColMajor, 909);
  const SvdReport r = check(c);
  EXPECT_TRUE(r.finite);
  EXPECT_TRUE(r.nonnegative);
  EXPECT_TRUE(r.backward_ok);
  EXPECT_TRUE(r.u_orthonormal);
  EXPECT_TRUE(r.v_orthonormal);
  EXPECT_TRUE(r.energy_ok);
  EXPECT_FALSE(r.descending);
  EXPECT_FALSE(r.ok());
}

// Equal neighbours must not be read as an ordering violation.
TEST(VerifySvd, AcceptsTiedSingularValuesAsDescending) {
  const SvdCase c = make_svd_case(4, 4, {4.0, 4.0, 1.0, 1.0},
                                  MatrixOrder::ColMajor, 910);
  EXPECT_TRUE(check(c).descending);
}

TEST(VerifySvd, RejectsNegativeSingularValue) {
  SvdCase c = make_svd_case(4, 4, {8.0, 4.0, 2.0, 1.0},
                            MatrixOrder::ColMajor, 911);
  ASSERT_TRUE(check(c).ok());

  c.s[3] = -1.0;

  const SvdReport r = check(c);
  EXPECT_FALSE(r.nonnegative);
  EXPECT_FALSE(r.ok());
}

TEST(VerifySvd, RejectsNanSingularValue) {
  SvdCase c = make_svd_case(4, 4, {8.0, 4.0, 2.0, 1.0},
                            MatrixOrder::ColMajor, 912);
  ASSERT_TRUE(check(c).ok());

  c.s[2] = opaque(quiet_nan());

  const SvdReport r = check(c);
  EXPECT_FALSE(r.finite);
  EXPECT_FALSE(r.nonnegative);
  EXPECT_FALSE(r.descending);
  EXPECT_FALSE(r.ok());
}

// Why the bound is compared in amplitude form and not squared.
//
//   amplitude:  ||R||   <= sqrt(D) + b*N
//   squared:    ||R||^2 <= D + b^2*N^2
//
// with D the discarded energy, N = ||M||_F and b = 64*max(rows,cols)*eps.
// Squaring the first gives D + 2*b*N*sqrt(D) + b^2*N^2, so the squared form
// silently drops 2*b*sqrt(D*N^2) -- and when truncation is heavy that term is
// the whole budget. The squared form then permits a round-off excess of only
// b^2*N^2 / (2*sqrt(D)), around 1e-26 here, which is far below one ulp of the
// residual itself: it rejects any real truncated factorisation, correct or
// not. The error the amplitude form is meant to tolerate is a *false
// rejection*, not a false acceptance.
//
// Constructed here by rotating the second kept pair (u1, v1) slightly towards
// the first discarded pair (u2, v2). That keeps U and V exactly orthonormal
// and the spectrum untouched, while lifting the residual above sqrt(D) by a
// chosen amount -- here half of the permitted b*N, so the factorisation is
// comfortably inside the backward-error budget.
TEST(VerifySvd, SquaredBoundWouldRejectAValidTruncatedFactorisation) {
  SvdCase c = make_svd_case(6, 4, {8.0, 4.0, 2.0, 1.0},
                            MatrixOrder::ColMajor, 31338);
  const int k = 2;

  const double eps = std::numeric_limits<double>::epsilon();
  const double b = 64.0 * 6.0 * eps;      // backward factor * max(rows,cols)
  const double norm_M = std::sqrt(85.0);  // 8^2 + 4^2 + 2^2 + 1^2
  const double discarded = 5.0;           // 2^2 + 1^2

  // ||R||^2 = D + 2*t*s1*(s1 - s2) exactly, with t = sin^2(theta); solve for
  // the t that puts ||R|| at sqrt(D) + excess.
  const double excess = 0.5 * b * norm_M;
  const double t = excess * std::sqrt(discarded) / (4.0 * (4.0 - 2.0));
  const double theta = std::sqrt(t);
  const double cos_t = std::cos(theta);
  const double sin_t = std::sin(theta);

  for (int i = 0; i < c.rows; ++i) {
    const std::size_t kept = static_cast<std::size_t>(1 * c.rows + i);
    const std::size_t dropped = static_cast<std::size_t>(2 * c.rows + i);
    c.u[kept] = cos_t * c.u[kept] + sin_t * c.u[dropped];
  }
  for (int i = 0; i < c.cols; ++i) {
    const std::size_t kept = static_cast<std::size_t>(1 * c.cols + i);
    const std::size_t dropped = static_cast<std::size_t>(2 * c.cols + i);
    c.v[kept] = cos_t * c.v[kept] + sin_t * c.v[dropped];
  }

  const SvdReport r = check_svd(c.m.data(), c.rows, c.cols, c.order, c.u.data(),
                                c.s.data(), c.v.data(), k);

  // The fixture is what the constants above assume it is.
  ASSERT_NEAR(r.norm_M, norm_M, 1e-12);
  ASSERT_NEAR(r.discarded_energy, discarded, 1e-12);

  const double squared_bound =
      std::sqrt(r.discarded_energy + b * b * r.norm_M * r.norm_M);

  EXPECT_TRUE(ReportAccepted(r)) << "amplitude form must accept";
  EXPECT_LT(r.residual, r.backward_bound);
  EXPECT_GT(r.residual, squared_bound)
      << "squared form would have rejected a valid factorisation; it allows "
      << (squared_bound - std::sqrt(r.discarded_energy))
      << " above the truncation floor where the amplitude form allows "
      << (r.backward_bound - std::sqrt(r.discarded_energy));
}

// --- input validation -----------------------------------------------------

TEST(VerifySvd, RejectsMalformedArguments) {
  const SvdCase c = make_svd_case(4, 4, {8.0, 4.0, 2.0, 1.0},
                                  MatrixOrder::ColMajor, 1);

  EXPECT_FALSE(check_svd(nullptr, 4, 4, c.order, c.u.data(), c.s.data(),
                         c.v.data(), 4)
                   .inputs_valid);
  EXPECT_FALSE(check_svd(c.m.data(), 0, 4, c.order, c.u.data(), c.s.data(),
                         c.v.data(), 4)
                   .inputs_valid);
  EXPECT_FALSE(check_svd(c.m.data(), 4, 4, c.order, c.u.data(), c.s.data(),
                         c.v.data(), 0)
                   .inputs_valid);
  // k above min(rows, cols) is not a thin factorisation.
  EXPECT_FALSE(check_svd(c.m.data(), 4, 4, c.order, c.u.data(), c.s.data(),
                         c.v.data(), 5)
                   .inputs_valid);
  EXPECT_FALSE(check_svd(c.m.data(), 4, 4, c.order, c.u.data(), c.s.data(),
                         c.v.data(), 4)
                   .ok() == false);
}

}  // namespace
