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

// Same shape as test_verify_svd.cpp: accept what is correct by construction,
// reject each way of breaking it.

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include "autonne/verify.hpp"
#include "test_support.hpp"

namespace {

using autonne::MatrixOrder;
using autonne::verify::EighReport;
using autonne::verify::check_eigh;
using autonne_test::Complex;
using autonne_test::EighCase;
using autonne_test::make_eigh_case;
using autonne_test::bits_of;
using autonne_test::kQuietNanBits;
using autonne_test::poke_bits;

EighReport check(const EighCase& c) {
  return check_eigh(c.a.data(), c.n, c.order, c.evals.data(), c.q.data());
}

::testing::AssertionResult ReportAccepted(const EighReport& r) {
  if (r.ok()) return ::testing::AssertionSuccess();
  return ::testing::AssertionFailure()
         << "check_eigh rejected a correct decomposition:"
         << "\n  inputs_valid    = " << r.inputs_valid
         << "\n  input_hermitian = " << r.input_hermitian << "  ("
         << r.hermitian_residual << " vs " << r.hermitian_bound << ")"
         << "\n  finite          = " << r.finite
         << "\n  ascending       = " << r.ascending
         << "\n  backward_ok     = " << r.backward_ok << "  (" << r.residual
         << " vs " << r.backward_bound << ")"
         << "\n  q_orthonormal   = " << r.q_orthonormal << "  ("
         << r.q_ortho_residual << " vs " << r.ortho_bound << ")"
         << "\n  trace_ok        = " << r.trace_ok << "  (" << r.trace_defect
         << " vs " << r.trace_bound << ")"
         << "\n  energy_ok       = " << r.energy_ok << "  (" << r.energy_defect
         << " vs " << r.energy_bound << ")";
}

// --- acceptance -----------------------------------------------------------

TEST(VerifyEigh, AcceptsExactDecomposition) {
  EXPECT_TRUE(ReportAccepted(check(
      make_eigh_case(4, {-3.0, -0.5, 1.0, 4.0}, MatrixOrder::ColMajor, 5150))));
}

TEST(VerifyEigh, AcceptsBothStorageOrders) {
  const std::vector<double> evals = {-2.0, 0.0, 0.5, 3.0, 7.0};
  EXPECT_TRUE(ReportAccepted(
      check(make_eigh_case(5, evals, MatrixOrder::RowMajor, 61))));
  EXPECT_TRUE(ReportAccepted(
      check(make_eigh_case(5, evals, MatrixOrder::ColMajor, 61))));
}

TEST(VerifyEigh, AcceptsDegenerateAndSingularSpectra) {
  EXPECT_TRUE(ReportAccepted(check(
      make_eigh_case(5, {1.0, 1.0, 1.0, 4.0, 4.0}, MatrixOrder::ColMajor, 71))));
  EXPECT_TRUE(ReportAccepted(check(
      make_eigh_case(4, {0.0, 0.0, 0.0, 2.0}, MatrixOrder::ColMajor, 72))));
  EXPECT_TRUE(ReportAccepted(
      check(make_eigh_case(3, {0.0, 0.0, 0.0}, MatrixOrder::ColMajor, 73))));
}

TEST(VerifyEigh, AcceptsSingletonMatrix) {
  EXPECT_TRUE(ReportAccepted(
      check(make_eigh_case(1, {2.5}, MatrixOrder::ColMajor, 74))));
}

// --- rejection ------------------------------------------------------------

TEST(VerifyEigh, RejectsNanInEigenvector) {
  EighCase c = make_eigh_case(4, {-3.0, -0.5, 1.0, 4.0},
                              MatrixOrder::ColMajor, 5150);
  ASSERT_TRUE(check(c).ok());

  poke_bits(c.q[static_cast<std::size_t>(2) * 4 + 1], kQuietNanBits,
            bits_of(0.0));

  const EighReport r = check(c);
  EXPECT_FALSE(r.finite);
  EXPECT_FALSE(r.ok());
}

TEST(VerifyEigh, RejectsPerturbedEigenvalue) {
  EighCase c = make_eigh_case(4, {-3.0, -0.5, 1.0, 4.0},
                              MatrixOrder::ColMajor, 5151);
  ASSERT_TRUE(check(c).ok());

  c.evals[2] += 1e-9;

  const EighReport r = check(c);
  EXPECT_FALSE(r.backward_ok);
  EXPECT_FALSE(r.trace_ok);
  EXPECT_FALSE(r.ok());
}

TEST(VerifyEigh, RejectsNonOrthogonalEigenvectors) {
  EighCase c = make_eigh_case(4, {-3.0, -0.5, 1.0, 4.0},
                              MatrixOrder::ColMajor, 5152);
  ASSERT_TRUE(check(c).ok());

  for (int i = 0; i < 4; ++i) {
    c.q[static_cast<std::size_t>(1) * 4 + static_cast<std::size_t>(i)] *= 1.5;
  }

  const EighReport r = check(c);
  EXPECT_FALSE(r.q_orthonormal);
  EXPECT_FALSE(r.ok());
}

// Built descending with the eigenvectors permuted to match, so the
// decomposition itself is exact and only the ordering contract is broken.
TEST(VerifyEigh, RejectsEigenvaluesOutOfAscendingOrder) {
  const EighCase c = make_eigh_case(4, {4.0, 1.0, -0.5, -3.0},
                                    MatrixOrder::ColMajor, 5153);
  const EighReport r = check(c);
  EXPECT_TRUE(r.input_hermitian);
  EXPECT_TRUE(r.backward_ok);
  EXPECT_TRUE(r.q_orthonormal);
  EXPECT_TRUE(r.trace_ok);
  EXPECT_TRUE(r.energy_ok);
  EXPECT_FALSE(r.ascending);
  EXPECT_FALSE(r.ok());
}

TEST(VerifyEigh, AcceptsTiedEigenvaluesAsAscending) {
  const EighCase c = make_eigh_case(4, {1.0, 1.0, 4.0, 4.0},
                                    MatrixOrder::ColMajor, 5154);
  EXPECT_TRUE(check(c).ascending);
}

// A non-Hermitian input is the caller's error, and is reported as such rather
// than being blamed on the decomposition.
TEST(VerifyEigh, FlagsNonHermitianInput) {
  EighCase c = make_eigh_case(4, {-3.0, -0.5, 1.0, 4.0},
                              MatrixOrder::ColMajor, 5155);
  ASSERT_TRUE(check(c).ok());

  c.a[static_cast<std::size_t>(1) * 4 + 0] += Complex(0.25, 0.0);

  const EighReport r = check(c);
  EXPECT_FALSE(r.input_hermitian);
  EXPECT_FALSE(r.ok());
}

// The same pair as in test_verify_svd.cpp, for the reason given there: the
// harness must measure a matrix at any scale, and must not accept a wrong
// decomposition of a tiny one because every squared quantity underflowed.
TEST(VerifyEigh, AcceptsACorrectDecompositionOfAHugeMatrix) {
  for (const int exponent : {700, 900, 1000}) {
    const double factor = std::ldexp(1.0, exponent);
    EighCase c = make_eigh_case(4, {-3.0, -0.5, 1.0, 4.0}, MatrixOrder::ColMajor, 3101);
    for (Complex& z : c.a) z *= factor;
    for (double& x : c.evals) x *= factor;
    EXPECT_TRUE(ReportAccepted(check(c))) << "2^" << exponent;
  }
}

TEST(VerifyEigh, RejectsAWrongDecompositionOfATinyMatrix) {
  for (const int exponent : {-700, -900, -1000}) {
    const double factor = std::ldexp(1.0, exponent);
    EighCase c = make_eigh_case(4, {-3.0, -0.5, 1.0, 4.0}, MatrixOrder::ColMajor, 3102);
    for (Complex& z : c.a) z *= factor;
    for (double& x : c.evals) x *= factor;
    ASSERT_TRUE(check(c).ok()) << "2^" << exponent;

    EighCase wrong = c;
    wrong.evals[3] *= 2.0;
    const EighReport r = check(wrong);
    EXPECT_FALSE(r.backward_ok) << "2^" << exponent;
    EXPECT_FALSE(r.ok()) << "2^" << exponent;
  }
}

// As in test_verify_svd.cpp: the power-of-two scaling must be representable
// for a matrix below the normal range, and a wrong decomposition of one must
// still be rejected.
TEST(VerifyEigh, MeasuresMatricesBelowTheNormalRange) {
  for (const int exponent : {-1024, -1025, -1030, -1060}) {
    const double tiny = std::ldexp(1.0, exponent);
    const std::vector<Complex> a = {Complex(tiny, 0.0)};
    const std::vector<double> evals = {tiny};
    const std::vector<Complex> q = {Complex(1.0, 0.0)};
    const EighReport r =
        check_eigh(a.data(), 1, MatrixOrder::ColMajor, evals.data(), q.data());
    EXPECT_TRUE(ReportAccepted(r)) << "2^" << exponent;
  }
}

TEST(VerifyEigh, RejectsAWrongDecompositionBelowTheNormalRange) {
  for (const int exponent : {-1025, -1030, -1060}) {
    const double tiny = std::ldexp(1.0, exponent);
    const std::vector<Complex> a = {Complex(tiny, 0.0), Complex(0.0, 0.0),
                                    Complex(0.0, 0.0), Complex(tiny, 0.0)};
    const std::vector<Complex> q = {Complex(1.0, 0.0), Complex(0.0, 0.0),
                                    Complex(0.0, 0.0), Complex(1.0, 0.0)};
    const std::vector<double> evals = {0.25 * tiny, 3.0 * tiny};
    const EighReport r =
        check_eigh(a.data(), 2, MatrixOrder::ColMajor, evals.data(), q.data());
    EXPECT_FALSE(r.ok()) << "2^" << exponent;
    EXPECT_FALSE(r.backward_ok) << "2^" << exponent;
  }
}

TEST(VerifyEigh, RejectsMalformedArguments) {
  const EighCase c = make_eigh_case(3, {1.0, 2.0, 3.0}, MatrixOrder::ColMajor, 1);
  EXPECT_FALSE(
      check_eigh(nullptr, 3, c.order, c.evals.data(), c.q.data()).inputs_valid);
  EXPECT_FALSE(
      check_eigh(c.a.data(), 0, c.order, c.evals.data(), c.q.data()).inputs_valid);
  EXPECT_FALSE(
      check_eigh(c.a.data(), 3, c.order, nullptr, c.q.data()).inputs_valid);
  EXPECT_FALSE(
      check_eigh(c.a.data(), 3, c.order, c.evals.data(), nullptr).inputs_valid);
}

}  // namespace
