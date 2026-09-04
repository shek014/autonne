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

// Verification harness: the arithmetic.
//
// This is a compiled translation unit rather than a header on purpose. A
// header-only check_svd has vague linkage: every translation unit that calls
// it emits its own copy, and the linker keeps one per binary chosen by mangled
// name, which does not record compile flags. A consumer that builds some files
// with -ffast-math and others without would then run whichever copy won the
// link, and a residual computed under a permissive floating-point model can
// come out too small -- the dangerous direction for a check whose job is to
// reject. Defining the entry points here gives them one definition each, built
// under the flags of this file, whatever the consumer does.
//
// The helpers below are in an anonymous namespace so that they too have
// exactly one home.

#include "autonne/verify.hpp"

#include <cmath>
#include <complex>

#include "autonne/detail/fp_bits.hpp"
#include "autonne/detail/matrix_view.hpp"

namespace autonne {
namespace verify {

namespace {

using autonne::detail::at;
using autonne::detail::cols_of;
using autonne::detail::fp_bad;
using autonne::detail::rows_of;

// A comparison is only meaningful once both sides are known finite; with a NaN
// operand the ordering predicates are themselves subject to the same
// -ffast-math assumptions. Every threshold test below routes through here.
bool within(const double& value, const double& bound) noexcept {
  if (fp_bad(value) || fp_bad(bound)) return false;
  return value <= bound;
}

template <typename View>
bool all_finite(const View& a) noexcept {
  for (int j = 0; j < cols_of(a); ++j) {
    for (int i = 0; i < rows_of(a); ++i) {
      if (fp_bad(at(a, i, j))) return false;
    }
  }
  return true;
}

bool all_finite(const double* v, int n) noexcept {
  for (int i = 0; i < n; ++i) {
    if (fp_bad(v[i])) return false;
  }
  return true;
}

// Sum of |a(i,j)|^2. Squared norms are accumulated rather than square-rooted
// per element so that the energy identities below compare like with like.
template <typename View>
double frobenius_sq(const View& a) noexcept {
  double acc = 0.0;
  for (int j = 0; j < cols_of(a); ++j) {
    for (int i = 0; i < rows_of(a); ++i) {
      const std::complex<double> z = at(a, i, j);
      acc += z.real() * z.real() + z.imag() * z.imag();
    }
  }
  return acc;
}

// ||X^* X - I||_F for an n x k matrix X.
template <typename View>
double orthonormality_residual(const View& x) noexcept {
  const int n = rows_of(x);
  const int k = cols_of(x);
  double acc = 0.0;
  for (int c = 0; c < k; ++c) {
    for (int d = 0; d < k; ++d) {
      std::complex<double> g(0.0, 0.0);
      for (int i = 0; i < n; ++i) g += std::conj(at(x, i, c)) * at(x, i, d);
      if (c == d) g -= std::complex<double>(1.0, 0.0);
      acc += g.real() * g.real() + g.imag() * g.imag();
    }
  }
  return std::sqrt(acc);
}

constexpr int min_int(int a, int b) noexcept { return a < b ? a : b; }
constexpr int max_int(int a, int b) noexcept { return a > b ? a : b; }

}  // namespace

// ---------------------------------------------------------------------------
// Thin / truncated SVD
// ---------------------------------------------------------------------------

SvdReport check_svd(const std::complex<double>* M, int rows, int cols,
                    MatrixOrder order, const std::complex<double>* U,
                    const double* S, const std::complex<double>* V, int k,
                    const Tolerances& tol) {
  SvdReport r;
  r.rows = rows;
  r.cols = cols;
  r.k = k;

  const int full_k = min_int(rows, cols);
  if (rows <= 0 || cols <= 0 || k <= 0 || k > full_k || M == nullptr ||
      U == nullptr || S == nullptr || V == nullptr) {
    return r;  // inputs_valid stays false; every verdict stays false
  }
  r.inputs_valid = true;
  r.truncated = k < full_k;

  const auto Mv = autonne::detail::ordered(M, rows, cols, order);
  const auto Uv = autonne::detail::col_major(U, rows, k);
  const auto Vv = autonne::detail::col_major(V, cols, k);

  // Non-finite scan over the kept slice, first, so that everything downstream
  // can be read as a measurement rather than a propagated NaN.
  r.finite = all_finite(Uv) && all_finite(S, k) && all_finite(Vv);

  const double eps = tol.eps;
  const double dim = static_cast<double>(max_int(rows, cols));

  const double energy_M = frobenius_sq(Mv);
  r.norm_M = std::sqrt(energy_M);

  double energy_S = 0.0;
  double s_max = 0.0;
  for (int t = 0; t < k; ++t) {
    energy_S += S[t] * S[t];
    if (!fp_bad(S[t]) && S[t] > s_max) s_max = S[t];
  }

  // The spectrum verdicts stand on their own: a NaN in S fails them here
  // rather than being inherited from a NaN elsewhere in U or V. Both operands
  // of the ordering step are guarded while they are still values in memory:
  // under -ffast-math their difference is a computed value the compiler may
  // assume finite, so a guard applied after the subtraction proves nothing.
  const double order_slack = tol.spectrum_factor * eps * s_max;
  bool nonneg = true;
  bool desc = true;
  for (int t = 0; t < k; ++t) {
    if (fp_bad(S[t]) || S[t] < 0.0) nonneg = false;
    if (t + 1 < k) {
      if (fp_bad(S[t]) || fp_bad(S[t + 1])) {
        desc = false;
      } else {
        const double step = S[t + 1] - S[t];
        if (!within(step, order_slack)) desc = false;
      }
    }
  }
  r.nonnegative = nonneg;
  r.descending = desc;

  r.energy_defect = energy_S - energy_M;
  r.energy_bound = tol.spectrum_factor * dim * eps * energy_M;
  const double energy_defect_abs = std::fabs(r.energy_defect);
  r.energy_ok = r.truncated ? within(r.energy_defect, r.energy_bound)
                            : within(energy_defect_abs, r.energy_bound);

  // Energy the kept spectrum does not account for. Clamped: rounding can push
  // the sum a few ulps past ||M||_F^2 on an untruncated factorisation.
  r.discarded_energy = (energy_M > energy_S) ? (energy_M - energy_S) : 0.0;

  // Backward error, in amplitude form.
  double residual_sq = 0.0;
  for (int j = 0; j < cols; ++j) {
    for (int i = 0; i < rows; ++i) {
      std::complex<double> approx(0.0, 0.0);
      for (int t = 0; t < k; ++t) {
        approx += at(Uv, i, t) * S[t] * std::conj(at(Vv, j, t));
      }
      const std::complex<double> d = at(Mv, i, j) - approx;
      residual_sq += d.real() * d.real() + d.imag() * d.imag();
    }
  }
  r.residual = std::sqrt(residual_sq);
  r.backward_bound =
      std::sqrt(r.discarded_energy) + tol.backward_factor * dim * eps * r.norm_M;
  r.backward_ok = r.finite && within(r.residual, r.backward_bound);

  // Orthonormality of the kept columns.
  r.u_ortho_residual = orthonormality_residual(Uv);
  r.v_ortho_residual = orthonormality_residual(Vv);
  r.ortho_bound = tol.ortho_factor * dim * eps;
  r.u_orthonormal = r.finite && within(r.u_ortho_residual, r.ortho_bound);
  r.v_orthonormal = r.finite && within(r.v_ortho_residual, r.ortho_bound);

  return r;
}

// ---------------------------------------------------------------------------
// Hermitian eigendecomposition
// ---------------------------------------------------------------------------

EighReport check_eigh(const std::complex<double>* A, int n, MatrixOrder order,
                      const double* evals, const std::complex<double>* evecs,
                      const Tolerances& tol) {
  EighReport r;
  r.n = n;
  if (n <= 0 || A == nullptr || evals == nullptr || evecs == nullptr) return r;
  r.inputs_valid = true;

  const auto Av = autonne::detail::ordered(A, n, n, order);
  const auto Qv = autonne::detail::col_major(evecs, n, n);

  r.finite = all_finite(Qv) && all_finite(evals, n);

  const double eps = tol.eps;
  const double dim = static_cast<double>(n);

  const double energy_A = frobenius_sq(Av);
  r.norm_A = std::sqrt(energy_A);

  double herm_sq = 0.0;
  double trace = 0.0;
  for (int j = 0; j < n; ++j) {
    trace += at(Av, j, j).real();
    for (int i = 0; i < n; ++i) {
      const std::complex<double> d = at(Av, i, j) - std::conj(at(Av, j, i));
      herm_sq += d.real() * d.real() + d.imag() * d.imag();
    }
  }
  r.hermitian_residual = std::sqrt(herm_sq);
  r.hermitian_bound = tol.backward_factor * dim * eps * r.norm_A;
  r.input_hermitian = within(r.hermitian_residual, r.hermitian_bound);

  double lambda_max = 0.0;
  double energy_L = 0.0;
  double sum_L = 0.0;
  for (int t = 0; t < n; ++t) {
    energy_L += evals[t] * evals[t];
    sum_L += evals[t];
    const double m = std::fabs(evals[t]);
    if (!fp_bad(m) && m > lambda_max) lambda_max = m;
  }

  // As in check_svd: guard the operands in memory, then subtract.
  const double order_slack = tol.spectrum_factor * eps * lambda_max;
  bool asc = true;
  for (int t = 0; t + 1 < n; ++t) {
    if (fp_bad(evals[t]) || fp_bad(evals[t + 1])) {
      asc = false;
    } else {
      const double step = evals[t] - evals[t + 1];
      if (!within(step, order_slack)) asc = false;
    }
  }
  r.ascending = asc;

  // A Q - Q diag(lambda), column by column.
  double residual_sq = 0.0;
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      std::complex<double> acc(0.0, 0.0);
      for (int t = 0; t < n; ++t) acc += at(Av, i, t) * at(Qv, t, j);
      const std::complex<double> d = acc - at(Qv, i, j) * evals[j];
      residual_sq += d.real() * d.real() + d.imag() * d.imag();
    }
  }
  r.residual = std::sqrt(residual_sq);
  r.backward_bound = tol.backward_factor * dim * eps * r.norm_A;
  r.backward_ok = r.finite && within(r.residual, r.backward_bound);

  r.q_ortho_residual = orthonormality_residual(Qv);
  r.ortho_bound = tol.ortho_factor * dim * eps;
  r.q_orthonormal = r.finite && within(r.q_ortho_residual, r.ortho_bound);

  r.trace_defect = sum_L - trace;
  r.trace_bound = tol.spectrum_factor * dim * eps * r.norm_A;
  const double trace_defect_abs = std::fabs(r.trace_defect);
  r.trace_ok = r.finite && within(trace_defect_abs, r.trace_bound);

  r.energy_defect = energy_L - energy_A;
  r.energy_bound = tol.spectrum_factor * dim * eps * energy_A;
  const double energy_defect_abs = std::fabs(r.energy_defect);
  r.energy_ok = r.finite && within(energy_defect_abs, r.energy_bound);

  return r;
}

}  // namespace verify
}  // namespace autonne
