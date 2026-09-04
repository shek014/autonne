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

// Out-of-line definitions for the verification harness.
//
// These three live here rather than in verify.hpp so that each has exactly one
// strong definition per binary. Defined inline, they would have vague linkage:
// every translation unit including the header emits its own weak symbol, the
// linker keeps one per mangled name, and compile flags are not part of that
// name. A consumer compiling one translation unit -fno-fast-math and another
// -ffast-math would get whichever copy won the link, which is neither of the
// two configurations they asked for.
//
// Tagging the mangled name on __FAST_MATH__ was considered and rejected: the
// macro is too coarse. It says nothing about -fno-signed-zeros or
// -freciprocal-math, so two translation units differing only in those would
// still share a name and still merge. Putting the definition in a compiled
// translation unit makes the floating-point flag travel with the code that
// needs it, whatever the flag happens to be.
//
// The arithmetic below is unchanged from when it lived in the header; only its
// linkage is different. Everything still inline there -- fp_bad, `within`,
// `all_finite`, `frobenius_sq` -- is either flag-insensitive or small enough
// that the caller's own flags are the ones that should apply.
//
// tests/test_symbol_linkage.cmake asserts the resulting linkage.

#include "autonne/verify.hpp"

#include <cmath>
#include <complex>

#include "autonne/detail/matrix_view.hpp"

namespace autonne {
namespace verify {
namespace detail {

double orthonormality_residual(const ConstColMajor& x) noexcept {
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

}  // namespace detail

SvdReport check_svd(const std::complex<double>* M, int rows, int cols,
                    MatrixOrder order, const std::complex<double>* U,
                    const double* S, const std::complex<double>* V,
                    int k, const Tolerances& tol) {
  SvdReport r;
  r.rows = rows;
  r.cols = cols;
  r.k = k;

  const int full_k = detail::min_int(rows, cols);
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
  r.finite = detail::all_finite(Uv) && detail::all_finite(S, k) &&
             detail::all_finite(Vv);

  const double eps = tol.eps;
  const double dim = static_cast<double>(detail::max_int(rows, cols));

  const double energy_M = detail::frobenius_sq(Mv);
  r.norm_M = std::sqrt(energy_M);

  double energy_S = 0.0;
  double s_max = 0.0;
  for (int t = 0; t < k; ++t) {
    energy_S += S[t] * S[t];
    if (!fp_bad(S[t]) && S[t] > s_max) s_max = S[t];
  }

  // The spectrum verdicts stand on their own: a NaN in S fails them here (via
  // the explicit fp_bad, and via `within` for the ordering test) rather than
  // being inherited from a NaN elsewhere in U or V.
  const double order_slack = tol.spectrum_factor * eps * s_max;
  bool nonneg = true;
  bool desc = true;
  for (int t = 0; t < k; ++t) {
    if (fp_bad(S[t]) || S[t] < 0.0) nonneg = false;
    if (t + 1 < k && !detail::within(S[t + 1] - S[t], order_slack)) desc = false;
  }
  r.nonnegative = nonneg;
  r.descending = desc;

  r.energy_defect = energy_S - energy_M;
  r.energy_bound = tol.spectrum_factor * dim * eps * energy_M;
  r.energy_ok = r.truncated
                    ? detail::within(r.energy_defect, r.energy_bound)
                    : detail::within(std::fabs(r.energy_defect), r.energy_bound);

  // Energy the kept spectrum does not account for. Clamped: rounding can push
  // the sum a few ulps past ||M||_F^2 on an untruncated factorisation.
  r.discarded_energy = (energy_M > energy_S) ? (energy_M - energy_S) : 0.0;

  // Backward error, in amplitude form.
  double residual_sq = 0.0;
  for (int j = 0; j < cols; ++j) {
    for (int i = 0; i < rows; ++i) {
      std::complex<double> approx(0.0, 0.0);
      for (int t = 0; t < k; ++t) {
        approx += detail::at(Uv, i, t) * S[t] * std::conj(detail::at(Vv, j, t));
      }
      const std::complex<double> d = detail::at(Mv, i, j) - approx;
      residual_sq += d.real() * d.real() + d.imag() * d.imag();
    }
  }
  r.residual = std::sqrt(residual_sq);
  r.backward_bound =
      std::sqrt(r.discarded_energy) + tol.backward_factor * dim * eps * r.norm_M;
  r.backward_ok = r.finite && detail::within(r.residual, r.backward_bound);

  // Orthonormality of the kept columns.
  r.u_ortho_residual = detail::orthonormality_residual(Uv);
  r.v_ortho_residual = detail::orthonormality_residual(Vv);
  r.ortho_bound = tol.ortho_factor * dim * eps;
  r.u_orthonormal = r.finite && detail::within(r.u_ortho_residual, r.ortho_bound);
  r.v_orthonormal = r.finite && detail::within(r.v_ortho_residual, r.ortho_bound);

  return r;
}

EighReport check_eigh(const std::complex<double>* A, int n,
                      MatrixOrder order, const double* evals,
                      const std::complex<double>* evecs,
                      const Tolerances& tol) {
  EighReport r;
  r.n = n;
  if (n <= 0 || A == nullptr || evals == nullptr || evecs == nullptr) return r;
  r.inputs_valid = true;

  const auto Av = autonne::detail::ordered(A, n, n, order);
  const auto Qv = autonne::detail::col_major(evecs, n, n);

  r.finite = detail::all_finite(Qv) && detail::all_finite(evals, n);

  const double eps = tol.eps;
  const double dim = static_cast<double>(n);

  const double energy_A = detail::frobenius_sq(Av);
  r.norm_A = std::sqrt(energy_A);

  double herm_sq = 0.0;
  double trace = 0.0;
  for (int j = 0; j < n; ++j) {
    trace += detail::at(Av, j, j).real();
    for (int i = 0; i < n; ++i) {
      const std::complex<double> d =
          detail::at(Av, i, j) - std::conj(detail::at(Av, j, i));
      herm_sq += d.real() * d.real() + d.imag() * d.imag();
    }
  }
  r.hermitian_residual = std::sqrt(herm_sq);
  r.hermitian_bound = tol.backward_factor * dim * eps * r.norm_A;
  r.input_hermitian = detail::within(r.hermitian_residual, r.hermitian_bound);

  double lambda_max = 0.0;
  double energy_L = 0.0;
  double sum_L = 0.0;
  for (int t = 0; t < n; ++t) {
    energy_L += evals[t] * evals[t];
    sum_L += evals[t];
    const double m = std::fabs(evals[t]);
    if (!fp_bad(m) && m > lambda_max) lambda_max = m;
  }

  const double order_slack = tol.spectrum_factor * eps * lambda_max;
  bool asc = true;
  for (int t = 0; t + 1 < n; ++t) {
    if (!detail::within(evals[t] - evals[t + 1], order_slack)) asc = false;
  }
  r.ascending = asc;

  // A Q - Q diag(lambda), column by column.
  double residual_sq = 0.0;
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      std::complex<double> acc(0.0, 0.0);
      for (int t = 0; t < n; ++t) {
        acc += detail::at(Av, i, t) * detail::at(Qv, t, j);
      }
      const std::complex<double> d = acc - detail::at(Qv, i, j) * evals[j];
      residual_sq += d.real() * d.real() + d.imag() * d.imag();
    }
  }
  r.residual = std::sqrt(residual_sq);
  r.backward_bound = tol.backward_factor * dim * eps * r.norm_A;
  r.backward_ok = r.finite && detail::within(r.residual, r.backward_bound);

  r.q_ortho_residual = detail::orthonormality_residual(Qv);
  r.ortho_bound = tol.ortho_factor * dim * eps;
  r.q_orthonormal = r.finite && detail::within(r.q_ortho_residual, r.ortho_bound);

  r.trace_defect = sum_L - trace;
  r.trace_bound = tol.spectrum_factor * dim * eps * r.norm_A;
  r.trace_ok = r.finite && detail::within(std::fabs(r.trace_defect), r.trace_bound);

  r.energy_defect = energy_L - energy_A;
  r.energy_bound = tol.spectrum_factor * dim * eps * energy_A;
  r.energy_ok = r.finite && detail::within(std::fabs(r.energy_defect), r.energy_bound);

  return r;
}

}  // namespace verify
}  // namespace autonne
