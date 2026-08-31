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

// Verification harness.
//
// autonne is judged by this file rather than trusted in place of it: a caller
// checks every factorisation against the matrix it came from, and a check that
// cannot reject a bad factorisation is worth nothing. Header-only, so each
// translation unit compiles it under its own floating-point flags.

#ifndef AUTONNE_VERIFY_HPP
#define AUTONNE_VERIFY_HPP

#include <bit>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>

#include "autonne/autonne.hpp"
#include "autonne/detail/matrix_view.hpp"

namespace autonne {
namespace verify {

// ---------------------------------------------------------------------------
// Non-finite detection
// ---------------------------------------------------------------------------

// True if x is NaN or an infinity.
//
// Deliberately does not call std::isnan or std::isfinite. Under -ffast-math
// (specifically -ffinite-math-only) the compiler is entitled to assume no NaN
// or infinity is ever produced and folds those predicates to a constant false,
// which deletes the guard entirely. This reads the IEEE-754 binary64 exponent
// field through std::bit_cast: an all-ones exponent means NaN or infinity. It
// is integer work on the object representation, so no assumption about the
// range of floating-point values can remove it.
constexpr bool fp_bad(double x) noexcept {
  static_assert(sizeof(double) == sizeof(std::uint64_t),
                "fp_bad assumes IEEE-754 binary64");
  const std::uint64_t bits = std::bit_cast<std::uint64_t>(x);
  return ((bits >> 52) & UINT64_C(0x7FF)) == UINT64_C(0x7FF);
}

inline bool fp_bad(const std::complex<double>& z) noexcept {
  return fp_bad(z.real()) || fp_bad(z.imag());
}

namespace detail {

using autonne::detail::at;
using autonne::detail::cols_of;
using autonne::detail::rows_of;

// A comparison is only meaningful once both sides are known finite; with a NaN
// operand the ordering predicates are themselves subject to the same
// -ffast-math assumptions. Every threshold test below routes through here.
constexpr bool within(double value, double bound) noexcept {
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

inline bool all_finite(const double* v, int n) noexcept {
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

}  // namespace detail

// ---------------------------------------------------------------------------
// Tolerances
// ---------------------------------------------------------------------------

// Every bound below is `factor * max(dimension) * eps * scale`. The factors are
// generous by design: the harness must reject a wrong factorisation, not grade
// a right one to the last ulp.
struct Tolerances {
  double eps = std::numeric_limits<double>::epsilon();
  double backward_factor = 64.0;
  double ortho_factor = 64.0;
  double spectrum_factor = 64.0;
};

// ---------------------------------------------------------------------------
// Thin / truncated SVD
// ---------------------------------------------------------------------------

struct SvdReport {
  // Shape actually checked.
  int rows = 0;
  int cols = 0;
  int k = 0;
  bool truncated = false;  // k < min(rows, cols)

  // Measured quantities.
  double norm_M = 0.0;            // ||M||_F
  double residual = 0.0;          // ||M - U_k S_k V_k^*||_F
  double discarded_energy = 0.0;  // max(0, ||M||_F^2 - sum s_i^2)
  double backward_bound = 0.0;
  double u_ortho_residual = 0.0;  // ||U^* U - I||_F
  double v_ortho_residual = 0.0;  // ||V^* V - I||_F
  double ortho_bound = 0.0;
  double energy_defect = 0.0;  // sum s_i^2 - ||M||_F^2
  double energy_bound = 0.0;

  // Verdicts.
  bool inputs_valid = false;
  bool finite = false;       // no NaN/Inf across the kept slice of U, S, V
  bool nonnegative = false;  // s_i >= 0
  bool descending = false;   // s_i >= s_{i+1}
  bool backward_ok = false;
  bool u_orthonormal = false;
  bool v_orthonormal = false;
  bool energy_ok = false;

  constexpr bool ok() const noexcept {
    return inputs_valid && finite && nonnegative && descending && backward_ok &&
           u_orthonormal && v_orthonormal && energy_ok;
  }
};

// Checks U_k, S_k, V_k against the matrix M they claim to factor.
//
//   M      rows x cols in `order`
//   U      rows x k, column-major
//   S      k values
//   V      cols x k, column-major (V itself, not V^*)
//
// The backward error is compared in amplitude form,
//
//   ||M - U_k S_k V_k^*||_F <= sqrt(discarded) + bwd * ||M||_F
//
// with bwd = backward_factor * max(rows, cols) * eps and `discarded` the
// energy in the singular values not kept. Squaring both sides is wrong: it
// drops the cross term 2 * bwd * sqrt(discarded * ||M||_F^2), which is the
// dominant term whenever the truncation is heavy.
//
// The spectral energy identity sum s_i^2 == ||M||_F^2 holds exactly when
// nothing was truncated; when k < min(rows, cols) the requirement weakens to
// the one-sided sum s_i^2 <= ||M||_F^2, since the missing energy is precisely
// what `discarded` accounts for.
inline SvdReport check_svd(const std::complex<double>* M, int rows, int cols,
                           MatrixOrder order, const std::complex<double>* U,
                           const double* S, const std::complex<double>* V,
                           int k, const Tolerances& tol = Tolerances()) {
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

// ---------------------------------------------------------------------------
// Hermitian eigendecomposition
// ---------------------------------------------------------------------------

struct EighReport {
  int n = 0;

  double norm_A = 0.0;              // ||A||_F
  double hermitian_residual = 0.0;  // ||A - A^*||_F
  double hermitian_bound = 0.0;
  double residual = 0.0;  // ||A Q - Q diag(lambda)||_F
  double backward_bound = 0.0;
  double q_ortho_residual = 0.0;  // ||Q^* Q - I||_F
  double ortho_bound = 0.0;
  double trace_defect = 0.0;  // sum lambda_i - Re tr(A)
  double trace_bound = 0.0;
  double energy_defect = 0.0;  // sum lambda_i^2 - ||A||_F^2
  double energy_bound = 0.0;

  bool inputs_valid = false;
  bool input_hermitian = false;
  bool finite = false;
  bool ascending = false;
  bool backward_ok = false;
  bool q_orthonormal = false;
  bool trace_ok = false;
  bool energy_ok = false;

  constexpr bool ok() const noexcept {
    return inputs_valid && input_hermitian && finite && ascending &&
           backward_ok && q_orthonormal && trace_ok && energy_ok;
  }
};

// Checks (evals, evecs) against the Hermitian matrix A they claim to
// diagonalise. Eigenvalues are required in ascending order, matching the
// contract in autonne.hpp. `input_hermitian` reports on the caller's input
// rather than on autonne: a non-Hermitian A makes every other verdict
// meaningless, so it is surfaced separately.
inline EighReport check_eigh(const std::complex<double>* A, int n,
                             MatrixOrder order, const double* evals,
                             const std::complex<double>* evecs,
                             const Tolerances& tol = Tolerances()) {
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

#endif  // AUTONNE_VERIFY_HPP
