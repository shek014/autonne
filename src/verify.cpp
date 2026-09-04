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

// Sum of |scale * a(i,j)|^2. Squared norms are accumulated rather than
// square-rooted per element so that the energy identities below compare like
// with like, and every caller passes a scale for the reason set out at
// scale_exponent below.
template <typename View>
double frobenius_sq(const View& a, double scale) noexcept {
  double acc = 0.0;
  for (int j = 0; j < cols_of(a); ++j) {
    for (int i = 0; i < rows_of(a); ++i) {
      const std::complex<double> z = at(a, i, j);
      const double re = z.real() * scale;
      const double im = z.imag() * scale;
      acc += re * re + im * im;
    }
  }
  return acc;
}

// Largest |re| or |im| across a view, and across a real vector. Non-finite
// entries are skipped: the finiteness verdict reports them, and they must not
// decide the scaling.
template <typename View>
double max_component(const View& a) noexcept {
  double m = 0.0;
  for (int j = 0; j < cols_of(a); ++j) {
    for (int i = 0; i < rows_of(a); ++i) {
      const std::complex<double> z = at(a, i, j);
      if (fp_bad(z)) continue;
      const double re = std::fabs(z.real());
      const double im = std::fabs(z.imag());
      if (re > m) m = re;
      if (im > m) m = im;
    }
  }
  return m;
}

double max_component(const double* v, int n) noexcept {
  double m = 0.0;
  for (int i = 0; i < n; ++i) {
    if (fp_bad(v[i])) continue;
    const double a = std::fabs(v[i]);
    if (a > m) m = a;
  }
  return m;
}

// The exponent e for which ldexp(x, -e) puts the largest component of the
// measured objects in [0.5, 1).
//
// Every quantity below is measured on the input scaled by 2^-e rather than on
// the input itself, because the sum of squares of an unscaled matrix is not
// representable across the range of matrices a caller may hold. A matrix with
// entries near 1e210 has a Frobenius norm squared near 1e420, which overflows
// to infinity and rejects a perfectly good factorisation; one with entries
// near 1e-170 has squares that underflow to zero, so the residual, the bound
// and the energies all come out exactly zero and the check silently accepts
// whatever it was given. The second failure is the dangerous one.
//
// Scaling by a power of two is exact, and it commutes with everything here:
// the residual and the norms scale by 2^-e, the energies by 2^-2e, and the
// ratios that decide each verdict do not move at all. The reported fields are
// scaled back afterwards, which is the only step that can overflow, and only
// for a matrix whose true Frobenius norm is itself past the end of binary64.
int scale_exponent(double largest) noexcept {
  if (!(largest > 0.0) || fp_bad(largest)) return 0;
  int e = 0;
  (void)std::frexp(largest, &e);
  return e;
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

  // Everything from here is measured on the input scaled by `scale`, an exact
  // power of two; see scale_exponent. `s_max` stays unscaled because the
  // ordering slack it feeds is compared against unscaled differences.
  const double m_max = max_component(Mv);
  const double s_max = max_component(S, k);
  const int exponent = scale_exponent(m_max > s_max ? m_max : s_max);
  const double scale = std::ldexp(1.0, -exponent);

  const double energy_M = frobenius_sq(Mv, scale);
  r.norm_M = std::ldexp(std::sqrt(energy_M), exponent);

  double energy_S = 0.0;
  for (int t = 0; t < k; ++t) {
    const double scaled = S[t] * scale;
    energy_S += scaled * scaled;
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

  const double energy_defect = energy_S - energy_M;
  const double energy_bound = tol.spectrum_factor * dim * eps * energy_M;
  const double energy_defect_abs = std::fabs(energy_defect);
  r.energy_ok = r.truncated ? within(energy_defect, energy_bound)
                            : within(energy_defect_abs, energy_bound);
  r.energy_defect = std::ldexp(energy_defect, 2 * exponent);
  r.energy_bound = std::ldexp(energy_bound, 2 * exponent);

  // Energy the kept spectrum does not account for. Clamped: rounding can push
  // the sum a few ulps past ||M||_F^2 on an untruncated factorisation.
  const double discarded = (energy_M > energy_S) ? (energy_M - energy_S) : 0.0;
  r.discarded_energy = std::ldexp(discarded, 2 * exponent);

  // Backward error, in amplitude form. The reconstruction is formed from the
  // scaled spectrum, so it is compared against the scaled matrix.
  double residual_sq = 0.0;
  for (int j = 0; j < cols; ++j) {
    for (int i = 0; i < rows; ++i) {
      std::complex<double> approx(0.0, 0.0);
      for (int t = 0; t < k; ++t) {
        approx += at(Uv, i, t) * (S[t] * scale) * std::conj(at(Vv, j, t));
      }
      const std::complex<double> target = at(Mv, i, j) * scale;
      const std::complex<double> d = target - approx;
      residual_sq += d.real() * d.real() + d.imag() * d.imag();
    }
  }
  const double residual = std::sqrt(residual_sq);
  const double norm_M_scaled = std::sqrt(energy_M);
  const double backward_bound =
      std::sqrt(discarded) + tol.backward_factor * dim * eps * norm_M_scaled;
  r.backward_ok = r.finite && within(residual, backward_bound);
  r.residual = std::ldexp(residual, exponent);
  r.backward_bound = std::ldexp(backward_bound, exponent);

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

  // As in check_svd, every measurement is taken on the input scaled by an
  // exact power of two; see scale_exponent.
  const double a_max = max_component(Av);
  const double lambda_max = max_component(evals, n);
  const int exponent = scale_exponent(a_max > lambda_max ? a_max : lambda_max);
  const double scale = std::ldexp(1.0, -exponent);

  const double energy_A = frobenius_sq(Av, scale);
  const double norm_A_scaled = std::sqrt(energy_A);
  r.norm_A = std::ldexp(norm_A_scaled, exponent);

  double herm_sq = 0.0;
  double trace = 0.0;
  for (int j = 0; j < n; ++j) {
    trace += at(Av, j, j).real() * scale;
    for (int i = 0; i < n; ++i) {
      const std::complex<double> d =
          (at(Av, i, j) - std::conj(at(Av, j, i))) * scale;
      herm_sq += d.real() * d.real() + d.imag() * d.imag();
    }
  }
  const double hermitian_residual = std::sqrt(herm_sq);
  const double hermitian_bound = tol.backward_factor * dim * eps * norm_A_scaled;
  r.input_hermitian = within(hermitian_residual, hermitian_bound);
  r.hermitian_residual = std::ldexp(hermitian_residual, exponent);
  r.hermitian_bound = std::ldexp(hermitian_bound, exponent);

  double energy_L = 0.0;
  double sum_L = 0.0;
  for (int t = 0; t < n; ++t) {
    const double scaled = evals[t] * scale;
    energy_L += scaled * scaled;
    sum_L += scaled;
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

  // A Q - Q diag(lambda), column by column, on the scaled matrix.
  double residual_sq = 0.0;
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      std::complex<double> acc(0.0, 0.0);
      for (int t = 0; t < n; ++t) acc += (at(Av, i, t) * scale) * at(Qv, t, j);
      const std::complex<double> d = acc - at(Qv, i, j) * (evals[j] * scale);
      residual_sq += d.real() * d.real() + d.imag() * d.imag();
    }
  }
  const double residual = std::sqrt(residual_sq);
  const double backward_bound = tol.backward_factor * dim * eps * norm_A_scaled;
  r.backward_ok = r.finite && within(residual, backward_bound);
  r.residual = std::ldexp(residual, exponent);
  r.backward_bound = std::ldexp(backward_bound, exponent);

  r.q_ortho_residual = orthonormality_residual(Qv);
  r.ortho_bound = tol.ortho_factor * dim * eps;
  r.q_orthonormal = r.finite && within(r.q_ortho_residual, r.ortho_bound);

  const double trace_defect = sum_L - trace;
  const double trace_bound = tol.spectrum_factor * dim * eps * norm_A_scaled;
  const double trace_defect_abs = std::fabs(trace_defect);
  r.trace_ok = r.finite && within(trace_defect_abs, trace_bound);
  r.trace_defect = std::ldexp(trace_defect, exponent);
  r.trace_bound = std::ldexp(trace_bound, exponent);

  const double energy_defect = energy_L - energy_A;
  const double energy_bound = tol.spectrum_factor * dim * eps * energy_A;
  const double energy_defect_abs = std::fabs(energy_defect);
  r.energy_ok = r.finite && within(energy_defect_abs, energy_bound);
  r.energy_defect = std::ldexp(energy_defect, 2 * exponent);
  r.energy_bound = std::ldexp(energy_bound, 2 * exponent);

  return r;
}

}  // namespace verify
}  // namespace autonne
