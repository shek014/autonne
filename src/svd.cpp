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

// Thin SVD by column-pivoted QR followed by one-sided Jacobi.
//
// The pipeline, for an input A of rows x cols. Steps 1, 2 and 8 are the
// frame every kernel shares, in detail/svd_common.hpp; the rest is this
// file's core.
//
//   1. Reject non-finite input by bit pattern.
//   2. Drop rows and columns that are exactly zero. They are structural: their
//      singular values are exactly zero and their singular vectors are
//      canonical basis vectors, and no arithmetic should be allowed to say
//      otherwise.
//   3. Scale by a power of two so the largest component is in [0.5, 1), then
//      choose between A and A^*: whichever carries its scaling in the columns
//      (see prefers_transpose below).
//   4. Householder QR with column pivoting: A P = Q R, for any shape. Pivoting
//      puts the dominant directions first and drives any remaining rank
//      deficiency into trailing rows of R that are exactly zero once the
//      remaining columns fall below the floor.
//   5. A second, unpivoted QR of X = R^*: X = Q_2 R_2. Columns of X are rows
//      of R, so the trailing zero rows become zero columns of X and zero rows
//      of R_2, which yield exact zero singular values. This is the
//      preconditioning of Drmac and Veselic: the pivoted R is scaled by its
//      diagonal along its rows, and the second factorisation leaves a
//      triangular factor on which Jacobi needs a fraction of the sweeps.
//   6. One-sided Jacobi on the columns of R_2^*: pairs of columns are
//      rotated until every pair is orthogonal to working precision, the
//      rotations accumulate in V_2, and R_2^* V_2 = U_2 S with U_2 the
//      normalised columns.
//   7. Assemble: R_2^* = U_2 S V_2^* gives A P = (Q U_2) S (Q_2 V_2)^*, so
//      U = Q U_2 and V = P Q_2 V_2. Sort descending, undo the scaling, the
//      transposition and the compression.
//   8. Scan the result by bit pattern before writing anything to the caller.
//
// Everything is judged afterwards by verify::check_svd; this file does not
// compute a residual of its own.

#include "autonne/autonne.hpp"

#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>

#include "detail/kernel_common.hpp"
#include "detail/svd_common.hpp"

namespace autonne {

namespace {

using detail::kernel::Complex;
using detail::kernel::Index;
using detail::kernel::at;
using detail::svd_common::CoreResult;
using detail::svd_common::QrFactors;
using detail::svd_common::apply_q;
using detail::svd_common::jacobi_orthogonalise;
using detail::svd_common::qr_householder;

// Whether the second, unpivoted QR of R^* (Drmac and Veselic's second
// preconditioning step) runs before Jacobi. Measured on this code with
// Clang 22, 128 x 128, median wall time: off / on = 6.6 / 9.9 ms on a flat
// spectrum, 16.2 / 18.7 ms rank-deficient, 18.3 / 19.1 ms decaying over
// sixteen decades. The pivoted QR alone already leaves Jacobi with few
// sweeps, and the extra factorisation costs more than it saves, so it stays
// off. Every test passes either way; the switch exists so the measurement
// can be repeated when the rotation kernel changes.
constexpr bool kSecondQr = false;

// --- orientation -----------------------------------------------------------

// Householder QR applied from the left has a column-wise backward error: the
// perturbation in column j is bounded by eps times the norm of column j. That
// keeps a column scaling B D intact and lets the singular values that live in
// the small columns come out with relative accuracy. It does the opposite to
// a row scaling D B, where the small rows are swamped by the error from the
// large ones. The cure is to factor A^* instead, whose columns carry that
// scaling. Choosing between A and A^* by the spread of their row and column
// magnitudes therefore serves both one-sided scalings; when both sides are
// scaled, the more strongly scaled side wins.
//
// Spread is measured on the largest component per row or column rather than
// on norms, so that nothing here can underflow to zero for a live row.
bool prefers_transpose(const std::vector<Complex>& b, Index m, Index n) {
  std::vector<double> row_max(m, 0.0);
  std::vector<double> col_max(n, 0.0);
  for (Index j = 0; j < n; ++j) {
    for (Index i = 0; i < m; ++i) {
      const Complex& z = b[at(i, j, m)];
      const double a = std::fabs(z.real());
      const double c = std::fabs(z.imag());
      const double mag = a > c ? a : c;
      if (mag > row_max[i]) row_max[i] = mag;
      if (mag > col_max[j]) col_max[j] = mag;
    }
  }
  // The spread is returned as its base-2 logarithm, never as hi / lo. That
  // quotient overflows to infinity once the gap passes about 2^1024, which a
  // live row whose largest component is subnormal reaches beside an ordinary
  // one -- and the whole fast-math argument for this file rests on no
  // infinity ever being formed. Splitting each value into mantissa and
  // exponent keeps the difference exact in the exponent and bounded in the
  // mantissa: frexp puts both mantissas in [0.5, 1), so their ratio lies in
  // (0.5, 2) and its logarithm in (-1, 1).
  auto spread_log2 = [](const std::vector<double>& v, bool& has_zero) {
    double lo = v[0];
    double hi = v[0];
    for (const double x : v) {
      if (x < lo) lo = x;
      if (x > hi) hi = x;
    }
    has_zero = (lo == 0.0);
    if (has_zero) return 0.0;
    int e_hi = 0;
    int e_lo = 0;
    const double m_hi = std::frexp(hi, &e_hi);
    const double m_lo = std::frexp(lo, &e_lo);
    return static_cast<double>(e_hi - e_lo) + std::log2(m_hi / m_lo);
  };
  bool rows_have_zero = false;
  bool cols_have_zero = false;
  const double row_spread = spread_log2(row_max, rows_have_zero);
  const double col_spread = spread_log2(col_max, cols_have_zero);
  if (cols_have_zero) return false;
  if (rows_have_zero) return true;
  return row_spread > col_spread;
}

// --- the core, on a matrix with no zero rows or columns --------------------

// `b` is m x n column-major with m, n >= 1, every entry finite, at least one
// nonzero. Overwritten.
bool svd_core(std::vector<Complex> b, Index m, Index n, CoreResult& out) {
  const Index m0 = m;
  const Index n0 = n;

  // Exact power-of-two scaling, first, so the orientation test below sees
  // the same numbers whatever the input's absolute scale.
  const double max_comp = detail::kernel::max_component(b.data(), m * n);
  const int exponent = detail::kernel::scaling_exponent(max_comp);
  detail::kernel::scale_in_place(b.data(), m * n, -exponent);

  const bool transposed = prefers_transpose(b, m, n);
  if (transposed) {
    std::vector<Complex> bt(n * m);
    for (Index j = 0; j < n; ++j) {
      for (Index i = 0; i < m; ++i) bt[at(j, i, n)] = std::conj(b[at(i, j, m)]);
    }
    b.swap(bt);
    m = n0;
    n = m0;
  }
  const Index r = m < n ? m : n;

  // A P = Q R, with R of size r x n (upper trapezoidal) and Q of size m x r.
  QrFactors qr;
  qr_householder(b, m, n, true, qr);

  // X = R^*, n x r. Rows of R at or beyond the rank are exactly zero, so the
  // corresponding columns of X are exactly zero. R is upper trapezoidal, so
  // R(j, i) is nonzero only for j <= i, and X(i, j) = conj(R(j, i)).
  std::vector<Complex> x(n * r);
  for (Index j = 0; j < r; ++j) {
    for (Index i = 0; i < n; ++i) {
      x[at(i, j, n)] = (j <= i && j < qr.rank) ? std::conj(b[at(j, i, m)]) : Complex(0.0, 0.0);
    }
  }

  // Second preconditioning (Drmac and Veselic): X = Q_2 R_2 by an unpivoted
  // QR, and Jacobi runs on the columns of R_2^* instead of X. The pivoted
  // factor is scaled by its diagonal along its rows; this step turns the
  // result into a factor that is close to diagonal up to that scaling, so
  // that Jacobi needs a fraction of the sweeps it would on X. It has the
  // same column-wise backward error as the first QR, so nothing the first
  // step preserved is lost, and a zero column of X stays a zero column of
  // R_2 and hence a zero row of R_2, i.e. a zero column of R_2^*.
  QrFactors qr2;
  Index xr = n;  // rows of the matrix Jacobi works on
  if (kSecondQr) {
    qr_householder(x, n, r, false, qr2);
    std::vector<Complex> x2(r * r);
    for (Index j = 0; j < r; ++j) {
      for (Index i = 0; i < r; ++i) {
        x2[at(i, j, r)] = (j <= i) ? std::conj(x[at(j, i, n)]) : Complex(0.0, 0.0);
      }
    }
    x.swap(x2);
    xr = r;  // from here `x` is R_2^*, r x r
  }

  std::vector<Complex> vx(r * r, Complex(0.0, 0.0));
  for (Index j = 0; j < r; ++j) vx[at(j, j, r)] = Complex(1.0, 0.0);

  std::vector<bool> dead;
  if (!jacobi_orthogonalise(x, xr, vx, r, dead)) return false;

  // Singular values and left vectors of R_2^*; zero columns are completed to
  // an orthonormal set afterwards.
  std::vector<double> s(r, 0.0);
  Index live = 0;
  for (Index j = 0; j < r; ++j) {
    if (dead[j]) continue;
    s[j] = std::sqrt(detail::kernel::norm_sq(&x[at(0, j, xr)], xr));
    ++live;
  }

  // Descending order, stable, zeros last.
  std::vector<Index> order;
  detail::kernel::sort_indices(s.data(), r, true, order);

  std::vector<double> s_sorted(r);
  std::vector<Complex> vx_sorted(r * r);
  detail::kernel::permute_columns(vx.data(), vx_sorted.data(), r, order);
  std::vector<Complex> ux(xr * r, Complex(0.0, 0.0));
  for (Index j = 0; j < r; ++j) {
    const Index src = order[j];
    s_sorted[j] = s[src];
    if (dead[src]) continue;
    const double inv = 1.0 / s[src];
    for (Index i = 0; i < xr; ++i) ux[at(i, j, xr)] = x[at(i, src, xr)] * inv;
  }
  {
    std::vector<Complex> work;
    detail::kernel::complete_orthonormal(ux.data(), xr, live, r, work);
  }

  // Assembly. R_2^* = U_2 S V_2^* (U_2 = ux, V_2 = vx), so R_2 = V_2 S U_2^*,
  // X = R^* = Q_2 R_2 = Q_2 V_2 S U_2^*, R = U_2 S (Q_2 V_2)^*, and
  // A P = Q R = (Q U_2) S (Q_2 V_2)^*: U = Q U_2 and V = P Q_2 V_2.

  std::vector<Complex> u(m * r, Complex(0.0, 0.0));
  std::vector<Complex> v_pre(n * r, Complex(0.0, 0.0));
  if (kSecondQr) {
    // U = Q U_2: extend U_2 (r x r) to m x r with zero rows, apply the
    // pivoted-QR reflectors. V = P Q_2 V_2: extend V_2 (r x r) to n x r,
    // apply the second QR's reflectors, then undo the pivoting below.
    for (Index j = 0; j < r; ++j) {
      for (Index i = 0; i < r; ++i) u[at(i, j, m)] = ux[at(i, j, xr)];
    }
    apply_q(qr, u, r);
    for (Index j = 0; j < r; ++j) {
      for (Index i = 0; i < r; ++i) v_pre[at(i, j, n)] = vx_sorted[at(i, j, r)];
    }
    apply_q(qr2, v_pre, r);
  } else {
    // Without the second QR, Jacobi worked on X = R^* = U_X S V_X^*, so
    // A P = Q R = (Q V_X) S U_X^*: U = Q V_X and V = P U_X.
    for (Index j = 0; j < r; ++j) {
      for (Index i = 0; i < r; ++i) u[at(i, j, m)] = vx_sorted[at(i, j, r)];
    }
    apply_q(qr, u, r);
    for (Index j = 0; j < r; ++j) {
      for (Index i = 0; i < n; ++i) v_pre[at(i, j, n)] = ux[at(i, j, xr)];
    }
  }
  std::vector<Complex> v(n * r);
  for (Index j = 0; j < r; ++j) {
    for (Index i = 0; i < n; ++i) v[at(qr.perm[i], j, n)] = v_pre[at(i, j, n)];
  }

  for (Index j = 0; j < r; ++j) s_sorted[j] = std::ldexp(s_sorted[j], exponent);

  out.m = m0;
  out.n = n0;
  out.k = r;
  if (transposed) {
    // B^* = U S V^*  =>  B = V S U^*.
    out.u.swap(v);
    out.v.swap(u);
  } else {
    out.u.swap(u);
    out.v.swap(v);
  }
  out.s.swap(s_sorted);
  return true;
}

}  // namespace

bool svd_thin(const std::complex<double>* data, int rows, int cols,
              MatrixOrder order, std::complex<double>* U_out, double* S_out,
              std::complex<double>* V_out) {
  return detail::svd_common::svd_entry(data, rows, cols, order, U_out, S_out, V_out,
                                       svd_core);
}

}  // namespace autonne
