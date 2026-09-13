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

// Thin SVD by Householder bidiagonalisation followed by divide and conquer.
//
// The pipeline, for an input A of rows x cols. Steps 1, 2 and 9 are the
// frame every kernel shares, in detail/svd_common.hpp; the rest is this
// file's core.
//
//   1. Reject non-finite input by bit pattern.
//   2. Drop rows and columns that are exactly zero. They are structural:
//      their singular values are exactly zero and their singular vectors are
//      canonical basis vectors, and no arithmetic should be allowed to say
//      otherwise.
//   3. Scale by a power of two so the largest component is in [0.5, 1), and
//      take A^* instead of A when there are more columns than rows, so that
//      the block is tall or square. Shape is all the orientation depends on:
//      this kernel promises absolute accuracy, and the choice that keeps a
//      one-sided scaling intact (svd.cpp) buys nothing here.
//   4. When the block is tall enough (m >= 1.6 n), an unpivoted Householder
//      QR first, A = Q R, so that the bidiagonalisation works on the n x n
//      triangle rather than on the whole block. The crossover is where the
//      flop counts of the two routes meet.
//   5. Householder bidiagonalisation, A = Q_bd B P_bd^*, with B complex
//      upper bidiagonal, Q_bd the product of n Hermitian reflectors applied
//      to the columns and P_bd the product of n - 1 applied to the rows.
//   6. Phase absorption, B = L B_r R^*, with L and R diagonal unitary and
//      B_r real with non-negative entries. A Hermitian reflector cannot make
//      the bidiagonal real by itself: I - tau v v^* maps x to beta e_1 only
//      when beta carries the phase of x_1. The phases are taken out here and
//      put back on the singular vectors in step 8.
//   7. Divide and conquer on B_r (detail/bidiag_dc.hpp): B_r = U_B S V_B^T
//      with U_B, V_B real orthogonal and S descending.
//   8. Assemble: A = (Q_bd L U_B) S (P_bd R V_B)^*, so U = Q_bd L U_B and
//      V = P_bd R V_B, with the reflectors applied to the factors directly
//      (neither Q_bd nor P_bd is ever formed); then Q from step 4 on U, the
//      transposition undone by swapping U and V, the scaling undone on S.
//   9. Scan the result by bit pattern before writing anything to the caller.
//
// What this kernel promises differs from svd.cpp in one respect: every
// singular value carries an error of order eps ||A||, so a value far below
// the largest is correct in absolute terms only. The bidiagonalisation is
// backward stable in the norm of the whole matrix, not column by column,
// and nothing downstream can recover what it mixed. svd_thin is the kernel
// for input whose small singular values must come out with relative
// accuracy.
//
// Everything is judged afterwards by verify::check_svd; this file does not
// compute a residual of its own.

#include "autonne/autonne.hpp"

#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>

#include "detail/bidiag_dc.hpp"
#include "detail/kernel_common.hpp"
#include "detail/svd_common.hpp"

namespace autonne {

namespace {

using detail::kernel::Complex;
using detail::kernel::Index;
using detail::kernel::at;
using detail::kernel::conj_mul;
using detail::kernel::mul;
using detail::kernel::scale;
using detail::svd_common::CoreResult;
using detail::svd_common::QrFactors;
using detail::svd_common::apply_q;
using detail::svd_common::qr_householder;

// Largest bidiagonal block solved densely (pivoted QR and one-sided Jacobi
// on its expansion) rather than split. Eigen's divide-and-conquer switches
// at sixteen; the benchmark sweeps this value.
constexpr Index kLeafSize = 16;

// Rows-to-columns ratio from which a QR factorisation precedes the
// bidiagonalisation (step 4). Bidiagonalising the block directly costs
// about 4 m n^2 - 4 n^3 / 3 flops; factoring first and bidiagonalising the
// triangle costs about 2 m n^2 + 2 n^3, and the two meet at m = 5 n / 3.
// LAPACK's dgesvd switches at 1.6.
constexpr double kQrCrossover = 1.6;

// --- Householder bidiagonalisation -----------------------------------------

// In place: `a` (m x n column-major, m >= n >= 1) becomes upper bidiagonal,
// with its diagonal in d[0..n) and its superdiagonal in e[0..n-1); e has n
// entries and the last is zero. The reflectors go to `left` and `right` in
// the shape apply_q reads, so that both back-transformations are that one
// routine:
//
//   * left.h column j holds v_j in rows j..m-1, H_j = I - tau_j v_j v_j^*
//     acts on the columns, and Q_bd = H_0 H_1 ... H_{n-1};
//   * right.h column j + 1 holds w_j in rows j+1..n-1, G_j = I - tau_j w_j w_j^*
//     acts on the rows, and P_bd = G_0 G_1 ... G_{n-2}. Column 0 of right.h
//     is empty with tau zero, which apply_q skips, so apply_q(right, y)
//     applies G_{n-2} first and G_0 last: that is P_bd y.
//
// The reflectors are qr_householder's exactly: Hermitian, beta = -e^{i arg
// x_1} ||x||, tau = 2 / ||v||^2. A right reflector is formed on the conjugate
// of the row, x = conj(a(j, j+1..n-1)): for a Hermitian G, row G = conj(G
// conj(row))^T, so G maps the row to conj(beta) e_1^T, and that conjugate is
// the superdiagonal entry.
//
// A column or row part whose norm is under the column floor gets no
// reflector. The part is declared exactly zero, a backward error of 2^-500
// relative to the scaled matrix and far under the eps ||A|| this kernel
// promises, and the bidiagonal entry is an exact zero, which the solver
// treats structurally: a zero superdiagonal decouples the matrix.
void bidiagonalise(std::vector<Complex>& a, Index m, Index n, QrFactors& left,
                   QrFactors& right, std::vector<Complex>& d, std::vector<Complex>& e) {
  left.m = m;
  left.n = n;
  left.rank = n;
  left.h.assign(m * n, Complex(0.0, 0.0));
  left.tau.assign(n, 0.0);
  left.perm.resize(n);
  right.m = n;
  right.n = n;
  right.rank = n;
  right.h.assign(n * n, Complex(0.0, 0.0));
  right.tau.assign(n, 0.0);
  right.perm.resize(n);
  for (Index j = 0; j < n; ++j) {
    left.perm[j] = j;
    right.perm[j] = j;
  }
  d.assign(n, Complex(0.0, 0.0));
  e.assign(n, Complex(0.0, 0.0));

  const double floor_sq = detail::kernel::column_floor() * detail::kernel::column_floor();
  std::vector<Complex> t(m);  // the trailing rows times w_j, one entry per row

  for (Index j = 0; j < n; ++j) {
    // Left reflector on column j, rows j..m-1.
    {
      Complex* x = &a[at(j, j, m)];
      const Index len = m - j;
      const double nrm_sq = detail::kernel::norm_sq(x, len);
      if (nrm_sq < floor_sq) {
        for (Index i = 0; i < len; ++i) x[i] = Complex(0.0, 0.0);
      } else {
        const double xnorm = std::sqrt(nrm_sq);
        const Complex alpha = x[0];
        const Complex phase = detail::kernel::unit_phase(alpha);
        const Complex beta(-phase.real() * xnorm, -phase.imag() * xnorm);
        Complex* v = &left.h[at(j, j, m)];
        v[0] = alpha - beta;
        for (Index i = 1; i < len; ++i) v[i] = x[i];
        // ||v||^2 >= ||x||^2 > 0: |v_1| = |alpha| + ||x||.
        const double tau = 2.0 / detail::kernel::norm_sq(v, len);
        left.tau[j] = tau;
        for (Index c = j + 1; c < n; ++c) {
          Complex* y = &a[at(j, c, m)];
          Complex dot(0.0, 0.0);
          for (Index i = 0; i < len; ++i) dot += conj_mul(v[i], y[i]);
          dot = scale(tau, dot);
          for (Index i = 0; i < len; ++i) y[i] -= mul(dot, v[i]);
        }
        x[0] = beta;
        for (Index i = 1; i < len; ++i) x[i] = Complex(0.0, 0.0);
        d[j] = beta;
      }
    }
    if (j + 1 >= n) break;

    // Right reflector on row j, columns j+1..n-1. The row is strided, so
    // its norm and the vector are gathered once; the trailing rows are then
    // updated column by column as a matrix-vector product followed by a
    // rank-one update, both on contiguous columns.
    {
      const Index len = n - j - 1;
      double nrm_sq = 0.0;
      for (Index c = 0; c < len; ++c) {
        const Complex z = a[at(j, j + 1 + c, m)];
        nrm_sq += z.real() * z.real() + z.imag() * z.imag();
      }
      if (nrm_sq < floor_sq) {
        for (Index c = 0; c < len; ++c) a[at(j, j + 1 + c, m)] = Complex(0.0, 0.0);
      } else {
        const double xnorm = std::sqrt(nrm_sq);
        const Complex alpha = std::conj(a[at(j, j + 1, m)]);
        const Complex phase = detail::kernel::unit_phase(alpha);
        const Complex beta(-phase.real() * xnorm, -phase.imag() * xnorm);
        Complex* w = &right.h[at(j + 1, j + 1, n)];
        w[0] = alpha - beta;
        for (Index c = 1; c < len; ++c) w[c] = std::conj(a[at(j, j + 1 + c, m)]);
        const double tau = 2.0 / detail::kernel::norm_sq(w, len);
        right.tau[j + 1] = tau;

        // Rows j+1..m-1: t = A(:, j+1..n-1) w, then A(:, c) -= tau t conj(w_c).
        const Index below = m - j - 1;
        for (Index i = 0; i < below; ++i) t[i] = Complex(0.0, 0.0);
        for (Index c = 0; c < len; ++c) {
          const Complex* col = &a[at(j + 1, j + 1 + c, m)];
          const Complex wc = w[c];
          for (Index i = 0; i < below; ++i) t[i] += mul(col[i], wc);
        }
        for (Index i = 0; i < below; ++i) t[i] = scale(tau, t[i]);
        for (Index c = 0; c < len; ++c) {
          Complex* col = &a[at(j + 1, j + 1 + c, m)];
          const Complex wc = std::conj(w[c]);
          for (Index i = 0; i < below; ++i) col[i] -= mul(t[i], wc);
        }
        a[at(j, j + 1, m)] = std::conj(beta);
        for (Index c = 1; c < len; ++c) a[at(j, j + 1 + c, m)] = Complex(0.0, 0.0);
        e[j] = std::conj(beta);
      }
    }
  }
}

// --- the core, on a matrix with no zero rows or columns --------------------

// `b` is m x n column-major with m, n >= 1, every entry finite, at least one
// nonzero. Overwritten.
bool svd_bdc_core(std::vector<Complex> b, Index m, Index n, CoreResult& out) {
  const Index m0 = m;
  const Index n0 = n;

  // Exact power-of-two scaling.
  const double max_comp = detail::kernel::max_component(b.data(), m * n);
  const int exponent = detail::kernel::scaling_exponent(max_comp);
  detail::kernel::scale_in_place(b.data(), m * n, -exponent);

  // Tall or square from here on.
  const bool transposed = m < n;
  if (transposed) {
    std::vector<Complex> bt(n * m);
    for (Index j = 0; j < n; ++j) {
      for (Index i = 0; i < m; ++i) bt[at(j, i, n)] = std::conj(b[at(i, j, m)]);
    }
    b.swap(bt);
    m = n0;
    n = m0;
  }

  // Step 4: for a tall block, A = Q R and the bidiagonalisation runs on the
  // n x n triangle R. Unpivoted, since rank deficiency need not be driven
  // into exact zeros here: it becomes small diagonal entries, which is what
  // absolute accuracy means. A column under the floor forms no reflector
  // and is left exactly zero, so R is still upper triangular; every entry
  // on or above the diagonal is copied whether or not a reflector was
  // formed for its column.
  QrFactors qr;
  const bool qr_first = static_cast<double>(m) >= kQrCrossover * static_cast<double>(n);
  Index mb = m;  // rows of the block that is bidiagonalised
  if (qr_first) {
    qr_householder(b, m, n, false, qr);
    std::vector<Complex> r(n * n, Complex(0.0, 0.0));
    for (Index j = 0; j < n; ++j) {
      for (Index i = 0; i <= j; ++i) r[at(i, j, n)] = b[at(i, j, m)];
    }
    b.swap(r);
    mb = n;
  }

  // Step 5.
  QrFactors left;
  QrFactors right;
  std::vector<Complex> d;
  std::vector<Complex> e;
  bidiagonalise(b, mb, n, left, right, d, e);

  // Step 6: B = L B_r R^*. With r_0 = 1, l_j takes the phase of d_j r_j so
  // that conj(l_j) d_j r_j = |d_j|, and r_{j+1} the phase that makes
  // conj(l_j) e_j r_{j+1} = |e_j|. unit_phase of zero is one, so an exact
  // zero in d or e stays an exact zero in B_r.
  std::vector<Complex> l(n);
  std::vector<Complex> r(n);
  std::vector<double> d_real(n, 0.0);
  std::vector<double> e_real(n, 0.0);
  r[0] = Complex(1.0, 0.0);
  for (Index j = 0; j < n; ++j) {
    l[j] = detail::kernel::unit_phase(mul(d[j], r[j]));
    d_real[j] = detail::kernel::modulus(d[j]);
    if (j + 1 < n) {
      r[j + 1] = detail::kernel::unit_phase(mul(l[j], std::conj(e[j])));
      e_real[j] = detail::kernel::modulus(e[j]);
    }
  }

  // The solver wants its input at most one in magnitude; the bidiagonal's
  // entries are norms of parts of the scaled block, so they can reach
  // sqrt(2 m n). A second exact scaling, undone together with the first.
  double top = 0.0;
  for (Index j = 0; j < n; ++j) {
    if (d_real[j] > top) top = d_real[j];
    if (e_real[j] > top) top = e_real[j];
  }
  const int exponent_b = detail::kernel::scaling_exponent(top);
  for (Index j = 0; j < n; ++j) {
    d_real[j] = std::ldexp(d_real[j], -exponent_b);
    e_real[j] = std::ldexp(e_real[j], -exponent_b);
  }

  // Step 7.
  std::vector<double> ub;
  std::vector<double> s;
  std::vector<double> vb;
  if (!detail::bidiag_dc::bidiag_svd(d_real.data(), e_real.data(), n, kLeafSize, ub, s, vb)) {
    return false;
  }

  // Step 8. U = Q_bd L U_B: L U_B is n x n on top of mb - n zero rows, then
  // the left reflectors, then the QR's if there was one. V = P_bd R V_B.
  std::vector<Complex> u_bd(mb * n, Complex(0.0, 0.0));
  for (Index j = 0; j < n; ++j) {
    for (Index i = 0; i < n; ++i) u_bd[at(i, j, mb)] = scale(ub[at(i, j, n)], l[i]);
  }
  apply_q(left, u_bd, n);
  std::vector<Complex> u;
  if (qr_first) {
    u.assign(m * n, Complex(0.0, 0.0));
    for (Index j = 0; j < n; ++j) {
      for (Index i = 0; i < n; ++i) u[at(i, j, m)] = u_bd[at(i, j, mb)];
    }
    apply_q(qr, u, n);
  } else {
    u.swap(u_bd);
  }

  std::vector<Complex> v(n * n);
  for (Index j = 0; j < n; ++j) {
    for (Index i = 0; i < n; ++i) v[at(i, j, n)] = scale(vb[at(i, j, n)], r[i]);
  }
  apply_q(right, v, n);

  for (Index j = 0; j < n; ++j) s[j] = std::ldexp(s[j], exponent + exponent_b);

  out.m = m0;
  out.n = n0;
  out.k = n;
  if (transposed) {
    // B^* = U S V^*  =>  B = V S U^*.
    out.u.swap(v);
    out.v.swap(u);
  } else {
    out.u.swap(u);
    out.v.swap(v);
  }
  out.s.swap(s);
  return true;
}

}  // namespace

bool svd_thin_bdc(const std::complex<double>* data, int rows, int cols,
                  MatrixOrder order, std::complex<double>* U_out, double* S_out,
                  std::complex<double>* V_out) {
  return detail::svd_common::svd_entry(data, rows, cols, order, U_out, S_out, V_out,
                                       svd_bdc_core);
}

}  // namespace autonne
