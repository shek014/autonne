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
// The pipeline, for an input A of rows x cols:
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
//   5. One-sided Jacobi on X = R^*. Columns of X are rows of R, so the
//      trailing zero rows become zero columns and yield exact zero singular
//      values; the nonzero rows, ordered by pivoting, are what Drmac and
//      Veselic showed Jacobi converges on quickly and with high relative
//      accuracy. Jacobi rotates pairs of columns until every pair is
//      orthogonal to working precision, accumulating the rotations in V_X;
//      then X V_X = U_X S with U_X the normalised columns.
//   6. Assemble: X = R^* = U_X S V_X^*, so R = V_X S U_X^* and
//      A P = Q V_X S U_X^*, giving U = Q V_X and V = P U_X. Sort descending,
//      undo the scaling, the transposition and the compression.
//   7. Scan the result by bit pattern before writing anything to the caller.
//
// Everything is judged afterwards by verify::check_svd; this file does not
// compute a residual of its own.

#include "autonne/autonne.hpp"

#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <vector>

#include "autonne/detail/fp_bits.hpp"
#include "autonne/detail/matrix_view.hpp"
#include "detail/kernel_common.hpp"

namespace autonne {

namespace {

using detail::kernel::Complex;
using detail::kernel::Index;
using detail::kernel::Rotation;
using detail::kernel::at;

constexpr int kMaxSweeps = 60;

// --- QR with column pivoting ------------------------------------------------

struct QrFactors {
  Index m = 0;
  Index n = 0;
  Index rank = 0;              // reflectors formed; rows of R beyond it are zero
  std::vector<Complex> h;      // m x n, column j holds the reflector vector v_j in rows j..m-1
  std::vector<double> tau;     // 2 / (v_j^* v_j)
  std::vector<Index> perm;     // A P has column perm[j] of A in position j
};

// In-place: on return `a` holds R in its upper triangle (rows >= rank exactly
// zero), and the factors hold the reflectors. Reflectors are Hermitian,
// P_j = I - tau_j v_j v_j^*, with P_j x = beta e_1 for beta = -e^{i arg x_1} ||x||
// (the sign that avoids cancellation in v_1). Q = P_0 P_1 ... P_{rank-1}.
//
// Column norms are recomputed from scratch at every step rather than
// downdated: it costs O(m n^2), the same order as the reflections, and it
// cannot suffer the cancellation that downdating formulas do.
void qr_column_pivoting(std::vector<Complex>& a, Index m, Index n, QrFactors& f) {
  f.m = m;
  f.n = n;
  f.rank = 0;
  f.h.assign(m * n, Complex(0.0, 0.0));
  f.tau.assign(n, 0.0);
  f.perm.resize(n);
  for (Index j = 0; j < n; ++j) f.perm[j] = j;

  const double floor_sq = detail::kernel::column_floor() * detail::kernel::column_floor();

  for (Index j = 0; j < n && j < m; ++j) {
    // Pivot: the trailing column of largest norm.
    Index pivot = j;
    double best = -1.0;
    for (Index c = j; c < n; ++c) {
      const double nrm = detail::kernel::norm_sq(&a[at(j, c, m)], m - j);
      if (nrm > best) {
        best = nrm;
        pivot = c;
      }
    }
    if (best < floor_sq) {
      // Nothing measurable is left. Declare the trailing block exactly zero
      // so that the rows of R from here on are exactly zero.
      for (Index c = j; c < n; ++c) {
        for (Index i = j; i < m; ++i) a[at(i, c, m)] = Complex(0.0, 0.0);
      }
      break;
    }
    if (pivot != j) {
      for (Index i = 0; i < m; ++i) {
        const Complex t = a[at(i, j, m)];
        a[at(i, j, m)] = a[at(i, pivot, m)];
        a[at(i, pivot, m)] = t;
      }
      const Index tp = f.perm[j];
      f.perm[j] = f.perm[pivot];
      f.perm[pivot] = tp;
    }

    // Reflector for column j, rows j..m-1.
    Complex* x = &a[at(j, j, m)];
    const Index len = m - j;
    const double xnorm = std::sqrt(best);  // best is exactly this column's norm^2
    const Complex alpha = x[0];
    const double alpha_abs = detail::kernel::modulus(alpha);
    const Complex phase = alpha_abs > 0.0
                              ? Complex(alpha.real() / alpha_abs, alpha.imag() / alpha_abs)
                              : Complex(1.0, 0.0);
    const Complex beta = -phase * xnorm;

    Complex* v = &f.h[at(j, j, m)];
    v[0] = alpha - beta;
    for (Index i = 1; i < len; ++i) v[i] = x[i];
    const double vnorm_sq = detail::kernel::norm_sq(v, len);
    // vnorm_sq >= xnorm^2 > 0 here: |v_1| = |alpha| + xnorm.
    const double tau = 2.0 / vnorm_sq;
    f.tau[j] = tau;

    // Apply P_j to the remaining columns.
    for (Index c = j + 1; c < n; ++c) {
      Complex* y = &a[at(j, c, m)];
      Complex w(0.0, 0.0);
      for (Index i = 0; i < len; ++i) w += std::conj(v[i]) * y[i];
      w *= tau;
      for (Index i = 0; i < len; ++i) y[i] -= w * v[i];
    }
    x[0] = beta;
    for (Index i = 1; i < len; ++i) x[i] = Complex(0.0, 0.0);
    f.rank = j + 1;
  }
}

// y (m x cols, column-major) <- Q y, with Q = P_0 P_1 ... P_{rank-1}.
void apply_q(const QrFactors& f, std::vector<Complex>& y, Index cols) {
  const Index m = f.m;
  for (Index jj = f.rank; jj > 0; --jj) {
    const Index j = jj - 1;
    const Complex* v = &f.h[at(j, j, m)];
    const Index len = m - j;
    const double tau = f.tau[j];
    for (Index c = 0; c < cols; ++c) {
      Complex* col = &y[at(j, c, m)];
      Complex w(0.0, 0.0);
      for (Index i = 0; i < len; ++i) w += std::conj(v[i]) * col[i];
      w *= tau;
      for (Index i = 0; i < len; ++i) col[i] -= w * v[i];
    }
  }
}

// --- one-sided Jacobi ------------------------------------------------------

// Orthogonalises the `cols` columns of x (rows x cols, column-major) by plane
// rotations, accumulating them in v (cols x cols, starts as identity). On
// return the nonzero columns of x are mutually orthogonal to working
// precision, `dead[j]` marks columns that are exactly zero, and
// x_original * v == x.
//
// A pair is rotated when |x_p^* x_q| > tol ||x_p|| ||x_q|| with
// tol = sqrt(rows) u, u the unit roundoff: below that the computed inner
// product is dominated by its own rounding and a rotation would be chasing
// noise. The resulting U = X S^{-1} then satisfies
// ||U^* U - I||_F <= cols sqrt(rows) u, comfortably inside the harness bound
// of 64 max(rows, cols) eps.
//
// Returns false if the sweep limit is reached without convergence.
bool jacobi_orthogonalise(std::vector<Complex>& x, Index rows, std::vector<Complex>& v,
                          Index cols, std::vector<bool>& dead) {
  const double tol = std::sqrt(static_cast<double>(rows)) * detail::kernel::unit_roundoff();
  const double floor_v = detail::kernel::column_floor();

  dead.assign(cols, false);
  auto refresh_dead = [&]() {
    for (Index j = 0; j < cols; ++j) {
      if (dead[j]) continue;
      if (std::sqrt(detail::kernel::norm_sq(&x[at(0, j, rows)], rows)) < floor_v) {
        dead[j] = true;
        for (Index i = 0; i < rows; ++i) x[at(i, j, rows)] = Complex(0.0, 0.0);
      }
    }
  };
  refresh_dead();

  for (int sweep = 0; sweep < kMaxSweeps; ++sweep) {
    Index rotations = 0;
    for (Index p = 0; p + 1 < cols; ++p) {
      if (dead[p]) continue;
      for (Index q = p + 1; q < cols; ++q) {
        if (dead[q]) continue;
        Complex* xp = &x[at(0, p, rows)];
        Complex* xq = &x[at(0, q, rows)];
        double alpha = 0.0;
        double beta = 0.0;
        Complex gamma(0.0, 0.0);
        for (Index i = 0; i < rows; ++i) {
          alpha += xp[i].real() * xp[i].real() + xp[i].imag() * xp[i].imag();
          beta += xq[i].real() * xq[i].real() + xq[i].imag() * xq[i].imag();
          gamma += std::conj(xp[i]) * xq[i];
        }
        const double gamma_abs = detail::kernel::modulus(gamma);
        if (!(gamma_abs > tol * std::sqrt(alpha) * std::sqrt(beta))) continue;

        const Rotation r = detail::kernel::make_rotation(alpha, beta, gamma);
        detail::kernel::rotate_columns(xp, xq, rows, r);
        detail::kernel::rotate_columns(&v[at(0, p, cols)], &v[at(0, q, cols)], cols, r);
        ++rotations;
      }
    }
    refresh_dead();
    if (rotations == 0) return true;
  }
  return false;
}

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
  auto spread = [](const std::vector<double>& v, bool& has_zero) {
    double lo = v[0];
    double hi = v[0];
    for (const double x : v) {
      if (x < lo) lo = x;
      if (x > hi) hi = x;
    }
    has_zero = (lo == 0.0);
    return has_zero ? 0.0 : hi / lo;
  };
  bool rows_have_zero = false;
  bool cols_have_zero = false;
  const double row_spread = spread(row_max, rows_have_zero);
  const double col_spread = spread(col_max, cols_have_zero);
  if (cols_have_zero) return false;
  if (rows_have_zero) return true;
  return row_spread > col_spread;
}

// --- the core, on a matrix with no zero rows or columns --------------------

struct CoreResult {
  Index m = 0;   // rows of the core input
  Index n = 0;   // cols of the core input
  Index k = 0;   // min(m, n)
  std::vector<Complex> u;  // m x k
  std::vector<double> s;   // k, descending
  std::vector<Complex> v;  // n x k
};

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
  qr_column_pivoting(b, m, n, qr);

  // X = R^*, n x r. Rows of R at or beyond the rank are exactly zero, so the
  // corresponding columns of X are exactly zero. R is upper trapezoidal, so
  // R(j, i) is nonzero only for j <= i, and X(i, j) = conj(R(j, i)).
  std::vector<Complex> x(n * r);
  for (Index j = 0; j < r; ++j) {
    for (Index i = 0; i < n; ++i) {
      x[at(i, j, n)] = (j <= i && j < qr.rank) ? std::conj(b[at(j, i, m)]) : Complex(0.0, 0.0);
    }
  }

  std::vector<Complex> vx(r * r, Complex(0.0, 0.0));
  for (Index j = 0; j < r; ++j) vx[at(j, j, r)] = Complex(1.0, 0.0);

  std::vector<bool> dead;
  if (!jacobi_orthogonalise(x, n, vx, r, dead)) return false;

  // Singular values and left vectors of X; zero columns are completed to an
  // orthonormal set afterwards.
  std::vector<double> s(r, 0.0);
  Index live = 0;
  for (Index j = 0; j < r; ++j) {
    if (dead[j]) continue;
    s[j] = std::sqrt(detail::kernel::norm_sq(&x[at(0, j, n)], n));
    ++live;
  }

  // Descending order, stable, zeros last.
  std::vector<Index> order;
  detail::kernel::sort_indices(s.data(), r, true, order);

  std::vector<double> s_sorted(r);
  std::vector<Complex> vx_sorted(r * r);
  detail::kernel::permute_columns(vx.data(), vx_sorted.data(), r, order);
  std::vector<Complex> ux(n * r, Complex(0.0, 0.0));
  for (Index j = 0; j < r; ++j) {
    const Index src = order[j];
    s_sorted[j] = s[src];
    if (dead[src]) continue;
    const double inv = 1.0 / s[src];
    for (Index i = 0; i < n; ++i) ux[at(i, j, n)] = x[at(i, src, n)] * inv;
  }
  {
    std::vector<Complex> work;
    detail::kernel::complete_orthonormal(ux.data(), n, live, r, work);
  }

  // U = Q V_X: extend V_X to m x r with zero rows, then apply the reflectors.
  std::vector<Complex> u(m * r, Complex(0.0, 0.0));
  for (Index j = 0; j < r; ++j) {
    for (Index i = 0; i < r; ++i) u[at(i, j, m)] = vx_sorted[at(i, j, r)];
  }
  apply_q(qr, u, r);

  // V = P U_X: row i of U_X becomes row perm[i] of V.
  std::vector<Complex> v(n * r);
  for (Index j = 0; j < r; ++j) {
    for (Index i = 0; i < n; ++i) v[at(qr.perm[i], j, n)] = ux[at(i, j, n)];
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

// --- driver ----------------------------------------------------------------

bool svd_impl(const Complex* data, Index rows, Index cols, MatrixOrder order,
              Complex* u_out, double* s_out, Complex* v_out) {
  const Index k = rows < cols ? rows : cols;
  if (detail::any_bad(data, static_cast<int>(rows * cols))) return false;

  const auto in = detail::ordered(data, static_cast<int>(rows), static_cast<int>(cols), order);

  // Structural zero rows and columns.
  std::vector<bool> row_live(rows, false);
  std::vector<bool> col_live(cols, false);
  for (int j = 0; j < static_cast<int>(cols); ++j) {
    for (int i = 0; i < static_cast<int>(rows); ++i) {
      const Complex z = detail::at(in, i, j);
      if (z.real() != 0.0 || z.imag() != 0.0) {
        row_live[static_cast<Index>(i)] = true;
        col_live[static_cast<Index>(j)] = true;
      }
    }
  }
  std::vector<Index> rows_kept;
  std::vector<Index> cols_kept;
  for (Index i = 0; i < rows; ++i) {
    if (row_live[i]) rows_kept.push_back(i);
  }
  for (Index j = 0; j < cols; ++j) {
    if (col_live[j]) cols_kept.push_back(j);
  }
  const Index mr = rows_kept.size();
  const Index nc = cols_kept.size();

  std::vector<Complex> u(rows * k, Complex(0.0, 0.0));
  std::vector<double> s(k, 0.0);
  std::vector<Complex> v(cols * k, Complex(0.0, 0.0));
  Index filled = 0;  // columns of U and V (and entries of S) already set

  if (mr > 0 && nc > 0) {
    std::vector<Complex> b(mr * nc);
    for (Index j = 0; j < nc; ++j) {
      for (Index i = 0; i < mr; ++i) {
        b[at(i, j, mr)] = detail::at(in, static_cast<int>(rows_kept[i]),
                                     static_cast<int>(cols_kept[j]));
      }
    }
    CoreResult core;
    if (!svd_core(std::move(b), mr, nc, core)) return false;

    // Embed: the core's factors live on the kept rows and columns.
    for (Index j = 0; j < core.k; ++j) {
      s[j] = core.s[j];
      for (Index i = 0; i < mr; ++i) u[at(rows_kept[i], j, rows)] = core.u[at(i, j, mr)];
      for (Index i = 0; i < nc; ++i) v[at(cols_kept[i], j, cols)] = core.v[at(i, j, nc)];
    }
    filled = core.k;
  }

  // The remaining k - filled singular values are exactly zero. Their vectors
  // come first from the dropped rows and columns, whose canonical basis
  // vectors are orthogonal to everything above by construction, and then
  // from a generic completion if the shape leaves any still to fill.
  {
    Index next_u = filled;
    for (Index i = 0; i < rows && next_u < k; ++i) {
      if (row_live[i]) continue;
      u[at(i, next_u, rows)] = Complex(1.0, 0.0);
      ++next_u;
    }
    Index next_v = filled;
    for (Index j = 0; j < cols && next_v < k; ++j) {
      if (col_live[j]) continue;
      v[at(j, next_v, cols)] = Complex(1.0, 0.0);
      ++next_v;
    }
    std::vector<Complex> work;
    detail::kernel::complete_orthonormal(u.data(), rows, next_u, k, work);
    detail::kernel::complete_orthonormal(v.data(), cols, next_v, k, work);
  }

  if (detail::any_bad(u.data(), static_cast<int>(rows * k)) ||
      detail::any_bad(s.data(), static_cast<int>(k)) ||
      detail::any_bad(v.data(), static_cast<int>(cols * k))) {
    return false;
  }
  for (Index i = 0; i < rows * k; ++i) u_out[i] = u[i];
  for (Index i = 0; i < k; ++i) s_out[i] = s[i];
  for (Index i = 0; i < cols * k; ++i) v_out[i] = v[i];
  return true;
}

}  // namespace

bool svd_thin(const std::complex<double>* data, int rows, int cols,
              MatrixOrder order, std::complex<double>* U_out, double* S_out,
              std::complex<double>* V_out) {
  if (data == nullptr || rows <= 0 || cols <= 0 || U_out == nullptr ||
      S_out == nullptr || V_out == nullptr) {
    return false;
  }
  // Element counts are handled as int throughout the public surface; a shape
  // whose element count does not fit is refused rather than truncated.
  if (static_cast<unsigned long long>(rows) * static_cast<unsigned long long>(cols) >
      static_cast<unsigned long long>(std::numeric_limits<int>::max())) {
    return false;
  }
  try {
    return svd_impl(data, static_cast<Index>(rows), static_cast<Index>(cols), order,
                    U_out, S_out, V_out);
  } catch (...) {
    // Allocation failure is the only thing that can throw in here, and the
    // contract is a false return rather than an exception.
    return false;
  }
}

}  // namespace autonne
