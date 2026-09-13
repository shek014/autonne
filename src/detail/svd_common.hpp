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

// The frame that both thin-SVD kernels share. Private to src/: nothing here
// is declared in a public header, and every function is `static` for the
// reason kernel_common.hpp gives: each kernel is compiled under the flags of
// its own translation unit, and a helper with vague linkage would be merged
// across translation units without regard to those flags.
//
// What is shared is the frame around a core, not the core. svd_entry and
// svd_driver check the input, strip its structural zeros, hand the rest to
// the core, and embed and scan what comes back; scaling and orientation are
// the core's, since each kernel has its own reasons for both. qr_householder
// and apply_q are the Householder QR, pivoted or not, that both kernels
// precondition or reduce with. jacobi_orthogonalise is the one-sided Jacobi
// sweep, which the Jacobi kernel runs on its whole preconditioned block and
// the divide-and-conquer kernel runs on its leaves.

#ifndef AUTONNE_SRC_DETAIL_SVD_COMMON_HPP
#define AUTONNE_SRC_DETAIL_SVD_COMMON_HPP

#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

#include "autonne/autonne.hpp"
#include "autonne/detail/fp_bits.hpp"
#include "autonne/detail/matrix_view.hpp"
#include "kernel_common.hpp"

namespace autonne {
namespace detail {
namespace svd_common {

using kernel::Complex;
using kernel::Index;
using kernel::Rotation;
using kernel::at;
using kernel::conj_mul;
using kernel::mul;

// Sweep budget. Cyclic Jacobi converges quadratically once the off-diagonal
// mass is small, so the count is low and grows slowly with n: lowering this
// until the suite breaks puts the hardest case in the corpus and the stress
// sweeps at twelve sweeps (a 128 x 128 random matrix, and a 96 x 96 with a
// three-level degenerate spectrum). Sixty is therefore a five-fold margin,
// not a guess. Exhausting it is reported as failure rather than papered over,
// which is what the false return in the contract is for.
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
static inline void qr_householder(std::vector<Complex>& a, Index m, Index n, bool pivot, QrFactors& f) {
  f.m = m;
  f.n = n;
  f.rank = 0;
  f.h.assign(m * n, Complex(0.0, 0.0));
  f.tau.assign(n, 0.0);
  f.perm.resize(n);
  for (Index j = 0; j < n; ++j) f.perm[j] = j;

  const double floor_sq = detail::kernel::column_floor() * detail::kernel::column_floor();

  for (Index j = 0; j < n && j < m; ++j) {
    // Pivot: the trailing column of largest norm. Without pivoting the
    // column norm is still needed for the reflector.
    Index pivot_col = j;
    double best = -1.0;
    if (pivot) {
      for (Index c = j; c < n; ++c) {
        const double nrm = detail::kernel::norm_sq(&a[at(j, c, m)], m - j);
        if (nrm > best) {
          best = nrm;
          pivot_col = c;
        }
      }
    } else {
      best = detail::kernel::norm_sq(&a[at(j, j, m)], m - j);
    }
    if (best < floor_sq) {
      if (pivot) {
        // Nothing measurable is left. Declare the trailing block exactly
        // zero so that the rows of R from here on are exactly zero.
        for (Index c = j; c < n; ++c) {
          for (Index i = j; i < m; ++i) a[at(i, c, m)] = Complex(0.0, 0.0);
        }
        break;
      }
      // Unpivoted: an empty column here is a zero column of the input (the
      // trailing zero columns of R^*). Leave it, form no reflector, and
      // move on; later columns may still carry something.
      for (Index i = j; i < m; ++i) a[at(i, j, m)] = Complex(0.0, 0.0);
      continue;
    }
    if (pivot_col != j) {
      for (Index i = 0; i < m; ++i) {
        const Complex t = a[at(i, j, m)];
        a[at(i, j, m)] = a[at(i, pivot_col, m)];
        a[at(i, pivot_col, m)] = t;
      }
      const Index tp = f.perm[j];
      f.perm[j] = f.perm[pivot_col];
      f.perm[pivot_col] = tp;
    }

    // Reflector for column j, rows j..m-1.
    Complex* x = &a[at(j, j, m)];
    const Index len = m - j;
    const double xnorm = std::sqrt(best);  // best is exactly this column's norm^2
    const Complex alpha = x[0];
    // Unimodular at any magnitude, including a subnormal alpha: see
    // unit_phase. |beta| must equal ||x|| exactly, because the reflection is
    // asserted below rather than recomputed, and a phase that is not
    // unimodular silently corrupts R while leaving Q unitary.
    const Complex phase = detail::kernel::unit_phase(alpha);
    const Complex beta(-phase.real() * xnorm, -phase.imag() * xnorm);

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
      for (Index i = 0; i < len; ++i) w += conj_mul(v[i], y[i]);
      w = Complex(tau * w.real(), tau * w.imag());
      for (Index i = 0; i < len; ++i) y[i] -= mul(w, v[i]);
    }
    x[0] = beta;
    for (Index i = 1; i < len; ++i) x[i] = Complex(0.0, 0.0);
    f.rank = j + 1;
  }
}

// y (m x cols, column-major) <- Q y, with Q = P_0 P_1 ... P_{rank-1}.
static inline void apply_q(const QrFactors& f, std::vector<Complex>& y, Index cols) {
  const Index m = f.m;
  for (Index jj = f.rank; jj > 0; --jj) {
    const Index j = jj - 1;
    const Complex* v = &f.h[at(j, j, m)];
    const Index len = m - j;
    const double tau = f.tau[j];
    if (tau == 0.0) continue;  // no reflector was formed for this column
    for (Index c = 0; c < cols; ++c) {
      Complex* col = &y[at(j, c, m)];
      Complex w(0.0, 0.0);
      for (Index i = 0; i < len; ++i) w += conj_mul(v[i], col[i]);
      w = Complex(tau * w.real(), tau * w.imag());
      for (Index i = 0; i < len; ++i) col[i] -= mul(w, v[i]);
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
static inline bool jacobi_orthogonalise(std::vector<Complex>& x, Index rows, std::vector<Complex>& v,
                          Index cols, std::vector<bool>& dead) {
  const double tol = std::sqrt(static_cast<double>(rows)) * detail::kernel::unit_roundoff();
  const double floor_v = detail::kernel::column_floor();

  dead.assign(cols, false);
  // Squared column norms, recomputed at the start of every sweep and updated
  // after each rotation in between: the rotation changes ||x_p||^2 and
  // ||x_q||^2 by exactly -t|gamma| and +t|gamma|, and the drift from using
  // that formula through one sweep is far below the tolerance it feeds.
  std::vector<double> norm2(cols, 0.0);
  auto refresh = [&]() {
    for (Index j = 0; j < cols; ++j) {
      if (dead[j]) continue;
      norm2[j] = detail::kernel::norm_sq(&x[at(0, j, rows)], rows);
      if (std::sqrt(norm2[j]) < floor_v) {
        dead[j] = true;
        norm2[j] = 0.0;
        for (Index i = 0; i < rows; ++i) x[at(i, j, rows)] = Complex(0.0, 0.0);
      }
    }
  };
  refresh();

  for (int sweep = 0; sweep < kMaxSweeps; ++sweep) {
    Index rotations = 0;
    for (Index p = 0; p + 1 < cols; ++p) {
      if (dead[p]) continue;
      for (Index q = p + 1; q < cols; ++q) {
        if (dead[q]) continue;
        Complex* xp = &x[at(0, p, rows)];
        Complex* xq = &x[at(0, q, rows)];
        double gr = 0.0;
        double gi = 0.0;
        for (Index i = 0; i < rows; ++i) {
          // conj(xp) * xq
          gr += xp[i].real() * xq[i].real() + xp[i].imag() * xq[i].imag();
          gi += xp[i].real() * xq[i].imag() - xp[i].imag() * xq[i].real();
        }
        const Complex gamma(gr, gi);
        const double gamma_abs = detail::kernel::modulus(gamma);
        const double alpha = norm2[p];
        const double beta = norm2[q];
        if (!(gamma_abs > tol * std::sqrt(alpha) * std::sqrt(beta))) continue;

        const Rotation r = detail::kernel::make_rotation(alpha, beta, gamma);
        detail::kernel::rotate_columns(xp, xq, rows, r);
        detail::kernel::rotate_columns(&v[at(0, p, cols)], &v[at(0, q, cols)], cols, r);
        const double shift = r.t * r.gamma_abs;
        norm2[p] = alpha - shift;
        norm2[q] = beta + shift;
        // Both are clamped: whichever of the two ends up carrying the smaller
        // Gram eigenvalue is the one a cancellation could push below zero,
        // and a negative value here would make the next sqrt a NaN.
        if (norm2[p] < 0.0) norm2[p] = 0.0;
        if (norm2[q] < 0.0) norm2[q] = 0.0;
        ++rotations;
      }
    }
    refresh();
    if (rotations == 0) return true;
  }

  // The sweep budget is exhausted, which is not by itself a failure.
  //
  // A pair can keep clearing the rotation threshold while the rotation that
  // would clear it is too small to change either column: both updates land at
  // or below an ulp, all that survives is a sign flip, and |x_p^* x_q| comes
  // back the same on the next sweep. The pair then cycles until the budget
  // runs out. Refusing there costs the caller a correct factorisation of an
  // ordinary matrix -- measured at about one in ten thousand 2x2 and 3x3
  // inputs -- and, worse, the cycle depends on ulp-level cancellation, so the
  // same input could be refused under one floating-point model and accepted
  // under another.
  //
  // What the caller needs is not that the iteration converged but that the
  // columns are orthogonal to the precision available. Measure that directly.
  // This is the counterpart of the acceptance test the two-sided sweep in
  // eigh.cpp already had.
  //
  // The bound is four times the rotation threshold. tol is itself the noise
  // floor -- an inner product accumulated over `rows` terms carries about
  // sqrt(rows) * u of rounding, so no rotation can drive a pair below it --
  // and the slack covers a pair sitting a little above it, which is where the
  // stall happens (measured at 1.09 * tol).
  //
  // What that costs, worst case: with every one of the pairs at the bound,
  // ||X^* X - I||_F is about cols * 4 * tol = 4 * cols * sqrt(rows) * u, and
  // the harness allows 64 * max(rows, cols) * eps. For a square input those
  // are in the ratio sqrt(n) / 32, so the test stays inside the harness bound
  // for every n up to 1024 -- eight times the largest matrix this library is
  // meant for. The ratio grows as sqrt(n), so it is stated here rather than
  // left to be discovered if that cap ever moves. In practice the worst case
  // is unreachable, since it needs every pair at the bound at once while a
  // converged sweep leaves almost all of them at u; the measured ratio is
  // 0.03.
  double worst_cosine = 0.0;
  for (Index p = 0; p + 1 < cols; ++p) {
    if (dead[p]) continue;
    for (Index q = p + 1; q < cols; ++q) {
      if (dead[q]) continue;
      const Complex* xp = &x[at(0, p, rows)];
      const Complex* xq = &x[at(0, q, rows)];
      double gr = 0.0;
      double gi = 0.0;
      for (Index i = 0; i < rows; ++i) {
        gr += xp[i].real() * xq[i].real() + xp[i].imag() * xq[i].imag();
        gi += xp[i].real() * xq[i].imag() - xp[i].imag() * xq[i].real();
      }
      const double denominator = std::sqrt(norm2[p]) * std::sqrt(norm2[q]);
      if (!(denominator > 0.0)) continue;
      const double cosine = detail::kernel::modulus(Complex(gr, gi)) / denominator;
      if (cosine > worst_cosine) worst_cosine = cosine;
    }
  }
  return worst_cosine <= 4.0 * tol;
}

// --- what a core returns ---------------------------------------------------

struct CoreResult {
  Index m = 0;   // rows of the core input
  Index n = 0;   // cols of the core input
  Index k = 0;   // min(m, n)
  std::vector<Complex> u;  // m x k
  std::vector<double> s;   // k, descending
  std::vector<Complex> v;  // n x k
};

// --- driver and entry ------------------------------------------------------

// The frame around a core. `core(b, m, n, out)` receives the m x n column-
// major block that remains after the structural zero rows and columns are
// stripped (m, n >= 1, every entry finite, at least one nonzero), overwrites
// it, and fills `out` or returns false. Everything else, the bit-pattern
// checks on the way in, the stripping, the embedding of the core's factors
// on the kept rows and columns, the completion of the zero-value vectors and
// the scan on the way out, is the same for every kernel and lives here.
template <typename Core>
static inline bool svd_driver(const Complex* data, Index rows, Index cols, MatrixOrder order,
                       Complex* u_out, double* s_out, Complex* v_out, Core&& core) {
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
    CoreResult result;
    if (!core(std::move(b), mr, nc, result)) return false;

    // Embed: the core's factors live on the kept rows and columns.
    for (Index j = 0; j < result.k; ++j) {
      s[j] = result.s[j];
      for (Index i = 0; i < mr; ++i) u[at(rows_kept[i], j, rows)] = result.u[at(i, j, mr)];
      for (Index i = 0; i < nc; ++i) v[at(cols_kept[i], j, cols)] = result.v[at(i, j, nc)];
    }
    filled = result.k;
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

// The public entry checks, shared word for word by every kernel: null
// pointers and non-positive dimensions are refused, a shape whose element
// count does not fit an int is refused rather than truncated (element counts
// are int throughout the public surface), and an allocation failure inside
// the driver is turned into the false return the contract promises. Nothing
// else in there can throw.
template <typename Core>
static inline bool svd_entry(const std::complex<double>* data, int rows, int cols,
                      MatrixOrder order, std::complex<double>* U_out, double* S_out,
                      std::complex<double>* V_out, Core&& core) {
  if (data == nullptr || rows <= 0 || cols <= 0 || U_out == nullptr ||
      S_out == nullptr || V_out == nullptr) {
    return false;
  }
  if (static_cast<unsigned long long>(rows) * static_cast<unsigned long long>(cols) >
      static_cast<unsigned long long>(std::numeric_limits<int>::max())) {
    return false;
  }
  try {
    return svd_driver(data, static_cast<Index>(rows), static_cast<Index>(cols), order,
                      U_out, S_out, V_out, std::forward<Core>(core));
  } catch (...) {
    return false;
  }
}

}  // namespace svd_common
}  // namespace detail
}  // namespace autonne

#endif  // AUTONNE_SRC_DETAIL_SVD_COMMON_HPP
