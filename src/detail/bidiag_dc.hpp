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

// Singular value decomposition of a real upper bidiagonal matrix by divide
// and conquer (Gu and Eisenstat), the real core of the BDC kernel. Private to
// src/ and `static` throughout, for the reason kernel_common.hpp gives.
//
// The input is the diagonal d[0..n) and superdiagonal e[0..n-1) of an n x m
// upper bidiagonal B, m = n or n + 1 (an n x (n + 1) block carries e[n-1] in
// its last column; it arises as the left half of every split, so the whole
// recursion works on that shape). Every entry is finite and at most 1 in
// magnitude, which the caller arranges by scaling. The output is B = U S V^T
// with U (n x n) and V (m x m) real orthogonal and S the n values in
// descending order; when m = n + 1 the last column of V is the null
// direction.
//
// The recursion:
//
//   * A block of order at most `leaf` is expanded to a dense n x m matrix and
//     handed to one-sided Jacobi on real columns (jacobi_real), the real twin
//     of the sweep in svd_common.hpp.
//   * A larger block is split at its middle row k. Rows above k form an
//     upper bidiagonal k x (k + 1) block, rows below form an
//     (n - k - 1) x (m - k - 1) block, and row k couples them through d[k]
//     and e[k]. Both halves are solved recursively.
//   * The halves are merged: with their SVDs substituted, B is orthogonally
//     equivalent to an arrow matrix, one full row z^T above a diagonal, and
//     the singular values of that are the roots of the secular equation
//     1 + sum_j z_j^2 / (d_j^2 - s^2) = 0, one in each gap of the sorted d.
//     Deflation removes every column with a negligible z and every pair of
//     equal d before the solve, which is where exact degeneracy and exact
//     zeros are handled: no gap the solver divides by can be smaller than
//     the deflation tolerance. The singular vectors are then formed not from
//     z but from the z-hat whose secular roots are exactly the computed
//     values (Loewner's theorem), which is what keeps them orthonormal when
//     two values are close.
//
// Nothing here can produce a NaN or an infinity: every quotient's
// denominator is either a norm bounded below by the column floor, a gap
// bounded below by the deflation tolerance, or a sum of positive terms. No
// result depends on the order of a summation: the secular iteration
// re-evaluates the function at its current point on every step, and each
// deflation decision is a comparison of finite values.
//
// A false return means the sweep or iteration budget was exhausted; nothing
// is written to the outputs in that case.

#ifndef AUTONNE_SRC_DETAIL_BIDIAG_DC_HPP
#define AUTONNE_SRC_DETAIL_BIDIAG_DC_HPP

#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

#include "kernel_common.hpp"
#include "svd_common.hpp"

namespace autonne {
namespace detail {
namespace bidiag_dc {

using kernel::Index;
using kernel::at;
using svd_common::kMaxSweeps;

// Secular iterations allowed per root. The rational iteration converges
// quadratically from a bracketed start, and a step that leaves the bracket
// is replaced by a bisection step, so twenty is a large margin over the
// handful a root takes; sixty is the same five-fold margin the sweep budget
// carries.
constexpr int kMaxSecularIterations = 60;

// --- real helpers ----------------------------------------------------------

static inline double norm_sq(const double* x, Index n) noexcept {
  double acc = 0.0;
  for (Index i = 0; i < n; ++i) acc += x[i] * x[i];
  return acc;
}

// Fills columns [have, want) of the n x want column-major matrix x with
// vectors orthonormal to columns [0, have), which are assumed orthonormal.
// The real twin of kernel::complete_orthonormal: the candidate is the
// canonical vector furthest from the current span, orthogonalised twice.
static inline void complete_orthonormal_real(double* x, Index n, Index have, Index want,
                                      std::vector<double>& work) {
  work.resize(n);
  for (Index col = have; col < want; ++col) {
    Index best = 0;
    double best_residual = -1.0;
    for (Index cand = 0; cand < n; ++cand) {
      double projected = 0.0;
      for (Index c = 0; c < col; ++c) {
        const double e = x[at(cand, c, n)];
        projected += e * e;
      }
      const double residual = 1.0 - projected;
      if (residual > best_residual) {
        best_residual = residual;
        best = cand;
      }
    }
    for (Index i = 0; i < n; ++i) work[i] = 0.0;
    work[best] = 1.0;
    for (int pass = 0; pass < 2; ++pass) {
      for (Index c = 0; c < col; ++c) {
        double dot = 0.0;
        for (Index i = 0; i < n; ++i) dot += x[at(i, c, n)] * work[i];
        for (Index i = 0; i < n; ++i) work[i] -= dot * x[at(i, c, n)];
      }
    }
    const double inv = 1.0 / std::sqrt(norm_sq(work.data(), n));
    for (Index i = 0; i < n; ++i) x[at(i, col, n)] = work[i] * inv;
  }
}

// --- one-sided Jacobi on real columns --------------------------------------

// Orthogonalises the `cols` columns of x (rows x cols, column-major) by plane
// rotations accumulated in v (cols x cols, starts as identity). Threshold,
// floor, norm bookkeeping and the acceptance test at budget exhaustion are
// those of svd_common::jacobi_orthogonalise, whose comments explain each;
// only the arithmetic is real. The 2 x 2 rotation comes from the same
// make_rotation, with a real gamma, so its phase is the sign of gamma.
static inline bool jacobi_real(std::vector<double>& x, Index rows, std::vector<double>& v,
                        Index cols, std::vector<bool>& dead) {
  const double tol = std::sqrt(static_cast<double>(rows)) * kernel::unit_roundoff();
  const double floor_v = kernel::column_floor();

  dead.assign(cols, false);
  std::vector<double> norm2(cols, 0.0);
  auto refresh = [&]() {
    for (Index j = 0; j < cols; ++j) {
      if (dead[j]) continue;
      norm2[j] = norm_sq(&x[at(0, j, rows)], rows);
      if (std::sqrt(norm2[j]) < floor_v) {
        dead[j] = true;
        norm2[j] = 0.0;
        for (Index i = 0; i < rows; ++i) x[at(i, j, rows)] = 0.0;
      }
    }
  };
  refresh();

  auto rotate = [](double* xp, double* xq, Index n, double c, double s, double sign) {
    for (Index i = 0; i < n; ++i) {
      const double a = xp[i];
      const double b = sign * xq[i];
      xp[i] = c * a - s * b;
      xq[i] = s * a + c * b;
    }
  };

  for (int sweep = 0; sweep < kMaxSweeps; ++sweep) {
    Index rotations = 0;
    for (Index p = 0; p + 1 < cols; ++p) {
      if (dead[p]) continue;
      for (Index q = p + 1; q < cols; ++q) {
        if (dead[q]) continue;
        double* xp = &x[at(0, p, rows)];
        double* xq = &x[at(0, q, rows)];
        double gamma = 0.0;
        for (Index i = 0; i < rows; ++i) gamma += xp[i] * xq[i];
        const double gamma_abs = std::fabs(gamma);
        const double alpha = norm2[p];
        const double beta = norm2[q];
        if (!(gamma_abs > tol * std::sqrt(alpha) * std::sqrt(beta))) continue;

        const kernel::Rotation r =
            kernel::make_rotation(alpha, beta, kernel::Complex(gamma, 0.0));
        const double sign = r.phase.real();
        rotate(xp, xq, rows, r.c, r.s, sign);
        rotate(&v[at(0, p, cols)], &v[at(0, q, cols)], cols, r.c, r.s, sign);
        const double shift = r.t * r.gamma_abs;
        norm2[p] = alpha - shift;
        norm2[q] = beta + shift;
        if (norm2[p] < 0.0) norm2[p] = 0.0;
        if (norm2[q] < 0.0) norm2[q] = 0.0;
        ++rotations;
      }
    }
    refresh();
    if (rotations == 0) return true;
  }

  double worst_cosine = 0.0;
  for (Index p = 0; p + 1 < cols; ++p) {
    if (dead[p]) continue;
    for (Index q = p + 1; q < cols; ++q) {
      if (dead[q]) continue;
      double gamma = 0.0;
      for (Index i = 0; i < rows; ++i) gamma += x[at(i, p, rows)] * x[at(i, q, rows)];
      const double denominator = std::sqrt(norm2[p]) * std::sqrt(norm2[q]);
      if (!(denominator > 0.0)) continue;
      const double cosine = std::fabs(gamma) / denominator;
      if (cosine > worst_cosine) worst_cosine = cosine;
    }
  }
  return worst_cosine <= 4.0 * tol;
}

// --- real Householder QR ---------------------------------------------------

// The real twin of svd_common::qr_householder and apply_q; the comments there
// explain the pivoting, the floor and the reflector convention. Reflectors
// are P_j = I - tau_j v_j v_j^T with P_j x = beta e_1, beta = -sign(x_1) ||x||.
struct QrFactorsReal {
  Index m = 0;
  Index n = 0;
  Index rank = 0;
  std::vector<double> h;     // m x n, column j holds v_j in rows j..m-1
  std::vector<double> tau;   // 2 / (v_j^T v_j)
  std::vector<Index> perm;   // A P has column perm[j] of A in position j
};

static inline void qr_householder_real(std::vector<double>& a, Index m, Index n, bool pivot,
                                       QrFactorsReal& f) {
  f.m = m;
  f.n = n;
  f.rank = 0;
  f.h.assign(m * n, 0.0);
  f.tau.assign(n, 0.0);
  f.perm.resize(n);
  for (Index j = 0; j < n; ++j) f.perm[j] = j;
  const double floor_sq = kernel::column_floor() * kernel::column_floor();

  for (Index j = 0; j < n && j < m; ++j) {
    Index pivot_col = j;
    double best = -1.0;
    if (pivot) {
      for (Index c = j; c < n; ++c) {
        const double nrm = norm_sq(&a[at(j, c, m)], m - j);
        if (nrm > best) {
          best = nrm;
          pivot_col = c;
        }
      }
    } else {
      best = norm_sq(&a[at(j, j, m)], m - j);
    }
    if (best < floor_sq) {
      if (pivot) {
        for (Index c = j; c < n; ++c) {
          for (Index i = j; i < m; ++i) a[at(i, c, m)] = 0.0;
        }
        break;
      }
      for (Index i = j; i < m; ++i) a[at(i, j, m)] = 0.0;
      continue;
    }
    if (pivot_col != j) {
      for (Index i = 0; i < m; ++i) {
        const double t = a[at(i, j, m)];
        a[at(i, j, m)] = a[at(i, pivot_col, m)];
        a[at(i, pivot_col, m)] = t;
      }
      const Index tp = f.perm[j];
      f.perm[j] = f.perm[pivot_col];
      f.perm[pivot_col] = tp;
    }
    double* x = &a[at(j, j, m)];
    const Index len = m - j;
    const double xnorm = std::sqrt(best);
    const double beta = (x[0] < 0.0) ? xnorm : -xnorm;
    double* v = &f.h[at(j, j, m)];
    v[0] = x[0] - beta;
    for (Index i = 1; i < len; ++i) v[i] = x[i];
    const double tau = 2.0 / norm_sq(v, len);  // |v_1| = |x_1| + xnorm > 0
    f.tau[j] = tau;
    for (Index c = j + 1; c < n; ++c) {
      double* y = &a[at(j, c, m)];
      double w = 0.0;
      for (Index i = 0; i < len; ++i) w += v[i] * y[i];
      w *= tau;
      for (Index i = 0; i < len; ++i) y[i] -= w * v[i];
    }
    x[0] = beta;
    for (Index i = 1; i < len; ++i) x[i] = 0.0;
    f.rank = j + 1;
  }
}

// y (m x cols, column-major) <- Q y, with Q = P_0 P_1 ... P_{rank-1}.
static inline void apply_q_real(const QrFactorsReal& f, std::vector<double>& y, Index cols) {
  const Index m = f.m;
  for (Index jj = f.rank; jj > 0; --jj) {
    const Index j = jj - 1;
    const double* v = &f.h[at(j, j, m)];
    const Index len = m - j;
    const double tau = f.tau[j];
    if (tau == 0.0) continue;
    for (Index c = 0; c < cols; ++c) {
      double* col = &y[at(j, c, m)];
      double w = 0.0;
      for (Index i = 0; i < len; ++i) w += v[i] * col[i];
      w *= tau;
      for (Index i = 0; i < len; ++i) col[i] -= w * v[i];
    }
  }
}

// --- leaves ----------------------------------------------------------------

// SVD of a dense real rows x cols block, cols = rows or rows + 1, by the
// route the complex kernel takes: Householder QR with column pivoting,
// A P = Q R, then one-sided Jacobi on the columns of X = R^T, then
// U = Q V_X and V = P U_X. The pivoted QR is what makes this safe on exact
// rank deficiency, whether from a zero on the diagonal or from the extra
// column: it drives the deficiency into rows of R that are exactly zero,
// which are exactly zero columns of X, so Jacobi never receives a column
// that is pure rounding noise. Handed such a column directly, the sweep
// cannot orthogonalise it below its threshold and cycles until the budget
// is spent. It also never receives more columns than rows, since X is
// cols x rows.
//
// On return u is rows x rows, s holds `rows` values in descending order and
// v is cols x cols; the columns of v beyond the rank, including the null
// direction when cols = rows + 1, are completed orthonormally.
static inline bool leaf_svd(std::vector<double> x, Index rows, Index cols,
                            std::vector<double>& u, std::vector<double>& s,
                            std::vector<double>& v) {
  const Index m = rows;
  const Index n = cols;
  const Index r = m;  // min(rows, cols)
  QrFactorsReal qr;
  qr_householder_real(x, m, n, true, qr);

  // X = R^T, n x r; rows of R at or beyond the rank are exactly zero.
  std::vector<double> xt(n * r, 0.0);
  for (Index j = 0; j < r; ++j) {
    for (Index i = 0; i < n; ++i) {
      xt[at(i, j, n)] = (j <= i && j < qr.rank) ? x[at(j, i, m)] : 0.0;
    }
  }
  std::vector<double> vx(r * r, 0.0);
  for (Index j = 0; j < r; ++j) vx[at(j, j, r)] = 1.0;
  std::vector<bool> dead;
  if (!jacobi_real(xt, n, vx, r, dead)) return false;

  std::vector<double> value(r, 0.0);
  Index live = 0;
  for (Index j = 0; j < r; ++j) {
    if (dead[j]) continue;
    value[j] = std::sqrt(norm_sq(&xt[at(0, j, n)], n));
    ++live;
  }
  std::vector<Index> order;
  kernel::sort_indices(value.data(), r, true, order);

  // Normalised columns of X in sorted order are U_X (n x r); the sorted
  // rotations are V_X (r x r).
  std::vector<double> ux(n * r, 0.0);
  std::vector<double> vx_sorted(r * r);
  s.assign(r, 0.0);
  for (Index jj = 0; jj < r; ++jj) {
    const Index src = order[jj];
    s[jj] = value[src];
    for (Index i = 0; i < r; ++i) vx_sorted[at(i, jj, r)] = vx[at(i, src, r)];
    if (dead[src]) continue;
    const double inv = 1.0 / value[src];
    for (Index i = 0; i < n; ++i) ux[at(i, jj, n)] = xt[at(i, src, n)] * inv;
  }
  std::vector<double> work;
  complete_orthonormal_real(ux.data(), n, live, r, work);

  // U = Q V_X (m x r, and m = r), V = P U_X completed to n x n.
  u.assign(m * r, 0.0);
  for (Index j = 0; j < r; ++j) {
    for (Index i = 0; i < r; ++i) u[at(i, j, m)] = vx_sorted[at(i, j, r)];
  }
  apply_q_real(qr, u, r);
  v.assign(n * n, 0.0);
  for (Index j = 0; j < r; ++j) {
    for (Index i = 0; i < n; ++i) v[at(qr.perm[i], j, n)] = ux[at(i, j, n)];
  }
  complete_orthonormal_real(v.data(), n, r, n, work);
  return true;
}

// --- the secular equation --------------------------------------------------

// A root of the secular equation, in the form that makes the vector
// formulas accurate: sigma = d[origin] + tau, with `origin` the nearer of
// the two poles bracketing the root and `tau` the small signed offset from
// it. Every d[j] - sigma the vector formulas need is then a difference of
// well-separated numbers minus a small one, rather than d[j] - sigma with
// the cancellation that carries when sigma is within rounding of d[j].
// Loewner's formula and the vectors built from it need exactly these
// differences to keep their orthogonality when a root sits close to a pole.
//
// The differences are stored, not recomputed. The three-term form
// (d[j] - d[origin]) - tau is what a fast-math build may reassociate into
// d[j] - (d[origin] + tau), and for j = origin that is d - fl(d + tau),
// which is zero whenever |tau| is below half an ulp of d: a zero
// denominator, and infinities in everything built on it. So d[j] - d[origin]
// is formed once into an array, with the origin's entry set to zero rather
// than computed, and every later use is the two-term dd[j] - tau, which has
// nothing to reassociate with. The same holds for the sums.
struct SecularRoot {
  double sigma = 0.0;
  Index origin = 0;
  double tau = 0.0;
  std::vector<double> delta;  // d[j] - sigma for every j
  std::vector<double> w;      // d[j] + sigma for every j
};

// The p roots of f(sigma) = 1 + sum_j z_j^2 / ((d_j - sigma)(d_j + sigma))
// for d[0..p) ascending and distinct by more than the deflation tolerance
// and z[0..p) every entry above it in magnitude. Root i lies in
// (d[i], d[i+1]); the last in (d[p-1], d[p-1] + ||z||], since
// sigma_max^2 <= d[p-1]^2 + ||z||^2.
//
// An interior root is found by the rational iteration of Bunch, Nielsen and
// Sorensen as Li arranged it: f is split into the terms whose poles lie
// below the root and those whose poles lie above, each part is modelled by
// a single pole at the nearer end of the interval plus a constant, matched
// to the part's value and slope at the current point, and the model's zero
// is the next point. The model is exact when only the two bracketing terms
// exist, and converges quadratically otherwise. A step that would leave the
// bracket, or would move away from the root, is replaced by bisection, so
// the bracket shrinks on every step regardless and the iteration cannot
// wander. The last root has poles on one side only, where f is increasing
// and concave, and Newton's method from the upper end of its bracket
// converges monotonically; the same bisection guard applies.
//
// Convergence is |f| at the rounding noise of evaluating it, or a step of
// zero. There is deliberately no exit on the width of the bracket: a root
// inside a gap of a few ulps must still be resolved to relative accuracy in
// tau, which is what the shifted form is for, and a width measured against
// d would stop the iteration while tau still carried an error of several
// percent. The rounding noise in f scales with the terms, which are large
// in exactly that regime, so the |f| test is tight in tau there too. Every
// denominator is a product of a gap the deflation tolerance keeps open and
// a sum of two positive numbers, so none can vanish.
static inline bool secular_roots(const double* d, const double* z, Index p,
                          std::vector<SecularRoot>& roots) {
  const double eps = std::numeric_limits<double>::epsilon();
  roots.assign(p, SecularRoot{});

  double z_norm_sq = 0.0;
  for (Index j = 0; j < p; ++j) z_norm_sq += z[j] * z[j];
  const double z_norm = std::sqrt(z_norm_sq);

  // f and its parts at sigma = d[origin] + tau, with slopes taken with
  // respect to lambda = sigma^2, the variable in which f is a sum of simple
  // poles: d/dlambda of z^2 / (d^2 - lambda) is z^2 / (d^2 - lambda)^2, and
  // d^2 - lambda is formed as delta * w. Terms j <= split are the part with
  // poles below the root.
  struct Eval {
    double f = 0.0;
    double abs_sum = 0.0;  // sum |term_j|, the scale of rounding in f
    double psi = 0.0;      // sum of terms j <= split
    double phi = 0.0;      // sum of terms j > split
    double dpsi = 0.0;     // d psi / d lambda
    double dphi = 0.0;     // d phi / d lambda
  };
  // dd[j] = d[j] - d[origin] and ds[j] = d[j] + d[origin] for the current
  // origin, formed once per root (see SecularRoot).
  std::vector<double> dd(p);
  std::vector<double> ds(p);
  auto set_origin = [&](Index origin) {
    for (Index j = 0; j < p; ++j) {
      dd[j] = d[j] - d[origin];
      ds[j] = d[j] + d[origin];
    }
    dd[origin] = 0.0;
  };
  auto evaluate = [&](double tau, Index split) {
    Eval e;
    for (Index j = 0; j < p; ++j) {
      const double delta = dd[j] - tau;
      const double w = ds[j] + tau;
      const double denom = delta * w;
      const double term = (z[j] * z[j]) / denom;
      e.abs_sum += std::fabs(term);
      const double slope = (z[j] * z[j]) / (denom * denom);
      if (j <= split) {
        e.psi += term;
        e.dpsi += slope;
      } else {
        e.phi += term;
        e.dphi += slope;
      }
    }
    e.f = 1.0 + e.psi + e.phi;
    return e;
  };

  // A step eta in lambda = sigma^2, applied to sigma = d[origin] + tau as a
  // step in tau: sigma_new = sqrt(sigma^2 + eta), formed without the
  // cancellation of taking the root and subtracting. Returns false when the
  // step would leave lambda non-positive, which the caller answers with
  // bisection.
  auto lambda_step = [](double sigma, double eta, double& eta_sigma) {
    const double next = sigma * sigma + eta;
    if (!(next > 0.0)) return false;
    eta_sigma = eta / (sigma + std::sqrt(next));
    return true;
  };

  for (Index i = 0; i < p; ++i) {
    SecularRoot& r = roots[i];
    if (p == 1) {
      // f = 1 + z^2 / (d^2 - sigma^2): sigma^2 = d^2 + z^2, in closed form.
      r.origin = 0;
      r.tau = std::hypot(d[0], z[0]) - d[0];
      r.sigma = d[0] + r.tau;
      r.delta.assign(1, -r.tau);
      r.w.assign(1, (d[0] + d[0]) + r.tau);
      continue;
    }

    const bool last = (i + 1 == p);
    double lo = 0.0;  // bracket in tau, open at both ends
    double hi = 0.0;
    if (last) {
      r.origin = p - 1;
      set_origin(r.origin);
      lo = 0.0;
      hi = z_norm;
      r.tau = hi;
    } else {
      // Which pole is nearer decides the origin: at the midpoint f is
      // positive when the root lies below it.
      const double half_gap = 0.5 * (d[i + 1] - d[i]);
      set_origin(i);
      const Eval mid = evaluate(half_gap, i);
      if (mid.f > 0.0) {
        r.origin = i;
        lo = 0.0;
        hi = half_gap;
        r.tau = half_gap;
      } else {
        r.origin = i + 1;
        set_origin(r.origin);
        lo = -half_gap;
        hi = 0.0;
        r.tau = -half_gap;
      }
    }

    bool converged = false;
    for (int it = 0; it < kMaxSecularIterations; ++it) {
      const Eval e = evaluate(r.tau, i);
      const double noise = 8.0 * eps * (1.0 + e.abs_sum);
      if (std::fabs(e.f) <= noise) {
        converged = true;
        break;
      }
      // f increases through the interval: a positive value means the root
      // lies below the current point.
      if (e.f > 0.0) {
        hi = r.tau;
      } else {
        lo = r.tau;
      }

      // The step, first in lambda, then converted.
      const double sigma = d[r.origin] + r.tau;
      const double slope = e.dpsi + e.dphi;  // df / dlambda, positive
      double eta_lambda = 0.0;
      if (last) {
        // One pole below, none above: f ~ c + a / (L - eta) with L the
        // lambda-distance to the last pole, matched to f and f'.
        const double L = (dd[p - 1] - r.tau) * (ds[p - 1] + r.tau);  // negative
        const double a = slope * L * L;
        const double c = e.f - slope * L;
        eta_lambda = (c != 0.0) ? (L + a / c) : (-e.f / slope);
      } else {
        // Two poles, one each side: f ~ c + a / (Li - eta) + b / (Lip - eta).
        const double Li = (dd[i] - r.tau) * (ds[i] + r.tau);          // negative
        const double Lip = (dd[i + 1] - r.tau) * (ds[i + 1] + r.tau);  // positive
        const double a = e.dpsi * Li * Li;
        const double b = e.dphi * Lip * Lip;
        const double c = e.f - e.dpsi * Li - e.dphi * Lip;
        // c (Li - eta)(Lip - eta) + a (Lip - eta) + b (Li - eta) = 0, i.e.
        // qa eta^2 - qb eta + qc = 0; the root between the poles is wanted.
        const double qa = c;
        const double qb = c * (Li + Lip) + a + b;
        const double qc = c * Li * Lip + a * Lip + b * Li;
        double disc = qb * qb - 4.0 * qa * qc;
        if (disc < 0.0) disc = 0.0;
        const double sq = std::sqrt(disc);
        double r1 = 0.0;
        double r2 = 0.0;
        if (qa == 0.0) {
          r1 = r2 = (qb != 0.0) ? qc / qb : 0.0;
        } else {
          // The larger-magnitude root without cancellation, the other by
          // the product of the roots.
          const double big = (qb >= 0.0) ? (qb + sq) / (2.0 * qa) : (qb - sq) / (2.0 * qa);
          r1 = big;
          r2 = (big != 0.0) ? qc / (qa * big) : 0.0;
        }
        const bool in1 = (r1 > Li && r1 < Lip);
        const bool in2 = (r2 > Li && r2 < Lip);
        if (in1 && !in2) {
          eta_lambda = r1;
        } else if (in2 && !in1) {
          eta_lambda = r2;
        } else if (in1 && in2) {
          eta_lambda = (std::fabs(r1) < std::fabs(r2)) ? r1 : r2;
        } else {
          eta_lambda = -e.f / slope;
        }
      }
      // The step must move toward the root: f > 0 means lambda must fall.
      if (!(e.f * eta_lambda < 0.0)) eta_lambda = -e.f / slope;
      double eta = 0.0;
      double next = lambda_step(sigma, eta_lambda, eta) ? r.tau + eta : 0.5 * (lo + hi);
      if (!(next > lo && next < hi)) next = 0.5 * (lo + hi);
      if (next == r.tau) {
        converged = true;
        break;
      }
      r.tau = next;
    }
    if (!converged) return false;
    r.sigma = d[r.origin] + r.tau;
    r.delta.resize(p);
    r.w.resize(p);
    for (Index j = 0; j < p; ++j) {
      r.delta[j] = dd[j] - r.tau;
      r.w[j] = ds[j] + r.tau;
    }
  }
  return true;
}

// --- the merge -------------------------------------------------------------

// B = U S V^T for an n x m block, m = n or n + 1.
struct Factors {
  Index n = 0;
  Index m = 0;
  std::vector<double> u;  // n x n
  std::vector<double> s;  // n, descending
  std::vector<double> v;  // m x m
};

// Applies the plane rotation [c s; -s c] to columns p and q of the ld x cols
// matrix x: col_p <- c col_p + s col_q, col_q <- -s col_p + c col_q.
static inline void rotate_pair(double* x, Index ld, Index p, Index q, double c,
                               double s) noexcept {
  double* xp = x + p * ld;
  double* xq = x + q * ld;
  for (Index i = 0; i < ld; ++i) {
    const double a = xp[i];
    const double b = xq[i];
    xp[i] = c * a + s * b;
    xq[i] = -s * a + c * b;
  }
}

// Merges the SVDs of the two halves of a split. `left` is the k x (k + 1)
// block above row k, `right` the block below it, alpha = d[k] and
// beta = e[k] the two entries of row k. With the halves' factors
// substituted, B = U_full A V_full^T where U_full = diag(U_l, 1, U_r),
// V_full = diag(V_l, V_r), and A is the arrow matrix: one full row z^T in
// row k, with z = (alpha * last row of V_l, beta * first row of V_r), and
// otherwise one entry per column, the halves' singular values, each in its
// own row. The columns of A that carry no such entry (the extra column of
// the left block, and the last column when m = n + 1) have d = 0.
//
// Deflation first. Columns are sorted by d, and every run of columns whose
// d agree within tol is rotated so that the run's z concentrates in one
// representative and the rest carry z = 0 exactly; a column with z = 0 is
// then a singular value d with its own left and right vector, and takes no
// part in the solve. The representative of the d = 0 run is always a
// column without a row of its own, so that the active arrow is square. A
// representative whose own |z| is at most tol is deflated too. What is
// left, p columns with distinct d and z above tol, is the secular problem.
// The rotations are applied to V_full's columns and, for a pair that both
// own a row, to U_full's; the off-diagonal the left rotation leaves behind
// is at most tol/2 and is dropped, which is the deflation's backward error.
//
// Then the roots, Loewner's z-hat, and the vectors of Gu and Eisenstat's
// Lemma 3.1: for root i with delta_j = d_j - sigma_i and w_j = d_j + sigma_i,
// v_i is proportional to (zhat_j / (delta_j w_j))_j and u_i to
// (-1, d_j zhat_j / (delta_j w_j))_j over the z row and the active rows.
// The merged vectors are U_full and V_full times those.
static inline bool merge(const Factors& left, const Factors& right, double alpha, double beta,
                  Factors& out) {
  const double eps = std::numeric_limits<double>::epsilon();
  const Index nl = left.n;
  const Index ml = left.m;  // nl + 1
  const Index nr = right.n;
  const Index mr = right.m;
  const Index n = nl + 1 + nr;
  const Index m = ml + mr;
  const Index zrow = nl;  // column of U_full that is the middle row

  // U_full and V_full, and the arrow's data per column.
  std::vector<double> uf(n * n, 0.0);
  for (Index j = 0; j < nl; ++j) {
    for (Index i = 0; i < nl; ++i) uf[at(i, j, n)] = left.u[at(i, j, nl)];
  }
  uf[at(zrow, zrow, n)] = 1.0;
  for (Index j = 0; j < nr; ++j) {
    for (Index i = 0; i < nr; ++i) uf[at(nl + 1 + i, nl + 1 + j, n)] = right.u[at(i, j, nr)];
  }
  std::vector<double> vf(m * m, 0.0);
  for (Index j = 0; j < ml; ++j) {
    for (Index i = 0; i < ml; ++i) vf[at(i, j, m)] = left.v[at(i, j, ml)];
  }
  for (Index j = 0; j < mr; ++j) {
    for (Index i = 0; i < mr; ++i) vf[at(ml + i, ml + j, m)] = right.v[at(i, j, mr)];
  }
  std::vector<double> d(m, 0.0);
  std::vector<double> z(m, 0.0);
  std::vector<bool> has_row(m, false);
  std::vector<Index> row(m, 0);
  for (Index c = 0; c < ml; ++c) {
    z[c] = alpha * left.v[at(nl, c, ml)];
    if (c < nl) {
      d[c] = left.s[c];
      has_row[c] = true;
      row[c] = c;
    }
  }
  for (Index j = 0; j < mr; ++j) {
    const Index c = ml + j;
    z[c] = beta * right.v[at(0, j, mr)];
    if (j < nr) {
      d[c] = right.s[j];
      has_row[c] = true;
      row[c] = nl + 1 + j;
    }
  }

  double scale = 0.0;
  for (Index c = 0; c < m; ++c) {
    if (std::fabs(d[c]) > scale) scale = std::fabs(d[c]);
    if (std::fabs(z[c]) > scale) scale = std::fabs(z[c]);
  }
  const double tol = 8.0 * eps * scale;

  // Deflation. `rep` is the current run's representative.
  std::vector<Index> order;
  kernel::sort_indices(d.data(), m, false, order);
  std::vector<bool> deflated(m, false);
  Index rep = order[0];
  for (Index t = 1; t < m; ++t) {
    const Index c = order[t];
    if (!(d[c] - d[rep] <= tol)) {
      rep = c;
      continue;
    }
    // Same run. Keep a rowless column as the representative if one exists.
    Index keep = rep;
    Index kill = c;
    if (has_row[keep] && !has_row[kill]) {
      keep = c;
      kill = rep;
    }
    const double r = std::hypot(z[keep], z[kill]);
    if (r > 0.0) {
      const double cs = z[keep] / r;
      const double sn = z[kill] / r;
      rotate_pair(vf.data(), m, keep, kill, cs, sn);
      if (has_row[keep] && has_row[kill]) rotate_pair(uf.data(), n, row[keep], row[kill], cs, sn);
      z[keep] = r;
    }
    z[kill] = 0.0;
    deflated[kill] = true;
    rep = keep;
  }
  for (Index c = 0; c < m; ++c) {
    if (!deflated[c] && std::fabs(z[c]) <= tol) {
      z[c] = 0.0;
      deflated[c] = true;
    }
  }

  // The active set, ascending in d.
  std::vector<Index> active;
  for (Index t = 0; t < m; ++t) {
    if (!deflated[order[t]]) active.push_back(order[t]);
  }
  const Index p = active.size();

  // Every column becomes one triple (sigma, u column or none, v column).
  std::vector<double> sigma(m, 0.0);
  std::vector<double> u_cols(n * m, 0.0);  // column c: the left vector for column c
  std::vector<bool> has_u(m, false);
  std::vector<double> v_cols(m * m, 0.0);

  for (Index c = 0; c < m; ++c) {
    if (deflated[c]) {
      sigma[c] = d[c];
      for (Index i = 0; i < m; ++i) v_cols[at(i, c, m)] = vf[at(i, c, m)];
      if (has_row[c]) {
        has_u[c] = true;
        for (Index i = 0; i < n; ++i) u_cols[at(i, c, n)] = uf[at(i, row[c], n)];
      }
    }
  }

  if (p > 0) {
    std::vector<double> da(p), za(p);
    for (Index j = 0; j < p; ++j) {
      da[j] = d[active[j]];
      za[j] = z[active[j]];
    }
    std::vector<SecularRoot> roots;
    if (!secular_roots(da.data(), za.data(), p, roots)) return false;

    // Loewner: zhat_j^2 = prod_i (sigma_i^2 - d_j^2) / prod_{i != j} (d_i^2 - d_j^2),
    // paired so that every ratio is positive and of moderate size.
    std::vector<double> zhat(p);
    for (Index j = 0; j < p; ++j) {
      double prod = -roots[p - 1].delta[j] * roots[p - 1].w[j];
      for (Index i = 0; i < j; ++i) {
        prod *= (roots[i].delta[j] * roots[i].w[j]) /
                ((da[j] - da[i]) * (da[j] + da[i]));
      }
      for (Index i = j; i + 1 < p; ++i) {
        prod *= (-roots[i].delta[j] * roots[i].w[j]) /
                ((da[i + 1] - da[j]) * (da[i + 1] + da[j]));
      }
      if (prod < 0.0) prod = 0.0;
      zhat[j] = (za[j] < 0.0) ? -std::sqrt(prod) : std::sqrt(prod);
    }

    // Vectors, then the products with U_full and V_full.
    std::vector<double> vsec(p), usec(p + 1);
    for (Index i = 0; i < p; ++i) {
      const Index c = active[i];
      sigma[c] = roots[i].sigma;
      double vn = 0.0;
      double un = 1.0;
      usec[0] = -1.0;
      for (Index j = 0; j < p; ++j) {
        const double denom = roots[i].delta[j] * roots[i].w[j];
        vsec[j] = zhat[j] / denom;
        vn += vsec[j] * vsec[j];
        usec[j + 1] = da[j] * vsec[j];
        un += usec[j + 1] * usec[j + 1];
      }
      const double vinv = 1.0 / std::sqrt(vn);
      const double uinv = 1.0 / std::sqrt(un);
      for (Index j = 0; j < p; ++j) {
        const double w = vsec[j] * vinv;
        const double* col = &vf[at(0, active[j], m)];
        for (Index r = 0; r < m; ++r) v_cols[at(r, c, m)] += col[r] * w;
      }
      has_u[c] = true;
      {
        const double w = usec[0] * uinv;
        const double* col = &uf[at(0, zrow, n)];
        for (Index r = 0; r < n; ++r) u_cols[at(r, c, n)] += col[r] * w;
      }
      for (Index j = 0; j < p; ++j) {
        if (!has_row[active[j]]) continue;
        const double w = usec[j + 1] * uinv;
        const double* col = &uf[at(0, row[active[j]], n)];
        for (Index r = 0; r < n; ++r) u_cols[at(r, c, n)] += col[r] * w;
      }
    }
  }

  // Descending, with the triples that own no left vector last among the
  // zeros: those are the null direction when m = n + 1, and the single
  // left vector still to be completed otherwise.
  std::vector<Index> by_sigma;
  kernel::sort_indices(sigma.data(), m, true, by_sigma);
  std::vector<Index> final_order;
  final_order.reserve(m);
  for (Index t = 0; t < m; ++t) {
    if (has_u[by_sigma[t]]) final_order.push_back(by_sigma[t]);
  }
  for (Index t = 0; t < m; ++t) {
    if (!has_u[by_sigma[t]]) final_order.push_back(by_sigma[t]);
  }

  out.n = n;
  out.m = m;
  out.s.assign(n, 0.0);
  out.u.assign(n * n, 0.0);
  out.v.assign(m * m, 0.0);
  Index filled = 0;
  for (Index t = 0; t < m; ++t) {
    const Index c = final_order[t];
    for (Index i = 0; i < m; ++i) out.v[at(i, t, m)] = v_cols[at(i, c, m)];
    if (t < n) {
      out.s[t] = sigma[c];
      if (has_u[c]) {
        for (Index i = 0; i < n; ++i) out.u[at(i, t, n)] = u_cols[at(i, c, n)];
        filled = t + 1;
      }
    }
  }
  if (filled < n) {
    std::vector<double> work;
    complete_orthonormal_real(out.u.data(), n, filled, n, work);
  }
  return true;
}

// --- recursion -------------------------------------------------------------

// SVD of the n x m upper bidiagonal block with diagonal d[0..n) and
// superdiagonal e[0..m-1) (so e[n-1] exists exactly when m = n + 1 and sits
// in the last column). Blocks of order at most `leaf` are solved densely.
static inline bool solve(const double* d, const double* e, Index n, Index m, Index leaf,
                  Factors& out) {
  if (n <= leaf) {
    std::vector<double> dense(n * m, 0.0);
    for (Index i = 0; i < n; ++i) {
      dense[at(i, i, n)] = d[i];
      if (i + 1 < m) dense[at(i, i + 1, n)] = e[i];
    }
    out.n = n;
    out.m = m;
    return leaf_svd(std::move(dense), n, m, out.u, out.s, out.v);
  }
  const Index k = n / 2;  // n >= 3 here, so both halves are non-empty
  Factors left;
  Factors right;
  if (!solve(d, e, k, k + 1, leaf, left)) return false;
  if (!solve(d + k + 1, e + k + 1, n - k - 1, m - k - 1, leaf, right)) return false;
  return merge(left, right, d[k], e[k], out);
}

// --- entry -----------------------------------------------------------------

// SVD of the n x n upper bidiagonal (d, e): on return U and V are n x n
// column-major real orthogonal and s holds the n singular values in
// descending order, with B = U diag(s) V^T. `leaf` is the largest order
// solved densely; the bench measures the choice, and the kernel passes its
// constant.
//
// An e[i] that is exactly zero decouples the matrix into independent
// blocks, each solved on its own; their factors are placed block-diagonally
// and the values sorted together. No zero superdiagonal therefore ever
// reaches the recursion, whose deflation would handle it in any case, and
// a block-diagonal input costs no merge across the blocks.
static inline bool bidiag_svd(const double* d, const double* e, Index n, Index leaf,
                       std::vector<double>& U, std::vector<double>& s,
                       std::vector<double>& V) {
  if (leaf < 2) leaf = 2;
  std::vector<double> u_all(n * n, 0.0);
  std::vector<double> v_all(n * n, 0.0);
  std::vector<double> s_all(n, 0.0);

  Index start = 0;
  while (start < n) {
    Index stop = start;  // the block is rows [start, stop]
    while (stop + 1 < n && e[stop] != 0.0) ++stop;
    const Index len = stop - start + 1;
    Factors f;
    if (!solve(d + start, e + start, len, len, leaf, f)) return false;
    for (Index j = 0; j < len; ++j) {
      s_all[start + j] = f.s[j];
      for (Index i = 0; i < len; ++i) {
        u_all[at(start + i, start + j, n)] = f.u[at(i, j, len)];
        v_all[at(start + i, start + j, n)] = f.v[at(i, j, len)];
      }
    }
    start = stop + 1;
  }

  std::vector<Index> order;
  kernel::sort_indices(s_all.data(), n, true, order);
  U.assign(n * n, 0.0);
  V.assign(n * n, 0.0);
  s.assign(n, 0.0);
  for (Index t = 0; t < n; ++t) {
    const Index src = order[t];
    s[t] = s_all[src];
    for (Index i = 0; i < n; ++i) {
      U[at(i, t, n)] = u_all[at(i, src, n)];
      V[at(i, t, n)] = v_all[at(i, src, n)];
    }
  }
  return true;
}

}  // namespace bidiag_dc
}  // namespace detail
}  // namespace autonne

#endif  // AUTONNE_SRC_DETAIL_BIDIAG_DC_HPP
