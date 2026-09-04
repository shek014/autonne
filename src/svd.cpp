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

// One-sided Jacobi SVD.
//
// The method orthogonalises the COLUMNS of the input by a sequence of plane
// rotations applied on the right. When no pair of columns is left to rotate,
// the working matrix is U * diag(S) and the accumulated rotations are V, so
//
//   A = U diag(S) V^*
//
// falls out by reading column norms and normalising.
//
// Chosen over divide-and-conquer for three properties this interface needs:
//
//   1. It forms the WHOLE spectrum. A caller sizing a truncation against a
//      fraction of total weight has to sum the discarded tail from individual
//      small values, because subtracting the kept weight from the total is
//      catastrophic cancellation on a normalised input: both sides sit near
//      1.0 while the real difference can be near 1e-30. Any method that never
//      forms the tail cannot serve that caller.
//
//   2. High relative accuracy on graded matrices (Demmel and Veselic). The
//      accuracy is governed by the condition number of the COLUMN-SCALED
//      matrix rather than of the matrix itself, so a spectrum spanning many
//      orders of magnitude is not intrinsically hostile. That is the regime
//      where general-purpose implementations have been observed to fail.
//
//   3. Column pairs within a sweep touch disjoint data, so the inner work
//      parallelises and vectorises without changing the arithmetic.
//
// The cost is O(rows * cols^2) per sweep at typically five to ten sweeps,
// which loses to divide-and-conquer as the matrix grows and is the right
// trade at the sizes this library targets.
//
// Every guard here reads an object representation through fp_bits.hpp rather
// than calling a floating-point predicate, so the guards survive a
// translation unit compiled with -ffinite-math-only.

#include "autonne/autonne.hpp"

#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <vector>

#include "autonne/detail/fp_bits.hpp"
#include "autonne/detail/matrix_view.hpp"

namespace autonne {
namespace {

using Complex = std::complex<double>;

// Cyclic Jacobi converges quadratically once the off-diagonal mass is small,
// and five to ten sweeps is typical. This is a runaway guard rather than a
// working limit: reaching it means the rotation sequence is not settling, and
// the honest answer to the caller is failure rather than a partial result.
constexpr int kMaxSweeps = 60;

// A pair is left alone once its inner product is negligible against the
// geometric mean of the two column norms. The comparison is relative for the
// reason the method is being used at all: an absolute threshold would call a
// pair converged purely because both columns are small.
//
// Machine epsilon is the tolerance rather than a chosen number, because the
// quantity being tested is a rounding residue: two columns are as orthogonal
// as double precision can represent once their normalised inner product falls
// to this scale, and demanding more would spin without converging.
constexpr double kOrthogonalityTolerance = std::numeric_limits<double>::epsilon();

// Squared Euclidean norm of column j of an m x n column-major buffer.
//
// Accumulated one element at a time in index order rather than through a
// tree reduction. The convergence test below compares against this value, and
// a convergence criterion that moves with how the target chose to partition a
// sum is not a criterion.
double column_norm_sq(const Complex* w, int m, int j) noexcept {
  const Complex* col = w + static_cast<std::size_t>(j) * static_cast<std::size_t>(m);
  double acc = 0.0;
  for (int i = 0; i < m; ++i) {
    acc += col[i].real() * col[i].real() + col[i].imag() * col[i].imag();
  }
  return acc;
}

// col_p^* col_q, the off-diagonal entry of the 2x2 Gram matrix for this pair.
Complex column_inner(const Complex* w, int m, int p, int q) noexcept {
  const Complex* cp = w + static_cast<std::size_t>(p) * static_cast<std::size_t>(m);
  const Complex* cq = w + static_cast<std::size_t>(q) * static_cast<std::size_t>(m);
  Complex acc(0.0, 0.0);
  for (int i = 0; i < m; ++i) acc += std::conj(cp[i]) * cq[i];
  return acc;
}

// Applies the 2x2 right-multiplication
//
//   col_p <- c * col_p - s * ph * col_q
//   col_q <- s * col_p + c * ph * col_q
//
// to columns p and q of an rows x * column-major buffer. Both columns are read
// before either is written, so the update is simultaneous rather than
// sequential.
void rotate_columns(Complex* w, int m, int p, int q, double c, double s,
                    Complex ph) noexcept {
  Complex* cp = w + static_cast<std::size_t>(p) * static_cast<std::size_t>(m);
  Complex* cq = w + static_cast<std::size_t>(q) * static_cast<std::size_t>(m);
  for (int i = 0; i < m; ++i) {
    const Complex wp = cp[i];
    const Complex wq = cq[i];
    cp[i] = c * wp - s * ph * wq;
    cq[i] = s * wp + c * ph * wq;
  }
}

// One cyclic sweep over every column pair. Returns the number of rotations
// actually applied, so the caller can stop when a sweep changes nothing.
int jacobi_sweep(Complex* w, Complex* v, int m, int n) noexcept {
  int rotations = 0;

  for (int p = 0; p < n - 1; ++p) {
    for (int q = p + 1; q < n; ++q) {
      const double app = column_norm_sq(w, m, p);
      const double aqq = column_norm_sq(w, m, q);

      // A zero column has no direction to rotate into, and the phase below
      // would divide by zero. Leaving it for the null-space completion is the
      // correct handling, not a shortcut.
      if (!(app > 0.0) || !(aqq > 0.0)) continue;
      if (detail::fp_bad(app) || detail::fp_bad(aqq)) continue;

      const Complex apq = column_inner(w, m, p, q);
      if (detail::fp_bad(apq)) continue;

      const double abs_apq = std::abs(apq);
      if (detail::fp_bad(abs_apq)) continue;

      // sqrt is taken of each norm separately rather than of their product,
      // which would overflow or underflow on the graded inputs this method
      // exists to handle.
      if (abs_apq <= kOrthogonalityTolerance * std::sqrt(app) * std::sqrt(aqq)) {
        continue;
      }

      // Reduce the Hermitian 2x2 to a real symmetric one by absorbing the
      // phase of the off-diagonal into column q. With apq = |apq| e^{i phi},
      // right-multiplying by diag(1, e^{-i phi}) makes both off-diagonal
      // entries equal to |apq|.
      const Complex ph = std::conj(apq) / abs_apq;
      if (detail::fp_bad(ph)) continue;

      // Real symmetric Jacobi on [[app, |apq|], [|apq|, aqq]]. The root of
      // smaller magnitude is taken, in the form that avoids cancellation when
      // theta is large.
      const double theta = (aqq - app) / (2.0 * abs_apq);
      const double sign = (theta >= 0.0) ? 1.0 : -1.0;
      const double t = sign / (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
      const double c = 1.0 / std::sqrt(t * t + 1.0);
      const double s = c * t;

      if (detail::fp_bad(c) || detail::fp_bad(s)) continue;

      rotate_columns(w, m, p, q, c, s, ph);
      rotate_columns(v, n, p, q, c, s, ph);
      ++rotations;
    }
  }

  return rotations;
}

// Writes a unit vector into column j of the m x k column-major buffer u,
// orthogonal to columns 0 through j-1. Returns false when no such vector was
// found, which cannot happen while j < m.
//
// This exists because a thin SVD of a rank-deficient matrix still owes the
// caller k orthonormal columns of U, and the columns past the rank have no
// direction to inherit: their singular value is zero, so normalising the
// working column is a division of zero by zero. Producing a NaN there is the
// exact failure this library is written to avoid, and it is invisible to a
// caller who only inspects the leading block.
//
// EVERY standard basis vector is tried and the one retaining the most length
// is taken, rather than the first to clear a fixed bar. The difference is not
// cosmetic. Summing over the basis gives
//
//   sum_i ||P_span e_i||^2 = j
//
// so as many as 2j basis vectors can each leave less than half their length,
// and a first-past-the-post scan with a one-half threshold can exhaust its
// candidates on a matrix it should handle: a square input at rank j = m/2 is
// already inside that budget. Taking the maximum cannot fail while j < m,
// because the same identity puts the largest residual at no less than
// (m - j) / m.
//
// `scratch` is m elements of caller-owned working space, so this allocates
// nothing.
bool complete_orthonormal_column(Complex* u, int m, int j, Complex* scratch) noexcept {
  Complex* target = u + static_cast<std::size_t>(j) * static_cast<std::size_t>(m);

  // Modified Gram-Schmidt, run twice. One pass loses orthogonality when the
  // candidate lies close to the span; the second restores it, and two passes
  // are known to be enough.
  const auto orthogonalise = [&](Complex* vec) noexcept {
    for (int pass = 0; pass < 2; ++pass) {
      for (int prior = 0; prior < j; ++prior) {
        const Complex* pcol =
            u + static_cast<std::size_t>(prior) * static_cast<std::size_t>(m);
        Complex proj(0.0, 0.0);
        for (int i = 0; i < m; ++i) proj += std::conj(pcol[i]) * vec[i];
        for (int i = 0; i < m; ++i) vec[i] -= proj * pcol[i];
      }
    }
    double norm_sq = 0.0;
    for (int i = 0; i < m; ++i) {
      norm_sq += vec[i].real() * vec[i].real() + vec[i].imag() * vec[i].imag();
    }
    return norm_sq;
  };

  int best = -1;
  double best_norm_sq = 0.0;
  for (int candidate = 0; candidate < m; ++candidate) {
    for (int i = 0; i < m; ++i) scratch[i] = Complex(0.0, 0.0);
    scratch[candidate] = Complex(1.0, 0.0);

    const double norm_sq = orthogonalise(scratch);
    if (!detail::fp_bad(norm_sq) && norm_sq > best_norm_sq) {
      best_norm_sq = norm_sq;
      best = candidate;
    }
  }

  if (best < 0 || !(best_norm_sq > 0.0)) return false;

  for (int i = 0; i < m; ++i) target[i] = Complex(0.0, 0.0);
  target[best] = Complex(1.0, 0.0);
  const double norm_sq = orthogonalise(target);
  if (detail::fp_bad(norm_sq) || !(norm_sq > 0.0)) return false;

  const double inv = 1.0 / std::sqrt(norm_sq);
  if (detail::fp_bad(inv)) return false;
  for (int i = 0; i < m; ++i) target[i] *= inv;
  return !detail::any_bad(target, static_cast<std::size_t>(m));
}

}  // namespace

bool svd_thin(const std::complex<double>* data, int rows, int cols,
              MatrixOrder order, std::complex<double>* U_out, double* S_out,
              std::complex<double>* V_out) {
  if (data == nullptr || U_out == nullptr || S_out == nullptr ||
      V_out == nullptr) {
    return false;
  }
  if (rows <= 0 || cols <= 0) return false;

  const int k = (rows < cols) ? rows : cols;

  // One-sided Jacobi orthogonalises columns, so it needs at least as many rows
  // as columns to have room for them. A wide matrix is handled by factoring
  // its adjoint and exchanging the two factors: from A^* = U' S V'^* follows
  // A = V' S U'^*, so U is V' and V is U'.
  const bool transposed = rows < cols;
  const int m = transposed ? cols : rows;
  const int n = transposed ? rows : cols;

  const auto in = detail::ordered(data, rows, cols, order);

  // Reject a non-finite input up front. Rotating on a NaN spreads it across
  // every column it touches, and the result would be a factorisation of
  // something the caller never passed.
  for (int j = 0; j < cols; ++j) {
    for (int i = 0; i < rows; ++i) {
      if (detail::fp_bad(detail::at(in, i, j))) return false;
    }
  }

  const std::size_t work_size =
      static_cast<std::size_t>(m) * static_cast<std::size_t>(n);
  std::vector<Complex> w(work_size);
  std::vector<Complex> v(static_cast<std::size_t>(n) * static_cast<std::size_t>(n),
                         Complex(0.0, 0.0));

  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < m; ++i) {
      w[static_cast<std::size_t>(j) * static_cast<std::size_t>(m) +
        static_cast<std::size_t>(i)] =
          transposed ? std::conj(detail::at(in, j, i)) : detail::at(in, i, j);
    }
    v[static_cast<std::size_t>(j) * static_cast<std::size_t>(n) +
      static_cast<std::size_t>(j)] = Complex(1.0, 0.0);
  }

  int sweep = 0;
  for (; sweep < kMaxSweeps; ++sweep) {
    if (jacobi_sweep(w.data(), v.data(), m, n) == 0) break;
  }
  // A sequence that never settles is reported as failure. Returning the
  // partially rotated factors would hand the caller something that passes a
  // shape check and fails a residual one.
  if (sweep == kMaxSweeps) return false;

  // Column norms are the singular values. Ordering is by descending norm,
  // established here by selection so that U, S and V are permuted together.
  std::vector<double> sigma(static_cast<std::size_t>(n));
  std::vector<int> perm(static_cast<std::size_t>(n));
  for (int j = 0; j < n; ++j) {
    const double norm_sq = column_norm_sq(w.data(), m, j);
    sigma[static_cast<std::size_t>(j)] =
        detail::fp_bad(norm_sq) ? 0.0 : std::sqrt(norm_sq);
    perm[static_cast<std::size_t>(j)] = j;
  }

  for (int a = 0; a < n - 1; ++a) {
    int best = a;
    for (int b = a + 1; b < n; ++b) {
      if (sigma[static_cast<std::size_t>(perm[static_cast<std::size_t>(b)])] >
          sigma[static_cast<std::size_t>(perm[static_cast<std::size_t>(best)])]) {
        best = b;
      }
    }
    const int tmp = perm[static_cast<std::size_t>(a)];
    perm[static_cast<std::size_t>(a)] = perm[static_cast<std::size_t>(best)];
    perm[static_cast<std::size_t>(best)] = tmp;
  }

  // U and V are built into their own buffers before anything is handed back,
  // because the wide case exchanges them and writing in place would alias.
  std::vector<Complex> u_built(static_cast<std::size_t>(m) *
                               static_cast<std::size_t>(n));
  std::vector<Complex> v_built(static_cast<std::size_t>(n) *
                               static_cast<std::size_t>(n));

  std::vector<Complex> scratch(static_cast<std::size_t>(m));
  for (int j = 0; j < n; ++j) {
    const int src = perm[static_cast<std::size_t>(j)];
    const double s = sigma[static_cast<std::size_t>(src)];

    const Complex* wcol =
        w.data() + static_cast<std::size_t>(src) * static_cast<std::size_t>(m);
    Complex* ucol =
        u_built.data() + static_cast<std::size_t>(j) * static_cast<std::size_t>(m);

    bool normalised = false;
    if (s > 0.0 && !detail::fp_bad(s)) {
      const double inv = 1.0 / s;
      normalised = !detail::fp_bad(inv);
      if (normalised) {
        for (int i = 0; i < m; ++i) ucol[i] = wcol[i] * inv;
        // A column that normalised into a non-finite value is rebuilt rather
        // than shipped. The check is over the column just written, so it costs
        // one pass and closes the case where the division underflowed.
        if (detail::any_bad(ucol, static_cast<std::size_t>(m))) normalised = false;
      }
    }

    if (!normalised) {
      // Rank deficiency, reached either by a genuinely zero column or by a
      // normalisation that could not be represented. Either way this direction
      // carries no weight, so the singular value is zero and the column is a
      // fresh orthonormal direction.
      sigma[static_cast<std::size_t>(src)] = 0.0;
      if (!complete_orthonormal_column(u_built.data(), m, j, scratch.data())) {
        return false;
      }
    }

    Complex* vdst =
        v_built.data() + static_cast<std::size_t>(j) * static_cast<std::size_t>(n);
    const Complex* vsrc =
        v.data() + static_cast<std::size_t>(src) * static_cast<std::size_t>(n);
    for (int i = 0; i < n; ++i) vdst[i] = vsrc[i];
  }

  // In the wide case the factors computed above belong to A^*, so they are
  // exchanged on the way out: U is what the working V holds and vice versa.
  const Complex* u_final = transposed ? v_built.data() : u_built.data();
  const Complex* v_final = transposed ? u_built.data() : v_built.data();

  const std::size_t u_count =
      static_cast<std::size_t>(rows) * static_cast<std::size_t>(k);
  const std::size_t v_count =
      static_cast<std::size_t>(cols) * static_cast<std::size_t>(k);

  // Nothing non-finite leaves this function. The caller's contract says the
  // outputs are unspecified on false, so a late rejection costs a scan and
  // never a wrong answer.
  if (detail::any_bad(u_final, u_count) || detail::any_bad(v_final, v_count)) {
    return false;
  }
  // The spectrum is ordered by the sort above, but a column that failed to
  // normalise has its singular value set to zero AFTER that sort, at whatever
  // position it already held. Only a denormal value can reach that branch with
  // a position to lose, and a denormal already sorts last, so the ordering
  // survives in every case reachable today. It is checked rather than argued:
  // the contract promises a descending spectrum, and shipping a factorisation
  // that breaks it would be a silent failure in a caller that trusts the order.
  std::vector<double> s_final(static_cast<std::size_t>(k));
  for (int j = 0; j < k; ++j) {
    const double s = sigma[static_cast<std::size_t>(perm[static_cast<std::size_t>(j)])];
    if (detail::fp_bad(s) || s < 0.0) return false;
    if (j > 0 && s > s_final[static_cast<std::size_t>(j - 1)]) return false;
    s_final[static_cast<std::size_t>(j)] = s;
  }

  for (std::size_t i = 0; i < u_count; ++i) U_out[i] = u_final[i];
  for (std::size_t i = 0; i < v_count; ++i) V_out[i] = v_final[i];
  for (int j = 0; j < k; ++j) S_out[j] = s_final[static_cast<std::size_t>(j)];

  return true;
}

}  // namespace autonne
