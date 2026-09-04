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

// Arithmetic shared by the two kernels. Private to src/: nothing here is
// declared in a public header.
//
// Every function is `static` on purpose. The kernels are compiled under the
// flags of their own translation unit, and a helper with vague linkage would
// be merged across translation units by the linker without regard to those
// flags. Internal linkage gives each kernel its own copy, built the way that
// kernel was built.
//
// Two rules apply throughout src/, both consequences of the fast-math test
// variant, in which this code must be as correct as under strict arithmetic:
//
//   * No NaN or infinity is ever produced. Inputs are scanned (by bit pattern,
//     see detail/fp_bits.hpp) and rejected if non-finite; the matrix is then
//     scaled by a power of two so that every entry has modulus at most sqrt 2;
//     every divisor is either bounded away from zero by construction or
//     guarded by a floor. A compiler that assumes finiteness is therefore
//     assuming something true.
//
//   * No result depends on the exact order of a summation. The iterations are
//     self-correcting: a Jacobi rotation orthogonalises the pair it is applied
//     to whatever rounding preceded it, and convergence is judged by a test on
//     the current state, not on an accumulated quantity. Reassociation changes
//     the path, not the destination.

#ifndef AUTONNE_SRC_DETAIL_KERNEL_COMMON_HPP
#define AUTONNE_SRC_DETAIL_KERNEL_COMMON_HPP

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace autonne {
namespace detail {
namespace kernel {

using Complex = std::complex<double>;
using Index = std::size_t;

// Column-major offset of element (i, j) with leading dimension ld.
static inline constexpr Index at(Index i, Index j, Index ld) noexcept {
  return i + j * ld;
}

// Unit roundoff, 2^-53. The Jacobi convergence thresholds are multiples of
// this rather than of epsilon (2^-52), following LAPACK's xGESVJ.
static inline constexpr double unit_roundoff() noexcept {
  return 1.1102230246251565404e-16;
}

// Columns whose norm falls below this are treated as exactly zero. It is far
// below anything a singular value can meaningfully be (2^-400 is about
// 4e-121, and the matrix has been scaled to unit size), and far above the
// range where the products of two such norms would underflow (2^-800 is
// still a normal double). It is the one place the kernel decides a value is
// zero rather than computing it.
static inline constexpr double column_floor() noexcept {
  return 3.8725919148493182986e-121;  // 2^-400
}

// |z| without forming |re|^2 + |im|^2, so a value whose components are far
// below sqrt(DBL_MIN) still comes out right. std::abs(std::complex) is
// implementation-defined in this respect and may be rewritten under
// -ffast-math.
static inline double modulus(const Complex& z) noexcept {
  double a = std::fabs(z.real());
  double b = std::fabs(z.imag());
  if (a < b) {
    const double t = a;
    a = b;
    b = t;
  }
  if (a == 0.0) return 0.0;
  const double r = b / a;
  return a * std::sqrt(1.0 + r * r);
}

// Sum of |x_i|^2 over a column, and its square root. The entries are at most
// sqrt 2 in modulus after scaling, so nothing overflows; entries so small
// that their squares underflow contribute nothing measurable to a norm that
// is itself above the floor.
static inline double norm_sq(const Complex* x, Index n) noexcept {
  double acc = 0.0;
  for (Index i = 0; i < n; ++i) {
    acc += x[i].real() * x[i].real() + x[i].imag() * x[i].imag();
  }
  return acc;
}

// Largest |re| or |im| over a buffer. Zero for an empty buffer.
static inline double max_component(const Complex* x, Index n) noexcept {
  double m = 0.0;
  for (Index i = 0; i < n; ++i) {
    const double a = std::fabs(x[i].real());
    const double b = std::fabs(x[i].imag());
    if (a > m) m = a;
    if (b > m) m = b;
  }
  return m;
}

// Exponent e such that value * 2^-e lies in [0.5, 1). Scaling by a power of
// two is exact, so the factorisation of the scaled matrix is the
// factorisation of the original with the singular values scaled back.
static inline int scaling_exponent(double value) noexcept {
  int e = 0;
  (void)std::frexp(value, &e);
  return e;
}

static inline void scale_in_place(Complex* x, Index n, int exponent) noexcept {
  for (Index i = 0; i < n; ++i) {
    x[i] = Complex(std::ldexp(x[i].real(), exponent), std::ldexp(x[i].imag(), exponent));
  }
}

// The unitary 2 x 2 rotation that diagonalises the Hermitian matrix
//
//   [ alpha    gamma ]
//   [ conj(g)  beta  ]
//
// with alpha, beta real and gamma = |gamma| e^{i phi}. It is J = D R with
// D = diag(1, e^{-i phi}) and R = [[c, s], [-s, c]] real, chosen so that
// t = s / c is the smaller root of t^2 + 2 zeta t - 1 = 0 for
// zeta = (beta - alpha) / (2 |gamma|). Applied to a pair of columns as
//
//   x_p' = c x_p - s e^{-i phi} x_q
//   x_q' = s x_p + c e^{-i phi} x_q
//
// it makes them orthogonal (one-sided Jacobi), and as a similarity J^* H J it
// zeroes the (p, q) entry while moving the diagonal to alpha - t |gamma| and
// beta + t |gamma| (two-sided Jacobi).
//
// Requires |gamma| > 0. zeta may be enormous when |gamma| is tiny, so the
// root is evaluated in a form that never squares it.
struct Rotation {
  double c = 1.0;
  double s = 0.0;
  double t = 0.0;
  Complex phase = Complex(1.0, 0.0);  // e^{-i phi}
  double gamma_abs = 0.0;
};

static inline Rotation make_rotation(double alpha, double beta,
                                     const Complex& gamma) noexcept {
  Rotation r;
  r.gamma_abs = modulus(gamma);
  r.phase = Complex(gamma.real() / r.gamma_abs, -gamma.imag() / r.gamma_abs);
  const double zeta = (beta - alpha) / (2.0 * r.gamma_abs);
  const double sign = zeta < 0.0 ? -1.0 : 1.0;
  const double zeta_abs = std::fabs(zeta);
  if (zeta_abs <= 1.0) {
    r.t = sign / (zeta_abs + std::sqrt(1.0 + zeta_abs * zeta_abs));
  } else {
    const double inv = 1.0 / zeta_abs;
    r.t = sign * inv / (1.0 + std::sqrt(1.0 + inv * inv));
  }
  r.c = 1.0 / std::sqrt(1.0 + r.t * r.t);
  r.s = r.c * r.t;
  return r;
}

// Applies the rotation to columns p and q (length n, contiguous).
static inline void rotate_columns(Complex* xp, Complex* xq, Index n,
                                  const Rotation& r) noexcept {
  for (Index i = 0; i < n; ++i) {
    const Complex a = xp[i];
    const Complex b = r.phase * xq[i];
    xp[i] = r.c * a - r.s * b;
    xq[i] = r.s * a + r.c * b;
  }
}

// Fills columns [have, want) of the n x want column-major matrix X with
// vectors orthonormal to the existing columns [0, have), which are assumed
// orthonormal. Each new column is the canonical basis vector that is
// furthest from the current span, orthogonalised twice (classical
// Gram-Schmidt, twice, is enough) and normalised. A candidate at least
// sqrt((n - have) / n) away always exists, so the normalisation is safe.
static inline void complete_orthonormal(Complex* x, Index n, Index have,
                                        Index want, std::vector<Complex>& work) {
  work.resize(n);
  for (Index col = have; col < want; ++col) {
    // Pick the canonical vector with the largest residual against the span.
    Index best = 0;
    double best_residual = -1.0;
    for (Index cand = 0; cand < n; ++cand) {
      double projected = 0.0;
      for (Index c = 0; c < col; ++c) {
        const Complex& e = x[at(cand, c, n)];
        projected += e.real() * e.real() + e.imag() * e.imag();
      }
      const double residual = 1.0 - projected;
      if (residual > best_residual) {
        best_residual = residual;
        best = cand;
      }
    }
    for (Index i = 0; i < n; ++i) work[i] = Complex(0.0, 0.0);
    work[best] = Complex(1.0, 0.0);
    for (int pass = 0; pass < 2; ++pass) {
      for (Index c = 0; c < col; ++c) {
        Complex dot(0.0, 0.0);
        for (Index i = 0; i < n; ++i) dot += std::conj(x[at(i, c, n)]) * work[i];
        for (Index i = 0; i < n; ++i) work[i] -= dot * x[at(i, c, n)];
      }
    }
    const double inv = 1.0 / std::sqrt(norm_sq(work.data(), n));
    for (Index i = 0; i < n; ++i) x[at(i, col, n)] = work[i] * inv;
  }
}

// Stable ordering of indices by key: descending if `descending`, else
// ascending. Insertion sort: n is small and stability is what matters, so
// that tied values (a degenerate spectrum) keep the order they came in.
static inline void sort_indices(const double* key, Index n, bool descending,
                                std::vector<Index>& order) {
  order.resize(n);
  for (Index i = 0; i < n; ++i) order[i] = i;
  for (Index i = 1; i < n; ++i) {
    const Index moving = order[i];
    Index j = i;
    while (j > 0) {
      const bool out_of_order =
          descending ? key[order[j - 1]] < key[moving] : key[order[j - 1]] > key[moving];
      if (!out_of_order) break;
      order[j] = order[j - 1];
      --j;
    }
    order[j] = moving;
  }
}

// Copies column `order[j]` of src (n rows, column-major) to column j of dst.
static inline void permute_columns(const Complex* src, Complex* dst, Index n,
                                   const std::vector<Index>& order) noexcept {
  for (Index j = 0; j < order.size(); ++j) {
    for (Index i = 0; i < n; ++i) dst[at(i, j, n)] = src[at(i, order[j], n)];
  }
}

}  // namespace kernel
}  // namespace detail
}  // namespace autonne

#endif  // AUTONNE_SRC_DETAIL_KERNEL_COMMON_HPP
