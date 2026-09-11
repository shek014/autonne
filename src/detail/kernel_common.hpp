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

// Complex products written out on the parts. std::complex's operator* is
// specified with the C99 Annex G recovery for infinite operands, and under
// strict floating point GCC and Clang implement it as a call to __muldc3 on
// every multiplication -- an order of magnitude slower than four
// multiplications and two additions, and pointless here, where every operand
// is finite by construction.
static inline Complex mul(const Complex& a, const Complex& b) noexcept {
  return Complex(a.real() * b.real() - a.imag() * b.imag(),
                 a.real() * b.imag() + a.imag() * b.real());
}

// conj(a) * b.
static inline Complex conj_mul(const Complex& a, const Complex& b) noexcept {
  return Complex(a.real() * b.real() + a.imag() * b.imag(),
                 a.real() * b.imag() - a.imag() * b.real());
}

static inline Complex scale(double c, const Complex& z) noexcept {
  return Complex(c * z.real(), c * z.imag());
}

// Unit roundoff, 2^-53. The Jacobi convergence thresholds are multiples of
// this rather than of epsilon (2^-52), following LAPACK's xGESVJ.
static inline constexpr double unit_roundoff() noexcept {
  return 1.1102230246251565404e-16;
}

// Columns whose norm falls below this are treated as exactly zero. It is the
// one place the kernel decides a value is zero rather than computing it, so
// it sits as low as the arithmetic allows rather than wherever is
// comfortable.
//
// The limit is the squaring. A column norm is formed by summing |x_i|^2, so a
// norm below about 2^-511 has a square that is no longer a normal double and
// the sum stops meaning anything. 2^-500 is the nearest round exponent above
// that, and it leaves the square (2^-1000) comfortably normal. The matrix has
// been scaled so its largest component is in [0.5, 1), so this is a relative
// floor: a singular value 150 decades below the largest is still returned.
//
// An earlier value of 2^-400 was chosen for safety and cost accuracy: it
// flushed the second singular value of diag(1, 2^-400) to zero although
// 3.87e-121 is an ordinary normal double, and made the two kernels disagree
// on that matrix, since eigh returned it correctly.
// One consequence of sitting this low: the smallest rotation threshold the
// one-sided sweep can form, tol * sqrt(alpha) * sqrt(beta) with both norms at
// the floor, is itself subnormal. Under a flush-to-zero rounding mode it
// becomes zero and the threshold degenerates to "rotate whenever the inner
// product is nonzero", which costs extra rotations and no accuracy -- and
// exhausting the sweep budget is no longer fatal, since the sweep ends with
// an acceptance test. It also means the sweep can now meet a subnormal
// inner product, which the old floor made impossible; that is safe only
// because unit_phase handles one, so these two choices are coupled.
static inline constexpr double column_floor() noexcept {
  return 3.0549363634996046820e-151;  // 2^-500
}

// |z|, computed without forming |re|^2 + |im|^2 and without leaving the
// normal range. std::abs(std::complex) is implementation-defined in this
// respect and may be rewritten under -ffast-math.
//
// The larger component is scaled to [0.5, 1) before the square root and the
// result scaled back, both by powers of two, both exact. Without that step
// the closing multiply `a * sqrt(1 + r*r)` is rounded onto the subnormal grid
// whenever `a` is subnormal: modulus(2^-1074 + 2^-1074 i) came back as
// 2^-1074 rather than 1.414 * 2^-1074, twenty-nine percent low. Callers
// divide by this to build a unit phase, so an error there is an error in a
// factor that is supposed to be unitary.
static inline double modulus(const Complex& z) noexcept {
  double a = std::fabs(z.real());
  double b = std::fabs(z.imag());
  if (a < b) {
    const double t = a;
    a = b;
    b = t;
  }
  if (a == 0.0) return 0.0;
  int e = 0;
  (void)std::frexp(a, &e);
  const double as = std::ldexp(a, -e);  // in [0.5, 1)
  const double bs = std::ldexp(b, -e);  // may underflow to zero; harmless
  const double r = bs / as;
  return std::ldexp(as * std::sqrt(1.0 + r * r), e);
}

// z / |z| for a nonzero z, unimodular to rounding at any magnitude, and
// (1, 0) for a zero z.
//
// Dividing the components by modulus(z) directly is not enough when z is
// subnormal: both the numerator and the denominator then carry only the few
// bits the subnormal grid offers, so the quotient is far from unit modulus.
// The components are lifted into the normal range first, which costs two
// exact scalings and makes the result good to an ulp everywhere.
static inline Complex unit_phase(const Complex& z) noexcept {
  double a = std::fabs(z.real());
  const double b = std::fabs(z.imag());
  if (a < b) a = b;
  if (a == 0.0) return Complex(1.0, 0.0);
  int e = 0;
  (void)std::frexp(a, &e);
  const double re = std::ldexp(z.real(), -e);
  const double im = std::ldexp(z.imag(), -e);
  const double m = modulus(Complex(re, im));
  return Complex(re / m, im / m);
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
  r.phase = std::conj(unit_phase(gamma));
  // zeta = (beta - alpha) / (2 |gamma|), but never formed as that quotient.
  //
  // Two ways it goes wrong. The denominator underflows to zero for a
  // subnormal |gamma| -- guaranteed under a flush-to-zero rounding mode,
  // which is what MSVC's /fp:fast selects -- and the quotient is then 0/0, a
  // NaN that propagates into c and s and out through every entry the rotation
  // touches. And when the diagonal is far apart zeta is enormous, so forming
  // it and then its reciprocal throws away the precision the reciprocal
  // needed.
  //
  // Both are avoided by dividing the smaller of the two quantities by the
  // larger, which is always representable, and by taking the equal case
  // explicitly.
  const double diff = beta - alpha;
  const double sign = diff < 0.0 ? -1.0 : 1.0;
  const double adiff = std::fabs(diff);
  const double two_gamma = r.gamma_abs + r.gamma_abs;
  if (adiff == 0.0) {
    r.t = 1.0;  // zeta = 0
  } else if (adiff <= two_gamma) {
    const double zeta_abs = adiff / two_gamma;
    r.t = sign / (zeta_abs + std::sqrt(1.0 + zeta_abs * zeta_abs));
  } else {
    const double inv = two_gamma / adiff;  // = 1 / |zeta|, in [0, 1)
    r.t = sign * inv / (1.0 + std::sqrt(1.0 + inv * inv));
  }
  r.c = 1.0 / std::sqrt(1.0 + r.t * r.t);
  r.s = r.c * r.t;
  return r;
}

// Applies the rotation to columns p and q (length n, contiguous).
static inline void rotate_columns(Complex* xp, Complex* xq, Index n,
                                  const Rotation& r) noexcept {
  const double c = r.c;
  const double s_ = r.s;
  const double pr = r.phase.real();
  const double pi = r.phase.imag();
  for (Index i = 0; i < n; ++i) {
    const double ar = xp[i].real();
    const double ai = xp[i].imag();
    const double qr = xq[i].real();
    const double qi = xq[i].imag();
    const double br = pr * qr - pi * qi;
    const double bi = pr * qi + pi * qr;
    xp[i] = Complex(c * ar - s_ * br, c * ai - s_ * bi);
    xq[i] = Complex(s_ * ar + c * br, s_ * ai + c * bi);
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
        for (Index i = 0; i < n; ++i) dot += conj_mul(x[at(i, c, n)], work[i]);
        for (Index i = 0; i < n; ++i) work[i] -= mul(dot, x[at(i, c, n)]);
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
