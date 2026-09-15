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
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "autonne/detail/fp_bits.hpp"
#include "autonne/detail/matrix_view.hpp"

namespace autonne {
namespace verify {

// Reads the macro in the translation unit whose floating-point model is the
// one in question. See the declaration in verify.hpp.
bool built_with_fast_math() noexcept {
#if defined(__FAST_MATH__)
  return true;
#else
  return false;
#endif
}

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

// x * 2^-e and z * 2^-e, exactly, for any e.
//
// Forming 2^-e as a double and multiplying by it is wrong once e passes 1023:
// the factor overflows to infinity and every product becomes a NaN, which is
// reachable for a matrix whose largest component is below 2^-1024. ldexp has
// no such limit, and scaling by a power of two is exact either way.
// Complex products written out on the parts.
//
// std::complex's operator* carries the C99 Annex G recovery for infinite
// operands, and under strict floating point -- which this file is compiled
// with in every variant -- GCC and Clang implement it as a call to __muldc3 on
// every multiplication. The recovery cannot be needed here: the bit-pattern
// scan establishes that every operand is finite before any of these loops run,
// and a non-finite input has already failed its verdict.
//
// These compute the same four products and two sums, in the same order, as the
// operator they replace, so no measured quantity moves by a single bit. It is
// the call that goes, not the arithmetic. src/detail/kernel_common.hpp does the
// same for the kernels and explains it at more length; these are separate
// because that header is private to src/ and verify.cpp is not a kernel.
std::complex<double> cmul(const std::complex<double>& a,
                          const std::complex<double>& b) noexcept {
  return std::complex<double>(a.real() * b.real() - a.imag() * b.imag(),
                              a.real() * b.imag() + a.imag() * b.real());
}

// conj(a) * b. Equal bit for bit to b * conj(a): complex multiplication is
// commutative in the products it forms as well as in value.
std::complex<double> conj_mul(const std::complex<double>& a,
                              const std::complex<double>& b) noexcept {
  return std::complex<double>(a.real() * b.real() + a.imag() * b.imag(),
                              a.real() * b.imag() - a.imag() * b.real());
}

double scaled(const double& x, int e) noexcept { return std::ldexp(x, -e); }

std::complex<double> scaled(const std::complex<double>& z, int e) noexcept {
  return std::complex<double>(std::ldexp(z.real(), -e), std::ldexp(z.imag(), -e));
}

// Sum of |2^-e a(i,j)|^2. Squared norms are accumulated rather than
// square-rooted per element so that the energy identities below compare like
// with like, and every caller passes an exponent for the reason set out at
// scale_exponent below.
template <typename View>
double frobenius_sq(const View& a, int e) noexcept {
  double acc = 0.0;
  for (int j = 0; j < cols_of(a); ++j) {
    for (int i = 0; i < rows_of(a); ++i) {
      const std::complex<double> z = scaled(at(a, i, j), e);
      acc += z.real() * z.real() + z.imag() * z.imag();
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
      for (int i = 0; i < n; ++i) g += conj_mul(at(x, i, c), at(x, i, d));
      if (c == d) g -= std::complex<double>(1.0, 0.0);
      acc += g.real() * g.real() + g.imag() * g.imag();
    }
  }
  return std::sqrt(acc);
}

constexpr int min_int(int a, int b) noexcept { return a < b ? a : b; }
constexpr int max_int(int a, int b) noexcept { return a > b ? a : b; }

// The spectrum verdicts, which stand on their own: a NaN in S fails them
// here rather than being inherited from a NaN elsewhere in U or V. Both
// operands of the ordering step are guarded while they are still values in
// memory: under -ffast-math their difference is a computed value the
// compiler may assume finite, so a guard applied after the subtraction
// proves nothing.
void spectrum_verdicts(const double* S, int k, double order_slack, bool& nonnegative,
                       bool& descending) noexcept {
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
  nonnegative = nonneg;
  descending = desc;
}

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

  const double energy_M = frobenius_sq(Mv, exponent);
  r.norm_M = std::ldexp(std::sqrt(energy_M), exponent);

  // The scaled spectrum, formed once: the residual below multiplies by it
  // inside its innermost loop.
  std::vector<double> S_scaled(static_cast<std::size_t>(k));
  double energy_S = 0.0;
  for (int t = 0; t < k; ++t) {
    const double v = scaled(S[t], exponent);
    S_scaled[static_cast<std::size_t>(t)] = v;
    energy_S += v * v;
  }

  spectrum_verdicts(S, k, tol.spectrum_factor * eps * s_max, r.nonnegative, r.descending);

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
        // (U * s) is formed first, as the operator form did, so the
        // products and their order are unchanged.
        const std::complex<double> u = at(Uv, i, t);
        const double st = S_scaled[static_cast<std::size_t>(t)];
        approx += conj_mul(at(Vv, j, t),
                           std::complex<double>(u.real() * st, u.imag() * st));
      }
      const std::complex<double> d = scaled(at(Mv, i, j), exponent) - approx;
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
// Thin / truncated SVD, screened
// ---------------------------------------------------------------------------

namespace {

// The probe vectors' bit source: SplitMix64, a few lines with one word of
// state and a fixed, platform-independent output, which is all a screen
// whose verdict must be repeatable needs from a generator.
struct SplitMix64 {
  std::uint64_t state;
  std::uint64_t next() noexcept {
    state += 0x9E3779B97F4A7C15ull;
    std::uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }
};

// The seed every call starts from. Digits of pi; nothing about the value
// matters except that it never changes.
constexpr std::uint64_t kScreenSeed = 0x243F6A8885A308D3ull;

// Fills x with entries from {1, -1, i, -i}, two bits per entry, so that
// ||x||^2 == n exactly and E |r^* x|^2 == ||r||^2 for any fixed r: the
// entries are independent with zero mean and unit second moment, which is
// all the identity needs.
void rademacher_fill(SplitMix64& g, std::complex<double>* x, int n) noexcept {
  std::uint64_t bits = 0;
  int left = 0;
  for (int j = 0; j < n; ++j) {
    if (left == 0) {
      bits = g.next();
      left = 32;
    }
    switch (bits & 3u) {
      case 0:
        x[j] = std::complex<double>(1.0, 0.0);
        break;
      case 1:
        x[j] = std::complex<double>(-1.0, 0.0);
        break;
      case 2:
        x[j] = std::complex<double>(0.0, 1.0);
        break;
      default:
        x[j] = std::complex<double>(0.0, -1.0);
        break;
    }
    bits >>= 2;
    --left;
  }
}

// Every product below is written on the parts. This file is built strict,
// and under strict floating point std::complex's operator* is a call to
// __muldc3 on every multiplication; the screen exists to be cheap, so it
// cannot pay that.

// out (n) = X^* x (X is n x k column-major, x has n entries): out_t = x_t^* x.
template <typename View>
void adjoint_times(const View& X, const std::complex<double>* x,
                   std::complex<double>* out) noexcept {
  const int n = rows_of(X);
  const int k = cols_of(X);
  for (int t = 0; t < k; ++t) {
    double ar = 0.0;
    double ai = 0.0;
    for (int i = 0; i < n; ++i) {
      const std::complex<double>& e = at(X, i, t);
      ar += e.real() * x[i].real() + e.imag() * x[i].imag();
      ai += e.real() * x[i].imag() - e.imag() * x[i].real();
    }
    out[t] = std::complex<double>(ar, ai);
  }
}

// out (n) = X w (X is n x k column-major, w has k entries), column by column.
template <typename View>
void times(const View& X, const std::complex<double>* w, std::complex<double>* out) noexcept {
  const int n = rows_of(X);
  const int k = cols_of(X);
  for (int i = 0; i < n; ++i) out[i] = std::complex<double>(0.0, 0.0);
  for (int t = 0; t < k; ++t) {
    const double wr = w[t].real();
    const double wi = w[t].imag();
    for (int i = 0; i < n; ++i) {
      const std::complex<double>& e = at(X, i, t);
      out[i] += std::complex<double>(e.real() * wr - e.imag() * wi,
                                     e.real() * wi + e.imag() * wr);
    }
  }
}

// out (rows) = 2^-e M x and out (cols) = 2^-e M^* z for M in either storage
// order, with the inner loop running along the storage so that each pass
// over M is contiguous: a row-major M is a column-major M^T, and each
// product is arranged so that the transposed view is what it multiplies by.
//
// The scaling is check_svd's, and where it is applied matters. Moving it
// onto the vector is exact and costs nothing per element, but a vector of
// unit entries scaled by 2^1024 is infinite, and a matrix below the normal
// range needs exactly that exponent. So the vector carries the scaling
// while |e| is moderate, and beyond that the elements are scaled one by one
// as check_svd scales them, which is what keeps such a matrix measurable at
// all. The limit is set by the energy pass below, which sums unscaled
// squares under the same rule: elements are under 2^e, their squares under
// 2^2e, and the sum has at most 2^32 terms, so it stays finite while
// 2e + 32 < 1024. The products are safer than that: a probe entry is at
// most sqrt(k) in modulus, so scaled by 2^-e it is normal well past this
// limit, and each product with an element under 2^e is of order one.
constexpr int kVectorScalingLimit = 480;

// ||2^-e M||_F^2 over the buffer, which holds the same elements in either
// order, so one linear pass serves both. Under the limit the squares are
// summed as they are and the sum scaled once; beyond it each element is
// scaled first, as check_svd's pass does throughout. A term whose square
// underflows on the unscaled path is at least 2^-112 below the largest term
// relative to it, and contributes nothing the sum could measure.
double energy_ordered(const std::complex<double>* M, int rows, int cols, int exponent) noexcept {
  const std::size_t count = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
  double acc = 0.0;
  if (exponent >= -kVectorScalingLimit && exponent <= kVectorScalingLimit) {
    for (std::size_t i = 0; i < count; ++i) {
      acc += M[i].real() * M[i].real() + M[i].imag() * M[i].imag();
    }
    return std::ldexp(acc, -2 * exponent);
  }
  for (std::size_t i = 0; i < count; ++i) {
    const std::complex<double> z = scaled(M[i], exponent);
    acc += z.real() * z.real() + z.imag() * z.imag();
  }
  return acc;
}

template <bool kScaleElements>
void times_ordered_impl(const std::complex<double>* M, int rows, int cols, MatrixOrder order,
                        int exponent, const std::complex<double>* x,
                        std::complex<double>* out) noexcept {
  auto load = [exponent](const std::complex<double>& e) noexcept {
    return kScaleElements ? scaled(e, exponent) : e;
  };
  if (order == MatrixOrder::ColMajor) {
    const auto Mv = autonne::detail::col_major(M, rows, cols);
    for (int i = 0; i < rows; ++i) out[i] = std::complex<double>(0.0, 0.0);
    for (int j = 0; j < cols; ++j) {
      const double xr = x[j].real();
      const double xi = x[j].imag();
      for (int i = 0; i < rows; ++i) {
        const std::complex<double> e = load(at(Mv, i, j));
        out[i] += std::complex<double>(e.real() * xr - e.imag() * xi,
                                       e.real() * xi + e.imag() * xr);
      }
    }
    return;
  }
  const auto Mt = autonne::detail::col_major(M, cols, rows);
  for (int i = 0; i < rows; ++i) {
    double ar = 0.0;
    double ai = 0.0;
    for (int j = 0; j < cols; ++j) {
      const std::complex<double> e = load(at(Mt, j, i));
      ar += e.real() * x[j].real() - e.imag() * x[j].imag();
      ai += e.real() * x[j].imag() + e.imag() * x[j].real();
    }
    out[i] = std::complex<double>(ar, ai);
  }
}

template <bool kScaleElements>
void adjoint_times_ordered_impl(const std::complex<double>* M, int rows, int cols,
                                MatrixOrder order, int exponent, const std::complex<double>* z,
                                std::complex<double>* out) noexcept {
  auto load = [exponent](const std::complex<double>& e) noexcept {
    return kScaleElements ? scaled(e, exponent) : e;
  };
  if (order == MatrixOrder::ColMajor) {
    const auto Mv = autonne::detail::col_major(M, rows, cols);
    for (int j = 0; j < cols; ++j) {
      double ar = 0.0;
      double ai = 0.0;
      for (int i = 0; i < rows; ++i) {
        const std::complex<double> e = load(at(Mv, i, j));
        ar += e.real() * z[i].real() + e.imag() * z[i].imag();
        ai += e.real() * z[i].imag() - e.imag() * z[i].real();
      }
      out[j] = std::complex<double>(ar, ai);
    }
    return;
  }
  // M^* z = conj(M^T conj(z)): multiply the transposed view by conj(z)
  // column by column and conjugate the result.
  const auto Mt = autonne::detail::col_major(M, cols, rows);
  for (int j = 0; j < cols; ++j) out[j] = std::complex<double>(0.0, 0.0);
  for (int i = 0; i < rows; ++i) {
    const double zr = z[i].real();
    const double zi = -z[i].imag();
    for (int j = 0; j < cols; ++j) {
      const std::complex<double> e = load(at(Mt, j, i));
      out[j] += std::complex<double>(e.real() * zr - e.imag() * zi, e.real() * zi + e.imag() * zr);
    }
  }
  for (int j = 0; j < cols; ++j) out[j] = std::conj(out[j]);
}

// The two products with the scaling placed as described above. `x` and `z`
// are scaled in place on the fast path, so they are inputs the caller is
// done with.
void times_ordered(const std::complex<double>* M, int rows, int cols, MatrixOrder order,
                   int exponent, std::complex<double>* x, std::complex<double>* out) noexcept {
  if (exponent >= -kVectorScalingLimit && exponent <= kVectorScalingLimit) {
    for (int j = 0; j < cols; ++j) x[j] = scaled(x[j], exponent);
    times_ordered_impl<false>(M, rows, cols, order, 0, x, out);
  } else {
    times_ordered_impl<true>(M, rows, cols, order, exponent, x, out);
  }
}

void adjoint_times_ordered(const std::complex<double>* M, int rows, int cols, MatrixOrder order,
                           int exponent, std::complex<double>* z,
                           std::complex<double>* out) noexcept {
  if (exponent >= -kVectorScalingLimit && exponent <= kVectorScalingLimit) {
    for (int i = 0; i < rows; ++i) z[i] = scaled(z[i], exponent);
    adjoint_times_ordered_impl<false>(M, rows, cols, order, 0, z, out);
  } else {
    adjoint_times_ordered_impl<true>(M, rows, cols, order, exponent, z, out);
  }
}

double norm_of(const std::complex<double>* x, int n) noexcept {
  double acc = 0.0;
  for (int i = 0; i < n; ++i) acc += x[i].real() * x[i].real() + x[i].imag() * x[i].imag();
  return std::sqrt(acc);
}

// ||X^* (X y) - y|| for a probe y with k entries: an estimate of
// ||X^* X - I||_F. `work` needs n entries, `back` k.
template <typename View>
double ortho_probe(const View& X, const std::complex<double>* y, std::complex<double>* work,
                   std::complex<double>* back) noexcept {
  const int k = cols_of(X);
  times(X, y, work);
  adjoint_times(X, work, back);
  for (int t = 0; t < k; ++t) back[t] -= y[t];
  return norm_of(back, k);
}

}  // namespace

SvdScreenReport screen_svd(const std::complex<double>* M, int rows, int cols,
                           MatrixOrder order, const std::complex<double>* U,
                           const double* S, const std::complex<double>* V, int k,
                           const Tolerances& tol, int probes) {
  SvdScreenReport r;
  r.rows = rows;
  r.cols = cols;
  r.k = k;
  r.probes = probes;

  const int full_k = min_int(rows, cols);
  if (rows <= 0 || cols <= 0 || k <= 0 || k > full_k || probes <= 0 || M == nullptr ||
      U == nullptr || S == nullptr || V == nullptr) {
    return r;  // inputs_valid stays false; every verdict stays false
  }
  r.inputs_valid = true;
  r.truncated = k < full_k;

  const auto Mv = autonne::detail::ordered(M, rows, cols, order);
  const auto Uv = autonne::detail::col_major(U, rows, k);
  const auto Vv = autonne::detail::col_major(V, cols, k);

  r.finite = all_finite(Uv) && all_finite(S, k) && all_finite(Vv);

  const double eps = tol.eps;
  const double dim = static_cast<double>(max_int(rows, cols));

  // Measured on the input scaled by an exact power of two, as check_svd is;
  // see scale_exponent, and times_ordered for where the scaling is applied
  // in the products.
  const double m_max = max_component(Mv);
  const double s_max = max_component(S, k);
  const int exponent = scale_exponent(m_max > s_max ? m_max : s_max);

  const double energy_M = energy_ordered(M, rows, cols, exponent);
  const double norm_M_scaled = std::sqrt(energy_M);
  r.norm_M = std::ldexp(norm_M_scaled, exponent);

  std::vector<double> S_scaled(static_cast<std::size_t>(k));
  double energy_S = 0.0;
  for (int t = 0; t < k; ++t) {
    const double v = scaled(S[t], exponent);
    S_scaled[static_cast<std::size_t>(t)] = v;
    energy_S += v * v;
  }

  spectrum_verdicts(S, k, tol.spectrum_factor * eps * s_max, r.nonnegative, r.descending);

  const double energy_defect = energy_S - energy_M;
  const double energy_bound = tol.spectrum_factor * dim * eps * energy_M;
  const double energy_defect_abs = std::fabs(energy_defect);
  r.energy_ok = r.truncated ? within(energy_defect, energy_bound)
                            : within(energy_defect_abs, energy_bound);
  r.energy_defect = std::ldexp(energy_defect, 2 * exponent);
  r.energy_bound = std::ldexp(energy_bound, 2 * exponent);

  const double discarded = (energy_M > energy_S) ? (energy_M - energy_S) : 0.0;
  r.discarded_energy = std::ldexp(discarded, 2 * exponent);

  // The probes. A residual probe draws y with k entries and forms both
  //
  //   M_s (V y) - U (S_s y)      and      M_s^* (U y) - V (S_s y),
  //
  // with M_s and S_s the scaled matrix and spectrum, as two matrix-vector
  // products with M and two with the factors. For a correct factorisation
  // M V_k = U_k S_k and M^* U_k = V_k S_k hold to rounding whatever k is,
  // so the truncated part of M never enters: the probes measure the
  // residual's error part alone, where the whole residual would carry the
  // discarded energy with a realisation-to-realisation spread of order one
  // for a low-rank discard. When k = min(rows, cols) one of V and U is
  // square unitary and the matching probe is exactly ||R x|| for the
  // harness's R = M - U S V^*, so the estimate is of the harness's own
  // residual; when V (or U) is merely orthonormal to the ortho bound the
  // two differ by at most s_max times that bound, inside the harness's
  // margin. Each orthonormality probe forms X^* (X y) - y. The worst
  // estimate over the probes is what is compared with the bound.
  const std::size_t sr = static_cast<std::size_t>(rows);
  const std::size_t sc = static_cast<std::size_t>(cols);
  const std::size_t sk = static_cast<std::size_t>(k);
  const std::size_t longest = sr > sc ? sr : sc;
  std::vector<std::complex<double>> y(sk);
  std::vector<std::complex<double>> w(sk);
  std::vector<std::complex<double>> x(sc);
  std::vector<std::complex<double>> z(sr);
  std::vector<std::complex<double>> lhs(longest);
  std::vector<std::complex<double>> rhs(longest);
  std::vector<std::complex<double>> back(sk);

  // Worst over the probes, with a non-finite estimate poisoning the worst
  // for good: a NaN here means the input holds one (M is never scanned, as
  // in check_svd), and the verdict must be false with the cause visible.
  auto take = [](double& worst, double value) noexcept {
    if (fp_bad(worst)) return;
    if (fp_bad(value)) {
      worst = std::numeric_limits<double>::quiet_NaN();
      return;
    }
    if (value > worst) worst = value;
  };

  SplitMix64 g{kScreenSeed};
  double worst_residual = 0.0;
  double worst_u = 0.0;
  double worst_v = 0.0;
  for (int p = 0; p < probes; ++p) {
    rademacher_fill(g, y.data(), k);
    for (int t = 0; t < k; ++t) {
      const double st = S_scaled[static_cast<std::size_t>(t)];
      w[static_cast<std::size_t>(t)] =
          std::complex<double>(st * y[static_cast<std::size_t>(t)].real(),
                               st * y[static_cast<std::size_t>(t)].imag());
    }
    times(Vv, y.data(), x.data());
    times_ordered(M, rows, cols, order, exponent, x.data(), lhs.data());
    times(Uv, w.data(), rhs.data());
    for (int i = 0; i < rows; ++i) lhs[static_cast<std::size_t>(i)] -= rhs[static_cast<std::size_t>(i)];
    const double right = norm_of(lhs.data(), rows);

    times(Uv, y.data(), z.data());
    adjoint_times_ordered(M, rows, cols, order, exponent, z.data(), lhs.data());
    times(Vv, w.data(), rhs.data());
    for (int j = 0; j < cols; ++j) lhs[static_cast<std::size_t>(j)] -= rhs[static_cast<std::size_t>(j)];
    const double left = norm_of(lhs.data(), cols);

    take(worst_residual, right);
    take(worst_residual, left);

    rademacher_fill(g, y.data(), k);
    take(worst_u, ortho_probe(Uv, y.data(), lhs.data(), back.data()));
    rademacher_fill(g, y.data(), k);
    take(worst_v, ortho_probe(Vv, y.data(), lhs.data(), back.data()));
  }

  // The harness adds sqrt(discarded) to this bound because it measures the
  // whole residual; the probes above measure only the part the truncation
  // does not account for, so the bound here is the error term alone.
  const double backward_bound = tol.backward_factor * dim * eps * norm_M_scaled;
  r.backward_ok = r.finite && within(worst_residual, backward_bound);
  r.residual_estimate = std::ldexp(worst_residual, exponent);
  r.backward_bound = std::ldexp(backward_bound, exponent);

  r.u_ortho_estimate = worst_u;
  r.v_ortho_estimate = worst_v;
  r.ortho_bound = tol.ortho_factor * dim * eps;
  r.u_orthonormal = r.finite && within(worst_u, r.ortho_bound);
  r.v_orthonormal = r.finite && within(worst_v, r.ortho_bound);

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

  const double energy_A = frobenius_sq(Av, exponent);
  const double norm_A_scaled = std::sqrt(energy_A);
  r.norm_A = std::ldexp(norm_A_scaled, exponent);

  double herm_sq = 0.0;
  double trace = 0.0;
  for (int j = 0; j < n; ++j) {
    trace += scaled(at(Av, j, j).real(), exponent);
    for (int i = 0; i < n; ++i) {
      const std::complex<double> d =
          scaled(at(Av, i, j) - std::conj(at(Av, j, i)), exponent);
      herm_sq += d.real() * d.real() + d.imag() * d.imag();
    }
  }
  const double hermitian_residual = std::sqrt(herm_sq);
  const double hermitian_bound = tol.backward_factor * dim * eps * norm_A_scaled;
  r.input_hermitian = within(hermitian_residual, hermitian_bound);
  r.hermitian_residual = std::ldexp(hermitian_residual, exponent);
  r.hermitian_bound = std::ldexp(hermitian_bound, exponent);

  std::vector<double> lambda_scaled(static_cast<std::size_t>(n));
  double energy_L = 0.0;
  double sum_L = 0.0;
  for (int t = 0; t < n; ++t) {
    const double v = scaled(evals[t], exponent);
    lambda_scaled[static_cast<std::size_t>(t)] = v;
    energy_L += v * v;
    sum_L += v;
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
      for (int t = 0; t < n; ++t) {
        acc += cmul(scaled(at(Av, i, t), exponent), at(Qv, t, j));
      }
      const std::complex<double> d =
          acc - at(Qv, i, j) * lambda_scaled[static_cast<std::size_t>(j)];
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
