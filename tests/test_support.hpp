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

// Test fixtures built from a chosen spectrum.
//
// Every case here is assembled forwards -- pick a spectrum and two unitaries,
// multiply out the matrix -- so the factorisation handed to the harness is
// correct by construction and any rejection is the harness's own doing.

#ifndef AUTONNE_TESTS_TEST_SUPPORT_HPP
#define AUTONNE_TESTS_TEST_SUPPORT_HPP

#include <bit>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "autonne/autonne.hpp"

namespace autonne_test {

using autonne::MatrixOrder;

using Complex = std::complex<double>;

// Deterministic, self-contained, and identical on every platform: the corpus
// must not shift when a standard library changes its distribution internals.
class Lcg {
 public:
  explicit Lcg(std::uint64_t seed) : state_(seed) { next_bits(); }

  // Uniform in [-1, 1).
  double next_uniform() {
    const std::uint64_t x = next_bits() >> 11;  // 53 significant bits
    return static_cast<double>(x) * (2.0 / 9007199254740992.0) - 1.0;
  }

  Complex next_complex() {
    const double re = next_uniform();
    const double im = next_uniform();
    return Complex(re, im);
  }

 private:
  std::uint64_t next_bits() {
    state_ = state_ * UINT64_C(6364136223846793005) +
             UINT64_C(1442695040888963407);
    return state_;
  }
  std::uint64_t state_;
};

// Non-finite values are handled as integers, never as doubles.
//
// A NaN or infinity held in a double-typed expression is not reliably a NaN or
// infinity in a translation unit built with -ffast-math. That flag implies
// -ffinite-math-only, which lets the compiler assume every floating-point
// value crossing a function boundary is finite; Clang 22 acts on the
// assumption hard enough that a NaN returned from a function, or handed to
// std::complex's constructor, does not even reach memory. Only the object
// representation survives, so the fixtures below write it with an integer
// store into a double that already exists, and read it back by reference.
constexpr std::uint64_t kQuietNanBits = UINT64_C(0x7FF8000000000000);
constexpr std::uint64_t kSignalingNanBits = UINT64_C(0x7FF0000000000001);
constexpr std::uint64_t kPositiveInfBits = UINT64_C(0x7FF0000000000000);
constexpr std::uint64_t kNegativeInfBits = UINT64_C(0xFFF0000000000000);
constexpr std::uint64_t kSmallestSubnormalBits = UINT64_C(0x0000000000000001);
constexpr std::uint64_t kLargestNormalBits = UINT64_C(0x7FEFFFFFFFFFFFFF);

// Overwrites the object representation of an existing double or complex.
inline void poke_bits(double& target, std::uint64_t bits) noexcept {
  std::memcpy(&target, &bits, sizeof bits);
}

inline void poke_bits(Complex& target, std::uint64_t re_bits,
                      std::uint64_t im_bits) noexcept {
  const std::uint64_t words[2] = {re_bits, im_bits};
  static_assert(sizeof(Complex) == sizeof words,
                "std::complex<double> must be two adjacent binary64 values");
  std::memcpy(&target, words, sizeof words);
}

// Bit pattern of a finite double, for mixing finite and non-finite parts in
// one poke_bits call.
inline std::uint64_t bits_of(double finite_value) noexcept {
  return std::bit_cast<std::uint64_t>(finite_value);
}

// A double whose representation is chosen bit for bit and lives in memory.
// get() returns a reference so the value reaches the guard under test as a
// load rather than as a by-value argument.
class Slot {
 public:
  explicit Slot(std::uint64_t bits) noexcept { poke_bits(value_, bits); }
  const double& get() const noexcept { return value_; }

 private:
  double value_ = 0.0;
};

// Finite values may still be built from bit patterns directly.
constexpr double bits_to_double(std::uint64_t bits) {
  return std::bit_cast<double>(bits);
}

constexpr double smallest_subnormal() { return bits_to_double(kSmallestSubnormalBits); }
constexpr double largest_normal() { return bits_to_double(kLargestNormalBits); }

// Launders a finite value through memory so the optimiser cannot carry a
// compile-time-known bit pattern into the caller and reason about it.
inline double opaque(double x) {
  volatile double v = x;
  return v;
}

// Applies H = I - 2 v v^* / (v^* v) to the n x n column-major matrix A, in
// place. H is unitary for any nonzero complex v, so a product of these is a
// unitary with no orthogonalisation step to go wrong.
inline void apply_householder(std::vector<Complex>& a, int n,
                              const std::vector<Complex>& v) {
  double beta = 0.0;
  for (int i = 0; i < n; ++i) {
    beta += v[static_cast<std::size_t>(i)].real() * v[static_cast<std::size_t>(i)].real() +
            v[static_cast<std::size_t>(i)].imag() * v[static_cast<std::size_t>(i)].imag();
  }
  if (beta == 0.0) return;
  for (int j = 0; j < n; ++j) {
    const std::size_t col = static_cast<std::size_t>(j) * static_cast<std::size_t>(n);
    Complex w(0.0, 0.0);
    for (int i = 0; i < n; ++i) {
      w += std::conj(v[static_cast<std::size_t>(i)]) * a[col + static_cast<std::size_t>(i)];
    }
    const Complex f = (2.0 / beta) * w;
    for (int i = 0; i < n; ++i) {
      a[col + static_cast<std::size_t>(i)] -= f * v[static_cast<std::size_t>(i)];
    }
  }
}

// n x n unitary, column-major, as a product of two complex Householders.
inline std::vector<Complex> make_unitary(int n, std::uint64_t seed) {
  const std::size_t sn = static_cast<std::size_t>(n);
  std::vector<Complex> a(sn * sn, Complex(0.0, 0.0));
  for (int i = 0; i < n; ++i) a[static_cast<std::size_t>(i) * sn + static_cast<std::size_t>(i)] = Complex(1.0, 0.0);
  Lcg rng(seed);
  for (int reflector = 0; reflector < 2; ++reflector) {
    std::vector<Complex> v(sn);
    for (int i = 0; i < n; ++i) v[static_cast<std::size_t>(i)] = rng.next_complex();
    apply_householder(a, n, v);
  }
  return a;
}

// A matrix together with the factorisation it was built from.
struct SvdCase {
  int rows = 0;
  int cols = 0;
  int k = 0;
  MatrixOrder order = MatrixOrder::ColMajor;
  std::vector<double> s;         // k
  std::vector<Complex> u;        // rows x k, column-major
  std::vector<Complex> v;        // cols x k, column-major
  std::vector<Complex> m;        // rows x cols, laid out per `order`
};

// Builds M = U_k diag(spectrum) V_k^* with U_k, V_k the leading columns of
// freshly generated unitaries. `spectrum` is used exactly as given, including
// its order, so a caller can construct a deliberately mis-ordered case.
inline SvdCase make_svd_case(int rows, int cols,
                             const std::vector<double>& spectrum,
                             MatrixOrder order, std::uint64_t seed) {
  SvdCase c;
  c.rows = rows;
  c.cols = cols;
  c.order = order;
  c.k = static_cast<int>(spectrum.size());
  c.s = spectrum;

  const std::vector<Complex> u_full = make_unitary(rows, seed);
  const std::vector<Complex> v_full = make_unitary(cols, seed + 0x9E3779B9u);
  const std::size_t uk = static_cast<std::size_t>(rows) * static_cast<std::size_t>(c.k);
  const std::size_t vk = static_cast<std::size_t>(cols) * static_cast<std::size_t>(c.k);
  c.u.assign(u_full.begin(), u_full.begin() + static_cast<std::ptrdiff_t>(uk));
  c.v.assign(v_full.begin(), v_full.begin() + static_cast<std::ptrdiff_t>(vk));

  c.m.assign(static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols),
             Complex(0.0, 0.0));
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      Complex acc(0.0, 0.0);
      for (int t = 0; t < c.k; ++t) {
        const Complex ut = c.u[static_cast<std::size_t>(t) * static_cast<std::size_t>(rows) +
                               static_cast<std::size_t>(i)];
        const Complex vt = c.v[static_cast<std::size_t>(t) * static_cast<std::size_t>(cols) +
                               static_cast<std::size_t>(j)];
        acc += ut * c.s[static_cast<std::size_t>(t)] * std::conj(vt);
      }
      const std::size_t idx =
          (order == MatrixOrder::RowMajor)
              ? static_cast<std::size_t>(i) * static_cast<std::size_t>(cols) + static_cast<std::size_t>(j)
              : static_cast<std::size_t>(j) * static_cast<std::size_t>(rows) + static_cast<std::size_t>(i);
      c.m[idx] = acc;
    }
  }
  return c;
}

struct EighCase {
  int n = 0;
  MatrixOrder order = MatrixOrder::ColMajor;
  std::vector<double> evals;  // n
  std::vector<Complex> q;     // n x n, column-major
  std::vector<Complex> a;     // n x n, laid out per `order`
};

// Builds A = Q diag(evals) Q^*, which is Hermitian to rounding.
inline EighCase make_eigh_case(int n, const std::vector<double>& evals,
                               MatrixOrder order, std::uint64_t seed) {
  EighCase c;
  c.n = n;
  c.order = order;
  c.evals = evals;
  c.q = make_unitary(n, seed);
  const std::size_t sn = static_cast<std::size_t>(n);
  c.a.assign(sn * sn, Complex(0.0, 0.0));
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      Complex acc(0.0, 0.0);
      for (int t = 0; t < n; ++t) {
        acc += c.q[static_cast<std::size_t>(t) * sn + static_cast<std::size_t>(i)] *
               c.evals[static_cast<std::size_t>(t)] *
               std::conj(c.q[static_cast<std::size_t>(t) * sn + static_cast<std::size_t>(j)]);
      }
      const std::size_t idx =
          (order == MatrixOrder::RowMajor)
              ? static_cast<std::size_t>(i) * sn + static_cast<std::size_t>(j)
              : static_cast<std::size_t>(j) * sn + static_cast<std::size_t>(i);
      c.a[idx] = acc;
    }
  }
  return c;
}

}  // namespace autonne_test

#endif  // AUTONNE_TESTS_TEST_SUPPORT_HPP
