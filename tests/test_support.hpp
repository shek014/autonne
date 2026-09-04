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
#include <cmath>
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


// ---------------------------------------------------------------------------
// Kernel fixtures
// ---------------------------------------------------------------------------

// Values a kernel must never leave behind on a false return; the tests check
// output buffers still hold them.
constexpr Complex kSentinel(-12345.0, 6789.0);
constexpr double kSentinelReal = -98765.0;

struct SvdResult {
  bool ok = false;
  std::vector<Complex> u;   // rows x k, column-major
  std::vector<double> s;    // k
  std::vector<Complex> v;   // cols x k, column-major
};

inline SvdResult run_svd(const Complex* m, int rows, int cols,
                         MatrixOrder order) {
  const int k = rows < cols ? rows : cols;
  SvdResult r;
  r.u.assign(static_cast<std::size_t>(rows) * static_cast<std::size_t>(k), kSentinel);
  r.s.assign(static_cast<std::size_t>(k), kSentinelReal);
  r.v.assign(static_cast<std::size_t>(cols) * static_cast<std::size_t>(k), kSentinel);
  r.ok = autonne::svd_thin(m, rows, cols, order, r.u.data(), r.s.data(), r.v.data());
  return r;
}

inline SvdResult run_svd(const SvdCase& c) {
  return run_svd(c.m.data(), c.rows, c.cols, c.order);
}

struct EighResult {
  bool ok = false;
  std::vector<double> evals;   // n
  std::vector<Complex> evecs;  // n x n, column-major
};

inline EighResult run_eigh(const Complex* a, int n, MatrixOrder order) {
  EighResult r;
  r.evals.assign(static_cast<std::size_t>(n), kSentinelReal);
  r.evecs.assign(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), kSentinel);
  r.ok = autonne::eigh(a, n, order, r.evals.data(), r.evecs.data());
  return r;
}

inline EighResult run_eigh(const EighCase& c) {
  return run_eigh(c.a.data(), c.n, c.order);
}

// Dense rows x cols matrix of uniform complex entries, column-major.
inline std::vector<Complex> random_matrix(int rows, int cols, std::uint64_t seed) {
  Lcg rng(seed);
  std::vector<Complex> m(static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols));
  for (Complex& z : m) z = rng.next_complex();
  return m;
}

// Random Hermitian n x n, column-major: (X + X^*) / 2 for random X.
inline std::vector<Complex> random_hermitian(int n, std::uint64_t seed) {
  std::vector<Complex> x = random_matrix(n, n, seed);
  const std::size_t sn = static_cast<std::size_t>(n);
  std::vector<Complex> h(sn * sn);
  for (std::size_t j = 0; j < sn; ++j) {
    for (std::size_t i = 0; i < sn; ++i) {
      h[i + j * sn] = 0.5 * (x[i + j * sn] + std::conj(x[j + i * sn]));
    }
    h[j + j * sn] = Complex(h[j + j * sn].real(), 0.0);
  }
  return h;
}

// The n x n discrete Fourier transform, unitary: F(j, k) = e^{-2 pi i j k / n} / sqrt(n).
// Every singular value is 1, so it is the fully degenerate case.
inline std::vector<Complex> dft_unitary(int n) {
  const std::size_t sn = static_cast<std::size_t>(n);
  std::vector<Complex> f(sn * sn);
  const double two_pi = 6.283185307179586476925286766559;
  const double scale = 1.0 / std::sqrt(static_cast<double>(n));
  for (int k = 0; k < n; ++k) {
    for (int j = 0; j < n; ++j) {
      const double angle = -two_pi * static_cast<double>((j * k) % n) / static_cast<double>(n);
      f[static_cast<std::size_t>(j) + static_cast<std::size_t>(k) * sn] =
          Complex(scale * std::cos(angle), scale * std::sin(angle));
    }
  }
  return f;
}

// Column-major product C = A (rows x inner) * B (inner x cols).
inline std::vector<Complex> matmul(const std::vector<Complex>& a, int rows, int inner,
                                   const std::vector<Complex>& b, int cols) {
  const std::size_t sr = static_cast<std::size_t>(rows);
  const std::size_t si = static_cast<std::size_t>(inner);
  std::vector<Complex> c(sr * static_cast<std::size_t>(cols), Complex(0.0, 0.0));
  for (int j = 0; j < cols; ++j) {
    const std::size_t sj = static_cast<std::size_t>(j);
    for (int t = 0; t < inner; ++t) {
      const std::size_t st = static_cast<std::size_t>(t);
      const Complex bt = b[st + sj * si];
      for (int i = 0; i < rows; ++i) {
        const std::size_t sii = static_cast<std::size_t>(i);
        c[sii + sj * sr] += a[sii + st * sr] * bt;
      }
    }
  }
  return c;
}

// n x n matrix B = F (I + 0.1 E) with F the DFT and ||E||_F <= 1, so every
// singular value of B lies in [0.9, 1.1]. The bound is what the relative
// accuracy tests lean on: sigma_k(B D) and sigma_k(D B) lie in
// [0.9, 1.1] * d_(k), whatever the scaling D.
inline std::vector<Complex> well_conditioned_matrix(int n, std::uint64_t seed) {
  const std::size_t sn = static_cast<std::size_t>(n);
  std::vector<Complex> e = random_matrix(n, n, seed);
  double norm_sq = 0.0;
  for (const Complex& z : e) norm_sq += std::norm(z);
  const double scale = 0.1 / std::sqrt(norm_sq);
  for (std::size_t j = 0; j < sn; ++j) {
    for (std::size_t i = 0; i < sn; ++i) {
      e[i + j * sn] *= scale;
      if (i == j) e[i + j * sn] += Complex(1.0, 0.0);
    }
  }
  return matmul(dft_unitary(n), n, n, e, n);
}

// n x n Hermitian B = I + 0.1 E with E Hermitian and ||E||_F <= 1, so every
// eigenvalue of B lies in [0.9, 1.1]; then lambda_k(D B D) lies in
// [0.9, 1.1] * d_(k)^2 (Ostrowski).
inline std::vector<Complex> well_conditioned_hermitian(int n, std::uint64_t seed) {
  const std::size_t sn = static_cast<std::size_t>(n);
  std::vector<Complex> e = random_hermitian(n, seed);
  double norm_sq = 0.0;
  for (const Complex& z : e) norm_sq += std::norm(z);
  const double scale = 0.1 / std::sqrt(norm_sq);
  for (std::size_t j = 0; j < sn; ++j) {
    for (std::size_t i = 0; i < sn; ++i) {
      e[i + j * sn] *= scale;
      if (i == j) e[i + j * sn] += Complex(1.0, 0.0);
    }
  }
  return e;
}

// Scales column j of the column-major rows x cols matrix by d[j].
inline void scale_columns(std::vector<Complex>& m, int rows, int cols,
                          const std::vector<double>& d) {
  const std::size_t sr = static_cast<std::size_t>(rows);
  for (int j = 0; j < cols; ++j) {
    for (int i = 0; i < rows; ++i) {
      m[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * sr] *=
          d[static_cast<std::size_t>(j)];
    }
  }
}

// Scales row i of the column-major rows x cols matrix by d[i].
inline void scale_rows(std::vector<Complex>& m, int rows, int cols,
                       const std::vector<double>& d) {
  const std::size_t sr = static_cast<std::size_t>(rows);
  for (int j = 0; j < cols; ++j) {
    for (int i = 0; i < rows; ++i) {
      m[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * sr] *=
          d[static_cast<std::size_t>(i)];
    }
  }
}

// The Simon-problem state (1/6) sum_x |x>|f(x)> over Z_6 x Z_6 with hidden
// shift s = (2, 4), as a 36 x 36 matrix M(x, y) = 1/6 if y = f(x), where f(x)
// is the lexicographic minimum of the coset x + <s>. The shift has order
// three, so there are twelve cosets of three elements each: twelve nonzero
// columns, mutually orthogonal, each of norm sqrt(3)/6. The exact spectrum is
// therefore twelve copies of 1/(2 sqrt 3) and twenty-four zeros, with sum of
// squares one. This is the structure of the matrix on which Eigen 3.4.0's
// divide-and-conquer SVD returned a wrong spectrum (verycareful/lindblad#95);
// the lindblad reproducer itself is not copied here.
//
// `residue` is added to M(0, 0) when nonzero, to mimic the rounding residue
// the original carried. Column-major.
inline std::vector<Complex> simon_coset_matrix(double residue = 0.0) {
  const int d = 6;
  const int n = d * d;
  const int s0 = 2;
  const int s1 = 4;
  std::vector<Complex> m(static_cast<std::size_t>(n) * static_cast<std::size_t>(n),
                         Complex(0.0, 0.0));
  for (int x = 0; x < n; ++x) {
    const int x0 = x % d;
    const int x1 = x / d;
    int best0 = x0;
    int best1 = x1;
    for (int k = 1; k < d; ++k) {
      const int c0 = (x0 + k * s0) % d;
      const int c1 = (x1 + k * s1) % d;
      if (c0 < best0 || (c0 == best0 && c1 < best1)) {
        best0 = c0;
        best1 = c1;
      }
    }
    const int y = best0 + d * best1;
    m[static_cast<std::size_t>(x) + static_cast<std::size_t>(y) * static_cast<std::size_t>(n)] =
        Complex(1.0 / 6.0, 0.0);
  }
  m[0] += Complex(residue, 0.0);
  return m;
}

// An 8 x 8 matrix with the structure of the "poison theta" from the same
// issue: four orthonormal columns of two entries of modulus 1/sqrt(2) each,
// on disjoint rows, followed by columns holding a few entries sixty-odd
// orders of magnitude smaller. The spectrum is four ones, then values below
// 1e-48, then zeros. Eigen's Jacobi SVD under -ffast-math returned NaN inside
// a null-space column of U on the original. Column-major.
inline std::vector<Complex> poison_theta_like() {
  const int n = 8;
  std::vector<Complex> m(static_cast<std::size_t>(n) * static_cast<std::size_t>(n),
                         Complex(0.0, 0.0));
  const double h = 0.70710678118654752440;
  for (int j = 0; j < 4; ++j) {
    m[static_cast<std::size_t>(2 * j) + static_cast<std::size_t>(j) * 8] = Complex(h, 0.0);
    m[static_cast<std::size_t>(2 * j + 1) + static_cast<std::size_t>(j) * 8] = Complex(0.0, h);
  }
  m[0 + 4 * 8] = Complex(1.57e-65, 0.0);
  m[7 + 5 * 8] = Complex(0.0, 2.2e-49);
  return m;
}

}  // namespace autonne_test

#endif  // AUTONNE_TESTS_TEST_SUPPORT_HPP
