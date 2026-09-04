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

// Verification harness.
//
// autonne is judged by this file rather than trusted in place of it: a caller
// checks every factorisation against the matrix it came from, and a check that
// cannot reject a bad factorisation is worth nothing.
//
// check_svd, check_eigh and orthonormality_residual are declared here and
// defined in src/verify.cpp, so each has exactly one strong definition per
// binary. Inline, they would have vague linkage: flags are not part of a
// mangled name, so a binary mixing -ffast-math and -fno-fast-math translation
// units would run whichever copy the linker happened to keep. Out-of-line, the
// floating-point flag travels with the code that needs it.
//
// fp_bad and the small helpers below stay inline. fp_bad is flag-insensitive
// by construction -- bit_cast plus an integer mask -- and the rest are cheap
// enough that the caller's own flags are the ones that should apply.

#ifndef AUTONNE_VERIFY_HPP
#define AUTONNE_VERIFY_HPP

#include <bit>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>

#include "autonne/autonne.hpp"
#include "autonne/detail/matrix_view.hpp"

namespace autonne {
namespace verify {

// ---------------------------------------------------------------------------
// Non-finite detection
// ---------------------------------------------------------------------------

// True if x is NaN or an infinity.
//
// Deliberately does not call std::isnan or std::isfinite. Under -ffast-math
// (specifically -ffinite-math-only) the compiler is entitled to assume no NaN
// or infinity is ever produced and folds those predicates to a constant false,
// which deletes the guard entirely. This reads the IEEE-754 binary64 exponent
// field through std::bit_cast: an all-ones exponent means NaN or infinity. It
// is integer work on the object representation, so no assumption about the
// range of floating-point values can remove it.
constexpr bool fp_bad(double x) noexcept {
  static_assert(sizeof(double) == sizeof(std::uint64_t),
                "fp_bad assumes IEEE-754 binary64");
  const std::uint64_t bits = std::bit_cast<std::uint64_t>(x);
  return ((bits >> 52) & UINT64_C(0x7FF)) == UINT64_C(0x7FF);
}

inline bool fp_bad(const std::complex<double>& z) noexcept {
  return fp_bad(z.real()) || fp_bad(z.imag());
}

namespace detail {

using autonne::detail::at;
using autonne::detail::cols_of;
using autonne::detail::rows_of;

// A comparison is only meaningful once both sides are known finite; with a NaN
// operand the ordering predicates are themselves subject to the same
// -ffast-math assumptions. Every threshold test below routes through here.
constexpr bool within(double value, double bound) noexcept {
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

inline bool all_finite(const double* v, int n) noexcept {
  for (int i = 0; i < n; ++i) {
    if (fp_bad(v[i])) return false;
  }
  return true;
}

// Sum of |a(i,j)|^2. Squared norms are accumulated rather than square-rooted
// per element so that the energy identities below compare like with like.
template <typename View>
double frobenius_sq(const View& a) noexcept {
  double acc = 0.0;
  for (int j = 0; j < cols_of(a); ++j) {
    for (int i = 0; i < rows_of(a); ++i) {
      const std::complex<double> z = at(a, i, j);
      acc += z.real() * z.real() + z.imag() * z.imag();
    }
  }
  return acc;
}

// The kept columns of U, V and the eigenvectors are all column-major, so this
// needs no template parameter; naming the type keeps it out of line.
using ConstColMajor = autonne::detail::ColMajorRef<const std::complex<double>>;

// ||X^* X - I||_F for an n x k matrix X. Defined in src/verify.cpp.
double orthonormality_residual(const ConstColMajor& x) noexcept;

constexpr int min_int(int a, int b) noexcept { return a < b ? a : b; }
constexpr int max_int(int a, int b) noexcept { return a > b ? a : b; }

}  // namespace detail

// ---------------------------------------------------------------------------
// Tolerances
// ---------------------------------------------------------------------------

// Every bound below is `factor * max(dimension) * eps * scale`. The factors are
// generous by design: the harness must reject a wrong factorisation, not grade
// a right one to the last ulp.
struct Tolerances {
  double eps = std::numeric_limits<double>::epsilon();
  double backward_factor = 64.0;
  double ortho_factor = 64.0;
  double spectrum_factor = 64.0;
};

// ---------------------------------------------------------------------------
// Thin / truncated SVD
// ---------------------------------------------------------------------------

struct SvdReport {
  // Shape actually checked.
  int rows = 0;
  int cols = 0;
  int k = 0;
  bool truncated = false;  // k < min(rows, cols)

  // Measured quantities.
  double norm_M = 0.0;            // ||M||_F
  double residual = 0.0;          // ||M - U_k S_k V_k^*||_F
  double discarded_energy = 0.0;  // max(0, ||M||_F^2 - sum s_i^2)
  double backward_bound = 0.0;
  double u_ortho_residual = 0.0;  // ||U^* U - I||_F
  double v_ortho_residual = 0.0;  // ||V^* V - I||_F
  double ortho_bound = 0.0;
  double energy_defect = 0.0;  // sum s_i^2 - ||M||_F^2
  double energy_bound = 0.0;

  // Verdicts.
  bool inputs_valid = false;
  bool finite = false;       // no NaN/Inf across the kept slice of U, S, V
  bool nonnegative = false;  // s_i >= 0
  bool descending = false;   // s_i >= s_{i+1}
  bool backward_ok = false;
  bool u_orthonormal = false;
  bool v_orthonormal = false;
  bool energy_ok = false;

  constexpr bool ok() const noexcept {
    return inputs_valid && finite && nonnegative && descending && backward_ok &&
           u_orthonormal && v_orthonormal && energy_ok;
  }
};

// Checks U_k, S_k, V_k against the matrix M they claim to factor.
//
//   M      rows x cols in `order`
//   U      rows x k, column-major
//   S      k values
//   V      cols x k, column-major (V itself, not V^*)
//
// The backward error is compared in amplitude form,
//
//   ||M - U_k S_k V_k^*||_F <= sqrt(discarded) + bwd * ||M||_F
//
// with bwd = backward_factor * max(rows, cols) * eps and `discarded` the
// energy in the singular values not kept. Squaring both sides is wrong: it
// drops the cross term 2 * bwd * sqrt(discarded * ||M||_F^2), which is the
// dominant term whenever the truncation is heavy.
//
// The spectral energy identity sum s_i^2 == ||M||_F^2 holds exactly when
// nothing was truncated; when k < min(rows, cols) the requirement weakens to
// the one-sided sum s_i^2 <= ||M||_F^2, since the missing energy is precisely
// what `discarded` accounts for.
SvdReport check_svd(const std::complex<double>* M, int rows, int cols,
                    MatrixOrder order, const std::complex<double>* U,
                    const double* S, const std::complex<double>* V, int k,
                    const Tolerances& tol = Tolerances());

// ---------------------------------------------------------------------------
// Hermitian eigendecomposition
// ---------------------------------------------------------------------------

struct EighReport {
  int n = 0;

  double norm_A = 0.0;              // ||A||_F
  double hermitian_residual = 0.0;  // ||A - A^*||_F
  double hermitian_bound = 0.0;
  double residual = 0.0;  // ||A Q - Q diag(lambda)||_F
  double backward_bound = 0.0;
  double q_ortho_residual = 0.0;  // ||Q^* Q - I||_F
  double ortho_bound = 0.0;
  double trace_defect = 0.0;  // sum lambda_i - Re tr(A)
  double trace_bound = 0.0;
  double energy_defect = 0.0;  // sum lambda_i^2 - ||A||_F^2
  double energy_bound = 0.0;

  bool inputs_valid = false;
  bool input_hermitian = false;
  bool finite = false;
  bool ascending = false;
  bool backward_ok = false;
  bool q_orthonormal = false;
  bool trace_ok = false;
  bool energy_ok = false;

  constexpr bool ok() const noexcept {
    return inputs_valid && input_hermitian && finite && ascending &&
           backward_ok && q_orthonormal && trace_ok && energy_ok;
  }
};

// Checks (evals, evecs) against the Hermitian matrix A they claim to
// diagonalise. Eigenvalues are required in ascending order, matching the
// contract in autonne.hpp. `input_hermitian` reports on the caller's input
// rather than on autonne: a non-Hermitian A makes every other verdict
// meaningless, so it is surfaced separately.
EighReport check_eigh(const std::complex<double>* A, int n, MatrixOrder order,
                      const double* evals, const std::complex<double>* evecs,
                      const Tolerances& tol = Tolerances());

}  // namespace verify
}  // namespace autonne

#endif  // AUTONNE_VERIFY_HPP
