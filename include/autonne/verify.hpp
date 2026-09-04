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
// The entry points are declared here and defined in src/verify.cpp, so that
// each has exactly one definition per binary, built under that file's
// floating-point flags. A header-only harness would be instantiated by every
// including translation unit and merged by the linker without regard to the
// flags each copy was built under; a consumer mixing -ffast-math and strict
// files would then be judged by whichever copy won the link. The guard
// fp_bad stays inline because it is integer work on the object representation
// and has no floating-point mode to be sensitive to.

#ifndef AUTONNE_VERIFY_HPP
#define AUTONNE_VERIFY_HPP

#include <complex>
#include <limits>

#include "autonne/autonne.hpp"
#include "autonne/detail/fp_bits.hpp"

namespace autonne {
namespace verify {

// ---------------------------------------------------------------------------
// Non-finite detection
// ---------------------------------------------------------------------------

// True if the object in memory is NaN or an infinity. Takes its argument by
// reference deliberately: see detail/fp_bits.hpp for why a by-value double is
// not a value a fast-math build can be trusted to keep non-finite.
using autonne::detail::fp_bad;

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
