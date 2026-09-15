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

// True if src/verify.cpp was itself compiled with -ffast-math.
//
// The harness is the acceptance gate, and nothing judges the judge: if its own
// residual and norm reductions are reassociated, a verdict moves with flags
// that have nothing to do with the factorisation under test. The build pins
// this file strict in every target, but a build file can be edited and the
// damage would be silent -- every test would still pass, just against a
// slightly different number. Reading the macro inside the translation unit
// that matters is the only way to check, which is why this is a function and
// not a constant.
bool built_with_fast_math() noexcept;

// ---------------------------------------------------------------------------
// Tolerances
// ---------------------------------------------------------------------------

// Every bound below is `factor * max(dimension) * eps * scale`. The factors are
// generous by design: the harness must reject a wrong factorisation, not grade
// a right one to the last ulp.
//
// Generous, but measured rather than picked. Lowering all three until the
// suite breaks puts autonne's own output through at 16 and not at 4, so the
// default sits about four times above what the kernel needs -- loose enough
// that rounding on a 128 x 128 case cannot trip it, tight enough that the
// deliberate perturbations in the test suite (a singular value moved by 1e-9,
// a column scaled by 1.5) are still rejected by a wide margin. A caller who
// wants a stricter check can tighten these; one who has to loosen them past
// the default should suspect the factorisation rather than the bound.
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
// Thin / truncated SVD, screened
// ---------------------------------------------------------------------------

struct SvdScreenReport {
  // Shape actually screened.
  int rows = 0;
  int cols = 0;
  int k = 0;
  bool truncated = false;  // k < min(rows, cols)
  int probes = 0;          // probe vectors drawn per estimate

  // Measured quantities. The estimates stand in for the harness's norms:
  // each is the worst over the probes, and each is compared with the same
  // bound check_svd applies to the norm it estimates.
  double norm_M = 0.0;             // ||M||_F
  double discarded_energy = 0.0;   // max(0, ||M||_F^2 - sum s_i^2)
  double energy_defect = 0.0;      // sum s_i^2 - ||M||_F^2
  double energy_bound = 0.0;
  double residual_estimate = 0.0;  // estimates max(||M V_k - U_k S_k||_F, ||M^* U_k - V_k S_k||_F)
  double backward_bound = 0.0;     // backward_factor * max(rows, cols) * eps * ||M||_F
  double u_ortho_estimate = 0.0;   // estimates ||U^* U - I||_F
  double v_ortho_estimate = 0.0;   // estimates ||V^* V - I||_F
  double ortho_bound = 0.0;

  // Verdicts.
  bool inputs_valid = false;
  bool finite = false;       // no NaN/Inf across the kept slice of U, S, V
  bool nonnegative = false;  // s_i >= 0
  bool descending = false;   // s_i >= s_{i+1}
  bool energy_ok = false;
  bool backward_ok = false;
  bool u_orthonormal = false;
  bool v_orthonormal = false;

  constexpr bool ok() const noexcept {
    return inputs_valid && finite && nonnegative && descending && energy_ok &&
           backward_ok && u_orthonormal && v_orthonormal;
  }
};

// The harness's checks at a fraction of its cost, for a caller who cannot
// afford check_svd's O(rows cols k) on every factorisation. Same arguments
// as check_svd; `probes` is the number of probe vectors behind each
// estimate.
//
// What is measured exactly, as in check_svd: the input validation, the
// bit-pattern scan of U, S and V, the ordering and sign of S, and the
// energy identity sum s_i^2 == ||M||_F^2 (one-sided when truncated), which
// costs one pass over M and is the check that catches a wrong singular
// value. What is estimated: the residual and the orthonormality defects
// ||U^* U - I||_F and ||V^* V - I||_F, each through probe vectors y with
// entries drawn from {1, -1, i, -i}. For such a y, E ||X y||^2 = ||X||_F^2
// for any X, so ||X y|| is an unbiased estimate of a Frobenius norm at the
// cost of matrix-vector products instead of matrix products.
//
// The residual is probed in the kept subspaces: M (V_k y) against U_k (S_k y)
// and M^* (U_k y) against V_k (S_k y). A correct factorisation has
// M V_k = U_k S_k and M^* U_k = V_k S_k to rounding whatever k is, so the
// truncated part of M never enters and no sqrt(discarded) term is needed,
// where the whole residual ||(M - U_k S_k V_k^*) x|| would carry the
// discarded energy with a spread of order one for a low-rank discard. When
// k = min(rows, cols) one of V and U is square unitary and the matching
// probe equals ||R x|| for the harness's own R = M - U S V^*, so the
// estimate is of the harness's residual. The estimate is compared with
// backward_factor * max(rows, cols) * eps * ||M||_F, the error term of
// check_svd's bound, and the worst probe decides. The whole screen is
// O(rows cols + (rows + cols) k) per probe.
//
// What that buys and what it costs. A defect the harness would reject is
// missed only when every probe lands well under the norm it estimates. For
// a defect aligned with a coordinate direction that cannot happen at all,
// since |y_j| = 1 for every entry; for a defect spread over many directions
// the estimate is close to Gaussian about the truth, and one probe lands
// under a tenth of it about one time in a hundred, so three probes miss at
// about one in a million and `probes` is the knob for a caller who wants
// less. In the other direction the estimate's own variance sits inside the
// margin the harness leaves (its bounds are about four times what autonne's
// output needs), so a factorisation the harness accepts is not rejected here
// in practice. The probe vectors come from a fixed seed, so the same inputs
// give the same report on every platform. The screen is no more able than
// the harness to see a loss of relative accuracy in the small singular
// values: both measure against ||M||.
//
// The report's estimates are at the scale of the input, as check_svd's
// norms are. M is not scanned for non-finite entries, as in check_svd; one
// makes the residual estimate NaN and the verdict false.
SvdScreenReport screen_svd(const std::complex<double>* M, int rows, int cols,
                           MatrixOrder order, const std::complex<double>* U,
                           const double* S, const std::complex<double>* V, int k,
                           const Tolerances& tol = Tolerances(), int probes = 3);

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
