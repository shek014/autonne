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

// Public interface of autonne.
//
// Nothing here names a type that is not either a builtin or from <complex>.
// Callers pass raw buffers; autonne owns no memory that outlives a call, and
// keeps no state between calls.
//
// Both functions are reentrant and safe to call concurrently on distinct
// buffers: there is no global or static mutable state, and every working
// array is allocated for the duration of the call. Concurrent calls sharing
// an output buffer race, as they would anywhere. Neither function reads or
// writes the floating-point environment: no rounding mode is changed, and no
// exception flag is consulted, so a caller's fenv settings survive a call
// unchanged.

#ifndef AUTONNE_AUTONNE_HPP
#define AUTONNE_AUTONNE_HPP

#include <complex>

namespace autonne {

// Storage order of a caller-supplied input matrix.
enum class MatrixOrder {
  RowMajor,
  ColMajor,
};

// Thin singular value decomposition of a dense complex matrix.
//
//   data   rows x cols, laid out according to `order`
//   k      = min(rows, cols)
//   U_out  rows x k, column-major, orthonormal columns
//   S_out  k singular values, non-negative, descending
//   V_out  cols x k, column-major, orthonormal columns
//
// V_out holds V itself, not V conjugate-transposed, so the reconstruction is
//
//   data == U_out * diag(S_out) * V_out^*
//
// Rows and columns of `data` that are exactly zero are treated as structural:
// the singular values they account for are exactly zero and their singular
// vectors are canonical basis vectors. The kernel scales its input by a power
// of two before any arithmetic, so scaling `data` by a power of two scales
// S_out by exactly that power and leaves U_out and V_out bit for bit unchanged.
//
// Returns false if the factorisation could not be produced: a null pointer, a
// non-positive dimension, a non-finite entry in `data`, an allocation failure
// or (in principle) a failure to converge. On false nothing has been written
// to U_out, S_out or V_out, and the caller must take its fallback path.
// Failure is reported by the return value only: this function never throws.
bool svd_thin(const std::complex<double>* data, int rows, int cols,
              MatrixOrder order,
              std::complex<double>* U_out, double* S_out,
              std::complex<double>* V_out);

// Eigendecomposition of a dense Hermitian complex matrix.
//
//   data       n x n, laid out according to `order`
//   evals_out  n real eigenvalues in ascending order
//   evecs_out  n x n, column-major; column j is the eigenvector for
//              evals_out[j]. May be null when only the spectrum is wanted;
//              the eigenvalues are then identical to those of the full call.
//
// so that
//
//   data * evecs_out == evecs_out * diag(evals_out)
//
// What is decomposed is the Hermitian part (data + data^*) / 2, with the
// imaginary parts of the diagonal discarded. For Hermitian input that is the
// input itself, bit for bit, in either storage order; for input that is
// Hermitian up to rounding it is the nearest Hermitian matrix in the Frobenius
// norm. A grossly non-Hermitian input is not rejected here -- the caller's
// verification (verify::check_eigh reports `input_hermitian`) is where that
// is caught.
//
// Rows and columns that are exactly zero are structural: their eigenvalue is
// exactly zero and their eigenvector is the canonical basis vector.
//
// Returns false if the decomposition could not be produced: a null `data` or
// `evals_out`, a non-positive n, a non-finite entry, an allocation failure or
// (in principle) a failure to converge. On false nothing has been written to
// evals_out or evecs_out, and the caller must take its fallback path. Failure
// is reported by the return value only: this function never throws.
bool eigh(const std::complex<double>* data, int n, MatrixOrder order,
          double* evals_out, std::complex<double>* evecs_out);

}  // namespace autonne

#endif  // AUTONNE_AUTONNE_HPP
