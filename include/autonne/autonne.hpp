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
// Callers pass raw buffers; autonne owns no memory and allocates nothing the
// caller can observe.

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
// Returns false if the factorisation could not be produced. On false the
// caller must take its fallback path and must not read U_out, S_out or V_out;
// their contents are unspecified. Failure is reported by the return value
// only: this function never throws.
bool svd_thin(const std::complex<double>* data, int rows, int cols,
              MatrixOrder order,
              std::complex<double>* U_out, double* S_out,
              std::complex<double>* V_out);

// Eigendecomposition of a dense Hermitian complex matrix.
//
//   data       n x n, laid out according to `order`; only Hermitian input is
//              meaningful, and the caller is responsible for supplying it
//   evals_out  n real eigenvalues in ascending order
//   evecs_out  n x n, column-major; column j is the eigenvector for
//              evals_out[j]
//
// so that
//
//   data * evecs_out == evecs_out * diag(evals_out)
//
// Returns false if the decomposition could not be produced. On false the
// caller must take its fallback path and must not read evals_out or
// evecs_out. Failure is reported by the return value only: this function
// never throws.
bool eigh(const std::complex<double>* data, int n, MatrixOrder order,
          double* evals_out, std::complex<double>* evecs_out);

}  // namespace autonne

#endif  // AUTONNE_AUTONNE_HPP
