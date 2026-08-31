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

#include "autonne/autonne.hpp"

#include "autonne/detail/matrix_view.hpp"

namespace autonne {

// Not implemented yet. Returning false is the documented "take your fallback
// path" answer, so callers written against this build are already on the
// contract the real kernel will honour; nothing is written to the outputs.
bool svd_thin(const std::complex<double>* /*data*/, int /*rows*/, int /*cols*/,
              MatrixOrder /*order*/, std::complex<double>* /*U_out*/,
              double* /*S_out*/, std::complex<double>* /*V_out*/) {
  return false;
}

}  // namespace autonne
