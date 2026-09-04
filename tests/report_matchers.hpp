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

// GoogleTest predicates over the harness reports and over spectra. A failure
// prints every measurement, so a regression says which bound moved rather
// than just "expected true".

#ifndef AUTONNE_TESTS_REPORT_MATCHERS_HPP
#define AUTONNE_TESTS_REPORT_MATCHERS_HPP

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "autonne/verify.hpp"

namespace autonne_test {

inline ::testing::AssertionResult SvdAccepted(const autonne::verify::SvdReport& r) {
  if (r.ok()) return ::testing::AssertionSuccess();
  return ::testing::AssertionFailure()
         << "check_svd rejected the factorisation:"
         << "\n  inputs_valid   = " << r.inputs_valid
         << "\n  finite         = " << r.finite
         << "\n  nonnegative    = " << r.nonnegative
         << "\n  descending     = " << r.descending
         << "\n  backward_ok    = " << r.backward_ok << "  (residual "
         << r.residual << " vs bound " << r.backward_bound << ")"
         << "\n  u_orthonormal  = " << r.u_orthonormal << "  (" << r.u_ortho_residual
         << " vs " << r.ortho_bound << ")"
         << "\n  v_orthonormal  = " << r.v_orthonormal << "  (" << r.v_ortho_residual
         << " vs " << r.ortho_bound << ")"
         << "\n  energy_ok      = " << r.energy_ok << "  (defect "
         << r.energy_defect << " vs bound " << r.energy_bound << ")";
}

inline ::testing::AssertionResult EighAccepted(const autonne::verify::EighReport& r) {
  if (r.ok()) return ::testing::AssertionSuccess();
  return ::testing::AssertionFailure()
         << "check_eigh rejected the decomposition:"
         << "\n  inputs_valid    = " << r.inputs_valid
         << "\n  input_hermitian = " << r.input_hermitian << "  ("
         << r.hermitian_residual << " vs " << r.hermitian_bound << ")"
         << "\n  finite          = " << r.finite
         << "\n  ascending       = " << r.ascending
         << "\n  backward_ok     = " << r.backward_ok << "  (" << r.residual
         << " vs " << r.backward_bound << ")"
         << "\n  q_orthonormal   = " << r.q_orthonormal << "  ("
         << r.q_ortho_residual << " vs " << r.ortho_bound << ")"
         << "\n  trace_ok        = " << r.trace_ok << "  (" << r.trace_defect
         << " vs " << r.trace_bound << ")"
         << "\n  energy_ok       = " << r.energy_ok << "  (" << r.energy_defect
         << " vs " << r.energy_bound << ")";
}

// Element-wise |got[i] - expected[i]| <= abs_tol after sorting `expected`
// into the order the kernel promises (`descending` true for singular values,
// false for eigenvalues). `got` is compared as given, so an unsorted kernel
// output fails here as well as in the harness.
inline ::testing::AssertionResult SpectrumClose(const std::vector<double>& got,
                                                std::vector<double> expected,
                                                double abs_tol, bool descending) {
  if (descending) {
    std::sort(expected.begin(), expected.end(), [](double a, double b) { return a > b; });
  } else {
    std::sort(expected.begin(), expected.end());
  }
  if (got.size() != expected.size()) {
    return ::testing::AssertionFailure()
           << "spectrum has " << got.size() << " values, expected " << expected.size();
  }
  for (std::size_t i = 0; i < got.size(); ++i) {
    const double diff = std::fabs(got[i] - expected[i]);
    if (!(diff <= abs_tol)) {
      return ::testing::AssertionFailure()
             << "value " << i << ": got " << got[i] << ", expected " << expected[i]
             << ", |diff| = " << diff << " > " << abs_tol;
    }
  }
  return ::testing::AssertionSuccess();
}

// Element-wise got[i] in [lo * ref[i], hi * ref[i]] after sorting `ref` the
// way the kernel sorts. Used for relative-accuracy claims on graded inputs.
inline ::testing::AssertionResult SpectrumWithinFactor(const std::vector<double>& got,
                                                       std::vector<double> ref,
                                                       double lo, double hi,
                                                       bool descending) {
  if (descending) {
    std::sort(ref.begin(), ref.end(), [](double a, double b) { return a > b; });
  } else {
    std::sort(ref.begin(), ref.end());
  }
  if (got.size() != ref.size()) {
    return ::testing::AssertionFailure()
           << "spectrum has " << got.size() << " values, expected " << ref.size();
  }
  for (std::size_t i = 0; i < got.size(); ++i) {
    const double low = lo * ref[i];
    const double high = hi * ref[i];
    if (!(got[i] >= low && got[i] <= high)) {
      return ::testing::AssertionFailure()
             << "value " << i << ": got " << got[i] << ", reference " << ref[i]
             << ", allowed [" << low << ", " << high << "]";
    }
  }
  return ::testing::AssertionSuccess();
}

}  // namespace autonne_test

#endif  // AUTONNE_TESTS_REPORT_MATCHERS_HPP
