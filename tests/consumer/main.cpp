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

// The smallest honest consumer: factor a matrix through the installed
// library, verify it through the installed harness, and hand a NaN to the
// kernel from a translation unit that was itself built with -ffast-math.
// Exit status is the verdict.

#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include <autonne/autonne.hpp>
#include <autonne/verify.hpp>

int main() {
  using Complex = std::complex<double>;
  const int rows = 3;
  const int cols = 2;
  // Column-major [[1, 2], [3i, 4], [5, 6i]].
  const std::vector<Complex> m = {Complex(1, 0), Complex(0, 3), Complex(5, 0),
                                  Complex(2, 0), Complex(4, 0), Complex(0, 6)};
  std::vector<Complex> u(6);
  std::vector<double> s(2);
  std::vector<Complex> v(4);
  if (!autonne::svd_thin(m.data(), rows, cols, autonne::MatrixOrder::ColMajor,
                         u.data(), s.data(), v.data())) {
    std::puts("svd_thin refused a valid matrix");
    return 1;
  }
  const autonne::verify::SvdReport r = autonne::verify::check_svd(
      m.data(), rows, cols, autonne::MatrixOrder::ColMajor, u.data(), s.data(),
      v.data(), 2);
  if (!r.ok()) {
    std::puts("harness rejected the factorisation");
    return 2;
  }

  // A NaN written as bits, from a fast-math translation unit, must be refused.
  std::vector<Complex> bad = m;
  const std::uint64_t nan_bits = UINT64_C(0x7FF8000000000000);
  std::memcpy(&bad[4], &nan_bits, sizeof nan_bits);
  if (autonne::svd_thin(bad.data(), rows, cols, autonne::MatrixOrder::ColMajor,
                        u.data(), s.data(), v.data())) {
    std::puts("svd_thin accepted a NaN");
    return 3;
  }

  std::printf("ok: s = %.17g %.17g\n", s[0], s[1]);
  return 0;
}
