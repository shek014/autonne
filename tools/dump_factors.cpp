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

// Factors one corpus matrix and writes the factors out as exact hex floats.
//
// Built twice, once per floating-point model, into two separate executables.
// That separation is the whole point. A comparison made inside one binary
// cannot measure what a compile flag does to this code: the kernels' inline
// helpers have vague linkage, the linker keeps one copy of each per binary
// without regard to the flags it was built under, and a suite comparing two
// "modes" in one process would be reading link order rather than floating
// point. Two executables, one input held fixed as a hex-float literal, and a
// diff of the outputs is the only arrangement that measures the flag.
//
// Usage: autonne_dump <corpus-file> <output-file> [svd|eigh]

#include <complex>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "autonne/autonne.hpp"
#include "autonne/hexfloat.hpp"

namespace {

int fail(const char* what, const char* detail) {
  std::fprintf(stderr, "autonne_dump: %s: %s\n", what, detail);
  return 1;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    return fail("usage", "autonne_dump <corpus-file> <output-file> [svd|eigh]");
  }
  const std::string in_path = argv[1];
  const std::string out_path = argv[2];
  const bool want_eigh = (argc > 3 && std::strcmp(argv[3], "eigh") == 0);

  std::ifstream in(in_path);
  if (!in) return fail("cannot open", in_path.c_str());
  autonne::hexfloat::Matrix m;
  if (!autonne::hexfloat::read_matrix(in, m)) return fail("bad matrix", in_path.c_str());

  std::ofstream out(out_path);
  if (!out) return fail("cannot write", out_path.c_str());

  if (want_eigh) {
    if (m.rows != m.cols) return fail("eigh needs a square matrix", in_path.c_str());
    const int n = m.rows;
    std::vector<double> evals(static_cast<std::size_t>(n));
    std::vector<std::complex<double>> evecs(static_cast<std::size_t>(n) *
                                            static_cast<std::size_t>(n));
    if (!autonne::eigh(m.data.data(), n, m.order, evals.data(), evecs.data())) {
      return fail("eigh refused", in_path.c_str());
    }
    if (!autonne::hexfloat::write_vector(out, evals.data(), n, "evals")) {
      return fail("write failed", "evals");
    }
    if (!autonne::hexfloat::write_matrix(out, evecs.data(), n, n,
                                         autonne::MatrixOrder::ColMajor, "evecs")) {
      return fail("write failed", "evecs");
    }
    return 0;
  }

  const int rows = m.rows;
  const int cols = m.cols;
  const int k = rows < cols ? rows : cols;
  std::vector<std::complex<double>> u(static_cast<std::size_t>(rows) *
                                      static_cast<std::size_t>(k));
  std::vector<double> s(static_cast<std::size_t>(k));
  std::vector<std::complex<double>> v(static_cast<std::size_t>(cols) *
                                      static_cast<std::size_t>(k));
  if (!autonne::svd_thin(m.data.data(), rows, cols, m.order, u.data(), s.data(),
                         v.data())) {
    return fail("svd_thin refused", in_path.c_str());
  }
  if (!autonne::hexfloat::write_vector(out, s.data(), k, "S")) {
    return fail("write failed", "S");
  }
  if (!autonne::hexfloat::write_matrix(out, u.data(), rows, k,
                                       autonne::MatrixOrder::ColMajor, "U")) {
    return fail("write failed", "U");
  }
  if (!autonne::hexfloat::write_matrix(out, v.data(), cols, k,
                                       autonne::MatrixOrder::ColMajor, "V")) {
    return fail("write failed", "V");
  }
  return 0;
}
