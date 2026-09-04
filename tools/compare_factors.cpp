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

// Compares two dumps written by autonne_dump, one per floating-point model.
//
// What is compared, and why not bit equality: -ffast-math permits
// reassociation, so the two builds may sum the same inner products in
// different orders and land a few ulps apart. Demanding bit equality would
// therefore fail for a reason that says nothing about correctness. What must
// hold is that the two builds agree to the accuracy either one claims. The
// spectra are compared in absolute terms against a tolerance scaled by the
// largest value, and the singular vectors are compared as subspaces --
// column by column up to a phase, since a factorisation is only ever
// determined up to the phase of each column, and a degenerate spectrum not
// even that far.
//
// Columns whose singular value is part of a degenerate group are not
// compared individually: any unitary mixing within the group is a correct
// answer, so a difference there is not a defect. The report says how many
// columns were skipped for that reason.
//
// Exit status is zero when every comparison is inside tolerance.
//
// Usage: autonne_compare <dump-a> <dump-b> [--label NAME]

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "autonne/hexfloat.hpp"

namespace {

using Complex = std::complex<double>;

// Generous relative to the 2^-53 of a single rounding, tight enough that a
// real divergence between the two models cannot hide inside it.
constexpr double kRelativeTolerance = 1e-11;

struct Dump {
  autonne::hexfloat::Vector spectrum;
  std::vector<autonne::hexfloat::Matrix> matrices;
};

bool read_dump(const std::string& path, Dump& out) {
  std::ifstream in(path);
  if (!in) {
    std::fprintf(stderr, "autonne_compare: cannot open %s\n", path.c_str());
    return false;
  }
  if (!autonne::hexfloat::read_vector(in, out.spectrum)) {
    std::fprintf(stderr, "autonne_compare: no leading vector in %s\n", path.c_str());
    return false;
  }
  autonne::hexfloat::Matrix m;
  while (autonne::hexfloat::read_matrix(in, m)) out.matrices.push_back(m);
  return true;
}

double largest(const std::vector<double>& v) {
  double m = 0.0;
  for (const double x : v) m = std::max(m, std::fabs(x));
  return m;
}

// Distance between two unit columns up to a global phase:
// min over theta of || a - e^{i theta} b || = sqrt(2 - 2 |<a, b>|).
double phase_free_distance(const Complex* a, const Complex* b, std::size_t n) {
  Complex dot(0.0, 0.0);
  for (std::size_t i = 0; i < n; ++i) dot += std::conj(a[i]) * b[i];
  const double overlap = std::abs(dot);
  const double squared = 2.0 - 2.0 * std::min(overlap, 1.0);
  return std::sqrt(std::max(squared, 0.0));
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr,
                 "usage: autonne_compare <dump-a> <dump-b> [--label NAME]\n");
    return 2;
  }
  std::string label = "comparison";
  for (int i = 3; i + 1 < argc; ++i) {
    if (std::strcmp(argv[i], "--label") == 0) label = argv[i + 1];
  }

  Dump a;
  Dump b;
  if (!read_dump(argv[1], a) || !read_dump(argv[2], b)) return 2;

  if (a.spectrum.data.size() != b.spectrum.data.size() ||
      a.matrices.size() != b.matrices.size()) {
    std::fprintf(stderr, "%s: the two dumps have different shapes\n", label.c_str());
    return 1;
  }

  const std::size_t k = a.spectrum.data.size();
  const double scale = std::max(largest(a.spectrum.data), largest(b.spectrum.data));
  const double spectrum_tolerance = kRelativeTolerance * (scale > 0.0 ? scale : 1.0);

  double worst_spectrum = 0.0;
  for (std::size_t i = 0; i < k; ++i) {
    worst_spectrum = std::max(
        worst_spectrum, std::fabs(a.spectrum.data[i] - b.spectrum.data[i]));
  }

  // A column may be compared on its own only when its value is separated from
  // its neighbours: within a degenerate group any unitary mixing is correct.
  const double separation = 1e-8 * (scale > 0.0 ? scale : 1.0);
  std::vector<bool> comparable(k, true);
  for (std::size_t i = 0; i < k; ++i) {
    const double value = a.spectrum.data[i];
    if (i > 0 && std::fabs(a.spectrum.data[i - 1] - value) <= separation) comparable[i] = false;
    if (i + 1 < k && std::fabs(a.spectrum.data[i + 1] - value) <= separation) comparable[i] = false;
    // A value at the level of the tolerance carries no direction worth
    // comparing: its column spans the null space, which is arbitrary.
    if (std::fabs(value) <= spectrum_tolerance) comparable[i] = false;
  }

  double worst_vector = 0.0;
  std::size_t compared = 0;
  std::size_t skipped = 0;
  for (std::size_t mi = 0; mi < a.matrices.size(); ++mi) {
    const autonne::hexfloat::Matrix& ma = a.matrices[mi];
    const autonne::hexfloat::Matrix& mb = b.matrices[mi];
    if (ma.rows != mb.rows || ma.cols != mb.cols) {
      std::fprintf(stderr, "%s: matrix %zu differs in shape\n", label.c_str(), mi);
      return 1;
    }
    const std::size_t rows = static_cast<std::size_t>(ma.rows);
    const std::size_t cols = static_cast<std::size_t>(ma.cols);
    for (std::size_t c = 0; c < cols; ++c) {
      if (c < k && !comparable[c]) {
        ++skipped;
        continue;
      }
      const double d = phase_free_distance(&ma.data[c * rows], &mb.data[c * rows], rows);
      worst_vector = std::max(worst_vector, d);
      ++compared;
    }
  }

  const bool spectrum_ok = worst_spectrum <= spectrum_tolerance;
  const bool vectors_ok = worst_vector <= std::sqrt(kRelativeTolerance);

  std::printf(
      "%-28s spectrum %8.2e (bound %8.2e)  vectors %8.2e over %zu columns "
      "(%zu skipped as degenerate or null)  %s\n",
      label.c_str(), worst_spectrum, spectrum_tolerance, worst_vector, compared,
      skipped, (spectrum_ok && vectors_ok) ? "ok" : "DIVERGED");

  return (spectrum_ok && vectors_ok) ? 0 : 1;
}
