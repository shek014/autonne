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

// Hermitian eigendecomposition by cyclic Jacobi.
//
//   1. Reject non-finite input by bit pattern.
//   2. Form H = (A + A^*) / 2 with a real diagonal. For Hermitian input this
//      is the input, bit for bit; otherwise it is the nearest Hermitian matrix
//      in the Frobenius norm, which is what a caller holding a rounded product
//      means.
//   3. Drop rows (hence columns) that are exactly zero: their eigenvalue is
//      exactly zero and their eigenvector is a canonical basis vector.
//   4. Scale by a power of two so the largest component is in [0.5, 1).
//   5. Sweep over pairs (p, q), applying the unitary similarity that zeroes
//      H(p, q) whenever |H(p, q)| exceeds sqrt(n) u sqrt(|H(p, p) H(q, q)|).
//      The threshold is relative to the diagonal, which is what gives small
//      eigenvalues of a graded positive definite matrix their relative
//      accuracy (Demmel and Veselic 1992). The rotations accumulate into Q.
//   6. Sort ascending, undo the scaling and the compression, scan the result
//      by bit pattern, write.
//
// The matrix is kept exactly Hermitian throughout: after a rotation the row
// entries are set to the conjugates of the freshly computed column entries,
// and the diagonal is carried as real numbers.

#include "autonne/autonne.hpp"

#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <vector>

#include "autonne/detail/fp_bits.hpp"
#include "autonne/detail/matrix_view.hpp"
#include "detail/kernel_common.hpp"

namespace autonne {

namespace {

using detail::kernel::Complex;
using detail::kernel::Index;
using detail::kernel::Rotation;
using detail::kernel::at;

// Sweep budget. Cyclic Jacobi converges quadratically once the off-diagonal
// mass is small, so the count is low and grows slowly with n: lowering this
// until the suite breaks puts the hardest case in the corpus and the stress
// sweeps at twelve sweeps (a 128 x 128 random matrix, and a 96 x 96 with a
// three-level degenerate spectrum). Sixty is therefore a five-fold margin,
// not a guess. Exhausting it is reported as failure rather than papered over,
// which is what the false return in the contract is for.
constexpr int kMaxSweeps = 60;

// Off-diagonal Frobenius norm squared and diagonal sum of squares, for the
// fallback acceptance test.
double off_norm_sq(const std::vector<Complex>& h, Index n) {
  double acc = 0.0;
  for (Index j = 0; j < n; ++j) {
    for (Index i = 0; i < n; ++i) {
      if (i == j) continue;
      const Complex& z = h[at(i, j, n)];
      acc += z.real() * z.real() + z.imag() * z.imag();
    }
  }
  return acc;
}

// Diagonalises h (n x n, exactly Hermitian, diagonal carried in d) in place.
// q, when non-null, accumulates the rotations (n x n, starts as identity).
bool jacobi_diagonalise(std::vector<Complex>& h, std::vector<double>& d, Index n,
                        std::vector<Complex>* q) {
  const double tol = std::sqrt(static_cast<double>(n)) * detail::kernel::unit_roundoff();

  for (int sweep = 0; sweep < kMaxSweeps; ++sweep) {
    Index rotations = 0;
    for (Index p = 0; p + 1 < n; ++p) {
      for (Index q_ = p + 1; q_ < n; ++q_) {
        const Complex g = h[at(p, q_, n)];
        const double g_abs = detail::kernel::modulus(g);
        if (g_abs == 0.0) continue;
        const double threshold = tol * std::sqrt(std::fabs(d[p])) * std::sqrt(std::fabs(d[q_]));
        if (!(g_abs > threshold)) continue;

        const Rotation r = detail::kernel::make_rotation(d[p], d[q_], g);

        // Columns p and q of every row other than p and q, then mirror.
        const double c = r.c;
        const double s_ = r.s;
        const double pr = r.phase.real();
        const double pi = r.phase.imag();
        for (Index k = 0; k < n; ++k) {
          if (k == p || k == q_) continue;
          const double ar = h[at(k, p, n)].real();
          const double ai = h[at(k, p, n)].imag();
          const double qr = h[at(k, q_, n)].real();
          const double qi = h[at(k, q_, n)].imag();
          const double br = pr * qr - pi * qi;
          const double bi = pr * qi + pi * qr;
          const double npr = c * ar - s_ * br;
          const double npi = c * ai - s_ * bi;
          const double nqr = s_ * ar + c * br;
          const double nqi = s_ * ai + c * bi;
          h[at(k, p, n)] = Complex(npr, npi);
          h[at(k, q_, n)] = Complex(nqr, nqi);
          h[at(p, k, n)] = Complex(npr, -npi);
          h[at(q_, k, n)] = Complex(nqr, -nqi);
        }
        d[p] -= r.t * r.gamma_abs;
        d[q_] += r.t * r.gamma_abs;
        h[at(p, q_, n)] = Complex(0.0, 0.0);
        h[at(q_, p, n)] = Complex(0.0, 0.0);

        if (q != nullptr) {
          detail::kernel::rotate_columns(&(*q)[at(0, p, n)], &(*q)[at(0, q_, n)], n, r);
        }
        ++rotations;
      }
    }
    if (rotations == 0) return true;
  }

  // The relative threshold was not met within the sweep budget. Accept if the
  // remaining off-diagonal mass is at the level of rounding in the whole
  // matrix, which is all a backward-stable method promises anyway.
  double diag_sq = 0.0;
  for (Index i = 0; i < n; ++i) diag_sq += d[i] * d[i];
  const double off_sq = off_norm_sq(h, n);
  const double total = std::sqrt(diag_sq + off_sq);
  return std::sqrt(off_sq) <= detail::kernel::unit_roundoff() * total;
}

bool eigh_impl(const Complex* data, Index n, MatrixOrder order, double* evals_out,
               Complex* evecs_out) {
  if (detail::any_bad(data, static_cast<int>(n * n))) return false;
  const auto in = detail::ordered(data, static_cast<int>(n), static_cast<int>(n), order);

  // Hermitian part, real diagonal.
  std::vector<Complex> a(n * n);
  for (int j = 0; j < static_cast<int>(n); ++j) {
    for (int i = 0; i < static_cast<int>(n); ++i) {
      const Complex z = detail::at(in, i, j);
      const Complex w = std::conj(detail::at(in, j, i));
      a[at(static_cast<Index>(i), static_cast<Index>(j), n)] =
          (i == j) ? Complex(z.real(), 0.0) : 0.5 * (z + w);
    }
  }

  // Structural zeros.
  std::vector<bool> live(n, false);
  for (Index j = 0; j < n; ++j) {
    for (Index i = 0; i < n; ++i) {
      const Complex& z = a[at(i, j, n)];
      if (z.real() != 0.0 || z.imag() != 0.0) live[i] = true;
    }
  }
  std::vector<Index> kept;
  for (Index i = 0; i < n; ++i) {
    if (live[i]) kept.push_back(i);
  }
  const Index nc = kept.size();

  std::vector<double> core_evals;      // ascending
  std::vector<Complex> core_evecs;     // nc x nc, or empty if not wanted

  if (nc > 0) {
    std::vector<Complex> h(nc * nc);
    std::vector<double> d(nc);
    for (Index j = 0; j < nc; ++j) {
      for (Index i = 0; i < nc; ++i) h[at(i, j, nc)] = a[at(kept[i], kept[j], n)];
      d[j] = h[at(j, j, nc)].real();
      h[at(j, j, nc)] = Complex(0.0, 0.0);
    }

    const double max_comp_off = detail::kernel::max_component(h.data(), nc * nc);
    double max_comp = max_comp_off;
    for (Index i = 0; i < nc; ++i) {
      if (std::fabs(d[i]) > max_comp) max_comp = std::fabs(d[i]);
    }
    const int exponent = detail::kernel::scaling_exponent(max_comp);
    detail::kernel::scale_in_place(h.data(), nc * nc, -exponent);
    for (Index i = 0; i < nc; ++i) d[i] = std::ldexp(d[i], -exponent);

    std::vector<Complex> q;
    if (evecs_out != nullptr) {
      q.assign(nc * nc, Complex(0.0, 0.0));
      for (Index j = 0; j < nc; ++j) q[at(j, j, nc)] = Complex(1.0, 0.0);
    }
    if (!jacobi_diagonalise(h, d, nc, evecs_out != nullptr ? &q : nullptr)) return false;

    std::vector<Index> ascending;
    detail::kernel::sort_indices(d.data(), nc, false, ascending);
    core_evals.resize(nc);
    for (Index j = 0; j < nc; ++j) core_evals[j] = std::ldexp(d[ascending[j]], exponent);
    if (evecs_out != nullptr) {
      core_evecs.resize(nc * nc);
      detail::kernel::permute_columns(q.data(), core_evecs.data(), nc, ascending);
    }
  }

  // Merge the structural zeros into ascending position: after every negative
  // value and before every positive one.
  const Index zeros = n - nc;
  Index negatives = 0;
  while (negatives < nc && core_evals[negatives] < 0.0) ++negatives;

  std::vector<double> evals(n);
  std::vector<Complex> evecs;
  if (evecs_out != nullptr) evecs.assign(n * n, Complex(0.0, 0.0));

  Index out = 0;
  auto place_core = [&](Index c) {
    evals[out] = core_evals[c];
    if (evecs_out != nullptr) {
      for (Index i = 0; i < nc; ++i) evecs[at(kept[i], out, n)] = core_evecs[at(i, c, nc)];
    }
    ++out;
  };
  for (Index c = 0; c < negatives; ++c) place_core(c);
  for (Index i = 0; i < n && zeros > 0; ++i) {
    if (live[i]) continue;
    evals[out] = 0.0;
    if (evecs_out != nullptr) evecs[at(i, out, n)] = Complex(1.0, 0.0);
    ++out;
  }
  for (Index c = negatives; c < nc; ++c) place_core(c);

  if (detail::any_bad(evals.data(), static_cast<int>(n))) return false;
  if (evecs_out != nullptr && detail::any_bad(evecs.data(), static_cast<int>(n * n))) return false;
  for (Index i = 0; i < n; ++i) evals_out[i] = evals[i];
  if (evecs_out != nullptr) {
    for (Index i = 0; i < n * n; ++i) evecs_out[i] = evecs[i];
  }
  return true;
}

}  // namespace

bool eigh(const std::complex<double>* data, int n, MatrixOrder order,
          double* evals_out, std::complex<double>* evecs_out) {
  if (data == nullptr || n <= 0 || evals_out == nullptr) return false;
  if (static_cast<unsigned long long>(n) * static_cast<unsigned long long>(n) >
      static_cast<unsigned long long>(std::numeric_limits<int>::max())) {
    return false;
  }
  try {
    return eigh_impl(data, static_cast<Index>(n), order, evals_out, evecs_out);
  } catch (...) {
    return false;
  }
}

}  // namespace autonne
