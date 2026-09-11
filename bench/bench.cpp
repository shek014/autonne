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

// autonne against Eigen, on the matrices the spec describes.
//
// For each size 2*bond (bond = 4 .. 64) and each spectrum shape, the median
// wall time over repeated calls of autonne::svd_thin, Eigen::JacobiSVD and
// Eigen::BDCSVD (thin U and V), and of autonne::eigh against
// Eigen::SelfAdjointEigenSolver. Every factorisation is also passed through
// the verification harness, and the worst ratio of measured residual to
// permitted bound is reported, because a fast wrong answer is not a result.
//
// Built strict (-fno-fast-math): Eigen under -ffast-math is the failure the
// project exists to avoid, not something to measure here.
//
// Usage: autonne_bench [--repeats N] [--csv]

#include <Eigen/Dense>
#include <Eigen/SVD>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "autonne/autonne.hpp"
#include "autonne/verify.hpp"
#include "test_support.hpp"

namespace {

using Complex = std::complex<double>;
using Clock = std::chrono::steady_clock;

struct Timing {
  double median_ms = 0.0;
  double worst_ratio = 0.0;  // max over runs of residual / bound (SVD: backward; eigh: backward)
  int rejected = 0;          // harness rejections
  int failed = 0;            // backend reported failure
};

// Spectrum shapes for a 2b x 2b theta with bond dimension b.
std::vector<double> spectrum(const std::string& shape, int n) {
  std::vector<double> s(static_cast<std::size_t>(n), 0.0);
  if (shape == "flat") {
    for (double& x : s) x = 1.0;
  } else if (shape == "decaying") {
    // Geometric decay over sixteen decades, the profile of a well-behaved
    // entangled state.
    for (int i = 0; i < n; ++i) {
      s[static_cast<std::size_t>(i)] = std::pow(10.0, -16.0 * static_cast<double>(i) / static_cast<double>(n - 1 > 0 ? n - 1 : 1));
    }
  } else if (shape == "rank-deficient") {
    // Half the spectrum degenerate, the other half exactly zero.
    for (int i = 0; i < n / 2; ++i) s[static_cast<std::size_t>(i)] = 1.0;
  }
  double norm = 0.0;
  for (const double x : s) norm += x * x;
  norm = std::sqrt(norm);
  for (double& x : s) x /= norm;
  return s;
}

template <typename Run>
Timing time_it(int repeats, Run run) {
  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(repeats));
  Timing t;
  for (int r = 0; r < repeats; ++r) {
    const auto start = Clock::now();
    const bool ok = run();
    const auto end = Clock::now();
    if (!ok) ++t.failed;
    samples.push_back(std::chrono::duration<double, std::milli>(end - start).count());
  }
  std::sort(samples.begin(), samples.end());
  t.median_ms = samples[samples.size() / 2];
  return t;
}

void print_row(const char* label, int n, const Timing& t, bool csv) {
  if (csv) {
    std::printf("%s,%d,%.6f,%.3e,%d,%d\n", label, n, t.median_ms, t.worst_ratio, t.rejected, t.failed);
  } else {
    std::printf("  %-22s %4dx%-4d %10.3f ms   worst residual/bound %8.2e   rejected %d   failed %d\n",
                label, n, n, t.median_ms, t.worst_ratio, t.rejected, t.failed);
  }
}

}  // namespace

int main(int argc, char** argv) {
  int repeats = 7;
  bool csv = false;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--repeats") == 0 && i + 1 < argc) {
      repeats = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--csv") == 0) {
      csv = true;
    }
  }
  if (repeats < 1) repeats = 1;

  if (csv) std::printf("method,n,median_ms,worst_ratio,rejected,failed\n");

  const std::vector<int> bonds = {4, 8, 16, 32, 64};
  const std::vector<std::string> shapes = {"decaying", "flat", "rank-deficient"};

  for (const std::string& shape : shapes) {
    if (!csv) std::printf("\n== SVD, %s spectrum ==\n", shape.c_str());
    for (const int bond : bonds) {
      const int n = 2 * bond;
      const autonne_test::SvdCase c = autonne_test::make_svd_case(
          n, n, spectrum(shape, n), autonne::MatrixOrder::ColMajor,
          static_cast<std::uint64_t>(bond) * 1000u + static_cast<std::uint64_t>(shape.size()));
      const Eigen::Map<const Eigen::MatrixXcd> M(c.m.data(), n, n);

      std::vector<Complex> u(static_cast<std::size_t>(n) * static_cast<std::size_t>(n));
      std::vector<double> s(static_cast<std::size_t>(n));
      std::vector<Complex> v(static_cast<std::size_t>(n) * static_cast<std::size_t>(n));

      auto judge = [&](Timing& t) {
        const autonne::verify::SvdReport r = autonne::verify::check_svd(
            c.m.data(), n, n, c.order, u.data(), s.data(), v.data(), n);
        if (!r.ok()) ++t.rejected;
        const double ratio = r.backward_bound > 0.0 ? r.residual / r.backward_bound : 0.0;
        if (ratio > t.worst_ratio) t.worst_ratio = ratio;
      };

      {
        Timing t = time_it(repeats, [&]() {
          return autonne::svd_thin(c.m.data(), n, n, c.order, u.data(), s.data(), v.data());
        });
        judge(t);
        print_row("autonne", n, t, csv);
      }
      {
        Timing t = time_it(repeats, [&]() {
          Eigen::JacobiSVD<Eigen::MatrixXcd> svd(M, Eigen::ComputeThinU | Eigen::ComputeThinV);
          if (svd.info() != Eigen::Success) return false;
          Eigen::Map<Eigen::MatrixXcd>(u.data(), n, n) = svd.matrixU();
          Eigen::Map<Eigen::MatrixXcd>(v.data(), n, n) = svd.matrixV();
          Eigen::Map<Eigen::VectorXd>(s.data(), n) = svd.singularValues();
          return true;
        });
        judge(t);
        print_row("eigen JacobiSVD", n, t, csv);
      }
      {
        Timing t = time_it(repeats, [&]() {
          Eigen::BDCSVD<Eigen::MatrixXcd> svd(M, Eigen::ComputeThinU | Eigen::ComputeThinV);
          if (svd.info() != Eigen::Success) return false;
          Eigen::Map<Eigen::MatrixXcd>(u.data(), n, n) = svd.matrixU();
          Eigen::Map<Eigen::MatrixXcd>(v.data(), n, n) = svd.matrixV();
          Eigen::Map<Eigen::VectorXd>(s.data(), n) = svd.singularValues();
          return true;
        });
        judge(t);
        print_row("eigen BDCSVD", n, t, csv);
      }
    }
  }

  if (!csv) std::printf("\n== Hermitian eigendecomposition ==\n");
  for (const int bond : bonds) {
    const int n = 2 * bond;
    const std::vector<Complex> a = autonne_test::random_hermitian(n, static_cast<std::uint64_t>(n) * 31u);
    const Eigen::Map<const Eigen::MatrixXcd> A(a.data(), n, n);
    std::vector<double> evals(static_cast<std::size_t>(n));
    std::vector<Complex> evecs(static_cast<std::size_t>(n) * static_cast<std::size_t>(n));

    auto judge = [&](Timing& t) {
      const autonne::verify::EighReport r = autonne::verify::check_eigh(
          a.data(), n, autonne::MatrixOrder::ColMajor, evals.data(), evecs.data());
      if (!r.ok()) ++t.rejected;
      const double ratio = r.backward_bound > 0.0 ? r.residual / r.backward_bound : 0.0;
      if (ratio > t.worst_ratio) t.worst_ratio = ratio;
    };

    {
      Timing t = time_it(repeats, [&]() {
        return autonne::eigh(a.data(), n, autonne::MatrixOrder::ColMajor, evals.data(), evecs.data());
      });
      judge(t);
      print_row("autonne eigh", n, t, csv);
    }
    {
      Timing t = time_it(repeats, [&]() {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(A, Eigen::ComputeEigenvectors);
        if (es.info() != Eigen::Success) return false;
        Eigen::Map<Eigen::VectorXd>(evals.data(), n) = es.eigenvalues();
        Eigen::Map<Eigen::MatrixXcd>(evecs.data(), n, n) = es.eigenvectors();
        return true;
      });
      judge(t);
      print_row("eigen SelfAdjoint", n, t, csv);
    }
  }

  // The two shapes from the spec, once each, judged rather than timed.
  if (!csv) {
    std::printf("\n== Spec reproducers (structure), harness verdicts ==\n");
    struct Named {
      const char* name;
      std::vector<Complex> m;
      int n;
    };
    const std::vector<Named> cases = {
        {"simon 36x36 rank 12", autonne_test::simon_coset_matrix(), 36},
        {"simon + residue", autonne_test::simon_coset_matrix(3.4e-17), 36},
        {"poison-like 8x8", autonne_test::poison_theta_like(), 8},
    };
    for (const Named& nc : cases) {
      const int n = nc.n;
      std::vector<Complex> u(static_cast<std::size_t>(n) * static_cast<std::size_t>(n));
      std::vector<double> s(static_cast<std::size_t>(n));
      std::vector<Complex> v(static_cast<std::size_t>(n) * static_cast<std::size_t>(n));
      auto verdict = [&](const char* method, bool ok) {
        const autonne::verify::SvdReport r = autonne::verify::check_svd(
            nc.m.data(), n, n, autonne::MatrixOrder::ColMajor, u.data(), s.data(), v.data(), n);
        double energy = 0.0;
        for (const double x : s) energy += x * x;
        std::printf("  %-22s %-18s %s  harness %s  sum s^2 = %.17g\n", nc.name, method,
                    ok ? "ok    " : "FAILED", r.ok() ? "accepts" : "REJECTS", energy);
      };
      verdict("autonne", autonne::svd_thin(nc.m.data(), n, n, autonne::MatrixOrder::ColMajor,
                                           u.data(), s.data(), v.data()));
      {
        const Eigen::Map<const Eigen::MatrixXcd> M(nc.m.data(), n, n);
        Eigen::JacobiSVD<Eigen::MatrixXcd> svd(M, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::Map<Eigen::MatrixXcd>(u.data(), n, n) = svd.matrixU();
        Eigen::Map<Eigen::MatrixXcd>(v.data(), n, n) = svd.matrixV();
        Eigen::Map<Eigen::VectorXd>(s.data(), n) = svd.singularValues();
        verdict("eigen JacobiSVD", svd.info() == Eigen::Success);
      }
      {
        const Eigen::Map<const Eigen::MatrixXcd> M(nc.m.data(), n, n);
        Eigen::BDCSVD<Eigen::MatrixXcd> svd(M, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::Map<Eigen::MatrixXcd>(u.data(), n, n) = svd.matrixU();
        Eigen::Map<Eigen::MatrixXcd>(v.data(), n, n) = svd.matrixV();
        Eigen::Map<Eigen::VectorXd>(s.data(), n) = svd.singularValues();
        verdict("eigen BDCSVD", svd.info() == Eigen::Success);
      }
    }
  }
  return 0;
}
