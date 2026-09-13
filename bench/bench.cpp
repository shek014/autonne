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
#include <random>
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

// Everything measured on one SVD result: the harness verdict, each of its
// quantities as measured / bound (so 1.0 is the harness's own line), the two
// cheap screens on the same scale, and what each cost.
struct SvdVerdict {
  autonne::verify::SvdReport report;
  double backward_ratio = 0.0;  // ||M - U S V^*||_F / bound
  double ortho_ratio = 0.0;     // max(||U^*U - I||, ||V^*V - I||) / bound
  double energy_ratio = 0.0;    // |sum s^2 - ||M||_F^2| / bound
  double probe_ratio = 0.0;     // max over probes of ||R x|| sqrt(n) / (||x|| bound)
  double gemm_ratio = 0.0;      // the backward residual, recomputed on the parts
  double harness_ms = 0.0;      // one check_svd call
  double screen_ms = 0.0;       // energy identity plus every probe
  double gemm_ms = 0.0;         // the recomputed residual
};

constexpr int kProbes = 3;

// Element (i, j) of a rows x cols matrix stored in `order`.
Complex elem(const Complex* m, int rows, int cols, autonne::MatrixOrder order, int i, int j) {
  const std::size_t r = static_cast<std::size_t>(rows);
  const std::size_t c = static_cast<std::size_t>(cols);
  const std::size_t ii = static_cast<std::size_t>(i);
  const std::size_t jj = static_cast<std::size_t>(j);
  return order == autonne::MatrixOrder::ColMajor ? m[ii + jj * r] : m[ii * c + jj];
}

double abs_sq(const Complex& z) { return z.real() * z.real() + z.imag() * z.imag(); }

// |sum s^2 - ||M||_F^2| against the harness's own energy bound. O(m n).
double energy_screen(const Complex* m, int rows, int cols, const double* s, int k) {
  double norm_sq = 0.0;
  const std::size_t count = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
  for (std::size_t i = 0; i < count; ++i) norm_sq += abs_sq(m[i]);
  double energy = 0.0;
  for (int t = 0; t < k; ++t) energy += s[t] * s[t];
  const autonne::verify::Tolerances tol;
  const double bound = tol.spectrum_factor * static_cast<double>(std::max(rows, cols)) * tol.eps * norm_sq;
  return bound > 0.0 ? std::fabs(energy - norm_sq) / bound : 0.0;
}

// ||(M - U S V^*) x|| for kProbes complex Gaussian x with E|x_j|^2 = 1, each
// scaled to the sphere of radius sqrt(cols) so that ||R x|| estimates
// ||R||_F, and compared with the harness's backward bound. Worst ratio over
// the probes. O(m n + (m + n) k) per probe: three matrix-vector products.
double probe_screen(const Complex* m, int rows, int cols, autonne::MatrixOrder order,
                    const Complex* u, const double* s, const Complex* v, int k,
                    double backward_bound, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> gauss(0.0, std::sqrt(0.5));
  std::vector<Complex> x(static_cast<std::size_t>(cols));
  std::vector<Complex> w(static_cast<std::size_t>(k));
  double worst = 0.0;
  for (int p = 0; p < kProbes; ++p) {
    double x_sq = 0.0;
    for (Complex& z : x) {
      z = Complex(gauss(rng), gauss(rng));
      x_sq += abs_sq(z);
    }
    // w = S V^* x, on the parts: the screen is being costed against the
    // harness, so it must not pay for std::complex's operator* either.
    for (int t = 0; t < k; ++t) {
      double ar = 0.0;
      double ai = 0.0;
      const Complex* vt = v + static_cast<std::size_t>(t) * static_cast<std::size_t>(cols);
      for (int j = 0; j < cols; ++j) {
        const Complex& xj = x[static_cast<std::size_t>(j)];
        ar += vt[j].real() * xj.real() + vt[j].imag() * xj.imag();
        ai += vt[j].real() * xj.imag() - vt[j].imag() * xj.real();
      }
      w[static_cast<std::size_t>(t)] = Complex(s[t] * ar, s[t] * ai);
    }
    // r = M x - U w
    double r_sq = 0.0;
    for (int i = 0; i < rows; ++i) {
      double yr = 0.0;
      double yi = 0.0;
      for (int j = 0; j < cols; ++j) {
        const Complex mij = elem(m, rows, cols, order, i, j);
        const Complex& xj = x[static_cast<std::size_t>(j)];
        yr += mij.real() * xj.real() - mij.imag() * xj.imag();
        yi += mij.real() * xj.imag() + mij.imag() * xj.real();
      }
      double zr = 0.0;
      double zi = 0.0;
      for (int t = 0; t < k; ++t) {
        const Complex& uit = u[static_cast<std::size_t>(i) + static_cast<std::size_t>(t) * static_cast<std::size_t>(rows)];
        const Complex& wt = w[static_cast<std::size_t>(t)];
        zr += uit.real() * wt.real() - uit.imag() * wt.imag();
        zi += uit.real() * wt.imag() + uit.imag() * wt.real();
      }
      r_sq += (yr - zr) * (yr - zr) + (yi - zi) * (yi - zi);
    }
    const double estimate = std::sqrt(r_sq) * std::sqrt(static_cast<double>(cols)) / std::sqrt(x_sq);
    const double ratio = backward_bound > 0.0 ? estimate / backward_bound : 0.0;
    if (ratio > worst) worst = ratio;
  }
  return worst;
}

// ||M - U S V^*||_F with every complex product written on the parts: the
// same O(m n k) the harness does, without std::complex's operator*.
double gemm_residual(const Complex* m, int rows, int cols, autonne::MatrixOrder order,
                     const Complex* u, const double* s, const Complex* v, int k) {
  std::vector<double> svr(static_cast<std::size_t>(k));
  std::vector<double> svi(static_cast<std::size_t>(k));
  double acc = 0.0;
  for (int j = 0; j < cols; ++j) {
    // s_t conj(V(j, t)) for every t, once per column.
    for (int t = 0; t < k; ++t) {
      const Complex& vjt = v[static_cast<std::size_t>(j) + static_cast<std::size_t>(t) * static_cast<std::size_t>(cols)];
      svr[static_cast<std::size_t>(t)] = s[t] * vjt.real();
      svi[static_cast<std::size_t>(t)] = -s[t] * vjt.imag();
    }
    for (int i = 0; i < rows; ++i) {
      double ar = 0.0;
      double ai = 0.0;
      for (int t = 0; t < k; ++t) {
        const Complex& uit = u[static_cast<std::size_t>(i) + static_cast<std::size_t>(t) * static_cast<std::size_t>(rows)];
        const double br = svr[static_cast<std::size_t>(t)];
        const double bi = svi[static_cast<std::size_t>(t)];
        ar += uit.real() * br - uit.imag() * bi;
        ai += uit.real() * bi + uit.imag() * br;
      }
      const Complex d = elem(m, rows, cols, order, i, j) - Complex(ar, ai);
      acc += abs_sq(d);
    }
  }
  return std::sqrt(acc);
}

double ms_since(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

SvdVerdict judge_svd(const Complex* m, int rows, int cols, autonne::MatrixOrder order,
                     const Complex* u, const double* s, const Complex* v, std::uint64_t seed) {
  const int k = std::min(rows, cols);
  SvdVerdict out;

  auto start = Clock::now();
  out.report = autonne::verify::check_svd(m, rows, cols, order, u, s, v, k);
  out.harness_ms = ms_since(start);
  const autonne::verify::SvdReport& r = out.report;
  out.backward_ratio = r.backward_bound > 0.0 ? r.residual / r.backward_bound : 0.0;
  out.ortho_ratio = r.ortho_bound > 0.0
      ? std::max(r.u_ortho_residual, r.v_ortho_residual) / r.ortho_bound : 0.0;

  start = Clock::now();
  out.energy_ratio = energy_screen(m, rows, cols, s, k);
  out.probe_ratio = probe_screen(m, rows, cols, order, u, s, v, k, r.backward_bound, seed);
  out.screen_ms = ms_since(start);

  start = Clock::now();
  const double gemm = gemm_residual(m, rows, cols, order, u, s, v, k);
  out.gemm_ms = ms_since(start);
  out.gemm_ratio = r.backward_bound > 0.0 ? gemm / r.backward_bound : 0.0;
  return out;
}

void print_verdict(const SvdVerdict& vd, bool csv) {
  if (csv) {
    std::printf(",%.3e,%.3e,%.3e,%.3e,%.3e,%.6f,%.6f,%.6f",
                vd.backward_ratio, vd.ortho_ratio, vd.energy_ratio, vd.probe_ratio,
                vd.gemm_ratio, vd.harness_ms, vd.screen_ms, vd.gemm_ms);
  } else {
    std::printf("   bwd %8.2e  orth %8.2e  energy %8.2e  probe %8.2e  gemm %8.2e"
                "   harness %7.3f ms  screen %7.3f ms  gemm %7.3f ms",
                vd.backward_ratio, vd.ortho_ratio, vd.energy_ratio, vd.probe_ratio,
                vd.gemm_ratio, vd.harness_ms, vd.screen_ms, vd.gemm_ms);
  }
}

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

void print_svd_row(const char* label, int n, const Timing& t, const SvdVerdict& vd, bool csv) {
  if (csv) {
    std::printf("%s,%d,%.6f,%d,%d", label, n, t.median_ms, t.rejected, t.failed);
    print_verdict(vd, csv);
    std::printf("\n");
  } else {
    std::printf("  %-22s %4dx%-4d %10.3f ms  %s  failed %d\n",
                label, n, n, t.median_ms, vd.report.ok() ? "accepted" : "REJECTED", t.failed);
    std::printf("  %-22s %9s", "", "");
    print_verdict(vd, csv);
    std::printf("\n");
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

  if (csv) {
    std::printf("method,n,median_ms,rejected,failed,bwd_ratio,orth_ratio,energy_ratio,"
                "probe_ratio,gemm_ratio,harness_ms,screen_ms,gemm_ms\n");
  }

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

      // Judged once, on the result of the last timed call: every call sees
      // the same input, so the last result is the result.
      const std::uint64_t probe_seed = static_cast<std::uint64_t>(n) * 7919u + 17u;
      auto judge = [&](Timing& t) {
        const SvdVerdict vd = judge_svd(c.m.data(), n, n, c.order, u.data(), s.data(), v.data(),
                                        probe_seed);
        if (!vd.report.ok()) ++t.rejected;
        t.worst_ratio = vd.backward_ratio;
        return vd;
      };

      {
        Timing t = time_it(repeats, [&]() {
          return autonne::svd_thin(c.m.data(), n, n, c.order, u.data(), s.data(), v.data());
        });
        const SvdVerdict vd = judge(t);
        print_svd_row("autonne", n, t, vd, csv);
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
        const SvdVerdict vd = judge(t);
        print_svd_row("eigen JacobiSVD", n, t, vd, csv);
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
        const SvdVerdict vd = judge(t);
        print_svd_row("eigen BDCSVD", n, t, vd, csv);
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
        const SvdVerdict vd = judge_svd(nc.m.data(), n, n, autonne::MatrixOrder::ColMajor,
                                        u.data(), s.data(), v.data(),
                                        static_cast<std::uint64_t>(n) * 7919u + 23u);
        double energy = 0.0;
        for (const double x : s) energy += x * x;
        std::printf("  %-22s %-18s %s  harness %s  sum s^2 = %.17g\n", nc.name, method,
                    ok ? "ok    " : "FAILED", vd.report.ok() ? "accepts" : "REJECTS", energy);
        std::printf("  %-22s %-18s", "", "");
        print_verdict(vd, false);
        std::printf("\n");
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
