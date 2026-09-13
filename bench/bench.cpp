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
// wall time over repeated calls of autonne::svd_thin, autonne::svd_thin_bdc,
// Eigen::JacobiSVD and Eigen::BDCSVD (thin U and V), and of autonne::eigh
// against Eigen::SelfAdjointEigenSolver. Every factorisation is also passed through
// the verification harness, and the worst ratio of measured residual to
// permitted bound is reported, because a fast wrong answer is not a result.
//
// Two further modes. `--leaf-sweep` times the divide-and-conquer core alone
// at every leaf size it might use, which is how the constant in the BDC
// kernel is chosen. `--mps <dir>` runs the two-site blocks a matrix-product-
// state simulator recorded (tests/corpus/lindblad_mps), per shape and
// weighted by how often each shape occurred, which is the workload the
// kernels exist for.
//
// Built strict (-fno-fast-math): Eigen under -ffast-math is the failure the
// project exists to avoid, not something to measure here.
//
// Usage: autonne_bench [--repeats N] [--csv] [--leaf-sweep] [--mps <dir>]

#include <Eigen/Dense>
#include <Eigen/SVD>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "autonne/autonne.hpp"
#include "autonne/hexfloat.hpp"
#include "autonne/verify.hpp"
#include "detail/bidiag_dc.hpp"
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

// --- leaf sweep -------------------------------------------------------------

// The divide-and-conquer core alone, bidiag_dc::bidiag_svd, on a random
// upper bidiagonal, at every leaf size the kernel might use. The leaf is the
// one free constant in the BDC kernel: below it a block is solved densely
// by pivoted QR and one-sided Jacobi, above it split and merged, and where
// the two costs cross is a measurement, not a derivation. The
// bidiagonalisation and the back-transformation do not depend on it, so
// they are left out and the core is timed on its own.
void run_leaf_sweep(int repeats) {
  using autonne::detail::bidiag_dc::bidiag_svd;
  const std::vector<int> sizes = {32, 64, 128, 256};
  const std::vector<std::size_t> leaves = {4, 8, 12, 16, 24, 32, 48, 64};

  std::printf("== bidiag_dc::bidiag_svd, random bidiagonal, median ms by leaf size ==\n");
  std::printf("  %5s", "n");
  for (const std::size_t leaf : leaves) std::printf(" %9zu", leaf);
  std::printf("\n");
  for (const int n : sizes) {
    std::mt19937_64 rng(static_cast<std::uint64_t>(n) * 101u);
    std::uniform_real_distribution<double> uniform(0.05, 1.0);
    std::vector<double> d(static_cast<std::size_t>(n));
    std::vector<double> e(static_cast<std::size_t>(n), 0.0);
    for (double& x : d) x = uniform(rng);
    for (int i = 0; i + 1 < n; ++i) e[static_cast<std::size_t>(i)] = uniform(rng);
    std::vector<double> U;
    std::vector<double> s;
    std::vector<double> V;
    std::printf("  %5d", n);
    for (const std::size_t leaf : leaves) {
      const Timing t = time_it(repeats, [&]() {
        return bidiag_svd(d.data(), e.data(), static_cast<std::size_t>(n), leaf, U, s, V);
      });
      std::printf(" %9.3f", t.median_ms);
      if (t.failed) std::printf("!");
    }
    std::printf("\n");
  }
}

// --- the MPS workload -------------------------------------------------------

// One representative block per shape, with the number of times that shape
// occurred in the run the corpus was captured from (its README gives the
// table). The weighted total is the time the run's 276 splits would have
// spent in each method, which is what the shape-by-shape table cannot show:
// squares are three quarters of the splits and 128 x 128 alone is an eighth.
struct MpsShape {
  const char* file;
  int count;
};

const std::vector<MpsShape> kMpsShapes = {
    {"svd_theta_2x2", 36},    {"svd_theta_2x8", 9},     {"svd_theta_4x4", 33},
    {"svd_theta_4x16", 9},    {"svd_theta_8x2", 9},     {"svd_theta_8x8", 30},
    {"svd_theta_8x32", 6},    {"svd_theta_16x4", 9},    {"svd_theta_16x16", 27},
    {"svd_theta_16x64", 6},   {"svd_theta_32x8", 6},    {"svd_theta_32x32", 24},
    {"svd_theta_32x128", 3},  {"svd_theta_64x16", 6},   {"svd_theta_64x64", 21},
    {"svd_theta_64x128", 3},  {"svd_theta_128x32", 3},  {"svd_theta_128x64", 3},
    {"svd_theta_128x128", 33},
};

using RowMajorXcd = Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

int run_mps(const std::string& dir, int repeats, bool csv) {
  const std::vector<const char*> methods = {"autonne", "autonne bdc", "eigen JacobiSVD",
                                            "eigen BDCSVD"};
  std::vector<double> weighted_ms(methods.size(), 0.0);
  std::vector<int> rejected(methods.size(), 0);
  int splits = 0;

  if (csv) {
    std::printf("method,shape,rows,cols,count,median_ms,rejected\n");
  } else {
    std::printf("== MPS two-site blocks (%s), median ms per call ==\n", dir.c_str());
    std::printf("  %-18s %5s", "block", "count");
    for (const char* m : methods) std::printf(" %16s", m);
    std::printf("\n");
  }

  for (const MpsShape& shape : kMpsShapes) {
    const std::string path = dir + "/" + shape.file + ".txt";
    std::ifstream in(path);
    autonne::hexfloat::Matrix m;
    if (!in || !autonne::hexfloat::read_matrix(in, m)) {
      std::fprintf(stderr, "cannot read %s\n", path.c_str());
      return 1;
    }
    const int rows = m.rows;
    const int cols = m.cols;
    const int k = rows < cols ? rows : cols;
    std::vector<Complex> u(static_cast<std::size_t>(rows) * static_cast<std::size_t>(k));
    std::vector<double> s(static_cast<std::size_t>(k));
    std::vector<Complex> v(static_cast<std::size_t>(cols) * static_cast<std::size_t>(k));
    const std::uint64_t probe_seed = static_cast<std::uint64_t>(rows * 131 + cols) * 7919u + 29u;

    // Eigen sees the block in the order it is stored.
    const Eigen::Map<const RowMajorXcd> M_row(m.data.data(), rows, cols);
    const Eigen::Map<const Eigen::MatrixXcd> M_col(m.data.data(), rows, cols);
    auto eigen_store = [&](const auto& svd) {
      Eigen::Map<Eigen::MatrixXcd>(u.data(), rows, k) = svd.matrixU();
      Eigen::Map<Eigen::MatrixXcd>(v.data(), cols, k) = svd.matrixV();
      Eigen::Map<Eigen::VectorXd>(s.data(), k) = svd.singularValues();
    };
    auto eigen_jacobi = [&]() {
      if (m.order == autonne::MatrixOrder::RowMajor) {
        Eigen::JacobiSVD<RowMajorXcd> svd(M_row, Eigen::ComputeThinU | Eigen::ComputeThinV);
        if (svd.info() != Eigen::Success) return false;
        eigen_store(svd);
      } else {
        Eigen::JacobiSVD<Eigen::MatrixXcd> svd(M_col, Eigen::ComputeThinU | Eigen::ComputeThinV);
        if (svd.info() != Eigen::Success) return false;
        eigen_store(svd);
      }
      return true;
    };
    auto eigen_bdc = [&]() {
      if (m.order == autonne::MatrixOrder::RowMajor) {
        Eigen::BDCSVD<RowMajorXcd> svd(M_row, Eigen::ComputeThinU | Eigen::ComputeThinV);
        if (svd.info() != Eigen::Success) return false;
        eigen_store(svd);
      } else {
        Eigen::BDCSVD<Eigen::MatrixXcd> svd(M_col, Eigen::ComputeThinU | Eigen::ComputeThinV);
        if (svd.info() != Eigen::Success) return false;
        eigen_store(svd);
      }
      return true;
    };

    if (!csv) std::printf("  %-18s %5d", shape.file + 4, shape.count);
    for (std::size_t which = 0; which < methods.size(); ++which) {
      Timing t;
      switch (which) {
        case 0:
          t = time_it(repeats, [&]() {
            return autonne::svd_thin(m.data.data(), rows, cols, m.order, u.data(), s.data(), v.data());
          });
          break;
        case 1:
          t = time_it(repeats, [&]() {
            return autonne::svd_thin_bdc(m.data.data(), rows, cols, m.order, u.data(), s.data(),
                                         v.data());
          });
          break;
        case 2:
          t = time_it(repeats, eigen_jacobi);
          break;
        default:
          t = time_it(repeats, eigen_bdc);
          break;
      }
      const SvdVerdict vd = judge_svd(m.data.data(), rows, cols, m.order, u.data(), s.data(),
                                      v.data(), probe_seed);
      const bool ok = vd.report.ok() && t.failed == 0;
      if (!ok) ++rejected[which];
      weighted_ms[which] += t.median_ms * shape.count;
      if (csv) {
        std::printf("%s,%s,%d,%d,%d,%.6f,%d\n", methods[which], shape.file + 4, rows, cols,
                    shape.count, t.median_ms, ok ? 0 : 1);
      } else {
        std::printf(" %13.3f %s", t.median_ms, ok ? "  " : "X ");
      }
    }
    if (!csv) std::printf("\n");
    splits += shape.count;
  }

  if (!csv) {
    std::printf("  %-18s %5d", "weighted total", splits);
    for (std::size_t which = 0; which < methods.size(); ++which) {
      std::printf(" %13.1f %s", weighted_ms[which], rejected[which] ? "X " : "  ");
    }
    std::printf("\n  (X marks a shape the harness rejected or the method refused)\n");
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  int repeats = 7;
  bool csv = false;
  bool leaf_sweep = false;
  std::string mps_dir;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--repeats") == 0 && i + 1 < argc) {
      repeats = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--csv") == 0) {
      csv = true;
    } else if (std::strcmp(argv[i], "--leaf-sweep") == 0) {
      leaf_sweep = true;
    } else if (std::strcmp(argv[i], "--mps") == 0 && i + 1 < argc) {
      mps_dir = argv[++i];
    }
  }
  if (repeats < 1) repeats = 1;

  if (leaf_sweep) {
    run_leaf_sweep(repeats);
    return 0;
  }
  if (!mps_dir.empty()) return run_mps(mps_dir, repeats, csv);

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
          return autonne::svd_thin_bdc(c.m.data(), n, n, c.order, u.data(), s.data(), v.data());
        });
        const SvdVerdict vd = judge(t);
        print_svd_row("autonne bdc", n, t, vd, csv);
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
      verdict("autonne bdc", autonne::svd_thin_bdc(nc.m.data(), n, n, autonne::MatrixOrder::ColMajor,
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
