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

// The frozen corpus: matrices written as exact hex-float literals by
// tools/make_corpus.py, each followed by the spectrum LAPACK (through numpy)
// computed for it. This is the one place the kernels are checked against an
// independent implementation rather than against the harness and against
// spectra known by construction.
//
// The comparison is absolute, at LAPACK's own accuracy: zgesdd and zheevd
// promise every value to within a modest multiple of eps times the largest,
// and no more, so a relative comparison on the graded cases would be testing
// LAPACK rather than autonne. The relative claims are made in test_svd.cpp
// and test_eigh.cpp on inputs whose exact spectrum is known.

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include "autonne/autonne.hpp"
#include "autonne/hexfloat.hpp"
#include "autonne/verify.hpp"
#include "report_matchers.hpp"
#include "test_support.hpp"

#ifndef AUTONNE_CORPUS_DIR
#error "AUTONNE_CORPUS_DIR must point at tests/corpus"
#endif

namespace {

using autonne::MatrixOrder;
using autonne_test::EighAccepted;
using autonne_test::EighResult;
using autonne_test::SpectrumClose;
using autonne_test::SvdAccepted;
using autonne_test::SvdResult;
using autonne_test::run_eigh;
using autonne_test::run_svd;

constexpr double kEps = std::numeric_limits<double>::epsilon();

struct Record {
  autonne::hexfloat::Matrix m;
  autonne::hexfloat::Vector reference;
};

Record load(const std::string& name) {
  const std::string path = std::string(AUTONNE_CORPUS_DIR) + "/" + name + ".txt";
  std::ifstream in(path);
  Record r;
  if (!in) {
    ADD_FAILURE() << "cannot open " << path;
    return r;
  }
  if (!autonne::hexfloat::read_matrix(in, r.m)) ADD_FAILURE() << "bad matrix in " << path;
  if (!autonne::hexfloat::read_vector(in, r.reference)) ADD_FAILURE() << "bad vector in " << path;
  return r;
}

double largest(const std::vector<double>& v) {
  double m = 0.0;
  for (const double x : v) {
    if (std::fabs(x) > m) m = std::fabs(x);
  }
  return m;
}

class SvdCorpus : public ::testing::TestWithParam<const char*> {};
class EighCorpus : public ::testing::TestWithParam<const char*> {};

TEST_P(SvdCorpus, AgreesWithLapackAndPassesHarness) {
  const Record rec = load(GetParam());
  if (rec.m.rows == 0) return;
  const int rows = rec.m.rows;
  const int cols = rec.m.cols;
  const SvdResult r = run_svd(rec.m.data.data(), rows, cols, rec.m.order);
  ASSERT_TRUE(r.ok);
  const int k = rows < cols ? rows : cols;
  EXPECT_TRUE(SvdAccepted(autonne::verify::check_svd(rec.m.data.data(), rows, cols, rec.m.order,
                                                     r.u.data(), r.s.data(), r.v.data(), k)));
  ASSERT_EQ(rec.reference.data.size(), static_cast<std::size_t>(k));
  const double tol = 32.0 * static_cast<double>(rows > cols ? rows : cols) * kEps * largest(rec.reference.data);
  EXPECT_TRUE(SpectrumClose(r.s, rec.reference.data, tol, true));
}

TEST_P(EighCorpus, AgreesWithLapackAndPassesHarness) {
  const Record rec = load(GetParam());
  if (rec.m.rows == 0) return;
  const int n = rec.m.rows;
  ASSERT_EQ(rec.m.cols, n);
  const EighResult r = run_eigh(rec.m.data.data(), n, rec.m.order);
  ASSERT_TRUE(r.ok);
  EXPECT_TRUE(EighAccepted(autonne::verify::check_eigh(rec.m.data.data(), n, rec.m.order,
                                                       r.evals.data(), r.evecs.data())));
  ASSERT_EQ(rec.reference.data.size(), static_cast<std::size_t>(n));
  const double tol = 32.0 * static_cast<double>(n) * kEps * largest(rec.reference.data);
  EXPECT_TRUE(SpectrumClose(r.evals, rec.reference.data, tol, false));
}

INSTANTIATE_TEST_SUITE_P(Files, SvdCorpus,
                         ::testing::Values("svd_simon36", "svd_simon36_residue", "svd_poison8",
                                           "svd_dft16", "svd_random_16x9", "svd_random_9x16",
                                           "svd_random_32x32", "svd_column_scaled_8",
                                           "svd_row_scaled_8", "svd_zero_rows_cols_24"));

INSTANTIATE_TEST_SUITE_P(Files, EighCorpus,
                         ::testing::Values("eigh_random_24", "eigh_random_64",
                                           "eigh_graded_pd_8", "eigh_degenerate_12"));

}  // namespace
