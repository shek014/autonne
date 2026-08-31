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

// Round-trip must be exact, not close. Comparisons here are on the bit
// pattern: a corpus that drifts by an ulp between write and read cannot pin
// down behaviour on degenerate spectra, which is the reason the corpus exists.

#include <gtest/gtest.h>

#include <bit>
#include <complex>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

#include "autonne/hexfloat.hpp"
#include "test_support.hpp"

namespace {

using autonne::MatrixOrder;
using autonne_test::Complex;
using autonne_test::make_svd_case;

std::uint64_t bits(double v) { return std::bit_cast<std::uint64_t>(v); }

// Values chosen to exercise the awkward corners: signed zero, subnormals, the
// ends of the normal range, and a value with no short decimal form.
std::vector<double> tricky_values() {
  return {0.0,
          -0.0,
          1.0,
          -1.0,
          0.1,
          -0.3333333333333333,
          3.141592653589793,
          1e-300,
          1e300,
          autonne_test::smallest_subnormal(),
          autonne_test::largest_normal(),
          -autonne_test::largest_normal(),
          2.2250738585072014e-308};
}

TEST(HexFloat, ScalarRoundTripIsBitExact) {
  for (const double v : tricky_values()) {
    const std::string text = autonne::hexfloat::to_string(v);
    ASSERT_FALSE(text.empty()) << "failed to render " << v;
    double back = 0.0;
    ASSERT_TRUE(autonne::hexfloat::from_string(text, back)) << text;
    EXPECT_EQ(bits(back), bits(v))
        << "round trip changed the bit pattern for " << text;
  }
}

TEST(HexFloat, RendersAsHexNotDecimal) {
  const std::string text = autonne::hexfloat::to_string(1.0);
  EXPECT_NE(text.find("0x"), std::string::npos) << text;
}

TEST(HexFloat, RejectsMalformedTokens) {
  double v = 0.0;
  EXPECT_FALSE(autonne::hexfloat::from_string("", v));
  EXPECT_FALSE(autonne::hexfloat::from_string("0x1p+0garbage", v));
  EXPECT_FALSE(autonne::hexfloat::from_string("not-a-number", v));
  EXPECT_FALSE(autonne::hexfloat::from_string("0x1p+0 0x1p+0", v));
}

TEST(HexFloat, MatrixRoundTripIsBitExact) {
  const auto c = make_svd_case(5, 3, {7.0, 2.5, 0.125}, MatrixOrder::ColMajor,
                               2718);

  std::stringstream stream;
  ASSERT_TRUE(autonne::hexfloat::write_matrix(stream, c.m.data(), c.rows,
                                              c.cols, c.order, "M"));

  autonne::hexfloat::Matrix back;
  ASSERT_TRUE(autonne::hexfloat::read_matrix(stream, back));

  EXPECT_EQ(back.name, "M");
  EXPECT_EQ(back.rows, c.rows);
  EXPECT_EQ(back.cols, c.cols);
  EXPECT_EQ(back.order == MatrixOrder::ColMajor, c.order == MatrixOrder::ColMajor);
  ASSERT_EQ(back.data.size(), c.m.size());
  for (std::size_t i = 0; i < c.m.size(); ++i) {
    EXPECT_EQ(bits(back.data[i].real()), bits(c.m[i].real())) << "element " << i;
    EXPECT_EQ(bits(back.data[i].imag()), bits(c.m[i].imag())) << "element " << i;
  }
}

TEST(HexFloat, MatrixRoundTripPreservesRowMajorOrder) {
  const auto c = make_svd_case(3, 4, {4.0, 2.0, 1.0}, MatrixOrder::RowMajor, 99);

  std::stringstream stream;
  ASSERT_TRUE(autonne::hexfloat::write_matrix(stream, c.m.data(), c.rows,
                                              c.cols, c.order));

  autonne::hexfloat::Matrix back;
  ASSERT_TRUE(autonne::hexfloat::read_matrix(stream, back));
  EXPECT_TRUE(back.order == MatrixOrder::RowMajor);
  EXPECT_EQ(back.name, "-");
  ASSERT_EQ(back.data.size(), c.m.size());
  for (std::size_t i = 0; i < c.m.size(); ++i) {
    EXPECT_EQ(bits(back.data[i].real()), bits(c.m[i].real()));
  }
}

TEST(HexFloat, VectorRoundTripIsBitExact) {
  const std::vector<double> s = tricky_values();

  std::stringstream stream;
  ASSERT_TRUE(autonne::hexfloat::write_vector(stream, s.data(),
                                              static_cast<int>(s.size()), "S"));

  autonne::hexfloat::Vector back;
  ASSERT_TRUE(autonne::hexfloat::read_vector(stream, back));
  EXPECT_EQ(back.name, "S");
  ASSERT_EQ(back.data.size(), s.size());
  for (std::size_t i = 0; i < s.size(); ++i) {
    EXPECT_EQ(bits(back.data[i]), bits(s[i])) << "element " << i;
  }
}

// A corpus file holds a matrix followed by its spectrum; reading them back in
// sequence from one stream has to work.
TEST(HexFloat, ConsecutiveRecordsInOneStream) {
  const auto c = make_svd_case(4, 4, {8.0, 4.0, 2.0, 1.0},
                               MatrixOrder::ColMajor, 1234);

  std::stringstream stream;
  ASSERT_TRUE(autonne::hexfloat::write_matrix(stream, c.m.data(), c.rows,
                                              c.cols, c.order, "M"));
  ASSERT_TRUE(autonne::hexfloat::write_vector(stream, c.s.data(), c.k, "S"));

  autonne::hexfloat::Matrix m;
  autonne::hexfloat::Vector s;
  ASSERT_TRUE(autonne::hexfloat::read_matrix(stream, m));
  ASSERT_TRUE(autonne::hexfloat::read_vector(stream, s));
  EXPECT_EQ(m.name, "M");
  EXPECT_EQ(s.name, "S");
  ASSERT_EQ(s.data.size(), static_cast<std::size_t>(c.k));
  EXPECT_EQ(bits(s.data[0]), bits(8.0));
}

TEST(HexFloat, RejectsWrongMagicVersionOrKind) {
  autonne::hexfloat::Matrix m;

  std::stringstream wrong_magic("not-autonne 1 matrix M 1 1 colmajor 0x1p+0 0x0p+0");
  EXPECT_FALSE(autonne::hexfloat::read_matrix(wrong_magic, m));

  std::stringstream wrong_version("autonne-hexfloat 99 matrix M 1 1 colmajor 0x1p+0 0x0p+0");
  EXPECT_FALSE(autonne::hexfloat::read_matrix(wrong_version, m));

  std::stringstream wrong_kind("autonne-hexfloat 1 vector S 1 0x1p+0");
  EXPECT_FALSE(autonne::hexfloat::read_matrix(wrong_kind, m));

  std::stringstream bad_order("autonne-hexfloat 1 matrix M 1 1 diagonal 0x1p+0 0x0p+0");
  EXPECT_FALSE(autonne::hexfloat::read_matrix(bad_order, m));

  std::stringstream truncated("autonne-hexfloat 1 matrix M 2 2 colmajor 0x1p+0 0x0p+0");
  EXPECT_FALSE(autonne::hexfloat::read_matrix(truncated, m));
}

}  // namespace
