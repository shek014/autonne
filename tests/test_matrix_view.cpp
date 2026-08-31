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

// The accessor is the one place index arithmetic is written, so an error in it
// would be invisible everywhere else: the harness would agree with itself and
// still be reading the wrong elements. These tests pin the mapping against
// literal offsets in the underlying buffer rather than against the accessor.

#include <gtest/gtest.h>

#include <complex>
#include <vector>

#include "autonne/detail/matrix_view.hpp"

namespace {

using autonne::MatrixOrder;
using autonne::detail::at;
using autonne::detail::col_major;
using autonne::detail::cols_of;
using autonne::detail::ordered;
using autonne::detail::rows_of;

using Complex = std::complex<double>;

// A 2x3 matrix whose (i, j) entry is i + 10*j, so a transposed read is
// immediately obvious.
std::vector<Complex> buffer_col_major() {
  // columns: (0,1) (10,11) (20,21)
  return {Complex(0, 0),  Complex(1, 0),  Complex(10, 0),
          Complex(11, 0), Complex(20, 0), Complex(21, 0)};
}

std::vector<Complex> buffer_row_major() {
  // rows: (0,10,20) (1,11,21)
  return {Complex(0, 0),  Complex(10, 0), Complex(20, 0),
          Complex(1, 0),  Complex(11, 0), Complex(21, 0)};
}

TEST(MatrixView, ColMajorMapsToLayoutLeftOffsets) {
  const std::vector<Complex> buf = buffer_col_major();
  const auto v = col_major(buf.data(), 2, 3);

  EXPECT_EQ(rows_of(v), 2);
  EXPECT_EQ(cols_of(v), 3);
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      EXPECT_DOUBLE_EQ(at(v, i, j).real(), static_cast<double>(i + 10 * j))
          << "at (" << i << ", " << j << ")";
      // layout_left: element (i, j) lives at i + j * rows.
      EXPECT_EQ(&at(v, i, j), &buf[static_cast<std::size_t>(i + j * 2)]);
    }
  }
}

TEST(MatrixView, OrderedReadsBothStorageOrdersIdentically) {
  const std::vector<Complex> col = buffer_col_major();
  const std::vector<Complex> row = buffer_row_major();
  const auto cv = ordered(col.data(), 2, 3, MatrixOrder::ColMajor);
  const auto rv = ordered(row.data(), 2, 3, MatrixOrder::RowMajor);

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      const double expected = static_cast<double>(i + 10 * j);
      EXPECT_DOUBLE_EQ(at(cv, i, j).real(), expected);
      EXPECT_DOUBLE_EQ(at(rv, i, j).real(), expected);
    }
  }
}

TEST(MatrixView, WritesThroughToTheUnderlyingBuffer) {
  std::vector<Complex> buf(6, Complex(0.0, 0.0));
  const auto v = col_major(buf.data(), 2, 3);
  at(v, 1, 2) = Complex(7.0, -8.0);

  EXPECT_DOUBLE_EQ(buf[5].real(), 7.0);
  EXPECT_DOUBLE_EQ(buf[5].imag(), -8.0);
}

// Non-square shapes are where a row/column mix-up stops being symmetric.
TEST(MatrixView, HandlesTallAndWideShapes) {
  std::vector<Complex> buf(12);
  for (std::size_t i = 0; i < buf.size(); ++i) {
    buf[i] = Complex(static_cast<double>(i), 0.0);
  }

  const auto tall = ordered(buf.data(), 4, 3, MatrixOrder::RowMajor);
  EXPECT_EQ(rows_of(tall), 4);
  EXPECT_EQ(cols_of(tall), 3);
  EXPECT_DOUBLE_EQ(at(tall, 3, 2).real(), 11.0);  // 3 * 3 + 2

  const auto wide = ordered(buf.data(), 3, 4, MatrixOrder::ColMajor);
  EXPECT_EQ(rows_of(wide), 3);
  EXPECT_EQ(cols_of(wide), 4);
  EXPECT_DOUBLE_EQ(at(wide, 2, 3).real(), 11.0);  // 3 * 3 + 2
}

}  // namespace
