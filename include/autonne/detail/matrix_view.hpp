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

// Internal element accessor.
//
// Not part of the public interface: autonne/autonne.hpp does not include this
// header, and nothing here appears in a public signature. It exists so that
// index arithmetic is written once rather than at every access site.
//
// Where the standard library ships <mdspan> this is a pair of thin aliases
// over std::mdspan -- layout_left for autonne's own column-major buffers,
// layout_stride for a caller-supplied matrix whose order is only known at run
// time. Otherwise the same surface is hand-rolled below. Both paths are
// subscripted as v[i, j], so call sites do not branch on which one is live.

#ifndef AUTONNE_DETAIL_MATRIX_VIEW_HPP
#define AUTONNE_DETAIL_MATRIX_VIEW_HPP

#include <array>
#include <complex>
#include <cstddef>

#include "autonne/autonne.hpp"

#if __has_include(<mdspan>)
#include <mdspan>
#endif

// Define AUTONNE_NO_STD_MDSPAN to compile the hand-rolled path even where
// <mdspan> exists. Without it that branch is dead code on any modern toolchain
// and would rot unnoticed; the test suite builds against both.
#if defined(__cpp_lib_mdspan) && __cpp_lib_mdspan >= 202207L && \
    !defined(AUTONNE_NO_STD_MDSPAN)
#define AUTONNE_HAS_STD_MDSPAN 1
#else
#define AUTONNE_HAS_STD_MDSPAN 0
#endif

namespace autonne {
namespace detail {

#if AUTONNE_HAS_STD_MDSPAN

using Extents2 = std::dextents<std::size_t, 2>;

// Column-major: element (i, j) sits at i + j * rows. layout_left is exactly
// that, so the mapping is not spelled out here.
template <typename T>
using ColMajorRef = std::mdspan<T, Extents2, std::layout_left>;

// Either order, chosen at run time, expressed as an explicit stride pair.
template <typename T>
using StridedRef = std::mdspan<T, Extents2, std::layout_stride>;

#else  // hand-rolled equivalents

// Mirrors the slice of std::mdspan that autonne uses: multidimensional
// subscript, extent(), and data_handle().
template <typename T>
class ColMajorRef {
 public:
  using element_type = T;
  using index_type = std::size_t;

  constexpr ColMajorRef(T* data, std::size_t rows, std::size_t cols) noexcept
      : data_(data), rows_(rows), cols_(cols) {}

  constexpr T& operator[](std::size_t i, std::size_t j) const noexcept {
    return data_[i + j * rows_];
  }

  constexpr std::size_t extent(std::size_t r) const noexcept {
    return r == 0 ? rows_ : cols_;
  }
  constexpr T* data_handle() const noexcept { return data_; }

 private:
  T* data_;
  std::size_t rows_;
  std::size_t cols_;
};

template <typename T>
class StridedRef {
 public:
  using element_type = T;
  using index_type = std::size_t;

  constexpr StridedRef(T* data, std::size_t rows, std::size_t cols,
                       std::size_t row_stride, std::size_t col_stride) noexcept
      : data_(data),
        rows_(rows),
        cols_(cols),
        row_stride_(row_stride),
        col_stride_(col_stride) {}

  constexpr T& operator[](std::size_t i, std::size_t j) const noexcept {
    return data_[i * row_stride_ + j * col_stride_];
  }

  constexpr std::size_t extent(std::size_t r) const noexcept {
    return r == 0 ? rows_ : cols_;
  }
  constexpr std::size_t stride(std::size_t r) const noexcept {
    return r == 0 ? row_stride_ : col_stride_;
  }
  constexpr T* data_handle() const noexcept { return data_; }

 private:
  T* data_;
  std::size_t rows_;
  std::size_t cols_;
  std::size_t row_stride_;
  std::size_t col_stride_;
};

#endif  // AUTONNE_HAS_STD_MDSPAN

// autonne's own outputs -- U, V, the eigenvectors -- are always column-major,
// so they are viewed through here rather than by naming a layout.
template <typename T>
constexpr ColMajorRef<T> col_major(T* data, int rows, int cols) noexcept {
  return ColMajorRef<T>(data, static_cast<std::size_t>(rows),
                        static_cast<std::size_t>(cols));
}

// A caller-supplied matrix, whose storage order is a run-time argument.
template <typename T>
constexpr StridedRef<T> ordered(T* data, int rows, int cols,
                                MatrixOrder order) noexcept {
  const std::size_t r = static_cast<std::size_t>(rows);
  const std::size_t c = static_cast<std::size_t>(cols);
  const std::size_t row_stride = (order == MatrixOrder::RowMajor) ? c : 1u;
  const std::size_t col_stride = (order == MatrixOrder::RowMajor) ? 1u : r;
#if AUTONNE_HAS_STD_MDSPAN
  const std::array<std::size_t, 2> strides{row_stride, col_stride};
  const std::layout_stride::mapping<Extents2> mapping(Extents2(r, c), strides);
  return StridedRef<T>(data, mapping);
#else
  return StridedRef<T>(data, r, c, row_stride, col_stride);
#endif
}

// Extents come back as an unsigned index type on both paths; autonne counts in
// int throughout, and mixing the two is exactly the sign-compare that -Werror
// rejects. Narrowing happens here, once.
template <typename View>
constexpr int rows_of(const View& v) noexcept {
  return static_cast<int>(v.extent(0));
}

template <typename View>
constexpr int cols_of(const View& v) noexcept {
  return static_cast<int>(v.extent(1));
}

// Subscripting with int arguments would convert to the unsigned index type at
// every site; this does it in one place.
template <typename View>
constexpr decltype(auto) at(const View& v, int i, int j) noexcept {
  return v[static_cast<std::size_t>(i), static_cast<std::size_t>(j)];
}

}  // namespace detail
}  // namespace autonne

#endif  // AUTONNE_DETAIL_MATRIX_VIEW_HPP
