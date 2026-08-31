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

// Hex-float matrix I/O.
//
// Matrices are written as C99 %a literals, which round-trip through binary64
// bit for bit. Decimal text does not: a corpus written as decimal and read
// back drifts by an ulp here and there, and a test suite that is supposed to
// pin down behaviour on degenerate spectra cannot afford that. Exact text also
// lets the same frozen corpus be shared with tools outside this repository.
//
// Header-only, and independent of everything but the standard library.

#ifndef AUTONNE_HEXFLOAT_HPP
#define AUTONNE_HEXFLOAT_HPP

#include <complex>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <istream>
#include <ostream>
#include <string>
#include <vector>

#include "autonne/autonne.hpp"

namespace autonne {
namespace hexfloat {

// Text format, one whitespace-separated token stream:
//
//   autonne-hexfloat 1 matrix <name> <rows> <cols> <rowmajor|colmajor>
//   <re> <im>            (rows*cols pairs, in the declared storage order)
//
//   autonne-hexfloat 1 vector <name> <n>
//   <value>              (n values)
//
// <name> is a single token; "-" stands for unnamed.
inline const char* magic() { return "autonne-hexfloat"; }
inline int format_version() { return 1; }

// Exact hex-float rendering of a binary64 value. NaN and the infinities come
// out as "nan" / "inf" / "-inf", which strtod reads back.
inline std::string to_string(double v) {
  char buf[64];
  const int n = std::snprintf(buf, sizeof(buf), "%a", v);
  if (n <= 0 || static_cast<std::size_t>(n) >= sizeof(buf)) return std::string();
  return std::string(buf, static_cast<std::size_t>(n));
}

// Parses one token. Requires the whole token to be consumed, so trailing
// garbage is an error rather than a silently truncated value.
inline bool from_string(const std::string& s, double& out) {
  if (s.empty()) return false;
  const char* first = s.c_str();
  char* last = nullptr;
  const double v = std::strtod(first, &last);
  if (last != first + s.size()) return false;
  out = v;
  return true;
}

inline std::string to_string(const std::complex<double>& z) {
  return to_string(z.real()) + " " + to_string(z.imag());
}

// A matrix as read back from a stream.
struct Matrix {
  std::string name;
  int rows = 0;
  int cols = 0;
  MatrixOrder order = MatrixOrder::ColMajor;
  std::vector<std::complex<double>> data;  // rows*cols, laid out per `order`
};

struct Vector {
  std::string name;
  std::vector<double> data;
};

namespace detail {

inline const char* order_token(MatrixOrder order) {
  return order == MatrixOrder::RowMajor ? "rowmajor" : "colmajor";
}

inline bool parse_order(const std::string& token, MatrixOrder& out) {
  if (token == "rowmajor") {
    out = MatrixOrder::RowMajor;
    return true;
  }
  if (token == "colmajor") {
    out = MatrixOrder::ColMajor;
    return true;
  }
  return false;
}

// Consumes the shared "<magic> <version> <kind>" prefix.
inline bool read_prefix(std::istream& in, const char* expected_kind) {
  std::string magic_token;
  int version = 0;
  std::string kind;
  if (!(in >> magic_token >> version >> kind)) return false;
  if (magic_token != magic()) return false;
  if (version != format_version()) return false;
  return kind == expected_kind;
}

}  // namespace detail

// `data` is rows*cols elements laid out according to `order`; both the layout
// and the element order on the wire follow it, so a write/read pair is a
// straight copy with no transposition.
inline bool write_matrix(std::ostream& out, const std::complex<double>* data,
                         int rows, int cols, MatrixOrder order,
                         const std::string& name = std::string("-")) {
  if (data == nullptr || rows <= 0 || cols <= 0) return false;
  out << magic() << ' ' << format_version() << " matrix "
      << (name.empty() ? std::string("-") : name) << ' ' << rows << ' ' << cols
      << ' ' << detail::order_token(order) << '\n';
  const std::size_t count = static_cast<std::size_t>(rows) *
                            static_cast<std::size_t>(cols);
  for (std::size_t i = 0; i < count; ++i) out << to_string(data[i]) << '\n';
  return static_cast<bool>(out);
}

inline bool read_matrix(std::istream& in, Matrix& out) {
  if (!detail::read_prefix(in, "matrix")) return false;
  std::string name;
  int rows = 0;
  int cols = 0;
  std::string order_token;
  if (!(in >> name >> rows >> cols >> order_token)) return false;
  MatrixOrder order = MatrixOrder::ColMajor;
  if (!detail::parse_order(order_token, order)) return false;
  if (rows <= 0 || cols <= 0) return false;

  const std::size_t count = static_cast<std::size_t>(rows) *
                            static_cast<std::size_t>(cols);
  std::vector<std::complex<double>> data;
  data.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    std::string re_token;
    std::string im_token;
    if (!(in >> re_token >> im_token)) return false;
    double re = 0.0;
    double im = 0.0;
    if (!from_string(re_token, re) || !from_string(im_token, im)) return false;
    data.push_back(std::complex<double>(re, im));
  }

  out.name = name;
  out.rows = rows;
  out.cols = cols;
  out.order = order;
  out.data.swap(data);
  return true;
}

inline bool write_vector(std::ostream& out, const double* data, int n,
                         const std::string& name = std::string("-")) {
  if (data == nullptr || n <= 0) return false;
  out << magic() << ' ' << format_version() << " vector "
      << (name.empty() ? std::string("-") : name) << ' ' << n << '\n';
  for (int i = 0; i < n; ++i) out << to_string(data[i]) << '\n';
  return static_cast<bool>(out);
}

inline bool read_vector(std::istream& in, Vector& out) {
  if (!detail::read_prefix(in, "vector")) return false;
  std::string name;
  int n = 0;
  if (!(in >> name >> n)) return false;
  if (n <= 0) return false;

  std::vector<double> data;
  data.reserve(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    std::string token;
    if (!(in >> token)) return false;
    double v = 0.0;
    if (!from_string(token, v)) return false;
    data.push_back(v);
  }

  out.name = name;
  out.data.swap(data);
  return true;
}

}  // namespace hexfloat
}  // namespace autonne

#endif  // AUTONNE_HEXFLOAT_HPP
