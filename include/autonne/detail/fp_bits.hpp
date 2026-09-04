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

// Non-finite detection that survives -ffast-math.
//
// Shared by the kernel and the verification harness. Everything here is
// integer work on the object representation of a double; no floating-point
// predicate, comparison or arithmetic is involved, so no assumption a compiler
// makes about the range of floating-point values can remove it.
//
// Two rules keep that true, and both are visible in the signatures:
//
//   1. Values are taken by reference, never by value. Under -ffinite-math-only
//      Clang marks every by-value floating-point parameter and return value as
//      finite (nofpclass), and from Clang 22 on it folds an exponent-field test
//      on such a value to "finite" -- even when the function is inlined and the
//      argument was just loaded from memory. A reference is a pointer; a load
//      through it carries no such mark.
//
//   2. A complex value is read as sixteen bytes, not as real() and imag().
//      Those accessors return by value and reintroduce the boundary of rule 1.
//
// The contract is therefore: fp_bad reports whether the object representation
// in memory is a NaN or an infinity. That is exactly what a kernel scanning a
// caller's buffer needs, and it is the only thing any guard can promise in a
// translation unit where the compiler is permitted to assume finiteness of
// every value it computes.

#ifndef AUTONNE_DETAIL_FP_BITS_HPP
#define AUTONNE_DETAIL_FP_BITS_HPP

#include <bit>
#include <complex>
#include <cstdint>
#include <cstring>

namespace autonne {
namespace detail {

// True if x is NaN or an infinity: the IEEE-754 binary64 exponent field is
// all ones for those and nothing else.
constexpr bool fp_bad(const double& x) noexcept {
  static_assert(sizeof(double) == sizeof(std::uint64_t),
                "fp_bad assumes IEEE-754 binary64");
  const std::uint64_t bits = std::bit_cast<std::uint64_t>(x);
  return ((bits >> 52) & UINT64_C(0x7FF)) == UINT64_C(0x7FF);
}

inline bool fp_bad(const std::complex<double>& z) noexcept {
  static_assert(sizeof(std::complex<double>) == 2 * sizeof(std::uint64_t),
                "std::complex<double> must be two adjacent binary64 values");
  std::uint64_t words[2];
  std::memcpy(words, &z, sizeof words);
  return (((words[0] >> 52) & UINT64_C(0x7FF)) == UINT64_C(0x7FF)) ||
         (((words[1] >> 52) & UINT64_C(0x7FF)) == UINT64_C(0x7FF));
}

// True if any element of v[0, n) is non-finite.
inline bool any_bad(const double* v, int n) noexcept {
  for (int i = 0; i < n; ++i) {
    if (fp_bad(v[i])) return true;
  }
  return false;
}

inline bool any_bad(const std::complex<double>* v, int n) noexcept {
  for (int i = 0; i < n; ++i) {
    if (fp_bad(v[i])) return true;
  }
  return false;
}

}  // namespace detail
}  // namespace autonne

#endif  // AUTONNE_DETAIL_FP_BITS_HPP
