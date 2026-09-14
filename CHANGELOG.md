# Changelog

Notable changes to autonne. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html) with 0.x
semantics: while the major version is `0`, the **minor** is the
breaking-change signal and the patch is everything else. The installed package
config is generated with `COMPATIBILITY SameMinorVersion` to match, so a
consumer pinned to `0.2` will not silently accept an installed `0.3`.

## [Unreleased]

## [0.2.0] - 2026-09-14

The first release with working kernels. `0.1.0` was carried unchanged from the
initial commit through everything below, so it meant "stubs returning `false`"
and "a working SVD and Hermitian eigensolver" at different times; that is what
this scheme exists to stop. Earlier states are not tagged, because none of them
was installable.

### Added

- `svd_thin`, a thin SVD of a dense complex matrix by Householder QR with
  column pivoting followed by one-sided Jacobi. Keeps relative accuracy on
  graded input, tested to `1e-70`, and returns exact zeros and canonical
  vectors for structurally zero rows and columns.
- `eigh`, a Hermitian eigendecomposition by cyclic Jacobi, eigenvalues
  ascending. `evecs_out` may be null when only the spectrum is wanted, and the
  eigenvalues are then bit-identical to the full call.
- `verify::check_svd` and `verify::check_eigh`, the verification harness
  callers are expected to run on every factorisation: backward error in
  amplitude form, orthonormality of every factor, the spectral energy
  identity, ordering, and a non-finite scan by bit pattern.
- `verify::built_with_fast_math`, which reports the floating-point model
  `src/verify.cpp` was itself compiled under.
- Hex-float matrix I/O (`autonne/hexfloat.hpp`), so a frozen corpus round-trips
  bit for bit.
- A frozen test corpus cross-checked against LAPACK through numpy, and 19
  matrix-product-state blocks captured from a running simulation.
- A benchmark against Eigen's `BDCSVD`, `JacobiSVD` and
  `SelfAdjointEigenSolver`, behind `AUTONNE_BUILD_BENCHMARKS` (off by default;
  Eigen is never a dependency of the library or the tests).
- Install and package-config rules, behind `AUTONNE_INSTALL`.

### Fixed

- The harness silently accepted **any** factorisation of a matrix whose entries
  were below about `1e-170`: every squared quantity underflowed, so the
  residual, the bounds and the energies were all exactly `0.0` and every
  comparison was `0 <= 0`. It also rejected correct factorisations above about
  `1e154`, where the sums of squares overflowed. Both are fixed by measuring on
  the input scaled by an exact power of two.
- Non-finite guards are defeated under `-ffast-math` on Clang 22 when a value
  crosses a function boundary by value: the compiler assumes it finite and
  folds an exponent-field test to a constant. The guards now take their
  argument by reference and read the object representation in memory.
- The orientation heuristic in `svd_thin` formed a ratio of magnitudes that
  overflowed to infinity for a matrix holding a subnormal beside an ordinary
  value, in a file whose correctness argument requires that no infinity is ever
  produced. It compares binary exponents instead.

### Changed

- `svd_thin`, `eigh`, `check_svd` and `check_eigh` are compiled in the library
  rather than instantiated from headers, so each has exactly one definition per
  binary and the floating-point flag travels with the code. CI verifies the
  linkage with `nm`.
- `src/verify.cpp` is compiled with strict floating point in every build
  variant, including the fast-math one, so the acceptance gate's own arithmetic
  cannot be reassociated by flags that say nothing about the factorisation
  under test.

[Unreleased]: https://github.com/shek014/autonne/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/shek014/autonne/releases/tag/v0.2.0
