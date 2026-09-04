# autonne

[![CI](https://github.com/shek014/autonne/actions/workflows/ci.yml/badge.svg)](https://github.com/shek014/autonne/actions/workflows/ci.yml)

Truncated singular value decomposition and Hermitian eigendecomposition for
small dense complex matrices.

Given a complex matrix, autonne returns its singular directions and the full
spectrum. It targets the case where matrices are small (n ≤ 128), shapes recur,
and the inputs are frequently rank-deficient with exactly degenerate spectra —
the regime of tensor-network bond splitting, though nothing in the interface is
specific to it.

Named for Léon Autonne, who extended the singular value decomposition to complex
matrices in 1915.

## Status

Early. The interface and verification harness are in place; the decomposition
kernel is not yet implemented. Not usable yet.

## Scope

Two operations, over `std::complex<double>`:

```cpp
enum class MatrixOrder { RowMajor, ColMajor };

bool svd_thin(const std::complex<double>* data, int rows, int cols,
              MatrixOrder order,
              std::complex<double>* U_out, double* S_out,
              std::complex<double>* V_out);

bool eigh(const std::complex<double>* data, int n, MatrixOrder order,
          double* evals_out, std::complex<double>* evecs_out);
```

With `k = min(rows, cols)`: `U_out` is `rows` by `k`, `V_out` is `cols` by `k`,
both column-major, and `S_out` holds `k` singular values in descending order.
`V` is returned as `V`, not conjugate-transposed. A `false` return means the
caller should take its fallback route and must not read the outputs.

`eigh` returns eigenvalues in ascending order, following the LAPACK convention.

Raw buffers cross the boundary, and no public header names a third-party type.

## Design constraints

**The full spectrum, not just the kept part.** Callers that truncate need the
discarded weight summed from the individual small values. Computing it as
`total - kept` fails by cancellation: on a normalised input both are near 1.0
while the true difference can be near 1e-30. This rules out methods that never
form the tail.

**Guards that survive `-ffast-math`.** Under that flag a compiler may assume no
infinity or NaN exists, so `isfinite` folds to a constant and the check becomes
dead code. autonne's finiteness guards read the exponent field of the object
representation in memory instead, by reference and never through a by-value
`double`: from Clang 22 on, a value that has crossed a function boundary by
value is assumed finite even by an integer test on its bits. The test suite is
built under strict and fast-math floating point (and once more with the
hand-rolled accessor path), and every variant must pass.

**One definition per entry point.** `svd_thin`, `eigh`, `check_svd` and
`check_eigh` are compiled in the library, not instantiated from headers. A
header-only definition is emitted by every including file and the linker keeps
one copy per binary without regard to the flags it was built under, so a
consumer mixing `-ffast-math` and strict files would run whichever copy won the
link. CI checks the linkage with `nm`.

**Accuracy on degenerate and rank-deficient input.** Repeated singular values
and hard zero blocks are the common case here, not the exception. Correctness on
such matrices is the primary design target.

**Verification is the caller's, not ours.** Consumers are expected to check every
factorisation against the matrix it came from. autonne is judged by that check
rather than trusted in place of it.

## Building

Requires a C++23 compiler and CMake 3.20+. No external dependencies; GoogleTest
is fetched for the test suite.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build
```

## Licence

Apache-2.0.