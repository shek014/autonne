# autonne

[![CI](https://github.com/shek014/autonne/actions/workflows/ci.yml/badge.svg)](https://github.com/shek014/autonne/actions/workflows/ci.yml)

Thin singular value decomposition and Hermitian eigendecomposition for small
dense complex matrices.

Given a complex matrix, autonne returns its singular directions and the full
spectrum. It targets the case where matrices are small (n ≤ 128), shapes recur,
and the inputs are frequently rank-deficient with exactly degenerate spectra —
the regime of tensor-network bond splitting, though nothing in the interface is
specific to it.

Named for Léon Autonne, who extended the singular value decomposition to complex
matrices in 1915.

## Status

Both kernels are implemented and pass the verification harness under strict
and fast-math floating point on GCC 13/14 and Clang 18/22. The interface is
the one below and is not expected to move. Performance has not yet been
measured against a baseline; see [Performance](#performance).

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
caller should take its fallback route; nothing has been written to the outputs.

`eigh` returns eigenvalues in ascending order, following the LAPACK convention.
It decomposes the Hermitian part `(A + A^*)/2` of its input, which for
Hermitian input is the input itself. `evecs_out` may be null when only the
spectrum is wanted.

Raw buffers cross the boundary, and no public header names a third-party type.

## What the kernels do

**`svd_thin`.** Rows and columns that are exactly zero are set aside first:
they are structural, their singular values are exactly zero and their singular
vectors are canonical basis vectors. The rest is scaled by a power of two, then
factored as `A P = Q R` by Householder QR with column pivoting — in whichever of
`A` and `A^*` carries its scaling in the columns, since a left-applied
Householder QR preserves column scalings and destroys row scalings. One-sided
Jacobi is then run on the columns of `R^*`: rows of `R` that pivoting has left
exactly zero become exactly zero singular values, and the rest, ordered by
pivoting, are what Drmač and Veselić showed Jacobi converges on quickly and
with high relative accuracy. `U = Q V_X` and `V = P U_X` are assembled from the
accumulated rotations, sorted, unscaled, and scanned for non-finite values
before anything is written.

**`eigh`.** The Hermitian part is formed with a real diagonal, structurally
zero rows are set aside, the matrix is scaled by a power of two, and cyclic
Jacobi is run with the rotation threshold relative to the diagonal
(`|H(p,q)| > sqrt(n) u sqrt(|H(p,p) H(q,q)|)`), which is what gives the small
eigenvalues of a graded positive definite matrix their relative accuracy. The
rotations accumulate into `Q`; the matrix is kept exactly Hermitian throughout.

Both are `O(sweeps · n³)` and allocate their workspace on the heap; an
allocation failure is reported as `false`.

## What is guaranteed

Every claim below is a test in `tests/test_svd.cpp` or `tests/test_eigh.cpp`,
run under both floating-point models.

- **The harness accepts every factorisation.** `verify::check_svd` and
  `verify::check_eigh` bound the backward error, the orthonormality of every
  factor, the spectral energy identity and the ordering, with factors of
  `64 · max(dimension) · eps`.
- **No non-finite value is ever returned.** Non-finite input is refused by bit
  pattern. Inside the kernel nothing can produce a NaN or an infinity: inputs
  are scaled to unit size, every divisor is either bounded below by construction
  or guarded by a floor, and the outputs are scanned once more before they are
  written.
- **Structural zeros are exact.** A zero row or column yields a singular value
  of exactly `0.0` and a canonical basis vector. The 36×36 rank-12 matrix of
  the Simon problem (twelve-fold degenerate, the shape on which Eigen 3.4.0's
  divide-and-conquer SVD returned a wrong spectrum) comes back with twelve
  copies of `1/(2√3)` and twenty-four exact zeros.
- **Power-of-two scaling is exact.** `svd_thin(2^k A)` returns `2^k S` and the
  same `U` and `V` bit for bit.
- **Storage order is invisible.** The same logical matrix in row-major and
  column-major order gives identical results bit for bit; so do repeated calls.
- **Relative accuracy on graded input.** For `A = B D` or `A = D B` with the
  singular values of `B` in `[0.9, 1.1]`, every singular value of `A` lands in
  `[0.9, 1.1]` times the matching entry of `D`, tested down to `1e-70`. For
  `H = D B D` with `B` positive definite in the same sense, every eigenvalue
  lands in `[0.9, 1.1]` times the matching `d_i²`, tested down to `1e-56`. A
  method with only absolute accuracy `eps · ‖A‖` fails these as soon as the
  scaled values drop below `eps`.

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
value is assumed finite even by an integer test on its bits, and a NaN built
as a `double` expression in a fast-math translation unit does not reliably
reach memory at all. The tests therefore inject non-finite values as integer
bit patterns. The suite is built under strict and fast-math floating point
(and once more with the hand-rolled accessor path), and every variant must
pass.

**One definition per entry point.** `svd_thin`, `eigh`, `check_svd` and
`check_eigh` are compiled in the library, not instantiated from headers. A
header-only definition is emitted by every including file and the linker keeps
one copy per binary without regard to the flags it was built under, so a
consumer mixing `-ffast-math` and strict files would run whichever copy won the
link. The kernels' private helpers have internal linkage for the same reason.
CI checks the linkage with `nm` (`tools/check_symbols.sh`).

**Accuracy on degenerate and rank-deficient input.** Repeated singular values
and hard zero blocks are the common case here, not the exception. Correctness on
such matrices is the primary design target, which is why the kernel is Jacobi
rather than bidiagonalisation plus QR or divide-and-conquer: rotations keep
every factor orthonormal by construction, whatever the spectrum does.

**Verification is the caller's, not ours.** Consumers are expected to check every
factorisation against the matrix it came from. autonne is judged by that check
rather than trusted in place of it:

```cpp
#include <autonne/autonne.hpp>
#include <autonne/verify.hpp>

const int k = std::min(rows, cols);
if (!autonne::svd_thin(M, rows, cols, order, U, S, V)) { /* fallback */ }
const autonne::verify::SvdReport r =
    autonne::verify::check_svd(M, rows, cols, order, U, S, V, k);
if (!r.ok()) { /* reject; r says which bound moved */ }
```

`check_svd` accepts `k < min(rows, cols)` for a truncated slice, comparing the
residual in amplitude form against `sqrt(discarded) + 64 · max(rows, cols) · eps · ‖M‖_F`.

## Performance

Not yet measured against a baseline. Jacobi costs `O(sweeps · n³)` with a
handful of sweeps after the pivoted QR, and a rank-deficient input costs
roughly `rank²` rather than `n²` per sweep because zero columns of `R^*` are
never rotated. The reference this is meant to be compared with (Eigen's
divide-and-conquer at about 5.5 ms for a 128×128 complex matrix) is set out in
[verycareful/lindblad#95](https://github.com/verycareful/lindblad/issues/95);
a benchmark target with that baseline is the next piece of work.

## Building

Requires a C++23 compiler and CMake 3.20+. No external dependencies; GoogleTest
is fetched for the test suite.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build
```

To consume from another CMake project:

```cmake
include(FetchContent)
FetchContent_Declare(autonne
  GIT_REPOSITORY https://github.com/shek014/autonne.git
  GIT_TAG <pinned tag>)
FetchContent_MakeAvailable(autonne)
target_link_libraries(your_target PRIVATE autonne::autonne)
```

The test suite is not built when autonne is a subproject.

## Licence

Apache-2.0.
