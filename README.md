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

Both kernels are implemented. The suite -- harness acceptance, exact spectra,
relative-accuracy bounds, and a frozen corpus cross-checked against LAPACK --
passes under strict and fast-math floating point on GCC 13/14, Clang 18/22,
MSVC 2022 and AppleClang. The interface is the one below and is not expected
to move. Measured against Eigen in [Performance](#performance).

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
- **Agreement with LAPACK.** `tests/corpus` holds fourteen matrices frozen as
  exact hex-float literals (`tools/make_corpus.py`), each with the spectrum
  numpy's `zgesdd` / `zheevd` computed for it. Every one is factored, judged
  by the harness, and compared with that reference to LAPACK's own absolute
  accuracy, in every build variant.

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

`bench/autonne_bench` (built with `-DAUTONNE_BUILD_BENCHMARKS=ON`, which
fetches Eigen 3.4.0 for that one target) times a thin SVD of a `2b × 2b`
matrix with a shaped spectrum, median of eleven calls, and passes every result
through the harness. Clang 22, `-O3`, strict floating point, one core of an
otherwise idle desktop; milliseconds:

| n   | spectrum       | autonne | Eigen BDCSVD  | Eigen JacobiSVD |
| --- | -------------- | ------: | ------------: | --------------: |
| 8   | decaying       |   0.007 |         0.018 |           0.017 |
| 16  | decaying       |   0.036 |         0.043 |           0.137 |
| 32  | decaying       |   0.32  |         0.34  |           1.23  |
| 64  | decaying       |   2.42  |         2.16  |          10.3   |
| 128 | decaying       |  17.3   |        12.8   |         101     |
| 8   | flat           |   0.005 |         0.037 |           0.037 |
| 16  | flat           |   0.039 |         0.063 |           0.37  |
| 32  | flat           |   0.19  |         0.26  |           3.28  |
| 64  | flat           |   1.21  |         1.85  |          27.7   |
| 128 | flat           |   7.05  |        10.7   |         283     |
| 8   | rank-deficient |   0.010 |         0.031 |           0.030 |
| 16  | rank-deficient |   0.050 | 0.046 (rejected) |        0.19  |
| 32  | rank-deficient |   0.37  |         0.24  |           1.55  |
| 64  | rank-deficient |   2.17  |         1.89  |          10.7   |
| 128 | rank-deficient |  19.4   | 11.7 (rejected) |       113     |

"Decaying" is a geometric spectrum over sixteen decades, "flat" is fully
degenerate, "rank-deficient" is half the spectrum degenerate and half exactly
zero. "Rejected" means the harness refused Eigen's factorisation: on the
rank-deficient input the divide-and-conquer result fails the backward-error
bound (residual 1.35 times the bound at 128×128), and on the 36×36 Simon coset
matrix it returns a spectrum with sum of squares 0.98611 against a norm of 1,
which is the defect the spec describes. autonne's factorisations were accepted
in every row.

Against the spec's bar -- the faster of Eigen's two methods -- autonne is
faster or equal up to 32×32 on every shape, faster on flat spectra at every
size, and within a factor of 1.5 on decaying and rank-deficient input at
128×128. The cost is dominated by Jacobi sweeps, each `O(n³)`; the pivoted QR
that precedes them is a few milliseconds at 128 and accounts for most of the
flat-spectrum time, where one or two sweeps suffice.

`eigh` is a plain cyclic Jacobi and pays for its accuracy guarantees: at
128×128 it takes 61 ms against 5 ms for Eigen's tridiagonalisation-based
solver, and about ten times longer at every size. A tridiagonal path would
close that gap for callers that do not need relative accuracy on graded input;
it is not implemented.

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

The test suite is not built when autonne is a subproject. `cmake --install`
exports `autonne::autonne` with a package config, so `find_package(autonne
CONFIG)` works too; CI builds `tests/consumer` that way, with `-ffast-math`,
against the installed tree.

The suite is built three times -- strict, fast-math, and strict with the
hand-rolled accessor path -- and `ctest` runs all of them. The benchmark is
opt-in (`-DAUTONNE_BUILD_BENCHMARKS=ON`) and is the only target that fetches
Eigen.

## Licence

Apache-2.0.
