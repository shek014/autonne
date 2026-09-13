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

Three entry points are implemented: two thin-SVD kernels and the Hermitian
eigensolver. The suite (harness acceptance, exact spectra, relative-accuracy
bounds, and a frozen corpus cross-checked against LAPACK) passes under strict
and fast-math floating point on GCC 13/14, Clang 18/22,
MSVC 2022 and AppleClang. The interface is the one below and is not expected
to move. Measured against Eigen in [Performance](#performance).

## Scope

Three operations, over `std::complex<double>`:

```cpp
enum class MatrixOrder { RowMajor, ColMajor };

bool svd_thin(const std::complex<double>* data, int rows, int cols,
              MatrixOrder order,
              std::complex<double>* U_out, double* S_out,
              std::complex<double>* V_out);

bool svd_thin_bdc(const std::complex<double>* data, int rows, int cols,
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
`svd_thin_bdc` takes the same arguments, returns the same layout and fails the
same way; it differs in what accuracy it promises and what it costs, which is
the whole of the next section. There is no entry point that chooses between
the two: nothing a caller can check on the result distinguishes them, so the
choice has to be made from what the caller knows about the input.

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

**`svd_thin_bdc`.** The same frame (structural zeros set aside, power-of-two
scaling, the scan on the way out), then Householder bidiagonalisation, after an
unpivoted QR when the block is more than 1.6 times as tall as it is wide. The
complex bidiagonal's phases are absorbed into two diagonal unitaries so that a
real non-negative bidiagonal remains, and that is solved by Gu and Eisenstat's
divide and conquer: a block of order at most eight is solved densely (pivoted
QR and one-sided Jacobi), a larger one is split at its middle row, and the
halves are merged through the secular equation, with negligible and repeated
entries deflated before the solve and the singular vectors rebuilt from the
values actually computed (Löwner's theorem), which is what keeps them
orthonormal when two values are close. The reflectors are applied to the
factors directly on the way back; neither `Q` nor `P` is formed. The
bidiagonalisation is backward stable in the norm of the whole matrix rather
than column by column, so every singular value carries an error of order
`eps · ‖A‖`: absolute accuracy, where `svd_thin` gives relative. On a
128×128 block it is 2.7 times faster than `svd_thin` on a decaying spectrum
and no faster on a flat one, where Jacobi needs few sweeps.

**`eigh`.** The Hermitian part is formed with a real diagonal, structurally
zero rows are set aside, the matrix is scaled by a power of two, and cyclic
Jacobi is run with the rotation threshold relative to the diagonal
(`|H(p,q)| > sqrt(n) u sqrt(|H(p,p) H(q,q)|)`), which is what gives the small
eigenvalues of a graded positive definite matrix their relative accuracy. The
rotations accumulate into `Q`; the matrix is kept exactly Hermitian throughout.

`svd_thin` and `eigh` are `O(sweeps · n³)`; `svd_thin_bdc` is `O(n³)` with a
constant that does not depend on the spectrum. All three allocate their
workspace on the heap; an allocation failure is reported as `false`.

## What is guaranteed

Every claim below is a test in `tests/test_svd.cpp`, `tests/test_svd_bdc.cpp`
or `tests/test_eigh.cpp`, run under both floating-point models. Every claim
holds for both SVD kernels except relative accuracy, which is `svd_thin`'s and
`eigh`'s alone: `svd_thin_bdc` promises `|s_i - s_i(true)| <= 64 · max(dimension)
· eps · s_max` on every value instead, and `tests/test_svd_bdc.cpp` pins exactly
that bound and nothing sharper.

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
- **Relative accuracy on graded input** (`svd_thin` and `eigh`). For `A = B D` or `A = D B` with the
  singular values of `B` in `[0.9, 1.1]`, every singular value of `A` lands in
  `[0.9, 1.1]` times the matching entry of `D`, tested down to `1e-70`. For
  `H = D B D` with `B` positive definite in the same sense, every eigenvalue
  lands in `[0.9, 1.1]` times the matching `d_i²`, tested down to `1e-56`. A
  method with only absolute accuracy `eps · ‖A‖` fails these as soon as the
  scaled values drop below `eps`.
- **Agreement with LAPACK.** `tests/corpus` holds fourteen matrices frozen as
  exact hex-float literals (`tools/make_corpus.py`), each with the spectrum
  numpy's `zgesdd` / `zheevd` computed for it. Every one is factored (the SVD
  matrices by both kernels), judged by the harness, and compared with that
  reference to LAPACK's own absolute accuracy, in every build variant. The
  nineteen two-site blocks recorded from a matrix-product-state simulation
  (`tests/corpus/lindblad_mps`) carry no reference; there both kernels must
  be accepted by the harness and by the screen, and agree with each other.
- **The two floating-point models agree.** The kernels are also built into one
  small executable per model, both are run on every corpus matrix through
  every kernel, and a third program diffs what they wrote. Bit equality is not required and would be the
  wrong test, since `-ffast-math` may reassociate; what is required is that the
  spectra agree to `1e-11` relative and that each singular vector agrees up to
  a phase, with degenerate groups excluded because any unitary mixing inside
  one is a correct answer. Measured divergence is around `1e-14` on the spectra
  and `1e-8` on the vectors.
- **Agreement with LAPACK beyond the frozen set.** `tools/lapack_sweep.py`
  draws matrices nobody chose -- arbitrary shapes, columns graded over thirty
  decades, exact rank deficiency down to the zero matrix, spectra repeated
  exactly, and whole matrices multiplied by `2^k` for `|k|` up to 900 -- and
  compares each spectrum against numpy. Over 1700 of them across the strict
  build, the fast-math build and an MSVC build, on five seeds, the worst
  relative disagreement was `4.5e-15`, about twenty ulps, and the fast-math
  build was no less accurate than the strict one. It is a developer tool, not
  a CI test: a failure there is a lead to turn into a fixed case in
  `tests/corpus`.
- **The screen agrees with the harness.** `verify::screen_svd` gives the
  harness's verdicts at `O(rows · cols)`: the energy identity and the spectrum
  checks exactly, the residual and the orthonormality defects through probe
  vectors whose products estimate the norms the harness computes. It accepts
  what is correct, from both kernels at every size in the suite and at every
  scale the harness handles; it rejects each corruption the harness's own tests
  apply, including the one only a residual can see (two singular vectors
  exchanged); and over 480 cases drawn from both kernels with each corruption
  applied, its verdict equals the harness's in every one
  (`tests/test_verify_screen.cpp`).
- **Subnormal entries are handled, not stumbled over.** An entry at the bottom
  of the subnormal range beside ordinary ones, or a scaling that drives one
  there, is where a phase computed as `z / |z|` stops being unimodular and a
  rotation stops being unitary. Both kernels lift such a value into the normal
  range before dividing, and never form a quotient whose denominator can
  underflow. Checked over 490,000 matrices in both floating-point models, on
  Clang and MSVC, with no refusal and no factorisation the harness rejects.
- **Any scale.** The harness measures on the input scaled by a power of two,
  applied with `ldexp` so the scaling itself is representable at any exponent.
  It therefore neither overflows on a matrix near `1e210` nor, more
  dangerously, underflows to all-zeros on one near `1e-170` and accepts
  whatever it is handed. Both directions are tested, in both floating-point
  models. One limit is inherent rather than fixable: for a matrix whose
  largest component is within a few ulps of the bottom of the subnormal range,
  the true spectrum is not representable, and the harness's relative bound is
  smaller than the one-ulp absolute error any implementation is forced into.

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

**One definition per entry point.** `svd_thin`, `svd_thin_bdc`, `eigh`,
`check_svd`, `screen_svd` and `check_eigh` are compiled in the library, not
instantiated from headers. A
header-only definition is emitted by every including file and the linker keeps
one copy per binary without regard to the flags it was built under, so a
consumer mixing `-ffast-math` and strict files would run whichever copy won the
link. The kernels' private helpers have internal linkage for the same reason.
CI checks the linkage with `nm` (`tools/check_symbols.sh`).

**A strict judge.** One definition is not enough on its own: it does not say
which one. The harness's residual, norms and orthonormality sums are ordinary
reductions, so a `-ffast-math` build of `verify.cpp` may reassociate them and
reach a different number, compared against the same fixed tolerance. That is
tolerable in a kernel, whose output the harness judges afterwards; it is not
tolerable in the harness, because nothing judges the judge. `src/verify.cpp` is
therefore compiled with strict floating point in every build variant, including
the fast-math one, so that a fast-math kernel is judged by a strict gate.
`verify::built_with_fast_math()` reports the model that file was actually
compiled under — the answer exists only inside that translation unit — and a
test asserts it is false everywhere, so removing the build rule fails the suite
rather than quietly shifting a number.

**Accuracy on degenerate and rank-deficient input.** Repeated singular values
and hard zero blocks are the common case here, not the exception. Correctness on
such matrices is the primary design target, which is why the default kernel,
`svd_thin`, is Jacobi rather than bidiagonalisation plus QR or divide and
conquer: rotations keep every factor orthonormal by construction, whatever the
spectrum does. `svd_thin_bdc` reaches the same verdicts by a different route,
deflation before the secular solve and singular vectors rebuilt from the values
actually computed, and is held to the same harness on the same matrices,
including the ones on which Eigen's divide and conquer is rejected. What it
gives up is the relative accuracy above, which no harness can see, and which
is why it is a separate name rather than a faster path behind `svd_thin`.

**A comparison between floating-point models has to cross a process boundary.**
Two variants of the same code linked into one binary do not measure two
floating-point models. The inline helpers have vague linkage, the linker keeps
one copy of each without regard to the flags it was built under, and the
comparison ends up reading link order. So `autonne_dump` is built once per
model, both executables are run on the same frozen matrix, and `autonne_compare`
diffs the files. It is the same argument as the one about entry points, applied
to measurement rather than to shipping code.

**Verification is the caller's, not ours.** Consumers are expected to check every
factorisation against the matrix it came from. autonne is judged by that check
rather than trusted in place of it:

```cpp
#include <autonne/autonne.hpp>
#include <autonne/verify.hpp>

const int k = std::min(rows, cols);
if (!autonne::svd_thin(M, rows, cols, order, U, S, V)) {
  return fallback();  // U, S and V have not been written
}
const autonne::verify::SvdReport r =
    autonne::verify::check_svd(M, rows, cols, order, U, S, V, k);
if (!r.ok()) {
  return fallback();  // r says which bound moved and by how much
}
```

`check_svd` accepts `k < min(rows, cols)` for a truncated slice, comparing the
residual in amplitude form against `sqrt(discarded) + 64 · max(rows, cols) · eps · ‖M‖_F`.

`check_svd` costs `O(rows · cols · k)`, which at 128×128 is about as long as
the factorisation itself. `verify::screen_svd` takes the same arguments and
returns the same verdicts at `O(rows · cols)`: the energy identity exactly, the
residual and the orthonormality defects estimated through probe vectors with
entries in `{1, -1, i, -i}`, for which `E ‖X y‖² = ‖X‖_F²`, so a matrix-vector
product stands in for a matrix product and the estimate is compared with the
harness's own bound. The residual is probed in the kept subspaces, `M V_k`
against `U_k S_k` and `M^* U_k` against `V_k S_k`, which is what makes a
truncated slice screenable as sharply as a full one. Three probes by default;
a defect the harness would reject is missed only when every probe lands well
under the norm it estimates, which for a defect spread over many directions
happens about once in a million and for one aligned with a coordinate
direction never. At 128×128 the screen takes 0.28 ms against 6 ms for the
harness. The probe vectors come from a fixed seed, so the same inputs give the
same report everywhere.

## Performance

`bench/autonne_bench` (built with `-DAUTONNE_BUILD_BENCHMARKS=ON`, which
fetches Eigen 3.4.0 for that one target) times a thin SVD of a `2b × 2b`
matrix with a shaped spectrum, median of fifteen calls, and passes every result
through the harness. Ryzen 9 7900X, Clang 22.1.8, `-O3`, strict floating
point, one core of an otherwise idle desktop; milliseconds:

| n   | spectrum       | autonne | autonne bdc | Eigen BDCSVD    | Eigen JacobiSVD |
| --- | -------------- | ------: | ----------: | --------------: | --------------: |
| 8   | decaying       |   0.007 |       0.007 |           0.015 |           0.015 |
| 16  | decaying       |   0.039 |       0.025 |           0.030 |           0.118 |
| 32  | decaying       |   0.26  |       0.12  |           0.16  |           1.06  |
| 64  | decaying       |   1.69  |       0.69  |           1.14  |           8.07  |
| 128 | decaying       |  12.4   |       4.67  |           6.60  |          88.8   |
| 8   | flat           |   0.003 |       0.004 |           0.026 |           0.026 |
| 16  | flat           |   0.016 |       0.018 |           0.018 |           0.26  |
| 32  | flat           |   0.089 |       0.087 |           0.081 |           2.23  |
| 64  | flat           |   0.56  |       0.55  |           0.84  |          18.9   |
| 128 | flat           |   4.35  |       4.39  |           5.70  |         229     |
| 8   | rank-deficient |   0.006 |       0.006 |           0.019 |           0.019 |
| 16  | rank-deficient |   0.037 |       0.022 |           0.019 |           0.12  |
| 32  | rank-deficient |   0.23  |       0.093 |           0.083 |           1.16  |
| 64  | rank-deficient |   1.65  |       0.57  | 0.87 (rejected) |           7.63  |
| 128 | rank-deficient |  11.8   |       4.50  |           5.84  |          88.2   |

Run-to-run variation on a desktop is around twenty percent, so treat a
difference smaller than that as noise; the ordering is stable across runs.

"Decaying" is a geometric spectrum over sixteen decades, "flat" is fully
degenerate, "rank-deficient" is half the spectrum degenerate and half exactly
zero. "Rejected" means the harness refused Eigen's factorisation: on the
rank-deficient input at 64×64 the divide-and-conquer result carries non-finite
values, and on the 36×36 Simon coset matrix it returns a spectrum with sum of
squares 0.98611 against a norm of 1, which is the defect the spec describes.
Which rows are rejected moves with the machine: an earlier run of the same
benchmark on a different desktop, same compiler and flags, rejected the 16 and
128 rows instead. `svd_thin` was accepted in every row on both machines, and
`svd_thin_bdc` in every row here.

Against the spec's bar, the faster of Eigen's two methods: `svd_thin_bdc` is
faster from 64×64 up on every spectrum, by 1.3 to 1.4 times at 128×128, and
within noise of or up to 1.2 times behind Eigen's divide and conquer at 16 and
32 on the flat and rank-deficient spectra, where that method solves the whole
block densely and this one already splits it. `svd_thin` is faster than either
Eigen method at 8×8, and faster or within noise at every size on the flat
spectrum; on the decaying spectrum it is within a factor of 1.9 of the faster
Eigen method at every size, and on the rank-deficient one within 2.0 except at
32×32, where it is 2.8 times slower. Its cost is the Jacobi sweeps, each
`O(n³)`, which the spectrum decides; `svd_thin_bdc`'s cost is three quarters
bidiagonalisation and back-transformation and one quarter the divide and
conquer itself, and the spectrum barely moves it.

Which sizes matter is a question about the caller's workload, and for one
workload there is a measurement. `tests/corpus/lindblad_mps` holds the
two-site blocks a matrix-product-state simulator recorded while running a
24-qubit brickwork circuit at a bond cap of 64: 276 splits over 19 shapes,
one representative matrix per shape, with the counts in that directory's
README. `autonne_bench --mps tests/corpus/lindblad_mps` times every method on
every shape and weights each by its count; the four heaviest shapes and the
weighted total, same machine, milliseconds:

| block   | count | autonne | autonne bdc | Eigen BDCSVD | Eigen JacobiSVD |
| ------- | ----: | ------: | ----------: | -----------: | --------------: |
| 16×16   |    27 |   0.049 |       0.025 |        0.029 |           0.145 |
| 32×32   |    24 |   0.32  |       0.14  |        0.21  |           1.30  |
| 64×64   |    21 |   2.19  |       0.78  |        1.39  |          12.1   |
| 128×128 |    33 |  17.6   |       5.05  |        7.24  |         259     |
| all 276 |   276 | 667     |     200     |      290     |        8930     |

Every shape was accepted by the harness for every method on this run. The
saturated 128×128 blocks are an eighth of the splits and three quarters of the
time, which is why the workload total follows the 128 column of the first
table. The broader question of which sizes a consuming project reaches at all
is open as
[verycareful/lindblad#100](https://github.com/verycareful/lindblad/issues/100).

`eigh` is a plain cyclic Jacobi and pays for its accuracy guarantees: at
128×128 it takes about 20 ms against 2.5 ms for Eigen's tridiagonalisation-based
solver, and between three and eight times longer at the smaller sizes. A
tridiagonal path would
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
hand-rolled accessor path -- and `ctest` runs all of them, along with the
cross-binary comparison of the two floating-point models. The benchmark is
opt-in (`-DAUTONNE_BUILD_BENCHMARKS=ON`) and is the only target that fetches
Eigen.

`-DAUTONNE_SANITIZERS=address,undefined` builds everything under those
sanitizers with `-fno-sanitize-recover`, which is how CI runs the suite on
one job.

## Licence

Apache-2.0.
