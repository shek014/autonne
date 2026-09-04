#!/usr/bin/env python3
# Copyright 2026 The autonne Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Writes the frozen corpus under tests/corpus, with LAPACK reference spectra.

Each file holds one matrix in autonne's hex-float format (see
include/autonne/hexfloat.hpp) followed by a reference spectrum computed by
numpy -- LAPACK zgesdd for singular values, zheevd for eigenvalues -- so the
test suite has an answer from an independent implementation to check against.
Everything is written as C99 hex-float literals and round-trips bit for bit.

The matrices are the structured cases from the spec (the Simon coset matrix,
the poison-theta analogue), graded matrices, and a few random shapes. They
are deterministic: rerunning this script reproduces the files exactly, given
the same numpy/LAPACK build for the reference spectra.

Usage: python tools/make_corpus.py [tests/corpus]
"""

import math
import os
import sys

import numpy as np

MAGIC = "autonne-hexfloat"
VERSION = 1


def hexfloat(x: float) -> str:
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return float(x).hex()


def write_matrix(out, m: np.ndarray, name: str) -> None:
    rows, cols = m.shape
    out.write(f"{MAGIC} {VERSION} matrix {name} {rows} {cols} colmajor\n")
    for j in range(cols):
        for i in range(rows):
            z = complex(m[i, j])
            out.write(f"{hexfloat(z.real)} {hexfloat(z.imag)}\n")


def write_vector(out, v: np.ndarray, name: str) -> None:
    out.write(f"{MAGIC} {VERSION} vector {name} {len(v)}\n")
    for x in v:
        out.write(f"{hexfloat(float(x))}\n")


def simon_coset_matrix(residue: float = 0.0) -> np.ndarray:
    d, s = 6, (2, 4)
    n = d * d
    m = np.zeros((n, n), dtype=complex)
    for x in range(n):
        x0, x1 = x % d, x // d
        best = (x0, x1)
        for k in range(1, d):
            cand = ((x0 + k * s[0]) % d, (x1 + k * s[1]) % d)
            if cand < best:
                best = cand
        y = best[0] + d * best[1]
        m[x, y] = 1.0 / 6.0
    m[0, 0] += residue
    return m


def poison_theta_like() -> np.ndarray:
    m = np.zeros((8, 8), dtype=complex)
    h = 0.70710678118654752440
    for j in range(4):
        m[2 * j, j] = h
        m[2 * j + 1, j] = 1j * h
    m[0, 4] = 1.57e-65
    m[7, 5] = 2.2e-49j
    return m


def dft(n: int) -> np.ndarray:
    j = np.arange(n)
    return np.exp(-2j * np.pi * np.outer(j, j) / n) / math.sqrt(n)


def well_conditioned(n: int, rng: np.random.Generator) -> np.ndarray:
    e = rng.uniform(-1, 1, (n, n)) + 1j * rng.uniform(-1, 1, (n, n))
    e *= 0.1 / np.linalg.norm(e)
    return dft(n) @ (np.eye(n) + e)


def random_complex(rows: int, cols: int, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(-1, 1, (rows, cols)) + 1j * rng.uniform(-1, 1, (rows, cols))


def hermitian(n: int, rng: np.random.Generator) -> np.ndarray:
    x = random_complex(n, n, rng)
    h = 0.5 * (x + x.conj().T)
    np.fill_diagonal(h, np.real(np.diag(h)))
    return h


def main() -> int:
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join("tests", "corpus")
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(20260904)

    svd_cases = {
        "svd_simon36": simon_coset_matrix(),
        "svd_simon36_residue": simon_coset_matrix(3.4e-17),
        "svd_poison8": poison_theta_like(),
        "svd_dft16": dft(16),
        "svd_random_16x9": random_complex(16, 9, rng),
        "svd_random_9x16": random_complex(9, 16, rng),
        "svd_random_32x32": random_complex(32, 32, rng),
        "svd_column_scaled_8": well_conditioned(8, rng) @ np.diag([1.0, 1e-3, 1e-6, 1e-9, 1e-12, 1e-15, 1e-18, 1e-21]),
        "svd_row_scaled_8": np.diag([1e-21, 1e-18, 1e-15, 1e-12, 1e-9, 1e-6, 1e-3, 1.0]) @ well_conditioned(8, rng),
    }
    # A rank-deficient 24x24 with zero rows and columns and a 4-fold
    # degenerate nonzero part.
    q1 = np.linalg.qr(random_complex(24, 24, rng))[0]
    q2 = np.linalg.qr(random_complex(24, 24, rng))[0]
    s = np.zeros(24)
    s[:4] = 0.5
    s[4:10] = 0.25
    m = (q1 * s) @ q2.conj().T
    m[[3, 11, 19], :] = 0.0
    m[:, [0, 5, 22]] = 0.0
    svd_cases["svd_zero_rows_cols_24"] = m

    for name, mat in svd_cases.items():
        ref = np.linalg.svd(mat, compute_uv=False)
        with open(os.path.join(out_dir, name + ".txt"), "w", newline="\n") as f:
            write_matrix(f, mat, name)
            write_vector(f, ref, "lapack_singular_values")

    eigh_cases = {
        "eigh_random_24": hermitian(24, rng),
        "eigh_random_64": hermitian(64, rng),
    }
    d = np.array([1.0, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14])
    b = hermitian(8, rng)
    b *= 0.1 / np.linalg.norm(b)
    b += np.eye(8)
    eigh_cases["eigh_graded_pd_8"] = (d[:, None] * b) * d[None, :]
    # Exactly degenerate: Q diag(1,1,1,1,-2,-2,0,0,...) Q^*.
    q = np.linalg.qr(random_complex(12, 12, rng))[0]
    lam = np.array([1.0] * 4 + [-2.0] * 2 + [0.0] * 6)
    eigh_cases["eigh_degenerate_12"] = (q * lam) @ q.conj().T

    for name, mat in eigh_cases.items():
        # zheevd through numpy; the input is Hermitian to rounding, and
        # numpy reads one triangle, so symmetrise exactly first to match what
        # autonne decomposes.
        mat = 0.5 * (mat + mat.conj().T)
        np.fill_diagonal(mat, np.real(np.diag(mat)))
        ref = np.linalg.eigvalsh(mat)
        with open(os.path.join(out_dir, name + ".txt"), "w", newline="\n") as f:
            write_matrix(f, mat, name)
            write_vector(f, ref, "lapack_eigenvalues")

    print(f"wrote {len(svd_cases) + len(eigh_cases)} files to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
