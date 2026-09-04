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

"""Differential test of autonne against LAPACK, over freshly drawn matrices.

The frozen corpus in tests/corpus pins a handful of matrices exactly, and the
test suite checks them on every build. This is the other half: a wide net,
thrown at run time, over matrices nobody chose. It writes each one out in
autonne's hex-float format, runs the dump executable on it, and compares the
spectrum against numpy -- zgesdd for singular values, zheevd for eigenvalues.

It is a developer tool rather than a test. It needs numpy, and its matrices
change with the seed, so a failure here is a lead to investigate and turn into
a fixed case in tests/corpus, not a red build.

Five families are drawn, each aimed at something the kernel claims:

  random      arbitrary shapes, no structure
  graded      columns scaled over thirty decades, where relative accuracy is
              the whole claim and an absolute-accuracy method fails
  rank        exact rank deficiency, from full rank down to the zero matrix
  degenerate  spectra drawn from four levels, so values repeat exactly
  scaled      the whole matrix multiplied by 2^k for |k| up to 900, near the
              ends of the exponent range

Usage:
  python tools/lapack_sweep.py build/autonne_dump_strict [--trials N] [--seed S]
"""

import argparse
import math
import os
import subprocess
import sys
import tempfile

import numpy as np

EPS = 2.220446049250313e-16


def hexfloat(x):
    x = float(x)
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return x.hex()


def write_matrix(path, m, name="m"):
    rows, cols = m.shape
    with open(path, "w", newline="\n") as f:
        f.write(f"autonne-hexfloat 1 matrix {name} {rows} {cols} colmajor\n")
        for j in range(cols):
            for i in range(rows):
                z = complex(m[i, j])
                f.write(f"{hexfloat(z.real)} {hexfloat(z.imag)}\n")


def read_leading_vector(path):
    with open(path) as f:
        tok = f.read().split()
    if tok[0] != "autonne-hexfloat" or tok[2] != "vector":
        raise ValueError(f"not a hex-float vector record: {tok[:3]}")
    n = int(tok[4])
    return np.array([float.fromhex(t) for t in tok[5:5 + n]])


def random_complex(rng, rows, cols):
    return rng.uniform(-1, 1, (rows, cols)) + 1j * rng.uniform(-1, 1, (rows, cols))


def make_case(rng, kind):
    if kind == "random":
        rows, cols = (int(v) for v in rng.integers(1, 40, size=2))
        return random_complex(rng, rows, cols)
    if kind == "graded":
        n = int(rng.integers(2, 25))
        return random_complex(rng, n, n) * np.power(10.0, -rng.uniform(0, 30, n))[None, :]
    if kind == "rank":
        n = int(rng.integers(2, 30))
        k = int(rng.integers(0, n + 1))
        u = np.linalg.qr(random_complex(rng, n, n))[0]
        v = np.linalg.qr(random_complex(rng, n, n))[0]
        s = np.zeros(n)
        s[:k] = 1.0
        return (u * s) @ v.conj().T
    if kind == "degenerate":
        n = int(rng.integers(2, 30))
        u = np.linalg.qr(random_complex(rng, n, n))[0]
        v = np.linalg.qr(random_complex(rng, n, n))[0]
        return (u * rng.choice([0.0, 0.25, 1.0, 4.0], size=n)) @ v.conj().T
    if kind == "scaled":
        n = int(rng.integers(2, 20))
        return random_complex(rng, n, n) * (2.0 ** int(rng.integers(-900, 900)))
    raise ValueError(kind)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump", help="path to autonne_dump_strict (or _fastmath)")
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=4242)
    args = parser.parse_args()

    # Absolute and normalised: on Windows CreateProcess does not reliably
    # accept a relative path spelled with forward slashes.
    dump = os.path.abspath(args.dump)
    if not os.path.exists(dump):
        print(f"no such executable: {dump}", file=sys.stderr)
        return 2

    rng = np.random.default_rng(args.seed)
    kinds = ["random", "graded", "rank", "degenerate", "scaled"]
    worst = {k: (0.0, None) for k in kinds}
    failures = []
    compared = 0

    with tempfile.TemporaryDirectory() as work:
        in_path = os.path.join(work, "in.txt")
        out_path = os.path.join(work, "out.txt")
        for trial in range(args.trials):
            kind = kinds[trial % len(kinds)]
            m = make_case(rng, kind)
            if not np.all(np.isfinite(m)):
                continue
            rows, cols = m.shape
            write_matrix(in_path, m)
            proc = subprocess.run([dump, in_path, out_path, "svd"],
                                  capture_output=True, text=True)
            if proc.returncode != 0:
                failures.append((kind, m.shape, "refused: " + proc.stderr.strip()))
                continue
            got = read_leading_vector(out_path)
            ref = np.linalg.svd(m, compute_uv=False)
            if got.shape != ref.shape:
                failures.append((kind, m.shape, f"shape {got.shape} vs {ref.shape}"))
                continue
            scale = ref[0] if ref[0] > 0 else 1.0
            err = float(np.max(np.abs(got - ref))) / scale
            compared += 1
            if err > worst[kind][0]:
                worst[kind] = (err, m.shape)
            # LAPACK promises its own values only to a modest multiple of eps
            # times the largest, so the comparison is absolute at that level.
            if err > 64 * max(rows, cols) * EPS:
                failures.append((kind, m.shape, f"relative spectrum error {err:.3e}"))

    print(f"compared {compared} matrices against numpy/LAPACK")
    for kind in kinds:
        err, shape = worst[kind]
        print(f"  {kind:11s} worst relative spectrum error {err:.3e}  at {shape}")
    if failures:
        print(f"\n{len(failures)} failures:")
        for f in failures[:20]:
            print("  ", f)
        return 1
    print("\nno failures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
