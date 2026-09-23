# AD Library Benchmarks

> **Quant-finance benchmarks for C++ automatic differentiation libraries.** Heston MC, SABR calibration, XVA CVA, and LIBOR swaption — finite differences, [XAD](https://github.com/auto-differentiation/xad), [CppAD](https://github.com/coin-or/CppAD), [Adept 2](https://github.com/rjhogan/Adept-2), [autodiff](https://github.com/autodiff/autodiff), [ddx](https://github.com/reach2sayan/ddx). Reproducible from source.

![Benchmark chart](results/chart.png)

*GCC 13.3, Intel Xeon Platinum 8488C, Ubuntu 24.04, `-O3 -mavx2 -mfma`, 10K MC paths. Canonical CSV: `results/results.csv`.*

## Results

**Gradient time** — the cost of computing all sensitivities. Bold = fastest in row. The **Primal** column is the cost of evaluating the same workload once with raw doubles (no AD machinery), so the gap between Primal and each AAD library shows the recording-and-reverse-sweep overhead, and FD ≈ (N+1) × Primal as expected for forward-difference bumping.

| # | Benchmark | Sensis | Primal | FD | XAD | XAD-Codegen | CppAD | Adept | autodiff |
|---|-----------|:------:|-------:|---:|----:|------------:|------:|------:|---------:|
| 1 | **Heston MC** | 8 | 9.3 ms | 83 ms | 40 ms | **21 ms** | 268 ms | 91 ms | DNF |
| 2 | **SABR Calib** | 15 | 2.1 ms | 33 ms | 8.4 ms | **4.9 ms** | 32 ms | 19 ms | 38 ms |
| 3 | **XVA CVA** | 40 | 591 ms | 24.1 s | 2.6 s | **0.57 s** | 8.0 s | 7.1 s | DNF |
| 4 | **LIBOR Swaption** | 161 | 138 ms | 21.6 s | 1.00 s | **0.31 s** | 4.57 s | 1.15 s | DNF |

Median of 10 measured iterations after warmup, reverse-mode for the AAD libraries. The Primal column is the median of the AAD libs' raw-double primal benchmarks, which all run the same underlying numerical kernel as the gradient timings (no RNG cost asymmetries). The sanity check holds: FD/Primal ≈ N+1 across all four rows.

## Observations

- **Adjoint AD vs finite differences.** FD scales O(N) with input count, so the gap to AAD widens from ~4× on Heston (8 inputs) to **~70× on LIBOR** (161 inputs). FD is fine for spot checks but quickly becomes the bottleneck once you need more than a handful of sensitivities.
- **Tape libraries cluster within an order of magnitude.** XAD's tape mode is the fastest tape library on every benchmark, by margins of ~1.1× (LIBOR vs Adept) up to 2.3× (Heston vs Adept). CppAD is consistently slowest of the three on the MC benchmarks but stays within an order of magnitude.
- **XAD-Codegen** compiles the recorded graph to AVX2 native code at runtime and is **roughly 2×–5× faster than XAD's own tape mode** on the four benchmarks. Switching from tape to Codegen requires expressing data-dependent branches via `xad::less(a,b).If(then,else)` so the recorded graph is branch-free; no code changes outside the per-path kernel.
- **autodiff** completes only 1 of 4 benchmarks (SABR, using forward `dual` mode). Forward mode is O(N) in input count and the alternative `var` reverse mode allocates a fresh heap-based expression tree on every gradient call — neither scales to MC pricing or larger calibrations. autodiff isn't a peer for the AAD workloads benchmarked here.

## Libraries

| Library | Modes available | Recording approach |
|---|---|---|
| [XAD](https://github.com/auto-differentiation/xad) | Forward & Adjoint, higher-order | Tape-based; optional [`xad-codegen`](https://www.xcelerit.com/xad-enterprise-support) backend compiles the recorded graph to AVX2 native code |
| [CppAD](https://github.com/coin-or/CppAD) | Forward & Reverse, higher-order | Tape-based `ADFun` record/replay |
| [Adept 2](https://github.com/rjhogan/Adept-2) | Forward & Reverse | Expression templates with stack recording |
| [autodiff](https://github.com/autodiff/autodiff) | Forward (`dual`) & Reverse (`var`) | Compile-time dual numbers / runtime expression tree |
| [ddx](https://github.com/reach2sayan/ddx) | Forward & Reverse, higher-order | Record-once graph; interpreted, or compiled by its LLVM backend (AVX2, 4 lanes) |

All five libraries support both forward and reverse modes; the suite exercises reverse mode, the standard choice for many-inputs/one-output workloads such as risk and pricing.

## Build & run

```bash
cmake -B build -GNinja -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/ad_benchmarks
```

To enable XAD-Codegen results (`xad-codegen` is [commercially licensed](https://www.xcelerit.com/xad-enterprise-support)):

```bash
cmake -B build -GNinja -DCMAKE_BUILD_TYPE=Release \
  -DXAD_DIR=/path/to/xad \
  -DXAD_CODEGEN_DIR=/path/to/xad-codegen \
  -DENABLE_XAD_JIT=ON
```

To enable the ddx rows (`ddx` and `ddx-JIT`), build ddx with the suite's flags, then configure against that build:

```bash
cd /path/to/ddx
cmake -S . -B build/adbench -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_NATIVE_ARCH=OFF -DDDX_BUILD_JIT=ON -DDDX_OPENCL=OFF \
  -DCMAKE_CXX_FLAGS="-mavx2 -mfma"
cmake --build build/adbench

cd /path/to/ad-benchmarks
cmake -B build -GNinja -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_DDX=ON -DDDX_DIR=/path/to/ddx/build/adbench
```

`./build/ad_benchmarks_ddx_validate` checks both ddx backends' gradients against central differences and their values against the double kernels, outside the timed binary.

Run `./build/ad_benchmarks --help` for CLI options (`--paths`, `--iters`, `--warmup`, `--csv`, `--only`, `--skip`).

To regenerate the chart from a CSV:

```bash
python scripts/plot_results.py results/results.csv results/chart.png
```

## Methodology

- **Identical compiler flags** across libraries: `-O3 -mavx2 -mfma` (GCC/Clang) or `/O2 /arch:AVX2 /fp:fast` (MSVC).
- **Idiomatic APIs.** Each library uses its own recommended pattern: XAD reverse-mode tape (`xad::adj<double>`), CppAD `ADFun` record/replay, Adept `Stack` recording, autodiff `dual` forward mode for SABR, ddx `rt::Builder` graph recorded once and replayed by its block interpreter (`ddx`) or compiled once with its LLVM backend at 4 lanes with only the replay timed (`ddx-JIT`). No micro-optimizations applied to one library that wouldn't be applied to another.
- **Median of measured iterations** after warmup; warmup excluded.
- **All five libraries' gradients** agree with finite differences within numerical tolerance during development.
- **Same machine, same run.** Re-running on a different machine scales all rows by roughly the same factor.

## Contributing

PRs and issues welcome — especially:

- **More AD libraries** (open an issue or PR with a wrapper following the pattern in `xad/`, `cppad/`, `adept/`, `autodiff/`).
- **More finance kernels** (Bermudan/American MC, multi-curve bootstrapping, equity vol surface fitters, real PFE / ECL).
- **Methodology improvements** or fairer wirings for any of the existing libraries.

## Acknowledgements

The LIBOR swaption benchmark is adapted from
[Prof. Mike Giles' canonical adjoint LMM C++ code](https://people.maths.ox.ac.uk/~gilesm/codes/libor_AD/testlinadj.cpp).
Thanks to the maintainers of [CppAD](https://github.com/coin-or/CppAD), [Adept 2](https://github.com/rjhogan/Adept-2), and [autodiff](https://github.com/autodiff/autodiff) — a meaningful comparison is only possible because their libraries exist.

## License

Copyright © 2026 Xcelerit Computing Ltd. Licensed under the [MIT License](LICENSE). See [CITATION.cff](CITATION.cff) for citation metadata.
