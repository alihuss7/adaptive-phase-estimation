# Adaptive Phase Estimation (Static vs Dynamic IPEA)

Benchmarking static (multi‑ancilla) vs dynamic (single‑ancilla with mid‑circuit measurement & reset) Iterative Phase Estimation using Qiskit 2.x. Focus: qubit savings, accuracy vs shots, and robustness to realistic noise.

## Quick Start

Environment & install:

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

Upgrade pip (optional but recommended):

```bash
python3 -m pip install --upgrade pip
```

Smoke test (imports only):

```bash
python3 -c "import adaptive_phase_estimation as m; print('OK')"
```

Version check (Qiskit + Python):

```bash
python3 -c "import sys, qiskit; print(sys.version); print(qiskit.__qiskit_version__)"
```

Run default benchmark:

```bash
python3 adaptive_phase_estimation.py
```

Creates: `ipea_benchmark_results.csv`, `ipea_benchmark_results.json`, `ipea_benchmark_results.png`.

## Minimal Python Usage

```python
from adaptive_phase_estimation import PhaseEstimationBenchmark
bench = PhaseEstimationBenchmark(target_phase=0.65625, precision_bits=5)
results = bench.run_full_benchmark(
    shots_list=[1024, 2048], noise_levels=[0.0, 1e-3], replicates=3
)
bench.plot_benchmark_results(results, save_path='results.png')
```

## CLI

```bash
python3 adaptive_phase_estimation.py \
  --angle 0.375 --bits 6 \
  --shots 1024 2048 4096 \
  --p1 0.0 0.001 0.005 \
  --replicates 5 --seed 42 \
  --csv out.csv --json out.json --fig out.png
```

Add `--show-circuits` for diagrams or `--quiet` for headless runs.

More examples:

```bash
# Minimal fast run (fewer shots)
python3 adaptive_phase_estimation.py --angle 0.5 --bits 3 --shots 512 1024 --replicates 2 --p1 0.0 0.001

# Irrational phase (harder case)
python3 adaptive_phase_estimation.py --angle 0.4487989505128276 --bits 6 --shots 2048 4096 --replicates 3

# Add second angle for comparison
python3 adaptive_phase_estimation.py --angle 0.5 --extra-angle 0.333 --bits 5 --shots 1024 2048 --replicates 3

# Headless + custom output filenames
python3 adaptive_phase_estimation.py --quiet --csv quick.csv --json quick.json --fig quick.png
```

## Core Features

- Static & dynamic IPEA circuit generation
- Resource comparison (qubits, depth, ops)
- Adaptive feedback via `if_test()` (Qiskit 2.x)
- Simple depolarizing + readout noise model
- Batch benchmarks with replicates (mean ± std)
- CSV/JSON export & Matplotlib plots
- Reproducible seeds

## Algorithm (Condensed)

Static: m ancillas kept coherent; phased corrections accumulate before final measurement.  
Dynamic: single ancilla measured each iteration → conditional phase corrections on subsequent rounds + reset.

## Output Artifacts

| File                          | Purpose                                           |
| ----------------------------- | ------------------------------------------------- |
| `ipea_benchmark_results.csv`  | Tabular per‑(theta, shots, noise, regime) metrics |
| `ipea_benchmark_results.json` | Full config + run records                         |
| `ipea_benchmark_results.png`  | Error vs shots / noise plots                      |

## Notebook

Open `example_usage.ipynb` for a curated walkthrough covering:

- Static vs dynamic IPEA sweeps with adjustable noise
- Plotting phase error vs shots and exporting artifacts
- Inspecting measurement histograms and convergence diagnostics
- Reproducing benchmark presets with seeded randomness

## Testing

Run the test suite:

```bash
pytest -v test_adaptive_phase_estimation.py
```

Need finer-grained checks? Useful variants include:

- ```bash
  pytest -k "Benchmark or Export" -v
  ```
  Run only the smoke-style benchmark or export tests.
- ```bash
  pytest test_adaptive_phase_estimation.py::TestSimulation::test_exact_phase_noiseless -v
  ```
  Focus on a single test class/case when iterating quickly.
- ```bash
  pytest -x -v
  ```
  Stop on first failure to speed up debugging.
- ```bash
  pytest -v --cov=adaptive_phase_estimation --cov-report=term-missing
  ```
  Capture coverage details when preparing releases.

## Noise Model (Default)

- 1Q depolarizing: p1 (e.g. 0.001)
- 2Q depolarizing: p2 = 5·p1
- Readout flip: 0.02

## Dependencies

Pinned versions in `requirements.txt` (Qiskit 2.x, Aer, NumPy, Matplotlib, Pandas). Optional: `qiskit-ibm-runtime` for hardware.

## License

MIT (see `LICENSE`).

## References

1. A. Kitaev, Quantum measurements and the Abelian Stabilizer Problem (1995).
2. IBM Quantum Docs – Dynamic Circuits.
3. Qiskit Textbook – Phase Estimation.
