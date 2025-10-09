"""Core tests for Adaptive Phase Estimation benchmark suite.

Run with: pytest test_adaptive_phase_estimation.py -v
"""

import pytest
import numpy as np
from pathlib import Path
import json

from adaptive_phase_estimation import (
    PhaseEstimationBenchmark,
    simple_noise_model,
    bitstring_to_phase_msb_first,
    export_results,
)

from qiskit_aer import AerSimulator
from qiskit import transpile


class TestCircuitConstruction:
    """Circuit construction."""

    def test_static_circuit_builds(self):
        bench = PhaseEstimationBenchmark(target_phase=0.5, precision_bits=4)
        qc = bench.build_static_ipea()

        assert qc is not None
        assert qc.num_qubits == 5
        assert qc.num_clbits == 4

    def test_dynamic_circuit_builds(self):
        bench = PhaseEstimationBenchmark(target_phase=0.5, precision_bits=4)
        qc = bench.build_dynamic_ipea()

        assert qc is not None
        assert qc.num_qubits == 2
        assert qc.num_clbits == 4

    def test_different_precision_levels(self):
        for m in [3, 5, 7, 8]:
            bench = PhaseEstimationBenchmark(target_phase=0.25, precision_bits=m)
            static_qc = bench.build_static_ipea()
            dynamic_qc = bench.build_dynamic_ipea()

            assert static_qc.num_qubits == m + 1
            assert dynamic_qc.num_qubits == 2


class TestTranspilation:
    """Transpilation to Aer."""

    def test_static_transpiles_to_aer(self):
        bench = PhaseEstimationBenchmark(target_phase=0.375, precision_bits=5)
        qc = bench.build_static_ipea()

        sim = AerSimulator()
        transpiled = transpile(qc, backend=sim, optimization_level=1)

        assert transpiled is not None
        assert transpiled.depth() > 0

    def test_dynamic_transpiles_to_aer(self):
        bench = PhaseEstimationBenchmark(target_phase=0.375, precision_bits=5)
        qc = bench.build_dynamic_ipea()

        sim = AerSimulator()
        transpiled = transpile(qc, backend=sim, optimization_level=1)

        assert transpiled is not None
        assert transpiled.depth() > 0

    def test_circuit_metrics(self):
        bench = PhaseEstimationBenchmark(target_phase=0.5, precision_bits=4)

        static_qc = bench.build_static_ipea()
        dynamic_qc = bench.build_dynamic_ipea()

        static_metrics = bench.analyze_circuit(static_qc, "static")
        dynamic_metrics = bench.analyze_circuit(dynamic_qc, "dynamic")

        assert static_metrics["qubits"] > dynamic_metrics["qubits"]

        assert static_metrics["depth"] > 0
        assert dynamic_metrics["depth"] > 0
        assert static_metrics["total_ops"] > 0
        assert dynamic_metrics["total_ops"] > 0


class TestSimulation:
    """Simulation correctness."""

    def test_exact_phase_noiseless(self):
        bench = PhaseEstimationBenchmark(target_phase=0.5, precision_bits=3)

        static_qc = bench.build_static_ipea()
        dynamic_qc = bench.build_dynamic_ipea()

        phase_s, counts_s = bench.simulate(static_qc, shots=1024, noise=None, seed=42)
        phase_d, counts_d = bench.simulate(dynamic_qc, shots=1024, noise=None, seed=42)

        assert abs(phase_s - 0.5) < 0.01
        assert abs(phase_d - 0.5) < 0.01

    def test_simulation_with_noise(self):
        bench = PhaseEstimationBenchmark(target_phase=0.25, precision_bits=4)
        qc = bench.build_static_ipea()

        noise = simple_noise_model(p1=0.001, p2=0.005, readout_p=0.02)
        phase, counts = bench.simulate(qc, shots=2048, noise=noise, seed=123)

        assert 0.0 <= phase <= 1.0
        assert abs(phase - 0.25) < 0.2


class TestUtilityFunctions:
    """Utility helpers."""

    def test_bitstring_to_phase_conversion(self):
        assert bitstring_to_phase_msb_first("1") == 0.5
        assert bitstring_to_phase_msb_first("10") == 0.5
        assert bitstring_to_phase_msb_first("11") == 0.75
        assert bitstring_to_phase_msb_first("101") == 0.625
        assert bitstring_to_phase_msb_first("10101") == 0.65625

    def test_noise_model_creation(self):
        nm = simple_noise_model(p1=0.001, p2=0.005, readout_p=0.02)
        assert nm is not None


class TestBenchmarkExecution:
    """Benchmark execution."""

    def test_minimal_benchmark(self):
        bench = PhaseEstimationBenchmark(target_phase=0.5, precision_bits=3)

        results = bench.run_full_benchmark(
            shots_list=[512],
            noise_levels=[0.0],
            replicates=2,
            verbose=False,
            show_circuits=False,
            base_seed=42,
        )

        assert "runs" in results
        assert "config" in results
        assert len(results["runs"]) == 1
        assert results["config"]["replicates"] == 2

    def test_multi_angle_benchmark(self):
        bench = PhaseEstimationBenchmark(target_phase=0.25, precision_bits=4)

        results = bench.run_full_benchmark(
            shots_list=[512],
            noise_levels=[0.0],
            replicates=1,
            extra_angle=0.375,
            verbose=False,
            base_seed=42,
        )

        assert len(results["runs"]) == 2
        assert results["runs"][0]["theta"] == 0.25
        assert results["runs"][1]["theta"] == 0.375


class TestDataExport:
    """Data export."""

    def test_export_creates_files(self, tmp_path):
        bench = PhaseEstimationBenchmark(target_phase=0.5, precision_bits=3)

        results = bench.run_full_benchmark(
            shots_list=[512], noise_levels=[0.0], replicates=1, verbose=False
        )

        csv_path = tmp_path / "test.csv"
        json_path = tmp_path / "test.json"

        export_results(results, str(csv_path), str(json_path))

        assert csv_path.exists()
        assert json_path.exists()

    def test_csv_has_expected_columns(self, tmp_path):
        import pandas as pd

        bench = PhaseEstimationBenchmark(target_phase=0.5, precision_bits=3)
        results = bench.run_full_benchmark(
            shots_list=[512], noise_levels=[0.0, 0.001], replicates=1, verbose=False
        )

        csv_path = tmp_path / "test.csv"
        json_path = tmp_path / "test.json"
        export_results(results, str(csv_path), str(json_path))

        df = pd.read_csv(csv_path)

        assert "theta" in df.columns
        assert "regime" in df.columns
        assert "shots" in df.columns

    def test_json_has_config(self, tmp_path):
        bench = PhaseEstimationBenchmark(target_phase=0.5, precision_bits=3)
        results = bench.run_full_benchmark(
            shots_list=[512], noise_levels=[0.0], replicates=2, verbose=False
        )

        csv_path = tmp_path / "test.csv"
        json_path = tmp_path / "test.json"
        export_results(results, str(csv_path), str(json_path))

        with open(json_path, "r") as f:
            data = json.load(f)

        assert "config" in data
        assert data["config"]["replicates"] == 2
        assert data["config"]["precision_bits"] == 3


class TestEndToEnd:
    """End-to-end workflow."""

    def test_full_workflow(self, tmp_path):
        bench = PhaseEstimationBenchmark(target_phase=0.375, precision_bits=4)

        results = bench.run_full_benchmark(
            shots_list=[512, 1024],
            noise_levels=[0.0, 0.001],
            replicates=2,
            verbose=False,
            base_seed=42,
        )

        csv_path = tmp_path / "results.csv"
        json_path = tmp_path / "results.json"
        export_results(results, str(csv_path), str(json_path))

        fig_path = tmp_path / "results.png"
        bench.plot_benchmark_results(results, save_path=str(fig_path))

        assert csv_path.exists()
        assert json_path.exists()
        assert fig_path.exists()

    def test_reproducibility(self):
        bench1 = PhaseEstimationBenchmark(target_phase=0.25, precision_bits=4)
        bench2 = PhaseEstimationBenchmark(target_phase=0.25, precision_bits=4)

        qc1 = bench1.build_static_ipea()
        qc2 = bench2.build_static_ipea()

        phase1, _ = bench1.simulate(qc1, shots=1024, noise=None, seed=42)
        phase2, _ = bench2.simulate(qc2, shots=1024, noise=None, seed=42)

        assert phase1 == phase2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
