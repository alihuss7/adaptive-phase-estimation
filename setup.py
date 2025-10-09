from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="adaptive-phase-estimation",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Benchmark suite for static vs. dynamic-circuit IPEA using Qiskit Runtime",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/adaptive-phase-estimation",
    py_modules=["adaptive_phase_estimation"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    install_requires=[
        "qiskit>=2.0.0",
        "qiskit-aer>=0.17.0",
        "numpy>=2.0.0",
        "matplotlib>=3.10.0",
        "pandas>=2.3.0",
    ],
    extras_require={
        "hardware": ["qiskit-ibm-runtime>=0.41.0"],
        "dev": [
            "pytest>=8.4.0",
            "black>=25.0.0",
            "pylint>=3.3.0",
            "jupyter>=1.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ipea-benchmark=adaptive_phase_estimation:main",
        ],
    },
)
