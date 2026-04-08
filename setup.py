"""Setup for vasp-grace package."""
from setuptools import setup, find_packages

setup(
    name="vasp-grace",
    version="0.1.0",
    description=(
        "Complete workflow for training and deploying GRACE machine-learning "
        "interatomic potentials from VASP ab initio data."
    ),
    author="vasp-grace contributors",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "ase>=3.22",
        "numpy>=1.23",
        "scipy>=1.9",
        "matplotlib>=3.6",
        "pandas>=1.5",
        "pymatgen>=2023.1.1",
        "pyyaml>=6.0",
        "tqdm>=4.64",
    ],
    extras_require={
        "grace": [
            "tensorpotential",   # gracemaker + GRACECalculator
            "python-ace",        # PyGRACEFSCalculator for GRACE/FS YAML models
        ],
        "dev": [
            "pytest>=7.0",
            "pytest-cov",
        ],
    },
    entry_points={
        "console_scripts": [
            "vasp-to-extxyz=scripts.vasp_to_extxyz:main",
            "collect-vasp-dataset=scripts.collect_vasp_dataset:main",
            "generate-grace-input=scripts.generate_gracemaker_input:main",
            "grace-ase=scripts.grace_ase_calculator:main",
            "grace-lammps=scripts.grace_lammps_input:main",
            "validate-grace=scripts.validate_grace_model:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
