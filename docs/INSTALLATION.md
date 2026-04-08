# Installation Guide

## Python environment

```bash
conda create -n vasp-grace python=3.10
conda activate vasp-grace
pip install -r requirements.txt
```

## gracemaker / TensorPotential

```bash
pip install tensorpotential   # GRACECalculator + gracemaker CLI
pip install python-ace        # PyGRACEFSCalculator for GRACE/FS YAML models
```

> TensorFlow ≥ 2.12 is installed as a dependency.  
> GPU support requires CUDA ≥ 11.8 and cuDNN ≥ 8.6 (matching your TF version).

### Verify installation

```bash
python -c "from tensorpotential.calculator.grace import GRACECalculator; print('OK')"
python -c "import pyace; print('python-ace OK')"
gracemaker --help
grace_collect --help
grace_models list
```

## LAMMPS with GRACE support

Build LAMMPS with the `ML-PACE` package enabled.
Follow the [official GRACE LAMMPS guide](https://gracemaker.readthedocs.io/en/latest/gracemaker/install/#lammps-with-grace).

Quick build outline:
```bash
git clone --depth=1 https://github.com/lammps/lammps.git
cd lammps
mkdir build && cd build
cmake ../cmake \
    -DPKG_ML-PACE=ON \
    -DPKG_MANYBODY=ON \
    -DPKG_RIGID=ON \
    -DBUILD_MPI=ON \
    -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

Verify:
```bash
lmp -h | grep -i "ML-PACE\|grace"
```

## Installing vasp-grace scripts

```bash
git clone https://github.com/YOUR_USERNAME/vasp-grace.git
cd vasp-grace
pip install -e .
```

This exposes CLI entry points:
```
vasp-to-extxyz
collect-vasp-dataset
generate-grace-input
grace-ase
grace-lammps
validate-grace
```
