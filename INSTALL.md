# Installation Guide for vasp-grace

`vasp-grace` is now organized as a proper Python package that can be installed via pip.

## Installation from the Repository

### 1. **Development Installation (Editable Mode)**

This is recommended for development or if you want to make changes to the code:

```bash
cd /path/to/vasp-grace
pip install -e .
```

The `-e` flag installs the package in editable mode, so changes to the source code are immediately reflected without reinstalling.

### 2. **Standard Installation**

To install the package normally:

```bash
cd /path/to/vasp-grace
pip install .
```

### 3. **Installation with Dependencies**

If dependencies (ASE, TensorFlow, etc.) are not yet installed:

```bash
pip install -e ".[dev]"  # Includes development dependencies
```

## Package Structure

```
vasp-grace/
├── vasp_grace/                 # Main package directory
│   ├── __init__.py            # Package initialization
│   ├── cli.py                 # Command-line interface
│   ├── parser.py              # INCAR file parsing
│   ├── calculator.py          # GRACE calculator setup
│   ├── calculations.py        # High-level calculation routines
│   ├── outputs.py             # VASP output file writing
│   └── utils.py               # Utility functions
├── pyproject.toml             # Modern build configuration
├── setup.py                   # Setup script (for compatibility)
├── README.md                  # Project description
└── tests/                     # Test directory
```

## Module Organization

The code is organized into logical modules:

- **`parser.py`**: INCAR parameter parsing with type inference
- **`calculator.py`**: GRACE model initialization and calculator utilities
- **`calculations.py`**: High-level calculation workflows (single-point, MD, optimization, phonons)
- **`outputs.py`**: VASP-compatible output file writing
- **`cli.py`**: Command-line interface and main entry point
- **`utils.py`**: Helper functions (mesh parsing, symbol grouping, etc.)

## Usage

### Command-line Usage

After installation, you can use `vasp-grace` from any directory:

```bash
vasp-grace --poscar POSCAR --incar INCAR
```

Or with defaults:

```bash
vasp-grace  # Uses CONTCAR and INCAR in current directory
```

### Programmatic Usage

You can also import and use the modules in your Python code:

```python
from vasp_grace import parse_incar, get_calculator
from vasp_grace.calculations import run_single_point
from ase.io import read

# Parse INCAR
incar = parse_incar("INCAR")

# Get GRACE calculator
calc = get_calculator(incar["GRACE_MODEL"])

# Read structure
atoms = read("POSCAR", format="vasp")

# Run calculation
atoms.calc = calc
energy = atoms.get_potential_energy()
```

## Adding New Calculation Types

To add new calculation types or properties:

1. **Add INCAR parameters** in `vasp_grace/parser.py`:
   ```python
   params = {
       "NEW_PROPERTY": 1.0,  # Will be parsed as float
       "NEW_BOOL": False,    # Will be parsed as bool
   }
   ```

2. **Create calculation function** in `vasp_grace/calculations.py`:
   ```python
   def run_new_calculation(atoms, incar):
       """Run new calculation type."""
       # Implementation
   ```

3. **Add trigger logic** in `vasp_grace/cli.py`:
   ```python
   if incar["NEW_TRIGGER"]:
       run_new_calculation(atoms, incar)
   ```

## Dependencies

- **numpy**: Numerical operations
- **ase**: Atomic Simulation Environment
- **tensorflow**: Machine learning backend
- **tensorpotential**: GRACE model interface
- **matplotlib**: Plotting (for phonon band structures, DOS)

## Troubleshooting

If you encounter installation issues:

1. **TensorFlow installation**: This can be slow. Use `--no-cache-dir` if needed:
   ```bash
   pip install --no-cache-dir -e .
   ```

2. **GPU support**: For GPU acceleration, ensure TensorFlow is built with CUDA support:
   ```bash
   pip install tensorflow[and-cuda]
   ```

3. **Specific Python version**: Ensure Python 3.8+ is used:
   ```bash
   python3.10 -m pip install -e .
   ```

## Uninstallation

To uninstall:

```bash
pip uninstall vasp-grace
```

## Further Development

The package is designed to be easily extended:

- Add new calculation types by creating functions in appropriate modules
- Add INCAR parameters by updating the `params` dict in `parser.py`
- Output formats can be extended in `outputs.py`
- CLI options can be added in `cli.py`
