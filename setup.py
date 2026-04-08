---

### 2. `setup.py`

```python
from setuptools import setup, find_packages

setup(
    name="vasp-grace",
    version="0.1.0",
    description="A wrapper to use GRACE Machine Learning Potentials as a drop-in replacement for VASP.",
    author="Your Name",
    author_email="asif.iqbal.bhatti@vub.be",
    packages=find_packages(),
    install_requires=[
        "ase>=3.27.0",
        "numpy>=1.20.0",
        "tensorpotential"
    ],
    entry_points={
        "console_scripts":[
            "vasp_grace=vasp_grace.main:main",
        ],
    },
)
