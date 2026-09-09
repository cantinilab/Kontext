<p align="center">
  <img src="assets/logo.png" alt="kontext logo" width="300"/>
</p>

<h1 align="center">kontext</h1>

<p align="center">
  Kernel-based method bridging spatial domain identification and cell-cell communication.
</p>


# kontext

**kontext** is a Python package for spatial transcriptomics data analysis.   
It provides tools for spatial domains identification, ligand-receptor scoring, niche analysis and local alignement scoring.

---

## Installation

### Requirements

- Python **>= 3.12**
- A working C/C++ compiler is recommended for `numba` / `scanpy` dependencies.

### From PyPI 

```bash
pip install kontext
```

### From github 

```bash
pip install git+https://github.com/cantinilab/kontext.git
```
### Local installation

```bash
# from the kontext directory
pip install -e .