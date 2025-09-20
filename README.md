# Geometric Eigenmode Model

<p align="center">
  <img src="data/cover.jpeg" alt="data/cover.jpeg" width="700">
</p>

Code and data for the manuscript "Geometric constraints on the architecture of mammalian cortical connectomes

---

## Repository Structure

## Getting Started

### 1. Download this repository

Simply download this Github repository

### 2. Download the data

Download the data from this OSF repository: https://osf.io/rz3hw/

Each folder downloaded from OSF needs to be placed in the "data/" folder structure of this GitHub repo.

### 2. Set up the conda environment

The repository includes an `environment.yml` file to recreate the python environment used for the analyses. The (hacky) last line
ensures that multiple jobs can be run at the same time within the environmen (SEEM TO BE ONLY NECESSARY ON LINUX BECAUS IT WORKS WITHOUT IT ON MAC).

```bash
conda env create -f environment.yml
conda activate geom_eigen_model
conda install -n geom_eigen_model numpy=1.26.* scipy=1.13.* numba=0.60.* mkl mkl-service -c defaults


