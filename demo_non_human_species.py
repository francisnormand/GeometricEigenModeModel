import numpy as np
import lapy
from scipy.spatial.distance import pdist
import utilities
import sys
import os
import json
import subprocess
# from connectome_models import generate_high_res_GEM_humans, generate_high_res_LBO_humans, generate_EDR_vertex_model, generate_random_vertex_model
from network_measures_and_statistics import compute_node_properties
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

def generate_geometric_modes(species):
    """
    Compute and save eigenvalues, eigenmodes of the 3 non-human species coritcal surfaces
    or volume (for the mouse isocortex).

    If the eigenmodes file already exists, the process is skipped.

    Parameters
    ----------
    path_data : str or Path
        Path to the data directory.
    lump : bool, default=False
        Whether to lump vertices during computation.
    cortex_mask : bool, default=True
        Whether the medial wall is masked.
    max_num_modes : int, default=500
        Maximum number of modes to compute.

    Outputs
    -------
    Saves the following files in `path_data`:
        - {species}_evals_lump_{lump}.npy
        - {species}_emodes_lump_{lump}.npy
    """
    lump = False

    cortex_mask = True
    # cortex_mask = False
  
    max_num_modes = 500

    surface, surface_path = utilities.get_human_template_surface()

    output_eval_filename = path_data + f"f{species}/{species}_evals_lump_{lump}.npy"
    output_emode_filename = path_data + f"/{species}/{species}_emodes_lump_{lump}.npy"
    output_B_matrix_filename = None # Not saving the B matrix

    if os.path.exists(output_emode_filename):
        print(f"{output_emode_filename}")
        print()
        print("geometric modes already exists and will be overwritten")
        print()

    if species == "mouse":
        path_mesh = path_data + f"/{specie}/rh_tet_mesh.vtk"
        mesh = lapy.TetMesh.read_vtk(path_mesh)
        solver = lapy.Solver(mesh)
        eigenvalues, eigenmodes = solver.eigs(k=500)

        np.save(output_eval_filename, eigenvalues)
        np.save(output_emode_filename, eigenmodes)
    
    else:
        cortex_mask_array = utilities.get_human_parcellated_cortex_mask(path_data, number_of_parcels)
        save_cut = 0  #Not saving temporary cut surface       
        evals, emodes, B_matrix = utilities.calc_surface_eigenmodes(surface_path, cortex_mask_array, output_eval_filename, output_emode_filename, output_B_matrix_filename, save_cut=save_cut, num_modes=max_num_modes, lump=lump)
    
    print()
    print("geometric modes were saved")

# Current working director
cwd = os.getcwd()

# Global variable here. So it only is defined once. 
path_data = f"{cwd}/data/non_human_species"

# Path to the Conda environment
conda_prefix = os.environ.get("CONDA_PREFIX")

env_path = conda_prefix.split("/conda/")[0]
# print(env_path, "base_conda_path")

# Current environment name
conda_env_name = os.environ.get("CONDA_DEFAULT_ENV")
# print(conda_env_name)


def mainFunction():
    
    """
    Main function to reproduce the analysis on the non-human species connectomes.

    These functions have to be run sequentially.
    """

    species = "mouse"
    species = "marmoset"
    species = "macaque"

    # 1. Generate the geometric eigenmodes
    generate_geometric_modes(species)

    # 2. Optimize the GEM (explore parameters landscape)
    # optimize_and_save_non_human_species_results(species)
    # print()

    #3. Visualize performance
    # visualize_GEM_non_human_species_results(species)


    # results = "main"
    # results = "modularity"
    # results = "spectral"
    
    #4. Generate benchmark models
    # generate_non_human_species_comparison_results(species, which_results=results)

    #5. Compare GEM performance with other models
    # compare_non_human_species_models(species, which_results=results)


if __name__ == "__main__":
    mainFunction()