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



def mainFunction():
    
    """
    Main function to reproduce the analysis on the non-human species connectomes.

    These functions have to be run sequentially.
    """

    species = "mouse"
    species = "marmoset"
    species = "macaque"

    # 1. Generate the geometric eigenmodes
    # generate_geometric_modes(species)

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