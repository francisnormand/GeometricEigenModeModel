import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

# Parameters
total_possible_connections = 9616305
n_connections_vertex = 442350
p_ij = np.random.rand(total_possible_connections)
p_ij /= p_ij.sum()

def method_choice():
    return np.random.choice(total_possible_connections, size=n_connections_vertex, replace=False, p=p_ij)

def method_gumbel():
    gumbels = -np.log(-np.log(np.random.rand(total_possible_connections)))
    logits = np.log(p_ij)
    return np.argsort(logits + gumbels)[-n_connections_vertex:]

def method_div_rand():
    return np.argsort(np.random.rand(total_possible_connections) / p_ij)[:n_connections_vertex]

# Run a single trial (sparse-aware)
inds_choice = method_choice()
inds_gumbel = method_gumbel()
inds_divrand = method_div_rand()

# Probabilities restricted to sampled indices
p_choice = p_ij[inds_choice]
p_gumbel = p_ij[inds_gumbel]
p_divrand = p_ij[inds_divrand]

# Normalize to sum to 1 for proper comparison
p_choice /= p_choice.sum()
p_gumbel /= p_gumbel.sum()
p_divrand /= p_divrand.sum()

# Correlation and KL divergence between each method and np.random.choice
corr_gumbel = np.corrcoef(p_choice, p_gumbel)[0,1]
corr_divrand = np.corrcoef(p_choice, p_divrand)[0,1]

kl_gumbel = entropy(p_choice, p_gumbel)
kl_divrand = entropy(p_choice, p_divrand)

print(f"Sparse-aware comparison (only sampled edges):")
print(f"Correlation with np.random.choice: Gumbel={corr_gumbel:.6f}, rand/p_ij={corr_divrand:.6f}")
print(f"KL divergence:                     Gumbel={kl_gumbel:.6e}, rand/p_ij={kl_divrand:.6e}")






























# Parameters (your real case)
# total_possible_connections = 9616305
# n_connections_vertex = 442350
# p_ij = np.random.rand(total_possible_connections)
# p_ij /= p_ij.sum()

# def method_choice():
#     return np.random.choice(
#         total_possible_connections,
#         size=n_connections_vertex,
#         replace=False,
#         p=p_ij
#     )

# def method_gumbel():
#     gumbels = -np.log(-np.log(np.random.rand(total_possible_connections)))
#     logits = np.log(p_ij)
#     return np.argsort(logits + gumbels)[-n_connections_vertex:]

# def method_div_rand():
#     return np.argsort(np.random.rand(total_possible_connections) / p_ij)[:n_connections_vertex]

# # Run fewer trials because this is expensive
# n_trials = 20
# counts_choice = np.zeros(total_possible_connections, dtype=np.int32)
# counts_gumbel = np.zeros_like(counts_choice)
# counts_divrand = np.zeros_like(counts_choice)

# for _ in range(n_trials):
#     counts_choice[method_choice()] += 1
#     counts_gumbel[method_gumbel()] += 1
#     counts_divrand[method_div_rand()] += 1

# # Convert to selection probabilities
# prob_choice = counts_choice / (n_trials * n_connections_vertex)
# prob_gumbel = counts_gumbel / (n_trials * n_connections_vertex)
# prob_divrand = counts_divrand / (n_trials * n_connections_vertex)

# # Quantitative comparisons
# corr_choice = np.corrcoef(prob_choice, p_ij)[0, 1]
# corr_gumbel = np.corrcoef(prob_gumbel, p_ij)[0, 1]
# corr_divrand = np.corrcoef(prob_divrand, p_ij)[0, 1]

# kl_choice = entropy(prob_choice + 1e-12, p_ij)  # KL divergence
# kl_gumbel = entropy(prob_gumbel + 1e-12, p_ij)
# kl_divrand = entropy(prob_divrand + 1e-12, p_ij)

# print(f"Correlation with p_ij:")
# print(f"  np.random.choice: {corr_choice:.6f}")
# print(f"  Gumbel-top:       {corr_gumbel:.6f}")
# print(f"  rand/p_ij:        {corr_divrand:.6f}")

# print(f"\nKL-divergence vs p_ij:")
# print(f"  np.random.choice: {kl_choice:.6e}")
# print(f"  Gumbel-top:       {kl_gumbel:.6e}")
# print(f"  rand/p_ij:        {kl_divrand:.6e}")