import numpy as np


def topsis(values, directions, nodes, pareto_allocations, weights=None):
    n_solutions = values.shape[0]
    n_criteria = values.shape[1]

    if weights is None:
        weights = np.ones(n_criteria) / n_criteria

    # 🔴 FIX: map node IDs → allocation indices
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    # Normalize decision matrix
    norm = np.sqrt((values ** 2).sum(axis=0))
    norm[norm == 0] = 1
    norm_values = values / norm

    weighted = norm_values * weights

    ideal = np.zeros(n_criteria)
    anti_ideal = np.zeros(n_criteria)

    for j in range(n_criteria):
        if directions[j] > 0:
            ideal[j] = weighted[:, j].max()
            anti_ideal[j] = weighted[:, j].min()
        else:
            ideal[j] = weighted[:, j].min()
            anti_ideal[j] = weighted[:, j].max()

    dist_ideal = np.sqrt(((weighted - ideal) ** 2).sum(axis=1))
    dist_anti = np.sqrt(((weighted - anti_ideal) ** 2).sum(axis=1))

    scores = dist_anti / (dist_ideal + dist_anti + 1e-12)

    # Optional: node greenness computation
    node_greenness = {}

    for node_id in nodes:
        idx = node_to_idx[node_id]
        node_greenness[node_id] = np.mean(
            [pareto_allocations[i][idx] for i in range(n_solutions)]
        )

    best_idx = int(np.argmax(scores))
    return best_idx, scores, node_greenness
