import numpy as np

def topsis(values, directions, weights=None, nodes=None, pareto_allocations=None):
    """
    values:    numpy array of shape (n_solutions, n_objectives)
    directions: list of 1 or -1 for each objective
    weights:   optional weight vector, default equal
    pareto_allocations: list of allocation vectors [optional, for Jalil's node greenness]
    """
    values = np.array(values, dtype=float)

    if values.shape[0] == 1:
        return 0, np.array([1.0]), {}

    n, m = values.shape

    # Step 1: Normalize
    norm = values / np.sqrt((values ** 2).sum(axis=0))

    # Step 2: Weights
    if weights is None:
        weights = np.ones(m) / m
    wnorm = norm * weights

    # Step 3: Ideal best & worst
    ideal_best  = np.zeros(m)
    ideal_worst = np.zeros(m)

    for j in range(m):
        if directions[j] == 1:   # maximize
            ideal_best[j]  = wnorm[:, j].max()
            ideal_worst[j] = wnorm[:, j].min()
        else:                    # minimize
            ideal_best[j]  = wnorm[:, j].min()
            ideal_worst[j] = wnorm[:, j].max()

    # Step 4: Distances
    d_best  = np.sqrt(((wnorm - ideal_best)  ** 2).sum(axis=1))
    d_worst = np.sqrt(((wnorm - ideal_worst) ** 2).sum(axis=1))

    # Step 5: TOPSIS score
    # print(f"d_worst:{d_worst}, d_best:{d_best}")
    score = d_worst / (d_best + d_worst)

    # Step 6: JALIL'S NODE GREENNESS [NEW]
    node_greenness = {}
    # if pareto_allocations is not None and len(pareto_allocations) == n:
    #     n_nodes = len(pareto_allocations[0])  # Assume all same length
        
    #     for node_idx in range(n_nodes):
    #         # TOPSIS scores where this node has >0 cache
    #         relevant_scores = [
    #             score[i] for i in range(n) 
    #             if pareto_allocations[i][node_idx] > 0
    #         ]
    #         node_greenness[node_idx] = np.mean(relevant_scores) if relevant_scores else 0.0
    # else:
    #     node_greenness = {}
    
    if pareto_allocations is not None and nodes is not None and len(pareto_allocations) == n:
        for node_idx, node in enumerate(nodes):
            # TOPSIS scores where this node has >0 cache
            weighted_sum = 0.0
            total_alloc  = 0.0

            for i in range(n):
                alloc_amount = pareto_allocations[i][node_idx]
                if alloc_amount > 0:
                    weighted_sum += score[i] * alloc_amount
                    total_alloc  += alloc_amount

            node_greenness[node] = 1 - (
                weighted_sum / total_alloc if total_alloc > 0 else 0.0
            )
    else:
        node_greenness = {}
    
    # Best solution index
    best_idx = np.argmax(score)

    return best_idx, score, node_greenness
