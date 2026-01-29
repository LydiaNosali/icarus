import numpy as np

def topsis(values, directions, weights=None, nodes=None, pareto_allocations=None):
    values = np.array(values, dtype=float)
    if values.shape[0] == 1:
        return 0, np.array([1.0]), {}

    n, m = values.shape
    
    # STEP 0: CARBON FIRST - Find minimum carbon
    carbon_col = 0  # assuming carbon is first column
    min_carbon_idx = np.argmin(values[:, carbon_col])
    min_carbon = values[min_carbon_idx, carbon_col]
    
    print(f"Min carbon: {min_carbon:.6f} at idx {min_carbon_idx}")
    
    # STEP 1: Normalize by VECTOR LENGTH (standard TOPSIS)
    norm = values / np.sqrt((values ** 2).sum(axis=0))
    
    # STEP 2: Weights - FORCE carbon dominance
    if weights is None:
        weights = np.array([0.9, 0.05, 0.05])
    wnorm = norm * weights
    
    # STEP 3: Ideal points
    ideal_best = np.array([np.min(wnorm[:,j]) if directions[j] == -1 else np.max(wnorm[:,j]) 
                          for j in range(m)])
    ideal_worst = np.array([np.max(wnorm[:,j]) if directions[j] == -1 else np.min(wnorm[:,j]) 
                           for j in range(m)])
    
    # STEP 4: Distances
    d_best = np.sqrt(((wnorm - ideal_best) ** 2).sum(axis=1))
    d_worst = np.sqrt(((wnorm - ideal_worst) ** 2).sum(axis=1))
    
    # STEP 5: Score
    score = d_worst / (d_best + d_worst + 1e-10)  # avoid div0
    
    print(f"All carbons: {values[:,0]}")
    print(f"TOPSIS scores: {score}")
    print(f"TOPSIS picks idx {np.argmax(score)} with carbon {values[np.argmax(score),0]:.6f}")
    
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
