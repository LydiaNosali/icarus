import numpy as np

def topsis(values, directions, weights=None):
    """
    values:    numpy array of shape (n_solutions, n_objectives)
    directions: list of 1 or -1 for each objective
                1  = maximize
                -1 = minimize
    weights:   optional weight vector, default equal
    """

    values = np.array(values, dtype=float)

    if values.shape[0] == 1:
        return 0, np.array([1.0])
    
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
    print(f"d_worst:{d_worst}, d_best:{d_best}")
    score = d_worst / (d_best + d_worst)

    # Best solution index
    best_idx = np.argmax(score)

    return best_idx, score
