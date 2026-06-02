def reciprocal_rank_fusion(ranked_lists: list, weights: list = None, k: int = 60) -> list:
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    
    scores = {}
    for ranked_list, weight in zip(ranked_lists, weights):
        for rank, item in enumerate(ranked_list):
            if item not in scores:
                scores[item] = 0
            scores[item] += weight * (1 / (k + rank + 1))
    
    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
