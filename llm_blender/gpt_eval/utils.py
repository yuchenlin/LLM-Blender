import numpy as np
from pathlib import Path
from itertools import combinations

def get_ranks_from_cmps(cmp_results, policy="max_logits"):
    """
    Args:
        cmp_results: ndarray of shape (n, c, c) where n is the number of samples, c is the number of candidates
            for each element, >0 means the first candidate is better than the second one, <0 means the second one is better
    Returns:
        ranks: ndarray of shape (n, c) where n is the number of samples, c is the number of candidates
    """
    if isinstance(cmp_results, list):
        cmp_results = np.array(cmp_results)
    bz, c, _ = cmp_results.shape
    ranks = np.zeros((bz, c), dtype=np.int32)
    for i in range(bz):
        if policy == "max_logits":
            scores = (cmp_results[i] - cmp_results[i].T).sum(axis=-1)
        elif policy == "max_wins":
            scores = (cmp_results[i] > 0).sum(axis=-1) + (cmp_results[i] < 0).sum(axis=-2)
        _ranks = get_ranks_from_scores(scores)
        ranks[i] = _ranks
    return ranks

def get_scores_from_cmps(cmp_results, policy="max_logits"):
    """
    Args:
        cmp_results: ndarray of shape (n, c, c) where n is the number of samples, c is the number of candidates
            for each element, >0 means the first candidate is better than the second one, <0 means the second one is better
    Returns:
        scores: ndarray of shape (n, c) where n is the number of samples, c is the number of candidates
    """
    if isinstance(cmp_results, list):
        cmp_results = np.array(cmp_results)
    bz, c, _ = cmp_results.shape
    scores = np.zeros((bz, c), dtype=np.float32)
    for i in range(bz):
        if policy == "max_logits":
            scores[i] = (cmp_results[i] - cmp_results[i].T).mean(axis=-1)
        elif policy == "max_wins":
            scores[i] = (cmp_results[i] > 0).sum(axis=-1) + (cmp_results[i] < 0).mean(axis=-2)
    return scores

def get_ranks_from_scores(scores):
    """
    Args:
        scores: ndarray of shape (n, c) or (c) where n is the number of samples, c is the number of candidates
        Treat same as higher one
        
    Returns:
        ranks: ndarray of shape (n, c) or (c) where n is the number of samples, c is the number of candidates
    """
    if isinstance(scores, list):
        scores = np.array(scores)
    orig_shape = scores.shape
    if len(scores.shape) == 1:
        scores = scores.reshape(1, -1)
    bz, c = scores.shape
    ranks = np.zeros((bz, c), dtype=np.int32)
    for i in range(bz):
        sorted_scores_i = list(sorted(list(scores[i]), reverse=True))
        for j in range(c):
            ranks[i, j] = sorted_scores_i.index(scores[i, j]) + 1
    
    ranks = ranks.reshape(orig_shape)
    return ranks

def get_ranks_from_chatgpt_cmps(ds_data):
    import numpy as np
    # transform chatgpt cmp_results to [bz, c, c]
    bz = len(ds_data)
    c = len(ds_data[0]['candidates'])

    chatgpt_cmp_results = np.zeros((bz, c, c))
    _models = [c['model'] for c in ds_data[0]['candidates']]
    for i, d in enumerate(ds_data):
        models = [c['model'] for c in d['candidates']]
        assert models == _models, f"models not match: {models} vs {_models}"
        for key, value in d['cmp_results'].items():
            idx1, idx2 = models.index(key.split(",")[0]), models.index(key.split(",")[1])
            if value == "A is better":
                chatgpt_cmp_results[i][idx1][idx2] += 1
                chatgpt_cmp_results[i][idx2][idx1] -= 1
            elif value == "B is better":
                chatgpt_cmp_results[i][idx1][idx2] -= 1
                chatgpt_cmp_results[i][idx2][idx1] += 1
            elif value == "Same good":
                chatgpt_cmp_results[i][idx1][idx2] += 0.5
                chatgpt_cmp_results[i][idx2][idx1] += 0.5
            elif value == "Same bad":
                chatgpt_cmp_results[i][idx1][idx2] -= 0.5
                chatgpt_cmp_results[i][idx2][idx1] -= 0.5
            else:
                raise ValueError("Unknown value: {}".format(value))

    chatgpt_cmp_ranks = get_ranks_from_cmps(chatgpt_cmp_results)

    model_ranks_map = {}
    for i, model_name in enumerate(_models):
        model_ranks_map[model_name] = chatgpt_cmp_ranks[:, i]
    return model_ranks_map, chatgpt_cmp_results

def draw_top_competitors(ranks, labels, save_path=None, top_k=3, verbose=False):
    """
    Args:
        ranks: ndarray of shape (n, c) where n is the number of samples, c is the number of candidates
            each element is the rank of the corresponding candidate
        labels: list of length c
            the labels of the candidates, can be the ranker model name
    Returns:
        fig, axes
        
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(top_k, 1, figsize=(10, 4+top_k*6))

    rank_idxs = np.argsort(ranks, axis=1)
    for rank in range(top_k):
        sizes = np.zeros(len(labels), dtype=np.int32)
        for i, idxs in enumerate(rank_idxs):
            sizes[idxs[rank]] += 1

        if verbose:
            print("rank-{} Competitiors".format(rank+1))
            for i in np.argsort(sizes)[::-1]:
                print("  {}: {} ({:.4f}%)".format(labels[i], sizes[i], sizes[i]/len(ranks) * 100))
            print()
        axes[rank].pie(sizes, labels=labels, autopct='%1.1f%%', shadow=False, startangle=90, labeldistance=1.0)
        axes[rank].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        axes[rank].set_title("rank-{} Competitiors".format(rank+1))
    if save_path:
        plt.suptitle(Path(save_path).stem)
        plt.savefig(save_path)
    else:
        return fig, axes

def deduplicate_string(string, repeat=4):

    result = ""
    sub_strings = string.split(" ")
    for i in range(len(sub_strings)):
        if " ".join(sub_strings[i:i+repeat]) in result:
            result += "..."
            break
        else:
            result += " " + sub_strings[i]
    return result.strip()

def is_evaluated(item):
    candidates = item['candidates']
    idxs = list(range(len(candidates)))
    if "cmp_results" not in item:
        return False
    cmp_results = item['cmp_results']
    all_pair_sets = set()
    for idx_A, idx_B in list(combinations(idxs, 2)):
        candidate_A = candidates[idx_A]
        candidate_B = candidates[idx_B]
        model_A = candidate_A['model']
        model_B = candidate_B['model']
        if model_A < model_B:
            all_pair_sets.add((model_A, model_B))
        else:
            all_pair_sets.add((model_B, model_A))
    
    eval_pair_sets = set()
    for key in cmp_results:
        model_A, model_B = key.split(",")
        if model_A < model_B:
            pair = (model_A, model_B)
        else:
            pair = (model_B, model_A)
        eval_pair_sets.add(pair)

    if eval_pair_sets < all_pair_sets:
        return False
    return True