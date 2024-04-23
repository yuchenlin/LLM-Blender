import torch
import torch.nn as nn
import numpy as np

PADDED_Y_VALUE = -1
PADDED_INDEX_VALUE = -1
DEFAULT_EPS = 1e-10


def permutation_prob(scores, level=1):
    """
    Args:
        scores: [batch_size, n_candidates]
        level: level of the permutation probs to compute
            when level is positive, we compute the top-pos permutation probs
            when level is negative, we compute the all permutation probs (same as top-n_candidates)
            when level is 0, we compute the top-1 permutation probs (same as top-1)
    Returns:
        prob: [batch_size, A(3,level)]
            represent the probability of each permutation.
            e.g. for input three scores [0.1, 0.2, 0.3], the original permutation is 0,1,2
            For the full level computation, the 2nd dim of probs is A(3,3)=6
            each representing probs of permutation
            0,1,2, 0,2,1, 1,0,2, 1,2,0, 2,0,1, 2,1,0
    """
    probs = []
    batch_size, n_candidates = scores.size()
    cur_probs = scores / scores.sum(dim=1, keepdim=True)
    if level <= -1 or level >= n_candidates:
        level = n_candidates
    if level > 1:
        for i in range(n_candidates):
            cur_prob = cur_probs[:, i].unsqueeze(1)
            scores_except_i = torch.cat([scores[:, :i], scores[:, i+1:]], dim=1)
            next_prob = permutation_prob(scores_except_i, level=level-1) # [batch_size, (n_candidates-1)*(n_candidates-2)*...(n_candidates-level)]
            probs.append(cur_prob * next_prob)
        probs = torch.cat(probs, dim=1)
        return probs
    else:
        return cur_probs

def ListNet_loss(pred_scores, scores, top_k_permutation=1):
    """
    Args:
        pred_scores: [batch_size, n_candidates]
        scores: [batch_size, n_candidates]
        top_k_permutation: int, top k permutation to compute the loss
    Return:
        loss: [1]
        preds: [batch_size, n_candidates]
    """
    # apply exp
    exp_pred_scores = torch.exp(pred_scores - torch.max(pred_scores, dim=1, keepdim=True)[0]) # [batch_size, n_candidates]
    exp_scores = torch.exp(scores - torch.max(scores, dim=1, keepdim=True)[0]) # [batch_size, n_candidates]
    # compute prob
    logits = permutation_prob(exp_pred_scores, top_k_permutation)
    labels = permutation_prob(exp_scores, top_k_permutation)
    # compute cross entropy loss
    loss = torch.mean(torch.sum(-labels * torch.log(logits + 1e-10), dim=1))
    return loss

def ListMLE_loss(pred_scores, scores):
    """
    Args:
        pred_scores: [batch_size, n_candidates]
        scores: [batch_size, n_candidates]
    Return:
        loss: [1]
    """
    batch_size, n_candidates = pred_scores.shape
    # apply exp
    exp_pred_scores = torch.exp(pred_scores - torch.max(pred_scores, dim=1, keepdim=True)[0]) # [batch_size, n_candidates]
    exp_sum_scores = torch.exp(scores - torch.max(scores, dim=1, keepdim=True)[0]) # [batch_size, n_candidates]

    sorted_indices = torch.argsort(exp_sum_scores, dim=1, descending=True) # [batch_size, n_candidates]
    probs = []
    for i in range(n_candidates):
        order_i_indices = sorted_indices[:, i] # [batch_size]
        left_indices = sorted_indices[:,i:] # [batch_size, n_candidates - i]
        denom_prob = -torch.log(exp_pred_scores[torch.arange(batch_size), order_i_indices])
        numer_prob = torch.log(torch.sum(exp_pred_scores[torch.arange(batch_size).unsqueeze(1), left_indices], dim=1))
        probs.append(denom_prob + numer_prob) # [batch_size]
    loss = torch.sum(torch.stack(probs, dim=1), dim=1) # [batch_size]
    loss = torch.mean(loss)
    return loss

def p_ListMLE_loss(pred_scores, scores):
    """
    Args:
        pred_scores: [batch_size, n_candidates]
        scores: [batch_size, n_candidates]
    Return:
        loss: [1]
    """
    batch_size, n_candidates = pred_scores.shape
    # apply exp
    exp_pred_scores = torch.exp(pred_scores - torch.max(pred_scores, dim=1, keepdim=True)[0]) # [batch_size, n_candidates]
    exp_sum_scores = torch.exp(scores - torch.max(scores, dim=1, keepdim=True)[0]) # [batch_size, n_candidates]

    sorted_indices = torch.argsort(exp_sum_scores, dim=1, descending=True) # [batch_size, n_candidates]
    probs = []
    for i in range(n_candidates):
        order_i_indices = sorted_indices[:, i] # [batch_size]
        left_indices = sorted_indices[:,i:] # [batch_size, n_candidates - i]
        denom_prob = -torch.log(exp_pred_scores[torch.arange(batch_size), order_i_indices])
        numer_prob = torch.log(torch.sum(exp_pred_scores[torch.arange(batch_size).unsqueeze(1), left_indices], dim=1))
        alpha = torch.tensor(2**(n_candidates - i) - 1, dtype=torch.float32).to(pred_scores.device)
        probs.append(alpha*(denom_prob + numer_prob)) # [batch_size]
    loss = torch.sum(torch.stack(probs, dim=1), dim=1) # [batch_size]
    loss = torch.mean(loss)
    return loss

def infoNCE_loss(sim_mat, labels, temperature=0.07):
    """
        InfoNCE loss
        See paper: https://arxiv.org/abs/2002.05709
    Args:
        sim_mat: [batch_size, n_candidates]
        labels: [batch_size, n_candidates]
        temperature: float
    Return:
        loss: [1]
    """
    # compute info loss
    pos_sim = sim_mat * labels / temperature
    neg_sim = sim_mat * (1 - labels) / temperature
    max_sim = torch.max(pos_sim+neg_sim, dim=1, keepdim=True)[0]
    pos_sim = torch.exp(pos_sim - max_sim)
    neg_sim = torch.exp(neg_sim - max_sim)
    pos_sim_sum = torch.sum(torch.exp(pos_sim ), dim=1)
    loss = -torch.log(pos_sim / (pos_sim + neg_sim)).mean()
    return loss

def simcls_loss(sim_mat, target_sim, scores):
    """
    Args:
        sim_mat: [batch_size, n_candidates]
        target_sim: [batch_size]
        scores: [batch_size, n_candidates]
    Return:
        loss: [1]
    """
    loss_func = nn.MarginRankingLoss(margin=0.0)
    loss = torch.tensor(0.0).to(sim_mat.device)
    gold_margin_loss = loss_func(target_sim.repeat(sim_mat.shape[1], 1).transpose(0, 1), sim_mat, torch.ones_like(sim_mat))
    loss += gold_margin_loss
    batch_size, n_candidates = sim_mat.shape
    sorted_idx = torch.argsort(scores, dim=1, descending=True) # [batch_size, n_candidates]
    for i in range(n_candidates):
        for j in range(i+1, n_candidates):
            sim_mat_i = sim_mat[torch.arange(batch_size), sorted_idx[:, i]]
            sim_mat_j = sim_mat[torch.arange(batch_size), sorted_idx[:, j]]
            loss_func = nn.MarginRankingLoss(margin=(j - i) / n_candidates)
            margin_loss = loss_func(sim_mat_i, sim_mat_j, torch.ones_like(sim_mat_i))
            loss += margin_loss
    return loss

def get_dcg(y_pred, y_true, k=10):
    """
    Args:
        y_pred: [size]
        y_true: [size]
        k: int
    Return:
        dcg: [size]
    """
    sorted_idx = torch.argsort(y_pred, descending=True)
    y_true = y_true[sorted_idx][:k]
    y_pred = y_pred[sorted_idx][:k]
    dcg = (torch.pow(2, y_true) - 1) / torch.log2(torch.arange(1, y_true.shape[0]+1, device=y_true.device) + 1)
    return dcg

def get_ndcg(scores, rels):
    """
    Args:
        scores: [batch_size, n_candidates], computed by model
        rels: [batch_size, n_candidates], relevance labels
    """
    if isinstance(scores, np.ndarray):
        scores = torch.tensor(scores)
    if isinstance(rels, np.ndarray):
        rels = torch.tensor(rels)
    batch_size, n_candidates = scores.shape
    # compute dcg
    dcg = [get_dcg(scores[i], rels[i]) for i in range(batch_size)]
    dcg = torch.stack(dcg, dim=0)
    # compute idcg
    idcg = [get_dcg(rels[i], rels[i]) for i in range(batch_size)]
    idcg = torch.stack(idcg, dim=0)
    # compute ndcg
    ndcg = dcg / idcg
    return 1 - ndcg.mean()



def ApproxNDCG_loss(scores, rels, temperature=0.1, k=10):
    """
    Args:
        scores: [batch_size, n_candidates], computed by model
        rels: [batch_size, n_candidates], relevance labels
    """

    def get_approxdcg(y_pred, y_true, k=10, temperature=0.5):
        y_pred = y_pred[:k]
        y_true = y_true[:k]
        approxrank = []
        for i in range(len(y_pred)):
            y_pred_except_i = torch.cat([y_pred[:i], y_pred[i+1:]])
            y_pred_except_i = (y_pred[i] - y_pred_except_i) / temperature
            approxrank_i = 1 + y_pred_except_i.exp()
            approxrank_i = 1 / approxrank_i
            approxrank_i = approxrank_i.sum() + 1
            approxrank.append(approxrank_i)
        approxrank = torch.stack(approxrank, dim=0)

        dcg = (torch.pow(2, y_true) - 1) / torch.log2(approxrank + 1)
        return dcg

    batch_size, n_candidates = scores.shape
    # compute approxdcg
    dcg = [get_approxdcg(scores[i], rels[i], k, temperature) for i in range(batch_size)]
    dcg = torch.stack(dcg, dim=0)
    # compute idcg
    idcg = [get_dcg(rels[i], rels[i], k) for i in range(batch_size)]
    idcg = torch.stack(idcg, dim=0)
    # compute ndcg
    ndcg = dcg / idcg
    return 1 - ndcg.mean()

def ranknet_loss(pred_scores, scores):
    """
    Args:
        pred_scores: [batch_size, n_candidates], 30, 30 -> 15
        scores: [batch_size, n_candidates]

    """
    dif_pred_scores = pred_scores.unsqueeze(1) - pred_scores.unsqueeze(2)
    dif_pred_scores = 1 / (1 + torch.exp(-dif_pred_scores))
    dif_scores = scores.unsqueeze(1) - scores.unsqueeze(2)
    dif_labels = torch.where(dif_scores > 0, torch.ones_like(dif_scores), torch.zeros_like(dif_scores))
    dif_labels = torch.where(dif_scores == 0, torch.ones_like(dif_scores) * 0.5, dif_labels)
    loss = -(dif_labels * torch.log(dif_pred_scores) + (1 - dif_labels) * torch.log(1 - dif_pred_scores)).mean()
    return loss

def lambdarank_loss(pred_scores, scores):
    """
    Args:
        pred_scores: [batch_size, n_candidates]
        scores: [batch_size, n_candidates]
    """
    batch_size, n_candidates = pred_scores.shape

    dif_pred_scores = pred_scores.unsqueeze(1) - pred_scores.unsqueeze(2)
    dif_pred_scores = 1 / (1 + torch.exp(-dif_pred_scores))

    # compute delta ndcg
    idcg = [get_dcg(scores[i], scores[i]) for i in range(batch_size)]
    idcg = torch.stack(idcg, dim=0).sum(dim=1)
    # print("idcg", idcg)
    ranks = torch.argsort(pred_scores, dim=1, descending=True) + 1
    # print("ranks", ranks)
    # print("scores", scores)
    # print("pred_scores", pred_scores)
    # print("dif_pred_scores", dif_pred_scores)
    gain_diff = scores.unsqueeze(1) - scores.unsqueeze(2)
    decay_diff = 1 / torch.log2(ranks.unsqueeze(1) + 1) - 1 / torch.log2(ranks.unsqueeze(2) + 1)
    delta_ndcg = gain_diff * decay_diff / idcg.unsqueeze(1).unsqueeze(2)
    delta_ndcg = torch.abs(delta_ndcg)
    # print("gain_diff", gain_diff)
    # print("decay_diff", decay_diff)
    # print("delta_ndcg", delta_ndcg)
    delta_ndcg = torch.where(delta_ndcg==0.0, torch.ones_like(delta_ndcg), delta_ndcg)
    # multiply delta ndcg
    dif_pred_scores = dif_pred_scores * delta_ndcg

    # compute labels
    dif_scores = scores.unsqueeze(1) - scores.unsqueeze(2)
    dif_labels = torch.where(dif_scores > 0, torch.ones_like(dif_scores), torch.zeros_like(dif_scores))
    dif_labels = torch.where(dif_scores == 0, torch.ones_like(dif_scores) * 0.5, dif_labels)

    # compute loss
    loss = -(dif_labels * torch.log(dif_pred_scores) + (1 - dif_labels) * torch.log(1 - dif_pred_scores)).mean()
    return loss
