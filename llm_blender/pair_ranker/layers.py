import torch
import torch.nn as nn

from .model_moe import MoE
class ModelMultitaskRegression(nn.Module):
    """
        This class is used to train the model for the multitask regression task.
        Use as a layer return the loss
    """
    def __init__(self, n_tasks, input_size, hidden_size):
        super(ModelMultitaskRegression, self).__init__()
        self.n_tasks = n_tasks
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, n_tasks)
        self.gelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        x = self.gelu(x)
        x = self.linear2(x)
        x = self.sigmoid(x) # do regression on [0, 1] scale
        return x, None # no loss


class MoERegression(nn.Module):
    """
        This class is modified from the original implementation of the paper:
        SummaReranker: A Multi-Task Mixture-of-Experts Re-ranking Framework for Abstractive Summarization
        paper: https://arxiv.org/abs/2203.06569
        code: https://github.com/Ravoxsg/SummaReranker-ACL-22-/blob/main/src/summareranker/model.py
        We thank the authors for sharing their code.

        In our implementation, we get passed in embedding from dual encoder and
        apply the multitask binary classification head on top of it.
        We only this layer to compute the auxiliary loss to help the generation.
        We don't use this layer for any prediction.
    """

    def __init__(self, n_tasks, input_size, hidden_size, num_experts=None, expert_hidden_size=1024, k=None):
        super(MoERegression, self).__init__()
        self.n_tasks = n_tasks
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.expert_hidden_size = expert_hidden_size
        if num_experts is None:
            num_experts = 2 * n_tasks
            self.num_experts = num_experts
        if k is None:
            k = num_experts // 2
            self.k = k
        # shared bottom
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # MoE
        self.moe = MoE(n_tasks, hidden_size, hidden_size, num_experts, expert_hidden_size, k)
        # towers - one for each task
        self.towers = nn.ModuleList([nn.Linear(hidden_size, 1) for _ in range(n_tasks)])
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        _, n_candidates, _ = x.size()
        pred_scores = []
        total_aux_loss = torch.tensor(0.0, device=x.device)
        for i in range(n_candidates):
            encs = x[:, i, :] # [CLS]
            preds_i = self.fc2(self.relu(self.fc1(encs))) # shared bottom
            train = self.training
            preds_i, aux_loss = self.moe(preds_i, train = train, collect_gates = not(train))
            pred_scores_i = []
            for j in range(self.n_tasks):
                # pred
                preds_i_j = self.towers[j](preds_i[j])[:, 0]
                pred_scors_i_j = self.sigmoid(preds_i_j)
                pred_scores_i.append(pred_scors_i_j)
            pred_scores_i = torch.stack(pred_scores_i, dim=1)
            pred_scores.append(pred_scores_i)
            total_aux_loss += aux_loss
        pred_scores = torch.stack(pred_scores, dim=1)
        return pred_scores, total_aux_loss

