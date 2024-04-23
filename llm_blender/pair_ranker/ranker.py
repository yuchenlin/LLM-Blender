import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .layers import (
    MoE,
)
from .loss import (
    simcls_loss,
)


class SummaReranker(nn.Module):
    """
        Sequence Classification Reranker

        Input format:
            [CLS] Source: <source> [SEP] Candidate: <candidate> [SEP]
        Output format:
            Using [CLS] token as the representation of the whole sequence.

        Support 3 objectives of reranking:
            2. multi-task classification (BCE loss)

    """
    def __init__(self, pretrained_model, args, tokenizer=None):
        super(SummaReranker, self).__init__()
        self.args = args
        self.n_tasks = self.args.n_tasks
        self.sub_sampling_mode = self.args.sub_sampling_mode
        self.sub_sampling_ratio = self.args.sub_sampling_ratio
        self.num_pos = self.args.num_pos
        self.num_neg = self.args.num_neg
        self.drop_out = self.args.drop_out

        # LM
        self.pretrained_model = pretrained_model
        self.hidden_size = self.pretrained_model.config.out_hidden_state_size
        self.sigmoid = nn.Sigmoid()
        self.tokenizer = tokenizer

        self.bottom_hidden_size = self.hidden_size
        # shared bottom
        self.fc1 = nn.Linear(self.hidden_size, self.bottom_hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.bottom_hidden_size, self.hidden_size)
        # MoE
        self.moe = MoE(self.n_tasks, self.hidden_size, self.hidden_size, 2*self.n_tasks, self.hidden_size, k=self.n_tasks)
        # towers - one for each task
        self.towers = nn.ModuleList([nn.Linear(self.hidden_size, 1) for i in range(self.n_tasks)])
        self.sigmoid = nn.Sigmoid()

    def _forawrd(self, input_ids, attention_mask):
        """
            SummareReranker
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        Return:
            preds: [batch_size, n_tasks]
            aus_loss: float
        """
        _, seq_len = input_ids.shape
        # encoding source
        to_model_input_ids = input_ids.view(-1, seq_len)
        to_model_attention_mask = attention_mask.view(-1, seq_len)
        outputs = self.pretrained_model(
            input_ids=to_model_input_ids,
            attention_mask=to_model_attention_mask,
            output_hidden_states = True
        )
        encs = outputs["hidden_states"][-1][:, 0, :] # [batch_size * n_candidates, hidden_size]
        # shared bottom
        encs = self.fc2(self.relu(self.fc1(encs)))
        # MoE
        moe_preds, aux_loss = self.moe(encs, train = self.training, collect_gates = not(self.training))
        # go to towers for different tasks
        pred_scores = torch.cat([
            tower(moe_pred) for moe_pred, tower in zip(moe_preds, self.towers)
        ], dim=-1)
        return pred_scores, aux_loss

    def forward(self, input_ids, attention_mask, scores=None):
        """
        Args:
            input_ids: [batch_size, n_candidates, seq_len]
            attention_mask: [batch_size, n_candidates, seq_len]
            scores: [batch_size, n_candidates, n_task]
        """
        if scores is not None:
            labels = torch.eq(scores, torch.max(scores, dim=1, keepdim=True)[0]).float().to(input_ids.device)
            if self.training:
                # sub sampling candidates if needed
                batch_size, n_candidates, seq_len = input_ids.shape
                selected_idx = sub_sampling(
                    self.sub_sampling_mode, self.num_pos, self.num_neg, self.sub_sampling_ratio, scores
                )
                input_ids = input_ids[torch.arange(batch_size).unsqueeze(-1), selected_idx]
                attention_mask = attention_mask[torch.arange(batch_size).unsqueeze(-1), selected_idx]
                scores = scores[torch.arange(batch_size).unsqueeze(-1), selected_idx]
                labels = labels[torch.arange(batch_size).unsqueeze(-1), selected_idx]

        # compute pred scores
        batch_size, n_candidates, seq_len = input_ids.shape
        pred_scores, aux_loss = self._forawrd(input_ids.view(-1, seq_len), attention_mask.view(-1, seq_len))
        pred_scores = pred_scores.reshape(batch_size, n_candidates, -1) # [batch_size, n_candidates, n_tasks]

        if scores is not None:
            # transpose scores and labels to let the last dim be the number of candidates
            scores = scores.transpose(1, 2).reshape(-1, n_candidates)
            labels = labels.transpose(1, 2).reshape(-1, n_candidates)
            pred_scores = pred_scores.transpose(1, 2).reshape(-1, n_candidates) # [batch_size * n_tasks, n_candidates]
            # compute loss
            loss = F.binary_cross_entropy_with_logits(pred_scores, labels)

            loss += aux_loss
        else:
            loss = torch.tensor(0.0).to(input_ids.device)
        # return loss and logits
        pred_scores = pred_scores.reshape(batch_size, -1, n_candidates).transpose(1, 2) # [batch_size, n_candidates, n_tasks]
        pred_scores = torch.mean(pred_scores, dim=-1).detach().reshape(batch_size, n_candidates)
        pred_scores = self.sigmoid(pred_scores)
        outputs = {
            'loss': loss,
            'logits': pred_scores,
        }
        return outputs
    
class DualReranker(nn.Module):
    """
        Dual Encoder Reranker
        Using Roberta as backbone.

        Input format:
            source encoder: [CLS] <source>
            candidate encoder: [CLS] <candiate>
        Output formate:
            Using [CLS] embedding to do rank according

        with the similarity function as follows:
            1. dot product (DP)
            2. L2 distance (L2)
            3. negative log likelihood base on softmax (NLL)
            4. cosine similarity (Cos)

        Using Loss function
            1. InfoNCE from SimCLR (Contrastive)
            2. ListMLE (Liswise ranking)
            3. MoCo (momentum contrastive)
            4. BYOL (bootstrap your own latent)
            5. Barlow Twins

        See DPR for details
    """
    def __init__(self, pretrained_model, args, tokenizer=None):
        super(DualReranker, self).__init__()
        self.args = args
        self.sub_sampling_mode = self.args.sub_sampling_mode
        self.sub_sampling_ratio = self.args.sub_sampling_ratio
        self.num_pos = self.args.num_pos
        self.num_neg = self.args.num_neg

        # LM
        self.source_encoder = pretrained_model
        # self.candidate_encoder = deepcopy(pretrained_model)
        self.candidate_encoder = pretrained_model
        self.hidden_size = self.source_encoder.config.hidden_size
        self.tokenizer = tokenizer

    def _forward(self,
        source_ids,
        source_attention_mask,
        target_ids,
        target_attention_mask,
        candidate_ids,
        candidate_attention_mask,
    ):
        """
            Compute scores for each candidate
        Args:
            source_ids: [batch_size, source_len]
            source_attention_mask: [batch_size, source_len]
            candidate_ids: [batch_size, n_candidates, candidate_len]
            candidate_attention_mask: [batch_size, n_candidates, candidate_len]
        Returns:
            scores: [batch_size, n_candidates]
            target_scores: [batch_size]
        """

        batch_size, n_candidates, candidate_seq_len = candidate_ids.shape
        _, source_seq_len = source_ids.shape

        source_ids = source_ids.view(-1, source_seq_len)
        source_attention_mask = source_attention_mask.view(-1, source_seq_len)
        candidate_ids = candidate_ids.view(-1, candidate_seq_len)
        candidate_attention_mask = candidate_attention_mask.view(-1, candidate_seq_len)

        source_encs = self.source_encoder(
            input_ids=source_ids,
            attention_mask=source_attention_mask,
            output_hidden_states = True
        )["last_hidden_state"][:, 0, :]
        source_encs = F.normalize(source_encs, dim=-1)

        candidate_encs = self.candidate_encoder(
            input_ids=candidate_ids,
            attention_mask=candidate_attention_mask,
            output_hidden_states = True
        )["last_hidden_state"][:, 0, :].reshape(batch_size, n_candidates, -1) # [batch_size, n_candidates, hidden_size]
        candidate_encs = F.normalize(candidate_encs, dim=-1)
        target_encs = self.candidate_encoder(
        input_ids=target_ids,
        attention_mask=target_attention_mask,
        output_hidden_states = True
        )["last_hidden_state"][:, 0, :].reshape(batch_size, 1, -1)
        target_encs = F.normalize(target_encs, dim=-1)
        sim_mat = torch.matmul(source_encs.unsqueeze(1), candidate_encs.transpose(1, 2)).squeeze(1) # [batch_size, n_candidates]
        target_sim_mat = torch.matmul(source_encs.unsqueeze(1), target_encs.transpose(1, 2)).squeeze()
        return sim_mat, target_sim_mat



    def forward(
        self,
        source_ids,
        source_attention_mask,
        target_ids,
        target_attention_mask,
        candidate_ids,
        candidate_attention_mask,
        scores=None):
        """
        Args:
            source_ids: [batch_size, seq_len]
            source_attention_mask: [batch_size, seq_len]
            candidate_ids: [batch_size, n_candidates, seq_len]
            candidate_attention_mask: [batch_size, n_candidates, seq_len]
            scores: [batch_size, n_candidates, n_task]
        """
        if scores is not None:
            labels = torch.eq(
                torch.sum(scores, dim=-1),
                torch.max(torch.sum(scores, dim=-1), dim=1, keepdim=True)[0]
            ).float().to(source_ids.device) # [batch_size, n_candidates]
            # subsampling
            if self.training:
                batch_size, n_candidates, seq_len = candidate_ids.shape
                selected_idx = sub_sampling(self.sub_sampling_mode, self.num_pos, self.num_neg, self.sub_sampling_ratio, scores)
                candidate_ids = candidate_ids[torch.arange(batch_size).unsqueeze(-1), selected_idx]
                candidate_attention_mask = candidate_attention_mask[torch.arange(batch_size).unsqueeze(-1), selected_idx]
                scores = scores[torch.arange(batch_size).unsqueeze(-1), selected_idx]
                labels = labels[torch.arange(batch_size).unsqueeze(-1), selected_idx]
        sim_mat, target_sim_mat = self._forward(
            source_ids, source_attention_mask,
            target_ids, target_attention_mask,
            candidate_ids, candidate_attention_mask)
        if scores is not None:
            sum_scores = torch.sum(scores, dim=-1) # [batch_size, n_candidates]
            loss = simcls_loss(sim_mat, target_sim_mat, sum_scores)
        else:
            loss = torch.tensor(0.0).to(source_ids.device)

        outputs = {
            'loss': loss,
            'logits': sim_mat,
        }
        return outputs

class CrossCompareReranker(nn.Module):
    """
        Cross Encoder Compare Reranker (Cross encoder version of Dual Encoder)
        Using Roberta as backbone

        Given a source text and 2 generated candidates,
        this ranker will compare the 2 candidates and give the better one by
        doing cross attention between query and 2 candidates .

        Input format:
            [CLS] source: <source> [SEP] candidate1: <candidate1> [SEP] candidate2: <candidate2> [SEP]
        Output format:
            the embeddings of the prompt 'source', 'candidate1', 'candidate2'

    """
    def __init__(self, pretrained_model, args, tokenizer):
        super(CrossCompareReranker, self).__init__()
        self.args = args
        self.config = pretrained_model.config
        self.n_tasks = self.args.n_tasks
        self.num_pos = self.args.num_pos
        self.num_neg = self.args.num_neg
        self.sub_sampling_mode = self.args.sub_sampling_mode
        self.sub_sampling_ratio = self.args.sub_sampling_ratio
        self.loss_type = self.args.loss_type
        self.drop_out = self.args.drop_out
        self.inference_mode = self.args.inference_mode
        if hasattr(pretrained_model.config, "is_encoder_decoder"):
            self.is_encoder_decoder = pretrained_model.config.is_encoder_decoder
        else:
            self.is_encoder_decoder = False
        # LM
        self.pretrained_model = pretrained_model
        self.hidden_size = pretrained_model.config.out_hidden_state_size
        self.sep_token_id = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
        self.tokenizer = tokenizer

        self.head_layer = nn.Sequential(
            nn.Dropout(self.drop_out),
            nn.Linear(2*self.hidden_size, 1*self.hidden_size),
            nn.Tanh(),
            nn.Dropout(self.drop_out),
            nn.Linear(1 * self.hidden_size, self.n_tasks),
        )
        self.sigmoid = nn.Sigmoid()

    def compute_loss(self, left_pred_scores, right_pred_scores, left_scores, right_scores):
        """
        Args:
            left_pred_scores: [n_candidates, n_task]
            right_pred_scores: [n_candidates, n_task]
            left_scores: [n_candidates, n_task]
            right_scores: [n_candidates, n_task]
        """

        device = left_pred_scores.device
        loss = torch.tensor(0.0).to(left_pred_scores.device)
        
        if self.loss_type == "BCE":
            dif_scores = (left_scores - right_scores)
            left_labels = (dif_scores > 0).float()
            right_labels = (dif_scores < 0).float()
            cls_loss = torch.tensor(0.0, device=device)
            cls_loss += F.binary_cross_entropy_with_logits(left_pred_scores, left_labels)
            cls_loss += F.binary_cross_entropy_with_logits(right_pred_scores, right_labels)
            cls_loss /= 2
        elif self.loss_type == "instructgpt":
            dif_scores = (left_scores - right_scores)
            left_pred_scores = left_pred_scores * dif_scores.sign()
            right_pred_scores = - right_pred_scores * dif_scores.sign()
            cls_loss = torch.tensor(0.0, device=device)
            cls_loss += - torch.log(torch.sigmoid(left_pred_scores+right_pred_scores)).mean()
        elif self.loss_type == "MSE":
            cls_loss = torch.tensor(0.0, device=device)
            cls_loss += F.mse_loss(left_pred_scores, left_scores)
            cls_loss += F.mse_loss(right_pred_scores, right_scores)
            cls_loss -= (2 * (left_pred_scores - right_pred_scores) * (left_scores - right_scores)).mean()
        elif self.loss_type == "open_instruct_BCE":
            assert all((left_scores == 1.0) + (left_scores == 0.0)), "open_instruct_BCE only support 0/1 labels"
            assert all((right_scores == 1.0) + (right_scores == 0.0)), "open_instruct_BCE only support 0/1 labels"
            left_labels = (left_scores == 1.0).float()
            right_labels = (right_scores == 1.0).float()
            cls_loss = torch.tensor(0.0, device=device)
            cls_loss += F.binary_cross_entropy_with_logits(left_pred_scores, left_labels)
            cls_loss += F.binary_cross_entropy_with_logits(right_pred_scores, right_labels)
            cls_loss /= 2
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        loss += cls_loss
        return loss
    
    def reduce(self, source_encs, cand1_encs, cand2_encs):
        """
        Args:
            source_encs: [batch_size, hidden_size]
            cand1_encs: [batch_size, hidden_size]
            cand2_encs: [batch_size, hidden_size]
        Returns:
            left_pred_scores: [batch_size, n_task]
            right_pred_scores: [batch_size, n_task]
        """
        # reduce
        aux_loss = torch.tensor(0.0, device=cand1_encs.device)
        if source_encs is not None:
            source_cand1_encs = torch.cat([source_encs, cand1_encs], dim=-1)
            source_cand2_encs = torch.cat([source_encs, cand2_encs], dim=-1)
            left_pred_scores = self.head_layer(source_cand1_encs)
            right_pred_scores = self.head_layer(source_cand2_encs)
        else:
            left_pred_scores = self.single_head_layer(cand1_encs)
            right_pred_scores = self.single_head_layer(cand2_encs)

        return left_pred_scores, right_pred_scores, aux_loss

    def _forward(
        self,
        source_ids,
        source_attention_mask,
        cand1_ids,
        cand1_attention_mask,
        cand2_ids,
        cand2_attention_mask,
        cand1_scores=None,
        cand2_scores=None,
    ):
        """
            Compute scores for each candidate pairs
        Args:
            source_ids: [batch_size, seq_len]
            source_attention_mask: [batch_size, seq_len]
            cand1_ids: [batch_size, cand_len]
            cand1_attention_mask: [batch_size, cand_len]
            cand2_ids: [batch_size, cand_len]
            cand2_attention_mask: [batch_size, cand_len]
            cand1_scores: [batch_size, n_task]
            cand2_scores: [batch_size, n_task]
        Returns:
            outputs dict:
                loss: scalar
                preds (optional): [batch_size, n_task]
        """
        device = source_ids.device
        # clone 
        cand1_ids = cand1_ids.clone()
        cand2_ids = cand2_ids.clone()
        # replace <candidate> with <candidate1> and <candidate2> respectively
        cand1_idxs = torch.where(cand1_ids == self.tokenizer.cand_prefix_id)
        cand2_idxs = torch.where(cand2_ids == self.tokenizer.cand_prefix_id)
        cand1_ids[cand1_idxs] = self.tokenizer.cand1_prefix_id
        cand2_ids[cand2_idxs] = self.tokenizer.cand2_prefix_id
        if self.is_encoder_decoder:
            decoder_input_ids, decoder_attention_mask = self.cat_ids(
                cand1_ids, cand1_attention_mask,
                cand2_ids, cand2_attention_mask,
            )
            decoder_input_ids = decoder_input_ids
            outputs = self.pretrained_model(
                input_ids=source_ids,
                attention_mask=source_attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                output_hidden_states=True,
            )
            # get the special token <source>, <candidate1> and <candidate2>
            # source_encs = None # not used
            source_idxs = torch.where(source_ids == self.tokenizer.source_prefix_id)
            source_encs = outputs.encoder_hidden_states[-1][source_idxs[0], source_idxs[1], :]
            cand1_idxs = torch.where(decoder_input_ids == self.tokenizer.cand1_prefix_id)
            cand1_encs = outputs.decoder_hidden_states[-1][cand1_idxs[0], cand1_idxs[1], :]
            cand2_idxs = torch.where(decoder_input_ids == self.tokenizer.cand2_prefix_id)
            cand2_encs = outputs.decoder_hidden_states[-1][cand2_idxs[0], cand2_idxs[1], :]
        else:
            input_ids, attention_mask = self.cat_ids(
                source_ids, source_attention_mask,
                cand1_ids, cand1_attention_mask,
                cand2_ids, cand2_attention_mask,
            )
            # trim batch padding ids
            keep_column_mask = attention_mask.ne(0).any(dim=0)
            input_ids = input_ids[:, keep_column_mask]
            attention_mask = attention_mask[:, keep_column_mask]
            outputs = self.pretrained_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            encs = outputs.hidden_states[-1]
            source_idxs = torch.where(input_ids == self.tokenizer.source_prefix_id)
            source_encs = encs[source_idxs[0], source_idxs[1], :]
            cand1_idxs = torch.where(input_ids == self.tokenizer.cand1_prefix_id)
            cand1_encs = encs[cand1_idxs[0], cand1_idxs[1], :]
            cand2_idxs = torch.where(input_ids == self.tokenizer.cand2_prefix_id)
            cand2_encs = encs[cand2_idxs[0], cand2_idxs[1], :]
        # reduce
        left_pred_scores, right_pred_scores, aux_loss = self.reduce(source_encs, cand1_encs, cand2_encs)

        loss = torch.tensor(0.0, device=device)
        if cand1_scores is not None and cand2_scores is not None:
            loss += self.compute_loss(left_pred_scores, right_pred_scores, cand1_scores, cand2_scores)
            loss += aux_loss

        preds = (left_pred_scores - right_pred_scores).mean(dim=-1)
        outputs = {
            'loss': loss,
            'logits': preds,
        }
        return outputs

    def sampling(
        self,
        candidate_ids,
        candidate_attention_mask, 
        scores):
        """
        Args:
            candidate_ids: [n_candidates, cand_len]
            candidate_attention_mask: [n_candidates, cand_len]
            scores: [n_candidates, n_task]
            n_pair: int
            device: torch.device
        """
        device = scores.device

        # remove duplicate candidates
        unique_idx = []
        unique_scores = []
        for idx, score in enumerate(scores.mean(dim=-1)):
            is_new = True
            for u_idx in unique_idx:
                if torch.all(candidate_ids[u_idx] == candidate_ids[idx]):
                    is_new = False
                    break
            if is_new:
                unique_idx.append(idx)
                unique_scores.append(score)
        unique_idx = torch.tensor(unique_idx, device=device)
        unique_scores = scores[unique_idx]
        unique_candidate_ids = candidate_ids[unique_idx]
        unique_candidate_attention_mask = candidate_attention_mask[unique_idx]
        unique_n_candidates = len(unique_idx)

        # NOTE: different sampling strategy
        if self.sub_sampling_mode == "top_bottom":
            n_pair = min(self.num_pos, self.num_neg)
            sorted_idx = torch.argsort(unique_scores.mean(-1), descending=True) # [batch_size, n_candidates]
            left_idx = sorted_idx[:n_pair]
            right_idx = sorted_idx[-n_pair:]
        elif self.sub_sampling_mode == "random":
            # 2. random sampling
            n_pair = max(int(unique_n_candidates * self.sub_sampling_ratio), 1)
            left_idx = torch.randint(0, unique_n_candidates, (n_pair), device=device)
            right_idx = torch.randint(0, unique_n_candidates, (n_pair), device=device)
        elif self.sub_sampling_mode == "uniform":
            # 3. uniform sampling
            step = torch.tensor(unique_n_candidates / (unique_n_candidates * self.sub_sampling_ratio), dtype=torch.long)
            sorted_idx = torch.argsort(unique_scores.mean(-1), descending=True) # [batch_size, n_candidates]
            left_idx = sorted_idx[0:-step:step]
            right_idx = sorted_idx[step::step]
        elif self.sub_sampling_mode == "all_pair":
            # 4. all pair C(n, 2)
            combs = torch.combinations(torch.arange(unique_n_candidates), r=2).to(device)
            if combs.shape[0] == 0:
                left_idx = torch.tensor([0], device=device)
                right_idx = torch.tensor([0], device=device)
            else:
                n_pair = min(self.num_pos, self.num_neg)
                rand_idx = torch.randperm(combs.shape[0], device=device)
                combs = combs[rand_idx[:n_pair]]
                left_idx = combs[:, 0]
                right_idx = combs[:, 1]
        else:
            raise ValueError(f"Unknown sampling mode: {self.sub_sampling_mode}")

        n_pair = left_idx.shape[0]
        shuffle_flag = torch.rand(n_pair, device=device) < 0.5
        _left_idx = torch.where(shuffle_flag, left_idx, right_idx)
        _right_idx = torch.where(shuffle_flag, right_idx, left_idx)
        left_idx, right_idx = _left_idx, _right_idx
        cand1_ids = unique_candidate_ids[left_idx]
        cand2_ids = unique_candidate_ids[right_idx]
        cand1_attention_mask = unique_candidate_attention_mask[left_idx]
        cand2_attention_mask = unique_candidate_attention_mask[right_idx]
        cand1_scores = unique_scores[left_idx]
        cand2_scores = unique_scores[right_idx]
        return {
            "cand1_ids": cand1_ids,
            "cand2_ids": cand2_ids,
            "cand1_attention_mask": cand1_attention_mask,
            "cand2_attention_mask": cand2_attention_mask,
            "cand1_scores": cand1_scores,
            "cand2_scores": cand2_scores,
            "n_pair": n_pair,
        }

    def cat_ids(self, ids1, masks1, ids2, masks2, ids3=None, masks3=None):
        """
        Concatenate ids and masks, move padding to the end
        Args:
            ids1, masks1: source ids and masks
            ids2, masks2: candidate ids and masks or the concatentated ids and masks
            ids3, masks3 (optional): candidate ids and masks
        """
        assert ids1.shape[:-1] == ids2.shape[:-1]
        assert ids1.shape[:-1] == ids3.shape[:-1] if ids3 is not None else True
        ori_shape = ids1.shape[:-1]
        ids1 = ids1.reshape(-1, ids1.shape[-1])
        ids2 = ids2.reshape(-1, ids2.shape[-1])
        masks1 = masks1.reshape(-1, masks1.shape[-1])
        masks2 = masks2.reshape(-1, masks2.shape[-1])
        bz = ids1.shape[0]
        sep_token_idx1 = ids1.eq(self.sep_token_id)
        sep_token_idx2 = ids2.eq(self.sep_token_id)
        assert sep_token_idx1.sum(-1).eq(sep_token_idx1.sum(-1)[0]).all(), sep_token_idx1.sum(-1)
        assert sep_token_idx2.sum(-1).eq(sep_token_idx2.sum(-1)[0]).all(), sep_token_idx2.sum(-1)
        assert sep_token_idx1.sum(-1).ge(1).all(), self.tokenizer.decode(ids1[0])
        assert sep_token_idx2.sum(-1).ge(1).all(), sep_token_idx2.sum(-1)
        sep_token_idx1 = sep_token_idx1.nonzero()[:, 1].reshape(bz, -1)[:, -1]
        sep_token_idx2 = sep_token_idx2.nonzero()[:, 1].reshape(bz, -1)[:, -1]
        cat_ids = []
        cat_masks = []
        if ids3 is not None:
            ids3 = ids3.view(-1, ids3.shape[-1])
            masks3 = masks3.view(-1, masks3.shape[-1])
            sep_token_idx3 = ids3.eq(self.sep_token_id)
            assert sep_token_idx3.sum(-1).eq(sep_token_idx3.sum(-1)[0]).all(), sep_token_idx3.sum(-1)
            sep_token_idx3 = sep_token_idx3.nonzero()[:, 1].reshape(bz, -1)[:, -1]
            for i in range(bz):
                cat_ids.append(torch.cat([
                    ids1[i, :sep_token_idx1[i] + 1],
                    ids2[i, :sep_token_idx2[i] + 1],
                    ids3[i, :sep_token_idx3[i] + 1],
                    ids1[i, sep_token_idx1[i] + 1:],
                    ids2[i, sep_token_idx2[i] + 1:],
                    ids3[i, sep_token_idx3[i] + 1:],
                ], dim=0))
                cat_masks.append(torch.cat([
                    masks1[i, :sep_token_idx1[i] + 1],
                    masks2[i, :sep_token_idx2[i] + 1],
                    masks3[i, :sep_token_idx3[i] + 1],
                    masks1[i, sep_token_idx1[i] + 1:],
                    masks2[i, sep_token_idx2[i] + 1:],
                    masks3[i, sep_token_idx3[i] + 1:],
                ], dim=0))
        else:
            for i in range(bz):
                cat_ids.append(torch.cat([
                    ids1[i, :sep_token_idx1[i] + 1],
                    ids2[i, :sep_token_idx2[i] + 1],
                    ids1[i, sep_token_idx1[i] + 1:],
                    ids2[i, sep_token_idx2[i] + 1:],
                ], dim=0))
                cat_masks.append(torch.cat([
                    masks1[i, :sep_token_idx1[i] + 1],
                    masks2[i, :sep_token_idx2[i] + 1],
                    masks1[i, sep_token_idx1[i] + 1:],
                    masks2[i, sep_token_idx2[i] + 1:],
                ], dim=0))
        cat_ids = torch.stack(cat_ids, dim=0)
        cat_masks = torch.stack(cat_masks, dim=0)
        cat_ids = cat_ids.reshape(ori_shape + (-1,))
        cat_masks = cat_masks.reshape(ori_shape + (-1,))
        return cat_ids, cat_masks

    def _bubble_predict(
        self,
        source_ids,
        source_attention_mask,
        candidate_ids,
        candidate_attention_mask,
        scores=None,
        num_runs=1,
        best_or_worst="best",
    ):
        """
            bubble prediction
        """
        device = source_ids.device
        outputs = {}
        batch_size, src_len = source_ids.shape
        batch_size, n_candidates, cand_len = candidate_ids.shape
        num_runs = n_candidates if num_runs < 0 else num_runs
        num_runs = np.clip(num_runs, 1, n_candidates)

        permu = torch.randperm(n_candidates).repeat(batch_size, 1).to(device) # [batch_size, n_candidates] random
        loss = torch.tensor(0.0).to(device)
        cur_idxs = []
        next_idxs = []
        better_idxs = []
        cand1_prefix_ids = torch.tensor(self.tokenizer.cand1_prefix_id).to(device)
        cand1_prefix_ids = cand1_prefix_ids.expand(batch_size, 1)
        cand2_prefix_ids = torch.tensor(self.tokenizer.cand2_prefix_id).to(device)
        cand2_prefix_ids = cand2_prefix_ids.expand(batch_size, 1)
        for i in range(num_runs):
            for j in range(i, n_candidates-1):
                cur_idx = permu[:, j].clone()
                next_idx = permu[:, j+1].clone() # [batch_size]
                batch_idx = torch.arange(batch_size).to(device)
                # left-right
                left_cand_ids = candidate_ids[batch_idx, cur_idx]
                right_cand_ids = candidate_ids[batch_idx, next_idx]
                left_cand_attention_mask = candidate_attention_mask[batch_idx, cur_idx]
                right_cand_attention_mask = candidate_attention_mask[batch_idx, next_idx]
                if scores is not None:
                    left_scores = scores[batch_idx, cur_idx]
                    right_scores = scores[batch_idx, next_idx]
                else:
                    left_scores = None
                    right_scores = None
                _outputs = self._forward(
                    source_ids, source_attention_mask,
                    left_cand_ids, left_cand_attention_mask,
                    right_cand_ids, right_cand_attention_mask,
                    left_scores, right_scores,
                )
                loss += _outputs['loss']
                preds = _outputs['logits']
                # right-left
                _outputs = self._forward(
                    source_ids, source_attention_mask,
                    right_cand_ids, right_cand_attention_mask,
                    left_cand_ids, left_cand_attention_mask,
                    right_scores, left_scores,
                )
                loss += _outputs['loss']
                preds_inv = -_outputs['logits']

                if best_or_worst == "best":
                    permu[:, j] = torch.where(preds + preds_inv <= 0, cur_idx, next_idx)
                    permu[:, j+1] = torch.where(preds + preds_inv > 0, cur_idx, next_idx)
                elif best_or_worst == "worst":
                    permu[:, j] = torch.where(preds + preds_inv >= 0, cur_idx, next_idx)
                    permu[:, j+1] = torch.where(preds + preds_inv < 0, cur_idx, next_idx)
                assert torch.ne(permu[:, j], permu[:, j+1]).all()
                better_idx = permu[:, j+1].clone()
                better_idxs.append(better_idx)
                next_idxs.append(next_idx)
                cur_idxs.append(cur_idx)

        outputs = {}
        outputs['loss'] = loss / 2
        outputs["select_process"] = []
        outputs["select_process"].append(torch.stack(cur_idxs, dim=1))
        outputs["select_process"].append(torch.stack(next_idxs, dim=1))
        outputs["select_process"].append(torch.stack(better_idxs, dim=1))
        outputs["select_process"] = torch.stack(outputs["select_process"], dim=1) # [batch_size, 3, n_candidates]
        outputs["loss"] /= outputs['select_process'].shape[-1]

        return outputs

    def _full_predict(
        self,
        source_ids,
        source_attention_mask,
        candidate_ids,
        candidate_attention_mask,
        scores=None,
    ):
        """
            Do predict over each group of candidates
        Args:
            source_ids: [batch_size, src_len]
            source_attention_mask: [batch_size, src_len]
            candidate_ids: [batch_size, n_candidates, cand_len]
            candidate_attention_mask: [batch_size, n_candidates, cand_len]
            scores: [batch_size, n_candidates, n_tasks] (optional)
        Returns:
            loss: scalar if scores is not None
            logits: [batch_size, n_candidates, n_candidates]
                complete pairwise comparison as a comparison matrix for each instance in the batch
        """
        device = source_ids.device
        outputs = {}
        batch_size, src_len = source_ids.shape
        batch_size, n_candidates, cand_len = candidate_ids.shape

        loss = torch.tensor(0.0).to(device)

        compare_results = torch.zeros(batch_size, n_candidates, n_candidates, device=device)
        for i in range(n_candidates):
            for j in range(n_candidates):
                if i == j:
                    continue
                left_cand_ids = candidate_ids[:, i]
                right_cand_ids = candidate_ids[:, j]
                left_cand_attention_mask = candidate_attention_mask[:, i]
                right_cand_attention_mask = candidate_attention_mask[:, j]
                if scores is not None:
                    left_scores = scores[:, i]
                    right_scores = scores[:, j]
                else:
                    left_scores = None
                    right_scores = None
                _outputs = self._forward(
                    source_ids, source_attention_mask,
                    left_cand_ids, left_cand_attention_mask,
                    right_cand_ids, right_cand_attention_mask,
                    left_scores, right_scores,
                )
                loss += _outputs['loss']
                preds = _outputs['logits']
                compare_results[:, i, j] = preds

        outputs['loss'] = loss / (n_candidates * (n_candidates - 1))
        outputs['logits'] = compare_results # [batch_size, n_candidates, n_candidates]

        return outputs

    def predict(
        self,
        source_ids,
        source_attention_mask,
        candidate_ids,
        candidate_attention_mask,
        scores=None,
        mode=None,
    ):
        """
            Do predict over each group of candidates
        Args:
            always:
                source_ids: [batch_size, src_len]
                source_attention_mask: [batch_size, src_len]
                candidate_ids: [batch_size, n_candidates, cand_len]
                candidate_attention_mask: [batch_size, n_candidates, cand_len]
                scores: [batch_size, n_candidates, n_tasks]
        """
        device = source_ids.device
        outputs = {}
        mode = mode or self.inference_mode
        if mode == "bubble":
            outputs = self._bubble_predict(
                source_ids,
                source_attention_mask,
                candidate_ids,
                candidate_attention_mask,
                scores,
            )
        elif mode == "full":
            outputs = self._full_predict(
                source_ids,
                source_attention_mask,
                candidate_ids,
                candidate_attention_mask,
                scores,
            )
        else:
            raise NotImplementedError
        return outputs

    def forward(
        self,
        source_ids,
        source_attention_mask,
        candidate_ids,
        candidate_attention_mask,
        scores,
    ):
        """
            Compute scores for each candidate
        Args:
            source_ids: [batch_size, src_len]
            source_attention_mask: [batch_size, src_len]
            target_ids: [batch_size, cand_len]
            target_attention_mask: [batch_size, cand_len]
            candidate_ids: [batch_size, n_candidates, cand_len]
            candidate_attention_mask: [batch_size, n_candidates, cand_len]
            scores: [batch_size, n_candidates, n_tasks]
        """
        device = source_ids.device
        outputs = {}
        # passing in as individual
        batch_size, src_len = source_ids.shape
        batch_size, n_candidates, cand_len = candidate_ids.shape
        if self.training:
            # subsampling
            batch_size, n_candidates, n_tasks = scores.shape

            cand1_ids, cand2_ids, cand1_attention_mask, cand2_attention_mask, cand1_scores, cand2_scores = [], [], [], [], [], []
            extended_source_ids, extended_source_attention_mask = [], []
            for i in range(batch_size):
                sampling_results = self.sampling(candidate_ids[i], candidate_attention_mask[i], scores[i])
                cand1_ids.append(sampling_results["cand1_ids"])
                cand2_ids.append(sampling_results["cand2_ids"])
                cand1_attention_mask.append(sampling_results["cand1_attention_mask"])
                cand2_attention_mask.append(sampling_results["cand2_attention_mask"])
                cand1_scores.append(sampling_results["cand1_scores"])
                cand2_scores.append(sampling_results["cand2_scores"])
                extended_source_ids.append(source_ids[i].unsqueeze(0).repeat(sampling_results["cand1_ids"].shape[0], 1))
                extended_source_attention_mask.append(source_attention_mask[i].unsqueeze(0).repeat(sampling_results["cand1_ids"].shape[0], 1))
            cand1_ids = torch.cat(cand1_ids, dim=0)
            cand2_ids = torch.cat(cand2_ids, dim=0)
            cand1_attention_mask = torch.cat(cand1_attention_mask, dim=0)
            cand2_attention_mask = torch.cat(cand2_attention_mask, dim=0)
            cand1_scores = torch.cat(cand1_scores, dim=0)
            cand2_scores = torch.cat(cand2_scores, dim=0)
            extended_source_ids = torch.cat(extended_source_ids, dim=0)
            extended_source_attention_mask = torch.cat(extended_source_attention_mask, dim=0)
            n_pair = cand1_ids.shape[0]
            outputs = self._forward(
                extended_source_ids,
                extended_source_attention_mask,
                cand1_ids,
                cand1_attention_mask,
                cand2_ids,
                cand2_attention_mask,
                cand1_scores,
                cand2_scores,
            )
        else:
            outputs = self.predict(
                source_ids,
                source_attention_mask,
                candidate_ids,
                candidate_attention_mask,
                scores,
            )

        return outputs

def sub_sampling(mode, num_pos, num_neg, ratio, scores):
    """
    Args:
        mode: sub sampling mode
        num_pos: number of positive samples
        num_neg: number of negative samples
        ratio: ratio of positive samples
        scores: [batch_size, candidate, n_task]

    Returns:
        selected_idx: [batch_size, n_pos+n_neg] or [batch_size, n_candidates * ratio]

    """
    batch_size, n_candidates, n_task = scores.shape

    if mode == "uniform":
        sorted_idx = torch.argsort(torch.sum(scores, dim=-1), dim=1, descending=True)
        step = torch.tensor(n_candidates / (n_candidates * ratio), dtype=torch.long)
        selected_idx = sorted_idx[:, ::step]
        shuffled_idx = torch.randperm(selected_idx.shape[1])
        selected_idx = selected_idx[:, shuffled_idx]
    elif mode == "random":
        selected_idx = torch.stack([
            torch.randperm(n_candidates)[:int(n_candidates * ratio)] for _ in range(batch_size)
        ], dim=0) # [batch_size, n_candidates * ratio]
    elif mode in ["top_bottom", "top_random", "random_bottom"]:
        selected_idx = []
        for i in range(batch_size):
            idx = np.arange(n_candidates)
            # remove duplicate candidates, cpu
            unique_idx = []
            unique_scores = []
            for j, score in enumerate(torch.sum(scores[i], dim=-1)):
                if score not in unique_scores:
                    unique_idx.append(idx[j])
                    unique_scores.append(score.item())
            unique_idx = np.array(unique_idx)
            unique_scores = np.array(unique_scores)
            # only select a few pos and neg candidates
            sorted_idx = np.argsort(unique_scores)[::-1]

            if mode == "top_bottom":
                pos_idx = sorted_idx[:num_pos] # top
                neg_idx = sorted_idx[-num_neg:] # bottom
            elif mode == "top_random":
                pos_idx = sorted_idx[:num_pos] # top
                neg_idx = np.random.choice(sorted_idx[num_pos:], num_neg, replace=False) # random
            elif mode == "random_bottom":
                pos_idx = np.random.choice(sorted_idx[:-num_neg], num_pos, replace=False) # random
                neg_idx = sorted_idx[-num_neg:] # bottom
            else:
                raise NotImplementedError
            idx = np.concatenate([pos_idx, neg_idx])
            np.random.shuffle(idx)
            idx = unique_idx[idx]
            selected_idx.append(idx)
        selected_idx = torch.tensor(selected_idx)
    elif mode == "none":
        selected_idx = torch.arange(n_candidates)
        selected_idx = selected_idx.unsqueeze(0).repeat(batch_size, 1)
    else:
        raise NotImplementedError

    return selected_idx

