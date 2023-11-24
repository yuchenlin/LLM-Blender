import torch
import torch.nn as nn
import numpy as np
import os
import wandb
import logging
import json
from transformers.trainer import Trainer
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers import EvalPrediction
from typing import Dict, List, Optional, Tuple, Union, Any
from torch.utils.data import Dataset
from dataclasses import asdict
logger = logging.getLogger(__name__)

class RerankerTrainer(Trainer):
    def evaluate(
        self,
        **kwargs,
    ) -> Dict[str, float]:
        metrics = super().evaluate(**kwargs)
        if self.is_world_process_zero():
            if "wandb" == self.args.report_to or "wandb" in self.args.report_to:
                wandb.log(metrics)
        return metrics

    def save_model(self, output_dir: Optional[str] = None, **kwargs):
        if self.is_world_process_zero():
            super().save_model(output_dir, **kwargs)
            model = self.model.module if hasattr(self.model, "module") else self.model
            json.dump(asdict(model.args), open(os.path.join(output_dir, "config.json"), "w"), indent=4)
            
class FiDTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
        )

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if self.model.config.use_aux_loss:
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                _, aux_loss = self.model.module.compute_auxiliary_loss(input["scores"])
            else:
                _, aux_loss = self.model.compute_auxiliary_loss(input["scores"])
            loss += aux_loss

        return (loss, outputs) if return_outputs else loss

def compute_metrics_for_scr(eval_pred: EvalPrediction) -> Dict[str, float]:
    preds, labels = eval_pred # pred_scores [batch_size, num_candidates], scores [batch_size, num_candidates, n_tasks]
    pred_scores = preds
    scores = labels
    agg_scores = np.mean(scores, axis=-1) # aggregate scores

    sort_indices = np.flip(np.argsort(agg_scores, axis=-1), axis=-1) # (batch_size, n_candidates), expected ranks
    ranks = np.zeros_like(sort_indices)
    ranks[np.arange(sort_indices.shape[0])[:, None], sort_indices] = np.arange(sort_indices.shape[-1])
    pred_sort_indices = np.flip(np.argsort(pred_scores, axis=-1), axis=-1) # (batch_size, n_candidates), predicted ranks
    pred_ranks = np.zeros_like(pred_sort_indices)
    pred_ranks[np.arange(pred_sort_indices.shape[0])[:, None], pred_sort_indices] = np.arange(pred_sort_indices.shape[-1])

    # compute selection scores
    sel_idx = np.argmax(pred_scores, axis=1) # [batch_size]
    sel_scores = scores[np.arange(scores.shape[0]), sel_idx] # [batch_size, n_task]
    sel_ranks = ranks[np.arange(ranks.shape[0]), sel_idx] # [batch_size]
    sel_acc = np.mean((sel_ranks == 0)) # scalar

    # compute oracle scores for reference
    oracle_sel_idx = np.argmax(agg_scores, axis=1) # [batch_size]
    oracle_sel_scores = scores[np.arange(scores.shape[0]), oracle_sel_idx] # [batch_size, n_task]
    oracle_sel_ranks = ranks[np.arange(ranks.shape[0]), oracle_sel_idx] # [batch_size]
    oracle_sel_acc = np.mean((oracle_sel_ranks == 0)) # scalar

    metrics = {
        "sel": {
            "acc": sel_acc,
            "rank": np.mean(sel_ranks),
        },
        "oracle": {
            "acc": oracle_sel_acc,
            "rank": np.mean(oracle_sel_ranks),
        },
        "dev_score": np.mean(sel_scores[:, 0]), # dev score used for save checkpoint,
    }
    for i in range(sel_scores.shape[-1]):
        metrics["sel"]["metric{}".format(i+1)] = np.mean(sel_scores[:, i])
        metrics["oracle"]["metric{}".format(i+1)] = np.mean(oracle_sel_scores[:, i])
    return metrics



def compute_metrics_for_pairranker(eval_pred: EvalPrediction) -> Dict[str, float]:
    """
    Compute metrics for the model.
    Args:

    """
    preds, labels = eval_pred # scores [batch_size, n_candidates, n_tasks]
    logits = preds
    
    scores = labels # [batch_size, n_candidates, n_tasks]
    # scores = scores[:, :-1] # debug
    mean_scores = np.mean(scores, axis=-1) # [batch_size, n_candidates]
    batch_size, n_candidates, n_tasks = scores.shape

    # get the predicted best index
    if logits.shape[1] == 3:
        # bubble
        pred_best_idx = logits[:, 2, -1]
    elif logits.shape == (batch_size, n_candidates, n_candidates):
        # full
        pred_best_idx = np.argmax(np.mean(logits, axis=2) - np.mean(logits, axis=1), axis=-1)
    else:
        raise ValueError("Invalid logits shape: {}".format(logits.shape))

    # metric_scores, denormalized these scores
    pred_best_scores = scores[np.arange(batch_size), pred_best_idx]
    oracle_best_scores = scores[np.arange(batch_size), np.argmax(mean_scores, axis=-1)]
    metrics = {
        "sel": {},
        "oracle": {},
        "top_beam": {},
        "gain": {},
    }
    for i in range(n_tasks):
        metrics["sel"]["metric_{}".format(i+1)] = np.mean(pred_best_scores[:, i])
        metrics["oracle"]["metric_{}".format(i+1)] = np.mean(oracle_best_scores[:, i])
        metrics["top_beam"]["metric_{}".format(i+1)] = np.mean(scores[:, 0, i])
        metrics["gain"]["metric_{}".format(i+1)] = metrics["sel"]["metric_{}".format(i+1)] / metrics["top_beam"]["metric_{}".format(i+1)] - 1
    metrics['dev_score'] = metrics['sel']['metric_1']

    return metrics


