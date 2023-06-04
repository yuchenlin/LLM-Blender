from sklearn.utils import shuffle
import torch
import random
import json
import numpy as np

def encode_texts(texts, tokenizer, max_length):
    """
    Args:
        texts List[str]: [n_texts]
    Returns:
        input_ids: [n_texts, max_length]
        attention_mask: [n_texts, max_length]
    """
    p = tokenizer.batch_encode_plus(
        texts,
        max_length=max_length,
        padding='max_length',
        return_tensors='pt',
        truncation=True
    )
    return p['input_ids'], p['attention_mask']

def encode_batch_text(batch_texts, tokenizer, max_length):
    """
    Args:
        batch_texts List[str]: [batch_size, n_texts]
    Returns:
        batch_input_ids: [batch_size, n_texts, max_length]
        batch_attention_mask: [batch_size, n_texts, max_length]
    """
    encoded_ids, encoded_masks = [], []
    for k, texts in enumerate(batch_texts):
        if isinstance(texts, str):
            texts = [texts]
        ids, mask = encode_texts(texts, tokenizer, max_length)
        encoded_ids.append(ids[None])
        encoded_masks.append(mask[None])
    encoded_ids = torch.cat(encoded_ids, dim=0)
    encoded_masks = torch.cat(encoded_masks, dim=0)
    return encoded_ids, encoded_masks

def get_truncated_text(texts, tokenizer, max_length):
    """
        Truncate the texts to max_length
    """
    truncated_texts = []
    for text in texts:
        tokens = tokenizer.encode(
            text,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
        )
        truncated_texts.append(tokenizer.decode(tokens, skip_special_tokens=True))
    return truncated_texts

class SCRCollator(object):
    def __init__(
        self,
        source_maxlength,
        tokenizer,
        candidate_maxlength,
        source_prefix=None,
        candidate_prefix=None,
    ):
        self.tokenizer = tokenizer
        self.source_maxlength = source_maxlength
        self.candidate_maxlength = candidate_maxlength

        self.sep_token = tokenizer.sep_token if tokenizer.sep_token is not None else tokenizer.eos_token
        self.cls_token = tokenizer.cls_token if tokenizer.cls_token is not None else tokenizer.bos_token
        assert self.sep_token is not None, 'sep_token is not found in the tokenizer'
        self.separate_token = self.sep_token
        self.source_prefix = source_prefix if source_prefix is not None else "<source>"
        self.candidate_prefix = candidate_prefix if candidate_prefix is not None else "<candidate>"
        self.model_max_length = min(tokenizer.model_max_length, self.source_maxlength+self.candidate_maxlength+3)


    def __call__(self, batch):
        batch_size = len(batch)
        batch_source = [b['source'] for b in batch]
        batch_candidates = [b['candidates'] for b in batch]
        if 'scores' in batch[0] and batch[0]['scores'] is not None:
            batch_scores = torch.tensor([b['scores'] for b in batch])

        batch_source = get_truncated_text(batch_source, self.tokenizer, self.source_maxlength)
        batch_candidates = [get_truncated_text(c, self.tokenizer, self.candidate_maxlength) for c in batch_candidates]

        source_texts = [[self.separate_token.join([self.source_prefix+s, self.candidate_prefix+c, c]) for c in cands] for s, cands in zip(batch_source, batch_candidates)] # concatenate source and target

        encoded_source_text_ids, encoded_source_text_masks = encode_batch_text(source_texts, self.tokenizer, self.model_max_length) # source


        return {
            'input_ids' : encoded_source_text_ids,
            'attention_mask' : encoded_source_text_masks,
            'scores' : batch_scores,
        }

class DualCollator(object):
    def __init__(
        self,
        source_maxlength,
        tokenizer,
        candidate_maxlength,
        source_prefix=None,
        candidate_prefix=None,
    ):
        self.tokenizer = tokenizer
        self.source_maxlength = source_maxlength
        self.candidate_maxlength = candidate_maxlength

        self.sep_token = tokenizer.sep_token if tokenizer.sep_token is not None else tokenizer.eos_token
        self.cls_token = tokenizer.cls_token if tokenizer.cls_token is not None else tokenizer.bos_token
        assert self.sep_token is not None, 'sep_token is not found in the tokenizer'
        self.cls_token = self.cls_token if self.cls_token is not None else ""
        self.separate_token = self.sep_token + ' ' + self.cls_token # used to separate 2 concatenated texts
        self.target_maxlength = self.candidate_maxlength
        self.source_prefix = source_prefix if source_prefix is not None else "<source>"
        self.candidate_prefix = candidate_prefix if candidate_prefix is not None else "<candidate>"

        tokenizer.add_tokens([self.source_prefix, self.candidate_prefix])


    def __call__(self, batch):
        batch_size = len(batch)
        batch_source = [b['source'] for b in batch]
        batch_target = [b['target'] for b in batch]
        batch_candidates = [b['candidates'] for b in batch]
        if 'scores' in batch[0] and batch[0]['scores'] is not None:
            batch_scores = torch.tensor([b['scores'] for b in batch])
        else:
            batch_scores = None
        

        # add prefix
        batch_source = [self.source_prefix + s for s in batch_source]
        batch_candidates = [[self.candidate_prefix + c for c in cands] for cands in batch_candidates]
        batch_target = [self.candidate_prefix + t for t in batch_target]

        # tokenize
        encoded_source_ids, encoded_source_masks = encode_texts(batch_source, self.tokenizer, self.source_maxlength) # source
        encoded_target_ids, encoded_target_masks = encode_texts(batch_target, self.tokenizer, self.candidate_maxlength) # target
        encoded_candidate_ids, encoded_candidate_masks = encode_batch_text(batch_candidates, self.tokenizer, self.candidate_maxlength) # candidates

        return {
            'source_ids' : encoded_source_ids,
            'source_attention_mask' : encoded_source_masks,
            'target_ids' : encoded_target_ids,
            'target_attention_mask' : encoded_target_masks,
            "candidate_ids" : encoded_candidate_ids,
            "candidate_attention_mask" : encoded_candidate_masks,
            'scores' : batch_scores,
        }

class CrossCompareCollator(object):
    def __init__(
        self,
        source_maxlength,
        tokenizer,
        candidate_maxlength,
        source_prefix=None,
        candidate1_prefix=None,
        candidate2_prefix=None,
    ):
        self.tokenizer = tokenizer
        self.source_maxlength = source_maxlength
        self.candidate_maxlength = candidate_maxlength

        self.sep_token = tokenizer.sep_token if tokenizer.sep_token is not None else tokenizer.eos_token
        self.cls_token = tokenizer.cls_token if tokenizer.cls_token is not None else tokenizer.bos_token
        assert self.sep_token is not None, 'sep_token is not found in the tokenizer'
        self.separate_token = self.sep_token
        self.target_maxlength = self.candidate_maxlength
        self.source_prefix = source_prefix if source_prefix is not None else "<|source|>"
        self.candidate1_prefix = candidate1_prefix if candidate1_prefix is not None else "<|candidate1|>"
        self.candidate2_prefix = candidate2_prefix if candidate2_prefix is not None else "<|candidate2|>"
        self.candidate_prefix = "<|candidate|>"
        self.max_length = min(self.tokenizer.model_max_length, self.source_maxlength + 2 * self.candidate_maxlength + 6)

        # add prefix
        tokenizer.add_tokens([self.source_prefix, self.candidate1_prefix, self.candidate2_prefix, self.candidate_prefix]) # debug
        tokenizer.source_prefix = self.source_prefix
        tokenizer.candidate1_prefix = self.candidate1_prefix
        tokenizer.candidate2_prefix = self.candidate2_prefix
        tokenizer.candidate_prefix = self.candidate_prefix
        tokenizer.source_prefix_id = tokenizer.convert_tokens_to_ids(self.source_prefix)
        tokenizer.cand1_prefix_id = tokenizer.convert_tokens_to_ids(self.candidate1_prefix)
        tokenizer.cand2_prefix_id = tokenizer.convert_tokens_to_ids(self.candidate2_prefix)
        tokenizer.cand_prefix_id = tokenizer.convert_tokens_to_ids(self.candidate_prefix)


    def __call__(self, batch):
        batch_source = [self.source_prefix+b['source'] for b in batch]
        batch_candidates = [[self.candidate_prefix+cand for cand in b['candidates']] for b in batch]
        # substitute special token into space
        batch_source = [s.replace(self.sep_token, ' ') for s in batch_source]
        batch_candidates = [[cand.replace(self.sep_token, ' ') for cand in cands] for cands in batch_candidates]
        if 'scores' in batch[0] and batch[0]['scores'] is not None:
            scores = torch.tensor([b['scores'] for b in batch])
        else:
            scores = None
            
        source_ids, source_masks = encode_texts(batch_source, self.tokenizer, self.source_maxlength)
        candidate_ids, candidate_masks = encode_batch_text(batch_candidates, self.tokenizer, self.candidate_maxlength)

        return {
            "source_ids" : source_ids,
            "source_attention_mask" : source_masks,
            "candidate_ids" : candidate_ids,
            "candidate_attention_mask" : candidate_masks,
            "scores" : scores,
        }