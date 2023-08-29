import logging
import torch
import numpy as np
import copy
from typing import List, Union
from .blender_utils import (
    load_ranker, 
    load_fuser,
    RankerDataset,
    GenFuserDataset,
    get_topk_candidates_from_ranks,
    get_torch_dtype
)
from ..gpt_eval.utils import (
    get_scores_from_cmps,
    get_ranks_from_scores
)
from ..pair_ranker.config import RankerConfig
from ..gen_fuser.config import GenFuserConfig
from .config import BlenderConfig
from tqdm import tqdm


class Blender:
    def __init__(
        self, 
        blender_config:BlenderConfig,
        ranker_config:RankerConfig=None,
        fuser_config:GenFuserConfig=None,
    ):
        self.ranker_config = ranker_config
        self.fuser_config = fuser_config
        self.blender_config = blender_config

        if self.ranker_config is None:
            logging.warning("No ranker config provided, no ranker loaded, please load ranker first through load_ranker()")
        else:
            self.ranker, self.ranker_tokenizer, self.ranker_collator = load_ranker(ranker_config)
            if self.blender_config.device == "cuda" and ranker_config.fp16:
                self.ranker = self.ranker.half()
            else:
                self.ranker = self.ranker.float()
            self.ranker = self.ranker.to(self.blender_config.device)
            self.ranker.eval()
        
        if self.fuser_config is None:
            logging.warning("No fuser config provided, no fuser loaded, please load fuser first through load_fuser()")
        else:
            fuser_config.device = self.blender_config.device
            self.fuser, self.fuser_tokenizer = load_fuser(fuser_config)
            self.fuser.eval()

    def rank(
        self, 
        inputs:List[str], 
        candidates:List[List[str]], 
        instructions:List[str]=None, 
        return_scores:bool=False,
        batch_size:int=8,
        **rank_kwargs
    ):
        """Rank candidates for each input
        Args:
            inputs List[str]: List of input texts
            candidates List[List[str]]: List of list of candidate texts, meaning each input can have multiple candidates
            instructions List[str]: List of instructions. if not None, will be prepended to the corresponding input
            return_scores bool: If True, will return scores instead of ranks
            batch_size int: batch size for ranking
        Returns:
            ranks List[List[int]]: Ranks of candidates for each input. Lower is better. ranks[i][j] is the rank of the j-th candidate for the i-th input
            or 
            scores List[List[float]]: Scores of candidates for each input. Higher is better. scores[i][j] is the score of the j-th candidate for the i-th input
        """
        if self.ranker is None:
            logging.warning("No ranker loaded, please load ranker first through load_ranker()")
            return None
        assert len(inputs) == len(candidates), "Number of inputs and candidates must be the same"
        collate_fn = copy.copy(self.ranker_collator)
        collate_fn.source_maxlength = rank_kwargs.get("source_max_length", None) or self.ranker_config.source_maxlength
        collate_fn.candidate_maxlength = rank_kwargs.get("candidate_max_length", None) or self.ranker_config.candidate_maxlength
        dataset = RankerDataset(inputs, candidates, instructions=instructions)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        scores = []
        with torch.no_grad():
            for batch in tqdm(iter(dataloader), desc="Ranking candidates"):
                batch = {k: v.to(self.blender_config.device) for k, v in batch.items() if v is not None}
                outputs = self.ranker._full_predict(**batch)
                preds = outputs['preds'].detach().cpu().numpy()
                batch_scores = get_scores_from_cmps(preds)
                scores.append(batch_scores)
        scores = np.concatenate(scores, axis=0)
        if return_scores:
            return scores
        else:
            return get_ranks_from_scores(scores)
    
    def compare(self, 
        inputs: List[str], 
        candidates_A: List[str], 
        candidates_B:List[str], 
        instructions:List[str]=None, 
        batch_size:int=4
    ):
        """Compare candidates for each input
        Args:
            inputs: List of input strings
            candidates_A: List of candidate strings
            candidates_B: List of candidate strings
            instructions: List of instruction strings. if not None, will be prepended to the corresponding input
            batch_size: Batch size
        Return:
            comparison_results: List[bool], True if A is better than B, False otherwise
            """
        if self.ranker is None:
            logging.warning("No ranker loaded, please load ranker first through load_ranker()")
            return None
        assert len(candidates_A) == len(candidates_B), "Number of candidates_A and candidates_B must be the same"
        assert len(inputs) == len(candidates_A), "Number of inputs and candidates must be the same"
        candidates = [[a, b] for a, b in zip(candidates_A, candidates_B)]
        scores = self.rank(inputs, candidates, return_scores=True, instructions=instructions, batch_size=batch_size)
        return scores[:, 0] > scores[:, 1]
    
    def fuse(
        self, 
        inputs:List[str], 
        candidates:List[List[str]], 
        instructions:List[str]=None, 
        batch_size:int=4,
        **generate_kwargs
    ):
        """Fuse candidates for each input
        Args:
            inputs List[str]: List of input texts
            candidates List[List[str]]: Candidates to fuse for each input. Normally, these candidates should be the top-ranked candidates by the ranker
            instructions List[str]: List of instructions. if not None, will be prepended to the corresponding input
            generate_kwargs: kwargs for fuser.generate()
        Returns:
            outputs List[str]: Fused outputs for each input
        """
        if self.fuser is None:
            logging.warning("No fuser loaded, please load fuser first through load_fuser()")
            return None
        generate_kwargs = generate_kwargs.copy()
        max_length = generate_kwargs.pop("source_max_length", None) or self.fuser_config.max_length
        candidate_maxlength = generate_kwargs.pop("candidate_max_length", None) or self.fuser_config.candidate_maxlength
        dataset = GenFuserDataset(inputs, candidates, self.fuser_tokenizer,
            instructions=instructions, max_length=max_length, 
            candidate_maxlength=candidate_maxlength)

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        generate_params = {
            "max_new_tokens": candidate_maxlength,
            "num_beams": 4,
            "num_return_sequences": 1,
        }
        if generate_kwargs:
            generate_params.update(generate_kwargs)
            
        generations = []
        for batch in tqdm(iter(dataloader), desc="Fusing candidates"):
            batch = {k: v.to(self.blender_config.device) for k, v in batch.items()}
            keep_column_mask = batch['attention_mask'].ne(0).any(dim=0)
            batch['input_ids'] = batch['input_ids'][:, keep_column_mask]
            batch['attention_mask'] = batch['attention_mask'][:, keep_column_mask]
            output_ids = self.fuser.generate(**batch, **generate_params)
            _generations = self.fuser_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            generations.extend(_generations)
        return generations
    
    def rank_and_fuse(self, inputs:List[str], candidates:List[List[str]], instructions:List[str]=None, return_scores=False, batch_size=4, top_k=3, **generate_kwargs):
        """Rank the candidates using ranker and fuse the top-k candidates with genfuser
        Args:
            inputs List[str]: List of input texts
            candidates List[List[str]]: List of list of candidate texts, meaning each input can have multiple candidates
            instructions List[str]: List of instructions. if not None, will be prepended to the corresponding input
            batch_size int: batch size for ranking
            top_k int: Number of the top-ranked candidates to fuse by the fuser
            generate_kwargs: kwargs for fuser.generate()
        Returns:
            fused_generations List[str]: Fused outputs for each input
            ranks_or_scores List[List[int]]: Ranks or scores of candidates for each input. element[i][j] is the rank or score of the j-th candidate for the i-th input
        """
        ranks_or_scores = self.rank(inputs, candidates, instructions=instructions, batch_size=batch_size, return_scores=return_scores)
        if return_scores:
            # if scores, transform to ranks. That is, from higher is better to lower is better
            topk_candidates = get_topk_candidates_from_ranks(-ranks_or_scores, candidates, top_k=top_k)
        else:
            topk_candidates = get_topk_candidates_from_ranks(ranks_or_scores, candidates, top_k=top_k)
        fused_generations = self.fuse(inputs, topk_candidates, instructions=instructions, batch_size=batch_size, **generate_kwargs)
        return fused_generations, ranks_or_scores

    
