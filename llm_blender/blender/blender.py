import logging
import torch
import numpy as np
import copy
import json
import os
import importlib
import transformers
from typing import List, Union
from pathlib import Path
from .blender_utils import (
    load_ranker, 
    load_other_ranker,
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
from huggingface_hub import snapshot_download
from transformers.utils.hub import TRANSFORMERS_CACHE
from tqdm import tqdm

# detect if vllm is installed
try:
    importlib.import_module("vllm")
    import vllm
    is_vllm_imported = True
except ImportError:
    is_vllm_imported = False


class Blender:
    def __init__(
        self, 
        blender_config:BlenderConfig=None,
        ranker_config:RankerConfig=None,
        fuser_config:GenFuserConfig=None,
    ):
        """Initialize Blender

        Args:
            blender_config (BlenderConfig, optional): 
                Defaults to None.
            ranker_config (RankerConfig, optional): 
                Defaults to None. 
                Load ranker from ranker_config with ranker_config.load_checkpoint
            fuser_config (GenFuserConfig, optional): 
                Defaults to None. 
                Load fuser from fuser_config with fuser_config.load_checkpoint
        """
        self.ranker_config = ranker_config
        self.fuser_config = fuser_config
        self.blender_config = blender_config or BlenderConfig()
        
        if self.ranker_config is None:
            logging.warning("No ranker config provided, no ranker loaded, please load ranker first through load_ranker()")
        else:
            ranker_path = self.ranker_config.load_checkpoint
            self.loadranker(ranker_path, **self.ranker_config.to_dict())
        
        if self.fuser_config is None:
            logging.warning("No fuser config provided, no fuser loaded, please load fuser first through load_fuser()")
        else:
            fuser_path = self.fuser_config.model_name
            self.loadfuser(fuser_path, **self.fuser_config.to_dict())
        
    def loadranker(self, ranker_path:str, device:str=None, **kwargs):
        """Load ranker from a path
            Supported rankers:
                - llm-blender/pair-ranker
                - llm-blender/pair-reward-model
                - llm-blender/PairRM
                - OpenAssistant/reward-model-deberta-v3-large-v2
                - openbmb/UltraRM-13b
                - berkeley-nest/Starling-RM-7B-alpha
                - Local path, e.g. "/path/to/ranker"

        Args:
            ranker_path (str):
                - Huggingface model path, e.g. "llm-blender/pair-ranker"
                - Local path, e.g. "/path/to/ranker"
            device (str): 
                cuda or cpu, or None. If None, will use self.blender_config.device
            kwargs: 
                kwargs for RankerConfig
                
        """
        cache_dir = kwargs.pop("cache_dir", TRANSFORMERS_CACHE)
        cache_dir = Path(cache_dir)
        
        if not os.path.exists(ranker_path):
            if not os.path.exists(cache_dir / ranker_path):
                logging.warning(f"Checkpoint '{ranker_path}' does not exist")
                try:
                    # try hugging face hub
                    logging.warning(f"Try dowloading checkpoint from huggingface hub: {ranker_path}")
                    snapshot_download(ranker_path, local_dir=cache_dir / ranker_path)
                    ranker_path = cache_dir / ranker_path
                    logging.warning(f"Successfully downloaded checkpoint to '{ranker_path}'")
                except Exception as e:
                    # try local path
                    logging.warning(f"Failed to download checkpoint from huggingface hub: {ranker_path}")
                    logging.warning(f"Erorr: {e}")
            else:
                ranker_path = cache_dir / ranker_path
        
        # load ranker config from ranker_path
        ranker_path = Path(ranker_path)
        if os.path.exists(ranker_path / "config.json"):
            with open(ranker_path / "config.json", "r") as f:
                ranker_config_json = json.load(f)
            ranker_config = RankerConfig.from_dict(ranker_config_json)
            ranker_config.load_checkpoint = str(ranker_path)
            ranker_config.cache_dir = cache_dir
            self.ranker_config = ranker_config
        else:
            ranker_config_json = {
                "ranker_type": None,
                "model_type": None,
                "model_name": str(ranker_path),
                "cache_dir": cache_dir,
            }
            ranker_config = RankerConfig.from_dict(ranker_config_json)
            self.ranker_config = ranker_config
        for k, v in kwargs.items():
            setattr(self.ranker_config, k, v)
        if ranker_config.model_name is None:
            ranker_config.model_name = str(ranker_path)
    
        # for other rms    
        if ranker_config.ranker_type not in ["pairranker", "summareranker", "simcls"]:
            # tell from the ranker_path
            if ranker_config.model_name.endswith("OpenAssistant/reward-model-deberta-v3-large-v2"):
                ranker_config.ranker_type = "deberta-rm"
                ranker_config.model_type = "deberta-rm"
            elif ranker_config.model_name.endswith("berkeley-nest/Starling-RM-7B-alpha"):
                ranker_config.ranker_type = "starling-rm"
                ranker_config.model_type = "starling-rm"
            elif ranker_config.model_name.endswith("openbmb/UltraRM-13b"):
                ranker_config.ranker_type = "ultra-rm"
                ranker_config.model_type = "ultra-rm"
            else:
                raise ValueError(f"reward model type {ranker_config.model_name} not supported")
            ranker_config.load_checkpoint = None
            
        self.ranker_config.device = device or self.ranker_config.device or self.blender_config.device
    
        self.ranker, self.ranker_tokenizer, self.ranker_collator = load_ranker(ranker_config)
        device = self.ranker_config.device
        if device in ["cuda", "mps"] and ranker_config.fp16:
            self.ranker = self.ranker.half()
        else:
            self.ranker = self.ranker.float()
        self.ranker = self.ranker.to(device)
        self.ranker.eval()
        print("Successfully loaded ranker from ", ranker_path)
        
    def loadfuser(self, fuser_path:str, device:str=None, **kwargs):
        """Load fuser from a path

        Args:
            fuser_path (str): 
                - Huggingface model path, e.g. "llm-blender/gen-fuser"
                - Local path, e.g. "/path/to/fuser"
            device (str): 
                cuda or cpu or None. If None, will use self.blender_config.device
            kwargs: 
                kwargs for GenFuserConfig
        """
        self.fuser_config = GenFuserConfig()
        self.fuser_config.model_name = fuser_path
        for k, v in kwargs.items():
            setattr(self.fuser_config, k, v)
        self.fuser_config.device = device or self.fuser_config.device or self.blender_config.device
        self.fuser, self.fuser_tokenizer = load_fuser(self.fuser_config)
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
        assert all([len(c) > 0 for c in candidates]), "Each input must have at least one candidate"
        assert all([len(c) == len(candidates[0]) for c in candidates]), "Number of candidates for each input must be the same"
        collate_fn = copy.copy(self.ranker_collator)
        collate_fn.source_maxlength = rank_kwargs.get("source_max_length", None) or self.ranker_config.source_maxlength
        collate_fn.candidate_maxlength = rank_kwargs.get("candidate_max_length", None) or self.ranker_config.candidate_maxlength
        dataset = RankerDataset(inputs, candidates, instructions=instructions)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        scores = []
        with torch.no_grad():
            for batch in tqdm(iter(dataloader), desc="Ranking candidates", disable=not self.blender_config.use_tqdm):
                batch = {k: v.to(self.ranker_config.device) for k, v in batch.items() if v is not None}
                if self.ranker_config.ranker_type == "pairranker":
                    outputs = self.ranker._full_predict(**batch)
                    preds = outputs['logits'].detach().cpu().numpy()
                    batch_scores = get_scores_from_cmps(preds)
                elif self.ranker_config.ranker_type in ["summareranker", "simcls"]:
                    outputs = self.ranker(**batch)
                    batch_scores = outputs['logits'].detach().cpu().numpy()
                elif self.ranker_config.ranker_type in ["deberta-rm"]:
                    outputs = self.ranker(**batch)
                    batch_scores = outputs.logits.detach().cpu().numpy()
                    batch_scores = batch_scores.squeeze(-1).reshape(-1, len(candidates[0]))
                else:
                    outputs = self.ranker(**batch) # outputs is a list of scores
                    batch_scores = outputs.detach().cpu().numpy()
                    batch_scores = batch_scores.reshape(-1, len(candidates[0]))
                scores.append(batch_scores)
        scores = np.concatenate(scores, axis=0)
        if return_scores:
            return scores
        else:
            return get_ranks_from_scores(scores)
    
    def compare_conversations(
        self,
        conversations_a:List[List[dict]],
        conversations_b:List[List[dict]],
        batch_size:int=4,
        return_logits:bool=False,
        mode:str="[A,B]+[B,A]"
    ):
        """Compare two conversations by takeing USER turns as inputs and ASSISTANT turns as candidates
            Multi-turn conversations comparison is also supportted.
            a conversation format is:
            ```python
            [
                {
                    "content": "hello",
                    "role": "USER"
                },
                {
                    "content": "hi",
                    "role": "ASSISTANT"
                },
                ...
            ]
            ```
        Args:
            conversations_a (List[List[dict]]): List of conversations
            conversations_b (List[List[dict]]): List of conversations
            batch_size (int, optional): batch size for ranking. Defaults to 4.
            return_logits (bool, optional): If True, will return logits instead of comparison results as bool. Defaults to False.
            mode: Control the compare mode, mianly deal with the effects of position bias if the model is pairwise scoring model.
                For typical reward models that do individual scoring, this mode makes no difference.
                - "[A,B]": 
                    concat A (left) and B (right) as the input. 
                - "[B,A]"
                    concat B (left) and A (right) as the input.
                - "[A,B]+[B,A]": 
                    1. concat A (left) and B (right) as the input for the first-time scoring.
                    2. concat B (left) and A (right) as the input for the second-time scoring.
                    3. The comparison result is the average of the two scoring results.
                    The comparison result is always consistent with the order of candidates
                "[A,B]+[B,A]" is recommended for pairwise scoring models.
        """
        # check role correctness
        for c in conversations_a + conversations_b:
            assert len(c) % 2 == 0, "Each conversation must have even number of turns"
            assert all([c[i]['role'] == 'USER' for i in range(0, len(c), 2)]), "Each even turn must be USER"
            assert all([c[i]['role'] == 'ASSISTANT' for i in range(1, len(c), 2)]), "Each odd turn must be ASSISTANT"
        # check conversations correctness
        assert len(conversations_a) == len(conversations_b), "Number of conversations must be the same"
        for c_a, c_b in zip(conversations_a, conversations_b):
            assert len(c_a) == len(c_b), "Number of turns in each conversation must be the same"
            assert all([c_a[i]['content'] == c_b[i]['content'] for i in range(0, len(c_a), 2)]), "USER turns must be the same"
        
        instructions = ["Finish the following coversation in each i-th turn by filling in <Response i> with your response."] * len(conversations_a)
        inputs = [
            "\n".join([
                "USER: " + x[i]['content'] +
                f"\nAssistant: <Response {i//2+1}>" for i in range(0, len(x), 2)
            ]) for x in conversations_a
        ]
        cand1_texts = [
            "\n".join([
                f"<Response {i//2+1}>: " + x[i]['content'] for i in range(1, len(x), 2)
            ]) for x in conversations_a
        ]
        cand2_texts = [
            "\n".join([
                f"<Response {i//2+1}>: " + x[i]['content'] for i in range(1, len(x), 2)
            ]) for x in conversations_b
        ]
        return self.compare(inputs, cand1_texts, cand2_texts, instructions, batch_size=batch_size, return_logits=return_logits, mode=mode)
    
    def get_best_of_n(
        self, 
        inputs:List[str], 
        candidates:List[List[str]], 
        instructions:List[str]=None,
        pairrm_cmp_type:str="bubble",
        return_all:bool=False,
        batch_size:int=8,
    ):
        """Get the best of n candidates for each input using the ranker
        Args:
            inputs List[str]: List of input texts
            candidates List[List[str]]: List of list of candidate texts, meaning each input can have multiple candidates
            instructions List[str]: List of instructions. if not None, will be prepended to the corresponding input
            pairrm_cmp_type str: one of ['bubble', 'full']
                - 'bubble': use a single run of bubble sort to get the best of n for quicker speed. Time complexity: O(n)
                - 'full': use full pairwise comparison matrix to get the best of n. Time complexity: O(n^2)
            return_all bool: 
                If True, will return all candidates instead of the best of n candidates
                The returned candidates will be sorted by the ranker, where the first candidate is the best
            batch_size int: batch size for ranking
        Returns:
            best_candidates
                - List[str]: Best candidates against the ranker for each input
                - List[List[str]]: All candidates against the ranker for each input, when return_all is True
        """
        if all([len(c) == 1 for c in candidates]):
            # no need to rank
            if not return_all:
                best_candidates = [x[0] for x in candidates]
            else:
                best_candidates = candidates
            return best_candidates
        if self.ranker_config.ranker_type == "pairranker" and pairrm_cmp_type == "bubble":
            # use bubble sort single run to get the best of n for quicker speed
            collate_fn = copy.copy(self.ranker_collator)
            dataset = RankerDataset(inputs, candidates, instructions=instructions)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
            best_idxs = []
            rest_idxs = []
            with torch.no_grad():
                for batch in tqdm(iter(dataloader), desc="Ranking candidates", disable=not self.blender_config.use_tqdm):
                    batch = {k: v.to(self.ranker_config.device) for k, v in batch.items() if v is not None}
                    outputs = self.ranker._bubble_predict(**batch)
                    select_process = outputs['select_process'].detach().cpu().numpy()
                    best_idx = select_process[:, 2, -1]
                    rest_idx = np.where(
                        select_process[:, 0, :] == select_process[:, 2, :], 
                        select_process[:, 1, :],
                        select_process[:, 0, :]
                    )
                    rest_idxs.append(rest_idx)
                    best_idxs.append(best_idx)
            best_idxs = np.concatenate(best_idxs, axis=0)
            if not return_all:
                best_candidates = np.array(candidates)[np.arange(len(candidates)), best_idxs].tolist()
            else:
                rest_idxs = np.concatenate(rest_idxs, axis=0)
                all_idxes = np.concatenate([best_idxs.reshape(-1, 1), rest_idxs], axis=1)
                best_candidates = []
                for i in range(len(candidates)):
                    best_candidates.append([candidates[i][x] for x in all_idxes[i]])
        else:
            ranks = self.rank(inputs, candidates, instructions=instructions, batch_size=batch_size)
            if not return_all:
                best_candidates = get_topk_candidates_from_ranks(ranks, candidates, top_k=1)
                best_candidates = [x[0] for x in best_candidates]
            else:
                best_candidates = get_topk_candidates_from_ranks(ranks, candidates, top_k=None)
        return best_candidates
    
    def get_worst_of_n(
        self, 
        inputs:List[str], 
        candidates:List[List[str]], 
        instructions:List[str]=None,
        pairrm_cmp_type:str="bubble",
        return_all:bool=False,
        batch_size:int=8,
    ):
        """Get the worst of n candidates for each input using the ranker
        Args:
            inputs List[str]: List of input texts
            candidates List[List[str]]: List of list of candidate texts, meaning each input can have multiple candidates
            instructions List[str]: List of instructions. if not None, will be prepended to the corresponding input
            pairrm_cmp_type str: one of ['bubble', 'full']
                - 'bubble': use a single run of bubble sort to get the worst of n for quicker speed. Time complexity: O(n)
                - 'full': use full pairwise comparison matrix to get the worst of n. Time complexity: O(n^2)
            return_all bool: 
                If True, will return all candidates instead of the worst of n candidates
                The returned candidates will be sorted by the ranker, where the first candidate is the worst
            batch_size int: batch size for ranking
        Returns:
            worst_candidates
                - List[str]: worst candidates against the ranker for each input
                - List[List[str]]: All candidates against the ranker for each input, when return_all is True
        """
        if all([len(c) == 1 for c in candidates]):
            # no need to rank
            if not return_all:
                worst_candidates = [x[0] for x in candidates]
            else:
                worst_candidates = candidates
            return worst_candidates
        if self.ranker_config.ranker_type == "pairranker" and pairrm_cmp_type == "bubble":
            # use bubble sort single run to get the worst of n for quicker speed
            collate_fn = copy.copy(self.ranker_collator)
            dataset = RankerDataset(inputs, candidates, instructions=instructions)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
            worst_idxs = []
            rest_idxs = []
            with torch.no_grad():
                for batch in tqdm(iter(dataloader), desc="Ranking candidates", disable=not self.blender_config.use_tqdm):
                    batch = {k: v.to(self.ranker_config.device) for k, v in batch.items() if v is not None}
                    outputs = self.ranker._bubble_predict(**batch, best_or_worst="worst")
                    select_process = outputs['select_process'].detach().cpu().numpy()
                    worst_idx = select_process[:, 2, -1]
                    rest_idx = np.where(
                        select_process[:, 0, :] == select_process[:, 2, :], 
                        select_process[:, 1, :],
                        select_process[:, 0, :]
                    )
                    rest_idxs.append(rest_idx)
                    worst_idxs.append(worst_idx)
            worst_idxs = np.concatenate(worst_idxs, axis=0)
            if not return_all:
                worst_candidates = np.array(candidates)[np.arange(len(candidates)), worst_idxs].tolist()
            else:
                rest_idxs = np.concatenate(rest_idxs, axis=0)
                all_idxes = np.concatenate([worst_idxs.reshape(-1, 1), rest_idxs], axis=1)
                worst_candidates = []
                for i in range(len(candidates)):
                    worst_candidates.append([candidates[i][x] for x in all_idxes[i]])
        else:
            ranks = self.rank(inputs, candidates, instructions=instructions, batch_size=batch_size)
            ranks = -ranks
            if not return_all:
                worst_candidates = get_topk_candidates_from_ranks(ranks, candidates, top_k=1)
                worst_candidates = [x[0] for x in worst_candidates]
            else:
                worst_candidates = get_topk_candidates_from_ranks(ranks, candidates, top_k=None)
        return worst_candidates
    
    
    def compare(self, 
        inputs: List[str], 
        candidates_A: List[str], 
        candidates_B:List[str], 
        instructions:List[str]=None, 
        batch_size:int=4,
        return_logits:bool=False,
        mode:str="[A,B]+[B,A]",
    ):
        """Compare candidates for each input
        Args:
            inputs: List of input strings
            candidates_A: List of candidate strings
            candidates_B: List of candidate strings
            instructions: List of instruction strings. if not None, will be prepended to the corresponding input
            batch_size: Batch size
            return_logits: If True, will return logits instead of comparison results as bool
            mode: 
                Control the compare mode, mianly deal with the effects of position bias if the model is pairwise scoring model.
                For typical reward models that do individual scoring, this mode makes no difference.
                - "[A,B]": 
                    concat A (left) and B (right) as the input. 
                - "[B,A]"
                    concat B (left) and A (right) as the input.
                - "[A,B]+[B,A]": 
                    1. concat A (left) and B (right) as the input for the first-time scoring.
                    2. concat B (left) and A (right) as the input for the second-time scoring.
                    3. The comparison result is the average of the two scoring results.
                    The comparison result is always consistent with the order of candidates
                "[A,B]+[B,A]" is recommended for pairwise scoring models.
        Return:
            comparison_results: 
                - List[float], logits as confidence that A is better than B. 
                    >0 means A is better than B, <0 means B is better than A
                - List[bool], True if A is better than B, False otherwise
            """
        if self.ranker is None:
            logging.warning("No ranker loaded, please load ranker first through load_ranker()")
            return None
        assert len(candidates_A) == len(candidates_B), "Number of candidates_A and candidates_B must be the same"
        assert len(inputs) == len(candidates_A), "Number of inputs and candidates must be the same"
        candidates = [[a, b] for a, b in zip(candidates_A, candidates_B)]
        
        if mode in ["[A,B]", "[B,A]"] and self.ranker_config.ranker_type == "pairranker":
            if mode == "[B,A]":
                candidates = [[b, a] for a, b in zip(candidates_A, candidates_B)]
            collate_fn = copy.copy(self.ranker_collator)
            dataset = RankerDataset(inputs, candidates, instructions=instructions)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
            cmp_results = []
            with torch.no_grad():
                for batch in tqdm(iter(dataloader), desc="Ranking candidates", disable=not self.blender_config.use_tqdm):
                    batch = {k: v.to(self.ranker_config.device) for k, v in batch.items() if v is not None}
                    source_ids, source_attention_mask = batch['source_ids'], batch['source_attention_mask']
                    left_cand_ids, left_cand_attention_mask = batch['candidate_ids'][:, 0], batch['candidate_attention_mask'][:, 0]
                    right_cand_ids, right_cand_attention_mask = batch['candidate_ids'][:, 1], batch['candidate_attention_mask'][:, 1]
                    if batch.get('scores', None) is None:
                        left_scores, right_scores = None, None
                    else:
                        left_scores, right_scores = batch['scores'][:, 0], batch['scores'][:, 1]
                    outputs = self.ranker._forward(
                        source_ids, source_attention_mask,
                        left_cand_ids, left_cand_attention_mask,
                        right_cand_ids, right_cand_attention_mask,
                        left_scores, right_scores,
                    )
                    cmp_results.append(outputs['logits'].detach().cpu().numpy())
            cmp_results = np.concatenate(cmp_results, axis=0)
        else:
            # other ranker type, simple rank
            scores = self.rank(inputs, candidates, return_scores=True, instructions=instructions, batch_size=batch_size)
            cmp_results = scores[:, 0] - scores[:, 1]
        if return_logits:
            return cmp_results
        else:
            return cmp_results > 0

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
        candidate_maxlength = generate_kwargs.pop("candidate_max_length", None) or self.fuser_config.candidate_maxlength
        dataset = GenFuserDataset(inputs, candidates, self.fuser_tokenizer,
            instructions=instructions, max_length=self.fuser_config.max_length, 
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
        for batch in tqdm(iter(dataloader), desc="Fusing candidates", disable=not self.blender_config.use_tqdm):
            batch = {k: v.to(self.fuser_config.device) for k, v in batch.items()}
            keep_column_mask = batch['attention_mask'].ne(0).any(dim=0)
            batch['input_ids'] = batch['input_ids'][:, keep_column_mask]
            batch['attention_mask'] = batch['attention_mask'][:, keep_column_mask]
            output_ids = self.fuser.generate(**batch, **generate_params)
            _generations = self.fuser_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            generations.extend(_generations)
        return generations
    
    def n_generate(
        self,
        model, # Union[transformers.PreTrainedModel, vllm.LLM]
        model_tokenizer:transformers.PreTrainedTokenizer,
        inputs:List[str],
        instructions:List[str]=None,
        n:int=5,
        sampling_mode:str="top_p_sampling",
        batch_size:int=4,
        **generate_kwargs:dict,
    ):
        """We will generate n generations for each input,

        Args:
            model: Union[transformers.PreTrainedModel, vllm.LLM]
                Huggingface model that could generate with .generate(**generate_kwargs)
            model_tokenizer: 
                Huggingface tokenizer that could tokenize with .__call__(**generate_kwargs)
            inputs List[str]: 
                List of input texts
            instructions List[str]: 
                List of instructions. if not None, will be prepended to the corresponding input
            n int: 
                the n parameter in best-of-n. That is, how many samples to generate for ranking for each input
            sampling_mode: 
                "top_k_sampling" or "top_p_sampling"
                if None, will use custom sampling strategy by generate_kwargs
            batch_size int: 
                batch size for generation
            generate_kwargs: 
                kwargs for model.generate()
                recommended kwargs:
                    - max_new_tokens: max length of the generation. If not specified, will use model_tokenizer.model_max_length
                    - top_k: if mode is "top_k_sampling", will use this top_k. if not specified, will use 50
                    - top_p: if mode is "top_p_sampling", will use this top_p. if not specified, will use 1.0
                    - temperature: temperature for sampling. if not specified, will use 0.7
                Note that num_return_sequences will be set to n, so you don't need to specify it
                    
        Returns:
            sampled_candidates
                - List[List[str]]: All sampled candidates against the ranker for each input
        """
        assert len(inputs) == len(instructions) if instructions is not None else True, "Number of inputs and instructions must be the same if instructions is not None"
        if sampling_mode == "top_k_sampling":
            generate_kwargs["do_sample"] = True
            generate_kwargs["top_k"] = generate_kwargs.get("top_k", 50)
            generate_kwargs["temperature"] = generate_kwargs.get("temperature", 0.7)
        elif sampling_mode == "top_p_sampling":
            generate_kwargs["do_sample"] = True
            generate_kwargs["top_p"] = generate_kwargs.get("top_p", 1.0)
            generate_kwargs["temperature"] = generate_kwargs.get("temperature", 0.7)
        elif sampling_mode is None:
            # custom sampling_mode by generate_kwargs
            pass
        else:
            raise ValueError(f"Unknown sampling_mode: {sampling_mode}")
        if "max_new_tokens" not in generate_kwargs:
            # limits of the generation is the default max_length of the model if max_new_tokes not specified
            generate_kwargs['max_length'] = generate_kwargs.get("max_length", model_tokenizer.model_max_length)
        generate_kwargs["num_return_sequences"] = n
        generate_kwargs["output_scores"] = True
        generate_kwargs['return_dict_in_generate'] = True
        
        prompts = [x + "\n" + y for x, y in zip(instructions, inputs)] if instructions is not None else inputs
        sampled_candidates: List[List[str]] = [] # sampled generations for each input [bz, n]
        if is_vllm_imported and isinstance(model, vllm.LLM):
            sampling_params = vllm.SamplingParams(
                n=n, max_tokens=generate_kwargs.get("max_tokens", generate_kwargs.get("max_new_tokens", generate_kwargs.get("max_length", model_tokenizer.model_max_length))),
            )
            for k, v in generate_kwargs.items():
                if hasattr(sampling_params, k):
                    print("set {} to {}".format(k, v))
                    setattr(sampling_params, k, v)
            outputs = model.generate(prompts, sampling_params=sampling_params)
            for output in outputs:
                sampled_candidates.append([output.outputs[i].text for i in range(len(output.outputs))])
        else:
            for i in tqdm(range(0, len(prompts), batch_size), desc="Sampling generations"):
                bz_start, bz_end = i, min(i+batch_size, len(inputs))
                
                bz_prompts = prompts[bz_start: bz_end]
                bz_encodings = model_tokenizer(bz_prompts, return_tensors="pt", padding=True, truncation=True)
                bz_encodings = {k: v.to(model.device) for k, v in bz_encodings.items()}
                bz_outputs = model.generate(**bz_encodings, **generate_kwargs)
                bz_output_ids = bz_outputs.sequences
                bz_output_scores = torch.stack(bz_outputs.scores, dim=0)
                if bz_output_ids.shape[1] == bz_encodings['input_ids'].shape[1] + bz_output_scores.shape[0]:
                    # for decoder-only models
                    bz_output_ids = bz_output_ids[:, bz_encodings['input_ids'].shape[1]:]
                # remove inputs part from outputs
                bz_outputs = model_tokenizer.batch_decode(bz_output_ids, skip_special_tokens=True)
                bz_sampled_candidates = [bz_outputs[i: i+n] for i in range(0, len(bz_outputs), n)]
                sampled_candidates.extend(bz_sampled_candidates)
        return sampled_candidates

    def best_of_n_generate(
        self,
        model, # Union[transformers.PreTrainedModel, vllm.LLM]
        model_tokenizer:transformers.PreTrainedTokenizer,
        inputs:List[str],
        instructions:List[str]=None,
        n:int=5,
        sampling_mode:str="top_p_sampling",
        batch_size:int=4,
        pairrm_cmp_type:str="bubble",
        return_all:bool=False,
        **generate_kwargs:dict,
    ):
        """Decoding enhance generate. 
            In this process, we will generate multiple generations for each input,
            Then we will rank these generations and only return the top-k generations,
            thus enhancing the quality of generations.

        Args:
            model: Union[transformers.PreTrainedModel, vllm.LLM]
                Huggingface model that could generate with .generate(**generate_kwargs)
            model_tokenizer: 
                Huggingface tokenizer that could tokenize with .__call__(**generate_kwargs)
            inputs List[str]: 
                List of input texts
            instructions List[str]: 
                List of instructions. if not None, will be prepended to the corresponding input
            n int: 
                the n parameter in best-of-n. That is, how many samples to generate for ranking for each input
            sampling_mode: 
                "top_k_sampling" or "top_p_sampling"
                if None, will use custom sampling strategy by generate_kwargs
            batch_size int: 
                batch size for generation
            pairrm_cmp_type str: one of ['bubble', 'full']
                - 'bubble': use a single run of bubble sort to get the best of n for quicker speed. Time complexity: O(n)
                - 'full': use full pairwise comparison matrix to get the best of n. Time complexity: O(n^2)
            return_all bool: 
                If True, will return all candidates instead of the best of n candidates
                The returned candidates will be sorted by the ranker, where the first candidate is the best
            generate_kwargs: 
                kwargs for model.generate()
                recommended kwargs:
                    - max_new_tokens: max length of the generation. If not specified, will use model_tokenizer.model_max_length
                    - top_k: if mode is "top_k_sampling", will use this top_k. if not specified, will use 50
                    - top_p: if mode is "top_p_sampling", will use this top_p. if not specified, will use 1.0
                    - temperature: temperature for sampling. if not specified, will use 0.7
                Note that num_return_sequences will be set to n, so you don't need to specify it
                    
        Returns:
            best_candidates
                - List[str]: Best candidates against the ranker for each input
                - List[List[str]]: All candidates against the ranker for each input, when return_all is True
        """
        sampled_candidates = self.n_generate(model, model_tokenizer, inputs, 
            instructions=instructions, n=n, sampling_mode=sampling_mode, batch_size=batch_size, **generate_kwargs)
        
        best_of_n_outputs = self.get_best_of_n(inputs, sampled_candidates, 
            instructions=instructions, batch_size=min(batch_size, 32),
            pairrm_cmp_type=pairrm_cmp_type, return_all=return_all)
        return best_of_n_outputs 
    
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

    
