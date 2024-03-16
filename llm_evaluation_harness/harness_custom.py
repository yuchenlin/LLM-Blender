import copy
import os
from collections import defaultdict
from importlib.util import find_spec
from typing import List, Literal, Optional, Tuple, Union

from tqdm import tqdm
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model


import sys
sys.path.append('/root/LLM-Blender-harness')
from llm_evaluation_harness.blender_pipe import blender_pipe
from llm_evaluation_harness.eval_args import get_args

@register_model("llm_blender", "llmblender")
class LlmBlender(LM):
    def __init__(self, 
                 batch_size: Optional[Union[int, str]] = 1,
                 device: Optional[str] = "cuda",) -> None:
        super().__init__()

        self.default_args = get_args(default=True)

    def loglikelihood(self, requests) -> list[tuple[float, bool]]:
        pass


    def loglikelihood_rolling(self, requests) -> list[tuple[float, bool]]:
        pass


    def generate_until(self, requests) -> list[str]:
        if not requests:
            return []

        res = []
        for request in tqdm([req.args for req in requests], disable=False):
            inp = request[0]

            dict_input = [{"instruction": inp, "input": ""}]
            result = blender_pipe(dict_input, self.default_args)
            res.append(result[0])
        
        return res
        