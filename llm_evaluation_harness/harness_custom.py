import copy
import os
import sys
from collections import defaultdict
from importlib.util import find_spec
from typing import List, Literal, Optional, Tuple, Union

import numpy as np
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm

sys.path.append("/home/azureuser/LLM-Blender-harness")
from llm_evaluation_harness.blender_pipe import blender_pipe
from llm_evaluation_harness.eval_args import get_args


@register_model("llm_blender", "llmblender")
class LlmBlender(LM):
    def __init__(
        self,
        batch_size: Optional[Union[int, str]] = 1,
        device: Optional[str] = "cuda",
    ) -> None:
        super().__init__()

        self.batch_size = batch_size

        self.default_args = get_args(default=True)

    def get_batched_requests(self, requests, batch_size: int = 64):
        inp_list = [request[0] for request in [req.args for req in requests]]
        batch_size = int(batch_size)
        num_batches = (len(inp_list) + batch_size - 1) // batch_size
        return [list(sub_arr) for sub_arr in np.array_split(inp_list, num_batches)]

    def loglikelihood(self, requests) -> list[tuple[float, bool]]:
        pass

    def loglikelihood_rolling(self, requests) -> list[tuple[float, bool]]:
        pass

    def generate_until(self, requests) -> list[str]:
        if not requests:
            return []

        requests_str_batch = self.get_batched_requests(
            requests=requests, batch_size=self.batch_size
        )
        print(
            f"Split requests into {len(requests_str_batch)} chunks with batch size {self.batch_size}."
        )

        res = []
        for request_list in tqdm(requests_str_batch, disable=False):
            dict_input = [{"instruction": inp, "input": ""} for inp in request_list]
            result = blender_pipe(dict_input, self.default_args)
            res.extend(result)

        return res
