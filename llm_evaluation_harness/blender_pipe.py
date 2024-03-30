import argparse
import ctypes
import gc
import os

import torch
from prompt_toolkit import PromptSession
from tqdm import tqdm

libc = ctypes.CDLL("libc.so.6")
import llm_blender
from llm_blender.blender.blender_utils import get_topk_candidates_from_ranks
from llm_evaluation_harness.config import supported_model
from llm_evaluation_harness.eval_args import get_args
from llm_evaluation_harness.tsp_pipe import TspPipeline

from llm_evaluation_harness.engine import stop_sequences_criteria

from llm_blender.blender.blender_utils import (
    GenFuserDataset
)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

example = """The following are multiple choice questions (with answers) about machine learning.
Q: To achieve an 0/1 loss estimate that is less than 1 percent of the true 0/1 loss (with probability 95%), according to Hoeffding's inequality the IID test set must have how many examples?
(A) around 10 examples (B) around 100 examples (C) between 100 and 500 examples (D) more than 1000 examples
A:  (D)

Q: A 6-sided die is rolled 15 times and the results are: side 1 comes up 0 times; side 2: 1 time; side 3: 2 times; side 4: 3 times; side 5: 4 times; side 6: 5 times. Based on these results, what is the probability of side 3 coming up when using Add-1 Smoothing?
(A) 2.0/15 (B) 1.0/7 (C) 3.0/16 (D) 1.0/5
A:  (B)

Q: Traditionally, when we have a real-valued input attribute during decision-tree learning we consider a binary split according to whether the attribute is above or below some threshold. Pat suggests that instead we should just have a multiway split with one branch for each of the distinct values of the attribute. From the list below choose the single biggest problem with Pat’s suggestion:
(A) It is too computationally expensive. (B) It would probably result in a decision tree that scores badly on the training set and a testset. (C) It would probably result in a decision tree that scores well on the training set but badly on a testset. (D) It would probably result in a decision tree that scores well on a testset but badly on a training set.
A:  (C)

Q: You are reviewing papers for the World’s Fanciest Machine Learning Conference, and you see submissions with the following claims. Which ones would you consider accepting?
(A) My method achieves a training error lower than all previous methods! (B) My method achieves a test error lower than all previous methods! (Footnote: When regularisation parameter λ is chosen so as to minimise test error.) (C) My method achieves a test error lower than all previous methods! (Footnote: When regularisation parameter λ is chosen so as to minimise cross-validaton error.) (D) My method achieves a cross-validation error lower than all previous methods! (Footnote: When regularisation parameter λ is chosen so as to minimise cross-validaton error.)
A:  (C)

Q: Which image data augmentation is most common for natural images?
(A) random crop and horizontal flip (B) random crop and vertical flip (C) posterization (D) dithering
A:  (A)

Q: Statement 1| Linear regression estimator has the smallest variance among all unbiased estimators. Statement 2| The coefficients α assigned to the classifiers assembled by AdaBoost are always non-negative.
(A) True, True (B) False, False (C) True, False (D) False, True
A: """

def init_llm_blender(device: torch.device) -> llm_blender.Blender:
    blender = llm_blender.Blender()
    # Load Ranker
    blender.loadranker("llm-blender/PairRM")  # load ranker checkpoint
    # blender.loadranker("OpenAssistant/reward-model-deberta-v3-large-v2") # load ranker checkpoint
    # Load Fuser
    blender.loadfuser(
        "llm-blender/gen_fuser_3b"
    )  # load fuser checkpoint if you want to use pre-trained fuser; or you can use ranker only
    return blender


def get_responses_from_supported_model(
    prompt: list[dict[str, str]], untils_list:list[str], args: argparse.Namespace
) -> list[list[str]]:
    tpipe = TspPipeline(supported_model_list=supported_model, args=args)

    total_responses = [[] for _ in range(len(prompt))]

    for model_name in tqdm(supported_model):
        tpipe.clean()
        response = tpipe.chat(model_id=model_name, chat_msg=prompt, untils_list=untils_list, args=args)
        for i, r in enumerate(response):
            total_responses[i].append(r)

    return total_responses


def get_ranks(
    llm_blender: llm_blender.Blender,
    prompt: list[dict[str, str]],
    total_responses: list[list[str]],
) -> list[list[int]]:
    insts = [x["instruction"] for x in prompt]
    inputs = [x["input"] for x in prompt]
    candidates_texts = total_responses
    ranks = llm_blender.rank(
        inputs, candidates_texts, instructions=insts, return_scores=False, batch_size=2
    )
    return ranks


def get_topk_candidates_and_fuse(
    llm_blender: llm_blender.Blender,
    prompt: list[dict[str, str]],
    total_responses: list[list[str]],
    ranks: list[list[int]],
    untils_list: list[str]=[],
    top_k: int = 3,
) -> list[list[str]]:
    insts = [x["instruction"] for x in prompt]
    inputs = [x["input"] for x in prompt]
    candidates_texts = total_responses
    topk_candidates = get_topk_candidates_from_ranks(
        ranks, candidates_texts, top_k=top_k
    )

    generate_kwargs={
        "max_new_tokens":128
    }

    fuse_generations = llm_blender.fuse(
        inputs, topk_candidates, instructions=insts, batch_size=32, stop_sequences=untils_list, **generate_kwargs
    )
    return fuse_generations


def blender_pipe(prompt: list[dict[str, str]], untils_list:list[str], args: argparse.Namespace) -> list[str]:
    untils_list.extend(["(A)", "(B)", "(C)", "(D)", "(E)"])
    untils_list.extend([" (A)", " (B)", " (C)", " (D)", " (E)"])
    untils_list.extend(["\n(A)", "\n(B)", "\n(C)", "\n(D)", "\n(E)"])
    untils_list.extend(["A)", "B)", "C)", "D)", "E)"])

    
    # print(f"{prompt=}")
    total_responses = get_responses_from_supported_model(prompt=prompt, untils_list=untils_list, args=args)

    llm_blender = init_llm_blender(device=torch.device("cuda" if args.cuda else "cpu"))

    total_responses = [
        [cad["candidates"][0]["text"] for cad in res] for res in total_responses
    ]
    # print(f"{total_responses=}")
    total_ranks = get_ranks(
        llm_blender=llm_blender,
        prompt=prompt,
        total_responses=total_responses,
    )

    total_result = get_topk_candidates_and_fuse(
        llm_blender=llm_blender,
        prompt=prompt,
        total_responses=total_responses,
        ranks=total_ranks,
        top_k=3,
        untils_list=untils_list
    )
    # print(f"{total_result=}")

    del llm_blender
    torch.cuda.empty_cache()
    _ = gc.collect()
    libc.malloc_trim(0)
    return total_result


if __name__ == "__main__":
    args = get_args()
    # print("Starting LLM Blender...")
    # llm_blender = init_llm_blender(device=torch.device("cuda" if args.cuda else "cpu"))

    input_session = PromptSession()
    while True:
        try:
            user_instruction = input_session.prompt("\n Instruction >> ")
            user_input = input_session.prompt(" Input       >> ")
            # if user_input == "":
            #     print("\nInput cannot be empty\n")
            #     continue
            dict_input = [{"instruction": example, "input": ""}]

            total_result = blender_pipe(prompt=dict_input, untils_list=[],args=args)

            print(total_result[0])

        except KeyboardInterrupt:
            print("quitting")
            break
