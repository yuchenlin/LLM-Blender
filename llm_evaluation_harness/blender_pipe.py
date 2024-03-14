import argparse
import os

import torch
from prompt_toolkit import PromptSession
from tqdm import tqdm

import llm_blender
from llm_blender.blender.blender_utils import get_topk_candidates_from_ranks
from llm_evaluation_harness.config import supported_model
from llm_evaluation_harness.eval_args import get_args
from llm_evaluation_harness.tsp_pipe import TspPipeline

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


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
    prompt: list[dict[str, str]], args: argparse.Namespace
) -> list[list[str]]:
    tpipe = TspPipeline(supported_model_list=supported_model, args=args)

    total_responses = [[] for _ in range(len(prompt))]

    for model_name in tqdm(supported_model):
        tpipe.clean()
        response = tpipe.chat(model_id=model_name, chat_msg=prompt, args=args)
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
    top_k: int = 3,
) -> list[list[str]]:
    insts = [x["instruction"] for x in prompt]
    inputs = [x["input"] for x in prompt]
    candidates_texts = total_responses
    topk_candidates = get_topk_candidates_from_ranks(
        ranks, candidates_texts, top_k=top_k
    )
    fuse_generations = llm_blender.fuse(
        inputs, topk_candidates, instructions=insts, batch_size=2
    )
    return fuse_generations


def blender_pipe(prompt: list[dict[str, str]], args: argparse.Namespace) -> list[str]:
    llm_blender = init_llm_blender(device=torch.device("cuda" if args.cuda else "cpu"))
    
    total_responses = get_responses_from_supported_model(
        prompt=prompt, args=args
    )

    total_responses = [[cad['candidates'][0]['text'] for cad in res] for res in total_responses]

    total_ranks = get_ranks(
        llm_blender=llm_blender,
        prompt=dict_input,
        total_responses=total_responses,
    )

    total_result = get_topk_candidates_and_fuse(
        llm_blender=llm_blender,
        prompt=dict_input,
        total_responses=total_responses,
        ranks=total_ranks,
        top_k=3,
    )
    return total_result


if __name__ == "__main__":
    args = get_args()
    print("Starting LLM Blender...")
    llm_blender = init_llm_blender(device=torch.device("cuda" if args.cuda else "cpu"))

    input_session = PromptSession()
    while True:
        try:
            user_instruction = input_session.prompt("\n Instruction >> ")
            user_input = input_session.prompt(" Input       >> ")
            if user_input == "":
                print("\nInput cannot be empty\n")
                continue
            dict_input = [{"instruction": user_instruction, "input": user_input}]

            total_responses = get_responses_from_supported_model(
                prompt=dict_input, args=args
            )

            total_responses = [[cad['candidates'][0]['text'] for cad in res] for res in total_responses]

            total_ranks = get_ranks(
                llm_blender=llm_blender,
                prompt=dict_input,
                total_responses=total_responses,
            )

            total_result = get_topk_candidates_and_fuse(
                llm_blender=llm_blender,
                prompt=dict_input,
                total_responses=total_responses,
                ranks=total_ranks,
                top_k=3,
            )

            print(total_result[0])

        except KeyboardInterrupt:
            print("quitting")
            break
