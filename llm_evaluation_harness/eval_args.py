import argparse
import logging
import os
from pathlib import Path

from llm_blender.common.utils import (append_jsonl, empty2None, empty2Noneint,
                                      load_json, load_jsonl, save_jsonl,
                                      seed_everything, str2bool)


def get_args(default: bool = False) -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda", type=str2bool, default=True)

    # data
    parser.add_argument("--data_dir", type=str, default="../../data")
    parser.add_argument("--batch_size", type=int, default=8)
    # parser.add_argument('--dataset', type = empty2None, required=True)
    parser.add_argument("--set", type=str, default="test")
    parser.add_argument("--max_size", type=int, default=None)
    parser.add_argument("--save_freq", type=int, default=10)

    # model
    parser.add_argument("--model", type=str, default="mosaicml/mpt-7b-chat")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float32", "float16", "bfloat16", "int8"],
    )
    parser.add_argument("--cache_dir", type=str, default=None)

    # candidate generation
    parser.add_argument("--inference_bs", type=int, default=2)
    parser.add_argument(
        "--decoding_method",
        type=str,
        default="top_p_sampling",
        choices=[
            "beam_search",
            "diverse_beam_search",
            "top_p_sampling",
            "top_k_sampling",
        ],
    )
    parser.add_argument("--num_return_sequences", type=int, default=1)
    parser.add_argument("--num_beams", type=int, default=2)  # for beam search
    parser.add_argument(
        "--num_beam_groups", type=int, default=2
    )  # for diverse beam search
    parser.add_argument(
        "--diversity_penalty", type=float, default=1.0
    )  # for diverse beam search
    parser.add_argument("--top_p", type=float, default=1.0)  # for top-p sampling
    parser.add_argument("--top_k", type=int, default=50)  # for top-k sampling
    parser.add_argument(
        "--temperature", type=float, default=1.0
    )  # for top-p and top-k sampling
    parser.add_argument("--stemmer", type=str2bool, default=True)

    # generation config
    parser.add_argument("--prompt_max_length", type=int, default=512)
    parser.add_argument("--output_max_length", type=int, default=512)
    parser.add_argument("--length_penalty", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0)

    parser.add_argument("--start_idx", type=empty2Noneint, default=None)
    parser.add_argument("--end_idx", type=empty2Noneint, default=None)

    parser.add_argument("--overwrite", type=str2bool, default=True)

    if default:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()

    # if args.cache_dir is None:
    #     args.cache_dir = (
    #         Path(os.path.abspath(__file__)).parent.parent.parent / "hf_models"
    #     )
    logging.basicConfig(level=logging.INFO)
    logging.info("*" * 50)
    logging.info(args)
    return args
