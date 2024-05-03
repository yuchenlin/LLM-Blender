"""
    This file is taken from This file is modified based on:
    https://github.com/Ravoxsg/SummaReranker-ACL-22-/blob/main/src/candidate_generation/engine.py
    We thank the authors for sharing their code.
"""

import gc
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from tqdm import tqdm
from transformers import StoppingCriteria, StoppingCriteriaList


# copt from : https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/models/huggingface.py
class MultiTokenEOSCriteria(transformers.StoppingCriteria):
    """Criteria to stop on the specified multi-token sequence."""

    def __init__(
        self,
        sequence: str,
        tokenizer: transformers.PreTrainedTokenizer,
        initial_decoder_input_length: int,
        batch_size: int,
    ) -> None:
        self.initial_decoder_input_length = initial_decoder_input_length
        self.done_tracker = [False] * batch_size
        self.sequence = sequence
        self.sequence_ids = tokenizer.encode(sequence, add_special_tokens=False)
        # print(sequence, self.sequence_ids)
        # we look back for 2 more tokens than it takes to encode our stop sequence
        # because tokenizers suck, and a model might generate `['\n', '\n']` but our `sequence` is `['\n\n']`
        # and we don't want to mistakenly not stop a generation because our
        # (string) stop sequence was output in a different tokenization

        # NOTE: there is a minor danger that this will end up looking back 2 tokens into the past, into the inputs to the model,
        # and stopping generation immediately as a result. With only 2 extra tokens of lookback, this risk is minimized
        # Additionally, in lookback_ids_batch we should prevent ever looking back into the inputs as described.
        self.sequence_id_len = len(self.sequence_ids) + 2
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        # For efficiency, we compare the last n tokens where n is the number of tokens in the stop_sequence
        lookback_ids_batch = input_ids[:, self.initial_decoder_input_length :]

        lookback_ids_batch = lookback_ids_batch[:, -self.sequence_id_len :]

        lookback_tokens_batch = self.tokenizer.batch_decode(lookback_ids_batch)

        for i, done in enumerate(self.done_tracker):
            if not done:
                self.done_tracker[i] = self.sequence in lookback_tokens_batch[i]
        return False not in self.done_tracker


def stop_sequences_criteria(
    tokenizer: transformers.PreTrainedTokenizer,
    stop_sequences: List[str],
    initial_decoder_input_length: int,
    batch_size: int,
) -> transformers.StoppingCriteriaList:
    return transformers.StoppingCriteriaList(
        [
            *[
                MultiTokenEOSCriteria(
                    sequence, tokenizer, initial_decoder_input_length, batch_size
                )
                for sequence in stop_sequences
            ],
        ]
    )


class StopTokenIdsCriteria(StoppingCriteria):
    """
    This class can be used to stop generation whenever the generated number of tokens exceeds `max_new_tokens`. Keep in
    mind for decoder-only type of transformers, this will **not** include the initial prompted tokens. This is very
    close to `MaxLengthCriteria` but ignores the number of initial tokens.

    Args:
        stop_token_ids (`List[int]`):
    """

    def __init__(self, stop_token_ids: List[int]):
        self.stop_token_ids = stop_token_ids

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        if self.stop_token_ids:
            return all(
                _input_ids[-1] in self.stop_token_ids for _input_ids in input_ids
            )
        return False


def beam_search_step(input_ids, attention_mask, tokenizer, base_model, args, **kwargs):
    kwargs["return_dict_in_generate"] = True
    kwargs["output_scores"] = True
    # if hasattr(args, "stop_token_ids") and args.stop_token_ids:
    #     kwargs['stopping_criteria'] = StoppingCriteriaList([
    #         StopTokenIdsCriteria(args.stop_token_ids),
    #     ])

    if hasattr(args, "extend_stop_str") and args.extend_stop_str:
        kwargs["stopping_criteria"] = stop_sequences_criteria(
            tokenizer, args.extend_stop_str, input_ids.shape[1], input_ids.shape[0]
        )

    # 1 - beam search
    if args.decoding_method == "beam_search":
        outputs = base_model.generate(
            input_ids,
            attention_mask=attention_mask,
            num_beams=args.num_beams,
            num_return_sequences=args.num_return_sequences,
            max_new_tokens=args.output_max_length,
            repetition_penalty=args.repetition_penalty,
            length_penalty=args.length_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            use_cache=True,
            early_stopping=True,
            temperature=args.temperature,
            **kwargs
        )
    # 2 - diverse beam search
    if args.decoding_method == "diverse_beam_search":
        outputs = base_model.generate(
            input_ids,
            attention_mask=attention_mask,
            num_beams=args.num_beams,
            num_beam_groups=args.num_beam_groups,
            num_return_sequences=args.num_return_sequences,
            max_new_tokens=args.output_max_length,
            diversity_penalty=args.diversity_penalty,
            repetition_penalty=args.repetition_penalty,
            length_penalty=args.length_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            use_cache=True,
            early_stopping=True,
            temperature=args.temperature,
            **kwargs
        )
    # 3 - top-p sampling
    if args.decoding_method == "top_p_sampling":
        outputs = base_model.generate(
            input_ids,
            attention_mask=attention_mask,
            num_beams=1,
            do_sample=True,
            top_p=args.top_p,
            num_return_sequences=args.num_return_sequences,
            max_new_tokens=args.output_max_length,
            repetition_penalty=args.repetition_penalty,
            length_penalty=args.length_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            use_cache=True,
            early_stopping=True,
            temperature=args.temperature,
            **kwargs
        )
    # 4 - top-k sampling
    if args.decoding_method == "top_k_sampling":
        outputs = base_model.generate(
            input_ids,
            attention_mask=attention_mask,
            num_beams=1,
            do_sample=True,
            top_k=args.top_k,
            num_return_sequences=args.num_return_sequences,
            max_new_tokens=args.output_max_length,
            repetition_penalty=args.repetition_penalty,
            length_penalty=args.length_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            use_cache=True,
            early_stopping=True,
            temperature=args.temperature,
            **kwargs
        )
    masked_logits = torch.stack(
        outputs.scores, dim=0
    )  # for top-p and top-k sampling, some scores will be masked as -inf. These scores are not processed by softmax and logrithm.
    masked_logits = F.log_softmax(masked_logits, dim=1)
    summary_ids = outputs.sequences
    logprobs = []
    # Different process for decoder-only models and encoder-decoder models
    if summary_ids.shape[1] == input_ids.shape[1] + masked_logits.shape[0]:
        # for decoder-only models
        summary_ids = summary_ids[:, input_ids.shape[1] :]  # remove input_ids
        for i in range(summary_ids.shape[0]):
            logprobs.append([])
            for j in range(summary_ids.shape[1]):  # token_idx
                if summary_ids[i][j] == tokenizer.eos_token_id:
                    break
                logprobs[i].append(masked_logits[j, i, summary_ids[i][j]].item())
    else:
        # for encoder-decoder models
        for i in range(summary_ids.shape[0]):
            logprobs.append([])
            # shift of decoder because of the additional bos_token
            for j in range(summary_ids.shape[1] - 1):  # token_idx
                if summary_ids[i][j + 1] == tokenizer.eos_token_id:
                    break
                logprobs[i].append(masked_logits[j, i, summary_ids[i][j + 1]].item())

    summary_ids_in_list = summary_ids.tolist()
    if hasattr(args, "stop_token_ids") and args.stop_token_ids:
        for i in range(len(summary_ids_in_list)):
            for j in range(len(summary_ids_in_list[i])):
                if summary_ids_in_list[i][j] in args.stop_token_ids:
                    summary_ids_in_list[i] = summary_ids_in_list[i][: j + 1]
                    logprobs[i] = logprobs[i][: j + 1]
                    break

    generated = []
    for i in range(len(summary_ids_in_list)):
        generated.append(
            tokenizer.decode(
                summary_ids_in_list[i],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
        )

    if hasattr(args, "stop_str") and args.stop_str:
        for i in range(len(generated)):
            pos = generated[i].find(args.stop_str)
            if pos != -1:
                generated[i] = generated[i][:pos]
                logprobs[i] = logprobs[i][:pos]

    # aggregate logprobs
    logprobs = [sum(_probs) for _probs in logprobs]
    del summary_ids
    gc.collect()

    batch_generated = []
    batch_logprobs = []
    for i in range(input_ids.shape[0]):
        batch_generated.append(
            generated[
                i * args.num_return_sequences : (i + 1) * args.num_return_sequences
            ]
        )
        batch_logprobs.append(
            logprobs[
                i * args.num_return_sequences : (i + 1) * args.num_return_sequences
            ]
        )
    return {"generated": batch_generated, "logprobs": batch_logprobs}
