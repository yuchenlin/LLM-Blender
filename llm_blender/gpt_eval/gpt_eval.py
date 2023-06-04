import argparse
import os
import json
import openai
import regex as re
import random
from pathlib import Path
from itertools import combinations
from string import Template
from tqdm import tqdm
from threading import get_ident
from concurrent.futures import ThreadPoolExecutor
from utils import (
    retry_handler, 
    openai_chat_request, 
    deduplicate_string, 
    is_evaluated
)
from functools import partial


TEMPLATE = """\
Instruction: 
${instruction}

Input: 
${input}

Candidate A: 
${candidate1}

Candidate B: 
${candidate2}

Given the instruction and input above, please compare the two candidates.
You only have 4 choices to output:
If you think A is better, please output: 1. A is better
If you think B is better, please output: 2. B is better
If you think both are good enough correctly give the answer, please output: 3. Same good
If you think both are bad and do not follow the instruction, please output: 4. Same bad
Do not output anything else except the 4 choices above.
Output your choice below:
"""



def gpt_cmp_eval(item, append_jsonl_file=None):
    candidates = item['candidates']
    idxs = list(range(len(candidates)))
    if "cmp_results" not in item:
        item['cmp_results'] = {}
    cmp_results = item['cmp_results']
    for idx_A, idx_B in tqdm(
        list(combinations(idxs, 2)),
        desc=f"Thread {get_ident()}",
        leave=False,
        disable=True
    ):
        if random.random() < 0.5:
            candidate_A = candidates[idx_A]
            candidate_B = candidates[idx_B]
        else:
            candidate_A = candidates[idx_B]
            candidate_B = candidates[idx_A]
        pair_id = f"{candidate_A['model']},{candidate_B['model']}"
        _pair_id = f"{candidate_B['model']},{candidate_A['model']}"
        if pair_id in cmp_results or _pair_id in cmp_results:
            continue
        candidate1 = deduplicate_string(candidate_A['text'])
        candidate2 = deduplicate_string(candidate_B['text'])
        prompt = Template(TEMPLATE).substitute(
            instruction=item['instruction'],
            input=item['input'],
            candidate1=candidate1,
            candidate2=candidate2
        )
        openai_args = {
            "prompt": prompt,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "stop": ["\n"]
        }
        if args.model:
            openai_args['model'] = args.model
        if args.engine:
            openai_args['engine'] = args.engine
        

        @retry_handler(retry_limit=3)
        def cmp_request(**kwargs):
            cmp_result = openai_chat_request(**kwargs)[0]
            cmp_choices = ["A is better", "B is better", "Same good", "Same bad"]
            pos = cmp_result.find(".")
            if pos != -1:
                cmp_result = cmp_result[:pos]
            choice_ids = [x for x in ["1", "2", "3", "4"] if x in cmp_result]
            if len(choice_ids) == 0:
                choice_ids = [i for i, x in enumerate(cmp_choices) if x in cmp_result]
            
            assert len(choice_ids) == 1, f"Invalid output: {cmp_result}"
            choice_id = int(choice_ids[0]) - 1
            cmp_result = cmp_choices[choice_id]
            return cmp_result
        
        try:
            cmp_results[pair_id] = cmp_request(**openai_args)
        except Exception as e:
            # raise e
            # if error, return None
            return None
    cmp_results = {k: v for k, v in sorted(cmp_results.items(), key=lambda x: x[0])}
    item['cmp_results'] = cmp_results

    if append_jsonl_file:
        with open(append_jsonl_file, 'a') as f:
            f.write(json.dumps(item) + "\n")
    return item

def main(args):
    with open(args.input_file, 'r') as f:
        data = json.load(f)

    # do slice
    if args.end_idx != None and args.end_idx > 0:
        data = data[:args.end_idx]
    if args.start_idx != None and args.start_idx > 0:
        data = data[args.start_idx:]
    args.tmp_output_file = Path(args.output_file).with_suffix(".tmp.jsonl")
    if args.tmp_output_file.exists():
        print("Found tmp output file, append to existing output file...")
        if Path(args.output_file).exists():
            with open(args.output_file, 'r') as f:
                eval_results = json.load(f)
            eval_results_map = {item['id']: item for item in eval_results}
            with open(args.tmp_output_file, 'r') as f:
                tmp_data = [json.loads(line) for line in f]
            for item in tmp_data:
                if item['id'] not in eval_results_map:
                    eval_results.append(item)
                else:
                    eval_results_map[item['id']]['cmp_results'] = item['cmp_results']            
            with open(args.output_file, 'w') as f:
                json.dump(eval_results, f, indent=4)
        else:
            with open(args.tmp_output_file, 'r') as f:
                tmp_data = [json.loads(line) for line in f]
            with open(args.output_file, 'w') as f:
                json.dump(tmp_data, f, indent=4)
    args.tmp_output_file = str(args.tmp_output_file)

    _gpt_cmp_eval = partial(gpt_cmp_eval, append_jsonl_file=args.tmp_output_file)


    if not Path(args.output_file).exists():
        print("Creating new output file...")
        with open(args.output_file, 'w') as f:
            json.dump(data, f, indent=4)

    print("Checking if all ids have been evaluated...")
    eval_results = json.load(open(args.output_file, 'r'))
    eval_results_map = {item['id']: item for item in eval_results}
    data_map = {item['id']: item for item in data}
    for id in eval_results_map:
        if id in data_map:
            if "cmp_results" in eval_results_map[id]:
                data_map[id]['cmp_results'] = eval_results_map[id]['cmp_results']
    eval_ids = set([item['id'] for item in data if is_evaluated(item)])

    ids = set([item['id'] for item in data])
    not_eval_ids = ids - eval_ids
    if len(not_eval_ids) > 0:
        print(f"Found {len(not_eval_ids)} not evaluated ids, start evaluating...")
        to_eval_data = [item for item in data if item['id'] in not_eval_ids]
        with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
            results = list(
                tqdm(
                    executor.map(_gpt_cmp_eval, to_eval_data), 
                    total=len(to_eval_data),
                    desc="Overall progress"
                )
            )
            results = [x for x in results if x is not None]
        with open(args.output_file, 'w') as f:
            json.dump(data, f, indent=4)
        
        print(f"Append {len(results)} results to {args.output_file}")

    else:
        print(f"All ids have been evaluated, skip evaluating...")

    # remove tmp file if everything is done
    # if os.path.exists(args.tmp_output_file):
    #     print("Removing tmp output file...")
    #     os.remove(args.tmp_output_file)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="input json file")
    parser.add_argument("--output_file", type=str, required=True, help="output json file")
    parser.add_argument("--num_threads", type=int, default=1, help="number of threads to call OpenAI API")

    # OpenAI Configs
    parser.add_argument("--api_key", type=str, default=None, help="OpenAI API key")
    parser.add_argument("--model", type=str, default=None, help="OpenAI API model")
    parser.add_argument("--engine", type=str, default=None, help="OpenAI API engine")
    parser.add_argument("--temperature", type=float, default=0.0, help="OpenAI API temperature")
    parser.add_argument("--max_tokens", type=int, default=2048, help="OpenAI API max tokens")
    parser.add_argument("--start_idx", type=int, default=None, help="start index of the input file")
    parser.add_argument("--end_idx", type=int, default=None, help="end index of the input file")
    args = parser.parse_args()

    if args.api_key is not None:
        openai.api_key = args.api_key

    random.seed(42) 
    main(args)