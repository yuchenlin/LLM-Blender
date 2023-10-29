# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import random
import json
import numpy as np
from ..common.evaluation import METRIC_WEIGHTS

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, n_candidates=None):
        self.data = data
        self.n_candidates = n_candidates if n_candidates is not None and n_candidates > 0 else None
        self.n_tasks = len(self.data[0]['candidates'][0]['scores']) if 'candidates' in self.data[0] else -1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        target = item['output'] if 'output' in item else None
        source = item['instruction'] + item['input']
        if isinstance(target, list):
            target = target[0]
        if "candidates" in item:
            candidates = [c for c in item['candidates']]
            if self.n_candidates is not None:
                candidates = candidates[:self.n_candidates]
                                        
            candidates_text = [c['text'] for c in candidates]
            candidates_scores = [[float(score) for score in c['scores'].values()] for c in candidates]
        else:
            candidates_text = None
            candidates_scores = None
        return {
            'index' : index,
            'source' : source,
            'target' : target,
            'candidates' : candidates_text,
            'scores' : candidates_scores,
        }

    def get_example(self, index):
        return self.data[index]


def load_data(data_path, args, max_size=None):
    random.seed(args.seed)
    assert data_path, "data_path is not specified"
    print("Loading data from {}".format(data_path))
    if data_path.endswith('.jsonl'):
        with open(data_path) as f:
            data = [json.loads(line) for line in f.readlines()]
    elif data_path.endswith('.json'):
        with open(data_path, 'r') as fin:
            data = json.load(fin)
    else:
        raise ValueError("Unknown data")
    if max_size is not None and max_size > 0:
        data = data[:max_size]
    examples = []

    for item in data:
        candidates = item['candidates']
        if args.candidate_models is not None:
            candidates = [candidate for candidate in candidates if candidate['model'] in args.candidate_models]
        if args.candidate_decoding_methods is not None:
            candidates = [candidate for candidate in candidates if candidate['decoding_method'] in args.candidate_decoding_methods]
        if len(candidates) == 0:
            available_model_methods = set([(candidate['model'], candidate['decoding_method']) for candidate in item["candidates"]])
            raise ValueError("No candidates left after filtering, available models and methods are: \n{}".format(
                "\n".join([str(x) for x in available_model_methods])))
        item['candidates'] = candidates
        for candidate in item['candidates']:
            candidate['scores'] = {
                metric: candidate['scores'][metric] for metric in args.metrics
            }

    for k, example in enumerate(data):
        if not 'id' in example:
            example['id'] = k
        examples.append(example)
        for candidate in example['candidates']:
            candidate['scores'] = {k:float(v) for k,v in list(candidate['scores'].items())}
    examples = check_and_normalize_scores(examples)
    return examples

def check_and_normalize_scores(examples):
    """
        Check the upper bound of the scores and print it
    """
    n_candidates = len(examples[0]['candidates'])
    task_names = list(examples[0]['candidates'][0]['scores'].keys())
    max_scores_per_group = {task:[] for task in task_names}
    scores = {task:[] for task in task_names}
    for example in examples:
        for task in task_names:
            scores[task].extend([c['scores'][task] for c in example['candidates']])
            max_scores_per_group[task].append(max([c['scores'][task] for c in example['candidates']]))
    # print checked scores
    for task in task_names:
        print(f"Selection Upper bound for task '{task}' is {np.mean(max_scores_per_group[task])}")
    candidate_scores = {task:[np.mean([ex['candidates'][i]['scores'][task] for ex in examples]) for i in range(n_candidates)] for task in task_names}
    for task in task_names:
        print(f"Candidate mean scores for task '{task}' are {candidate_scores[task]}")

    # normalize scores if training dataset
    metric_weights = METRIC_WEIGHTS

    for example in examples:
        for candidate in example['candidates']:
            for task in task_names:
                if task in metric_weights:
                    candidate['scores'][task] *= metric_weights[task]
    return examples





