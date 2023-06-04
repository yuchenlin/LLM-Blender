import random
import os
import numpy as np
import torch
import argparse
import hashlib
import json
import prettytable as pt
import tabulate
from collections import defaultdict
from typing import List, Dict
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def empty2None(x):
    if x == '':
        return None
    elif isinstance(x, str):
        return x
    else:
        raise argparse.ArgumentTypeError('String value expected.')

def empty2Noneint(x):
    if x == '':
        return None
    elif isinstance(x, int):
        return x
    elif isinstance(x, str):
        return int(x)
    else:
        raise argparse.ArgumentTypeError('Integer value expected.')

def empty2zero(x):
    if x == '':
        return 0
    elif isinstance(x, int):
        return x
    elif isinstance(x, str):
        return int(x)
    else:
        raise argparse.ArgumentTypeError('Integer value expected.')



def generate_hash_code(text):
    # Convert the text to bytes and create a hash object
    hash_object = hashlib.sha256(text.encode())

    # Get the hexadecimal representation of the hash code
    hex_code = hash_object.hexdigest()

    # Return the first 16 digits of the hexadecimal code
    return hex_code[:16]

def load_json(path):
    with open(path) as f:
        data = json.load(f)
    return data

def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def load_jsonl(path):
    with open(path) as f:
        data = [json.loads(line) for line in f if line.strip()]
    return data

def save_jsonl(data, path):
    with open(path, "w") as f:
        for line in data:
            json.dump(line, f, ensure_ascii=False)
            f.write("\n")

def append_jsonl(data, path):
    with open(path, "a") as f:
        for line in data:
            json.dump(line, f, ensure_ascii=False)
            f.write("\n")


def tabulate_data_stats(ds_data, sources=None):

    source_count_map = defaultdict(int)
    if sources is not None:
        ds_data = [x for x in ds_data if x["id"].split('/')[0] in sources]
    for item in ds_data:
        source_count_map[item["id"].split('/')[0]] += 1

    metrics = list(ds_data[0]["candidates"][0]["scores"].keys())
    models = sorted(list(set([x["model"] for x in ds_data[0]["candidates"]])))
    headers = ["Models (down) / Metircs (right)"] + metrics # models + ["Best Model", "Oracle", "Oracle - Best Model"]
    model_metric_perf_map = defaultdict(dict)
    oracle_perf_map = {metric: 0 for metric in metrics}
    for metric in metrics:
        for model in models:
            model_metric_perf_map[model][metric] = 0
        for item in ds_data:
            best_pref = 0
            for candidate in item["candidates"]:
                model_metric_perf_map[candidate["model"]][metric] += candidate["scores"][metric]
                if candidate["scores"][metric] > best_pref:
                    best_pref = candidate["scores"][metric]
            oracle_perf_map[metric] += best_pref
        for model in models:
            model_metric_perf_map[model][metric] /= len(ds_data)
        oracle_perf_map[metric] /= len(ds_data)

    # print the table
    table_data = []
    for model in models:
        model_perfs = [model_metric_perf_map[model][metric] for metric in metrics]
        table_data.append([model] + model_perfs)
    best_model_name_row = ["Best Model Name"]
    best_model_perf_row = ["Best Model Metric Perf"]
    gap_row = ["Oracle-Best_Model Gap"]
    for metric in metrics:
        model_perfs = [model_metric_perf_map[model][metric] for model in models]
        max_model_perf = max(model_perfs)
        max_model_idx = model_perfs.index(max_model_perf)
        max_model_name = models[max_model_idx]
        best_model_name_row.append(max_model_name)
        best_model_perf_row.append(max_model_perf)
        gap_row.append(oracle_perf_map[metric]-max_model_perf)
    table_data.append(best_model_name_row)
    table_data.append(best_model_perf_row)
    table_data.append(["Oracle"] + [oracle_perf_map[metric] for metric in metrics])
    table_data.append(gap_row)

    # control the precision
    for row in table_data:
        for i in range(len(row)):
            if isinstance(row[i], float):
                row[i] = round(row[i], 4)
    if sources is not None:
        print("Table for {}:".format(sources))
    else:
        print("Table for all sources")
    if len(source_count_map) < 10:
        print("Source distribution:")
        print(source_count_map)
    maxcolwidths = [max([len(str(x)), 15]) for x in headers]
    print(tabulate.tabulate(table_data, headers=headers, tablefmt="pipe", maxcolwidths=maxcolwidths))

def deduplicate_string(string, min_ngram=2, max_ngram=10, repeat=4):

    result = ""
    
    sub_strings = string.split(" ")
    assert repeat >= 2, "repeat should be larger than 2"
    for i in range(len(sub_strings)):
        stop = False
        for ngram in range(min_ngram, max_ngram):
            current_ngrams = sub_strings[i:i+ngram]
            # at least one alpha in the ngram
            if not any([re.search(r"[a-zA-Z]", ngra) for ngra in current_ngrams]):
                continue
            if len(set([" ".join(sub_strings[i+j*ngram:i+j*ngram+ngram]) for j in range(repeat)])) == 1:
                stop = True
                # keep the first occurrence
                result += " " + " ".join(sub_strings[i:i+ngram])
                break
        if stop:
            break
        else:
            result += " " + sub_strings[i]
    return result.strip()
