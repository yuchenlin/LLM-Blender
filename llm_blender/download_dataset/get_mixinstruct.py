# Description: Download datasets GPT4all, Dolly 15k, ITwGPT4, ShareGPT

import json
import os
import random
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from pathlib import Path
dev_num = 5000
test_num = 5000
train_num = 100000
DATA_DIR = Path("../../data")
DATA_DIR.mkdir(exist_ok=True)
gpt4all_file = DATA_DIR / "gpt4all.json"
doll15k_file = DATA_DIR / "dolly_15k.json"
itwgpt4_file = DATA_DIR / "itwgpt4.json"
sharegpt_file = DATA_DIR / "sharegpt.json"
gpt4all_num = 100000
doll15k_num = 15000
itwgpt4_num = 52000
sharegpt_num = 50000
mix_dir = DATA_DIR / "mixinstruct"
overwrite=False # overwrite the downloaded files, not overwrite the mixed datasets
max_input_length = 128
max_output_length = 128
if __name__ == "__main__":

    mix_data = []
    source_nums = {}

    # <============== Download GPT4all data ==============>
    print("# Downloading GPT4all data")
    if not os.path.exists(gpt4all_file) or overwrite:
        DS = load_dataset("nomic-ai/gpt4all_prompt_generations")
        DS_data = []
        for x in tqdm(DS['train'], desc="Processing GPT4all"):
            if x['source'] in ['laion/unified_chip2', 'unified_chip2']:
                x['id'] = x['source'] + '/' + str(source_nums.get(x['source'], 0))
                DS_data.append({
                    'id': x['id'],
                    'instruction': "",
                    'input': x['prompt'],
                    'output': x['response'],
                })
                source_nums[x['source']] = source_nums.get(x['source'], 0) + 1
        with open(gpt4all_file, 'w') as f:
            json.dump(DS_data, f, indent=4, ensure_ascii=False)
    else:
        print("File existing! Loading GPT4all from file")
        with open(gpt4all_file, 'r') as f:
            DS_data = json.load(f)
    print("{} examples in GPT4all".format(len(DS_data)))
    random.seed(42)
    random.shuffle(DS_data)
    mix_data.extend(DS_data[:gpt4all_num])

    # <============== Download Dolly 15k ==============>
    print("# Downloading Dolly 15k")
    if not os.path.exists(doll15k_file) or overwrite:
        DS = load_dataset("HuggingFaceH4/databricks_dolly_15k")
        DS_data = []
        for x in tqdm(DS['train'], desc="Processing Dolly 15k"):
            _id = "dolly_15k/" + x['category']
            DS_data.append({
                'id': _id + '/' + str(source_nums.get(_id, 0)),
                'instruction': x['instruction'],
                'input': x['input'],
                'output': x['output'],
            })
            source_nums[_id] = source_nums.get(_id, 0) + 1

        with open(doll15k_file, 'w') as f:
            json.dump(DS_data, f, indent=4, ensure_ascii=False)
    else:
        print("File existing! Loading Dolly 15k from file")
        with open(doll15k_file, 'r') as f:
            DS_data = json.load(f)
    print("{} examples in Dolly 15k".format(len(DS_data)))
    random.seed(42)
    random.shuffle(DS_data)
    mix_data.extend(DS_data[:doll15k_num])


    # <============== Download ITwGPT4 ==============>
    print("# Downloading ITwGPT4")
    if not os.path.exists(itwgpt4_file) or overwrite:
        DS_data = []
        os.system(f"wget https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/raw/main/data/alpaca_gpt4_data.json -O {itwgpt4_file}")
        with open(itwgpt4_file, 'r') as f:
            DS = json.load(f)
        for x in tqdm(DS, desc="ITwGPT4"):
            DS_data.append({
                'id': "itwgpt4/" + str(source_nums.get("itwgpt4", 0)),
                'instruction': x['instruction'],
                'input': x['input'],
                'output': x['output'],
            })
            source_nums["itwgpt4"] = source_nums.get("itwgpt4", 0) + 1
        with open(itwgpt4_file, 'w') as f:
            json.dump(DS_data, f, indent=4, ensure_ascii=False)
    else:
        print("File existing! Loading ITwGPT4 from file")
        with open(itwgpt4_file, 'r') as f:
            DS_data = json.load(f)
    print("{} examples in ITwGPT4".format(len(DS_data)))
    random.seed(42)
    random.shuffle(DS_data)
    mix_data.extend(DS_data[:itwgpt4_num])

    # <============== Download ShareGPT ==============>
    print("# Downloading ShareGPT")
    if not os.path.exists(sharegpt_file) or overwrite:
        DS_data = []
        cleaned_sharegpt_file = DATA_DIR / "sharegpt_cleaned.json"
        if not os.path.exists(cleaned_sharegpt_file):
            os.system(f"wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json -O {cleaned_sharegpt_file}")
        with open(cleaned_sharegpt_file, 'r') as f:
            DS = json.load(f)
        for x in tqdm(DS, desc="Processing ShareGPT"):
            # Here, experimentally, we only keep the first human input as the prompt
            # and the following gpt outputs as the response
            # Since ShareGPT v3 is split to fit the input length no more than 2048
            # the first item in the conversation might comes from gpt to serve as the context
            # We take that as the instruction in that case.
            conversations = x['conversations']
            if len(conversations) < 2:
                # Skip the conversation with only one item or no item
                continue
            first_item = conversations[0]
            if conversations[0]['from'] == 'human' and conversations[1]['from'] == 'gpt':
                instruction = "" 
                input = conversations[0]['value'] # from 'human'
                output = conversations[1]['value'] # from 'gpt'
            else:
                if  len(conversations) < 3 or \
                    not conversations[0]['from'] in ['gpt', 'system'] or \
                    not conversations[1]['from'] == 'human' or \
                    not conversations[2]['from'] == 'gpt':
                    continue
                instruction = conversations[0]['value'] # from 'gpt' or 'system'
                input = conversations[1]['value'] # from 'human'
                output = conversations[2]['value'] # from 'gpt'
            
            # filtering outputs that not informative
            ban_words = ["i'm sorry", "i'am here", "i'am ready", "sure", "okay", "ok", "yes", "no", "yeah", "nope", "yep", "yup", "no problem", "no worries", "how can i", "of course"]
            if any([x in output.lower() for x in ban_words]):
                continue

            DS_data.append({
                'id': f"sharegpt/{x['id']}",
                'instruction': instruction,
                'input': input,
                'output': output,
            })
            source_nums["sharegpt"] = source_nums.get("sharegpt", 0) + 1
        with open(sharegpt_file, 'w') as f:
            json.dump(DS_data, f, indent=4, ensure_ascii=False)
    else:
        print("File existing! Loading ShareGPT from file")
        with open(sharegpt_file, 'r') as f:
            DS_data = json.load(f)
    print("{} examples in ShareGPT".format(len(DS_data)))
    random.seed(42)
    random.shuffle(DS_data)
    mix_data.extend(DS_data[:sharegpt_num])

    # <============== Mix and filtering ==============>
    print("# Mixing and filtering...")
    tokenizer = AutoTokenizer.from_pretrained("chavinlo/alpaca-native")
    print("Total {} examples after mixing".format(len(mix_data)))

    print("# Removing duplicated examples...")
    dedup_mix_data = list({tuple(sorted(d.items())): d for d in tqdm(mix_data, desc="Deduplicating")}.values())
    print("Total {} examples after deduplication".format(len(dedup_mix_data)))
    
    print("# Removing examples with too short and too long output...")
    output_lengths = [len(tokenizer.encode(x['output'])) for x in tqdm(dedup_mix_data, desc="Tokenizing outputs")]
    dedup_mix_data = [x for x, length in zip(dedup_mix_data, output_lengths) if length > 10 and length < max_output_length]
    print("Total {} examples after removing short output".format(len(dedup_mix_data)))

    print("# Removing examples with too short too long instruction+input...")
    input_lengths = [len(tokenizer.encode(x['instruction']+x['input'])) for x in tqdm(dedup_mix_data, desc="Tokenizing inputs")]
    dedup_mix_data = [x for x, length in zip(dedup_mix_data, input_lengths) if length >= 5 and length < max_input_length]
    print("Total {} examples after removing short input".format(len(dedup_mix_data)))

    # <============== Split ==============>
    print("# Shuffling and splitting...")
    random.seed(42)
    random.shuffle(dedup_mix_data)
    dev_data = dedup_mix_data[:dev_num]
    test_data = dedup_mix_data[dev_num:dev_num+test_num]
    train_data = dedup_mix_data[dev_num+test_num:dev_num+test_num+train_num]
    print("Train: {}, Dev: {}, Test: {}".format(len(train_data), len(dev_data), len(test_data)))
    
    mix_dir.mkdir(exist_ok=True)
    with open(mix_dir / "train_data.json", 'w') as f:
        json.dump(train_data, f, indent=4, ensure_ascii=False)
    with open(mix_dir / "val_data.json", 'w') as f:
        json.dump(dev_data, f, indent=4, ensure_ascii=False)
    with open(mix_dir / "test_data.json", 'w') as f:
        json.dump(test_data, f, indent=4, ensure_ascii=False)
    print("Done!")

    # <============== Dataset Statistics ==============>
    print("# Datapoint source statistics:")
    data_sources = {}
    for x in dedup_mix_data:
        data_sources[x['id'].split('/')[0]] = data_sources.get(x['id'].split('/')[0], 0) + 1
    for k, v in data_sources.items():
        print("{}: {}".format(k, v))

    print("# Text length statistics:")
    instruction_lens = [len(tokenizer.encode(x['instruction'])) for x in tqdm(dedup_mix_data, desc="Tokenizing instructions")]
    input_lens = [len(tokenizer.encode(x['input'])) for x in tqdm(dedup_mix_data, desc="Tokenizing inputs")]
    output_lens = [len(tokenizer.encode(x['output'])) for x in tqdm(dedup_mix_data, desc="Tokenizing outputs")]
    print("Avg. Instruction length: {:.2f}".format(sum(instruction_lens) / len(instruction_lens)))
    print("Avg. Input length: {:.2f}".format(sum(input_lens) / len(input_lens)))
    print("Avg. Output length: {:.2f}".format(sum(output_lens) / len(output_lens)))
    print("Max. Instruction length: {}".format(max(instruction_lens)))
    print("Max. Input length: {}".format(max(input_lens)))
    print("Max. Output length: {}".format(max(output_lens)))
    print("Min. Instruction length: {}".format(min(instruction_lens)))
    print("Min. Input length: {}".format(min(input_lens)))
    print("Min. Output length: {}".format(min(output_lens)))
    
    print("Done!")
