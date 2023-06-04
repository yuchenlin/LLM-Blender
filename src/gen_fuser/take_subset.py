import json
import random
import re
# input_file = '../../data/fuse_gen/val/top3_deberta-bartscore-test.jsonl'
# output_file = '../../data/fuse_gen/val/top3_deberta-bartscore-test.mini.jsonl'
# subset_size = 1500

# input_file = '../../data/fuse_gen/train/top3_deberta-bartscore.jsonl'
# output_file = '../../data/fuse_gen/train/top3_deberta-bartscore.clean.jsonl'
# subset_size = -1


input_file = '../../data/fuse_gen/val/top3_deberta-bartscore-test.jsonl'
output_file = '../../data/fuse_gen/val/top3_deberta-bartscore-test.clean.jsonl'
subset_size = -1

# Read the input file and load the JSON lines
with open(input_file, 'r') as f:
    lines = f.readlines()

# Randomly select the subset of lines


def remove_repeated_substrings(s):
    # Find substrings longer than one-word which repeat
    # print(s)
    try:
        words = s.split()
        repeating_substrings = []
        for i in range(len(words)):
            for j in range(i+2, len(words)+1):
                substring = " ".join(words[i:j])
                if s.count(substring) > 1 and words[j:j+j-i] == words[i:j]:
                    repeating_substrings.append(substring)

        # Keep only the first occurrence of each repeating substring
        unique_substring = s
        for r in sorted(repeating_substrings, key=len, reverse=True):
            unique_substring = re.sub(r, "", unique_substring, count=s.count(r) - 1)
            if unique_substring.endswith(r):
                break
        
        return unique_substring
    except Exception as e:
        print(e)
        print(s)
        return s 

if subset_size > 0:
    random_subset = random.sample(lines, subset_size)
else:
    random_subset = lines 

# Write the subset to the output file
with open(output_file, 'w') as f:
    for line in random_subset:
        instance = json.loads(line.strip())
        # instance["input"] = remove_repeated_substrings(instance["input"])
        # instance["output"] = remove_repeated_substrings(instance["output"])
        if 'source_models' in instance:
            del instance["source_models"]
        line = json.dumps(instance) + "\n"
        f.write(line)

print(f"A random subset of {subset_size} lines has been created in {output_file}.")
