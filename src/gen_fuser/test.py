
from datasets import load_dataset, load_metric
data_files = {} 
data_files["train"] = "../../data/fuse_gen/train/top3_deberta-bartscore.clean.jsonl"
data_files["validation"] = "../../data/fuse_gen/val/top3_deberta-bartscore-test.mini.jsonl"
# data_files["test"] = None
print(data_files)
raw_datasets = load_dataset('json', data_files=data_files)