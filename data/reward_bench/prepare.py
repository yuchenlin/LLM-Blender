import fire
import datasets
import json
import random
from typing import List
def get_pair_from_conv_for_single_turn(convAs: List[str], convBs: List[str]):
    """Compare two conversations by takeing USER turns as inputs and ASSISTANT turns as candidates
        Multi-turn conversations comparison is also supportted.
        a conversation format is:
        ```python
        [
            {
                "content": "hello",
                "role": "USER"
            },
            {
                "content": "hi",
                "role": "ASSISTANT"
            },
            ...
        ]
        ```
    Args:
        convAs (List[List[dict]]): List of conversations
        convAs (List[List[dict]]): List of conversations
    """
    for c in convAs + convBs:
        assert len(c) % 2 == 0, "Each conversation must have even number of turns"
        assert all([c[i]['role'].upper() == 'USER' for i in range(0, len(c), 2)]), "Each even turn must be USER"
        assert all([c[i]['role'].upper() == 'ASSISTANT' for i in range(1, len(c), 2)]), "Each odd turn must be ASSISTANT"
    # check conversations correctness
    assert len(convAs) == len(convBs), "Number of conversations must be the same"
    for c_a, c_b in zip(convAs, convBs):
        assert len(c_a) == len(c_b), "Number of turns in each conversation must be the same"
        assert all([c_a[i]['content'] == c_b[i]['content'] for i in range(0, len(c_a), 2)]), "USER turns must be the same"
    
    inputs = [
        convAs[i][0]['content'] for i in range(len(convAs))
    ]
    cand1_texts = [
        convAs[i][1]['content'] for i in range(len(convAs))
    ]
    cand2_texts = [
        convBs[i][1]['content'] for i in range(len(convBs))
    ]
    return inputs, cand1_texts, cand2_texts


def get_pair_from_conv(convAs: List[str], convBs: List[str]):
    """Compare two conversations by takeing USER turns as inputs and ASSISTANT turns as candidates
        Multi-turn conversations comparison is also supportted.
        a conversation format is:
        ```python
        [
            {
                "content": "hello",
                "role": "USER"
            },
            {
                "content": "hi",
                "role": "ASSISTANT"
            },
            ...
        ]
        ```
    Args:
        convAs (List[List[dict]]): List of conversations
        convAs (List[List[dict]]): List of conversations
    """
    for c in convAs + convBs:
        assert len(c) % 2 == 0, "Each conversation must have even number of turns"
        assert all([c[i]['role'].upper() == 'USER' for i in range(0, len(c), 2)]), "Each even turn must be USER"
        assert all([c[i]['role'].upper() == 'ASSISTANT' for i in range(1, len(c), 2)]), "Each odd turn must be ASSISTANT"
    # check conversations correctness
    assert len(convAs) == len(convBs), "Number of conversations must be the same"
    for c_a, c_b in zip(convAs, convBs):
        assert len(c_a) == len(c_b), "Number of turns in each conversation must be the same"
        assert all([c_a[i]['content'] == c_b[i]['content'] for i in range(0, len(c_a), 2)]), "USER turns must be the same"
    
    instructions = ["Finish the following coversation in each i-th turn by filling in <Response i> with your response."] * len(convAs)
    inputs = [
        "\n".join([
            "USER: " + x[i]['content'] +
            f"\nAssistant: <Response {i//2+1}>" for i in range(0, len(x), 2)
        ]) for x in convAs
    ]
    cand1_texts = [
        "\n".join([
            f"<Response {i//2+1}>: " + x[i]['content'] for i in range(1, len(x), 2)
        ]) for x in convAs
    ]
    cand2_texts = [
        "\n".join([
            f"<Response {i//2+1}>: " + x[i]['content'] for i in range(1, len(x), 2)
        ]) for x in convBs
    ]
    inputs = [inst + inp for inst, inp in zip(instructions, inputs)]
    return inputs, cand1_texts, cand2_texts


def main(
    seed=42
):
    random.seed(seed)
    pref_test_set = datasets.load_dataset('allenai/preference-test-sets')
    pref_items = []
    for subset in pref_test_set:
        sub_dataset = pref_test_set[subset]
        for item in sub_dataset:
            convA = item['prompt'] + [
                {
                    "content": item['chosen'],
                    "role": "assistant"
                }
            ]
            convB = item['prompt'] + [
                {
                    "content": item['rejected'],
                    "role": "assistant"
                }
            ]
            num_turn = len(convA) // 2
            if num_turn > 1:
                inputs, cand1_texts, cand2_texts = get_pair_from_conv([convA], [convB])
                input_text, cand1_text, cand2_text = inputs[0], cand1_texts[0], cand2_texts[0]
            else:
                inputs, cand1_texts, cand2_texts = get_pair_from_conv_for_single_turn([convA], [convB])
                input_text, cand1_text, cand2_text = inputs[0], cand1_texts[0], cand2_texts[0]
            
            pref_items.append({
                "id": f"{subset}_{item['id']}",
                "instruction": "",
                "input": input_text,
                "candidates": [
                    {
                        "text": cand1_text,
                        "model": "unknown",
                        "decoding_method": "unknown",
                        "scores": {
                            "human_preference": 1
                        }
                    },
                    {
                        "text": cand2_text,
                        "model": "unknown",
                        "decoding_method": "unknown",
                        "scores": {
                            "human_preference": 0
                        }
                    }
                ]
            })
            random.shuffle(pref_items[-1]['candidates'])
    with open('pref_test_set.json', 'w') as f:
        json.dump(pref_items, f, indent=4)
    
    random.seed(seed)
    reward_bench_eval_data = datasets.load_dataset('allenai/reward-bench', split='filtered')
    reward_bench_items = []
    for item in reward_bench_eval_data:
        reward_bench_items.append({
            "id": f"reward_bench_{item['subset']}_{item['id']}",
            "instruction": "",
            "input": item['prompt'],
            "candidates": [
                {
                    "text": item['chosen'],
                    "model": item['chosen_model'],
                    "decoding_method": "unknown",
                    "scores": {
                        "human_preference": 1,
                    }
                },
                {
                    "text": item['rejected'],
                    "model": item['rejected_model'],
                    "decoding_method": "unknown",
                    "scores": {
                        "human_preference": 0
                    }
                }
            ]
        })
        random.shuffle(reward_bench_items[-1]['candidates'])
    with open('reward_bench.json', 'w') as f:
        json.dump(reward_bench_items, f, indent=4)

    all_test_items = pref_items + reward_bench_items
    with open('all_test_items.json', 'w') as f:
        json.dump(all_test_items, f, indent=4)

if __name__ == '__main__':
    fire.Fire(main)
