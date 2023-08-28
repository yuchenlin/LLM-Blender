import gradio as gr
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
from datasets import load_dataset
from typing import List

MAX_BASE_LLM_NUM = 20
MIN_BASE_LLM_NUM = 3
DESCRIPTIONS = """
"""
MAX_MAX_NEW_TOKENS=1024
DEFAULT_MAX_NEW_TOKENS=256
EXAMPLES_DATASET = load_dataset("llm-blender/mix-instruct", split='validation', streaming=True)
SHUFFLED_EXAMPLES_DATASET = EXAMPLES_DATASET.shuffle(seed=42, buffer_size=1000)
EXAMPLES = []
CANDIDATE_EXAMPLES = {}
for example in SHUFFLED_EXAMPLES_DATASET.take(100):
    EXAMPLES.append([
        example['instruction'],
        example['input'],
    ])
    CANDIDATE_EXAMPLES[example['instruction']+example['input']] = example['candidates']

# Load Blender
import llm_blender
from llm_blender.blender.blender_utils import get_topk_candidates_from_ranks
ranker_config = llm_blender.RankerConfig()
ranker_config.ranker_type = "pairranker"
ranker_config.model_type = "deberta"
ranker_config.model_name = "microsoft/deberta-v3-large" # ranker backbone
ranker_config.load_checkpoint = "../checkpoint-best" # ranker checkpoint <your checkpoint path>
ranker_config.cache_dir = "../hf_models" # hugging face model cache dir
ranker_config.source_maxlength = 128
ranker_config.candidate_maxlength = 128
ranker_config.n_tasks = 1 # number of singal that has been used to train the ranker. This checkpoint is trained using BARTScore only, thus being 1.
fuser_config = llm_blender.GenFuserConfig()
fuser_config.model_name = "llm-blender/gen_fuser_3b" # our pre-trained fuser
fuser_config.cache_dir = "../hf_models"
fuser_config.max_length = 1024
fuser_config.candidate_maxlength = 128
blender_config = llm_blender.BlenderConfig()
blender_config.device = "cuda" # blender ranker and fuser device
blender = llm_blender.Blender(blender_config, ranker_config, fuser_config)

def update_base_llms_num(k, llm_outputs):
    k = int(k)
    return [gr.Dropdown.update(choices=[f"LLM-{i+1}" for i in range(k)], 
        value=f"LLM-1" if k >= 1 else "", visible=True),
        {f"LLM-{i+1}": llm_outputs.get(f"LLM-{i+1}", "") for i in range(k)}]
    

def display_llm_output(llm_outputs, selected_base_llm_name):
    return gr.Textbox.update(value=llm_outputs.get(selected_base_llm_name, ""), 
        label=selected_base_llm_name + " (Click Save to save current content)", 
        placeholder=f"Enter {selected_base_llm_name} output here", show_label=True)

def save_llm_output(selected_base_llm_name, selected_base_llm_output, llm_outputs):
    llm_outputs.update({selected_base_llm_name: selected_base_llm_output})
    return llm_outputs

def get_preprocess_examples(inst, input):
    # get the num_of_base_llms
    candidates = CANDIDATE_EXAMPLES[inst+input]
    num_candiates = len(candidates)
    dummy_text = inst+input
    return inst, input, num_candiates, dummy_text

def update_base_llm_dropdown_along_examples(dummy_text):
    candidates = CANDIDATE_EXAMPLES[dummy_text]
    ex_llm_outputs = {f"LLM-{i+1}": candidates[i]['text'] for i in range(len(candidates))}
    return ex_llm_outputs
    
def check_save_ranker_inputs(inst, input, llm_outputs):
    if not inst and not input:
        raise gr.Error("Please enter instruction or input context")
    
    if not all([x for x in llm_outputs.values()]):
        empty_llm_names = [llm_name for llm_name, llm_output in llm_outputs.items() if not llm_output]
        raise gr.Error("Please enter base LLM outputs for LLMs: {}").format(empty_llm_names)
    return {
        "inst": inst,
        "input": input,
        "candidates": list(llm_outputs.values()),
    }

def check_fuser_inputs(blender_state, top_k_for_fuser, ranks):
    pass

def llms_rank(inst, input, llm_outputs):
    candidates = list(llm_outputs.values())
    
    return blender.rank(instructions=[inst], inputs=[input], candidates=[candidates])[0]

def display_ranks(ranks):
    return ",  ".join([f"LLM-{i+1}: {rank}" for i, rank in enumerate(ranks)])

def llms_fuse(blender_state, top_k_for_fuser, ranks):
    inst = blender_state['inst']
    input = blender_state['input']
    candidates = blender_state['candidates']
    top_k_candidates = get_topk_candidates_from_ranks([ranks], [candidates], top_k=top_k_for_fuser)[0]
    return blender.fuse(instructions=[inst], inputs=[input], candidates=[top_k_candidates])[0]

def display_fuser_output(fuser_output):
    return fuser_output

        
with gr.Blocks(theme='ParityError/Anime') as demo:
    gr.Markdown(DESCRIPTIONS)
    with gr.Row():
        with gr.Column():
            inst_textbox = gr.Textbox(lines=1, label="Instruction", placeholder="Enter instruction here", show_label=True)
            input_textbox = gr.Textbox(lines=4, label="Input Context", placeholder="Enter input context here", show_label=True)
        with gr.Column():
            saved_llm_outputs = gr.State(value={})
            selected_base_llm_name_dropdown = gr.Dropdown(label="Base LLM",
                choices=[f"LLM-{i+1}" for i in range(MIN_BASE_LLM_NUM)], value="LLM-1", show_label=True)
            selected_base_llm_output = gr.Textbox(lines=4, label="LLM-1 (Click Save to save current content)",
                placeholder="Enter LLM-1 output here", show_label=True)
            with gr.Row():
                base_llm_outputs_save_button = gr.Button('Save', variant='primary')
                
                base_llm_outputs_clear_single_button = gr.Button('Clear Single', variant='primary')
                
                base_llm_outputs_clear_all_button = gr.Button('Clear All', variant='primary')
            base_llms_num = gr.Slider(
                    label='Number of base llms',
                    minimum=MIN_BASE_LLM_NUM,
                    maximum=MAX_BASE_LLM_NUM,
                    step=1,
                    value=MIN_BASE_LLM_NUM,
                )
    
    blender_state = gr.State(value={})
    with gr.Tab("Ranking outputs"):
        saved_rank_outputs = gr.State(value=[])
        rank_outputs = gr.Textbox(lines=4, label="Ranking outputs", placeholder="Ranking outputs", show_label=True)
    with gr.Tab("Fusing outputs"):
        saved_fuse_outputs = gr.State(value=[])
        fuser_outputs = gr.Textbox(lines=4, label="Fusing outputs", placeholder="Fusing outputs", show_label=True)
    with gr.Row():
        rank_button = gr.Button('Rank LLM Outputs', variant='primary',
            scale=1, min_width=0)
        fuse_button = gr.Button('Fuse Top-K ranked outputs', variant='primary',
            scale=1, min_width=0)
        clear_button = gr.Button('Clear Blender', variant='primary',
            scale=1, min_width=0)
        
    with gr.Accordion(label='Advanced options', open=False):
        
        top_k_for_fuser = gr.Slider(
            label='Top k for fuser',
            minimum=1,
            maximum=3,
            step=1,
            value=1,
        )
    
    examples_dummy_textbox = gr.Textbox(lines=1, label="", placeholder="", show_label=False, visible=False)     
    batch_examples = gr.Examples(
        examples=EXAMPLES,
        fn=get_preprocess_examples,
        cache_examples=True,
        examples_per_page=5,
        inputs=[inst_textbox, input_textbox],
        outputs=[inst_textbox, input_textbox, base_llms_num, examples_dummy_textbox],
    )
        
    base_llms_num.change(
        fn=update_base_llms_num,
        inputs=[base_llms_num, saved_llm_outputs],
        outputs=[selected_base_llm_name_dropdown, saved_llm_outputs],
    )
    
    examples_dummy_textbox.change(
        fn=update_base_llm_dropdown_along_examples,
        inputs=[examples_dummy_textbox],
        outputs=saved_llm_outputs,
    ).then(
        fn=display_llm_output,
        inputs=[saved_llm_outputs, selected_base_llm_name_dropdown],
        outputs=selected_base_llm_output,
    )
    
    selected_base_llm_name_dropdown.change(
        fn=display_llm_output,
        inputs=[saved_llm_outputs, selected_base_llm_name_dropdown],
        outputs=selected_base_llm_output,
    )
    
    base_llm_outputs_save_button.click(
        fn=save_llm_output,
        inputs=[selected_base_llm_name_dropdown, selected_base_llm_output, saved_llm_outputs],
        outputs=saved_llm_outputs,
    )
    base_llm_outputs_clear_all_button.click(
        fn=lambda: [{}, ""],
        inputs=[],
        outputs=[saved_llm_outputs, selected_base_llm_output],
    )
    base_llm_outputs_clear_single_button.click(
        fn=lambda: "",
        inputs=[],
        outputs=selected_base_llm_output,
    )
        

    rank_button.click(
        fn=check_save_ranker_inputs,
        inputs=[inst_textbox, input_textbox, saved_llm_outputs],
        outputs=blender_state,
    ).success(
        fn=llms_rank,
        inputs=[inst_textbox, input_textbox, saved_llm_outputs],
        outputs=[saved_rank_outputs],
    ).then(
        fn=display_ranks,
        inputs=[saved_rank_outputs],
        outputs=rank_outputs,
    )
    
    fuse_button.click(
        fn=check_fuser_inputs,
        inputs=[blender_state, top_k_for_fuser, saved_rank_outputs],
        outputs=[],
    ).success(
        fn=llms_fuse,
        inputs=[blender_state, top_k_for_fuser, saved_rank_outputs],
        outputs=[saved_fuse_outputs],
    ).then(
        fn=display_fuser_output,
        inputs=[saved_fuse_outputs],
        outputs=fuser_outputs,
    )
    
    clear_button.click(
        fn=lambda: ["", "", {}, []],
        inputs=[],
        outputs=[rank_outputs, fuser_outputs, blender_state, saved_rank_outputs],
    )
    
        
    

demo.queue(max_size=20).launch()