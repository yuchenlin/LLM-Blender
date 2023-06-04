from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
# from transformers import LlamaTokenizer, LlamaForCausalLM 
import torch 
import os
import json  

class ModelManager:
    def __init__(self, model_path, model_name):
        self.model_path = model_path
        self.model_name = model_name
    
    def load_model(self):
        # Load model from disk
        pass
    
    def infer_logits(self, input_data):
        # Run model inference to get logits
        pass
    
    def infer_generate(self, input_data):
        # Run model inference to generate output
        pass


class EncDecModelManager(ModelManager):
    def __init__(self, model_path, model_name, cache_dir):
        super().__init__(model_path, model_name)
        self.model = None 
        self.tokenizer = None 
        self.cache_dir = cache_dir
        self.bf16 = True

    def load_model(self):
        print("loading model: ", self.model_name, "from", self.model_path)
        cd = None 
        if self.cache_dir != "none":
            cd = self.cache_dir
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, cache_dir=cd)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path, device_map="auto", torch_dtype=torch.bfloat16, cache_dir=cd).cuda()
        print("model device:", self.model.device)
        if torch.cuda.is_available():
            self.model = self.model.to("cuda:0")
        print("model device:", self.model.device)
        self.model.eval()

    def clean_newlines(self, texts):
        return [t.replace("\n", " </s> " ) for t in texts]
    
    def infer_logits(self, flatten_inputs, flatten_options):
        # Run T5 model inference to get logits
        flatten_inputs = self.clean_newlines(flatten_inputs)
        flatten_options = self.clean_newlines(flatten_options)
        inputs = self.tokenizer(flatten_inputs, padding=True, add_special_tokens=False)
        outputs = self.tokenizer(flatten_options, padding=True, add_special_tokens=False)
        inputs = {k: torch.tensor(v) for k, v in inputs.items()}
        outputs = {k: torch.tensor(v) for k, v in outputs.items()}
        model_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": outputs["input_ids"],
        }
        with torch.no_grad():
            logits = self.model(**model_inputs).logits
        masked_log_probs = outputs["attention_mask"].unsqueeze(-1) * torch.log_softmax(logits.float(), dim=-1)
        seq_token_log_probs = torch.gather(masked_log_probs, -1, outputs["input_ids"].unsqueeze(-1))
        seq_log_prob = seq_token_log_probs.squeeze(dim=-1).sum(dim=-1)
        return seq_log_prob 
    
    def infer_generate(self, input_data, args):
        # Run T5 model inference to generate output
        input_data = self.clean_newlines(input_data)
        inputs = self.tokenizer(input_data, return_tensors="pt", padding=True)
        outputs = self.model.generate(
                    input_ids=inputs['input_ids'].to(self.model.device), 
                    attention_mask=inputs['attention_mask'].to(self.model.device),
                    pad_token_id=self.tokenizer.eos_token_id, 
                    do_sample=False, 
                    num_return_sequences=args.num_outputs,
                    num_beams=max(args.beam_size, args.num_outputs),
                    max_new_tokens=args.max_output_tokens, # for the outputs
                )   
        decoded_outputs = [self.tokenizer.decode(y, skip_special_tokens=True) for y in outputs]
        n = args.num_outputs
        decoded_outputs = [decoded_outputs[j:j+n] for j in range(0, len(decoded_outputs), n)]
        return decoded_outputs



