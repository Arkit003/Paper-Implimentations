from typing import Optional
from pathlib import Path
import time
import json

from safetensors.torch import load_file
from sentencepiece import SentencePieceProcessor
import torch
import torch.nn as nn
import tqdm

from model import Transformer,ModelArgs


def convert_hf_state_dict(checkpoint):
    ''' for changing meta style initialization of architecture to hf style'''
    converted = {}
    replacements = (
        ("model.embed_tokens.", "tok_embeddings."),
        (".self_attn.q_proj.", ".attention.wq."),
        (".self_attn.k_proj.", ".attention.wk."),
        (".self_attn.v_proj.", ".attention.wv."),
        (".self_attn.o_proj.", ".attention.wo."),
        (".input_layernorm.", ".attention_norm."),
        (".post_attention_layernorm.", ".ffn_norm."),
        (".mlp.gate_proj.", ".feed_forward.w1."),
        (".mlp.down_proj.", ".feed_forward.w2."),
        (".mlp.up_proj.", ".feed_forward.w3."),
    )
    for name, tensor in checkpoint.items():
        if name.endswith("self_attn.rotary_emb.inv_freq"):
            continue
        if name == "model.norm.weight":
            converted["norm.weight"] = tensor
            continue
        if name == "lm_head.weight":
            converted["output.weight"] = tensor
            continue
        converted_name = name
        for old, new in replacements:
            converted_name = converted_name.replace(old, new)
        converted[converted_name.replace("model.layers.", "layers.")] = tensor
    return converted


class Llama(nn.Module):
    
    def __init__(self,model:Transformer,tokenizer:SentencePieceProcessor,model_args:ModelArgs):
        super().__init__()
        
        self.model=model
        self.tokenizer=tokenizer
        self.args=model_args
        
    @staticmethod
    def build(checkpoints_dir:str, tokenizer_path:str, load_model:bool,max_seq_len:int,max_batch_size:int, device:str):
        prev_time=time.time()
        if load_model:
            # checkpoints = sorted(Path(checkpoints_dir).glob('*.safetensors'))
            checkpoint = {}
            checkpoints = sorted(Path(checkpoints_dir).glob("model-*.safetensors"))
            for path in checkpoints:
                checkpoint.update(load_file(str(path), device="cpu"))
            assert len(checkpoints)>0, "No model files found "
            print(f"Loading checkpoint {checkpoints}")
            print(f'loaded checkpoint in {(time.time()-prev_time):.2f}s')
            prev_time=time.time()
        
        with open(Path(checkpoints_dir)/'config.json','r') as f:
            params=json.loads(f.read())
            
        # model_args:ModelArgs=ModelArgs(
        #     max_seq_len=max_seq_len,
        #     max_batch_size=max_batch_size,
        #     device=device,
        #     **params
                
        # )
        model_args = ModelArgs(
            dim=params["hidden_size"],
            n_layers=params["num_hidden_layers"],
            n_heads=params["num_attention_heads"],
            n_kv_heads=params["num_key_value_heads"],
            vocab_size=params["vocab_size"],
            norm_eps=params["rms_norm_eps"],
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            device=device,
        )
        
        tokenizer=SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
        model_args.vocab_size=tokenizer.vocab_size()
        
        if device=='cuda':
            torch.set_default_dtype(torch.float16)
        else:
            torch.set_default_dtype(torch.bfloat16)
        
        model = Transformer(model_args).to(device)
        
        if load_model:
            checkpoint = convert_hf_state_dict(checkpoint)
            model.load_state_dict(checkpoint,strict=True)
            print(f'load state dict in {(time.time()-prev_time):.2f}s')
            
        return Llama(model,tokenizer,model_args)
    
if __name__=="__main__":
    torch.manual_seed(0)
    
    allow_cuda= False
    device='cuda' if torch.cuda.is_available() and allow_cuda else "cpu"
    
    model=Llama.build(
        checkpoints_dir='models/llama-2-7b-hf/',
        tokenizer_path='models/llama-2-7b-hf/tokenizer.model',
        load_model=True,
        max_seq_len=1024,
        max_batch_size=3,
        device=device,
        
    )
    
    print("model loaded succesfully")
