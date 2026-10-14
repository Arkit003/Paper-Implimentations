from typing import Optional
from pathlib import Path
import time
import json

from safetensors.torch import load_file
from sentencepiece import SentencePieceProcessor
import torch
import torch.nn as nn
from tqdm import tqdm

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
            intermediate_size=params.get("intermediate_size"),
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
    
    def _sample_top_p(self, probs:torch.Tensor, p):
        probs_sort , probs_idx =torch.sort(probs,dim=-1,descending=True)
        probs_sum = torch.cumsum(probs_sort,dim=-1)
        mask = probs_sum - probs_sort >p
        probs_sort[mask]=0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(probs_sort,num_samples=1)
        next_token = torch.gather(probs_idx,-1,next_token)
        return next_token
        
    
    def text_completion(self,prompts:list[str], temperature:float=0.6, top_p:float=0.9, max_gen_len:Optional[int]=None, use_prefill:bool=True):
        """output generation -> tokenizing prompt ,prefilling and decoding """
        if max_gen_len is None:
            max_gen_len = self.args.max_seq_len
            
        #coverting each prompt into tokens using our tokenizer
        prompt_tokens=[self.tokenizer.tokenize(prompt,out_type=int,add_eos=False,add_bos=True) for prompt in prompts]
        
        #making sure the batch size is not to large
        batch_size = len(prompt_tokens)
        assert batch_size<=self.args.max_batch_size, "batch size is too large"
        
        max_prompt_len = max(len(prompt) for prompt in prompt_tokens)
        
        #making sure the maxing prompt_len is not larger than our seq_len
        assert max_prompt_len<=self.args.max_seq_len, "prompt size is too large"
        
        #how many tokens we need to generate
        total_len = min(self.args.max_seq_len, max_gen_len + max_prompt_len)
        
        #creating the list that will contain the generated tokens as well as prompt
        device = self.args.device
        pad_id = self.tokenizer.pad_id()
        tokens=torch.full((batch_size,total_len),pad_id,dtype=torch.long, device = device)
        
        for k,t in enumerate(prompt_tokens):
            #filling the prompt tokens
            tokens[k,:len(t)]=torch.tensor(t,dtype=torch.long,device=device)
            
        eos_reached = torch.zeros(batch_size, dtype=torch.bool, device=device)
        prompt_token_mask = tokens!=pad_id #true if the token is a prompt token,False otherwise
        
        #adding prefill generation before decoding if the user wants
        prompt_lens=torch.tensor([len(t) for t in prompt_tokens],device=device)
        positions = torch.arange(total_len,device=device) #(1,total_len)
        attention_mask = positions<prompt_lens.unsqueeze(1)#(B,total_len)
        
        if use_prefill:
            with torch.no_grad():
                logits = self.model.forward(
                    tokens[:, :max_prompt_len].clamp(min=0),   # clamp: pad slots must be embeddable
                    0,
                    valid_len=max_prompt_len,
                    attention_mask=attention_mask,
                )
            last_logits = logits[:, -1, :]
        else:
            last_logits = None

        start = max_prompt_len if use_prefill else 1
        
        for curr_pos in tqdm(range(start,total_len),desc = 'generating tokens'):
            if last_logits is None:
                with torch.no_grad():
                    logits= self.model.forward(tokens[:,curr_pos-1:curr_pos],curr_pos-1,attention_mask=attention_mask)
                last_logits=logits[:,-1,:]
            
            #sampling
            if temperature>0:
                #the temperature is applied before the softmax
                probs  = torch.softmax(last_logits/temperature,dim= -1)
                next_token = self._sample_top_p(probs,top_p)
            else:
                #select greddly
                next_token= torch.argmax(last_logits,dim=-1)

            
            
            next_token = next_token.reshape(-1)
            #only replace the token if its a padding token
            next_token  = torch.where(prompt_token_mask[:,curr_pos],tokens[:,curr_pos],next_token)
            tokens[:,curr_pos]= next_token
            #breaking when finding the eos token instead of pad token
            eos_reached |= (~prompt_token_mask[:,curr_pos])& (next_token == self.tokenizer.eos_id())
            if all(eos_reached):
                break
            
            last_logits = None
            
        out_tokens=[]
        out_text=[]
        for prompt_index, current_prompt_tokens in enumerate(tokens.tolist()):
            #cutting out the eos token if found 
            if self.tokenizer.eos_id() in current_prompt_tokens:
                eos_idx = current_prompt_tokens.index(self.tokenizer.eos_id())
                current_prompt_tokens = current_prompt_tokens[:eos_idx]
            #drop unwritten pad slots (pad_id is -1) before decoding
            current_prompt_tokens = [t for t in current_prompt_tokens if t >= 0]
            out_tokens.append(current_prompt_tokens)
            current_prompt_text = self.tokenizer.decode(current_prompt_tokens)
            out_text.append(current_prompt_text)
        return (out_tokens,out_text)
            
            

    
if __name__=="__main__":
    torch.manual_seed(0)
    
    allow_cuda= False
    device='cuda' if torch.cuda.is_available() and allow_cuda else "cpu"
    
    prompts = [
        "Simply put, the theory of relativity states that ",
        "If Google was an Italian company founded in Milan, it would",
        # Few shot promt
        """Translate English to French:
        
        sea otter => loutre de mer
        peppermint => menthe poivrée
        plush girafe => girafe peluche
        cheese =>""",
        # Zero shot prompt
        """Tell me if the following person is actually Doraemon disguised as human:
        Name: Arkit Kumar
        Decision: 
        """
    ]
    
    model=Llama.build(
        checkpoints_dir='models/llama-2-7b-hf/',
        tokenizer_path='models/llama-2-7b-hf/tokenizer.model',
        load_model=True,
        max_seq_len=1024,
        max_batch_size=len(prompts),
        device=device,
        
    )
    
    print("model loaded succesfully")

    #inferncing 
    
    
    
    out_tokens, out_text = model.text_completion(prompts,max_gen_len=64,use_prefill=True)
    assert (len(out_text))==len(prompts)
    for i in range(len(out_text)):
        print(f'{out_text[i]}')
        print("-"*50)