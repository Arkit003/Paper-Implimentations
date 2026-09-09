import torch.nn as nn
import torch
from dataclasses import dataclass
import math
import torch.nn.functional as F

@dataclass
class GPTConfig:
    block_size:int=1024#seq length #T
    vocab_size:int=50257 #  50,000 bpe merges + 256 bytes tokens +1 special token(eos)
    n_layer:int=12  
    n_head:int =12
    n_embd:int =768 #channel
    
    
"""                                    Token IDs
                                    [B, T]
                                    │
                                    │ Token Embedding
                                    ▼
                                    [B, T, C]
                                    [4, 256, 384]
                                    │
                                    │ + Position Embedding
                                    ▼
                                    [4, 256, 384]
                                    │
                                    ▼
                                    ┌─────────────────────────────┐
                                    │ Transformer Block × 6        │
                                    │                             │
                                    │  LayerNorm                  │
                                    │      ↓                      │
                                    │  Causal Self-Attention      │
                                    │      ↓                      │
                                    │  Residual                   │
                                    │      ↓                      │
                                    │  LayerNorm                  │
                                    │      ↓                      │
                                    │  MLP                        │
                                    │      ↓                      │
                                    │  Residual                   │
                                    └─────────────────────────────┘
                                    │
                                    ▼
                                    [4, 256, 384]
                                    │
                                    │ Final Linear
                                    ▼
                                    [4, 256, 65]
                                    │
                                    ▼
                                    65 vocabulary logits 
"""

    
#we just concatinated the Head and the MultiAttentionHead from our gpt one  
class CasualSelfAttention(nn.Module):
    def __init__(self,config:GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head==0
        #taking the values of key,values and query but in a single batch
        self.c_attn=nn.Linear(config.n_embd,3*config.n_embd)
        #output projection
        self.c_proj=nn.Linear(config.n_embd,config.n_embd)
        self.n_embd=config.n_embd
        self.n_head=config.n_head
        
        #openai/hf naming it bias instead of trill or mask(mask)
        self.register_buffer("bias",torch.tril(torch.ones(config.block_size,config.block_size)))
        
    
    def forward(self,x):
        B,T,C = x.shape# Batch-size,seqence Lenght,n_embd #[4,284,384]
        
        #calculating the value of kqv in a batch 
        #it is equivalent to what we did in gpt1 paper creating diff linear layer for each
        #nh =no.of heads,hs=head_size and c=no_op
        # in GPT(124M) model they used hs=12,nh=64 c=n_embd=hs*nh
        
        qkv=self.c_attn(x)#[4,256,1152]
        q,k,v=qkv.split(self.n_embd,dim=2) #each [384,384] # we are just splitting the |k=384|q=384|v=384|
        # this is also same as the k,q,v from gpt1 paper just computer calculates much faster this way
        k=k.view(B,T,self.n_head,C//self.n_head).transpose(1,2)#(B,nh,T,hs)
        q=q.view(B,T,self.n_head,C//self.n_head).transpose(1,2)#(B,nh,T,hs)
        v=v.view(B,T,self.n_head,C//self.n_head).transpose(1,2)#(B,nh,T,hs)
        
        att=(q @ k.transpose(-2,-1))*(1.0/math.sqrt(k.size(-1)))
        att=att.masked_fill(self.bias[:,:,:T,:T]==0,float("-inf"))
        att=F.softmax(att,dim=-1)
        y=att@v
        #concatination
        y=y.transpose(1,2).contiguous().view(B,T,C)#reassembling all the head output side by side
        # output_projection
        y=self.c_proj(y)
        return y
        
        
        
class MLP(nn.Module):
    def __init__(self, config:GPTConfig):
        super().__init__()
        self.c_fc=nn.Linear(config.n_embd,config.n_embd *4)
        self.gelu=nn.GELU(approximate='tanh')
        self.c_proj=nn.Linear(config.n_embd * 4,config.n_embd)
        
    def forward(self,x):
        x=self.c_fc(x)
        x=self.gelu(x)
        x=self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self,config:GPTConfig):
        super().__init__()
        
        self.ln_1=nn.LayerNorm(config.n_embd)
        self.attn=CasualSelfAttention(config)
        self.ln_2=nn.LayerNorm(config.n_embd)
        self.mlp=MLP(config)
    
    def forward(self,x):
        x=x+self.attn(self.ln1(x))
        x=x+self.mlp(self.ln2(x))
        return x
    
class GPT2(nn.Module):
    def __init__(self, config:GPTConfig):
        super().__init__()
        self.config=config
        self.transformer=nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size,config.n_embd),
            wpe=nn.Embedding(config.block_size,config.n_embd),
            h=nn.ModuleList([Block(config)for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head=nn.Linear(config.n_embd,config.vocab_size,bias=False)
        
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT2(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
model=GPT2.from_pretrained("gpt2")
print("it worked")