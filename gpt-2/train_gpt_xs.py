import torch.nn as nn
import torch
from dataclasses import dataclass
import math
import torch.nn.functional as F
import tiktoken
import time

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
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_embd=config.n_embd
        self.n_head=config.n_head
        
        #openai/hf naming it bias instead of trill or mask(mask)
        self.register_buffer("bias",torch.tril(torch.ones(config.block_size,config.block_size))
                             .view(1,1,config.block_size,config.block_size))
                                                
        
    
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
        
        #casual self attention
        # att=(q @ k.transpose(-2,-1))*(1.0/math.sqrt(k.size(-1)))
        # att=att.masked_fill(self.bias[:,:,:T,:T]==0,float("-inf"))
        # att=F.softmax(att,dim=-1)
        # y=att@v
        
        #flash attention
        y=F.scaled_dot_product_attention(q,k,v,is_causal=True)
        
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
        self.c_proj.NANOGPT_SCALE_INIT = 1
        
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
        x=x+self.attn(self.ln_1(x))
        x=x+self.mlp(self.ln_2(x))
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
        
        #weight sharing scheme
        #the output layer could share up the same vectors in the ou
        self.transformer.wte.weight=self.lm_head.weight
        
        #initing params
        self.apply(self._init_weights)
        
    def _init_weights(self,module):
        '''applies the std and mean during model weights initialization so that random weights should get initalized differently'''
        std = 0.02
        if hasattr(module, 'NANOGPT_SCALE_INIT'):# if the module has that attribute name then the std will change.
            std *= (2 * self.config.n_layer) ** -0.5 
        if isinstance(module,nn.Linear):
            torch.nn.init.normal_(module.weight,mean=0.0,std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module,nn.Embedding):
            torch.nn.init.normal_(module.weight,mean=0.0 ,std=0.02)
    
    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
        
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

#we preload the weights from the hf and trying infernce on our architecture

num_return_sequences=5
max_length=30
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)


class DataLoaderLite:
    def __init__(self,B,T,device):
        self.B=B
        self.T=T
        self.device=device
    
        #creating tokens during intializing of our DataLoader Object
        with open("input.txt",'r') as f:
            data=f.read()
    
        self.encoder=tiktoken.get_encoding("gpt2")
        tokens=self.encoder.encode(data)
        self.tokens=torch.tensor(tokens)
        print(f"Loaded {len(self.tokens)} tokens")
        print(f"1 epoch={len(self.tokens)//(B*T)} batches")
        
        #state
        self.current_position=0
        
    def next_batch(self,):
        B,T=self.B,self.T
        #creating buffer 
        buf=self.tokens[self.current_position:self.current_position+B*T+1]
        x=buf[:-1].view(B,T).to(self.device)
        y=buf[1:].view(B,T).to(self.device)
        
        #getting the next set of tokens for the next batch
        self.current_position+=B*T
        if self.current_position >len(self.tokens):
            self.current_position=0
        return x,y    
    
train_loader = DataLoaderLite(B=1,T=1024,device=device) #for 16,1024 our vram was less #use power of 2 for optimal performance

torch.set_float32_matmul_precision('high')#tf-32 bit precision whereever pytorch could use it so 
#so that we could trade off bw speed and lower floating point accuracy it doesn't that much damage though 

# model=GPT2.from_pretrained("gpt2")
model =GPT2(GPTConfig(vocab_size=50304))#modifying ugly nbr 
# model.eval() # not training but evaluating
model.to('cuda')
model=torch.compile(model)
optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4)
for i in range(50):
    t0=time.time()
    x,y= train_loader.next_batch()
    optimizer.zero_grad()
    with torch.autocast(device_type=device,dtype=torch.bfloat16):
        logits,loss=model(x,y)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()#to sync the cpu time and the gpu time 
    t1=time.time()
    dt=(t1-t0)*1000 #time differnce in milisec
    #through put
    token_per_sec=(train_loader.B*train_loader.T)/(t1-t0)
    print(f"step{i} ,loss :{loss.item()}, dt:{dt:.2f}ms, tok/sec:{token_per_sec:.2f}")

print(loss)



import sys
sys.exit(0)

print("not working2")
#prefix tokens

import tiktoken
enc=tiktoken.get_encoding("gpt2")
tokens=enc.encode("Hello,I'm a language model,")
tokens=torch.tensor(tokens,dtype=torch.long)#(8,)
tokens=tokens.unsqueeze(0).repeat(num_return_sequences,1)#(5,8)
x=tokens.to('cuda')

#x=(B,T) where B=5,T=8

torch.manual_seed(42)
torch.cuda.manual_seed(42)

while x.size(1)<max_length: # meaning till 1 dim < 30 tokens or so 
    #forwarding the model to get the logits for the next token
    with torch.no_grad():
        logits,_=model(x)#(B,T,vocab_size)
        #take the logits at the last position
        logits=logits[:,-1,:] #(B,vocab_size)
        #getting the probs using softmax
        probs=F.softmax(logits,-1)
        #doing tok_k for 50 next tokens (used in the hf pipeline deafaultly )
        #top_k in here becomes (5,50),top_k_indicies(5,50)
        topk_probs,topk_indicies=torch.topk(probs,50,dim=-1)
        #selcting tokens from top_k
        ix=torch.multinomial(topk_probs,1)#(B,1) #sampling 1 prob from top_k_probs randomly
        xcol=torch.gather(topk_indicies,-1,ix) #(B,1) # gathering the token from the ix choosen
        # append to the seq
        x=torch.cat((x,xcol),dim=1) #appending the token to our tokens
        
for i in range(num_return_sequences):
    tokens=x[i,:max_length].tolist()
    decoded=enc.decode(tokens)
    print(">",decoded)
        
        
        
