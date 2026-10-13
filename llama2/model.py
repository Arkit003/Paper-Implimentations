import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from dataclasses import dataclass

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1 # Later set in the build method
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    intermediate_size: Optional[int] = None
    norm_eps: float = 1e-5

    # Needed for KV cache
    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = None
    
def precompute_theta_pos_frequencies(head_dim:int,seq_len:int,device:str,theta:float=10000.0):
    #as written in paper the head_dim should be even
    assert head_dim%2==0,"the head dim should be even"
    #building the theta parameters
    #acc to the formula theta_i= 10000 ^(-2(i-1)/dim) for i=[1,2,3...dim/2]
    #shape(hea_dim/2)#we are taking the head dim as it applies after calculating W_k and W_V
    theta_numerator=torch.arange(0,head_dim,2).float()
    #shape : (head_dim/2)
    theta=1/(theta**(theta_numerator/head_dim)).to(device)
    #constructing the "m" parameter (the postion)
    #shape:(seq_len)
    m=torch.arange(seq_len,device=device)
    #multipliying each theta with the position m using outer product
    #Shape : (seq_len) outer_product* (Head_dim/2) -> (seq_len,head_dim/2)
    freqs=torch.outer(m,theta).float()
    #we can compute  complex number  in the polar form as c=R*exp(i*m*theta) where R=1:
    freqs_complex=torch.polar(torch.ones_like(freqs),freqs)
    return freqs_complex
      
def apply_rotatary_embedding(x:torch.Tensor , freqs_complex:torch.Tensor, device:str):
    ''' it will be applied to only K and V not Q '''
    #grouping two consecutive embedding and converting into complex nbr
    #shape : (B,seq_len,H,head_dim) -> (B,seq_len,h,head_dim/2)
    x_complex=torch.view_as_complex(x.float().reshape(*x.shape[:-1],-1,2))
    
    #adding the two dimension that the freq complex is missing (B,H)
    #shape:(seq_len,head_dim/2) -> (1,seq_len,1,head_dim/2)
    freqs_complex=freqs_complex.unsqueeze(0).unsqueeze(2)
    
    #(B,seq_len,H,head_dim/2) * (1,seq_len,1,head_dim/2) ->(B,seq_len,h,head_dim/2)
    x_rotated= x_complex*freqs_complex
    
    #(B,seq_len,h,head_dim/2) -> (B,seq_len,h,head_dim/2,2)
    x_out = torch.view_as_real(x_rotated)
    
    #flatting the real values
    #(B,seq_len,h,head_dim/2,2) -> (B,seq_len,h,head_dim)
    x_out=x_out.reshape(*x.shape)
    
    return x_out.type_as(x).to(device)

def repeat_kv(x:torch.Tensor,n_rep:int):
    batch_size,seq_len,n_kv_heads,head_dim = x.shape
    if n_rep==1:
        return x
    return (
        #(B,seq_len,n_kv_head,1,head_dim)
        x[:,:,:,None,:]
        .expand(batch_size,seq_len,n_kv_heads,n_rep,head_dim)
        .reshape(batch_size,seq_len,n_kv_heads*n_rep,head_dim)
    )

class RMSNorm(nn.Module):
    def __init__(self,dim:int , eps:float=1e-6):
        super().__init__()
        self.eps=eps
        #gamma
        self.weight = nn.Parameter(torch.ones(dim))
        
    def _norm(self,x:torch.Tensor):
        #(B,seq_len,dim)
        return x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)

    def forward(self,x:torch.Tensor):
        #(dim) * (B,seq_len,dim) -> (B,seq_len,dim)
        return self.weight * self._norm(x.float()).type_as(x)

class SelfAttention(nn.Module):
    def __init__(self,args:ModelArgs):
        super().__init__()
        self.n_kv_heads=args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_head_q = args.n_heads
        
        #the ratio bw kv head and q head
        self.n_rep = self.n_head_q//self.n_kv_heads
        #the head_dim
        self.head_dim = args.dim//args.n_heads
        
        self.wq=nn.Linear(args.dim,self.n_head_q * self.head_dim, bias= False)
        self.wk=nn.Linear(args.dim,self.n_kv_heads*self.head_dim,bias=False)
        self.wv=nn.Linear(args.dim,self.n_kv_heads*self.head_dim,bias=False)
        self.wo=nn.Linear(self.head_dim*args.n_heads, args.dim , bias=False)
        
        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len , self.n_kv_heads,self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads,self.head_dim))
        
    def forward(self, x:torch.Tensor , start_pos:int, freq_complex:torch.Tensor):
        batch_size,seq_len, _ =x.shape #(B,1,dim) seq_len ==1 as we are inferencing 1 token at a time 
        
        #(B,1,H_q * head_dim)
        xq=self.wq(x)
        #here the kv_head*head_dim may not equal to dim as we are using GQA
        #(B,1,H_KV * head_dim)
        xk=self.wk(x)
        xv=self.wv(x)
        
        #(B,1,H_q * head_dim)->(B,1,H_Q,head_dim)
        xq=xq.view(batch_size,seq_len,self.n_head_q,self.head_dim)
        #(B,1,H_KV*head_dim) -> (B,1,H_KV , head_dim)
        xk=xk.view(batch_size,seq_len,self.n_kv_heads,self.head_dim)
        xv=xv.view(batch_size,seq_len,self.n_kv_heads,self.head_dim)
        
        #applying rotary_embedding to xq and xk
        #does not changes the shapes
        xq = apply_rotatary_embedding(xq,freq_complex,device=x.device)
        xk = apply_rotatary_embedding(xk,freq_complex,device=x.device)
        
        #updating the cache with the newer token keys and values
        self.cache_k[:batch_size,start_pos:start_pos+seq_len] = xk
        self.cache_v[:batch_size, start_pos:start_pos+seq_len]= xv
        
        #retriving all the cached key and values till now 
        #(B,Seq_len_KV , H_KV,head_dim) #here seq_len_kv = start_pos+seq_len
        keys = self.cache_k[:batch_size,:start_pos+seq_len]
        values = self.cache_v[:batch_size,:start_pos+seq_len]
        
        #since the kv_head and q_head is unmatched so in the paper:
        #they repeated the same key and values head for 2 query head
        keys=repeat_kv(keys,self.n_rep)        
        values=repeat_kv(values,self.n_rep)
        
        # Attention is computed independently for each head, 
        # so heads should behave like a batch dimension. 
        # torch.matmul treats all leading dimensions as batch dimensions 
        # and multiplies only the last two. After the transpose:
        # scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(head_dim)
        #(B,1,H_Q,head_dim) ->(B,H_Q,1,head_dim)
        xq=xq.transpose(1,2)
        # keys,values:   (B, seq_len_kv, H_Q, head_dim) ->  (B, H_Q,   seq_len_kv, head_dim)
        keys=keys.transpose(1,2)
        values = values.transpose(1,2)
        
        #(B,H_Q,1,head_dim) @ (B,H_Q,seq_len_kv,head_dim) -> (B,H_Q,1,seq_len_kv)
        scores=torch.matmul(xq,keys.transpose(2,3))/math.sqrt(self.head_dim)
        scores=F.softmax(scores.float(),dim=-1).type_as(xq)
        
        #(B,H_Q,1,seq_len_kv) @ (B,H_Q,seq_len_kv,head_dim)->(B,H_Q,1,head_dim)
        out=torch.matmul(scores,values)
        out=(out.transpose(1,2).contiguous().view(batch_size,seq_len,-1))
        return self.wo(out) #(B,1,dim) -> (B,1,dim)

class FeedForward(nn.Module):
    def __init__(self,args:ModelArgs):
        super().__init__()
         
        if args.intermediate_size is not None:
            hidden_dim = args.intermediate_size
        else:
            hidden_dim = int(2 * (4 * args.dim) / 3)
            if args.ffn_dim_multiplier is not None:
                hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
            # Round to the nearest multiple_of when no explicit size is given.
            hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)
        
        self.w1=nn.Linear(args.dim,hidden_dim,bias=False)
        self.w2=nn.Linear(hidden_dim,args.dim,bias=False)
        self.w3 = nn.Linear(args.dim,hidden_dim,bias=False)
        
    def forward(self,x:torch.Tensor):
        swish=F.silu(self.w1(x))
        x_v=self.w3(x)
        x=swish * x_v
        x=self.w2(x)
        return x
        
        
class EncoderBlock(nn.Module):
    def __init__(self, args:ModelArgs):
        super().__init__()
        
        self.eps=args.norm_eps
        self.dim=args.dim
        self.n_heads=args.n_heads
        self.head_dim=self.dim/self.n_heads
        
        #normalization before attention
        self.attention_norm = RMSNorm(args.dim,eps=self.eps)
        
        self.attention=SelfAttention(args)
        
        #normalization before ffn
        self.ffn_norm = RMSNorm(args.dim,eps=self.eps)
        
        self.feed_forward = FeedForward(args)
        
    def forward(self,x:torch.Tensor, start_pos:int , freqs_complex:torch.Tensor):
        #(B,seq_len,dim) + (B,seq_len ,dim) -> (B,seq_len,dim)
        h = x + self.attention.forward(
            self.attention_norm(x), start_pos, freqs_complex
        )
        
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out

class Transformer(nn.Module):
    def __init__(self,args:ModelArgs):
        super().__init__()
        
        assert args.vocab_size!=-1, "vocab size is not set"
        self.args=args
        self.vocab_size=args.vocab_size
        self.n_layers=args.n_layers
        
        self.tok_embeddings=nn.Embedding(self.vocab_size,args.dim)
        
        self.layers=nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))
        
        self.norm =RMSNorm(args.dim,eps=args.norm_eps)
        
        self.output=nn.Linear(args.dim,args.vocab_size,bias =False)
        
        self.freqs_complex=precompute_theta_pos_frequencies(
            args.dim//args.n_heads,
            self.args.max_seq_len*2,
            device =self.args.device           
            )
        
    def forward(self,tokens:torch.Tensor,start_pos:int):
        #(B,seqlen)
        batch_size,seq_len=tokens.shape
        assert seq_len==1,"only one token a time can be procesed" #as we are not training the model
        
        #converting the tokens into embeddings
        # (B,seq_len)->(B,seq_len,dim)
        h=self.tok_embeddings(tokens)
        
        #retrive the pair (m,theta) correspoding to the positions[start_pos,start_pos+seq_len]
        freqs_complex=self.freqs_complex[start_pos:start_pos+seq_len]
        
        #applying to all the encoders layers
        for layer in self.layers:
            h=layer(h,start_pos,freqs_complex)
        h=self.norm(h)
        output=self.output(h).float()
        return output    
        