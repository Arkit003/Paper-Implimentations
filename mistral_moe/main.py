import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class ModelArgs:
    dim:int = 4096
    n_layers:int  = 32
    head_dim:int = 128
    hidden_dim : int = 14336
    n_head:int = 9
    window_size:int = 4096
    vocab_size:int=32000
    num_experts_per_token:int=2
    num_experts:int =8
    
    
class Mistral(nn.Module):
    def __init__(self,args:ModelArgs):
        super().__init__()
        
        self.args=args
        self.tok_embedding = nn.Embedding(args.vocab_size,args.dim)
        
        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(DecoderBlock(args)) 
        self.out_norm =RMSNorm(args)
        self.linear = nn.Linear(args)
        
        self.freq_complex=
        
        def forward(self,x:torch.Tensor,start_pos:int):
            
            batch_size , seq_len= x.shape
            
            freq_complex=self.freq_complex()
            tokens=self.embedding(x)
            for layer in self.layers:
                out =layer(tokens)
            out = self.out_norm(x)
            out = self.linear(x)
            out =F.softmax(out)
            return out
        
        
        
        
        
        

