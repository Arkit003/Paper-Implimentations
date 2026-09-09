import torch
import torch.nn as nn

class BasicTokenizer():
    def __init__(self):
        self.merges={}
    
    def train(self,text,vocab_size,verbose=False):
        tokens = text.encode("utf-8") # raw bytes
        tokens = list(map(int, tokens)) 
        ids=list(tokens)
        num_merges=vocab_size-256
        
        
        for i in range(num_merges):
            stats=self._get_stats(ids)
            top_pair=self._top_pair(stats)
            idx=256+i
            if verbose:
                print(f"merging {top_pair} into {idx}")

            ids=self._merge(ids,top_pair,idx)
            self.merges[top_pair]=idx
        return ids
        
    def encode(self,text):
        tokens=list(text.encode("utf-8"))
        while len(tokens)>=2:
            stats=self._get_stats(tokens)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx=self.merges[pair]
            tokens=self._merge(tokens,pair,idx)
        return tokens
                
            
    
    def decode(self,ids,):
        vocab={idx:bytes([idx]) for idx in range(256)}
        for (p0,p1),idx in self.merges.items():
            vocab[idx]=vocab[p0]+vocab[p1]
        
        tokens=b"".join(vocab[idx] for idx in ids)
        text=tokens.decode('utf-8',errors="replace")
        return text
    
    def _get_stats(self,ids):
        counts={}
        for pair in zip(ids,ids[1:]):
            counts[pair]=counts.get(pair,0)+1
        return counts
    
    def _top_pair(self,stats):
        return max(stats,key=stats.get)
    
    def _merge(self,ids,pair,idx):
        new_ids=[]
        i=0
        while i<len(ids):
            if i<len(ids)-1 and ids[i]==pair[0] and ids[i+1]==pair[1]:
                new_ids.append(idx)
                i+=2
            else:
                new_ids.append(ids[i])
                i+=1
        return new_ids
            
if __name__=='__main__':
    tokenizer=BasicTokenizer()
    with open("train.txt","r") as f:
        text=f.read()
    vocab_size=500
    tokenizer.train(text,vocab_size,verbose=True)
    with open("taylor_swift.txt","r") as f:
        text=f.read()
    text=tokenizer.encode(text)
    ids=[105, 118, 257, 67, 417, 109, 263, 267, 65, 116, 116, 262, 98, 117, 116, 375, 45, 83, 459, 101, 65, 108, 105, 107, 257, 76, 344, 298, 115, 257, 52, 46, 48, 59, 32, 383, 284, 116, 375, 272, 264, 268, 109, 267, 109, 97, 330, 97, 112, 443, 121, 261, 66, 330, 117, 115, 305, 264, 427, 115, 419, 101, 325, 121, 424, 32, 356, 460, 257, 364, 271, 84, 268, 109, 267, 380, 85, 115, 257, 332, 80, 262, 118, 97, 99, 330, 80, 111, 108, 344, 121, 261, 87, 105, 107, 105, 112, 101, 284, 97, 194, 174, 32, 105, 267, 296, 460, 103, 105, 300, 268, 266, 116, 114, 383, 293, 276, 107, 32, 386, 87, 105, 107, 428, 101, 284, 296, 70, 111, 336, 100, 446, 325, 285, 99, 46, 325, 296, 110, 263, 45, 112, 280, 102, 105, 273, 277, 103, 270, 105, 122, 446, 46, 10, 80, 262, 118, 97, 99, 330, 112, 111, 108, 344, 121, 65, 98, 424, 273, 87, 105, 107, 105, 112, 101, 284, 97, 68, 465, 99, 108, 97, 428, 268, 115, 67, 263, 116, 97, 99, 273, 87, 105, 107, 105, 112, 101, 284, 97, 67, 111, 100, 257, 380, 67, 263, 100, 117, 99, 116, 68, 101, 118, 432, 111, 112, 268, 115, 83, 116, 292, 105, 300, 344, 115, 67, 111, 111, 107, 105, 257, 300, 292, 293, 377, 77, 111, 98, 415, 257, 118, 105, 101, 119, 10, 87, 105, 107, 428, 101, 284, 296, 70, 111, 336, 100, 446, 80, 448, 268, 266, 98, 330, 77, 101, 284, 97, 87, 105, 107, 105]
    print(tokenizer.decode(ids))
    # print(text)
    
    