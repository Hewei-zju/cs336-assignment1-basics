import torch
import torch.nn as nn
from einops import rearrange, einsum
import math

class Linear(nn.Module) :
    def __init__(self,in_features,out_features,device=None,dtype=None):
        super().__init__()
        self.w = nn.Parameter(torch.randn(out_features,in_features))
        sigma = 2/(in_features+out_features)
        std_sigma = math.sqrt(sigma)
        nn.init.trunc_normal_(self.w,mean=0,std=sigma,a=-3*sigma,b=3*sigma)
        self.device = device
        self.dtype = dtype
    
    def forward(self,x:torch.Tensor) -> torch.Tensor :
        """
        input(...,in_features)@L.w.T(in_features,out_features)
        """
        return einsum(self.w,x,"out_features in_features,... in_features -> ... out_features")


class Embedding(nn.Module) :
    def __init__(self,num_embeddings,embedding_dim,device=None,dtype=None):
        """
        num_embeddings : vocab_size
        embedding_dim : C
        """
        super().__init__()
        self.embedding_matrix = nn.Parameter(torch.randn(num_embeddings,embedding_dim))
        self.device = device
        self.dtype = dtype
    def forward(self,token_ids:torch.Tensor) -> torch.Tensor :
        return self.embedding_matrix[token_ids.long()]



class RMSNorm(nn.Module):
    def __init__(self,d_model:int,eps:float = 1e-5,device=None,dtype=None):
        """
        d_model: C
        """
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        self.g = nn.Parameter(torch.randn(d_model))
        self.device = device
        self.dtype = dtype
    def forward(self,x:torch.Tensor) ->torch.Tensor :
        x = x.to(torch.float32)
        in_dtype = x.dtype
        rms = torch.sqrt(self.eps+(x**2).mean(dim=-1,keepdim = True))
        result = (x/rms)*self.g
        # result = einsum(self.g,x,"")
        return result.to(in_dtype)

class FFN(nn.Module):
    """
    position wise feed forward network
    """
    def __init__(self,d_model,d_ff = None):
        super().__init__()
        self.d_model = d_model
        if not d_ff :
            d_ff = (8/3)*d_model
            self.d_ff = 64*math.ceil(d_ff/64)
        else :
            self.d_ff = d_ff
        self.w1 = Linear(self.d_model,self.d_ff)
        self.w3 = Linear(self.d_model,self.d_ff)
        self.w2 = Linear(self.d_ff,self.d_model)
    def SiLU(self,x : torch.Tensor):
        return torch.sigmoid(x)*x
    def forward(self,x:torch.Tensor) :
        # x(....,d_model) -> ... -> x(...,d_model)
        return self.w2(self.SiLU(self.w1(x))*self.w3(x))





def main() :
    pass

if __name__ == "__main__" :
    main()