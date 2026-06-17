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



def main():
    l = Linear(3,4)
    x = torch.Tensor(5,3)
    output = l.forward(x)
    print(f"weight shape: {l.w.size()}")
    print(f"input : {x.size()}")
    print(f"output : {output.size()}")


if __name__ == "__main__" :
    main()