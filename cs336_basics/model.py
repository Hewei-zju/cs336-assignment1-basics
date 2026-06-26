import torch
import torch.nn as nn
from einops import rearrange, einsum
import math

class Linear(nn.Module) :
    def __init__(self,in_features,out_features,device=None,dtype=None):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(torch.randn(out_features,in_features,device=self.device,dtype=self.dtype))
        sigma = 2/(in_features+out_features)
        std_sigma = math.sqrt(sigma)
        nn.init.trunc_normal_(self.weight,mean=0,std=sigma,a=-3*sigma,b=3*sigma)
    
    def forward(self,x:torch.Tensor) -> torch.Tensor :
        """
        input(...,in_features)@L.w.T(in_features,out_features)
        """
        return einsum(self.weight,x,"out_features in_features,... in_features -> ... out_features")


class Embedding(nn.Module) :
    def __init__(self,num_embeddings,embedding_dim,device=None,dtype=None):
        """
        embedding_dim : C
        """
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_embeddings,embedding_dim,device=device,dtype=dtype))
        self.device = device
        self.dtype = dtype
    def forward(self,token_ids:torch.Tensor) -> torch.Tensor :
        return self.weight[token_ids.long()]



class RMSNorm(nn.Module):
    def __init__(self,d_model:int,eps:float = 1e-5,device=None,dtype=None):
        """
        d_model: C
        """
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        self.weight = nn.Parameter(torch.randn(d_model,device=device,dtype=dtype))
        self.device = device
        self.dtype = dtype
    def forward(self,x:torch.Tensor) ->torch.Tensor :
        x = x.to(torch.float32)
        in_dtype = x.dtype
        rms = torch.sqrt(self.eps+(x**2).mean(dim=-1,keepdim = True))
        result = (x/rms)*self.weight
        # result = einsum(self.g,x,"")
        return result.to(in_dtype)

class FFN(nn.Module):
    """
    position wise feed forward network
    """
    def __init__(self,d_model,d_ff = None,device=None,dtype=None):
        super().__init__()
        self.d_model = d_model
        if not d_ff :
            d_ff = (8/3)*d_model
            self.d_ff = 64*math.ceil(d_ff/64)
        else :
            self.d_ff = d_ff
        self.w1 = Linear(self.d_model,self.d_ff,device,dtype)
        self.w3 = Linear(self.d_model,self.d_ff,device,dtype)
        self.w2 = Linear(self.d_ff,self.d_model,device,dtype)
    def SiLU(self,x : torch.Tensor):
        return torch.sigmoid(x)*x
    def forward(self,x:torch.Tensor):
        # x(....,d_model) -> ... -> x(...,d_model)
        return self.w2(self.SiLU(self.w1(x))*self.w3(x))

class RotaryPositionEmbedding(nn.Module):
    def __init__(self,theta,d_k:int,max_seq_len:int, device=None,dtype=None) :
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        i = torch.arange(self.max_seq_len,device=device,dtype=dtype).float() # shape (max_seq_len)
        k = torch.arange(self.d_k//2,device=device,dtype=dtype).float()+1.0 # k = [1,...,d_k/2]
        w = 1.0/(self.theta**((2*k-2)/self.d_k)) # shape (d_k/2)
        angle = einsum(i,w,"i,w->i w") #(max_seq_len,d_k/2)
        self.register_buffer("sin_cache",torch.sin(angle),persistent = False) #shape = (max_seq_len,d_k/2)
        self.register_buffer("cos_cache",torch.cos(angle),persistent = False) 
    def forward(self, x:torch.Tensor, token_positions : torch.Tensor = None) -> torch.Tensor : # x shape (...,seq_len,d_k)
        if token_positions is None:
            token_positions = arrange(x.shape[-2],device = x.device)
        sin = self.sin_cache[token_positions] #shape : (...,seq_len,d_k/2)
        cos = self.cos_cache[token_positions]
        even = x[...,0::2]*cos - x[...,1::2]*sin
        odd = x[...,0::2]*sin + x[...,1::2]*cos
        out_put = torch.empty_like(x)
        out_put[...,0::2] = even
        out_put[...,1::2] = odd
        return out_put # (...,seq_len,d_k)

def softmax(x : torch.Tensor, i : int) :
    """
    apply softmax on the i-th dimension of the tensor
    """
    max_value = x.max(dim=i,keepdim = True).values
    x_ = x - max_value 
    s = torch.sum(torch.exp(x_),dim = i,keepdim = True)
    out_put = torch.exp(x_)/s
    return out_put

def scaled_dot_product_attention(Q,K,V,mask=None) :
    Q_K = einsum(Q,K,"... seq_len_q d_k,... seq_len_k d_k->... seq_len_q seq_len_k")/math.sqrt(float(Q.shape[-1]))
    seq_len = Q.shape[-2]
    if mask is None:
        mask = torch.tril(torch.ones(seq_len,seq_len,dtype=torch.bool)) 
        # print(f"mask : {mask}")
    QK_masked = Q_K.masked_fill(~mask,float("-inf"))
    A = softmax(QK_masked,-1)
    out_put = einsum(A,V,"... q k,... k v->... q v")
    return out_put

class Multihead_Self_Attention(nn.Module):
    def __init__(self,d_model:int,num_heads:int,rope=None,device=None,dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model/num_heads
        self.d_v = d_model/num_heads
        self.q_proj = Linear(d_model,d_model,device,dtype) #(d_model,h*d_k)
        self.k_proj = Linear(d_model,d_model,device,dtype) #(d_model,h*d_k)
        self.v_proj = Linear(d_model,d_model,device,dtype) #(h*d_v,d_model)
        self.output_proj = Linear(d_model,d_model,device,dtype) #(d_model,h*d_v)
        self.rope = rope
    def forward(self,x : torch.Tensor,token_positions = None):
        Q_cat = self.q_proj(x) 
        K_cat = self.k_proj(x)
        V_cat = self.v_proj(x)
        Q = rearrange(Q_cat,"... T (h d_k)->... h T d_k",h=self.num_heads)
        K = rearrange(K_cat,"... T (h d_k)->... h T d_k",h=self.num_heads)
        if self.rope :
            Q = self.rope(Q,token_positions)
            K = self.rope(K,token_positions)
        V = rearrange(V_cat,"... T (h d_v)->... h T d_v",h=self.num_heads)
        mh = scaled_dot_product_attention(Q,K,V) #mh shape : (...,h,seq_len,d_v)
        mh_rearan= rearrange(mh,"... h seq_len d_v->... seq_len (h d_v)")
        output = self.output_proj(mh_rearan)
        return output


class Transformer_block(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,theta,max_seq_len,device=None,dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.ffn = FFN(d_model=d_model,d_ff=d_ff,device=device,dtype=dtype)
        self.rope = RotaryPositionEmbedding(theta=theta,d_k = d_model/num_heads,max_seq_len=max_seq_len,device=device,dtype=dtype)
        self.attn = Multihead_Self_Attention(d_model,num_heads,self.rope,device=device,dtype=dtype)
        self.ln1 = RMSNorm(d_model,device=device,dtype=dtype)
        self.ln2 = RMSNorm(d_model,device=device,dtype=dtype)
    def forward(self,x): #x.shape (...,seq_len,d_model)
        token_positions = torch.arange(x.shape[-2],device=x.device)
        x_ = x + self.attn(self.ln1(x),token_positions)
        output = x_ + self.ffn(self.ln2(x_))
        return output

class Transformer_LM(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,theta,vocab_size,context_length,num_layers,device=None,dtype=None):
        super().__init__()
        self.token_embeddings = Embedding(num_embeddings=vocab_size,embedding_dim=d_model,device=device,dtype=dtype)
        self.ln_final = RMSNorm(d_model=d_model,device=device,dtype=dtype)
        self.lm_head = Linear(d_model,vocab_size,device=device,dtype=dtype)
        self.layers = nn.ModuleList([Transformer_block(d_model,num_heads,d_ff,theta,max_seq_len=context_length,device=device,dtype=dtype) for i in range(num_layers)])
    def forward(self,x):
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.ln_final(x))

def cross_entropy(logits,targets):
    """
    logits : output of transformer_lm, shape : (batch_size,vocab_size)
    targets : shape (batch_size,)
    """
    # stable softmax
    max_logits = torch.max(logits,dim=-1,keepdim=True).values
    logits = logits - max_logits #(B,V)
    output = torch.log(torch.sum(torch.exp(logits),dim=-1,keepdim=True)) - logits
    result = output[torch.arange(targets.shape[0]),targets].mean()
    return result

def main():
    a = torch.tensor([[1,2,3],[4,5,6]])
    b = torch.max(a,dim=0,keepdim=True).values
    c = torch.sum(a,dim=1)
    print(a)
    print(b)
    print(c)

if __name__ == "__main__" :
    main()