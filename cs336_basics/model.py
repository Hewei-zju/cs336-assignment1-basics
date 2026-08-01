import torch
import torch.nn as nn
from einops import rearrange, einsum
import math
from collections.abc import Callable,Iterable
from typing import Optional

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
def silu(x):
    return torch.sigmoid(x)*x

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
        mask = torch.tril(torch.ones(seq_len,seq_len,dtype=torch.bool,device=Q.device)) 
        # print(f"mask : {mask}")
    QK_masked = Q_K.masked_fill(~mask,float("-inf"))
    A = softmax(QK_masked,-1)
    out_put = einsum(A,V,"... q k,... k v->... q v")
    return out_put

def learning_rate_schedule(t,alpha_max,alpha_min,T_w,T_c):
    if t < T_w :
        alpha_t = (t*alpha_max)/T_w
    elif T_w <= t <= T_c :
        alpha_t = alpha_min + 0.5*(1+math.cos(((t-T_w)*math.pi)/(T_c-T_w)))*(alpha_max-alpha_min)
    else :
        alpha_t = alpha_min
    return alpha_t

def gradient_clipping(parameters,max_l2_norm,eps = 1e-6):
    total_norm = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        total_norm += p.grad.norm(2).item()**2
    total_norm = math.sqrt(total_norm)
    scale_w = max_l2_norm/(total_norm+eps)
    if total_norm >= max_l2_norm :
        for p in parameters:
            if p.grad is not None:
                p.grad *= scale_w

def data_loading(tokens,batch_size,context_length,device:str = "cpu"):
    """
    x :[token_id,token_id,...]
    input : (batch_size,context_length)
    target : (batch_size,context_length)
    """
    # tokens = torch.as_tensor(tokens,device=device)
    starts = torch.randint(0,len(tokens)-context_length,(batch_size,),device=device)
    # offset = torch.arange(context_length,device=device)
    # input_idx = starts.unsqueeze(1) + offset
    # target_idx = input_idx + 1
    # print(f"input index : {input_idx}")
    # print(f"target index : {target_idx}")
    input = torch.tensor([tokens[start.item():start.item()+context_length] for start in starts]).to(device)
    target = torch.tensor([tokens[start.item()+1:start.item()+context_length+1] for start in starts]).to(device)
    return (input , target)

def save_checkpoint(model,optimizer,iteration,out,model_config = {}):
    checkpoint = {"model":model.state_dict(),"optimizer":optimizer.state_dict(),"iteration":iteration,"model_config":model_config}
    torch.save(checkpoint,out)

def load_checkpoint(src,model=None,optimizer=None):
    checkpoint = torch.load(src)
    if model is not None :
        model.load_state_dict(checkpoint["model"])
    if optimizer is not None :
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]


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
        self.device = device
        self.dtype = dtype
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
        mh= rearrange(mh,"... h seq_len d_v->... seq_len (h d_v)")
        return self.output_proj(mh)


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
        x = x + self.attn(self.ln1(x),token_positions)
        output = x + self.ffn(self.ln2(x))
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

class SGD(torch.optim.Optimizer):
    def __init__(self,params,lr=1e-3):
        if lr < 0 :
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr":lr}
        super().__init__(params,defaults)
    
    def step(self,closure:Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t",0)
                grad = p.grad.data
                p.data -= lr/math.sqrt(t+1)*grad
                state["t"] = t+1
        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(self,params:Iterable[nn.Parameter],lr,eps,betas,weight_decay):
        if lr < 0 :
            raise ValueError(f"Invalid learning rate : {lr}")
        defaults = {"lr":lr,"eps":eps,"weight_decay":weight_decay,"betas":betas}
        super().__init__(params,defaults)
    def step(self,closure:Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            betas = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                m = state.get("m",torch.zeros_like(p))
                v = state.get("v",torch.zeros_like(p))
                t = state.get("t",1)
                grad = p.grad.data
                
                lr_t = lr*math.sqrt(1-betas[1]**t)/(1-betas[0]**t)
                p.data -= lr*weight_decay*p.data
                m = betas[0]*m+(1-betas[0])*grad
                v = betas[1]*v+(1-betas[1])*(grad**2)
                p.data -= lr_t*(m/(torch.sqrt(v)+eps))

                #update 
                state["m"] = m
                state["v"] = v
                state["t"] = t+1
        return loss

def main():
    # weights = torch.nn.Parameter(5*torch.randn(10,10))
    # opt = SGD([weights],lr=1e1)
    # for t in range(100):
    #     opt.zero_grad()
    #     loss = (weights**2).mean()
    #     print(loss.cpu().item())
    #     loss.backward()
    #     opt.step()
    x = [0,1,2,3,4,5,6,7]
    data_loading(x=x,batch_size=4,context_length=4)

if __name__ == "__main__" :
    main()