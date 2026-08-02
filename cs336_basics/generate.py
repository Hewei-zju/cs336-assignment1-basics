import argparse
from tokenizer import Tokenizer
from model import *
import numpy as np
from pathlib import Path
from tqdm import tqdm
parser = argparse.ArgumentParser()
"""
generate a text with a trained model and a given prompt.

1.tokenize the given prompt
2.load the model
3.logit = model(x)
4.generate the next token
5.update the prompt, add the new token to the prompt
6.loop until endoftext or the context limit
7.decode prompt
"""
parser.add_argument("--vocab_filepath",type=str,default="data/vocab_ts.pkl")
parser.add_argument("--merges_filepath",type=str,default="data/merges_ts.pkl")
parser.add_argument("--special_tokens",type=str,nargs= "+",default=["<|endoftext|>"])
parser.add_argument("--prompt",type=str,default = "Once upon a time, there was a little girl named Lily.")
parser.add_argument("--model_path",type=str,default = "data/checkpoint_4h.pt")
parser.add_argument("--max_context",type=int,default = 256)
parser.add_argument("--temperature",type=float,default = 0.5)
parser.add_argument("--p",type=float,default = 0.8)

def top_p_choose(prob:torch.tensor,p:float):
    sorted_prob, sorted_indice = torch.sort(prob,dim=-1,descending = True)
    cumulative_prob = torch.cumsum(sorted_prob,dim=-1)
    mask = cumulative_prob >= p
    #right shift mask
    mask[...,1:] = mask[...,:-1].clone()
    mask[...,0] = False
    sorted_prob = sorted_prob.masked_fill(mask,0.0)
    sorted_prob = sorted_prob/torch.sum(sorted_prob,dim=-1,keepdim=True)
    sampled_indice = torch.multinomial(sorted_prob,num_samples=1)
    next_token_id = sorted_indice.gather(dim=-1,index = sampled_indice)
    return next_token_id
def main():
    args = parser.parse_args()
    #tokenize the prompt
    tokenizer = Tokenizer.from_files(
        vocab_filepath=args.vocab_filepath,
        merges_filepath=args.merges_filepath,
        special_tokens=args.special_tokens,
        )
    end_of_text_id = tokenizer.reversed_vocab["<|endoftext|>".encode("utf-8")]
    prompt_tokens = tokenizer.encode(args.prompt) 
    x = torch.tensor([prompt_tokens],device = "cuda")
    # print(f"prompt : {args.prompt}")
    # print(f"prompt tokens : {prompt_tokens}")
    #load the model
    model_config = {
        "d_model" : 768,
        "num_heads" : 4,
        "d_ff" : 2048,
        "theta" : 10000.0,
        "vocab_size" : 10000,
        "context_length" : 256,
        "num_layers" : 12,
        "device" : "cuda",
        "dtype" : torch.float32,
    }
    model = Transformer_LM(**model_config)
    load_checkpoint(args.model_path,model)
    print(args.prompt,end="")
    with torch.no_grad():
        while x.shape[-1] < args.max_context:
            #compute logits
            logits = model(x)[:,-1,:]/args.temperature #logits.shape : (prompt_length,vocab_size)
            prob = softmax(logits,i=-1)
            # next_token_id = torch.argmax(prob[:,-1,:],dim=-1).item()
            next_token_tensor = top_p_choose(prob,args.p)
            # prompt_tokens.append(next_token_id)
            x = torch.cat([x,next_token_tensor],dim=-1)
            next_token_id = next_token_tensor.item()
            if next_token_id == end_of_text_id:
                break
            print(tokenizer.decode([next_token_id]), end="", flush=True)
    print()
if __name__ == "__main__":
    main()