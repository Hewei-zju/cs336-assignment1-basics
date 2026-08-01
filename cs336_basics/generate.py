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
parser.add_argument("--prompt",type=str,default = "Once upon a time, there was a pretty girl named Lily. She loved to eat gum, especially the big black one. One day, Lily's mom asked her to help cook dinner. Lily was so excited! She loved to help her mom.")
parser.add_argument("--model_path",type=str,default = "data/checkpoint_4h.pt")
parser.add_argument("--max_context",type=int,default = 256)

def main():
    args = parser.parse_args()
    #tokenize the prompt
    tokenizer = Tokenizer.from_files(
        vocab_filepath=args.vocab_filepath,
        merges_filepath=args.merges_filepath,
        special_tokens=args.special_tokens,
        )
    prompt_tokens = tokenizer.encode(args.prompt) 
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
        while True:
            #compute logits
            x = torch.tensor([prompt_tokens])
            logits = model(x) #logits.shape : (prompt_length,vocab_size)
            next_token_id = torch.argmax(logits[:,-1,:],dim=-1).item()
            if next_token_id == 256:
                break
            prompt_tokens.append(next_token_id)
            print(tokenizer.decode([next_token_id]), end="", flush=True)

            if len(prompt_tokens) > args.max_context :
                break
    print()
if __name__ == "__main__":
    main()