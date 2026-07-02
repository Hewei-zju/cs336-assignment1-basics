import argparse
from tokenizer import Tokenizer
from model import *
parser = argparse.ArgumentParser()
parser.add_argument("--device",type=str,default="cpu")
parser.add_argument("--filepath",type=str,default="/home/hewei/cs336-assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt")
parser.add_argument("--lr",type=float,default=1e-3)
parser.add_argument("--batch_size",type=int,default=8)
parser.add_argument("--vocab_filepath",type=str,default="/home/hewei/cs336-assignment1-basics/data/vocab_ts.pkl")
parser.add_argument("--merges_filepath",type=str,default="/home/hewei/cs336-assignment1-basics/data/merges_ts.pkl")
parser.add_argument("--special_tokens",type=str,nargs= "+",default=["<|endoftext|>"])
parser.add_argument("--d_model",type=int,default=768)
parser.add_argument("--num_heads",type=int,default=4)
parser.add_argument("--d_ff",type=int,default=2048)
parser.add_argument("--theta",type=float,default=10000.0)
parser.add_argument("--vocab_size",type=int,default=50257)
parser.add_argument("--context_length",type=int,default=256)
parser.add_argument("--num_layers",type=int,default=12) # num of transformer blocks
parser.add_argument("--dtype",type=str,default="float32")
parser.add_argument("--iterations",type=int,default=10000)
parser.add_argument("--eps",type=float,default=1e-6)
parser.add_argument("--betas",type=float,nargs = 2,default=(0.9,0.999))
parser.add_argument("--weight_decay",type=float,default=0.01)

dtype_mapping = {
    "float16" : torch.float16,
    "float32" : torch.float32,
    "int32" : torch.int32,
    "int64" : torch.int64
}

def main():
    args = parser.parse_args()
    #
    filepath = args.filepath
    vocab_filepath=args.vocab_filepath
    merges_filepath=args.merges_filepath
    special_tokens = args.special_tokens
    d_model = args.d_model
    d_ff = args.d_ff
    num_heads = args.num_heads
    theta = args.theta
    vocab_size = args.vocab_size
    batch_size = args.batch_size
    context_length = args.context_length
    num_layers = args.num_layers
    device = args.device
    dtype = dtype_mapping[args.dtype]
    lr = args.lr
    eps = args.eps
    betas = args.betas
    weight_decay = args.weight_decay
    iterations = args.iterations

    tokenizer = Tokenizer.from_files(vocab_filepath=vocab_filepath,merges_filepath=merges_filepath,special_tokens=special_tokens)
    
    with open(filepath,"r") as f:
        training_dataset = f.read()
    print(f"training data loaded")
    
    tokens = tokenizer.encode(training_dataset)
    training_dataset = None
    print(f"tokens shape {len(tokens)}")
    model = Transformer_LM(d_model=d_model,num_heads=num_heads,d_ff=d_ff,theta=theta,vocab_size=vocab_size,context_length=context_length,num_layers=num_layers,device=device,dtype=dtype)
    optimizer = AdamW(params=model.parameters(),lr=lr,eps=eps,betas=betas,weight_decay=weight_decay)
    #training loop
    print(f"training starts,total traning loop {iterations}")
    for it in range(iterations):
        # print(f"this is training loop {it}")
        x,target = data_loading(tokens=tokens,batch_size=batch_size,context_length=context_length,device=device)
        # print(f"x : \n{x}, \ntarget : \n{target}")
        logits = model(x) #shape (batch_size,context_length,vocab_size)
        # print(f"logits of model : {logits}")
        loss = cross_entropy(logits.reshape(-1,logits.shape[-1]),target.reshape(-1))
        if it % 100 == 0 : 
            print(f"iteration {it}, loss : {loss}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

if __name__ == "__main__":
    main()