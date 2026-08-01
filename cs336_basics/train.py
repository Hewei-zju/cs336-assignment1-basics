import argparse
from tokenizer import Tokenizer
from model import *
import numpy as np
from pathlib import Path
from tqdm import tqdm
parser = argparse.ArgumentParser()
parser.add_argument("--device",type=str,default="cuda")
parser.add_argument("--filepath_training",type=str,default="data/TinyStoriesV2-GPT4-train.txt")
parser.add_argument("--filepath_validation",type=str,default="data/TinyStoriesV2-GPT4-valid.txt")
parser.add_argument("--lr",type=float,default=1e-4)
parser.add_argument("--batch_size",type=int,default=8)
parser.add_argument("--vocab_filepath",type=str,default="data/vocab_ts.pkl")
parser.add_argument("--merges_filepath",type=str,default="data/merges_ts.pkl")
parser.add_argument("--special_tokens",type=str,nargs= "+",default=["<|endoftext|>"])
parser.add_argument("--d_model",type=int,default=768)
parser.add_argument("--num_heads",type=int,default=4)
parser.add_argument("--d_ff",type=int,default=2048)
parser.add_argument("--theta",type=float,default=10000.0)
parser.add_argument("--vocab_size",type=int,default=10000)
parser.add_argument("--context_length",type=int,default=256)
parser.add_argument("--num_layers",type=int,default=12) # num of transformer blocks
parser.add_argument("--dtype",type=str,default="float32")
parser.add_argument("--iterations",type=int,default=10000)
parser.add_argument("--eps",type=float,default=1e-6)
parser.add_argument("--betas",type=float,nargs = 2,default=(0.9,0.999))
parser.add_argument("--weight_decay",type=float,default=0.01)
parser.add_argument("--resume",action="store_true",help="resume training from checkpoint")

dtype_mapping = {
    "float16" : torch.float16,
    "float32" : torch.float32,
    "int32" : torch.int32,
    "int64" : torch.int64
}
class Trainer():
    def __init__(self):
        pass
    def train(self):
        pass

def main():
    args = parser.parse_args()
    #
    filepath_training = args.filepath_training
    filepath_validation = args.filepath_validation
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
    training_dataset_path = Path("data/train.bin")
    valid_path = Path("data/valid.bin")
    checkpoint_path = Path("data/checkpoint_local.pt")

    tokenizer = Tokenizer.from_files(vocab_filepath=vocab_filepath,merges_filepath=merges_filepath,special_tokens=special_tokens)
    #load the valid data
    print(f"validation dataset loading...")
    if not valid_path.exists():
            buffer = []
            memery_chunk_size = 1000000
            num_chunks = 0
            with open(filepath_validation,"r") as f,open(valid_path,"wb") as out:
                valid_tokens = tokenizer.encode_iterable(f)
                for token in valid_tokens:
                    buffer.append(token)
                    if len(buffer) >= memery_chunk_size:
                        out.write(np.asarray(buffer,dtype=np.int32).tobytes())
                        buffer.clear()
                if buffer :
                    np.asarray(buffer,dtype = np.int32).tofile(out)
    print(f"validation dataset loaded")
        
    #load the training data, tokenize and save
    print(f"training dataset loading...")
    if not training_dataset_path.exists():
        buffer = []
        memery_chunk_size = 1000000
        num_chunks = 0
        with open(filepath_training,"r") as f,open(training_dataset_path,"wb") as out:
            training_tokens = tokenizer.encode_iterable(f)
            for token in training_tokens:
                buffer.append(token)
                if len(buffer) >= memery_chunk_size:
                    out.write(np.asarray(buffer,dtype=np.int32).tobytes())
                    buffer.clear()
            if buffer :
                np.asarray(buffer,dtype = np.int32).tofile(out)
            
    print(f"training data loaded in file : {training_dataset_path}")

    tokens = np.memmap(training_dataset_path,
                        dtype=np.int32,
                        mode="r",
                    )
    validation_tokens = np.memmap(valid_path,
                                    dtype=np.int32,
                                    mode = "r",
                                    )
    print(f"tokens shape {len(tokens)}")
    #initialize model and otpimizer 
    model_config = {
        "d_model" : d_model,
        "num_heads" : num_heads,
        "d_ff" : d_ff,
        "theta" : theta,
        "vocab_size" : vocab_size,
        "context_length" : context_length,
        "num_layers" : num_layers,
        "device" : device,
        "dtype" : dtype_mapping[dtype],
    }
    model = Transformer_LM(**model_config)
    optimizer = AdamW(params=model.parameters(),lr=lr,eps=eps,betas=betas,weight_decay=weight_decay)
    start_iteration = 0
    if args.resume and checkpoint_path.exists():
        start_iteration = load_checkpoint(src=checkpoint_path,model=model,optimizer=optimizer)+1
    elif args.resume and not checkpoint_path.exists():
        raise FileNotFoundError(f"file not found : {checkpoint_path}")
    else :
        pass
    #training loop
    print(f"training starts,total traning loop {iterations}")
    print(f"training on device : {device}")
    for it in tqdm(range(start_iteration,iterations)):
        # print(f"this is training loop {it}")
        x,target = data_loading(tokens=tokens,batch_size=batch_size,context_length=context_length,device=device)
        # print(f"x : \n{x}, \ntarget : \n{target}")
        logits = model(x) #shape (batch_size,context_length,vocab_size)
        # print(f"logits of model : {logits}")
        loss = cross_entropy(logits.reshape(-1,logits.shape[-1]),target.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if it%1000 == 0 or it+1 == iterations: 
            tqdm.write(f"iteration {it}, training loss: {loss.item():.8f}")
            save_checkpoint(model=model,optimizer=optimizer,iteration=it,model_config=model_config,out=checkpoint_path)
            #validation
            with torch.no_grad():
                x_v, target_v = data_loading(tokens=validation_tokens,batch_size=batch_size,context_length=context_length,device=device)
                logits_v = model(x_v) #shape (batch_size,context_length,vocab_size)
                loss_v = cross_entropy(logits_v.reshape(-1,logits_v.shape[-1]),target_v.reshape(-1))
                tqdm.write(f"iteration {it}, validation loss: {loss_v.item():.8f}")

if __name__ == "__main__":
    main()