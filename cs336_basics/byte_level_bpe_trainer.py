import regex as re
from multiprocessing import Pool
from typing import BinaryIO
import os
import time
from datetime import datetime
import pickle
import math

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def merge_token_tuple(token_tuple : tuple[bytes,...],merged_pair : tuple[bytes,bytes]) -> tuple[bytes,...] :
    l1 = merged_pair[0]
    l2 = merged_pair[1]
    if (l1 not in token_tuple) or (l2 not in token_tuple) : return token_tuple

    new_token_tuple = []
    i = 0
    while i < len(token_tuple):
        if token_tuple[i] == l1 and i+1 < len(token_tuple) and token_tuple[i+1] == l2 :
            new_token_tuple.append(token_tuple[i]+token_tuple[i+1])
            i+=2
        else :
            new_token_tuple.append(token_tuple[i])
            i+=1
    return tuple(new_token_tuple)


def pre_tokenize(text : str , special_tokens : list[str] ,**kwargs) -> list[str] :
    result = []
    #divide by special tokens
    if  special_tokens : 
        special_pat = "|".join(re.escape(token) for token in sorted(special_tokens,key=len,reverse= True))
        text_splited_st = re.split(special_pat,text)
    else :
        text_splited_st = text
    #divide by re
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for st in text_splited_st :
        result.extend(re.findall(PAT,st))
    return result

def pre_tokenize_for_chunk(start:int,end:int,file_path : str ,special_tokens:list[str]) ->dict[tuple[bytes,bytes] : int]:
    with open(file_path,"rb") as f :
        f.seek(start)
        chunk_text = f.read(end-start).decode("utf-8",errors = "ignore")
    words_list =   pre_tokenize(chunk_text,special_tokens)
    freq_table = {}
    for token_block in words_list :
        token_block_unicode = token_block.encode("utf-8")
        key = tuple(bytes([b]) for b in token_block_unicode)
        freq_table[key] = freq_table.get(key,0) + 1
    return freq_table

def merge_freq_tables(freq_tables : list[dict[tuple[bytes,bytes] : int]]) -> dict[tuple[bytes,bytes] : int] :
    freq_table = {}
    for d in freq_tables :
        for token_tuple, count in d.items():
            freq_table[token_tuple] = freq_table.get(token_tuple,0) + count
    return freq_table


def parallel_arg(file_path:str) :
    file_size = os.path.getsize(file_path)/(1024**2)
    cpu_kernls = os.cpu_count() or 1
    print(cpu_kernls)
    chunk_size = 32
    num_chunks =  math.ceil(file_size/chunk_size)
    num_process = min(max(num_chunks//8,1),cpu_kernls)
    return num_process,num_chunks



def train_bpe(file_path ,vocab_size , special_tokens :list[bytes],**kwargs) ->tuple[dict[int,bytes],list[tuple[bytes,bytes]]] :
    time_start = time.perf_counter()
    print(f"training start at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    vocab_dict = {}
    for i in range(256) :
        vocab_dict[i] = bytes([i])
    for st in special_tokens :
        vocab_dict[len(vocab_dict)] = st.encode("utf-8")
    merges = list()


    num_process , num_chunks = parallel_arg(file_path) 

    print(f"process_num : {num_process}, chunk_nums:  {num_chunks}")
    with open(file_path,"rb") as f :
        split_special_token = "<|endoftext|>".encode("utf-8")
        boundaries = find_chunk_boundaries(f,num_chunks,split_special_token)
        with Pool(processes=num_process) as pool :
            data_tuple = [tuple([start,end,file_path,special_tokens]) for start,end in zip(boundaries[:-1],boundaries[1:])]
            freq_tables = pool.starmap(pre_tokenize_for_chunk,data_tuple)

    freq_table = merge_freq_tables(freq_tables)
    print(f"pre-tokenization done, elapsed: {time.perf_counter() - time_start:.2f}s")

    #maintain a dict in which {pair->(tuples in freq_table)}
    # pair_tuple : dict[tuple[bytes,bytes],tuple[bytes,...]]= {}
    #maintain a priority queue for pair

    pair_freq : dict[tuple[bytes,bytes],int] = {} 
    train_count = 1
    while(len(vocab_dict) < vocab_size) :
        #only first time, iterate every item to update pair_freq
        if train_count == 1 :
            for tp in freq_table :
                for p in zip(tp,tp[1:]) :
                    pair_freq[p] = pair_freq.get(p,0) + freq_table[tp]
        if not pair_freq : break
        if train_count % 1000 == 0 : print(f"training count: {train_count}s,elapsed: {time.perf_counter() - time_start:.2f}s")
        max_pair = max(pair_freq,key=lambda k : (pair_freq[k],k))
        merges.append(max_pair)
        vocab_dict[len(vocab_dict)] = max_pair[0] + max_pair[1]
        #update the freq_table and pair_freq
        new_freq_table = {}
        for token_tuple, count in freq_table.items() :
            new_token_tuple = merge_token_tuple(token_tuple,max_pair)
            update_pair_freq(token_tuple,new_token_tuple,pair_freq,count)
            new_freq_table[new_token_tuple] = new_freq_table.get(new_token_tuple,0) + count
        freq_table = new_freq_table
        train_count += 1
    
    return tuple([vocab_dict,merges])
    
def update_pair_freq(old_token_tuple,new_token_tuple,pair_freq,count) :
    if old_token_tuple == new_token_tuple : return
    #add new_token_tuple to pair_freq
    changed_pair = set()
    for p in zip(new_token_tuple,new_token_tuple[1:]) :
        pair_freq[p] = pair_freq.get(p,0) + count
        changed_pair.add(p)
    #delete old_token_tuple from pair_freq
    for p in zip(old_token_tuple,old_token_tuple[1:]) :
        pair_freq[p] = pair_freq.get(p,0) - count
        changed_pair.add(p)
    for p in changed_pair :
        if pair_freq.get(p,0) <= 0 :
            pair_freq.pop(p,None)
    return

def save(vocab_dict,merges,output_path_vocab,output_path_merges) :
    with open(output_path_vocab,"wb") as f:
        pickle.dump(vocab_dict,f)
    with open(output_path_merges,"wb") as f:
        pickle.dump(merges,f)
    
def main():
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]
    file_path = "/home/hewei/cs336-assignment1-basics/data/TinyStoriesV2-GPT4-train.txt"
    output_path_vocab = "/home/hewei/cs336-assignment1-basics/data/vocab_ts.pkl"
    output_path_merges ="/home/hewei/cs336-assignment1-basics/data/merges_ts.pkl" 
    vocab_dict, merges = train_bpe(file_path,vocab_size,special_tokens)
    save(vocab_dict,merges,output_path_vocab,output_path_merges)
    print(max(vocab_dict.items(),key=lambda item : len(item[1])))
if __name__ == "__main__":
    main()