import regex as re
def merge_token_tuple(token_tuple : tuple[bytes,...],max_pair : tuple[bytes,bytes]) -> tuple[bytes,...] :
    l1 = max_pair[0]
    l2 = max_pair[1]
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


def pre_tokenize(text : str , special_tokens : list[str],**kwargs) -> list[str] :
    result = []
    #divide by special tokens
    special_pat = "|".join(re.escape(token) for token in sorted(special_tokens,key=len,reverse= True))
    text_splited = re.split(special_pat,text)
    #divide by re
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for st in text_splited :
        result.extend(re.findall(PAT,st))
    return result

def train_bpe(file_path ,vocab_size , special_tokens ,**kwargs) ->tuple[dict[int,bytes],list[tuple[bytes,bytes]]] :
    with open(file_path,"r",encoding="utf-8") as f :
        text = f.read()
    #initialize the vocab_dict and merges
    vocab_dict = {}
    for i in range(256) :
        vocab_dict[i] = bytes([i])
    for st in special_tokens :
        vocab_dict[len(vocab_dict)] = st.encode("utf-8")
    merges = list()
    #pre-tokenize -> list of str
    text_splited = pre_tokenize(text,special_tokens)


    freq_table :dict[tuple[bytes,],int] = {}
    for token_block in text_splited :
        token_block_unicode = token_block.encode("utf-8")
        key = tuple(bytes([b]) for b in token_block_unicode)
        freq_table[key] = freq_table.get(key,0) + 1
    
    while(len(vocab_dict) < vocab_size) :
        pair_freq : dict[tuple[bytes,bytes],int] = {} 
        for tp in freq_table :
            for p in zip(tp,tp[1:]) :
                pair_freq[p] = pair_freq.get(p,0) + freq_table[tp]
        if not pair_freq : break
        max_pair = max(pair_freq,key=lambda k : (pair_freq[k],k))
        merges.append(max_pair)
        vocab_dict[len(vocab_dict)] = max_pair[0] + max_pair[1]
        #update the freq_table
        new_freq_table = {}
        for token_tuple, count in freq_table.items() :
            new_token_tuple = merge_token_tuple(token_tuple,max_pair)
            new_freq_table[new_token_tuple] = new_freq_table.get(new_token_tuple,0) + count
        freq_table = new_freq_table
    
    return tuple([vocab_dict,merges])
    



    
def main():
    vocab_size = 50304
    special_tokens = ["<|endoftext|>"]
    file_path = "/mnt/d/用户/Desktop/CS336/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt"
    vocab_dict, merges = train_bpe(file_path,vocab_size,special_tokens)

if __name__ == "__main__":
    main()