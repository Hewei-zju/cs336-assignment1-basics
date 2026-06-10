import regex as re
def merge_token_tuple(token_tuple : tuple[bytes,...],count : int ,max_pair : tuple[bytes,bytes],pair_freq : dict[tuple[bytes,bytes]:int]) -> tuple[bytes,...] :
    l1 = max_pair[0]
    l2 = max_pair[1]
    if (l1 not in token_tuple) or (l2 not in token_tuple) : return token_tuple

    new_token_tuple = []
    i = 0
    while i < len(token_tuple):
        if token_tuple[i] == l1 and i+1 < len(token_tuple) and token_tuple[i+1] == l2 :
            #update pair_freq destructively
            if i > 0 : 
                l1_pre = token_tuple[i-1]
                pre_token_tuple = tuple([l1_pre,l1])
                new_pre_token_tuple = tuple([l1_pre,l1+l2])
                pair_freq[pre_token_tuple] = pair_freq.get(pre_token_tuple,0) - count 
                pair_freq[new_pre_token_tuple] = pair_freq.get(new_pre_token_tuple,0) + count
            if i+2 < len(token_tuple) :
                l2_post = token_tuple[i+2]
                post_token_tuple = tuple([l2,l2_post])
                new_post_token_tuple = tuple([l1+l2,l2_post])
                pair_freq[post_token_tuple] = pair_freq.get(post_token_tuple,0) - count
                pair_freq[new_post_token_tuple] = pair_freq.get(new_post_token_tuple,0) + count
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
    
    pair_freq : dict[tuple[bytes,bytes],int] = {} 
    train_count = 1
    while(len(vocab_dict) < vocab_size) :
        #only first time, iterate every item to update pair_freq
        if train_count == 1 :
            for tp in freq_table :
                for p in zip(tp,tp[1:]) :
                    pair_freq[p] = pair_freq.get(p,0) + freq_table[tp]
        if (not pair_freq) and train_count != 1 : break
        
        max_pair = max(pair_freq,key=lambda k : (pair_freq[k],k))
        merges.append(max_pair)
        vocab_dict[len(vocab_dict)] = max_pair[0] + max_pair[1]
        #update the freq_table and pair_freq
        new_freq_table = {}
        for token_tuple, count in freq_table.items() :
            new_token_tuple = merge_token_tuple(token_tuple,count,max_pair,pair_freq)
            new_freq_table[new_token_tuple] = new_freq_table.get(new_token_tuple,0) + count
        pair_freq.pop(max_pair,None)
        freq_table = new_freq_table
        train_count += 1
    
    return tuple([vocab_dict,merges])
    



    
def main():
    vocab_size = 50304
    special_tokens = ["<|endoftext|>"]
    file_path = "/mnt/d/用户/Desktop/CS336/assignment1-basics/data/test.txt"
    vocab_dict, merges = train_bpe(file_path,vocab_size,special_tokens)
    print(vocab_dict)
    print(merges)

if __name__ == "__main__":
    main()