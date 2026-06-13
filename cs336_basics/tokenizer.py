import pickle
from .byte_level_bpe_trainer import *
from collections.abc import Iterable,Iterator
class Tokenizer :
    def __init__(self,vocab,merges,special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.merges_set = set(merges)
        self.special_tokens = special_tokens
        self.reversed_vocab = {bytes_token : id for id,bytes_token in vocab.items()}
        self.merges_priority = {pair : priority for priority, pair in enumerate(self.merges)}

    @classmethod
    def from_files(cls,vocab_filepath,merges_filepath,special_tokens=None):
        with open(vocab_filepath,"rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath,"rb") as f:
            merges = pickle.load(f)
        return cls(vocab,merges,special_tokens)
    # def get_priority(self ,p) :
    #     if p in self.merges_priority : 
    #         return self.merges_priority[p]
    #     else :
    #         return INF
    def merge(self,token_tuple) :
        """
        merge all the pair until the end of merges
        """
        while True :
            pairs = [p for p in zip(token_tuple,token_tuple[1:]) if p in self.merges_priority]
            if (not pairs) : break 
            highest_pri_pair = min(pairs,key=lambda p : self.merges_priority[p])
            # merge highese_pri_pair
            l1 = highest_pri_pair[0]
            l2 = highest_pri_pair[1]
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
            token_tuple =  new_token_tuple

        return tuple(token_tuple)
    def encode(self,text:str) ->list[int] :
        """
        encode an input text into a sequence of token IDs
        """
        pre_tokenized_text = pre_tokenize(text,self.special_tokens)
        pre_tokenized_tokens : dict[tuple[bytes,...]]  = []
        for sub_word in pre_tokenized_text :
            sub_word_unicode = sub_word.encode("utf-8")
            sub_token = tuple(bytes([b]) for b in sub_word_unicode)
            pre_tokenized_tokens.append(sub_token)
        merged_tokens : list[bytes] = []
        for token_tuple in pre_tokenized_tokens:
            new_token_tuple = self.merge(token_tuple)
            merged_tokens.extend(new_token_tuple)
        token_ids = []
        for token in merged_tokens :
            if token in self.reversed_vocab :
                token_ids.append(self.reversed_vocab[token])
            else :
                token_ids.append(self.insert_token(token))
        return token_ids
    
    def insert_token(self,token) -> int :
        """
        insert a new token to vocab
        """
        id = len(self.reversed_vocab)
        self.reversed_vocab[id] = token
        self.vocab[token] = id
        return id

    def encode_iterable(self,iterable:Iterable[str]) -> Iterator[int] :
        """
        given an iterable of strings, return a generator that lazily yields token IDs
        """
        for text in iterable :
            for token_id in self.encode(text) :
                yield token_id

    def decode(self,ids:list[int])->str :
        """
        decode a sequence of token IDs into text
        """
        return [self.vocab[i] for i in ids]



def main() :
    vocab_filepath = "/mnt/d/用户/Desktop/CS336/assignment1-basics/data/vocab_ts.pkl"
    merges_filepath = "/mnt/d/用户/Desktop/CS336/assignment1-basics/data/merges_ts.pkl"
    special_tokens = ["<|endoftext|>"]
    text = "the cat ate mice"
    tokenizer = Tokenizer.from_files(vocab_filepath,merges_filepath,special_tokens)
    ids = tokenizer.encode(text)
    decode_text = tokenizer.decode(ids)
    print(ids)
    print(decode_text)
    it = tokenizer.encode_iterable([text])
    print(next(it))
    print(next(it))
    print(next(it))
    print(next(it))

if __name__ == "__main__":
    main()