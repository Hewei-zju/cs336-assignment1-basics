import torch
import torch.nn as nn
from einops import rearrange, einsum

class Embedding(nn.Module) :
    def __init__(self,num_embeddings,embedding_dim,device=None,dtype=None):
        """
        num_embeddings : vocab_size
        embedding_dim : C
        """
        super().__init__()
        self.embedding_matrix = nn.Parameter(torch.randn(num_embeddings,embedding_dim))
    def forward(self,token_ids:torch.Tensor) -> torch.Tensor :
        return self.embedding_matrix[token_ids.long()]
    


def main() :
    embedding = Embedding(10,8)
    input = torch.randn(3,5)
    output = embedding(input)
    print(f"input : {input.size()}")
    print(f"output : {output.size()}")

if __name__ == "__main__":
    main()