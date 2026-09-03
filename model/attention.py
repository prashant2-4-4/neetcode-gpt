import torch
import torch.nn as nn
from torchtyping import TensorType
import math
import torch.nn.functional as F

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        # Create three linear projections (Key, Query, Value) with bias=False
        # Instantiation order matters for reproducible weights: key, query, value
        self.key = nn.Linear(embedding_dim , attention_dim , bias = False)
        self.query = nn.Linear(embedding_dim , attention_dim , bias = False)
        self.value = nn.Linear(embedding_dim  , attention_dim ,bias = False)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # 1. Project input through K, Q, V linear layers
        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,
        #    then masked_fill positions where mask == 0 with float('-inf')
        # 4. Apply softmax(dim=2) to masked scores
        # 5. Return (scores @ V) rounded to 4 decimal places

        batch , seq_len , _ = embedded.shape

        Q = self.query(embedded)
        K = self.key(embedded)
        V = self.value(embedded)

        # embedded is of batch , seq_len , attn_dimension
        #k is required in batch , attn_dimension ,seq_len that's why last two transposed only
        score = torch.matmul(Q , K.transpose(-2 , -1))
        score = score/math.sqrt(Q.shape[-1])

        # creating so that token can only see back token only not future token 
        #[1 , 0 , 0 ] , [1 , 1, 0] , [1, ,1 ,1]

        tril = torch.tril(torch.ones(seq_len , seq_len))

        # this is mask but out score is of shape batch , seq_len , seq_len so either unsequeeze to make tril of that same shape

        tril  = tril.unsqueeze(0)

        # tril = tril.masked_fill(tril == 0 , float('-inf'))

        # score = torch.masked_fill(tril==1 , score)
        score = score.masked_fill(tril==0 , float('-inf'))
        #apply softmax
        score = F.softmax(score , dim = 2) # only on last dimension score applied

        # multiply with v

        score = torch.matmul(score ,  V)

        return torch.round(score , decimals = 4)


