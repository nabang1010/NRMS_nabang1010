import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Scaled_Dot_Product_Attention(nn.Module):
    def __init__(self, d_k):
        super(Scaled_Dot_Product_Attention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask=None):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        scores = torch.exp(scores)
        if attn_mask is not None:
            scores = scores * attn_mask
        attn = scores / (torch.sum(scores, dim=-1, keepdim=True) + 1e-8)

        context = torch.matmul(attn, V)
        return context, attn
    
    
class Multi_Head_Self_Attention(nn.Module):
    def __init__(self, d_model, num_attention_heads):
        super(Multi_Head_Self_Attention, self).__init__()
        self.d_model = d_model
        self.num_attention_heads = num_attention_heads
        assert d_model % num_attention_heads == 0
        self.d_k = d_model // num_attention_heads
        self.d_v = d_model // num_attention_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, Q, K=None, V=None, length=None):
        if K is None:
            K = Q
        if V is None:
            V = Q
        batch_size = Q.size(0)

        q_s = self.W_Q(Q).view(batch_size, -1, self.num_attention_heads,
                               self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.num_attention_heads,
                               self.d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.num_attention_heads,
                               self.d_v).transpose(1, 2)

        if length is not None:
            attn_mask = torch.arange(q_s.size(-1)).repeat(q_s.size(0), 1).to(device)
            attn_mask = attn_mask < length.repeat(q_s.size(-1), 1).transpose(0, 1)
            attn_mask = attn_mask.unsqueeze(1).repeat(1, q_s.size(1), 1, 1)
        else:
            attn_mask = None

        context, attn = Scaled_Dot_Product_Attention(self.d_k)(q_s, k_s, v_s, attn_mask=attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_attention_heads * self.d_v)
        return context, attn
    
    
    
class Additive_Attention(nn.Module):
    
    def __init__(self, query_vector_dim, candidate_vector_dim, writer=None, tag=None, names=None):
        super(Additive_Attention, self).__init__()
        self.linear = nn.Linear(candidate_vector_dim, query_vector_dim)
        self.attention_query = nn.Parameter(torch.randn(query_vector_dim).uniform_(-0.1, 0.1))
        self.writer = writer
        self.tag = tag
        self.names = names
        self.local_step = 1
        
    def forward(self, candidate_vector):
        temp = torch.tanh(self.linear(candidate_vector))
        candidate_weights = F.softmax(torch.matmul(temp, self.attention_query), dim=1)
        
        if  self.writer is not None:
            assert candidate_weights.size(0) == len(self.names)
            if self.local_step % 10 == 0:
                self.writer.add_scalars(self.tage, {x: y for x, y in zip(self.names, candidate_weights.mean(dim=0))}, self.local_step)
        self.local_step += 1
        
        target = torch.bmm(candidate_weights.unsqueeze(1), candidate_vector).squeeze(1)
class News_Encoder(nn.Module):
    def __init__(self, config, pretrained_word_embedding ):
        super(News_Encoder, self).__init__()
        self.config = config
        if pretrained_word_embedding is None:
            self.word_embedding = nn.Embedding(
                config["BASE_CONFIG"]["num_words"],
                config["BASE_CONFIG"]["word_embedding_dim"],
                padding_idx=0,
            )
        else:
            self.word_embedding = nn.Embedding.from_pretrained(
                pretrained_word_embedding, freeze=False, padding_idx=0
            )
        self.multihead_self_attention = Multi_Head_Self_Attention(
            config["BASE_CONFIG"]["word_embedding_dim"],
            config["NRMS_CONFIG"]["num_attention_heads"],
        )
        self.additive_attention = Additive_Attention(
            config["BASE_CONFIG"]["query_vector_dim"],
            config["BASE_CONFIG"]["word_embedding_dim"],
        )
    
class NRMS(nn.Module):
    def __init__(self, config, pretrained_word_embedding=None):
        super(NRMS, self).__init__()
        self.config = config
        if pretrained_word_embedding is None:
            self.word_embedding = nn.Embedding()
            
        
     


    





