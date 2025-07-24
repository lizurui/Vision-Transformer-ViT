import torch
import torch.nn as nn
import time
import os
import torch
import torch.nn as nn
import math



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout = 0.1, max_len=5000):
        """
        :param d_model: 词嵌入的维度
        :param max_len: 句子的最大长度
        """
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        
        # 创建一个足够长的位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        
        # 创建位置张量 [0, 1, 2, ..., max_len-1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 计算除数项 1 / (10000^(2i/d_model))
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 应用正弦和余弦函数
        pe[:, 0::2] = torch.sin(position * div_term) # 偶数索引
        pe[:, 1::2] = torch.cos(position * div_term) # 奇数索引
        
        pe = pe.unsqueeze(0) 
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: 输入张量，形状为 [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
    

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model                      # 模型总维度
        self.n_heads = n_heads                      # 头数
        self.d_k = d_model // n_heads               # 每个头的维度
        
        # Q, K, V 的线性变换层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # 输出的线性变换层
        self.W_o = nn.Linear(d_model, d_model)
        
    def dot_product_attention(self, Q, K, V, mask=None):
        ktrans = K.transpose(-2, -1)
        scores = torch.matmul(Q, ktrans)
        scores = scores / math.sqrt(self.d_k)
        
        # 应用掩码 (如果提供)
        if mask is not None:
            scores = scores + mask
            
        attn_weights = nn.functional.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        return output, attn_weights

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # 1. 线性变换
        # q, k, v 形状: [batch_size, seq_len, d_model] [2, 10, 128]
        Q = self.W_q(q)
        K = self.W_k(k)
        V = self.W_v(v)
        
        # 2. 拆分成多个头
        # view 操作将 d_model 维度拆分为 n_heads * d_k
        # [batch_size, seq_len, n_heads, d_k] [2, 10, 4, 32]
        # [batch_size, n_heads, seq_len, d_k] [2, 4, 10, 32]
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 3. 计算点积注意力
        x, attn_weights = self.dot_product_attention(Q, K, V, mask)
        
        # 4. 合并头
        x = x.transpose(1, 2)
        x = x.reshape(batch_size, -1, self.d_model)
        
        # 5. 最终的线性变换
        output = self.W_o(x)
        return output
    
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x 形状: [batch_size, seq_len, d_model]
        return self.fc2(self.dropout(self.relu(self.fc1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        # 1. 多头自注意力
        # 残差连接：输入 src + 注意力输出
        attn_output = self.self_attn(src, src, src, src_mask)
        src = self.norm1(src + self.dropout1(attn_output))
        
        # 2. 前馈网络
        # 残差连接：上一步的输出 + 前馈网络输出
        ff_output = self.feed_forward(src)
        src = self.norm2(src + self.dropout2(ff_output))
        
        return src
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.enc_dec_attn = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask, memory_mask):
        # 1. 带掩码的多头自注意力 (Masked Self-Attention)
        attn_output = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = self.norm1(tgt + self.dropout1(attn_output))
        
        # 2. 编码器-解码器注意力 (Encoder-Decoder Attention)
        # Query 来自解码器，Key 和 Value 来自编码器的输出 (memory)
        attn_output = self.enc_dec_attn(tgt, memory, memory, memory_mask)
        tgt = self.norm2(tgt + self.dropout2(attn_output))
        
        # 3. 前馈网络
        ff_output = self.feed_forward(tgt)
        tgt = self.norm3(tgt + self.dropout3(ff_output))
        
        return tgt
    

class Encoder(nn.Module):
    def __init__(self, d_model, n_layers, n_heads, d_ff, dropout=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.pos_encoding = PositionalEncoding(d_model)
        self.encoderLayer = [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        self.layers = nn.ModuleList(self.encoderLayer)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask = None):
        src = self.pos_encoding(src)
        src = self.dropout(src)
        for layer in self.layers:
            src = layer(src, src_mask)
            
        return src

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, dropout=0.1):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.decoderLayer = [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        self.layers = nn.ModuleList(self.decoderLayer)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask = None, memory_mask = None):
        tgt = self.pos_encoding(tgt)
        tgt = self.dropout(tgt)
        
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask, memory_mask)
            
        return tgt
    

class Transformer(nn.Module):
    def __init__(self, tgt_num_classes, d_model, n_heads, n_layers, d_ff, dropout=0.1):
        super(Transformer, self).__init__()
        
        self.encoder = Encoder(d_model, n_layers, n_heads, d_ff, dropout)
        self.decoder = Decoder(tgt_num_classes, d_model, n_layers, n_heads, d_ff, dropout)
 
    def forward(self, src, tgt, tgt_mask):
        # src: [batch_size, src_len, d_model]
        # tgt: [batch_size, tgt_len, d_model]

        memory = self.encoder(src)
        output = self.decoder(tgt, memory, tgt_mask)
        
        return output