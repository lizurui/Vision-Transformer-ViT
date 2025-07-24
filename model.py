# 文件名: model.py

import torch
import torch.nn as nn
import math
import transformer                                     # type: ignore


d_ff = 512
dropout = 0.1
n_heads = 4
n_layers = 4
# n_layers = 2
d_model = 128
patch_size = 8
img_size = (1, 64, 64)

class ImageToCharViT(nn.Module):
    def __init__(self, tgt_num_classes, image_size):
        super().__init__()
        # 1. 图像分块与投影
        assert image_size % patch_size == 0, "Image size must be divisible by patch size."
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 1 * patch_size ** 2

        self.mlp_embeding = nn.Linear(patch_dim, d_model)

        self.transformer = transformer.Transformer(
            tgt_num_classes,
            d_model = d_model,
            n_heads = n_heads,
            n_layers = n_layers,
            d_ff = d_ff,
            dropout = dropout
        )
        
        # 4. 目标词嵌入（只为 <sos> token）
        self.tgt_embedding = nn.Embedding(tgt_num_classes, d_model)

        # 5. 输出层
        self.output_layer = nn.Linear(d_model, tgt_num_classes)

    def make_mask(self, seq, isCausal):
        # 忽略序列中的填充词元
        seq_pad_mask = (seq == 0).unsqueeze(1).unsqueeze(2) # 形状: [batch_size, 1, 1, tgt_len]
        
        # 创建一个后续词的掩码，防止看到未来的词
        # 防止看到未来的词元,避免未卜先知
        if isCausal:
            seq_len = seq.shape[1]
            seq_causal_mask = torch.triu(torch.ones((seq_len, seq_len), device=seq.device), diagonal=1).bool()
            
            # 合并两种掩码
            seq_mask = (seq_pad_mask | seq_causal_mask) # 形状: [batch_size, 1, tgt_len, tgt_len]
            seq_mask = torch.where(seq_mask, -1e9, 0)
            return seq_mask
        
        seq_pad_mask = torch.where(seq_pad_mask, -1e9, 0)
        return seq_pad_mask

    def forward(self, src, tgt):
        
        batch_size, c, h, w = src.shape
        
        in_features = self.mlp_embeding.in_features
        p_size = int(math.sqrt(in_features))
        
        # a. 图像分块、展平、投影
        src_patches = src.unfold(2, p_size, p_size)
        src_patches = src_patches.unfold(3, p_size, p_size)     # [B, C, num_patches_h, num_patches_w, patch_h, patch_w]
        src_patches = src_patches.permute(0, 2, 3, 1, 4, 5)     # [B, num_patches_h, num_patches_w, C, patch_h, patch_w]
        src_patches = src_patches.reshape(batch_size, -1, p_size * p_size)  # [batch_size, time, feature]
        src_embed = self.mlp_embeding(src_patches)              # [batch_size, time, feature]


        # b. 准备 Decoder 输入
        tgt_embed = self.tgt_embedding(tgt)

        test_mask = self.make_mask(tgt, True)
        
        # c. 送入 Transformer
        # 在这个例子中,无填充掩码
        output = self.transformer(
            src_embed,
            tgt_embed,
            tgt_mask = test_mask
        )
        
        # d. 输出预测
        prediction = self.output_layer(output) # -> [1, batch, num_classes]
        
        return prediction # -> [batch, num_classes]