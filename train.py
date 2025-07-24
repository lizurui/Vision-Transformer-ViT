# 文件名: train.py

import torch
import torch.nn as nn
import torch.optim as optim
import generate_data
from model import ImageToCharViT
import time
import os
import numpy as np
import pred
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# --- 1. 配置 ---
# 尝试更复杂的字符集
# CHARS = "0123456789"                                              # 青铜
# CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"                    # 白银
# CHARS = "天地玄黄宇宙洪荒日月盈昃辰宿列张寒来暑往秋收冬藏"            # 黄金
CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789天地玄黄宇宙洪荒日月盈昃辰宿列张寒来暑往秋收冬藏"  # 王者
FONT_PATH = '.\\font\\simfang.ttf'
pad_idx = 0                                                         # 空白标记
sos_idx = 1                                                         # 开始标记
eos_idx = 2                                                         # 结束标记
NUM_CLASSES = len(CHARS) + 3
epoch = 1000
# 训练参数
BATCH_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 确保你的字体文件存在
def prepare():
    print(f"Using device: {DEVICE}")
    print("正在初始化...")
    if not os.path.exists(FONT_PATH):
        raise FileNotFoundError(f"字体文件未找到: {FONT_PATH}. 请下载一个 ttf 字体并放到项目目录中。")



# --- 3. 训练循环 ---
def train_fun(model, optimizer, loss_fun, loader, start_epoch):
    print("进入训练循环")
    model.train()
    start_time = time.time()
    for step in range(start_epoch, epoch):
        images, labels = loader.get_batch(BATCH_SIZE)
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        tgt_input = labels[:, :-1]
        
        optimizer.zero_grad()
        output = model(images, tgt_input)

        output_flat = output.reshape(-1, output.shape[-1])
        tgt_output = labels[:, 1:]
        tgt_output_flat = tgt_output.reshape(-1)
        loss = loss_fun(output_flat, tgt_output_flat)
        
        loss.backward()
        optimizer.step()
    
        if (step + 1) % 100 == 0:
            elapsed_time = time.time() - start_time
            print(f"Step [{step+1}/{epoch}], Loss: {loss.item():.4f}, steps: {elapsed_time:.2f}s")
            start_time = time.time()

        save_model(model, optimizer, step, check_path)

def eval(model, loader):
    print("进入测试循环")
    model.eval()
    correct = 0
    total = 100 # 测试100个样本
    with torch.no_grad():
        for i in range(total):
            test_image, test_label_idx = loader.get_batch(1)
            test_image = test_image.to(DEVICE)
            tgt_tensor = torch.tensor([sos_idx], dtype=torch.long).unsqueeze(0).to(DEVICE)

            true_idx = test_label_idx[0, 1].item()
            true_char = loader.idx_to_char[true_idx]
            
            output = model(test_image, tgt_tensor)
            predicted_idx = output.squeeze().argmax().item()
            predicted_char = loader.idx_to_char.get(predicted_idx, '?')
            
            if predicted_char == true_char:
                correct += 1
            
            if i < 5: # 只打印前5个例子
                print(f"  测试样本 {i+1}: 真实值='{true_char}', 预测值='{predicted_char}' {'✅' if predicted_char == true_char else '❌'}")


    accuracy = 100 * correct / total
    print(f"\n测试准确率: {accuracy:.2f}% ({correct}/{total})")

def save_model(model, optimizer, epoch, dir):

    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, dir)

def load_model(dir, imgSize):
    start_epoch = 0
    model = ImageToCharViT(NUM_CLASSES, imgSize).to(DEVICE)
    loss_fun = nn.CrossEntropyLoss(ignore_index = 0)
    optimizer = optim.AdamW(model.parameters(), lr = 0.0003, weight_decay=0.01)

    if os.path.exists(dir):
        print(f"Loading Checkpoint From {dir}")
        checkpoint = torch.load(dir, map_location=DEVICE, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"--- Starting training from {start_epoch} ---")
    else:
        print("--- Starting training from scratch ---")

    return model, optimizer, loss_fun, start_epoch



if __name__=="__main__":
    check_path = ".\\pth\\char_vit_model"
    imgSize = (64, 64)
    print("图片大小为64 * 64")
    print(f"输出维度为:{NUM_CLASSES}")
    prepare()
    data_generator = generate_data.CharImageGenerator(list(CHARS), FONT_PATH, imgSize)
    model, optimizer, loss_fun, start_epoch = load_model(check_path, 64)
    train_fun(model, optimizer, loss_fun, data_generator, start_epoch)
    
    eval(model, data_generator)

    pred.test_one_char(model, data_generator, CHARS[5])