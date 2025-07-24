# 文件名: generate_data.py

import torch
import random
import numpy as np
from torchvision.utils import save_image
from PIL import Image, ImageDraw, ImageFont

class CharImageGenerator:
    """
    一个能生成带有随机变换的单个字符图片的生成器。
    """
    def __init__(self, char_list, font_path, image_size):
        self.char_list = char_list
        self.font_path = font_path
        self.image_size = image_size
        self.len_chars = len(char_list)
        self.pad_idx = 0                    # 空白标记
        self.sos_idx = 1                    # 开始标记
        self.eos_idx = 2                    # 结束标记
        
        # 创建字符到索引的映射
        self.char_to_idx = {ch: i + 3 for i, ch in enumerate(char_list)}
        self.idx_to_char = {i + 3: ch for i, ch in enumerate(char_list)}

    def generate_image(self, char_idx):
        """生成单个字符图片"""
        char = self.char_list[char_idx]
        
        # 随机化参数
        font_size = random.randint(self.image_size[0] // 2, self.image_size[0] - 10)
        rotation_angle = random.randint(-45, 45)
        offset_x = random.randint(-self.image_size[0] // 8, self.image_size[0] // 8)
        offset_y = random.randint(-self.image_size[1] // 8, self.image_size[1] // 8)

        # 创建一个大的画布以容纳旋转后的字符
        canvas_size = int(self.image_size[0] * 1.5)
        try:
            font = ImageFont.truetype(self.font_path, font_size)
        except IOError:
            print(f"错误: 无法加载指定字体文件于 '{self.font_path}'。请确保文件存在。")
            print(f"尝试载入默认字体文件")
            # 使用默认字体作为后备
            font = ImageFont.load_default()

        # 在一个透明背景的大画布上绘制字符
        char_image = Image.new('L', (canvas_size, canvas_size), 0)
        draw = ImageDraw.Draw(char_image)
        
        text_bbox = draw.textbbox((0, 0), char, font=font)
        text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        
        position = ((canvas_size - text_width) / 2, (canvas_size - text_height) / 2)
        draw.text(position, char, font=font, fill=255) # 填充为白色 (255)
        
        char_image = char_image.rotate(rotation_angle, resample=Image.BICUBIC)
        
        # 创建最终的黑色背景画布
        final_image = Image.new('L', self.image_size, 0) # 背景为黑色 (0)
        
        paste_x = (self.image_size[0] - char_image.width) // 2 + offset_x
        paste_y = (self.image_size[1] - char_image.height) // 2 + offset_y
        
        final_image.paste(char_image, (paste_x, paste_y))
        
        return final_image

    def get_batch(self, batch_size):
        """生成一个批次的数据"""
        images = []
        labels = []
        for _ in range(batch_size):
            char_idx = random.randint(0, self.len_chars - 1)
            image = self.generate_image(char_idx)
            
            image_np = np.array(image, dtype=np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np).unsqueeze(0)
            images.append(image_tensor)
            # labels.append(char_idx)
            labels.append([1, char_idx + 3, 2])
            
        return torch.stack(images), torch.LongTensor(labels)

# --- 使用示例 ---
if __name__ == '__main__':
    # chars = "0123456789"
    # chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    chars = "天地玄黄宇宙洪荒日月盈昃辰宿列张寒来暑往秋收冬藏" # 进阶版
    font_file = '.\\font\\simfang.ttf' # 确保你有一个名为 font.ttf 的字体文件

    char_to_idx = {ch: idx + 3 for idx, ch in enumerate(chars)}
    idx_to_char = {idx + 3: ch for idx, ch in enumerate(chars)}
    
    generator = CharImageGenerator(list(chars), font_file, (64,64))
    img_batch, lbl_batch = generator.get_batch(4)

    for idx, ch in enumerate(lbl_batch):
        lbl_batch_n = lbl_batch[idx, 1].item()
        true = idx_to_char[lbl_batch_n]
        print(true)

    print(lbl_batch)
    print(f"Image batch shape: {img_batch.shape}")
    print(f"Label batch shape: {lbl_batch}")
    
    save_image(img_batch, ".\\img\\sample_batch.png", nrow=4)
    print("已保存一个批次的示例图片到 sample_batch.png")