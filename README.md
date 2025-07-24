# ✍️ 基于 Vision Transformer (ViT) 的字符图像识别
# ✍️ Character Image Recognition with Vision Transformer (ViT)

## 项目简介与核心特点
## Project Introduction & Key Features

本项目实现了一个基于 Vision Transformer (ViT) 的单个字符图像识别系统。我们致力于提供一个极度友好、易于上手且资源要求低的机器学习项目，尤其适合初学者和对 Transformer 架构不甚熟悉的用户。

This project implements a single character image recognition system based on the Vision Transformer (ViT) architecture. Our goal is to provide an extremely user-friendly, easy-to-get-started, and low-resource machine learning project, especially suitable for beginners and those less familiar with Transformer architectures.

**核心优势 (Core Advantages):**

*   **极简启动 (Minimal Setup):** 无需手动准备数据集，项目内置高效的字符图像数据生成器，开箱即用。
    *   No manual dataset preparation needed; the project includes an efficient character image data generator, ready to use out-of-the-box.
*   **资源友好 (Resource-Friendly):** 对硬件资源要求不高，在普通笔记本电脑上也能顺利运行训练和推理。
    *   Low hardware requirements, allowing smooth training and inference on a standard laptop.
*   **上手迅速 (Quick Start):** 代码结构清晰，依赖项极少，可快速理解并运行。
    *   Clear code structure with minimal dependencies, enabling quick understanding and execution.
*   **Transformer 核心实践 (Core Transformer Practice):** 提供相对完整且模块化的 Transformer 核心组件实现（如多头注意力、编码器、解码器），是学习 Transformer 工作原理的绝佳实践。
    *   Offers a relatively complete and modular implementation of core Transformer components (e.g., Multi-Head Attention, Encoder, Decoder), serving as an excellent practical guide for understanding how Transformers work.
*   **内置数据增强 (Built-in Data Augmentation):** 数据生成器在生成图像时自动加入随机字体大小、旋转和位置偏移，提高了模型对字符形变的泛化能力，无需复杂外部库。
    *   The data generator automatically incorporates random font sizes, rotations, and positional offsets during image creation, enhancing the model's generalization to character variations without needing complex external libraries.
*   **高度可定制 (Highly Customizable):** 轻松修改字符集（数字、字母、汉字甚至更多），适应各种识别需求。
    *   Easily modify the character set (numbers, letters, Chinese characters, or more) to adapt to various recognition needs.
*   **断点续训支持 (Checkpointing Support):** 训练过程支持模型检查点保存与加载，方便中断后继续训练。
    *   Training supports saving and loading model checkpoints, allowing continuation after interruptions.
*   **纯 PyTorch 实现 (Pure PyTorch Implementation):** 核心逻辑基于 PyTorch 构建，没有引入额外的复杂依赖，保持项目轻量化。
    *   Core logic built purely on PyTorch, without introducing extra complex dependencies, keeping the project lightweight.

## 功能概览
## Feature Overview

*   **字符图像数据生成 (Character Image Data Generation):** 自动生成带有随机变换（大小、旋转、位置）的单字符黑白图像。
    *   Automatically generates single character grayscale images with random transformations (size, rotation, position).
    *   
*   **Vision Transformer (ViT) 编码器 (Vision Transformer (ViT) Encoder):** 将输入图像分割成小块，展平并投影为序列嵌入，作为 Transformer 编码器的输入。
    *   Divides input images into patches, flattens them, and projects them into sequential embeddings for the Transformer encoder.
    *   
*   **Transformer 解码器 (Transformer Decoder):** 基于编码器输出的图像特征，结合自回归机制，预测目标字符序列（`<SOS>` -> `字符` -> `<EOS>`）。
    *   Predicts the target character sequence (`<SOS>` -> `character` -> `<EOS>`) based on the encoder's output image features, using an auto-regressive mechanism.
    *   
*   **模型训练与评估 (Model Training & Evaluation):** 提供完整的训练循环和简单的评估准确率计算。
    *   Provides a complete training loop and simple accuracy evaluation.
    *   
*   **实时单字符预测 (Real-time Single Character Prediction):** 训练完成后，可即时生成并预测指定字符的图像。
    *   After training, instantly generates and predicts the image of a specified character.

## 快速开始
## Quick Start

为了能够顺利运行项目，请确保你已经安装了 Python (建议 3.8+) 和 pip。

To run the project smoothly, please ensure you have Python (3.8+ recommended) and pip installed.

### 1. 准备字体文件
### 1. Prepare Font File

本项目需要一个 `.ttf` 格式的字体文件来生成训练图像。请下载一个字体文件（例如，可以搜索 "simfang.ttf" 或 "方正仿宋简体" 下载）并将其放入 `font/` 目录下。

This project requires a `.ttf` format font file to generate training images. Please download a font file (e.g., search for "simfang.ttf" or "方正仿宋简体" and download it) and place it into the `font/` directory.

### 2. 安装依赖
### 2. Install Dependencies

使用 pip 安装所有必需的库：
Install all required libraries using pip:

pip install torch torchvision numpy Pillow

### 3. 训练模型
### 3. Train the Model
运行 train.py 脚本开始训练。脚本会自动生成训练数据，并在 pth/ 目录下保存模型检查点。
Run the train.py script to start training. The script will automatically generate training data and save model checkpoints in the pth/ directory.

### 4. 进行预测
### 4. Perform Prediction
训练结束后，train.py 会自动调用 pred.test_one_char 函数进行一次实时单字符预测。你也可以在 train.py 的最后修改 CHARS[5] 为你想要测试的字符，例如 pred.test_one_char(model, data_generator, "天")，然后再次运行 train.py。生成的测试图片将保存到 img/realtime_test.png。
After training, train.py will automatically call pred.test_one_char to perform a real-time single character prediction. You can also modify CHARS[5] at the end of train.py to the character you want to test, for example, pred.test_one_char(model, data_generator, "天"), and then run train.py again. The generated test image will be saved to


### 核心技术亮点
### Core Technical Highlights

图像分块与嵌入 (Image Patching & Embedding): 项目将图像视为一系列独立的图像块，通过线性投影将其转换为 Transformer 可处理的序列，这是 Vision Transformer 的核心思想，无需复杂的 CNN 特征提取。

The project treats images as sequences of independent patches, transforming them into a Transformer-processable sequence via linear projection, which is the core idea of Vision Transformer, without needing complex CNN feature extraction.

完整的 Transformer 架构 (Complete Transformer Architecture): 从位置编码到多头注意力、前馈网络、编码器层和解码器层，本项目提供了 Transformer 模型的完整实现细节，而非仅仅调用高级 API，极大地促进了学习。

From positional encoding to multi-head attention, feed-forward networks, encoder layers, and decoder layers, this project provides a complete implementation detail of the Transformer model, rather than just calling high-level APIs, greatly facilitating learning.

序列解码的精确控制 (Precise Control of Sequence Decoding): 通过明确的 pad_idx (填充)、sos_idx (开始) 和 eos_idx (结束) 标记，精准控制了 Transformer 解码器的输入和输出序列，符合序列生成任务的最佳实践。

Through explicit pad_idx (padding), sos_idx (start-of-sequence), and eos_idx (end-of-sequence) tokens, the Transformer decoder's input and output sequences are precisely controlled, adhering to best practices for sequence generation tasks.
### 鸣谢
### Acknowledgements
本项目得益于 PyTorch 深度学习框架和 Vision Transformer 与 Attention Is All You Need 论文中提出的核心理念。

This project benefits from the PyTorch deep learning framework and the core concepts presented in the Vision Transformer and Attention Is All You Need papers.

### 作者 / 联系方式
### Author / Contact Information
ytingalp@gmail.com
