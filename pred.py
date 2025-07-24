import torch
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def test_one_char(model, data_generator, char_to_test):
    print(f"\n--- 实时测试单个字符: '{char_to_test}' ---")
    model.eval()
    with torch.no_grad():
        if char_to_test not in data_generator.char_to_idx:
            print("字符不在词汇表中！")
            return
            
        char_idx = data_generator.char_to_idx[char_to_test]
        image_pil = data_generator.generate_image(char_idx - 3)
        image_pil.save(".\\img\\realtime_test.png")
        print("生成的测试图片已保存到 realtime_test.png")
        
        image_np = np.array(image_pil, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).unsqueeze(0).unsqueeze(0).to(DEVICE)

        sos_token = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
        tgt_tensor = torch.tensor([1], dtype=torch.long).unsqueeze(0).to(DEVICE)

        output = model(image_tensor, tgt_tensor)
        predicted_idx = output.squeeze()
        predicted_idx = predicted_idx.argmax(0).item()
        predicted_char = data_generator.idx_to_char.get(predicted_idx, '?')
        
        print(f"模型预测: '{predicted_char}'")