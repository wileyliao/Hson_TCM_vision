from PIL import Image
import torch
from torchvision import models, transforms
import io

idx_to_label = {
    0: "乾薑飲片(C0038-1)", 1: "杜仲(炒)飲片(C0022-1)", 2: "桑葉飲片(C0035-1)",
    3: "生地黃飲片(C0012-1)", 4: "白朮(炒)飲片(C0013-1)", 5: "白芍(炒)飲片(C0014-1)",
    6: "白芨飲片(C0076-1)", 7: "白芷飲片(C0015-1)", 8: "白茅根飲片(C0063-1)", 9: "黃芩飲片(C0044-1)"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型載入（初始化一次）
model = models.mobilenet_v2()
model.classifier[1] = torch.nn.Linear(model.last_channel, len(idx_to_label))
model.load_state_dict(torch.load("model/model_無乾薑.pth", map_location=device))
model.eval().to(device)

# 處理流程
class PadToSquare:
    def __call__(self, image):
        w, h = image.size
        max_dim = max(w, h)
        pad_w = (max_dim - w) // 2
        pad_h = (max_dim - h) // 2
        padding = (pad_w, pad_h, max_dim - w - pad_w, max_dim - h - pad_h)
        return transforms.functional.pad(image, padding, fill=0, padding_mode='constant')

transform = transforms.Compose([
    PadToSquare(),
    transforms.Resize((640, 640)),
    transforms.ToTensor()
])

def predict_herb_image(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        prob = torch.nn.functional.softmax(output, dim=1)[0]
        pred_idx = torch.argmax(prob).item()
        return {
            "label": idx_to_label[pred_idx],
            "confidence": f"{prob[pred_idx].item() * 100:.2f}%"
        }
