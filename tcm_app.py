from flask import Flask, request, jsonify
from PIL import Image
import torch
from torchvision import transforms, models
import base64
import io
import time
import os

# === 固定 label 對應 ===
idx_to_label = {
    0: "乾薑飲片(C0038-1)",
    1: "杜仲(炒)飲片(C0022-1)",
    2: "桑葉飲片(C0035-1)",
    3: "生地黃飲片(C0012-1)",
    4: "白朮(炒)飲片(C0013-1)",
    5: "白芍(炒)飲片(C0014-1)",
    6: "白芨飲片(C0076-1)",
    7: "白芷飲片(C0015-1)",
    8: "白茅根飲片(C0063-1)",
    9: "黃芩飲片(C0044-1)"
}

# === 圖像補正與處理 ===
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

# === 模型載入 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.mobilenet_v2()
model.classifier[1] = torch.nn.Linear(model.last_channel, len(idx_to_label))
model.load_state_dict(torch.load("model/model.pth", map_location=device))
model.eval().to(device)

# === Flask app ===
app = Flask(__name__)

@app.route("/herb_predict", methods=["POST"])
def predict():
    start_time = time.time()
    try:
        data = request.json["Data"][0]
        guid = data.get("GUID", "Unknown")
        b64 = data["base64"]

        if b64.startswith("data:image"):
            b64 = b64.split(",", 1)[1]

        image_bytes = base64.b64decode(b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)
            probs = torch.nn.functional.softmax(output, dim=1)[0]
            pred_idx = torch.argmax(probs).item()
            pred_label = idx_to_label[pred_idx]
            confidence = probs[pred_idx].item()

        result = {
            "GUID": guid,
            "Prediction": pred_label,
            "Confidence": f"{confidence * 100:.2f}%"
        }

        return jsonify({
            "Data": [result],
            "Code": 200,
            "TimeTaken": f"{time.time() - start_time:.2f}秒"
        })

    except Exception as e:
        return jsonify({"Code": 500, "Error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3005, debug=True)
