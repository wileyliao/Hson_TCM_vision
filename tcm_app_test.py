import requests
import json
import base64

# 圖片轉 base64
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        b64 = base64.b64encode(img_file.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

# API URL
url = "http://localhost:3005/herb_predict"

# 測試圖片路徑
image_path = r"C:\pycharm\tcm\augmented_highres_seq_test\黃芩飲片(C0044-1)_009.jpg"  # ⚠️ 替換為你的圖片路徑
base64_img = encode_image_to_base64(image_path)

payload = {
    "Data": [
        {
            "GUID": "herb-test-001",
            "base64": base64_img
        }
    ]
}

headers = {"Content-Type": "application/json"}

response = requests.post(url, headers=headers, data=json.dumps(payload))

if response.status_code == 200:
    print("✅ 回傳結果：")
    print(json.dumps(response.json(), indent=4, ensure_ascii=False))
else:
    print("❌ API 錯誤：", response.status_code)
