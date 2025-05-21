from flask import Flask, request, jsonify
import base64
import time
from tcm_main import predict_herb_image

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

        result = predict_herb_image(image_bytes)
        return jsonify({
            "Data": [{
                "GUID": guid,
                "Prediction": result["label"],
                "Confidence": result["confidence"]
            }],
            "Code": 200,
            "TimeTaken": f"{time.time() - start_time:.2f}秒"
        })

    except Exception as e:
        return jsonify({"Code": 500, "Error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3005, debug=True)
