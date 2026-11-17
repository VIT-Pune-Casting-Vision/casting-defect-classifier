import os
import io
import base64
import numpy as np
import onnxruntime as ort
from PIL import Image
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Smart ONNX Model Locator
def locate_onnx_model():
    target = "casting_defect_onnx_model"
    if os.path.isfile(target):
        return target
    if os.path.isdir(target):
        for root, dirs, files in os.walk(target):
            for f in files:
                if f.endswith('.onnx'):
                    return os.path.join(root, f)
    for root, dirs, files in os.walk("."):
        for f in files:
            if f.endswith('.onnx'):
                return os.path.join(root, f)
    return None

model_path = locate_onnx_model()

if model_path:
    try:
        session = ort.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        model_status = f"✅ Model Loaded: {os.path.basename(model_path)}"
    except Exception as e:
        session = None
        input_shape = None
        model_status = f"❌ Model Load Error: {e}"
else:
    session = None
    input_shape = None
    model_status = "❌ No ONNX model file found!"

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Casting Defect Classifier</title>
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; background-color: #f4f7f6; text-align: center; padding: 40px; }
        .card { background: white; max-width: 520px; margin: 0 auto; padding: 30px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
        h2 { color: #1e293b; margin-bottom: 5px; }
        p { color: #64748b; font-size: 14px; margin-top: 0; }
        .status { padding: 12px; background-color: #e2f0d9; color: #2e6930; border-radius: 6px; font-weight: bold; margin: 20px 0; font-size: 13px; }
        input[type="file"] { margin: 15px 0; }
        input[type="submit"] { background: #2563eb; color: white; border: none; padding: 10px 24px; border-radius: 6px; cursor: pointer; font-size: 15px; font-weight: bold; }
        input[type="submit"]:hover { background: #1d4ed8; }
        .result-box { margin-top: 20px; padding: 15px; border-radius: 8px; font-size: 18px; font-weight: bold; }
        .pass { background-color: #dcfce7; color: #15803d; border: 1px solid #bbf7d0; }
        .fail { background-color: #fee2e2; color: #b91c1c; border: 1px solid #fecaca; }
        .raw-score { font-size: 12px; color: #64748b; margin-top: 8px; }
        img { max-width: 100%; height: auto; border-radius: 8px; margin-top: 15px; border: 1px solid #e2e8f0; }
    </style>
</head>
<body>
    <div class="card">
        <h2>⚙️ Surface Defect Classifier</h2>
        <p>Mechanical Engineering Quality Control Project</p>
        <div class="status">{{ status }}</div>

        <form method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/*" required><br>
            <input type="submit" value="🔍 Inspect Part">
        </form>

        {% if result %}
            <div class="result-box {{ res_class }}">
                {{ result }}
                {% if raw_val %}
                    <div class="raw-score">(Raw Model Output: {{ raw_val }})</div>
                {% endif %}
            </div>
            {% if img_data %}
                <img src="data:image/jpeg;base64,{{ img_data }}" alt="Uploaded Image">
            {% endif %}
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    result = None
    res_class = ""
    img_data = None
    raw_val = None

    if request.method == 'POST':
        file = request.files.get('file')
        if file and session is not None:
            # 1. Image load & Base64 preview
            image = Image.open(file.stream).convert('RGB')
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            img_data = base64.b64encode(buffered.getvalue()).decode('utf-8')

            # 2. Exact Keras Preprocessing Replication (150x150, Rescaling 1./255)
            img_resized = image.resize((150, 150), Image.BILINEAR)
            img_array = np.array(img_resized, dtype=np.float32) / 255.0
            
            # Check if model expects 3-channel (RGB) or 1-channel (Grayscale)
            if input_shape and len(input_shape) == 4 and input_shape[-1] == 1:
                # Convert to Grayscale array if model requires 1 channel
                img_array = np.mean(img_array, axis=2, keepdims=True)

            img_reshape = np.expand_dims(img_array, axis=0)

            # 3. Model Inference
            outputs = session.run(None, {input_name: img_reshape})
            raw_output = float(outputs[0][0][0])
            raw_val = f"{raw_output:.4f}"

            # 4. Standard Kaggle Dataset Threshold Mapping:
            # OK_front = Class 1 (Output >= 0.5)
            # DEF_front = Class 0 (Output < 0.5)
            if raw_output >= 0.5:
                conf = raw_output * 100
                result = f"✅ QUALITY PASS (OK PART) — Confidence: {conf:.2f}%"
                res_class = "pass"
            else:
                conf = (1.0 - raw_output) * 100
                result = f"❌ DEFECT DETECTED (DEFECTIVE PART) — Confidence: {conf:.2f}%"
                res_class = "fail"

    return render_template_string(
        HTML_TEMPLATE, 
        status=model_status, 
        result=result, 
        res_class=res_class, 
        img_data=img_data, 
        raw_val=raw_val
    )

if __name__ == '__main__':
    print("\n👉 Website live at: http://127.0.0.1:5000\n")
    app.run(port=5000, debug=False)
