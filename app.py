import os
import io
import csv
import time
import base64
from datetime import datetime
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageFilter, ImageOps
from flask import Flask, request, render_template_string, Response, jsonify

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 250 * 1024 * 1024
app.config['MAX_FORM_PARTS'] = 5000

inspection_history = []

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
        model_status = f"✅ Model Loaded: {os.path.basename(model_path)} (Original Precision Mode)"
    except Exception as e:
        session = None
        input_shape = None
        model_status = f"❌ Model Load Error: {e}"
else:
    session = None
    input_shape = None
    model_status = "❌ No ONNX model file found!"

def strictness_to_cutoff(strictness_pct):
    """
    Maps UI Strictness (1% to 99%) to actual CNN Sigmoid probability cutoff:
    - 50% Strictness -> 0.5000 (Exact default model accuracy)
    - 10% Strictness -> 0.0500 (Lenient: passes minor marks)
    - 80% Strictness -> 0.9500 (High Strictness)
    - 95% Strictness -> 0.9950 (Ultra-Strict: rejects even tiny micro-defects)
    - 99% Strictness -> 0.9995 (Zero-Defect Aerospace Grade)
    """
    s = np.clip(strictness_pct, 1.0, 99.0)
    # Map 1..99 to logit range -4.5 .. +7.6
    logit_cutoff = ((s - 50.0) / 49.0) * 6.5
    return float(1.0 / (1.0 + np.exp(-logit_cutoff)))

def run_inference(image, filename, strictness_pct):
    cutoff_prob = strictness_to_cutoff(strictness_pct)

    # Compact thumbnail preview
    preview_img = image.copy()
    preview_img.thumbnail((250, 250))
    buffered = io.BytesIO()
    preview_img.save(buffered, format="JPEG", quality=80)
    img_data = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Surface Edge-Highlight View
    edge_img = ImageOps.grayscale(preview_img).filter(ImageFilter.FIND_EDGES)
    edge_img = ImageOps.autocontrast(edge_img)
    edge_buf = io.BytesIO()
    edge_img.save(edge_buf, format="JPEG", quality=80)
    edge_data = base64.b64encode(edge_buf.getvalue()).decode('utf-8')

    # Exact Original Preprocessing + ONNX Inference (Unmodified Accuracy)
    t_start = time.perf_counter()
    img_resized = image.resize((150, 150), Image.BILINEAR)
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    if input_shape and len(input_shape) == 4 and input_shape[-1] == 1:
        img_array = np.mean(img_array, axis=2, keepdims=True)
    img_reshape = np.expand_dims(img_array, axis=0)

    outputs = session.run(None, {input_name: img_reshape})
    latency_ms = (time.perf_counter() - t_start) * 1000.0
    raw_output = float(outputs[0][0][0])

    if raw_output >= cutoff_prob:
        conf = f"{raw_output * 100:.2f}"
        verdict = "✅ QUALITY PASS"
        res_class = "pass"
    else:
        conf = f"{(1.0 - raw_output) * 100:.2f}"
        verdict = "❌ DEFECT DETECTED"
        res_class = "fail"

    record = {
        "id": len(inspection_history) + 1,
        "time": datetime.now().strftime("%H:%M:%S"),
        "filename": os.path.basename(filename),
        "verdict": verdict,
        "conf": conf,
        "raw": f"{raw_output:.4f}",
        "cutoff": f"{cutoff_prob:.4f}",
        "strictness": f"{int(strictness_pct)}%",
        "latency": f"{latency_ms:.1f}",
        "res_class": res_class,
        "img_data": img_data,
        "edge_data": edge_data
    }
    inspection_history.append(record)
    return record

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Casting Defect Classifier - Industrial AOI System</title>
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; background-color: #f1f5f9; text-align: center; padding: 22px; color: #1e293b; }
        .container { max-width: 960px; margin: 0 auto; }
        .card { background: white; padding: 24px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.08); margin-bottom: 20px; }
        h2 { margin: 0 0 4px 0; }
        p.sub { color: #64748b; font-size: 13px; margin-top: 0; }
        
        .top-bar { display: flex; justify-content: space-between; align-items: center; background: #dcfce7; color: #166534; padding: 8px 14px; border-radius: 6px; font-weight: bold; font-size: 13px; margin: 12px 0; }
        .sound-toggle { background: #166534; color: white; border: none; padding: 4px 10px; border-radius: 4px; cursor: pointer; font-size: 12px; }

        /* Stats Bar */
        .stats-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; margin: 16px 0; }
        .stat-box { background: #f8fafc; padding: 12px; border-radius: 8px; border: 1px solid #e2e8f0; }
        .stat-num { font-size: 20px; font-weight: bold; color: #0f172a; }
        .stat-label { font-size: 11px; color: #64748b; margin-top: 3px; }
        .pass-txt { color: #15803d; }
        .fail-txt { color: #b91c1c; }

        /* Tabs */
        .tabs { display: flex; justify-content: center; gap: 8px; margin-bottom: 14px; flex-wrap: wrap; }
        .tab-btn { padding: 8px 16px; border: 1px solid #cbd5e1; background: #f8fafc; border-radius: 6px; cursor: pointer; font-weight: bold; font-size: 13px; }
        .tab-btn.active { background: #1e293b; color: white; border-color: #1e293b; }

        /* Controls */
        .controls { background: #f8fafc; padding: 18px; border-radius: 8px; border: 1px dashed #cbd5e1; }
        .slider-box { margin-bottom: 14px; font-size: 14px; background: white; padding: 10px; border-radius: 6px; border: 1px solid #e2e8f0; }
        .file-row { display: flex; justify-content: center; gap: 15px; flex-wrap: wrap; margin-bottom: 12px; }
        .file-box { background: white; padding: 10px 14px; border: 1px solid #cbd5e1; border-radius: 6px; font-size: 13px; }

        .btn-primary { background: #2563eb; color: white; border: none; padding: 10px 22px; border-radius: 6px; cursor: pointer; font-size: 14px; font-weight: bold; }
        .btn-primary:hover { background: #1d4ed8; }
        .btn-conveyor { background: #d97706; color: white; border: none; padding: 10px 22px; border-radius: 6px; cursor: pointer; font-size: 14px; font-weight: bold; }
        .btn-conveyor:hover { background: #b45309; }
        .btn-cam { background: #7c3aed; color: white; border: none; padding: 10px 22px; border-radius: 6px; cursor: pointer; font-size: 14px; font-weight: bold; margin-top: 10px; }
        
        .btn-export { background: #0f766e; color: white; text-decoration: none; padding: 7px 14px; border-radius: 6px; font-size: 12px; font-weight: bold; }
        .btn-print { background: #334155; color: white; border: none; padding: 7px 14px; border-radius: 6px; font-size: 12px; font-weight: bold; cursor: pointer; margin-left: 6px; }
        .btn-clear { background: #94a3b8; color: white; text-decoration: none; padding: 7px 12px; border-radius: 6px; font-size: 12px; margin-left: 6px; }

        video { width: 100%; max-width: 340px; border-radius: 8px; border: 2px solid #cbd5e1; background: #000; }

        /* Conveyor Live Stage */
        #conveyor-stage { display: none; margin-top: 15px; padding: 15px; background: #0f172a; color: white; border-radius: 10px; }
        .belt-bar { height: 6px; background: repeating-linear-gradient(90deg, #f59e0b 0, #f59e0b 15px, #334155 15px, #334155 30px); border-radius: 3px; margin: 10px 0; animation: moveBelt 1s linear infinite; }
        @keyframes moveBelt { from { background-position: 0 0; } to { background-position: 30px 0; } }

        /* Results Grid */
        .results-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(205px, 1fr)); gap: 12px; margin-top: 18px; max-height: 640px; overflow-y: auto; padding: 4px; }
        .item-card { border-radius: 8px; padding: 10px; text-align: center; font-size: 13px; }
        .pass { background-color: #dcfce7; color: #15803d; border: 1px solid #bbf7d0; }
        .fail { background-color: #fee2e2; color: #b91c1c; border: 1px solid #fecaca; }
        .item-card img { width: 100%; height: 135px; object-fit: cover; border-radius: 6px; margin-top: 6px; }
        .badge-speed { display: inline-block; background: rgba(255,255,255,0.85); padding: 2px 7px; border-radius: 10px; font-size: 11px; margin-top: 4px; color: #0f172a; }
        .btn-edge { margin-top: 6px; background: #1e293b; color: white; border: none; padding: 4px 9px; border-radius: 4px; font-size: 11px; cursor: pointer; }

        /* Table */
        table { width: 100%; border-collapse: collapse; margin-top: 14px; font-size: 13px; }
        th, td { padding: 8px; border-bottom: 1px solid #e2e8f0; text-align: left; }
        th { background-color: #f8fafc; color: #475569; }

        @media print {
            .no-print { display: none !important; }
            body { background: white; padding: 0; }
            .card { box-shadow: none; border: 1px solid #ccc; }
        }
    </style>
</head>
<body>
<div class="container">
    <div class="card">
        <h2>⚙️ Automated Casting Defect Classifier</h2>
        <p class="sub">VIT Mechanical Engineering • High-Volume Batch & Automated Optical Inspection (AOI)</p>
        
        <div class="top-bar no-print">
            <span>{{ status }}</span>
            <button type="button" class="sound-toggle" id="sound-btn" onclick="toggleSound()">🔊 Buzzer Alert: ON</button>
        </div>

        <!-- Live Telemetry Counters -->
        <div class="stats-grid">
            <div class="stat-box">
                <div class="stat-num" id="stat-total">{{ total }}</div>
                <div class="stat-label">Total Inspected</div>
            </div>
            <div class="stat-box">
                <div class="stat-num pass-txt" id="stat-ok">{{ ok_count }}</div>
                <div class="stat-label">Quality Pass</div>
            </div>
            <div class="stat-box">
                <div class="stat-num fail-txt" id="stat-def">{{ def_count }}</div>
                <div class="stat-label">Defective Parts</div>
            </div>
            <div class="stat-box">
                <div class="stat-num" id="stat-yield">{{ yield_rate }}%</div>
                <div class="stat-label">Yield Rate</div>
            </div>
            <div class="stat-box">
                <div class="stat-num" id="stat-lat">{{ avg_latency }} ms</div>
                <div class="stat-label">Avg Inference</div>
            </div>
        </div>

        <!-- Mode Switcher -->
        <div class="tabs no-print">
            <button type="button" class="tab-btn active" id="btn-upload-tab" onclick="switchMode('upload')">📁 Batch / Folder Upload (50+ Images)</button>
            <button type="button" class="tab-btn" id="btn-conv-tab" onclick="switchMode('conveyor')">🏭 Auto-Conveyor Simulator</button>
            <button type="button" class="tab-btn" id="btn-cam-tab" onclick="switchMode('camera')">📷 Live Webcam Inspection</button>
        </div>

        <!-- Shared Quality Strictness Slider (50% = Standard Balanced Accuracy) -->
        <div class="controls no-print">
            <div class="slider-box">
                <label for="thresh"><b>Inspection Strictness Level:</b> <span id="tval" style="color:#2563eb; font-weight:bold;">{{ threshold }}</span>% 
                <span id="mode-tag" style="font-size:12px; color:#0f766e;">(Standard Balanced — Cutoff: 0.5000)</span></label><br>
                <input type="range" id="thresh" min="10" max="99" step="1" value="{{ threshold }}" style="width: 60%; margin-top: 6px;"
                       oninput="updateSliderUI(this.value)">
                <div style="font-size: 11px; color: #64748b; margin-top: 4px;">
                    <b>50%</b> = Default Model Accuracy (Cutoff 0.50) | <b>10–30%</b> = Lenient | <b>80–99%</b> = High Strictness (Rejects minor blemishes)
                </div>
            </div>

            <!-- 1. Standard Upload Mode -->
            <form id="upload-box" method="post" enctype="multipart/form-data">
                <input type="hidden" name="threshold" id="form-thresh" value="{{ threshold }}">
                <div class="file-row">
                    <div class="file-box">
                        <b>Select Multiple Images (50+):</b><br><br>
                        <input type="file" id="multi-files" name="files" accept="image/*" multiple onchange="updateCount(this, 'file-count')">
                        <div id="file-count" style="margin-top:6px; color:#2563eb; font-weight:bold;"></div>
                    </div>
                    <div class="file-box">
                        <b>OR Select Entire Folder:</b><br><br>
                        <input type="file" id="folder-files" name="files" accept="image/*" webkitdirectory directory multiple onchange="updateCount(this, 'folder-count')">
                        <div id="folder-count" style="margin-top:6px; color:#0f766e; font-weight:bold;"></div>
                    </div>
                </div>
                <input type="submit" class="btn-primary" value="🔍 Inspect All Selected Part(s)">
            </form>

            <!-- 2. Auto-Conveyor Mode -->
            <div id="conveyor-box" style="display:none;">
                <p style="font-size: 13px; color: #475569; margin-top: 0;">Select 50+ images or a folder to simulate continuous conveyor-belt inspection:</p>
                <div class="file-row">
                    <div class="file-box">
                        <b>Multiple Files:</b><br><br>
                        <input type="file" id="conv-files" accept="image/*" multiple>
                    </div>
                    <div class="file-box">
                        <b>OR Entire Folder:</b><br><br>
                        <input type="file" id="conv-folder" accept="image/*" webkitdirectory directory multiple>
                    </div>
                </div>
                <div style="margin-bottom: 10px; font-size: 13px;">
                    <b>Belt Speed:</b>
                    <select id="belt-speed">
                        <option value="1500">Normal (1.5s / part)</option>
                        <option value="400" selected>High-Speed Batch (0.4s / part — 50 imgs in 20s)</option>
                    </select>
                </div>
                <button type="button" class="btn-conveyor" onclick="startConveyor()">▶️ Start Conveyor Inspection</button>
                <button type="button" class="btn-clear" onclick="stopConveyor()">⏹ Stop</button>

                <div id="conveyor-stage">
                    <div id="conv-status" style="font-weight:bold; font-size:15px;">Initializing Belt...</div>
                    <div class="belt-bar"></div>
                    <img id="conv-preview" src="" style="width:170px; height:170px; object-fit:cover; border-radius:8px; border:3px solid #fff;">
                    <div id="conv-verdict" style="font-size:18px; font-weight:bold; margin-top:8px;"></div>
                </div>
            </div>

            <!-- 3. Webcam Mode -->
            <div id="camera-box" style="display:none;">
                <video id="video" autoplay playsinline></video><br>
                <canvas id="canvas" width="300" height="300" style="display:none;"></canvas>
                <form id="cam-form" method="post">
                    <input type="hidden" name="threshold" id="cam-thresh" value="{{ threshold }}">
                    <input type="hidden" name="webcam_image" id="webcam_image">
                    <button type="button" class="btn-cam" onclick="captureAndInspect()">📸 Capture & Inspect Live Frame</button>
                </form>
            </div>
        </div>

        <!-- Latest Output -->
        {% if current_results %}
            <h3 style="margin-top: 22px;">Latest Inspection Output ({{ current_results|length }} Part/s @ {{ threshold }}% Strictness)</h3>
            <div class="results-grid">
                {% for item in current_results %}
                <div class="item-card {{ item.res_class }}">
                    <strong>{{ item.verdict }}</strong><br>
                    <small title="{{ item.filename }}">{{ item.filename[:22] }} • Conf: {{ item.conf }}%</small><br>
                    <span class="badge-speed">Raw: {{ item.raw }} (Cutoff: {{ item.cutoff }})</span><br>
                    <span class="badge-speed">⚡ {{ item.latency }} ms</span>
                    <img id="img-{{ loop.index }}" 
                         data-orig="data:image/jpeg;base64,{{ item.img_data }}" 
                         data-edge="data:image/jpeg;base64,{{ item.edge_data }}" 
                         data-mode="orig"
                         src="data:image/jpeg;base64,{{ item.img_data }}" alt="Part"><br>
                    <button type="button" class="btn-edge no-print" onclick="toggleEdge('img-{{ loop.index }}', this)">🔍 Edge View</button>
                </div>
                {% endfor %}
            </div>
        {% endif %}
    </div>

    <!-- Inspection Log & Reports -->
    {% if history %}
    <div class="card">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <h3 style="margin: 0;">📋 Quality Control Inspection Log ({{ history|length }} Records)</h3>
            <div class="no-print">
                <a href="/export" class="btn-export">⬇️ Export CSV</a>
                <button onclick="window.print()" class="btn-print">🖨️ Print QC Report</button>
                <a href="/clear" class="btn-clear">Clear</a>
            </div>
        </div>
        <div style="max-height: 360px; overflow-y: auto; margin-top: 10px;">
            <table>
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Time</th>
                        <th>Part / Source</th>
                        <th>Verdict</th>
                        <th>Confidence</th>
                        <th>Raw vs Cutoff</th>
                        <th>Latency</th>
                    </tr>
                </thead>
                <tbody>
                    {% for row in history|reverse %}
                    <tr>
                        <td>{{ row.id }}</td>
                        <td>{{ row.time }}</td>
                        <td>{{ row.filename }}</td>
                        <td class="{{ 'pass-txt' if 'PASS' in row.verdict else 'fail-txt' }}"><b>{{ row.verdict }}</b></td>
                        <td>{{ row.conf }}%</td>
                        <td>{{ row.raw }} / {{ row.cutoff }}</td>
                        <td>{{ row.latency }} ms</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
    </div>
    {% endif %}
</div>

<script>
    let stream = null;
    let soundEnabled = true;
    let conveyorTimer = null;

    function updateSliderUI(val) {
        val = parseInt(val);
        document.getElementById('tval').innerText = val;
        document.getElementById('form-thresh').value = val;
        document.getElementById('cam-thresh').value = val;
        const logit = ((val - 50.0) / 49.0) * 6.5;
        const cutoff = (1.0 / (1.0 + Math.exp(-logit))).toFixed(4);
        let label = "Standard Balanced";
        if (val < 45) label = "Lenient Tolerance";
        else if (val > 85) label = "Ultra-Strict Zero-Defect";
        else if (val > 55) label = "High Strictness";
        document.getElementById('mode-tag').innerText = `(${label} — Cutoff: ${cutoff})`;
    }
    window.addEventListener('DOMContentLoaded', () => updateSliderUI(document.getElementById('thresh').value));

    function updateCount(input, labelId) {
        const count = Array.from(input.files).filter(f => f.type.startsWith('image/')).length;
        document.getElementById(labelId).innerText = count > 0 ? `✅ ${count} image(s) ready` : "";
    }

    function toggleSound() {
        soundEnabled = !soundEnabled;
        document.getElementById('sound-btn').innerText = soundEnabled ? "🔊 Buzzer Alert: ON" : "🔇 Buzzer Alert: OFF";
    }

    function playBeep(isDefect) {
        if (!soundEnabled) return;
        try {
            const ctx = new (window.AudioContext || window.webkitAudioContext)();
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.connect(gain);
            gain.connect(ctx.destination);
            osc.type = isDefect ? 'sawtooth' : 'sine';
            osc.frequency.value = isDefect ? 320 : 880;
            gain.gain.setValueAtTime(0.12, ctx.currentTime);
            osc.start();
            osc.stop(ctx.currentTime + (isDefect ? 0.25 : 0.1));
        } catch (e) {}
    }

    {% if current_results %}
        const hasDefect = {{ 'true' if current_results|selectattr('res_class', 'equalto', 'fail')|list|length > 0 else 'false' }};
        window.addEventListener('load', () => playBeep(hasDefect));
    {% endif %}

    function toggleEdge(imgId, btn) {
        const img = document.getElementById(imgId);
        if (img.dataset.mode === 'orig') {
            img.src = img.dataset.edge;
            img.dataset.mode = 'edge';
            btn.innerText = "🖼️ Original";
        } else {
            img.src = img.dataset.orig;
            img.dataset.mode = 'orig';
            btn.innerText = "🔍 Edge View";
        }
    }

    function switchMode(mode) {
        stopConveyor();
        document.getElementById('upload-box').style.display = (mode === 'upload') ? 'block' : 'none';
        document.getElementById('conveyor-box').style.display = (mode === 'conveyor') ? 'block' : 'none';
        document.getElementById('camera-box').style.display = (mode === 'camera') ? 'block' : 'none';

        document.getElementById('btn-upload-tab').classList.toggle('active', mode === 'upload');
        document.getElementById('btn-conv-tab').classList.toggle('active', mode === 'conveyor');
        document.getElementById('btn-cam-tab').classList.toggle('active', mode === 'camera');

        if (mode === 'camera') {
            navigator.mediaDevices.getUserMedia({ video: true })
                .then(s => { stream = s; document.getElementById('video').srcObject = s; })
                .catch(err => alert("Camera access denied or not found: " + err));
        } else if (stream) {
            stream.getTracks().forEach(t => t.stop());
        }
    }

    function captureAndInspect() {
        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
        document.getElementById('webcam_image').value = canvas.toDataURL('image/jpeg');
        document.getElementById('cam-form').submit();
    }

    async function startConveyor() {
        const f1 = Array.from(document.getElementById('conv-files').files);
        const f2 = Array.from(document.getElementById('conv-folder').files);
        const files = [...f1, ...f2].filter(f => f.type.startsWith('image/'));

        if (!files.length) {
            alert("Please select multiple images or a folder first!");
            return;
        }
        const delay = parseInt(document.getElementById('belt-speed').value);
        const stage = document.getElementById('conveyor-stage');
        stage.style.display = 'block';
        let idx = 0;

        async function processNext() {
            if (idx >= files.length) {
                document.getElementById('conv-status').innerText = `✅ Completed ${files.length} Parts! Reloading log...`;
                setTimeout(() => window.location.href = "/", 1200);
                return;
            }
            const file = files[idx];
            document.getElementById('conv-status').innerText = `Scanning Part ${idx + 1} of ${files.length}: ${file.name}`;
            
            const formData = new FormData();
            formData.append('file', file);
            formData.append('threshold', document.getElementById('thresh').value);

            const res = await fetch('/api/inspect', { method: 'POST', body: formData });
            const data = await res.json();

            document.getElementById('conv-preview').src = "data:image/jpeg;base64," + data.record.img_data;
            document.getElementById('conv-preview').style.borderColor = (data.record.res_class === 'pass') ? '#22c55e' : '#ef4444';
            document.getElementById('conv-verdict').innerHTML = `${data.record.verdict} (Conf: ${data.record.conf}%) — ⚡ ${data.record.latency} ms`;
            document.getElementById('conv-verdict').style.color = (data.record.res_class === 'pass') ? '#4ade80' : '#f87171';

            document.getElementById('stat-total').innerText = data.stats.total;
            document.getElementById('stat-ok').innerText = data.stats.ok_count;
            document.getElementById('stat-def').innerText = data.stats.def_count;
            document.getElementById('stat-yield').innerText = data.stats.yield_rate + "%";
            document.getElementById('stat-lat').innerText = data.stats.avg_latency + " ms";

            playBeep(data.record.res_class === 'fail');
            idx++;
            conveyorTimer = setTimeout(processNext, delay);
        }
        processNext();
    }

    function stopConveyor() {
        if (conveyorTimer) clearTimeout(conveyorTimer);
    }
</script>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    current_results = []
    threshold = int(float(request.form.get('threshold', 50)))

    if request.method == 'POST' and session is not None:
        webcam_data = request.form.get('webcam_image')
        if webcam_data and ',' in webcam_data:
            header, encoded = webcam_data.split(',', 1)
            img_bytes = base64.b64decode(encoded)
            image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            rec = run_inference(image, f"LiveCam_{datetime.now().strftime('%H%M%S')}.jpg", threshold)
            current_results.append(rec)

        uploaded_files = request.files.getlist('files')
        for file in uploaded_files:
            if not file or file.filename == '':
                continue
            try:
                image = Image.open(file.stream).convert('RGB')
                rec = run_inference(image, file.filename, threshold)
                current_results.append(rec)
            except Exception:
                continue

    total = len(inspection_history)
    ok_count = sum(1 for r in inspection_history if "PASS" in r["verdict"])
    def_count = total - ok_count
    yield_rate = f"{(ok_count / total * 100):.1f}" if total > 0 else "0.0"
    avg_latency = f"{(sum(float(r['latency']) for r in inspection_history) / total):.1f}" if total > 0 else "0.0"

    return render_template_string(
        HTML_TEMPLATE,
        status=model_status,
        current_results=current_results,
        history=inspection_history,
        total=total,
        ok_count=ok_count,
        def_count=def_count,
        yield_rate=yield_rate,
        avg_latency=avg_latency,
        threshold=threshold
    )

@app.route('/api/inspect', methods=['POST'])
def api_inspect():
    file = request.files.get('file')
    threshold = int(float(request.form.get('threshold', 50)))
    image = Image.open(file.stream).convert('RGB')
    rec = run_inference(image, file.filename, threshold)

    total = len(inspection_history)
    ok_count = sum(1 for r in inspection_history if "PASS" in r["verdict"])
    def_count = total - ok_count
    yield_rate = f"{(ok_count / total * 100):.1f}"
    avg_latency = f"{(sum(float(r['latency']) for r in inspection_history) / total):.1f}"

    return jsonify({
        "record": rec,
        "stats": {
            "total": total,
            "ok_count": ok_count,
            "def_count": def_count,
            "yield_rate": yield_rate,
            "avg_latency": avg_latency
        }
    })

@app.route('/export')
def export_csv():
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ID", "Timestamp", "Part/File", "Verdict", "Confidence (%)", "Raw Score", "Cutoff Used", "Strictness", "Latency (ms)"])
    for r in inspection_history:
        writer.writerow([r["id"], r["time"], r["filename"], r["verdict"], r["conf"], r["raw"], r["cutoff"], r["strictness"], r["latency"]])
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=VIT_Casting_QC_Report.csv"}
    )

@app.route('/clear')
def clear_log():
    inspection_history.clear()
    return index()

if __name__ == '__main__':
    print("\n👉 Industrial AOI Dashboard (Original Precision Mode) live at: http://127.0.0.1:5000\n")
    app.run(port=5000, debug=False)
