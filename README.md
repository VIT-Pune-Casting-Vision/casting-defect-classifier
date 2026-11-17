# ⚙️ Surface Defect Classifier for Cast Metal Parts

An automated visual quality control web application that detects surface defects in manufactured casting parts using a deep learning model exported to **ONNX**.

🔗 **Live Demo:** `https://<your-username>.github.io/<repo-name>/` *(Update after enabling GitHub Pages)*

---

## 📋 Overview

Manual visual inspection of cast metal components is time-consuming and prone to human fatigue. This project automates quality assurance by classifying top-view images of submersible pump impeller castings into two categories:
* **OK Part (`QUALITY PASS`)** — Free of surface anomalies.
* **Defective Part (`DEFECT DETECTED`)** — Contains casting flaws such as blowholes, pinholes, shrinkage, or rough surfaces.

The repository includes both a **client-side browser version** (for zero-server hosting on GitHub Pages / Netlify) and a **Python Flask backend** for local or server-based deployment.

---

## 🛠️ Tech Stack

* **Machine Learning & Inference:** ONNX Runtime (`onnxruntime` / `onnxruntime-web`), NumPy
* **Backend (Local / Server):** Python, Flask, Pillow (PIL)
* **Frontend (Static / Browser):** HTML5, CSS3, Vanilla JavaScript (HTML5 Canvas for image preprocessing)
* **Dataset:** Casting Product Image Data for Quality Inspection (`150 × 150` normalized input)

---

## 📂 Repository Structure

```text
├── index.html          # Browser-based app using ONNX Runtime Web (GitHub Pages ready)
├── app.py              # Local Python Flask web server using ONNX Runtime
├── model.onnx          # Trained CNN classification model in ONNX format
├── requirements.txt    # Python dependencies for local execution
└── README.md           # Project documentation
