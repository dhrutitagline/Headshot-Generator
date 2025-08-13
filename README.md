# AI Business Headshot Generator

This project generates **professional corporate headshots** from your uploaded image using **IP-Adapter FaceID** and **Stable Diffusion**.  
It detects your face, extracts facial embeddings, and produces a **realistic business headshot** with various suit, pose, and background combinations.

---

## 🚀 Features
- Automatic face detection using **InsightFace**
- **IP-Adapter FaceID** for identity preservation
- **Stable Diffusion** for photorealistic generation
- Randomized suit colors, poses, and backgrounds
- Gradio web interface for easy usage
- Works on **Local PC** and **Google Colab GPU**

---

## 📦 Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/ai-business-headshot.git
cd ai-business-headshot
```

### 2️⃣ Install Requirements
```bash
pip install -r requirements.txt
```

### 📥 IP-Adapter FaceID Model
The file ip-adapter-faceid_sd15.bin is required.
The script will automatically download the model if it’s not found in the folder.

If automatic download fails, download it manually:
#### 🔹 Local PC
```bash
wget https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sd15.bin
```

#### 🔹 Google Colab (GPU)
```bash
!wget https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sd15.bin
```

The file should be placed in the same directory as main.py.

### ▶️ Run the Application
#### Local PC
```bash
python main.py
```
#### Google Colab
1. Open your Colab notebook
2. Enable GPU:
Runtime → Change runtime type → Hardware accelerator → GPU
3. Clone the repo, install requirements, and run:
```bash
!git clone https://github.com/yourusername/ai-business-headshot.git
%cd ai-business-headshot
!pip install -r requirements.txt
!python main.py
```

### 📂 Project Structure
```bash
ai-business-headshot/
│-- main.py                  # Main script with Gradio interface
│-- requirements.txt         # Python dependencies
│-- ip-adapter-faceid_sd15.bin  # Model file (auto/manual download)
│-- README.md                # This file
```
