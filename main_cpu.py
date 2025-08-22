import os
import random
import torch
import cv2
import numpy as np
from PIL import Image
import gradio as gr
from diffusers import StableDiffusionPipeline
from diffusers.utils import load_image
from insightface.app import FaceAnalysis
import subprocess
import sys

# -------------------
# Auto-download model if missing
# -------------------
MODEL_URL = "https://huggingface.co/h94/IP-Adapter-FaceID/resolve/main/ip-adapter-faceid_sd15.bin"
MODEL_FILE = "ip-adapter-faceid_sd15.bin"

if not os.path.exists(MODEL_FILE):
    print(f"[INFO] '{MODEL_FILE}' not found. Downloading from Hugging Face...")
    try:
        if sys.platform.startswith("win"):
            subprocess.run(["curl", "-L", MODEL_URL, "-o", MODEL_FILE], check=True)
        else:
            subprocess.run(["curl", "-L", MODEL_URL, "-o", MODEL_FILE], check=True)
        print("[INFO] Model downloaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to download model: {e}")
        sys.exit(1)

# -----------------------------------
# Fix PyTorch 2.6+ UnpicklingError
# -----------------------------------
if not hasattr(torch, "_patched_load"):
    _orig_torch_load = torch.load
    def torch_load_weights_only_false(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return _orig_torch_load(*args, **kwargs)
    torch.load = torch_load_weights_only_false
    torch._patched_load = True

# Import after patch
from ip_adapter.ip_adapter_faceid import IPAdapterFaceID

# -------------------
# Device setup (CPU only)
# -------------------
device = "cpu"
dtype = torch.float32

# -------------------
# InsightFace setup
# -------------------
app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1, det_size=(640, 640))

# -------------------
# Model paths
# -------------------
sd_base = "SG161222/Realistic_Vision_V4.0_noVAE"
ip_ckpt = MODEL_FILE

# -------------------
# Load Stable Diffusion
# -------------------
pipe_sd = StableDiffusionPipeline.from_pretrained(
    sd_base, torch_dtype=dtype, safety_checker=None
).to(device)

# Move models to float32 for CPU
pipe_sd.unet.to(dtype)
pipe_sd.vae.to(dtype)
pipe_sd.text_encoder.to(dtype)

ipai_model = IPAdapterFaceID(pipe_sd, MODEL_FILE, device=device)
ipai_model.pipe.unet.to(dtype)
ipai_model.pipe.vae.to(dtype)
ipai_model.pipe.text_encoder.to(dtype)

# -------------------
# Prompt variations
# -------------------
suit_colors = ["navy blue", "charcoal gray", "black", "light gray", "royal blue"]
backgrounds = [
    "modern glass office", "warm wooden office", "conference room (blurred)",
    "corporate plaza outdoors", "studio backdrop"
]
poses = [
    "arms crossed, smiling confidently",
    "hands in pockets, slight smile, looking to the side",
    "neutral expression with straight posture",
    "leaning forward slightly with engaging smile",
    "standing upright with confident smile, facing camera directly",
    "arms crossed with professional posture, neutral facial expression",
    "standing with hands loosely clasped in front, slight professional smile",
    "slight turn towards the side, hands by sides, confident look",
    "seated at desk, straight posture, hands resting naturally, professional expression"
]

def generate_variation():
    return random.choice(suit_colors), random.choice(backgrounds), random.choice(poses)

# -------------------
# Main generation function
# -------------------
def generate_headshot(upload):
    img = np.array(upload.convert("RGB"))
    faces = app.get(img)
    if not faces:
        return "No face detected in input image.", None

    face = faces[0]
    gender = "male" if face.gender == 1 else "female"
    suit, bg, pose = generate_variation()

    prompt = (
        f"Professional corporate headshot of a confident { 'businessman' if gender=='male' else 'businesswoman' }, "
        f"wearing a tailored {suit} suit, {pose}, standing in a {bg}, "
        "studio-quality lighting, photorealistic, ultra high detail, sharp focus"
    )
    negative_prompt = "blurry, distorted, cartoon, lowres, watermark, bad hands, deformed face"

    embedding = torch.from_numpy(face.normed_embedding).unsqueeze(0).to(dtype=dtype, device=device)
    print(f"[DEBUG] Embedding shape: {embedding.shape}, dtype: {embedding.dtype}")

    width, height = 512, 768
    generated = ipai_model.generate(
        prompt=prompt,
        negative_prompt=negative_prompt,
        faceid_embeds=embedding,
        num_samples=1,
        width=width,
        height=height,
        num_inference_steps=40,
        scale=1.2 if device == "cuda" else 1.0
    )[0]

    output_path = "output_gen.jpg"
    generated.save(output_path)
    return f"Generated via {device.upper()}", output_path

# -------------------
# Gradio UI
# -------------------
iface = gr.Interface(
    fn=generate_headshot,
    inputs=[gr.Image(type="pil", label="Upload your photo")],
    outputs=[
        gr.Text(label="Status"),
        gr.Image(type="filepath", label="Generated Headshot")
    ],
    title="AI Business Headshot Generator",
    description="Upload your image to generate a professional business headshot using IP-Adapter FaceID."
)

if __name__ == "__main__":
    iface.launch(debug=True,share=True)
