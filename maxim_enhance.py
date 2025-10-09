import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image
import sys, os

input_path = sys.argv[1]
output_path = sys.argv[2]

# Load model
print("🚀 Loading MAXIM (Enhance) model from TensorFlow Hub...")
model = hub.load("https://tfhub.dev/sayakpaul/maxim_enhancement_s-3/1")


# Load and preprocess image
img = Image.open(input_path).convert("RGB")
w, h = img.size
# Resize safely for GPU memory
max_dim = 2048
scale = min(max_dim / max(w, h), 1.0)
new_size = (int(w * scale), int(h * scale))
if scale < 1.0:
    print(f"Resizing from {w}x{h} → {new_size}")
    img = img.resize(new_size, Image.LANCZOS)

x = np.array(img, dtype=np.float32) / 255.0
x = np.expand_dims(x, axis=0)

# Run inference
print("✨ Running enhancement...")
y = model(x)
out = np.clip(y[0].numpy() * 255.0, 0, 255).astype(np.uint8)

# Save result
os.makedirs(os.path.dirname(output_path), exist_ok=True)
Image.fromarray(out).save(output_path)
print(f"✅ Saved enhanced image → {output_path}")
