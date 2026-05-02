"""
Extracts penultimate layer embeddings from ResNet50 for a folder of images.

Folder structure expected:
  images/
    airplane/
      img1.png
      img2.png
    bear/
      ...
    ...  (any of the 16 Geirhos categories)

Output:
  normal_images_embeddings.json  <- {filename: [2048-dim vector]}
  normal_images_shape_classes.json <- {filename: {shape, category, dir}}
"""

import os
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from collections import defaultdict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# ── Configuration ──────────────────────────────────────────────────────────────
IMAGE_DIR   = "./path/to/your/stimuli/file/" # root folder containing category subfolders
OUTPUT_EMB  = "./path/to/your/output/file/" # name of json output file
OUTPUT_SC   = "./path/to/your/stimuli/file/" # name of shape classes output file
BATCH_SIZE  = 32

CATEGORIES = [
    'airplane', 'bear', 'bicycle', 'bird', 'boat', 'bottle',
    'car', 'cat', 'chair', 'clock', 'dog', 'elephant',
    'keyboard', 'knife', 'oven', 'truck'
]

# ── Load model ─────────────────────────────────────────────────────────────────
print('Loading ResNet50...')
model = models.resnet50(pretrained=True)
model.eval()

# Strip final FC layer — same as main.py
modules = list(model.children())[:-1]
penult_model = nn.Sequential(*modules).to(device)
penult_model.eval()

# ImageNet transforms — same as main.py
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ── Collect image paths ────────────────────────────────────────────────────────
image_paths = []
shape_classes = {}

for cat in CATEGORIES:
    cat_dir = os.path.join(IMAGE_DIR, cat)
    if not os.path.exists(cat_dir):
        print(f'  WARNING: {cat_dir} not found, skipping')
        continue
    for fname in sorted(os.listdir(cat_dir)):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        full_path = os.path.join(cat_dir, fname)
        rel_dir   = os.path.join(cat, fname)
        image_paths.append((fname, full_path, cat, rel_dir))
        shape_classes[fname] = {
            'shape':    cat,
            'category': cat,
            'dir':      rel_dir,
            'type':     'normal'
        }

print(f'Found {len(image_paths)} images across {len(set(p[2] for p in image_paths))} categories')

# ── Extract embeddings in batches ─────────────────────────────────────────────
embedding_dict = {}

with torch.no_grad():
    for i in range(0, len(image_paths), BATCH_SIZE):
        batch_paths = image_paths[i:i+BATCH_SIZE]
        batch_imgs  = []
        batch_names = []

        for fname, full_path, cat, rel_dir in batch_paths:
            try:
                img = Image.open(full_path).convert('RGB')
                img = transform(img)
                batch_imgs.append(img)
                batch_names.append(fname)
            except Exception as e:
                print(f'  ERROR loading {full_path}: {e}')

        if not batch_imgs:
            continue

        batch_tensor = torch.stack(batch_imgs).to(device)
        embeddings   = penult_model(batch_tensor)
        embeddings   = torch.squeeze(embeddings)

        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)

        for j, name in enumerate(batch_names):
            embedding_dict[name] = embeddings[j].cpu().tolist()

        print(f'  Processed {min(i+BATCH_SIZE, len(image_paths))}/{len(image_paths)}')

# ── Save ───────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUTPUT_EMB), exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_SC),  exist_ok=True)

with open(OUTPUT_EMB, 'w') as f:
    json.dump(embedding_dict, f)
print(f'\nSaved {len(embedding_dict)} embeddings -> {OUTPUT_EMB}')

with open(OUTPUT_SC, 'w') as f:
    json.dump(shape_classes, f, indent=2)
print(f'Saved shape_classes -> {OUTPUT_SC}')
