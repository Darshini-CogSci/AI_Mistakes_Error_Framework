import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from collections import Counter

from torchvision import transforms, models
from torchvision.io import read_image

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# -----------------------------
# CONFIG
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
IMG_SIZE = 224
DEBUG = True

POOLING = np.mean   # use SAME function everywhere

input_stimuli_path = Path("./path/to/your/stimuli/folder/")
cnn_local_weights_path = Path("./path/to/your/resnet50/model/")

output_results_csv = Path("./path/to/your/confidence/score/output/file/")
output_trial_csv = Path("./path/to/your/trial/data/output/file/")
confusion_matrix_path = Path("./path/to/your/confusion/matrix/png/file/")

# -----------------------------
# LOAD HUMAN CATEGORY SYSTEM (SINGLE SOURCE OF TRUTH)
# -----------------------------
import helper.human_categories as hc
from probabilities_to_decision import ImageNetProbabilitiesTo16ClassesMapping

CATEGORIES = hc.get_human_object_recognition_categories()   # authoritative order
HC = hc.HumanCategories()
mapping = ImageNetProbabilitiesTo16ClassesMapping(aggregation_function=POOLING)

print("Using category order:", CATEGORIES)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = models.resnet50(weights=None)
state_dict = torch.load(cnn_local_weights_path, map_location=device)
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model.eval().to(device)

# -----------------------------
# IMAGE TRANSFORM
# -----------------------------

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ConvertImageDtype(torch.float32),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# -----------------------------
# LOAD IMAGES
# -----------------------------
all_images = list(input_stimuli_path.rglob("*.png")) + \
             list(input_stimuli_path.rglob("*.jpg")) + \
             list(input_stimuli_path.rglob("*.jfif")) + \
             list(input_stimuli_path.rglob("*.jpeg"))

print(f"Found {len(all_images)} images.")

# -----------------------------
# RUN INFERENCE
# -----------------------------
results_df_list = []
trial_data_list = []
predictions = []
ground_truths = []
trial_number = 1

for i in tqdm(range(0, len(all_images), BATCH_SIZE), desc="Running inference"):
    batch_paths = all_images[i:i+BATCH_SIZE]
    imgs, info = [], []

    for img_path in batch_paths:
        gt = img_path.parent.name.lower()
        if gt not in CATEGORIES:
            continue

        try:
            img = transform(read_image(str(img_path)))
            imgs.append(img)
            info.append((img_path.name, gt))
        except Exception as e:
            if DEBUG:
                print(f"[SKIP] {img_path}: {e}")

    if not imgs:
        continue

    batch = torch.stack(imgs).to(device)

    with torch.no_grad():
        probs_1000 = F.softmax(model(batch), dim=1).cpu().numpy()

    for (img_name, gt), prob_vec in zip(info, probs_1000):

        # -----------------------------
        # AGGREGATE INTO 16 CATEGORIES (CONSISTENT ORDER)
        # -----------------------------
        broad_probs = {}
        for cat in CATEGORIES:
            indices = HC.get_imagenet_indices_for_category(cat)
            values = prob_vec[indices]
            broad_probs[cat] = float(POOLING(values)) if len(values) else 0.0

        # Normalize
        total = sum(broad_probs.values())
        if total > 0:
            for c in broad_probs:
                broad_probs[c] /= total

        # -----------------------------
        # DECISION (ARGMAX ON SAME DISTRIBUTION)
        # -----------------------------
        pred = max(broad_probs, key=broad_probs.get)
        confidence = broad_probs[pred] * 100

        # 🔒 SAFETY ASSERT (THIS CANNOT FAIL NOW)
        assert pred == mapping.probabilities_to_decision(prob_vec)

        predictions.append(pred)
        ground_truths.append(gt)

        # -----------------------------
        # SAVE ROW
        # -----------------------------
        results_df_list.append({
            "image_name": img_name,
            "ground_truth_category": gt,
            "predicted_category": pred,
            "confidence_percent": confidence,
            **{f"prob_{c}": broad_probs[c] for c in CATEGORIES}
        })

        trial_data_list.append({
            "subj": "resnet50",
            "session": 1,
            "trial": trial_number,
            "rt": np.nan,
            "object_response": pred,
            "category": gt,
            "condition": np.nan,
            "imagename": img_name
        })

        trial_number += 1

# -----------------------------
# SAVE FILES
# -----------------------------
pd.DataFrame(results_df_list).to_csv(output_results_csv, index=False)
pd.DataFrame(trial_data_list).to_csv(output_trial_csv, index=False)

print("Saved FIXED CSVs")

# -----------------------------
# ACCURACY
# -----------------------------
acc = np.mean([p == g for p, g in zip(predictions, ground_truths)])
print(f"Top-1 accuracy (16-cat): {acc:.4f}")
print("Most common predictions:", Counter(predictions).most_common(10))

# -----------------------------
# CONFUSION MATRIX (Synced Reversed Y-Axis)
# -----------------------------
cm = confusion_matrix(ground_truths, predictions, labels=CATEGORIES, normalize='true')

# 1. Transpose as per your original preference (True on X, Pred on Y)
cm = cm.T

# 2. REVERSE the matrix rows so the data matches the inverted Y-axis
cm_reversed = cm[::-1, :] 

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(cm_reversed, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)

# 3. Use the inverted labels
REVERSED_CATEGORIES = CATEGORIES[::-1]

ax.set(xticks=np.arange(len(CATEGORIES)),
       yticks=np.arange(len(CATEGORIES)),
       xticklabels=CATEGORIES, 
       yticklabels=REVERSED_CATEGORIES, 
       title='ResNet50: Error Patterns on Edge Stimuli',
       xlabel='True label (Human Category)',
       ylabel='Predicted label (Machine Decision)')

plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# 4. Annotate using the reversed matrix
fmt = '.2f'
thresh = cm_reversed.max() / 2.
for i in range(len(CATEGORIES)):
    for j in range(len(CATEGORIES)):
        ax.text(j, i, format(cm_reversed[i, j], fmt),
                ha="center", va="center",
                color="white" if cm_reversed[i, j] > thresh else "black")

fig.tight_layout()
plt.savefig(confusion_matrix_path, dpi=300)