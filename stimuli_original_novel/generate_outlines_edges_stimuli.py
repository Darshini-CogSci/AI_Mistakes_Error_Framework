import cv2
import numpy as np
from pathlib import Path

# --- Configuration ---
mask_root = Path("./path/to/your/silhouette/stimuli/folder/")
output_root = Path("./path/to/your/output/outlines/stimuli/folder/")
BORDER_WIDTH = 1  # Exact pixel thickness of the black border
SUPER_SAMPLE = 4  # Internal multiplier for smoother edges

def create_high_quality_outline(mask_path, border_thickness=6):
    # 1. Load the mask
    img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    
    # 2. Upscale for Super-Sampling (makes edges much sharper)
    h, w = img.shape
    img_big = cv2.resize(img, (w * SUPER_SAMPLE, h * SUPER_SAMPLE), interpolation=cv2.INTER_LANCZOS4)
    
    # 3. Threshold to get clean object (Object = 255)
    _, mask = cv2.threshold(img_big, 127, 255, cv2.THRESH_BINARY_INV)
    
    # 4. Distance Transform: Find pixels far from the edge
    # This ensures the "shrink" is uniform regardless of aspect ratio
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    
    # 5. Create the Interior Mask (White part)
    # We keep only pixels that are further than 'border_thickness' from the edge
    interior_mask = np.zeros_like(mask)
    interior_mask[dist_transform > (border_thickness * SUPER_SAMPLE)] = 255
    
    # 6. Create the Base Black Silhouette (Black = 0)
    # We create a white canvas and put the black silhouette on it
    final_big = np.ones((h * SUPER_SAMPLE, w * SUPER_SAMPLE, 3), dtype=np.uint8) * 255
    final_big[mask == 255] = (0, 0, 0) # Black base
    
    # 7. Superimpose the White Interior
    final_big[interior_mask == 255] = (255, 255, 255) # White interior
    
    # 8. Downscale to original size with high-quality interpolation
    # This naturally anti-aliases the edges without using "Blur"
    final_image = cv2.resize(final_big, (w, h), interpolation=cv2.INTER_AREA)
    
    return final_image

# --- Batch Processing ---
if not output_root.exists():
    output_root.mkdir(parents=True)

for cat_folder in mask_root.iterdir():
    if not cat_folder.is_dir(): continue
    out_cat_path = output_root / cat_folder.name
    out_cat_path.mkdir(exist_ok=True)
    
    print(f"Generating high-quality images for: {cat_folder.name}")
    
    for mask_file in cat_folder.glob("*.png"):
        result = create_high_quality_outline(mask_file, border_thickness=BORDER_WIDTH)
        if result is not None:
            new_filename = f"{mask_file.stem}_hq_outline.png"
            cv2.imwrite(str(out_cat_path / new_filename), result)

print(f"High-quality stimuli saved at: {output_root}")