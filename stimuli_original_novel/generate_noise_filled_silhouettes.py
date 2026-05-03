import cv2
import numpy as np
import os

def create_noise_filled_stimulus(image_path, thickness=1):
    # 1. Load the mask
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    
    # 2. Threshold (Object = 255)
    _, mask = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    h, w = mask.shape
    
    # 3. Generate White Noise (0-255)
    # We create a 3-channel noise to keep it consistent for ResNet
    noise = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    
    # 4. Create a White Background (3 channels)
    final_image = np.ones((h, w, 3), dtype=np.uint8) * 255
    
    # 5. Mask the Noise into the Silhouette
    # Everywhere the mask is 255, we put the noise
    final_image[mask == 255] = noise[mask == 255]
    
    return final_image

# --- Path Setup ---
root_dir = "./path/to/your/silhouette/stimuli/folder/"
output_root = "./path/to/your/output/noise/stimuli/folder/"

if not os.path.exists(output_root):
    os.makedirs(output_root)

for cat in os.listdir(root_dir):
    cat_path = os.path.join(root_dir, cat)
    if not os.path.isdir(cat_path): continue
    
    out_cat_path = os.path.join(output_root, cat)
    os.makedirs(out_cat_path, exist_ok=True)
    
    for filename in os.listdir(cat_path):
        if filename.lower().endswith('.png'):
            in_file = os.path.join(cat_path, filename)
            noise_img = create_noise_filled_stimulus(in_file)
            
            if noise_img is not None:
                cv2.imwrite(os.path.join(out_cat_path, filename), noise_img)

print(f"Noise-filled stimuli generated in: {output_root}")