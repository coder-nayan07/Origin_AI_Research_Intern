import torch, cv2, os, json, numpy as np
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor
from PIL import Image

# 1. Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)
model.load_state_dict(torch.load("final_model.pth"))
model.eval()

# 2. Rebuild the Isolated Map to match the training logic
path_map = {}
def map_json(json_path, img_dir, prefix):
    if not os.path.exists(json_path): return
    with open(json_path, 'r') as f:
        data = json.load(f)
    for img_obj in data['images']:
        # Ensure we use the exact same unique key as the mask filenames
        unique_key = f"{prefix}_{img_obj['id']}"
        path_map[unique_key] = os.path.join(img_dir, img_obj['file_name'])

map_json('Drywall-Join-Detect-2/valid/_annotations.coco.json', 'Drywall-Join-Detect-2/valid', 'drywall')
map_json('cracks-1/valid/_annotations.coco.json', 'cracks-1/valid', 'crack')

def save_visual_strip(unique_id, prompt, mask_dir, out_name):
    img_path = path_map.get(unique_id)
    if not img_path: 
        print(f"Skipping {unique_id}: Path not found.")
        return
    
    # Load original image
    image = Image.open(img_path).convert("RGB")
    
    # Model Inference
    inputs = processor(text=[prompt], images=[image], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Process prediction (Squeeze and Resize)
    pred_raw = torch.sigmoid(outputs.logits.squeeze()).cpu().numpy()
    pred_mask = cv2.resize((pred_raw > 0.3).astype(np.uint8) * 255, (image.size[0], image.size[1]), interpolation=cv2.INTER_NEAREST)
    
    # Load Ground Truth mask (single-channel, {0, 255})
    gt_name = f"{unique_id}__{prompt.replace(' ', '_')}.png"
    gt_path = os.path.join(mask_dir, gt_name)
    gt_mask = cv2.imread(gt_path, 0)
    
    if gt_mask is None:
        print(f"Skipping {unique_id}: GT Mask not found at {gt_path}.")
        return

    # Create Horizontal Strip: orig | GT | pred
    orig_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gt_viz = cv2.cvtColor(gt_mask, cv2.COLOR_GRAY2BGR)
    pred_viz = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2BGR)
    
    # Ensure all images have the same height for hstack
    combined = np.hstack((orig_np, gt_viz, pred_viz))
    cv2.imwrite(out_name, combined)
    print(f"Successfully generated: {out_name}")

# 3. Execution for Report Examples
os.makedirs("visualizations", exist_ok=True)
all_masks = [f for f in os.listdir('data_v/masks') if f.endswith('.png')]

# Correctly filter masks based on the isolated prefixes
crack_ids = [m.split("__")[0] for m in all_masks if m.startswith("crack_")][5:10]
drywall_ids = [m.split("__")[0] for m in all_masks if m.startswith("drywall_")][5:10]

for i, cid in enumerate(crack_ids):
    save_visual_strip(cid, "segment crack", "data_v/masks", f"visualizations/viz_crack_{i}.png")

for i, did in enumerate(drywall_ids):
    save_visual_strip(did, "segment taping area", "data_v/masks", f"visualizations/viz_drywall_{i}.png")