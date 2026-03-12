import torch, cv2, os, json, numpy as np, random
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor
from PIL import Image

# 1. Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)
model.load_state_dict(torch.load("final_model.pth"))
model.eval()

# 2. Rebuild the map to get the ACTUAL arbitrary filenames
path_map = {}
def map_json(json_path, img_dir):
    if not os.path.exists(json_path): return
    with open(json_path, 'r') as f:
        data = json.load(f)
    for img_obj in data['images']:
        path_map[str(img_obj['id'])] = os.path.join(img_dir, img_obj['file_name'])

map_json('Drywall-Join-Detect-2/train/_annotations.coco.json', 'Drywall-Join-Detect-2/train')
map_json('cracks-1/train/_annotations.coco.json', 'cracks-1/train')

def save_visual_strip(img_id, prompt, mask_dir, out_name):
    img_path = path_map.get(str(img_id))
    if not img_path: return
    
    image = Image.open(img_path).convert("RGB")
    inputs = processor(text=[prompt], images=[image], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    
    pred_raw = torch.sigmoid(outputs.logits.squeeze()).cpu().numpy()
    pred_mask = cv2.resize((pred_raw > 0.3).astype(np.uint8) * 255, (image.size[0], image.size[1]))
    
    gt_name = f"{img_id}__{prompt.replace(' ', '_')}.png"
    gt_mask = cv2.imread(os.path.join(mask_dir, gt_name), 0)
    
    # orig | GT | pred
    orig_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gt_viz = cv2.cvtColor(gt_mask, cv2.COLOR_GRAY2BGR)
    pred_viz = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2BGR)
    
    cv2.imwrite(out_name, np.hstack((orig_np, gt_viz, pred_viz)))
    print(f"Generated {out_name} using real file: {os.path.basename(img_path)}")

# 3. Automatically pick 3 IDs from your generated masks
all_masks = [f for f in os.listdir('data/masks') if f.endswith('.png')]
crack_masks = [m for m in all_masks if "crack" in m][5:10]
drywall_masks = [m for m in all_masks if "taping" in m][5:10]

for i, m in enumerate(crack_masks):
    save_visual_strip(m.split("__")[0], "segment crack", "data/masks", f"visualizations/viz_crack_{i}.png")

for i, m in enumerate(drywall_masks):
    save_visual_strip(m.split("__")[0], "segment taping area", "data/masks", f"visualizations/viz_drywall_{i}.png")