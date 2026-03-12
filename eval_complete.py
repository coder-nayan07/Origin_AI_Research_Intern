import torch, cv2, os, json, numpy as np
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor
from PIL import Image
from tqdm import tqdm

# 1. Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)
model.load_state_dict(torch.load("final_model.pth"))
model.eval()

# 2. Build Isolated Path Map (Dataset Isolation)
path_map = {}
def map_json(json_path, img_dir, prefix):
    if not os.path.exists(json_path): return
    with open(json_path, 'r') as f:
        data = json.load(f)
    for img_obj in data['images']:
        # Unique ID prevents collisions between datasets
        unique_key = f"{prefix}_{img_obj['id']}"
        path_map[unique_key] = os.path.join(img_dir, img_obj['file_name'])

map_json('Drywall-Join-Detect-2/valid/_annotations.coco.json', 'Drywall-Join-Detect-2/valid', 'drywall')
map_json('cracks-1/valid/_annotations.coco.json', 'cracks-1/valid', 'crack')

def calculate_metrics(pred, gt):
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    
    iou = intersection / union if union > 0 else 1.0
    dice = (2 * intersection) / (pred.sum() + gt.sum()) if (pred.sum() + gt.sum()) > 0 else 1.0
    return iou, dice

# 3. Aggregate Metrics over ALL examples
all_masks = [f for f in os.listdir('data_v/masks') if f.endswith('crack.png')]
total_iou = 0
total_dice = 0
valid_count = 0

print(f"Starting global evaluation on {len(all_masks)} masks...")

with torch.no_grad():
    for mask_name in tqdm(all_masks):
        # Extract metadata from filename
        unique_id = mask_name.split("__")[0]
        prompt = mask_name.split("__")[1].replace(".png", "").replace("_", " ")
        
        img_path = path_map.get(unique_id)
        if not img_path or not os.path.exists(img_path):
            continue
            
        # Load data
        image = Image.open(img_path).convert("RGB")
        gt_mask = cv2.imread(os.path.join('data_v/masks', mask_name), 0)
        gt_mask = (gt_mask > 128).astype(np.uint8)
        
        # Inference
        inputs = processor(text=[prompt], images=[image], return_tensors="pt", padding=True).to(device)
        outputs = model(**inputs)
        
        # Post-process
        logits = outputs.logits.squeeze()
        pred_raw = torch.sigmoid(logits).cpu().numpy()
        pred_mask = cv2.resize((pred_raw > 0.3).astype(np.uint8), (image.size[0], image.size[1]), interpolation=cv2.INTER_NEAREST)
        
        # Accumulate
        iou, dice = calculate_metrics(pred_mask, gt_mask)
        total_iou += iou
        total_dice += dice
        valid_count += 1

# 4. Final Report Output
final_miou = total_iou / valid_count
final_dice = total_dice / valid_count

print(f"\n--- FINAL DATASET METRICS ---")
print(f"Total Samples Processed: {valid_count}")
print(f"Mean IoU (mIoU): {final_miou:.4f}")
print(f"Mean Dice Score: {final_dice:.4f}")