import torch, cv2, os, json, numpy as np
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor
from PIL import Image

# 1. Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)
model.load_state_dict(torch.load("final_model.pth"))
model.eval()

# 2. Build Isolated Path Map
path_map = {}
def map_json(json_path, img_dir, prefix):
    if not os.path.exists(json_path): return
    with open(json_path, 'r') as f:
        data = json.load(f)
    for img_obj in data['images']:
        unique_key = f"{prefix}_{img_obj['id']}"
        path_map[unique_key] = os.path.join(img_dir, img_obj['file_name'])

map_json('Drywall-Join-Detect-2/train/_annotations.coco.json', 'Drywall-Join-Detect-2/train', 'drywall')
map_json('cracks-1/train/_annotations.coco.json', 'cracks-1/train', 'crack')

def get_metrics(pred, gt):
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    iou = intersection / union if union > 0 else 1.0
    dice = (2 * intersection) / (pred.sum() + gt.sum()) if (pred.sum() + gt.sum()) > 0 else 1.0
    return iou, dice

def run_visual_eval(unique_id, prompt, mask_dir, out_name):
    img_path = path_map.get(unique_id)
    if not img_path: return None
    
    image = Image.open(img_path).convert("RGB")
    inputs = processor(text=[prompt], images=[image], return_tensors="pt", padding=True).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Process Prediction
    logits = outputs.logits.squeeze()
    pred_raw = torch.sigmoid(logits).cpu().numpy()
    pred_mask = cv2.resize((pred_raw > 0.3).astype(np.uint8), (image.size[0], image.size[1]), interpolation=cv2.INTER_NEAREST)
    
    # Load GT Mask
    gt_path = os.path.join(mask_dir, f"{unique_id}__{prompt.replace(' ', '_')}.png")
    gt_mask = cv2.imread(gt_path, 0)
    if gt_mask is None: return None
    gt_mask = (gt_mask > 128).astype(np.uint8)
    
    iou, dice = get_metrics(pred_mask, gt_mask)
    
    # Create Visual Strip: Orig | GT | Pred
    orig_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gt_viz = cv2.cvtColor(gt_mask * 255, cv2.COLOR_GRAY2BGR)
    pred_viz = cv2.cvtColor(pred_mask * 255, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(out_name, np.hstack((orig_np, gt_viz, pred_viz)))
    
    return iou, dice

# --- Run for Report ---
results = []
# Evaluate Crack (Polygon GT)
r_crack = run_visual_eval('crack_0', 'segment crack', 'data/masks', 'final_crack_viz.png')
# Evaluate Drywall (BBox GT)
r_drywall = run_visual_eval('drywall_0', 'segment taping area', 'data/masks', 'final_drywall_viz.png')

metrics = [r for r in [r_crack, r_drywall] if r is not None]
if metrics:
    print(f"\nFinal mIoU: {np.mean([m[0] for m in metrics]):.4f}")
    print(f"Final Dice: {np.mean([m[1] for m in metrics]):.4f}")