import torch, cv2, os, json, numpy as np
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor
from PIL import Image

# 1. Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)
model.load_state_dict(torch.load("final_model.pth"))
model.eval()

# Build path lookup from JSONs
path_map = {}
def map_json(json_path, img_dir):
    if not os.path.exists(json_path): return
    with open(json_path, 'r') as f:
        data = json.load(f)
    for img_obj in data['images']:
        path_map[str(img_obj['id'])] = os.path.join(img_dir, img_obj['file_name'])

map_json('Drywall-Join-Detect-2/train/_annotations.coco.json', 'Drywall-Join-Detect-2/train')
map_json('cracks-1/train/_annotations.coco.json', 'cracks-1/train')
def get_metrics(pred_mask, gt_mask):
    # Ensure both are boolean/binary for logical ops
    pred = (pred_mask > 0).astype(np.uint8)
    gt = (gt_mask > 0).astype(np.uint8)
    
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    
    iou = intersection / union if union > 0 else 1.0
    dice = (2 * intersection) / (pred.sum() + gt.sum()) if (pred.sum() + gt.sum()) > 0 else 1.0
    return iou, dice

def run_eval(img_id, prompt, mask_dir, out_name):
    img_path = path_map.get(str(img_id))
    if not img_path or not os.path.exists(img_path):
        print(f"Skipping ID {img_id}: image not found.")
        return None
    
    image = Image.open(img_path).convert("RGB")
    inputs = processor(text=[prompt], images=[image], return_tensors="pt", padding=True).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # FIX: Squeeze to remove extra dimensions before resizing
    # CLIPSeg logits shape is usually [1, 352, 352]
    logits = outputs.logits.squeeze() # Result: [352, 352]
    
    pred_raw = torch.sigmoid(logits).cpu().numpy()
    # Resize 2D prediction to match original image size
    pred_mask = cv2.resize((pred_raw > 0.3).astype(np.uint8), (image.size[0], image.size[1]), interpolation=cv2.INTER_NEAREST)
    
    # Load GT Mask and ensure it matches image size
    gt_name = f"{img_id}__{prompt.replace(' ', '_')}.png"
    gt_path = os.path.join(mask_dir, gt_name)
    if not os.path.exists(gt_path):
        print(f"GT not found: {gt_path}")
        return None
        
    gt_mask = cv2.imread(gt_path, 0)
    if gt_mask is None: return None
    gt_mask = (gt_mask > 128).astype(np.uint8) # Binary 0 or 1
    
    # Calculate Metrics
    iou, dice = get_metrics(pred_mask, gt_mask)
    
    # Visual Strip Generation
    orig_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gt_viz = cv2.cvtColor(gt_mask * 255, cv2.COLOR_GRAY2BGR)
    pred_viz = cv2.cvtColor(pred_mask * 255, cv2.COLOR_GRAY2BGR)
    strip = np.hstack((orig_np, gt_viz, pred_viz))
    cv2.imwrite(out_name, strip)
    
    return iou, dice

# Pick IDs that actually exist in your 'data/masks' folder
# Check your folder to see what IDs were generated
results = []
r1 = run_eval('5', 'segment taping area', 'data/masks', 'viz_taping.png')
r2 = run_eval('5', 'segment crack', 'data/masks', 'viz_crack.png') 
# ... [Rest of print logic] ...
# Filter None results
metrics = [r for r in [r1, r2] if r is not None]
if metrics:
    avg_iou = np.mean([m[0] for m in metrics])
    avg_dice = np.mean([m[1] for m in metrics])
    print(f"\nReport Metrics:\nmIoU: {avg_iou:.4f}\nDice: {avg_dice:.4f}")