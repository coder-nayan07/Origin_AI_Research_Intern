import os, cv2, numpy as np
from pycocotools.coco import COCO

def generate_prefixed_masks(json_path, output_dir, prompt_label, prefix, use_polygons=False):
    if not os.path.exists(json_path): return
    coco = COCO(json_path)
    os.makedirs(output_dir, exist_ok=True)
    
    for img_id in coco.imgs:
        img_info = coco.loadImgs(img_id)[0]
        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        
        for ann in anns:
            if use_polygons and 'segmentation' in ann and ann['segmentation']:
                # Specific: Use pixel-perfect polygons for cracks
                mask = np.maximum(mask, coco.annToMask(ann) * 255)
            elif 'bbox' in ann:
                # Specific: Use bounding box for drywall
                x, y, w, h = [int(v) for v in ann['bbox']]
                mask[y:y+h, x:x+w] = 255
            
        # New Naming Format: {prefix}_{id}__{prompt}.png
        clean_prompt = prompt_label.replace(" ", "_")
        filename = f"{prefix}_{img_id}__{clean_prompt}.png"
        cv2.imwrite(os.path.join(output_dir, filename), mask)

# Run with prefixes to avoid collisions
generate_prefixed_masks('Drywall-Join-Detect-2/train/_annotations.coco.json', 'data/masks', 'segment_taping_area', 'drywall', use_polygons=False)
generate_prefixed_masks('cracks-1/train/_annotations.coco.json', 'data/masks', 'segment_crack', 'crack', use_polygons=True)