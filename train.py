import torch, os, json, numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor
from PIL import Image

class PrefixedHybridDataset(Dataset):
    def __init__(self, mask_dir, drywall_json, cracks_json, drywall_img_dir, cracks_img_dir, processor):
        self.mask_dir = mask_dir
        # Only load masks that match our new prefixed naming convention
        self.mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.png')]
        self.processor = processor
        
        # Build the Isolated Source of Truth: { "prefix_id": "full/path.jpg" }
        self.path_map = {}
        self._map_json(drywall_json, drywall_img_dir, "drywall")
        self._map_json(cracks_json, cracks_img_dir, "crack")

    def _map_json(self, json_path, img_dir, prefix):
        if not os.path.exists(json_path):
            print(f"Warning: {json_path} not found.")
            return
        with open(json_path, 'r') as f:
            data = json.load(f)
        # Use a prefix to prevent ID 0 in crack from overwriting ID 0 in drywall
        for img_obj in data['images']:
            unique_key = f"{prefix}_{img_obj['id']}"
            self.path_map[unique_key] = os.path.join(img_dir, img_obj['file_name'])

    def __len__(self): 
        return len(self.mask_files)

    def __getitem__(self, idx):
        mask_name = self.mask_files[idx]
        # Filename expected: "crack_123__segment_crack.png" or "drywall_0__segment_taping_area.png"
        full_id = mask_name.split("__")[0] # e.g., "crack_123"
        prompt = mask_name.split("__")[1].replace(".png", "").replace("_", " ")
        
        img_path = self.path_map.get(full_id)
        if not img_path or not os.path.exists(img_path):
            raise FileNotFoundError(f"Lookup failed for {full_id}. Path: {img_path}")
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, mask_name)).convert("L")
        
        # CLIPSeg preprocessing
        inputs = self.processor(text=[prompt], images=[image], return_tensors="pt", padding="max_length")
        target = torch.tensor(np.array(mask.resize((352, 352), Image.NEAREST))).float() / 255.0
        
        return inputs.pixel_values.squeeze(0), inputs.input_ids.squeeze(0), target

# --- Setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)

ds = PrefixedHybridDataset(
    mask_dir='data/masks',
    drywall_json='Drywall-Join-Detect-2/train/_annotations.coco.json',
    cracks_json='cracks-1/train/_annotations.coco.json',
    drywall_img_dir='Drywall-Join-Detect-2/train',
    cracks_img_dir='cracks-1/train',
    processor=processor
)

loader = DataLoader(ds, batch_size=8, shuffle=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# --- Training Loop ---
model.train()
print(f"Dataset loaded: {len(ds)} samples correctly isolated.")
for epoch in range(5):
    total_loss = 0
    for p, i, m in loader:
        optimizer.zero_grad()
        outputs = model(pixel_values=p.to(device), input_ids=i.to(device))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs.logits, m.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} Loss: {total_loss/len(loader):.4f}")

torch.save(model.state_dict(), "final_model.pth")