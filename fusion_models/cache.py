import os
import torch
import pydicom
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import BertTokenizer
from torchvision import transforms
from concurrent.futures import ThreadPoolExecutor, as_completed

cache_dir = "./cached_data"
image_dir = "/mnt/e/ecs289l/mimic-cxr-download/imageData/"
report_dir = "/mnt/e/ecs289l/mimic-cxr-download/textData/"
labels_file = "../download_data/metadata/edema+pleural_effusion_samples_v2.csv"
model_name_text = 'dmis-lab/biobert-base-cased-v1.1'
model_name_vision = 'google/vit-base-patch16-224-in21k'
max_length = 128

# ---------------------
# Labels and File Loading
# ---------------------

# Load metadata
meta = pd.read_csv(labels_file, dtype={'study_id': str})
meta['study_id'] = 's' + meta['study_id']
label_map = meta.set_index('study_id')[['edema', 'effusion']].to_dict(orient='index')

# Collect image paths and labels
all_image_paths = []
for root, _, files in os.walk(image_dir):
    for f in files:
        if f.endswith('.dcm'):
            all_image_paths.append(os.path.join(root, f))
paths, labels = [], []
for p in all_image_paths:
    sid = os.path.basename(os.path.dirname(p))
    if sid in label_map:
        paths.append(p)
        labels.append(label_map[sid]['edema'] + label_map[sid]['effusion']*2)  # placeholder, we will use list below
# Actually build multi-label list
labels = [ [label_map[os.path.basename(os.path.dirname(p))]['edema'],
            label_map[os.path.basename(os.path.dirname(p))]['effusion']]
          for p in paths ]
os.makedirs(cache_dir, exist_ok=True)

# ---------------------
# Image and Text Processing
# ---------------------

tokenizer = BertTokenizer.from_pretrained(model_name_text)
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(0.2,0.2,0.2,0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# ---------------------
# Process and Save Function
# ---------------------

def process_and_save(i, path, label):
    try:
        # Ignore if the file already exists
        cache_path = os.path.join(cache_dir, f'{i}.pt')
        if os.path.exists(cache_path):
            return
        # Image
        dcm = pydicom.dcmread(path)
        arr = dcm.pixel_array.astype(np.float32)
        img = Image.fromarray((arr / arr.max() * 255).astype(np.uint8)).convert('RGB')
        img_tensor = transform(img)

        # Text
        sid = os.path.basename(os.path.dirname(path))
        report_path = os.path.join(report_dir, sid, 'report.txt')
        with open(report_path, 'r', encoding='utf-8') as f:
            text = f.read()
        encoding = tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        # Label
        label_tensor = torch.tensor(label, dtype=torch.float32)

        # Save all
        torch.save({
            'pixel_values': img_tensor,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': label_tensor
        }, os.path.join(cache_dir, f'{i}.pt'))

    except Exception as e:
        print(f"Failed to process index {i}: {e}")

# Set number of threads based on your CPU (e.g., 8 or 16)
num_threads = 8

with ThreadPoolExecutor(max_workers=num_threads) as executor:
    futures = [
        executor.submit(process_and_save, i, path, label)
        for i, (path, label) in enumerate(zip(paths, labels))
    ]

    for _ in tqdm(as_completed(futures), total=len(futures), desc="Caching"):
        pass
