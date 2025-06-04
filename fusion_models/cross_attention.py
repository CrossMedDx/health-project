import os
import random
import json
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)
from transformers import (
    BertModel,
    get_linear_schedule_with_warmup,
    ViTModel
)

print("finished imports")


# ---------------------
# Reproducibility
# ---------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
cache_dir = "./cached_data"

########################################################################################################################################



# Custom collate for batching
model_name_vision = 'google/vit-base-patch16-224-in21k'
def fusion_collate_fn(batch):
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    return {
        'pixel_values': pixel_values,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

class CachedFusionDataset(Dataset):
    def __init__(self, cache_dir, indices):
        self.cache_dir = cache_dir
        self.indices = indices  # list of indices used in pre-split sets

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        data = torch.load(os.path.join(self.cache_dir, f'{self.indices[idx]}.pt'))
        return data

###################################################################################


labels = []
with open('labels.json', 'r') as f:
    labels = json.load(f)

indices = list(range(len(labels)))
train_idx, test_val_idx, train_labels, test_val_labels = train_test_split(
    indices, labels, test_size=0.3, random_state=SEED, shuffle=True, stratify=labels
)
test_idx, val_idx, test_labels, val_labels = train_test_split(
    test_val_idx, test_val_labels, test_size=1/3, random_state=SEED, shuffle=True, stratify=test_val_labels
)

print("Train / Val / Test sizes:", len(train_idx), len(val_idx), len(test_idx))


batch_size = 32

train_ds = CachedFusionDataset(cache_dir, train_idx)
val_ds = CachedFusionDataset(cache_dir, val_idx)
test_ds = CachedFusionDataset(cache_dir, test_idx)

def get_loader(ds, bs, shuffle=False):
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, collate_fn=fusion_collate_fn, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)

##################################################################################


class CrossAttentionFusionModel(nn.Module):
    def __init__(self, vision_model_name, text_model_name, vision_drop=0.1, text_drop=0.1, hidden_dim=256, num_heads=8):
        super().__init__()
        self.image_model = ViTModel.from_pretrained(vision_model_name)
        self.text_model = BertModel.from_pretrained(text_model_name)
        self.vision_proj = nn.Linear(self.image_model.config.hidden_size, hidden_dim)
        self.text_proj = nn.Linear(self.text_model.config.hidden_size, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=0.1)
        self.dropout_img = nn.Dropout(vision_drop)
        self.dropout_txt = nn.Dropout(text_drop)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim//2, 2)
        )

    def forward(self, pixel_values, input_ids, attention_mask):
        img_feats = self.image_model(pixel_values=pixel_values).pooler_output  # [B, img_dim]
        txt_feats = self.text_model(input_ids=input_ids, attention_mask=attention_mask).pooler_output  # [B, txt_dim]
        img_proj = self.dropout_img(self.vision_proj(img_feats)).unsqueeze(0)  # [1, B, H]
        txt_proj = self.dropout_txt(self.text_proj(txt_feats)).unsqueeze(0)    # [1, B, H]
        # cross-attention: image as query, text+image as key/value
        seq = torch.cat([img_proj, txt_proj], dim=0)  # [2, B, H]
        attn_out, _ = self.cross_attn(query=img_proj, key=seq, value=seq)  # [1, B, H]
        fusion_vec = attn_out.squeeze(0)  # [B, H]
        logits = self.classifier(fusion_vec)
        return logits

##############################################################


def compute_metrics(y_true, y_pred, y_probs):
    metrics = {}
    for i, name in enumerate(['edema','effusion']):
        acc = accuracy_score(y_true[:,i], y_pred[:,i])
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true[:,i], y_pred[:,i], zero_division=0
        )
        try:
            auroc = roc_auc_score(y_true[:,i], y_probs[:,i])
        except ValueError:
            auroc = float('nan')
        try:
            auprc = average_precision_score(y_true[:,i], y_probs[:,i])
        except ValueError:
            auprc = float('nan')
        tn, fp, fn, tp = confusion_matrix(y_true[:,i], y_pred[:,i]).ravel()
        sens = tp/(tp+fn) if (tp+fn)>0 else 0.0
        spec = tn/(tn+fp) if (tn+fp)>0 else 0.0
        metrics[name] = {
            'accuracy': acc,
            'precision': precision.to_list(),
            'recall': recall.to_list(),
            'f1': f1.to_list(),
            'auroc': auroc,
            'auprc': auprc,
            'sensitivity': sens,
            'specificity': spec
        }
    return metrics

###############################################################################################


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="Training", leave=False):
        optimizer.zero_grad()
        pixel_values = batch['pixel_values'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        logits = model(pixel_values, input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def validate_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", leave=False):
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            logits = model(pixel_values, input_ids, attention_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()
    avg_loss = total_loss / len(loader)

    return avg_loss


########################################################################



def evaluate(model, loader):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluation", leave=False):
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].cpu().numpy()
            logits = model(pixel_values, input_ids, attention_mask).cpu().numpy()
            probs = torch.sigmoid(torch.tensor(logits)).numpy()
            preds = (probs > 0.5).astype(int)
            all_labels.append(labels)
            all_preds.append(preds)
            all_probs.append(probs)
    y_true = np.vstack(all_labels)
    y_pred = np.vstack(all_preds)
    y_probs= np.vstack(all_probs)
    return compute_metrics(y_true, y_pred, y_probs)

##############################################################################


model_name_text = 'dmis-lab/biobert-base-cased-v1.1'
model_name_vision = 'google/vit-base-patch16-224-in21k'
hyperparameter_combinations = []
for vision_drop in [0.1, 0.2]:
    for text_drop in [0.1, 0.2]:
        for lr in [1e-5, 5e-5, 2e-4]:
            for wd in [0, 0.01, 0.1]:
                for bs in [16, 32]:
                    hyperparameter_combinations.append({
                        'vision_drop': vision_drop,
                        'text_drop':   text_drop,
                        'learning_rate': lr,
                        'weight_decay': wd,
                        'batch_size': bs,
                        'num_epochs': 20
                    })

results_file = 'cross_attn_results.json'
if not os.path.exists(results_file):
    with open(results_file, 'w') as f:
        json.dump([], f)

# for combo in hyperparameter_combinations:
def run_experiment(combo):
    name = f"CA_VD{combo['vision_drop']}_TD{combo['text_drop']}_LR{combo['learning_rate']}_WD{combo['weight_decay']}_BS{combo['batch_size']}_EP{combo['num_epochs']}"
    print(f"Running combo: {name}")

    train_loader = get_loader(train_ds, combo['batch_size'], shuffle=True)
    val_loader = get_loader(val_ds, combo['batch_size'])
    test_loader = get_loader(test_ds, combo['batch_size'])

    model = CrossAttentionFusionModel(
        vision_model_name=model_name_vision,
        text_model_name=model_name_text,
        vision_drop=combo['vision_drop'],
        text_drop=combo['text_drop']
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=combo['learning_rate'],
        weight_decay=combo['weight_decay']
    )

    total_steps = len(train_loader) * combo['num_epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1*total_steps),
        num_training_steps=total_steps
    )

    best_val_loss = float('inf')
    patience = 3
    no_improve = 0

    for epoch in range(1, combo['num_epochs']+1):
        train_loss = train_epoch(model, train_loader,optimizer, criterion)
        val_loss = validate_epoch(model, val_loader, criterion)
        print(f"Epoch {epoch}/{combo['num_epochs']} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    test_metrics = evaluate(model, test_loader)
    print(f"Test Metrics for {name}: {test_metrics}")

    with open(results_file, 'r') as f:
        results = json.load(f)
    results.append({'name': name, 'combo': combo, 'metrics': test_metrics})
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Saved results for {name}\n")
###################################################
combo = {
    "vision_drop": 0.2,
    "text_drop": 0.2,
    "weight_decay": 0.01,
    "learning_rate": 5e-05,
    "batch_size": 32,
    "num_epochs": 20,
    "patience": 3
}
run_experiment(combo)


