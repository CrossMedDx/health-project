import json
import random
from collections import defaultdict, Counter

SEED = 42
random.seed(SEED)

# Load labels
with open("labels.json", "r") as f:
    labels = json.load(f)

# Group indices by label combination
label_groups = defaultdict(list)
for idx, label in enumerate(labels):
    key = tuple(label)
    label_groups[key].append(idx)

target_total = 3600
total_available = sum(len(v) for v in label_groups.values())

# First, calculate proportional targets
proportions = {k: len(v) / total_available for k, v in label_groups.items()}
proportional_targets = {k: int(p * target_total) for k, p in proportions.items()}

selected_indices = []

# Step 1: Select all from groups with fewer samples than target
leftover = target_total
for label, group in label_groups.items():
    if len(group) < proportional_targets[label]:
        selected_indices.extend(group)
        leftover -= len(group)
        proportional_targets[label] = len(group)  # adjust target

# Step 2: For groups with enough samples, allocate leftover proportionally
# Calculate total proportion of remaining groups
remaining_groups = [label for label in label_groups if len(label_groups[label]) >= proportional_targets[label]]
remaining_total = sum(len(label_groups[g]) for g in remaining_groups)

for label in remaining_groups:
    group = label_groups[label]
    # New target proportional to leftover
    new_target = int(len(group) / remaining_total * leftover)
    selected = random.sample(group, new_target)
    selected_indices.extend(selected)

# Shuffle final selected indices
random.shuffle(selected_indices)

selected_labels = [labels[i] for i in selected_indices]

# Save selected indices and labels
with open("selected_indices.json", "w") as f:
    json.dump(selected_indices, f, indent=4)
with open("selected_labels.json", "w") as f:
    json.dump(selected_labels, f, indent=4)

# Print counts
label_counter = Counter(tuple(label) for label in selected_labels)
for label in sorted(label_counter.keys()):
    print(f"Label {label}: {label_counter[label]} samples")
