import json
import pandas as pd

# Load the dataset
filename = "metadata/mimic-cxr-2.0.0-chexpert.csv"
df = pd.read_csv(filename)

# Drop unnecessary columns and calculate counts
counts = (
    df.drop(columns=["subject_id", "study_id"])
    .apply(
        lambda x: x.value_counts().reindex([-1, 0, 1], fill_value=0).to_dict(), axis=0
    )
    .to_dict()
)

# Sort and reformat the counts
formatted_counts = {
    key: {"1": value[1], "0": value[0], "-1": value[-1]}
    for key, value in sorted(
        counts.items(), key=lambda x: (-x[1][1], -x[1][0], -x[1][-1], x[0])
    )
}

# Save the counts to a JSON file
with open("metadata/counts.json", "w") as f:
    json.dump(formatted_counts, f, indent=4, ensure_ascii=False)

# Define the number of samples to select
NUM_SAMPLES = 150
SAMPLES_PER_BUCKET = NUM_SAMPLES // 3

# Filter the dataset for the column "Pleural Effusion" and sample data
column_name = "Pleural Effusion"
samples = []
for label in [-1, 0, 1]:
    # Filter the dataset for the current label
    filtered_df = df[df[column_name] == label].sample(
        SAMPLES_PER_BUCKET, random_state=42
    )

    # Create a DataFrame with the required columns
    sample_df = filtered_df[["subject_id", "study_id"]].copy()
    sample_df["class"] = label

    # Append the sample DataFrame to the list
    samples.append(sample_df)

# Combine the samples and save to a CSV file
pd.concat(samples)[["subject_id", "study_id", "class"]].to_csv(
    f"metadata/{column_name.lower().replace(' ', '_')}_samples.csv", index=False
)
