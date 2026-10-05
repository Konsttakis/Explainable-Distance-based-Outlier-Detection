import os
import json
import pandas as pd

# Paths (relative to this script, i.e. Code/data/)
data_dir = os.path.dirname(os.path.abspath(__file__))
nab_dir = os.path.join(data_dir, "NAB_Archive")
labels_file = os.path.join(nab_dir, "labels", "combined_labels.json")
output_dir = os.path.join(data_dir, "NAB_formatted")

os.makedirs(output_dir, exist_ok=True)

print("Loading labels...")
with open(labels_file, "r") as f:
    labels_dict = json.load(f)

count = 0
for file_path, anomaly_timestamps in labels_dict.items():
    full_csv_path = os.path.join(nab_dir, "data", file_path)
    if not os.path.exists(full_csv_path):
        continue

    # Read original CSV
    df = pd.read_csv(full_csv_path)

    # Skip files without the standard NAB columns
    if 'timestamp' not in df.columns or 'value' not in df.columns:
        continue

    # Create label column, default 0
    df['label'] = 0

    # Set 1 for anomaly timestamps
    if anomaly_timestamps:
        df.loc[df['timestamp'].isin(anomaly_timestamps), 'label'] = 1

    # Same layout as the Yahoo files: column 0 = value, column 1 = label
    output_df = df[['value', 'label']]

    # Flatten the category folder into the file name
    flat_name = file_path.replace("/", "_")
    out_path = os.path.join(output_dir, flat_name)

    output_df.to_csv(out_path, index=False)
    count += 1

print(f"Successfully formatted {count} NAB datasets into {output_dir}")
