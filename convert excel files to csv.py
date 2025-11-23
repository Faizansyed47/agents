import os
import pandas as pd

# ===== CONFIGURATION =====
source_folder = r"Provid input folder here"
target_folder = r"Provide output folder here"
# =========================

# Create target folder if it doesn't exist
os.makedirs(target_folder, exist_ok=True)

# Collect all Excel files in the folder (no subfolders)
excel_files = [
    os.path.join(source_folder, f)
    for f in os.listdir(source_folder)
    if f.lower().endswith((".xlsx", ".xls")) and os.path.isfile(os.path.join(source_folder, f))
]

print(f"Found {len(excel_files)} Excel files to convert in the folder.")

# Process each file
for file_path in excel_files:
    try:
        filename = os.path.basename(file_path)
        name_only = os.path.splitext(filename)[0]
        target_file = os.path.join(target_folder, f"{name_only}.csv")

        # Read the Excel file (first sheet)
        df = pd.read_excel(file_path, sheet_name=0, engine='openpyxl')
        
        # Save as CSV
        df.to_csv(target_file, index=False, encoding='utf-8-sig')
        print(f"Converted: {filename} → {target_file}")

    except Exception as e:
        print(f"Error converting {filename}: {e}")

print("All files converted successfully!")
