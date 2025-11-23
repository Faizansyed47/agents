import os
import pandas as pd
import csv
from tqdm import tqdm
from collections import Counter

# ===== CONFIG =====
input_folder = r"Provide path here"
output_folder = r"provide path here"
output_base = "unique_by_email"
csv_chunk_size = 500_000
rows_per_output_file = 1_000_000
email_header = "Email"
encoding_in = "latin1"
encoding_out = "utf-8-sig"
# ==================

os.makedirs(output_folder, exist_ok=True)

file_list = [
    os.path.join(input_folder, f)
    for f in os.listdir(input_folder)
    if f.lower().endswith(".csv")
]

if not file_list:
    raise SystemExit("No CSV files found in input folder.")

# -------- PASS 1: COUNT EMAIL OCCURRENCES --------
print("PASS 1: Counting email occurrences...")

email_counter = Counter()

for file_path in tqdm(file_list, desc="Counting emails", unit="file"):
    if os.stat(file_path).st_size == 0:
        continue

    try:
        for chunk in pd.read_csv(
            file_path,
            usecols=[email_header],
            dtype=str,
            encoding=encoding_in,
            chunksize=csv_chunk_size,
            on_bad_lines="skip"     # FIXED
        ):
            chunk = chunk.dropna(subset=[email_header])
            if chunk.empty:
                continue

            emails = (
                chunk[email_header]
                .astype(str)
                .str.strip()
                .str.lower()
            )

            email_counter.update(emails.tolist())

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        continue

unique_emails_set = {email for email, cnt in email_counter.items() if cnt == 1}
unique_count = len(unique_emails_set)

print(f"Total unique emails (appear once): {unique_count:,}")
if unique_count == 0:
    raise SystemExit("No unique emails found — STOPPING.")

# -------- PASS 2: UNION HEADERS --------
print("PASS 2: Collecting headers...")

all_headers = []
seen = set()

for file_path in file_list:
    try:
        df_header = pd.read_csv(file_path, nrows=0, encoding=encoding_in, on_bad_lines="skip")
        for col in df_header.columns:
            if col not in seen:
                seen.add(col)
                all_headers.append(col)
    except Exception as e:
        print(f"Header read error in {file_path}: {e}")
        continue

if email_header not in seen:
    raise SystemExit(f"Email column '{email_header}' missing!")

all_headers = [h for h in all_headers if h != email_header]
all_fieldnames = [email_header] + all_headers

print(f"Total combined columns: {len(all_fieldnames)}")

# -------- PASS 3: WRITE UNIQUE ROWS --------
print("PASS 3: Writing unique rows...")

part_num = 1
row_written = 0

def new_output_file():
    global part_num, row_written
    file_path = os.path.join(output_folder, f"{output_base}_part{part_num}.csv")
    f = open(file_path, "w", newline="", encoding=encoding_out)
    writer = csv.DictWriter(f, fieldnames=all_fieldnames)
    writer.writeheader()
    part_num += 1
    return f, writer, file_path

f_out, writer, current_file = new_output_file()
print(f"Writing → {current_file}")

for file_path in tqdm(file_list, desc="Exporting rows", unit="file"):
    try:
        for chunk in pd.read_csv(
            file_path,
            dtype=str,
            encoding=encoding_in,
            chunksize=csv_chunk_size,
            on_bad_lines="skip"
        ):
            if email_header not in chunk.columns:
                continue

            chunk[email_header] = (
                chunk[email_header]
                .astype(str)
                .str.strip()
                .str.lower()
            )

            matches = chunk[chunk[email_header].isin(unique_emails_set)]
            if matches.empty:
                continue

            for _, row in matches.iterrows():

                if row_written % rows_per_output_file == 0 and row_written > 0:
                    f_out.close()
                    f_out, writer, current_file = new_output_file()
                    print(f"→ New output file: {current_file}")

                record = {col: "" for col in all_fieldnames}
                for col, val in row.items():
                    record[col] = "" if pd.isna(val) else str(val)

                writer.writerow(record)
                row_written += 1

    except Exception as e:
        print(f"Processing error in {file_path}: {e}")
        continue

f_out.close()
print(f"Done. Total unique rows saved: {row_written:,}")
print(f"CSV files saved under: {output_folder}")
