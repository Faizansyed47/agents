#!/usr/bin/env python3
import os
import json
import torch
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from transformers import AutoTokenizer, AutoModel

# -----------------------------
# CONFIG
# -----------------------------
CSV_FILE = "data.csv"
VECTOR_DIR = "./vectorizers"
COLLECTION_NAME = "csv_single_vector"
PROFILE_COL = "profile"
TOP_K = 5
MILVUS_BATCH_SIZE = 16
MAX_VECTOR_DIM = 1024
DEVICE = "cpu"
MODEL_PATH = "Provide path"
DENSE_DIM = 512  # fixed dense embedding size
SPARSE_DIM = 512  # TF-IDF size

# -----------------------------
# 1. Load CSV
# -----------------------------
df = pd.read_csv(CSV_FILE)

if PROFILE_COL not in df.columns:
    df[PROFILE_COL] = [f"user_{i+1}" for i in range(len(df))]
else:
    df[PROFILE_COL] = df[PROFILE_COL].astype(str).str.lower()

text_columns = df.drop(columns=[PROFILE_COL], errors="ignore").columns
print(f"[1] Loaded {len(df)} rows and {len(text_columns)} text columns.")

# -----------------------------
# 2. Load local BGE-M3 model
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
model = AutoModel.from_pretrained(MODEL_PATH, local_files_only=True)
model.to(DEVICE)
model.eval()
print("[2] Local BGE-M3 model loaded.")

# Optional projection if model hidden_size != 512
if model.config.hidden_size != DENSE_DIM:
    projection_layer = torch.nn.Linear(model.config.hidden_size, DENSE_DIM).to(DEVICE)
else:
    projection_layer = None

# -----------------------------
# 3. TF-IDF vectorizer
# -----------------------------
os.makedirs(VECTOR_DIR, exist_ok=True)
vectorizer_path = os.path.join(VECTOR_DIR, "vectorizer_row.pkl")
row_texts = df[text_columns].astype(str).agg(" ".join, axis=1).tolist()

if os.path.exists(vectorizer_path):
    vectorizer = joblib.load(vectorizer_path)
    print("[3] Loaded existing TF-IDF vectorizer.")
else:
    vectorizer = TfidfVectorizer(max_features=SPARSE_DIM)
    vectorizer.fit(row_texts)
    joblib.dump(vectorizer, vectorizer_path)
    print("[3] Created new TF-IDF vectorizer.")

print(f"[3] Sparse TF-IDF vector dim: {SPARSE_DIM}")

# -----------------------------
# 4. Connect Milvus
# -----------------------------
connections.connect("default", host="127.0.0.1", port="19530")
print("[4] Connected to Milvus.")

# -----------------------------
# 5. Create Collection
# -----------------------------
final_vector_dim = DENSE_DIM + SPARSE_DIM
if COLLECTION_NAME in utility.list_collections():
    print(f"[5] Dropping existing collection '{COLLECTION_NAME}'...")
    utility.drop_collection(COLLECTION_NAME)

fields = [
    FieldSchema(name=PROFILE_COL, dtype=DataType.VARCHAR, max_length=2000, is_primary=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=final_vector_dim),
    FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=2000)
]
schema = CollectionSchema(fields, description="Dynamic vector per row")
collection = Collection(name=COLLECTION_NAME, schema=schema)
print(f"[5] Collection '{COLLECTION_NAME}' created with vector dim: {final_vector_dim}")

# -----------------------------
# 6. Embedding generator
# -----------------------------
def generate_embeddings(df_input):
    texts = df_input[text_columns].astype(str).agg(" ".join, axis=1).tolist()

    # Dense embeddings
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**encoded)
        dense = outputs.last_hidden_state[:, 0, :]  # [CLS] token

        if projection_layer is not None:
            dense = projection_layer(dense)

        dense = torch.nn.functional.normalize(dense, dim=-1)
        dense = dense.cpu().numpy().astype(np.float32)

    # Sparse TF-IDF embeddings
    sparse = vectorizer.transform(texts).toarray().astype(np.float32)

    # Combine dense + sparse
    combined = np.hstack([dense, sparse])

    # Ensure final_vector_dim = 1024
    if combined.shape[1] < final_vector_dim:
        pad_len = final_vector_dim - combined.shape[1]
        combined = np.hstack([combined, np.zeros((combined.shape[0], pad_len), dtype=np.float32)])
    elif combined.shape[1] > final_vector_dim:
        pca = PCA(n_components=final_vector_dim)
        combined = pca.fit_transform(combined)

    metadata = df_input.apply(lambda row: row.to_json(), axis=1).tolist()
    return combined, metadata

# -----------------------------
# 7. Upsert / ingest profiles
# -----------------------------
def upsert_profiles_incremental(df_input):
    vectors, metadata = generate_embeddings(df_input)
    num_rows = len(df_input)

    for start in tqdm(range(0, num_rows, MILVUS_BATCH_SIZE), desc="Insert Batches"):
        end = start + MILVUS_BATCH_SIZE
        batch_profile_ids = df_input[PROFILE_COL].iloc[start:end].tolist()
        batch_vectors = vectors[start:end].tolist()
        batch_metadata = [json.dumps({**json.loads(meta)}) for meta in metadata[start:end]]

        entities = [
            batch_profile_ids,
            batch_vectors,
            batch_metadata
        ]
        collection.insert(entities)

    collection.flush()
    print(f"[7] Inserted {num_rows} profiles into Milvus.")

# -----------------------------
# 8. Create index
# -----------------------------
index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "IP",
    "params": {"nlist": 128}
}
collection.create_index(field_name="vector", index_params=index_params)
collection.load()
print("[8] Collection loaded and index created for search.")

# -----------------------------
# 9. Hybrid Search
# -----------------------------
def hybrid_search(query_text, top_k=TOP_K):
    encoded = tokenizer(
        [query_text],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**encoded)
        dense_query = outputs.last_hidden_state[:, 0, :]
        if projection_layer is not None:
            dense_query = projection_layer(dense_query)
        dense_query = torch.nn.functional.normalize(dense_query, dim=-1)
        dense_query = dense_query.cpu().numpy().astype(np.float32)

    sparse_query = vectorizer.transform([query_text]).toarray().astype(np.float32)
    query_vector = np.hstack([dense_query, sparse_query])

    if query_vector.shape[1] < final_vector_dim:
        pad_len = final_vector_dim - query_vector.shape[1]
        query_vector = np.hstack([query_vector, np.zeros((1, pad_len), dtype=np.float32)])
    elif query_vector.shape[1] > final_vector_dim:
        pca = PCA(n_components=final_vector_dim)
        query_vector = pca.fit_transform(query_vector)

    results = collection.search(
        data=query_vector.tolist(),
        anns_field="vector",
        param={"metric_type": "IP", "params": {"nprobe": 10}},
        limit=top_k,
        output_fields=["metadata"]
    )

    print(f"[9] Top {top_k} results for '{query_text}':")
    for i, hit in enumerate(results[0]):
        print(f"{i+1}. Score: {hit.distance:.4f}, Metadata: {hit.entity.get('metadata')}")

# -----------------------------
# 10. Run ingestion
# -----------------------------
upsert_profiles_incremental(df)

# -----------------------------
# 11. Example search
# -----------------------------
hybrid_search("example query")

# -----------------------------
# 12. Compute user compatibility matrix
# -----------------------------
def compute_user_compatibility(df_input, top_k=TOP_K):
    vectors, _ = generate_embeddings(df_input)
    vectors = vectors.astype(np.float32)

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors_norm = vectors / (norms + 1e-10)

    results = []
    for i, profile_id in enumerate(df_input[PROFILE_COL]):
        query_vector = vectors_norm[i:i+1]
        scores = np.dot(vectors_norm, query_vector.T).flatten()
        scores[i] = -1.0  # exclude self
        top_indices = scores.argsort()[::-1][:top_k]

        for idx in top_indices:
            results.append({
                "profile": profile_id,
                "compatible_user": df_input[PROFILE_COL].iloc[idx],
                "similarity_score": scores[idx]
            })

    return pd.DataFrame(results)

# -----------------------------
# 13. Generate CSV with compatible users
# -----------------------------
compatibility_df = compute_user_compatibility(df, top_k=TOP_K)
compatibility_csv_file = "user_compatibility.csv"
compatibility_df.to_csv(compatibility_csv_file, index=False)
print(f"[12] User compatibility CSV saved as '{compatibility_csv_file}'")
