import os
import glob
import torch
import esm
import numpy as np
import pandas as pd
from tqdm import tqdm

# Load pretrained ESM-2 model
model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
batch_converter = alphabet.get_batch_converter()

# Device: MPS (Apple Silicon) or fallback to CPU
try:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
except Exception:
    device = torch.device("cpu")

model = model.eval().to(device)

# Directories
p_values_dir = "p_values"
seq_aug_dir = "sequence_augmented"
output_dir = "plm_embeddings"
os.makedirs(output_dir, exist_ok=True)

# Set of organisms to process based on p_values
pvalue_basenames = set(
    os.path.splitext(os.path.basename(p))[0]
    for p in glob.glob(f"{p_values_dir}/*.csv")
)

# Set of already processed organisms in plm_embeddings
already_embedded = set(
    os.path.splitext(os.path.basename(p))[0]
    for p in glob.glob(f"{output_dir}/*.csv")
)

def embed_sequence(seq):
    data = [("protein", seq)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[6], return_contacts=False)
    token_representations = results["representations"][6]
    sequence_repr = token_representations[0, 1:-1].mean(0).cpu().numpy()
    return sequence_repr

# Process each file from sequence_augmented that has a match in p_values and hasn't been embedded
for csv_file in tqdm(glob.glob(f"{seq_aug_dir}/*.csv"), desc="Checking organisms"):
    base_name = os.path.splitext(os.path.basename(csv_file))[0]

    if base_name not in pvalue_basenames:
        continue  # skip if not a p-value file

    if base_name in already_embedded:
        print(f"⏭️  Skipping {base_name} — already embedded")
        continue  # skip if already embedded

    print(f"🧬 Embedding: {base_name}")
    df = pd.read_csv(csv_file)
    embeddings1 = []
    embeddings2 = []

    for _, row in tqdm(df.iterrows(), total=len(df), leave=False):
        seq1 = row.get("seq1")
        seq2 = row.get("seq2")

        try:
            emb1 = embed_sequence(seq1) if pd.notna(seq1) and isinstance(seq1, str) else None
        except Exception:
            emb1 = None

        try:
            emb2 = embed_sequence(seq2) if pd.notna(seq2) and isinstance(seq2, str) else None
        except Exception:
            emb2 = None

        # Convert numpy arrays to strings for CSV
        emb1_str = ";".join(map(str, emb1)) if emb1 is not None else ""
        emb2_str = ";".join(map(str, emb2)) if emb2 is not None else ""

        embeddings1.append(emb1_str)
        embeddings2.append(emb2_str)

    df["embedding1"] = embeddings1
    df["embedding2"] = embeddings2

    out_path = os.path.join(output_dir, base_name + ".csv")
    df.to_csv(out_path, index=False)
    print(f"✅ Saved embeddings CSV: {out_path}")


# For loading the embeddings from CSV since they are in string format in the csv
# def parse_embedding_str(s):
#     if not s:
#         return None
#     return np.array(list(map(float, s.split(";"))))

# df = pd.read_csv("plm_embeddings/somefile.csv")
# embedding1 = df["embedding1"].apply(parse_embedding_str)
# embedding2 = df["embedding2"].apply(parse_embedding_str)