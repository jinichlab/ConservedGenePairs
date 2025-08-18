import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# === Edit these paths ===
csv_path_1 = "plm_embeddings/fit_organism_pseudo5_N2C3_1_p_values.csv"
csv_path_2 = "plm_embeddings/fit_organism_pseudo6_N2E2_p_values.csv"
# ========================

def parse_embedding(emb_str):
    if pd.isna(emb_str) or not isinstance(emb_str, str) or emb_str.strip() == "":
        return None
    try:
        return np.array([float(x) for x in emb_str.split(";")])
    except:
        return None

def pairwise_protein_similarity(pair1, pair2):
    A1, A2 = pair1
    B1, B2 = pair2

    A1, A2, B1, B2 = map(np.asarray, [A1, A2, B1, B2])
    A1 = A1.reshape(1, -1)
    A2 = A2.reshape(1, -1)
    B1 = B1.reshape(1, -1)
    B2 = B2.reshape(1, -1)

    direct = cosine_similarity(A1, B1)[0][0] + cosine_similarity(A2, B2)[0][0]
    flipped = cosine_similarity(A1, B2)[0][0] + cosine_similarity(A2, B1)[0][0]
    return min(direct, flipped) / 2

# === Load and prepare data ===
df1 = pd.read_csv(csv_path_1)
df2 = pd.read_csv(csv_path_2)

for df in [df1, df2]:
    df["embedding1"] = df["embedding1"].apply(parse_embedding)
    df["embedding2"] = df["embedding2"].apply(parse_embedding)
    df.dropna(subset=["embedding1", "embedding2"], inplace=True)

# === Use top pair from df1 ===
top_row = df1.iloc[0]
query_pair = (top_row["embedding1"], top_row["embedding2"])
query_meta = (top_row["locus1"], top_row["locus2"])

# === Compare to all in df2 ===
results = []
for row in df2.itertuples(index=False):
    emb_pair = (row.embedding1, row.embedding2)
    similarity = pairwise_protein_similarity(query_pair, emb_pair)
    results.append({
        "query_locus1": query_meta[0],
        "query_locus2": query_meta[1],
        "target_locus1": row.locus1,
        "target_locus2": row.locus2,
        "similarity": similarity
    })

results_df = pd.DataFrame(results).sort_values(by="similarity", ascending=False)

# === Output ===
print(f"\n🔍 Top pair from {csv_path_1}: {query_meta}")
print(f"Compared to all pairs in {csv_path_2}\n")
print(results_df.head(100))  # top 10 matches
