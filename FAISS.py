import os
import pandas as pd
import numpy as np
import faiss
from tqdm import tqdm
import itertools

# -----------------------
# Parameters
# -----------------------
input_dir = "./plm_embeddings"  # folder with organism CSVs
output_csv = "gene_pair_similarity.csv"
K = 50  # number of valid pairs from organism A
J = 50  # number of valid pairs from organism B

# -----------------------
# Helper function to parse embedding string into numpy array
# Returns None if parsing fails or embedding is invalid
# -----------------------
def parse_embedding(embedding):
    try:
        if isinstance(embedding, str):
            if embedding.strip() == "":
                return None
            vec = np.array([float(x) for x in embedding.split(';')], dtype=np.float32)
        elif isinstance(embedding, (float, np.floating)):
            if np.isnan(embedding):
                return None
            vec = np.array([embedding], dtype=np.float32)
        elif isinstance(embedding, (list, np.ndarray)):
            vec = np.array(embedding, dtype=np.float32)
        else:
            return None

        # Normalize the vector (L2 norm)
        norm = np.linalg.norm(vec)
        if norm == 0:
            return None
        return vec / norm

    except Exception:
        return None

    


# -----------------------
# Validate if both embeddings exist and are valid (no None or NaN)
# -----------------------
def valid_embeddings(row):
    emb1 = parse_embedding(row.get("embedding1"))
    emb2 = parse_embedding(row.get("embedding2"))
    if emb1 is None or emb2 is None:
        return False
    if np.isnan(emb1).any() or np.isnan(emb2).any():
        return False
    return True

# -----------------------
# Load CSV files into dict of DataFrames, filtering for valid rows
# -----------------------
csv_files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]
organism_data = {}

for file in csv_files:
    df = pd.read_csv(os.path.join(input_dir, file))
    df_valid = df[df.apply(valid_embeddings, axis=1)].reset_index(drop=True)
    organism_data[file] = df_valid

# -----------------------
# Prepare output and total comparisons count for progress bar
# -----------------------
results = []
total_comparisons = 0
for org1, org2 in itertools.combinations(organism_data.keys(), 2):
    total_comparisons += min(K, len(organism_data[org1])) * min(J, len(organism_data[org2]))

pbar = tqdm(total=total_comparisons, desc="Comparing gene pairs", ncols=100)

# -----------------------
# Compare pairs across organisms
# -----------------------
for org1, org2 in itertools.combinations(organism_data.keys(), 2):
    df1 = organism_data[org1].head(K)
    df2 = organism_data[org2].head(J)

    for i, row1 in df1.iterrows():
        embA1 = parse_embedding(row1.get("embedding1"))
        embA2 = parse_embedding(row1.get("embedding2"))
        seqA1 = row1.get("seq1")
        seqA2 = row1.get("seq2")
        gene_pair_A = f"{row1.get('locus1', f'{org1}_locus1_{i}')}_{row1.get('locus2', f'{org1}_locus2_{i}')}"

        for j, row2 in df2.iterrows():
            embB1 = parse_embedding(row2.get("embedding1"))
            embB2 = parse_embedding(row2.get("embedding2"))
            seqB1 = row2.get("seq1")
            seqB2 = row2.get("seq2")
            gene_pair_B = f"{row2.get('locus1', f'{org2}_locus1_{j}')}_{row2.get('locus2', f'{org2}_locus2_{j}')}"

            # Check embedding length compatibility
            if len(embA1) != len(embB1) or len(embA2) != len(embB2):
                print(f"Warning: embedding length mismatch for pairs ({org1} {i}) vs ({org2} {j}), skipping")
                pbar.update(1)
                continue

            try:
                # Similarity: geneA1 vs geneB1
                index_a1_b1 = faiss.IndexFlatIP(len(embA1))
                index_a1_b1.add(embB1.reshape(1, -1))
                sim_a1_b1 = index_a1_b1.search(embA1.reshape(1, -1), 1)[0][0][0]

                # Similarity: geneA1 vs geneB2
                index_a1_b2 = faiss.IndexFlatIP(len(embA1))
                index_a1_b2.add(embB2.reshape(1, -1))
                sim_a1_b2 = index_a1_b2.search(embA1.reshape(1, -1), 1)[0][0][0]

                # Similarity: geneA2 vs geneB1
                index_a2_b1 = faiss.IndexFlatIP(len(embA2))
                index_a2_b1.add(embB1.reshape(1, -1))
                sim_a2_b1 = index_a2_b1.search(embA2.reshape(1, -1), 1)[0][0][0]

                # Similarity: geneA2 vs geneB2
                index_a2_b2 = faiss.IndexFlatIP(len(embA2))
                index_a2_b2.add(embB2.reshape(1, -1))
                sim_a2_b2 = index_a2_b2.search(embA2.reshape(1, -1), 1)[0][0][0]

                avg_sim_pair_1 = (sim_a1_b1 + sim_a2_b2)/2
                avg_sim_pair_2 = (sim_a2_b1 + sim_a1_b2)/2

                max_sim = max([avg_sim_pair_1,avg_sim_pair_2])

# Only include if above threshold
                if max_sim >= 0.95:
                    results.append([
                        org1, seqA1, seqA2, org2, 
                        seqB1, seqB2, i, j,
                        gene_pair_A, gene_pair_B,
                        max_sim
                    ])
            except Exception as e:
                print(f"Warning: FAISS failed for pairs ({org1} {i}) vs ({org2} {j}), skipping. Error: {e}")

            pbar.update(1)

pbar.close()

# -----------------------
# Save results with concatenated gene pairs
# -----------------------
results_df = pd.DataFrame(results, columns=[
    "Organism_A","SeqA1", "SeqA2", "Organism_B", "SeqB1",
    "SeqB2", "Pair_Index_A", "Pair_Index_B",
    "Gene_Pair_A", "Gene_Pair_B",
    "Max_Avg_Similarity"
])
results_df.to_csv(output_csv, index=False)

print(f"✅ Saved results to {output_csv}")
