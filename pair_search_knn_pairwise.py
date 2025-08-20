import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import faiss

# -----------------------
# Utilities
# -----------------------

def load_embeddings(csv_file):
    """Load CSV file with protein pairs and embeddings (semicolon-delimited vectors)."""
    df = pd.read_csv(csv_file)

    def parse_embedding(x):
        if pd.isna(x):  # handle NaN
            return np.array([], dtype=np.float32)
        s = str(x).strip()
        if not s:  # handle empty string
            return np.array([], dtype=np.float32)
        try:
            arr = np.fromstring(s, sep=";")
            return arr.astype(np.float32)
        except Exception:
            print(f"⚠️ Warning: could not parse embedding: {x}")
            return np.array([], dtype=np.float32)

    df["embedding1"] = df["embedding1"].apply(parse_embedding)
    df["embedding2"] = df["embedding2"].apply(parse_embedding)

    # Drop rows where either embedding is empty
    df = df[df["embedding1"].apply(len) > 0]
    df = df[df["embedding2"].apply(len) > 0]

    return df


def build_pair_arrays(df, org):
    """Extract arrays of embeddings and metadata for an organism."""
    B1 = np.vstack(df["embedding1"].values)
    B2 = np.vstack(df["embedding2"].values)

    # Ensure we always have a unique pair_id
    if "pair_id" in df.columns:
        meta = df[["pair_id", "locus1", "locus2", "seq1", "seq2"]].copy()
    else:
        meta = df[["locus1", "locus2", "seq1", "seq2"]].copy()
        meta.insert(0, "pair_id", [f"{org}_pair{i}" for i in range(len(df))])

    return B1, B2, meta


def cosine_faiss(x, y_index, k):
    """Search FAISS index for top-k cosine similarity matches."""
    # Normalize x before search
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    sims, idxs = y_index.search(x.astype('float32'), k)
    return sims, idxs


def build_faiss_index(vectors):
    """Build a FAISS index for cosine similarity."""
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product index
    # Normalize vectors for cosine similarity
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    index.add(vectors.astype('float32'))
    return index


# -----------------------
# Main logic
# -----------------------

def main(args):
    # Collect all CSVs in input_dir
    org_files = [f for f in os.listdir(args.input_dir) if f.endswith('.csv')]
    orgs = {os.path.splitext(f)[0]: os.path.join(args.input_dir, f) for f in org_files}

    if len(orgs) < 2:
        raise ValueError("Need at least 2 organisms (CSV files) in input_dir")

    results = []

    org_names = list(orgs.keys())

    # Compare each pair of organisms
    for i in range(len(org_names)):
        for j in range(i+1, len(org_names)):
            orgA, orgB = org_names[i], org_names[j]

            print(f"🔍 Comparing {orgA} vs {orgB}")

            dfA = load_embeddings(orgs[orgA])
            A1 = np.vstack(dfA["embedding1"].values)
            A2 = np.vstack(dfA["embedding2"].values)

            dfB = load_embeddings(orgs[orgB])
            B1, B2, metaB = build_pair_arrays(dfB, orgB)

            # Build FAISS indexes for B1 and B2
            indexB1 = build_faiss_index(B1)
            indexB2 = build_faiss_index(B2)

            for k in tqdm(range(len(A1)), desc=f"Pairs in {orgA}"):
                a1 = A1[k].reshape(1, -1)
                a2 = A2[k].reshape(1, -1)

                # Search candidates separately for a1 and a2 against B1 and B2
                sims1, idxs1 = cosine_faiss(a1, indexB1, args.top_n * 5)
                sims2, idxs2 = cosine_faiss(a2, indexB2, args.top_n * 5)

                candidates = set(idxs1.flatten()).union(set(idxs2.flatten()))

                scores = []
                for idx in candidates:
                    b1 = B1[idx].reshape(1, -1)
                    b2 = B2[idx].reshape(1, -1)

                    # Normalize for cosine
                    a1n = a1 / np.linalg.norm(a1)
                    a2n = a2 / np.linalg.norm(a2)
                    b1n = b1 / np.linalg.norm(b1)
                    b2n = b2 / np.linalg.norm(b2)

                    o1 = (np.dot(a1n, b1n.T)[0,0] + np.dot(a2n, b2n.T)[0,0]) / 2
                    o2 = (np.dot(a1n, b2n.T)[0,0] + np.dot(a2n, b1n.T)[0,0]) / 2

                    if o1 >= o2:
                        score, s1, s2, orient = o1, o1, o2, 1
                    else:
                        score, s1, s2, orient = o2, o1, o2, 2

                    if score >= args.min_score:
                        scores.append((idx, score, s1, s2, orient))

                # Sort candidates
                scores.sort(key=lambda x: x[1], reverse=True)
                top_scores = scores[:args.top_n]

                for rank, (idx, score, s1, s2, orient) in enumerate(top_scores, 1):
                    results.append({
                        "orgA": orgA,
                        "orgA_pair_id": dfA.loc[k, "pair_id"] if "pair_id" in dfA.columns else f"{orgA}_pair{k}",
                        "orgA_locus1": dfA.iloc[k]["locus1"],
                        "orgA_locus2": dfA.iloc[k]["locus2"],
                        "orgB": orgB,
                        "orgB_pair_id": metaB.iloc[idx]["pair_id"],
                        "orgB_locus1": metaB.iloc[idx]["locus1"],
                        "orgB_locus2": metaB.iloc[idx]["locus2"],
                        "score": float(score),
                        "comp_score_o1": float(s1),
                        "comp_score_o2": float(s2),
                        "score_orientation": int(orient),
                        "rank_in_B_pairs": int(rank)
                    })

    # Save results
    out_df = pd.DataFrame(results)
    out_df.to_csv(args.output_csv, index=False)
    print(f"✅ Results saved to {args.output_csv}")


# -----------------------
# CLI
# -----------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pairwise protein pair similarity search (FAISS accelerated)")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing organism CSVs")
    parser.add_argument("--output_csv", type=str, default="pairwise_similarity.csv", help="Output CSV file")
    parser.add_argument("--top_n", type=int, default=5, help="Number of top matches to keep per pair")
    parser.add_argument("--min_score", type=float, default=0.85, help="Minimum cosine similarity to keep")
    args = parser.parse_args()
    main(args)
