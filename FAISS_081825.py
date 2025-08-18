#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Two-stage KNN for conserved PPI search across organisms using PLM embeddings.

Workflow (Option A):
- Load per-organism CSVs that contain *pair rows*: (seq1, seq2, locus1, locus2, embedding1, embedding2).
- Canonicalize & extract a *single-protein* table per organism (dedup proteins).
- Build one FAISS index per organism over single-protein embeddings (L2-normalized).
- For each pair (a1,a2) in org A, query org B’s index:
    B1 = top-k neighbors of a1, B2 = top-k neighbors of a2
  Score only the k×k candidate pairs with the correct order-invariant score:
    score = max( (a1·b + a2·b')/2, (a1·b' + a2·b)/2 )
- Keep the best (b,b') per query pair and write results.

Assumptions:
- Embeddings are semicolon-separated strings (or lists) and will be L2-normalized here.
- Input directory structure:
    input_dir/
      ecoli.csv
      mtb.csv
      ...
  where each CSV has columns:
    ["locus1","locus2","seq1","seq2","embedding1","embedding2", ...]
- Output: a CSV of best matches and scores per cross-organism pair.

Usage:
  python pair_search_knn.py \
      --input_dir ./plm_embeddings \
      --output_csv conserved_pairs_knn.csv \
      --k 50 --min_score 0.85
"""

import os
import argparse
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
import faiss

# -----------------------
# Parsing / normalization
# -----------------------

def parse_embedding(x):
    """Parse embedding from str/list/np.ndarray to float32 np.array and L2-normalize."""
    if x is None:
        return None
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        vec = np.array([float(v) for v in s.split(";")], dtype=np.float32)
    elif isinstance(x, (list, tuple, np.ndarray)):
        vec = np.asarray(x, dtype=np.float32)
    else:
        # handle weird types like np.float32 scalar (not expected here)
        try:
            vec = np.array([float(x)], dtype=np.float32)
        except Exception:
            return None

    if not np.all(np.isfinite(vec)):
        return None
    n = np.linalg.norm(vec)
    if n == 0.0:
        return None
    return vec / n


def valid_pair_row(row):
    """Return True if both embeddings parse & are finite."""
    e1 = parse_embedding(row.get("embedding1"))
    e2 = parse_embedding(row.get("embedding2"))
    return (e1 is not None) and (e2 is not None)


# -----------------------
# Single-protein table
# -----------------------

def build_protein_table_from_pairs(df_pairs, org_name):
    """
    From a pair-rows DataFrame, produce a single-protein table with columns:
      ["protein_id","organism","locus","seq","emb"]
    Deduplicates proteins by 'locus' (falls back to sequence if locus missing).
    """
    records = []

    # left protein
    for _, r in df_pairs.iterrows():
        emb = parse_embedding(r["embedding1"])
        if emb is None: 
            continue
        locus = r.get("locus1")
        seq = r.get("seq1")
        if pd.isna(locus) or locus is None:
            # fallback canonical id if locus missing
            locus = f"{org_name}::seq1::{hash(seq)%10**9}"
        records.append({"protein_id": f"{org_name}::{locus}",
                        "organism": org_name, "locus": locus, "seq": seq, "emb": emb})

    # right protein
    for _, r in df_pairs.iterrows():
        emb = parse_embedding(r["embedding2"])
        if emb is None:
            continue
        locus = r.get("locus2")
        seq = r.get("seq2")
        if pd.isna(locus) or locus is None:
            locus = f"{org_name}::seq2::{hash(seq)%10**9}"
        records.append({"protein_id": f"{org_name}::{locus}",
                        "organism": org_name, "locus": locus, "seq": seq, "emb": emb})

    prot_df = pd.DataFrame.from_records(records)

    # dedup by locus; keep first
    prot_df = prot_df.drop_duplicates(subset=["protein_id"]).reset_index(drop=True)
    # stack embeddings into a matrix for FAISS
    emb_mat = np.stack(prot_df["emb"].values, axis=0).astype(np.float32)
    d = emb_mat.shape[1]
    return prot_df, emb_mat, d


# -----------------------
# FAISS index helpers
# -----------------------

def build_faiss_index(emb_mat, use_ivf=False, nlist=1024):
    """
    Build a FAISS index over L2-normalized embeddings using inner product.
    If use_ivf=True, builds IVF+Flat for larger corpora.
    """
    d = emb_mat.shape[1]
    if not use_ivf:
        index = faiss.IndexFlatIP(d)
        index.add(emb_mat)
        return index
    # IVF coarse quantizer + Flat for good recall/speed tradeoff
    quant = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quant, d, nlist, faiss.METRIC_INNER_PRODUCT)
    assert not index.is_trained
    index.train(emb_mat)
    index.add(emb_mat)
    # sensible default; caller may override
    index.nprobe = min(32, nlist)
    return index


# -----------------------
# Order-invariant pair score
# -----------------------

def order_invariant_pair_score(a1, a2, B, Bprime):
    """
    Compute the k×k matrix of order-invariant scores between (a1,a2) and
    candidate sets B (neighbors of a1) and Bprime (neighbors of a2).

    Returns:
      score_mat: (k,k) with score[i,j] = max( (a1·B[i] + a2·Bprime[j])/2,
                                             (a1·Bprime[j] + a2·B[i])/2 )
    """
    # a1, a2: (d,)
    # B:      (k,d)
    # Bprime: (k,d)

    # Similarity vectors (cosine == dot because embeddings are L2-normalized)
    S1 = B @ a1              # (k,)
    S2 = Bprime @ a2         # (k,)
    S3 = Bprime @ a1         # (k,)
    S4 = B @ a2              # (k,)

    # Broadcast to (k,k)
    pairing1 = (S1[:, None] + S2[None, :]) * 0.5
    pairing2 = (S3[None, :] + S4[:, None]) * 0.5
    return np.maximum(pairing1, pairing2)


# -----------------------
# Main
# -----------------------

def main(args):
    # Load organism CSVs (pair rows), filter to valid rows
    csv_files = [f for f in os.listdir(args.input_dir) if f.endswith(".csv")]
    if len(csv_files) < 2:
        raise SystemExit("Need at least two organism CSVs in --input_dir.")

    org_pairs = {}
    for f in csv_files:
        org = os.path.splitext(f)[0]
        df = pd.read_csv(os.path.join(args.input_dir, f))
        df = df[df.apply(valid_pair_row, axis=1)].reset_index(drop=True)
        if df.empty:
            continue
        # Canonicalize pair id to avoid (a,b)/(b,a) duplicates inside an organism file
        def canon_pair_id(r):
            l1 = str(r.get("locus1", f"{org}_locus1"))
            l2 = str(r.get("locus2", f"{org}_locus2"))
            return "|".join(sorted([l1, l2]))
        df["pair_id"] = df.apply(canon_pair_id, axis=1)
        df = df.drop_duplicates(subset=["pair_id"]).reset_index(drop=True)
        org_pairs[org] = df

    if len(org_pairs) < 2:
        raise SystemExit("After filtering, fewer than two organisms had valid data.")

    # Build single-protein tables + FAISS per organism
    org_prot = {}
    for org, df_pairs in org_pairs.items():
        prot_df, emb_mat, d = build_protein_table_from_pairs(df_pairs, org)
        if emb_mat.size == 0:
            continue
        index = build_faiss_index(
            emb_mat,
            use_ivf=args.use_ivf,
            nlist=args.ivf_nlist
        )
        # Store
        org_prot[org] = {
            "proteins": prot_df,
            "emb": emb_mat,
            "dim": d,
            "index": index
        }

    if len(org_prot) < 2:
        raise SystemExit("Fewer than two organisms had protein embeddings for indexing.")

    # Prepare output
    out_rows = []
    org_pairs_list = sorted(org_pairs.keys())
    pairs_to_compare = list(itertools.combinations(org_pairs_list, 2))
    pbar = tqdm(total=len(pairs_to_compare), desc="Org-org comparisons", ncols=90)

    for orgA, orgB in pairs_to_compare:
        dfA = org_pairs[orgA]
        dfB = org_pairs[orgB]  # only needed for metadata joins

        # indices for B
        idxB = org_prot[orgB]["index"]
        embB = org_prot[orgB]["emb"]
        protB = org_prot[orgB]["proteins"]

        # iterate pairs in A (optionally head() to cap)
        dfA_iter = dfA.head(args.max_pairs_A) if args.max_pairs_A > 0 else dfA

        for _, rowA in dfA_iter.iterrows():
            a1 = parse_embedding(rowA["embedding1"])
            a2 = parse_embedding(rowA["embedding2"])
            if a1 is None or a2 is None:
                continue

            # KNN for a1 and a2 in orgB
            # ensure correct shape (1,d)
            q1 = a1.reshape(1, -1).astype(np.float32)
            q2 = a2.reshape(1, -1).astype(np.float32)

            D1, I1 = idxB.search(q1, args.k)  # (1,k)
            D2, I2 = idxB.search(q2, args.k)  # (1,k)

            # gather candidate embeddings
            B  = embB[I1[0]]  # (k,d)
            Bp = embB[I2[0]]  # (k,d)

            # compute k×k order-invariant score matrix
            score_mat = order_invariant_pair_score(a1, a2, B, Bp)
            max_idx = np.unravel_index(np.argmax(score_mat), score_mat.shape)
            best_i, best_j = int(max_idx[0]), int(max_idx[1])
            best_score = float(score_mat[best_i, best_j])

            if best_score < args.min_score:
                continue

            # metadata for best neighbors
            nb_i = protB.iloc[I1[0, best_i]]
            nb_j = protB.iloc[I2[0, best_j]]

            out_rows.append({
                "orgA": orgA,
                "orgB": orgB,
                "pairA_locus1": rowA.get("locus1"),
                "pairA_locus2": rowA.get("locus2"),
                "pairA_seq1": rowA.get("seq1"),
                "pairA_seq2": rowA.get("seq2"),
                "pairA_id": rowA.get("pair_id"),
                "bestB_locus_from_a1": nb_i["locus"],
                "bestB_locus_from_a2": nb_j["locus"],
                "best_score": best_score
            })

        pbar.update(1)

    pbar.close()

    if not out_rows:
        print("No matches above min_score; writing empty file.")
    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(args.output_csv, index=False)
    print(f"✅ Saved results to {args.output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Folder with per-organism CSVs (pair rows).")
    parser.add_argument("--output_csv", type=str, default="conserved_pairs_knn.csv")
    parser.add_argument("--k", type=int, default=50, help="Top-k neighbors per protein.")
    parser.add_argument("--min_score", type=float, default=0.85,
                        help="Minimum order-invariant pair score to report.")
    parser.add_argument("--max_pairs_A", type=int, default=0,
                        help="Cap pairs from orgA (0 = all).")
    # IVF options (off by default; useful for large corpora)
    parser.add_argument("--use_ivf", action="store_true",
                        help="Use IVF+Flat index instead of FlatIP.")
    parser.add_argument("--ivf_nlist", type=int, default=1024,
                        help="Number of IVF lists (coarse centroids).")
    args = parser.parse_args()
    main(args)
