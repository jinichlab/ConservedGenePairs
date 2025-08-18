import pandas as pd
import numpy as np
from scipy.special import stdtr
import os
import glob
from tqdm import tqdm
import gc

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

class GLSRegression:
    def __init__(self, screens):
        self.screens_df = pd.DataFrame(screens).fillna(0)
        self.Warped_Screens = None
        self.Warped_Intercept = None
        self.GLS_coef = None
        self.GLS_se = None
        self.GLS_pW_df = None

    def whiten_data(self):
        cov_matrix = np.cov(self.screens_df.T)
        cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-8
        try:
            cholsigmainv = np.linalg.cholesky(np.linalg.inv(cov_matrix))
        except np.linalg.LinAlgError:
            raise ValueError("Covariance matrix is singular even after regularization")
        self.Warped_Screens = self.screens_df.values @ cholsigmainv
        self.Warped_Intercept = cholsigmainv.sum(axis=0)

    def linear_regression(self):
        num_genes = self.Warped_Screens.shape[0]
        num_samples = self.Warped_Screens.shape[1]
        self.GLS_coef = np.zeros((num_genes, num_genes))
        self.GLS_se = np.zeros((num_genes, num_genes))
        df = num_samples - 2

        for gene_index in tqdm(range(num_genes), desc="Running GLS linear regression"):
            x = np.column_stack([np.ones(num_samples), self.Warped_Screens[gene_index]])
            try:
                xtx_inv = np.linalg.inv(x.T @ x)
            except np.linalg.LinAlgError:
                xtx_inv = np.linalg.pinv(x.T @ x)

            Y = self.Warped_Screens.T
            coef = xtx_inv @ x.T @ Y
            self.GLS_coef[gene_index, :] = coef[1, :]
            Y_pred = x @ coef
            residuals = Y - Y_pred
            ssr = np.sum(residuals**2, axis=0)
            sigma_squared = ssr / df
            se_slope = np.sqrt(sigma_squared * xtx_inv[1, 1])
            self.GLS_se[gene_index, :] = se_slope

    def calculate_p_values(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            t_stat = np.divide(self.GLS_coef, self.GLS_se,
                               out=np.zeros_like(self.GLS_coef),
                               where=self.GLS_se != 0)

        df = self.screens_df.shape[1] - 2
        GLS_pW = 2 * (1 - stdtr(df, np.abs(t_stat)))
        np.fill_diagonal(GLS_pW, 1)
        self.GLS_pW_df = pd.DataFrame(GLS_pW)

def analyze_fitness(GLS_coef, significant_results, locus_ids):
    if len(significant_results) == 0:
        return pd.DataFrame(columns=['gene1', 'gene2', 'fitness_metric', 'locus1', 'locus2'])

    g1 = significant_results['gene1_idx'].to_numpy()
    g2 = significant_results['gene2_idx'].to_numpy()
    fitness_metric = (GLS_coef[g1, g2] + GLS_coef[g2, g1]) / 2

    return pd.DataFrame({
        'gene1': [f"Gene_{i}" for i in g1],
        'gene2': [f"Gene_{i}" for i in g2],
        'locus1': [locus_ids[i] for i in g1],
        'locus2': [locus_ids[j] for j in g2],
        'fitness_metric': fitness_metric
    })

def get_top_gls_pairs(GLS_coef, significant_pairs, locus_ids, top_n=100):
    pairs = []
    for _, row in significant_pairs.iterrows():
        i, j = int(row['gene1_idx']), int(row['gene2_idx'])
        avg_coef = (GLS_coef[i, j] + GLS_coef[j, i]) / 2
        pairs.append({
            'gene1_index': i,
            'gene2_index': j,
            'gene1_name': f"Gene_{i}",
            'gene2_name': f"Gene_{j}",
            'locus1': locus_ids[i],
            'locus2': locus_ids[j],
            'avg_gls_coef': avg_coef
        })
    return pd.DataFrame(sorted(pairs, key=lambda x: abs(x['avg_gls_coef']), reverse=True)[:top_n])

def run_gls_on_file(tsv_path):
    print(f"\nProcessing: {tsv_path}")
    base_name = os.path.splitext(os.path.basename(tsv_path))[0]

    try:
        data = pd.read_csv(tsv_path, sep='\t')
        print(f"Loaded data shape: {data.shape}")
    except Exception as e:
        print(f"Failed to read {tsv_path}: {e}")
        return

    if 'locusId' not in data.columns:
        print(f"Missing 'locusId' column in {tsv_path}, skipping.")
        return

    locus_ids = data['locusId'].tolist()
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    screens_data = data[numeric_cols]

    if screens_data.shape[0] < 3 or screens_data.shape[1] < 3:
        print(f"Insufficient data in {tsv_path}")
        return

    gls_model = GLSRegression(screens_data)

    try:
        gls_model.whiten_data()
        gls_model.linear_regression()
        gls_model.calculate_p_values()

        gls_model.GLS_pW_df.index = locus_ids
        gls_model.GLS_pW_df.columns = locus_ids

        # Get upper triangle of p-values
        p_matrix = gls_model.GLS_pW_df.values
        n_genes = p_matrix.shape[0]
        upper_tri_rows, upper_tri_cols = np.triu_indices(n_genes, k=1)
        upper_tri_p_values = p_matrix[upper_tri_rows, upper_tri_cols]

        # Get top 5000 p-values
        if len(upper_tri_p_values) > 5000:
            top_indices = np.argsort(upper_tri_p_values)[:5000]
        else:
            top_indices = np.argsort(upper_tri_p_values)

        sig_rows = upper_tri_rows[top_indices]
        sig_cols = upper_tri_cols[top_indices]
        sig_pvals = upper_tri_p_values[top_indices]

        sig_results = pd.DataFrame({
            'gene1_idx': sig_rows,
            'gene2_idx': sig_cols,
            'p_value': sig_pvals,
            'locus1': [locus_ids[i] for i in sig_rows],
            'locus2': [locus_ids[j] for j in sig_cols]
        })

        # Save filtered p-values
        ensure_dir("p_values")
        sig_results.to_csv(f"p_values/{base_name}_p_values.csv", index=False)

        # Fitness results
        fitness_results = analyze_fitness(gls_model.GLS_coef, sig_results, locus_ids)
        ensure_dir("fitness_results")
        fitness_results.to_csv(f"fitness_results/{base_name}_fitness_results.csv", index=False)

        # Top GLS pairs
        top_gls_pairs = get_top_gls_pairs(gls_model.GLS_coef, sig_results, locus_ids, top_n=100)
        ensure_dir("top_gls_pairs")
        top_gls_pairs.to_csv(f"top_gls_pairs/{base_name}_top_gls_pairs.csv", index=False)

        # Full coefficient matrix
        coef_df = pd.DataFrame(gls_model.GLS_coef, index=locus_ids, columns=locus_ids)
        ensure_dir("gls_coefficients")
        coef_df.to_csv(f"gls_coefficients/{base_name}_gls_coefficients.csv")

        del gls_model.GLS_pW_df
        gc.collect()

        print(f"✔ Done: {base_name} — {len(sig_results)} top p-value gene pairs")

    except Exception as e:
        print(f"Error in {tsv_path}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    folder_path = "./organism_fitness_data"
    tsv_files = sorted(glob.glob(os.path.join(folder_path, "*.tsv")))

    if not tsv_files:
        print("No .tsv files found in './organism_fitness_data'")
    else:
        for tsv_file in tqdm(tsv_files, desc="Processing files"):
            run_gls_on_file(tsv_file)
    print("\nAll files processed.")
