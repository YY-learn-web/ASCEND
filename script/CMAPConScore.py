import pandas as pd
import numpy as np
from joblib import Parallel, delayed

def _calculate_score_a(
    disease_up_genes_ranks_sorted, # A sorted (ascending) 1-based rank list of disease-upregulated genes found in the compound spectrum
    p_original, # Original number of genes in the input disease upregulated gene set
    N_total_profile_genes # Total number of genes in the compound expression profile
):
    """
    Calculate score 'a' according to the formula: a = max_{m=1~p'} [m/p_original - R_m/N_total_profile_genes]
    - disease_up_genes_ranks_sorted: contains r_1, r_2, ..., r_p' (ranks sorted in ascending order)
    - p_original: 'p' in the formula (total number of genes in the disease-upregulated gene set)
    - N_total_profile_genes: 'n' in the formula
    - 'm' loops from 1 to p' (p_actual_count, i.e., the actual number of disease-upregulated genes found in the compound profile).
    - R_m is the m-th value in disease_up_genes_ranks_sorted.
    """
    if not disease_up_genes_ranks_sorted or p_original == 0:
        return 0.0

    p_actual_count = len(disease_up_genes_ranks_sorted) # Number of disease upregulated genes actually found in compound spectrum
    es_terms = []
    # The 'm' in the formula corresponds to 'j' in this loop (from 1 to p_actual_count).
    for j in range(1, p_actual_count + 1):
        rank_R_m = disease_up_genes_ranks_sorted[j-1] 
        term = (j / p_original) - (rank_R_m / N_total_profile_genes)
        es_terms.append(term)
    
    return np.max(es_terms) if es_terms else 0.0

def _calculate_score_b_internal_max(
    disease_down_genes_ranks_sorted, 
    q_original, # Original number of genes in the input disease downregulated gene set
    N_total_profile_genes 
):
    """
    Calculating the inner maximum part of score 'b': max_{m=1~q'} [R(m-1)/N - (m-1)/q_original]
    - disease_down_genes_ranks_sorted: contains s_1, s_2, ..., s_q' (ranks sorted in ascending order)
    - q_original: 'q' in the formula (total number of genes in the disease-downregulated gene set)
    - N_total_profile_genes: 'n' in the formula
    - The loop for 'm' goes from 1 to q' (q_actual_count, which is the actual number of disease-downregulated genes found in the compound profile).
    - R(m-1) is interpreted as s_m (the m-th value in disease_down_genes_ranks_sorted).
    - (m-1) is literally (iteration index - 1).
    """
    if not disease_down_genes_ranks_sorted or q_original == 0:
        return 0.0

    q_actual_count = len(disease_down_genes_ranks_sorted) # Number of disease-downregulated genes actually found in compound spectrum
    es_terms = []
    # The 'm' in the formula corresponds to 'j' in this loop (from 1 to q_actual_count).
    for j in range(1, q_actual_count + 1):
        rank_S_j = disease_down_genes_ranks_sorted[j-1]
        m_minus_1_val = j - 1
        
        term = (rank_S_j / N_total_profile_genes) - (m_minus_1_val / q_original)
        es_terms.append(term)
        
    return np.max(es_terms) if es_terms else 0.0


def calculate_cmap_score_from_image(disease_deg, compound_profile, n_jobs=-1):
    """
    Args:
        disease_deg (dict): A dictionary containing 'up' and 'down' keys.
                            Values are lists of gene IDs/symbols representing up-regulated and down-regulated genes in the disease.
        compound_profile (pd.DataFrame): DataFrame with compound identifiers as index,
                                         gene identifiers/symbols as columns, and gene expression values (e.g., z-scores) as values.
        n_jobs (int): Number of parallel jobs to run. -1 means using all processors.

    Returns:
        pd.DataFrame: DataFrame indexed by compound, containing 'CMap_Score', 'a_score', 'b_score' columns.
    """
    # Standardize disease feature gene names/IDs for consistent matching
    disease_up_set = set(str(g).strip().upper() for g in disease_deg['up'])
    disease_down_set = set(str(g).strip().upper() for g in disease_deg['down'])
    
    # Original p and q values (total number of genes in disease up/down-regulated gene sets)
    p_original = len(disease_up_set)
    q_original = len(disease_down_set)

    # Standardize gene names/IDs in compound spectrum
    compound_profile_genes = {str(col).strip().upper() for col in compound_profile.columns}
    
    # n: Total number of genes in the compound expression profile
    N_total_profile_genes = len(compound_profile_genes)

    if N_total_profile_genes == 0:
        raise ValueError("After normalization, the compound expression profile does not contain any genes.")


    normalized_compound_profile = compound_profile.copy()
    normalized_compound_profile.columns = [str(col).strip().upper() for col in compound_profile.columns]
    # normalized_compound_profile = normalized_compound_profile[list(compound_profile_genes)]


    def process_compound(compound_id):
        # Get expression data for the current compound (as Series)
        compound_data_series = normalized_compound_profile.loc[compound_id]
        
        # Create a sorted list of all genes in the compound profile (high expression to low expression)
        # The index of this sorted Series will be the standardized gene names
        ranked_genes_for_compound_list = compound_data_series.sort_values(ascending=False).index.tolist()

        # Create a mapping from gene to its 1-based rank for quick lookups
        gene_to_rank_map = {gene: i + 1 for i, gene in enumerate(ranked_genes_for_compound_list)}

        # --- 计算 a_score ---
        # Find the ranking of disease-upregulated genes present in the compound's spectrum
        up_genes_in_profile_ranks = []
        for gene in disease_up_set: # Iterate through normalized disease up-regulated genes
            if gene in gene_to_rank_map: # Check if it exists in the (normalized) spectrum of the current compound
                up_genes_in_profile_ranks.append(gene_to_rank_map[gene])
        up_genes_in_profile_ranks.sort() # Sort the ranks in ascending order: r_1, r_2, ...

        a_score = _calculate_score_a(
            up_genes_in_profile_ranks,
            p_original,
            N_total_profile_genes
        )

        # --- 计算 b_score ---
        # Rank of disease-downregulated genes found in the compound's spectrum
        down_genes_in_profile_ranks = []
        for gene in disease_down_set: # Iterate through normalized disease down-regulated genes
            if gene in gene_to_rank_map:
                down_genes_in_profile_ranks.append(gene_to_rank_map[gene])
        down_genes_in_profile_ranks.sort()
        
        b_internal_max = _calculate_score_b_internal_max(
            down_genes_in_profile_ranks,
            q_original,
            N_total_profile_genes
        )
        b_score = -b_internal_max
        
        # --- Calculate the final CMap score ---
        cmap_score = 0.0
        if a_score * b_score < 0:
            cmap_score = a_score - b_score
        
        return compound_id, cmap_score, a_score, b_score

    # Run the processing in parallel for all compounds
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_compound)(compound_idx) for compound_idx in normalized_compound_profile.index
    )

    # Run the processing in parallel for all compounds
    results_df = pd.DataFrame(results, columns=['Compound', 'CMap_Score', 'a_score', 'b_score'])
    return results_df.set_index('Compound')


# =======================
# Example usage
# =======================
gene_info = pd.read_csv('../data/geneinfo_beta.txt',sep='\t')
sym2id = dict(zip(gene_info['gene_symbol'], gene_info['gene_id'].astype(str)))
ensb2symbol = dict(zip(gene_info['ensembl_id'], gene_info['gene_id'].astype(str)))

# 1. Load disease DEGs
df_disease = pd.read_csv(f'../data/DrugRepurposing/DiseaseProfile/PAAD_DEGs.csv',index_col=0) # Change the path to the disease DEG file

df_disease.rename(index=ensb2symbol, inplace=True)
df_disease.rename(index=sym2id, inplace=True)

# 2. Read compound gene expression profiles
compound_profile = pd.read_csv(f'../data/DrugRepurposing/DrugResponseDEGs/PC3_PRISM_12328DEGs.csv', index_col=1)
compound_profile.columns = compound_profile.columns.astype(str)
ol_genes = list(set(compound_profile.columns) & set(df_disease.index))
print(f"Number of genes in the compound signature that overlap with the disease gene set: {len(ol_genes)}")
compound_profile = compound_profile.loc[:, ol_genes] 
df_disease = df_disease.loc[ol_genes, :] 
threshold = 3 # Adjustable
ups = df_disease[(df_disease['logFC'] > threshold) & (df_disease['adj.P.Val'] < 0.05)].index.tolist()
downs = df_disease[(df_disease['logFC'] < -threshold) & (df_disease['adj.P.Val'] < 0.05)].index.tolist()
print(f"Number of upregulated genes: {len(ups)}")
print(f"Number of downregulated genes: {len(downs)}")
disease_deg = {
    'up': ups,  
    'down': downs  
}

# compound_profile.drop(columns=['cell_iname'], inplace=True)
compound_profile = compound_profile.apply(lambda x: (x - x.mean()) / x.std(), axis=0)



# 3. Calculate CMap score
cmap_scores = calculate_cmap_score_from_image(disease_deg, compound_profile, n_jobs=5)

cmap_scores.sort_values(by='CMap_Score', ascending=False, inplace=True)

# Output the top 10 most promising compounds
print(cmap_scores.head(10))

# Save results to file
output_file = f'../data/...' # Change save path
cmap_scores.to_csv(output_file, sep='\t')
