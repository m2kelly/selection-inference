
#filter warnings
import warnings

import pandas as pd
warnings.filterwarnings("ignore")
import pickle
from joblib import Parallel, delayed
import qvalues as qv
import time
from pathlib import Path
from predictor import Predictor
import essentials as es
from graphpoint import GraphPointAbstract as GPA
from parse_genes_pkl import load_codon_table, summarize_genes

def get_sig_genes(qvals_file):
    qvals_df = pd.read_csv(qvals_file, sep='\t', index_col=0, skiprows=1)
    qvals_df_sig = qvals_df[qvals_df['q_phi_neg'] < 0.05]
    qvals_df_sig = qvals_df_sig[qvals_df_sig['m_obs'] >= 1]
    #sort by dn/ds ascending
    #qvals_df_sig = qvals_df_sig.sort_values(by='d(m+k)/ds', ascending=True)
    
    #sample 100 genes randomly
    #qvals_df_sig = qvals_df_sig.sample(100)
    return qvals_df_sig.index.tolist()

def calculate_general_muts_matrix_old(pop_size):
    calculate_general_muts_matrix(pop_size)

def calculate_general_muts_matrix(pop_size):
    mutation_matrix_path = Path('files/best_in_df_mutations_4_allele.csv')
    try:
        muts_matrix = pd.read_csv(mutation_matrix_path, index_col=0, header=0, sep='\t')
    except FileNotFoundError:
        from cuda_operations import get_mut_matrix_from_sub_matrix

        sub_matrix = pd.read_csv('files/best_in_df.csv', index_col=0, header=0, sep='\t')
        muts_matrix = get_mut_matrix_from_sub_matrix(sub_matrix, pop_size)

    muts_matrix.to_csv(mutation_matrix_path, sep='\t')

if __name__ == '__main__':

    run_name = 'cactus'
    episode = 'Anc4->hg38'
    population_size = 200000

    qvals_file = f'Output/{run_name}/{episode}/q_values_{run_name}.txt'
    sig_genes = get_sig_genes(qvals_file)
    try: analyzed_genes = pd.read_csv('analyzed_genes.csv', sep='\t', index_col=0).index
    except FileNotFoundError: analyzed_genes = []
     #TP53,BRCA1
    #sig_genes=['ENSG00000198626.19']
    #TP53,BRCA1
    #from cbase run with gene coding, this is mut matrix of trinucs->alt
    #edit so input trinucs count and df of trinucs->alts for allowed syn muts per gene
    occs_dfs = pickle.load(open(f'files/trinucs.pkl', 'rb')) #occs_dfs = {'SMG6': occs_dfs['SMG6'], 'AANAT': occs_dfs['AANAT'], 'BRCA1': occs_dfs['BRCA1']}
    #syn_target_dfs=pickle.load(open(f'Output/{run_name}/{episode}/genes/occs_per_gene.pkl', 'rb'))
    syn_target_dfs=pickle.load(open('files/Anc4_syn_frac.pkl', 'rb'))
    valid_genes=set(occs_dfs.keys()) & set(syn_target_dfs.keys())
    
    sig_genes = [s for s in sig_genes if s in occs_dfs]
    sig_genes=set(sig_genes) & set(valid_genes)
    
    #sig_genes = ['ENSG00000012048.26','ENSG00000141510.19'] #these genes not sig?
    print(sig_genes)  
    #maria not sure what this is doing
    #sig_genes=sig_genes[:int((len(sig_genes)*1)/3)]
    print(sig_genes)
    print(len(sig_genes) - len(analyzed_genes))

    occs_dfs = {gene: occs_dfs[gene] for gene in sig_genes if gene not in analyzed_genes}
    syn_target_dfs = {gene: syn_target_dfs[gene] for gene in sig_genes if gene not in analyzed_genes}
    
    working_dir = f'Output/{run_name}/{episode}'
    output_name = f'{run_name}'
    output_dir = f'Output/{run_name}'
    with open(f'{working_dir}/genes/genes.pkl', 'rb') as handle:
        gene_frames = pickle.load(handle)
    site_summary, _ = summarize_genes(
        gene_frames,
        load_codon_table(Path(__file__).parent / 'Auxiliary' / 'codon_table.json'),
        genes=sig_genes,
    )
    nonsyn_sites_by_gene = {gene: [] for gene in sig_genes}
    for site in site_summary.to_dict(orient='records'):
        nonsyn_sites_by_gene[site['gene']].append({
            'trinucleotide': site['trinucleotide'],
            'reference_base': site['reference_base'],
            'nonsyn_alt_bases': site['nonsyn_alt_bases'],
            'count': site['count'],
        })
    print('calculating mutation matrix from transition matrix')
    calculate_general_muts_matrix(population_size)

    start = time.time()
    qv.compute_q_values(
        working_dir,
        output_name,
        output_dir,
        occs_dfs,
        syn_target_dfs,
        population_size,
        nonsyn_sites_by_gene,
    )
    print(f'elabsed time = {time.time() - start}')
