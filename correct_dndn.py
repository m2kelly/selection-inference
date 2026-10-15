
#filter warnings
import warnings

import argparse

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



def calculate_general_muts_matrix(pop_size, force=False):
    mutation_matrix_path = Path('files/best_in_df_mutations_4_allele.csv')
    try:
        if force:
            raise FileNotFoundError
        muts_matrix = pd.read_csv(mutation_matrix_path, index_col=0, header=0, sep='\t')
    except FileNotFoundError:
        from cuda_operations import get_mut_matrix_from_sub_matrix

        sub_matrix = pd.read_csv('files/best_in_df.csv', index_col=0, header=0, sep='\t')
        muts_matrix = get_mut_matrix_from_sub_matrix(sub_matrix, pop_size)

    muts_matrix.to_csv(mutation_matrix_path, sep='\t')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--correct-rates-for-recurrence', action='store_true',
        help=('infer files/best_in_df.csv from files/rates using Q**generations '
              'before running the existing pipeline'),
    )
    parser.add_argument('--rates', default='files/rates')
    parser.add_argument('--trinucs', default='files/trinucs')
    parser.add_argument('--generations', type=int, default=160000)
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()
    if args.correct_rates_for_recurrence:
        from cuda_operations import get_transition_cpmatrix_from_rates

        rates = pd.read_csv(args.rates, index_col=0, sep='\t')
        trinucs = pd.read_csv(args.trinucs, index_col=0, sep='\t')
        sub_matrix = get_transition_cpmatrix_from_rates(
            rates, trinucs, generations=args.generations
        )
        sub_matrix.to_csv('files/best_in_df.csv', sep='\t')

        #test for comparison
        '''
        from cuda_operations import get_transition_cpmatrix_from_rates_greedy
        
        rates = pd.read_csv(args.rates, index_col=0, sep='\t')
        trinucs = pd.read_csv(args.trinucs, index_col=0, sep='\t')
        sub_matrix = get_transition_cpmatrix_from_rates_greedy(
            rates, trinucs, generations=args.generations
        )
        sub_matrix.to_csv('files/best_in_df_greedy.csv', sep='\t')
        '''
        
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

    print(len(sig_genes) - len(analyzed_genes))

    occs_dfs = {gene: occs_dfs[gene] for gene in sig_genes if gene not in analyzed_genes}
    syn_target_dfs = {gene: syn_target_dfs[gene] for gene in sig_genes if gene not in analyzed_genes}
    
    working_dir = f'Output/{run_name}/{episode}'
    output_name = f'{run_name}'
    output_dir = f'Output/{run_name}'
    with open(f'{working_dir}/genes/genes.pkl', 'rb') as handle:
        gene_frames = pickle.load(handle)

    #from parse genes_pkl.py to take codon and trinucs saved in cbase run
    #then calculate number of non syn mutations per gene
    #summarising into tallys of number of trinucs with each/alls alts as non syn
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
    #using 4 allele model+inputted N to infer mutation matrix
    calculate_general_muts_matrix(
        population_size, force=args.correct_rates_for_recurrence
    )

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
