
#filter warnings
import warnings

import pandas as pd
warnings.filterwarnings("ignore")
import pickle
from joblib import Parallel, delayed
import qvalues as qv
import time
from predictor import Predictor
import essentials as es
from graphpoint import GraphPointAbstract as GPA

def get_sig_genes(qvals_file):
    qvals_df = pd.read_csv(qvals_file, sep='\t', index_col=0, skiprows=1)
    qvals_df_sig = qvals_df[qvals_df['q_phi_neg'] < 0.05]
    qvals_df_sig = qvals_df_sig[qvals_df_sig['m_obs'] >= 1]
    #sort by dn/ds ascending
    qvals_df_sig = qvals_df_sig.sort_values(by='d(m+k)/ds', ascending=True)
    
    #sample 100 genes randomly
    #qvals_df_sig = qvals_df_sig.sample(100)
    return qvals_df_sig.index.tolist()

def calculate_general_muts_matrix_old(pop_size):
    #check if the csv file exists
    try: muts_matrix = pd.read_csv('files/best_in_df_mutations.csv', index_col=0, header=0, sep='\t')
    except FileNotFoundError:
        muts_matrix = pd.DataFrame(0, index=es.CONTEXTS, columns=es.CONTEXTS)
        sub = pd.read_csv('files/best_in_df.csv', index_col=0, header=0, sep='\t')
        p = Predictor()
        for tria in es.CONTEXTS:
            for trib in es.CONTEXTS:
                subf = sub[tria][trib]
                if subf == 0: continue
                if subf > 0.5: continue
                subb = sub[trib][tria]
                point = GPA(0, subb, pop_size, mean_sub_rate=subf)
                mu = p.get_mu(point)
                muts_matrix[tria][trib] = mu


        for tri in es.CONTEXTS:
            muts_matrix[tri][tri] = 1-muts_matrix.T[tri].sum()

        muts_matrix.to_csv('files/best_in_df_mutations.csv', sep='\t')

def calculate_general_muts_matrix(pop_size):
    try: muts_matrix = pd.read_csv('files/best_in_df_mutations.csv', index_col=0, header=0, sep='\t')
    except FileNotFoundError:
        from cuda_operations import get_mut_matrix_from_sub_matrix

        sub_matrix = pd.read_csv('files/best_in_df.csv', index_col=0, header=0, sep='\t')
        muts_matrix = get_mut_matrix_from_sub_matrix(sub_matrix, pop_size)

    muts_matrix.to_csv('files/best_in_df_mutations.csv', sep='\t')

if __name__ == '__main__':

    run_name = 'cactus'
    episode = 'Anc4->hg38'
    population_size = 220000

    qvals_file = f'Output/{run_name}/{episode}/q_values_{run_name}.txt'
    sig_genes = get_sig_genes(qvals_file)
    try: analyzed_genes = pd.read_csv('analyzed_genes.csv', sep='\t', index_col=0).index
    except FileNotFoundError: analyzed_genes = []
     #TP53,BRCA1
    #sig_genes = ['ENSG00000141510.19','ENSG00000012048.26'] #these genes not sig?
    sig_genes=['ENSG00000198626.19']
    #TP53,BRCA1
    #from cbase run with gene coding, this is mut matrix of trinucs->alt
    #edit so input trinucs count and df of trinucs->alts for allowed syn muts per gene
    occs_dfs = pickle.load(open(f'files/trinucs.pkl', 'rb')) #occs_dfs = {'SMG6': occs_dfs['SMG6'], 'AANAT': occs_dfs['AANAT'], 'BRCA1': occs_dfs['BRCA1']}
    #syn_target_dfs=pickle.load(open(f'Output/{run_name}/{episode}/genes/occs_per_gene.pkl', 'rb'))
    syn_target_dfs=pickle.load(open('files/Anc4_syn_frac.pkl', 'rb'))
    
    
    sig_genes = [s for s in sig_genes if s in occs_dfs]
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
    print('calculating mutation matrix from transition matrix')
    calculate_general_muts_matrix(population_size)

    start = time.time()
    qv.compute_q_values(working_dir, output_name, output_dir, occs_dfs, syn_target_dfs,population_size)
    print(f'elabsed time = {time.time() - start}')
