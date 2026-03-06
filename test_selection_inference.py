import math
import subprocess
import time
import cupy
import pandas as pd
import essentials as es
import cuda_operations as co
import numpy as np
from joblib import Parallel, delayed
import pickle
from predictor import Predictor
from graphpoint import GraphPointAbstract as GPA


'''
first test so have wokring function
then will simulate spectrum and re-calculate
'''

selection_range=[ 0.0000001,0.01]
population_size=200000
predictor = Predictor()

BASES = ['A', 'C', 'G', 'T']
CONTEXTS = [f'{b1}{b2}{b3}' for b1 in BASES for b2 in BASES for b3 in BASES]

gene='ENSG00000000460.17'
nobs=5
sobs=8
nexp=17.70151418551893
gene_scale=2.15
def selection_from_rescaled_mut_matrix(mutations_matrix_host,occurences,syn_target):
    '''
    nonsyn_target=1-syn_target
    syn_prob= es.matri64x4_to_64x64(syn_target)
    syn_prob.to_csv('test/syn_prob.csv')
    nonsyn_prob=es.matri64x4_to_64x64(nonsyn_target)
    nonsyn_prob.to_csv('test/nonsyn_prob.csv')
    '''
    syn_prob=pd.read_csv('test/syn_prob.csv',index_col=0)
    nonsyn_prob=pd.read_csv('test/nonsyn_prob.csv',index_col=0)

    sub_matrix_no_s=co.get_sub_matrix_from_mut_matrix_host(mutations_matrix_host, population_size=population_size,
                                                            selection=0)
    #sub_matrix_no_s.to_csv('test/sub_no_s.csv')
    
    obs_subs=nobs+sobs
    neg_direction = True if nobs < nexp else False
    def error_function(selection):
        
        try:
            sub_matrix = co.get_sub_matrix_from_mut_matrix_host(mutations_matrix_host, population_size=population_size,
                                                        selection=selection)
            
            print(sub_matrix) #if have nan entries then conitnue
            #check if sub_matrix is an identity matrix or cntains na values
            if sub_matrix is None or not cupy.isfinite(sub_matrix).all():
                return np.inf
            if not neg_direction and (cupy.all(cupy.eye(64) == sub_matrix)):
                return np.inf
            #TO DO EDIT THESE FUNCTIONS
            print('calculating expected_n')
            expected_n = co.get_number_of_mutations_syn(occurences, sub_matrix,nonsyn_prob)
            if not np.isfinite(expected_n):
                return np.inf
            print(f'expected n {expected_n}')
            print('calculating expected_s')
            expected_s =co.get_number_of_mutations_syn(occurences, sub_matrix_no_s,syn_prob)
            print(f'expected s {expected_s}')
            expected_subs=expected_n+expected_s
            print(f'expected sub {expected_subs}')
            return -1 if (expected_subs <= 0.01 and obs_subs== 0) else abs(expected_subs-obs_subs)

        except OverflowError as e:
            return np.inf

    selection = predictor.minimize_by_search(error_function, selection_range)

    return selection

#first run do this
'''
occs_dfs = pickle.load(open(f'files/trinucs.pkl', 'rb')) #occs_dfs = {'SMG6': occs_dfs['SMG6'], 'AANAT': occs_dfs['AANAT'], 'BRCA1': occs_dfs['BRCA1']}
occs_df=occs_dfs[gene]  
occs_df.to_csv('test/occs.csv')
syn_target_dfs=pickle.load(open('files/Anc4_syn_frac.pkl', 'rb'))
syn_target=syn_target_dfs[gene]
'''
occs_df=pd.read_csv('test/occs.csv',index_col=0)
muts_matrix = pd.read_csv('files/best_in_df_mutations.csv',index_col=0,sep='\t')
muts_matrix=muts_matrix.multiply(gene_scale)
#convert mut matrix indexes from tirnuc to corresponding strings
mutations_matrix_host=pd.DataFrame(0.0,index=range(64),columns=range(64))
for i in range(64):
    for j in range(64):
        mutations_matrix_host.loc[i,j] = muts_matrix.loc[CONTEXTS[i],CONTEXTS[j]]
print(mutations_matrix_host)
print(selection_from_rescaled_mut_matrix(mutations_matrix_host,occs_df,''))