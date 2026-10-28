import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from math import e
from scipy.optimize import fsolve
import numpy as np
plt.rcParams.update({'font.size': 14})

pd.set_option('display.float_format', '{:.2E}'.format)
dir='/home/dweghorngroup/selection_per_gene_4_allele_'
results_file=f'{dir}/analyzed_genes_nonsyn.csv'
gene_id_file=f'{dir}/maria_output/gene_name_id.csv'
paml_id_file=f'{dir}/maria_output/idmapping_S2.tsv'
shet_file=f'{dir}/maria_output/shet_Supplementary_Table_1.txt'
paml_file=f'{dir}/maria_output/Uncovering_adaptive_evolution_in_the_human_lineage_S2.csv'
qval_file=f'{dir}/Output/cactus/Anc4->hg38/q_values_cactus.txt'
data_prep_file=f'{dir}/Output/cactus/Anc4->hg38/nonsyn_output_data_preparation_cactus.txt'



data_prep_df=pd.read_csv(data_prep_file,sep='\t',index_col=0)
data_prep_df['l_n']=data_prep_df['l_m']+data_prep_df['l_k']

qval_df=pd.read_csv(qval_file,sep='\t',index_col=0,skiprows=1)
qval_df['n_obs_q']=qval_df['m_obs']+qval_df['k_obs']

results_df=pd.read_csv(results_file,sep='\t',names=['gene','s','s_upper','s_lower','n_exp','n_exp_lower','n_exp_upper','n_obs']).iloc[1:]
results_df['s']=results_df['s'].astype(float)


#coparing with dn/ds and obs ubs
qval_df = results_df.set_index('gene').join(qval_df,how='inner')
qval_df.sort_values(by='s',inplace=True)
qval_df=qval_df.join(data_prep_df[['l_n','l_s']],how='inner')
#but need to multiply by target sizes per genes!!! , 
#s exp =2N mu l_s/2N (fixation prob=1/2N)
#n exp = 2N mu l_n u
#where u  = (1-e^)/(1-e^(-4Ns)) ~ 2s/(1-e^(-4Ns))

#we approximate n_obs/n_exp in dn/ds - so should be comparing to this? 
Ne = 200000
def dndn(s):
    
    return 2*Ne * ((1-e**(-s))/(1-(e**(-2*Ne*s))))

'''given dnds want to solve for what s would be inferred if we assumed no parallel/recurrent mutations, i.e. dnds = n_obs/n_exp'''

def find_niave_s(dnds):
    
    def func(s):
        return dndn(s) - dnds
    s_guess = 0.000001
    s_solution = fsolve(func, s_guess)
    return s_solution[0]



#qval_df['n/nexp']=qval_df.apply(lambda row: row['n_obs']/(0.000001*(row['l_n'])*dndn(row['s'])),axis=1)
qval_df['n/nexp']=qval_df.apply(lambda row: dndn(row['s']),axis=1)
qval_df['naive_s']=qval_df.apply(lambda row: find_niave_s(row['d(m+k)/ds']),axis=1)

#qval_df=qval_df[qval_df['s']>-0.00001]
#cmap='gray'
plt.scatter(qval_df['d(m+k)/ds'],abs(qval_df['s']),alpha=0.5,label='inferred s',cmap='gray')
plt.scatter(qval_df['d(m+k)/ds'],abs(qval_df['naive_s']),color='red',label='no parallel/recurrent mutations',alpha=0.5)
plt.xlabel('dn_obs/dn_exp')
plt.ylabel('negative selection |s|')
s_range=np.linspace(min(qval_df['s']),max(qval_df['s']),100)
plt.plot(dndn(s_range),abs(s_range),label='no parallel/recurrent mutations',color='red')
plt.axhline(y=2.5e-6,label='drift barrier')
plt.legend()
plt.savefig('s_dnds_no_parallel.png')

plt.figure()
plt.scatter(qval_df['s'],qval_df['n/nexp'],color='red',label='theory',alpha=0.3)
plt.xlabel('s')
plt.ylabel('dn/dnexp predicted')
plt.legend()
plt.savefig('s_dnds_theoretical.png')

gene_id_df=pd.read_csv(gene_id_file)
results_df = qval_df.merge(gene_id_df,on='gene').set_index('gene_name')
results_df.sort_values(by='s',inplace=True)

s_het_df=pd.read_csv(shet_file,sep='\t',index_col=0)
s_het_df = s_het_df.join(results_df,how='inner')



print(len(s_het_df))
#print(s_het_df[['s','s_lower','s_upper','s_het_drift','d(m+k)/ds']])
print(f'shet s spearman r {stats.spearmanr(s_het_df['s'],s_het_df['s_het_drift'])}')
print('shet s pearson r ',stats.pearsonr(s_het_df['s'],s_het_df['s_het_drift']))
# Print the Spearman correlation
print('s_het vs dn/ds')
print('shet dn/ds spearmanr',stats.spearmanr(s_het_df['s_het_drift'], s_het_df['d(m+k)/ds']))


#compare dk/ds and s shet
print('dn/ds vs s_het')
qval_df=pd.read_csv(qval_file,sep='\t',index_col=0,skiprows=1)
s_het_df=pd.read_csv(shet_file,sep='\t',index_col=0)
qval_df = qval_df.merge(gene_id_df,on='gene').set_index('gene_name')
s_het_df=s_het_df.join(qval_df,how='inner')
s_het_df=s_het_df[s_het_df['q_phi_neg']<0.05]
s_het_df=s_het_df[s_het_df['m_obs']>=1]
print(len(s_het_df))
print(stats.spearmanr(s_het_df['s_het_drift'], s_het_df['d(m+k)/ds']))


paml_df=pd.read_csv(paml_file,sep='\t',index_col=0)
paml_id_df=pd.read_csv(paml_id_file,sep='\t',index_col=0)

paml_df=paml_df.join(paml_id_df[['Gene Names']],how='inner').set_index('Gene Names')



paml_df = paml_df.copy()

# Store original PAML gene names
paml_df["paml_gene"] = paml_df.index

# Split aliases into separate rows
paml_df["gene"] = paml_df.index.str.split()
paml_df = paml_df.explode("gene").set_index("gene")

# Join on exact gene name
paml_df = paml_df.join(results_df, how="inner")

# Keep only one match per original PAML row
paml_df = paml_df.drop_duplicates(subset="paml_gene")


print(paml_df[['d(m+k)/ds','dN/dS Human']])
print(stats.spearmanr(paml_df['dN/dS Human'], paml_df['d(m+k)/ds']))

plt.figure()
plt.scatter(paml_df['dN/dS Human'],paml_df['d(m+k)/ds'],
            s=10*paml_df['s_obs'],alpha=0.5,label='size=s_obs')
plt.plot(paml_df['dN/dS Human'],paml_df['dN/dS Human'],label='y=x')
plt.ylabel('cbase dn/dnexp')
plt.xlabel('paml dn/ds')
plt.legend()
plt.tight_layout()
plt.savefig('paml_cbase.png')


plt.figure()
plt.scatter(paml_df['dN/dS Human'],abs(paml_df['s']),
            s=10000*paml_df['s_obs']/paml_df['Length'],alpha=0.5,label='size=s_obs/length')
s_range=np.linspace(min(paml_df['s']),max(paml_df['s']),100)
plt.plot(dndn(s_range),abs(s_range),label='no parallel/recurrent mutations',color='red')
plt.ylabel('|s|')
plt.xlabel('paml dn/ds')
plt.legend()
plt.tight_layout()
plt.savefig('paml_s.png')

#hossam thesis results to compare 
'''
LARS2 -3.65E-06 -2.74E-06 -4.37E-06 11.7065 10 35 1
MPND -3.16E-06 -1.78E-06 -4.07E-06 17.225 4 21 1
RRP12 -1.49E-06 -4.22E-07 -2.17E-06 15.8756 8 31 6
'''
