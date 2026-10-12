import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from math import e

pd.set_option('display.float_format', '{:.2E}'.format)
results_file='/home/dweghorngroup/selection_per_gene_final_step_intron_all_occ_fixed/analyzed_genes_using_all_muts.csv'
gene_id_file='/home/dweghorngroup/selection_per_gene_final_step_intron_all_occ_fixed/maria_output/gene_name_id.csv'
shet_file='/home/dweghorngroup/selection_per_gene_final_step_intron_all_occ_fixed/maria_output/shet_Supplementary_Table_1.txt'
qval_file='/home/dweghorngroup/selection_per_gene_final_step_intron_all_occ_fixed/Output/cactus/Anc4->hg38/q_values_cactus.txt'
data_prep_file='/home/dweghorngroup/selection_per_gene_final_step_intron_all_occ_fixed/Output/cactus/Anc4->hg38/nonsyn_output_data_preparation_cactus.txt'

data_prep_df=pd.read_csv(data_prep_file,sep='\t',index_col=0)
print(data_prep_df)
data_prep_df['l_n']=data_prep_df['l_m']+data_prep_df['l_k']

qval_df=pd.read_csv(qval_file,sep='\t',index_col=0,skiprows=1)
qval_df['n_obs_q']=qval_df['m_obs']+qval_df['k_obs']

results_df=pd.read_csv(results_file,sep='\t',names=['gene','s','s_upper','s_lower','n_exp','n_exp_lower','n_exp_upper','n_obs']).iloc[1:]

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
    return 2*Ne * ((2*s)/(1-(e**(-4*Ne*s))))

#qval_df['n/nexp']=qval_df.apply(lambda row: row['n_obs']/(0.000001*(row['l_n'])*dndn(row['s'])),axis=1)
qval_df['n/nexp']=qval_df.apply(lambda row: dndn(row['s']),axis=1)
print(qval_df[['s','d(m+k)/ds','s_obs','n_obs','n_obs_q']])
#qval_df=qval_df[qval_df['s']>-0.00001]
#cmap='gray'
plt.scatter(qval_df['s'],qval_df['d(m+k)/ds'],alpha=0.3,c=(qval_df['n_obs']+qval_df['s_obs'])/(qval_df['n_obs']+qval_df['s_obs']).sum(),label='observed dn/dnexp',cmap='gray')
plt.scatter(qval_df['s'],qval_df['n/nexp'],color='red',label='theoretical if no parallel/recurrent mutations',alpha=0.3)
plt.xlabel('s')
plt.ylabel('dn/dnexp')
plt.axvline(x=-2.5e-6,label='drift barrier')
plt.legend()
plt.savefig('s_dnds.png')

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
print(s_het_df[['s','s_lower','s_upper','s_het_drift','d(m+k)/ds']])
print(stats.spearmanr(s_het_df['s'],s_het_df['s_het_drift']))
print(f'pearson r {stats.pearsonr(s_het_df['s'],s_het_df['s_het_drift'])}')
# Print the Spearman correlation
print('s_het vs dn/ds')
print(stats.spearmanr(s_het_df['s_het_drift'], s_het_df['d(m+k)/ds']))


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


#hossam thesis results to compare 
'''
LARS2 -3.65E-06 -2.74E-06 -4.37E-06 11.7065 10 35 1
MPND -3.16E-06 -1.78E-06 -4.07E-06 17.225 4 21 1
RRP12 -1.49E-06 -4.22E-07 -2.17E-06 15.8756 8 31 6
'''
