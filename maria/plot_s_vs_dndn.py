import pickle
import numpy as np
from math import e
import pandas as pd
import essentials as es
import matplotlib.pyplot as plt



#but need to multiply by target sizes per genes!!! , 
#s exp =2N mu l_s/2N (fixation prob=1/2N)
#n exp = 2N mu l_n u
#where u  = (1-e^)/(1-e^(-4Ns)) ~ 2s/(1-e^(-4Ns))

#we approximate n_obs/n_exp in dn/ds - so should be comparing to this? 
Ne = 349786
def dndn(s):
    return 2*Ne * ((2*s)/(1-(e**(-4*Ne*s))))

file = pd.read_csv('analyzed_genes.csv', sep='\t', index_col=0,names=['gene','expected_selection','s_upper','s_lower','expected_n','n_exp_lower','n_exp_upper','observed_m_+_k'])

file = file[file['expected_selection'] < -10**-9]
file = file[file['expected_selection'] > -4*(10**-6)]

file['dn_dn'] = file['observed_m_+_k'] / file['expected_n']

with open('errored_genes.logs', 'r') as f:
    genes_to_exclude = f.read().splitlines()

# Filter the DataFrame to exclude genes in the list
file = file[~file.index.isin(genes_to_exclude)]

gene_names = file.index.tolist()
print(len(file))

dn_dn = file['observed_m_+_k'] / file['expected_n']
s = file['expected_selection']
#ABCC8
#plot scatter and add gene names to the points
fig, ax = plt.subplots()
ax.scatter(s, dn_dn)
#ax.scatter(s[1], dn_dn[1], color='red')
s_curve = np.linspace(min(s*1.1), max(s*0.9), 1000)
ax.plot(s_curve, dndn(s_curve), color='red', label="uncorrected expected dndn(s)")

plt.ylabel('$dN_{obs}/dN_{exp}$', fontsize=16)
plt.xlabel('Selection in log scale -log(S)', fontsize=16)
plt.title('Relationship between $dN_{obs}/dN_{exp}$ and selection', fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=16)
plt.legend()
plt.savefig('s_dn.png')
plt.show()
