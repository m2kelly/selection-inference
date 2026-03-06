import pickle
import numpy as np
from math import e
import pandas as pd
import essentials as es
import matplotlib.pyplot as plt


Ne = 349786
def dndn(s):
    return 2*Ne * ((2*s)/(1-(e**(-4*Ne*s))))

file = pd.read_csv('../analyzed_genes.csv', sep='\t', index_col=0)

file = file[file['expected_selection'] < -10**-9]
file = file[file['expected_selection'] > -4*(10**-6)]

file['dn_dn'] = file['observed_m_+_k'] / file['expected_n']

with open('../errored_genes.logs', 'r') as f:
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
ax.plot(s_curve, dndn(s_curve), color='red', label="e^-s")

plt.ylabel('$dN_{obs}/dN_{exp}$', fontsize=32)
plt.xlabel('Selection in log scale -log(S)', fontsize=32)
plt.title('Relationship between $dN_{obs}/dN_{exp}$ and selection', fontsize=32)
ax.tick_params(axis='both', which='major', labelsize=16)
plt.show()
