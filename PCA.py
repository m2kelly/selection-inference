import pickle
import pandas as pd
import essentials as es
import matplotlib.pyplot as plt


outlier_genes = ['EPS8', 'LIMK2', 'MCM10', 'MROH6', 'RPN2', 'RYR2', 'TRIM26', 'ATP10B', 'CPPED1', 'CRYZ', 'MAP3K6', 'PRDM10', 'SH3BP1', 'TRIM36']

run_name = 'primates_10_Jun25'
episode = 'HCLCA->homo_sapiens'
occs = pickle.load(open(f'Output/{run_name}/{episode}/genes/occs_per_gene.pkl', 'rb')) #occs_dfs = {'SMG6': occs_dfs['SMG6'], 'AANAT': occs_dfs['AANAT'], 'BRCA1': occs_dfs['BRCA1']}
qvals_file = f'Output/{run_name}/{episode}/q_values_{run_name}.txt'
qvals_df = pd.read_csv(qvals_file, sep='\t', index_col=0, skiprows=1).T


def get_syn(gene_name):
      syn_count = qvals_df[gene_name]['s_obs']
      size = occs[gene_name].sum().sum()
      return syn_count/size

def get_CG_ratio(gene_name):
    CG = sum([occs[gene_name][tri].sum() for tri in es.CONTEXTS if 'CG' in tri])
    all = occs[gene_name].sum().sum()

    return float(CG)/all

file = pd.read_csv('analyzed_genes.csv', sep='\t', index_col=0)
file = file[file['expected_selection'] < -10**-9]
file['dn_dn'] = file['observed_m_+_k'] / file['expected_n']
file['CG_ratio'] = file.index.map(lambda gene: get_CG_ratio(gene))
file['syn_count'] = file.index.map(lambda gene: get_syn(gene))

x = file['CG_ratio']
y = file['syn_count']

# Create a boolean mask to identify outlier genes
is_outlier = file.index.isin(outlier_genes)

# Plot all genes in blue
plt.scatter(x[~is_outlier], y[~is_outlier], color='blue', label='Genes')

# Plot outlier genes in red
plt.scatter(x[is_outlier], y[is_outlier], color='red', label='Outlier Genes')

# Add labels and title
plt.xlabel('CG Ratio')
plt.ylabel('Synonymous Count')
plt.title('CG Ratio vs Synonymous Count for Genes')

# Add a legend
plt.legend()

# Show the plot
plt.show()