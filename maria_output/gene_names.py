import pandas as pd

file='/home/dweghorngroup/selection_per_gene_4_allele_/maria_output/Uncovering_adaptive_evolution_in_the_human_lineage_S2.csv'

df=pd.read_csv(file,sep='\t')
print(df.columns)
df[['Human Protein']].to_csv(f'{file}_genes',index=False,header=False)