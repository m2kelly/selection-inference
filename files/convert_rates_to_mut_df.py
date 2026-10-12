import pandas as pd

'''
expand to 64x64 matrix
divide by number generations
set diagoinal entries to be 1-(sum of entries in row)
'''

input = 'rates'
ref='best_in_df_ref.csv'
out='best_in_df.csv'
generations = 160000
bases=['A','C','G','T']
trinucleotides= [a+b+c for a in bases for b in bases for c in bases]
out_df = pd.DataFrame(0.0,index=trinucleotides,columns=trinucleotides)

rates=pd.read_csv(input,index_col=0,sep='\t')
for trinuc, row in rates.iterrows():
    left_flank=trinuc[0]
    right_flank = trinuc[2]
    ref=trinuc[1]
    for alt in bases:
        if alt != ref:
            out_df.loc[trinuc,left_flank+alt+right_flank] = row[alt]/generations
#probability of staying the smae should 1-prob of changing?
for trinuc, row in out_df.copy().iterrows():
    out_df.loc[trinuc,trinuc] = 1-row.sum()

print(out_df.sum(axis=1))
out_df.to_csv(out,sep='\t')
#aim=pd.read_csv(ref,sep='\t',index_col=0)
#print(aim)
#has 1s on diagonal and headers = cols 


#test ref 
#its noamlised so sum over axis = 1 is 1: so also normalise matrix as this

import pandas as pd
file='/home/dweghorngroup/selection_per_gene_final_step_intron_all_occ_fixed/files/best_in_df_ref.csv'
file='/home/dweghorngroup/selection_per_gene_final_step_intron_all_occ_fixed/files/best_in_df_mutations.csv'
df= pd.read_csv(file,index_col=0,sep='\t')
l=0
for i in range(len(df)):
    for j in range(len(df)):
        if df.iloc[i,j] == 0.0:
            l+=1
print(l)
print(df.sum(axis=0).sum())
print(df.sum(axis=1).sum())


for i in range(len(df)):
    for j in range(len(df)):
        if df.iloc[i,j]!=df.iloc[j,i]:
            print(df.iloc[i,j],df.iloc[j,i])

'''