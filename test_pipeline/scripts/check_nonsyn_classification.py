import pandas as pd

df = pd.read_csv("../../files/four_allele_site_summary.tsv.gz", sep="\t")

#count is wrong 

# Number of possible mutations, accounting for site counts
nonsyn = (df["count"] * df["n_nonsyn"]).sum()
syn = (df["count"] * df["syn_alt_bases"].fillna("").str.len()).sum()

print(f"Nonsynonymous mutations: {nonsyn:,}")
print(f"Synonymous mutations:    {syn:,}")
print(f"Total mutations:         {nonsyn + syn:,}")
print(f"Nonsyn/Syn ratio:        {nonsyn / syn:.4f}")