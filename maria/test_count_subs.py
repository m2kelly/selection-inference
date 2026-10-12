import cupy
import pandas as pd 
import numpy as np

generations=10
def get_number_of_mutations_syn(occurences, transition_matrix_original, syn_fracs):
    """occurences: df index =trinucs, coln counts=trinuc counts in gene
       transition matrix = 64 by 64 matrix, indexes = numbers correpsomding to tirnucs eg AAA=0 ...TTT=64
       syn_fracs = 64x64 matrix, indexes=trinucs
    """

    P = cupy.array(transition_matrix_original)

    # ensure vector
    pi0 = cupy.array(occurences.values.flatten())

    #propagate generations
    Pt = cupy.linalg.matrix_power(P, generations)
    
    #consider only subs changing trinuc
    cupy.fill_diagonal(Pt, 0)

    #only allow syn subs
    Pt_syn = Pt * cupy.array(syn_fracs)

    expected_per_context = Pt_syn.sum(axis=1)

    #expected syn subs giving starting composition of gene
    mutations = cupy.dot(pi0, expected_per_context)

    return float(mutations.get())

contexts = ["AAA","AAC","AAG"]

P = pd.DataFrame(
[[0.8,0.1,0.1],
 [0.1,0.8,0.1],
 [0.2,0.1,0.7]],
index=[0,1,2],
columns=[0,1,2]
)

Pt = pd.DataFrame(
[[0.0,3.0,2.0],
 [4.0,0.0,2.0],
 [3.0,1.0,0.0]],
index=[0,1,2],
columns=[0,1,2]
)

syn = pd.DataFrame(
[[0,1,0],
 [1,0,0],
 [0,0,0]],
index=contexts,
columns=contexts
)

occ = pd.DataFrame(
{"count":[10,5,0]},
index=contexts
)

print(get_number_of_mutations_syn(occ, Pt, syn))
