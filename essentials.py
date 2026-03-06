
import re
import cupy
import numpy as np
import pandas as pd

BASES = ['A', 'C', 'G', 'T']
CONTEXTS = [a+b+c for a in BASES for b in BASES for c in BASES]
CHROMS = [f'chr{str(i)}' for i in range(1, 23)] + ['chrX', 'chrY']

def get_observed_cpmatrix(matrix, occurences):
    observed_matrix = pd.DataFrame(0, columns=CONTEXTS, index=CONTEXTS)
    for source_context in CONTEXTS:
        for base in set(BASES) - {source_context[1]}:
            target_context = source_context[0] + base + source_context[2]
            observed_matrix[source_context][target_context] = matrix[base][source_context]*(max(occurences.T[source_context]))
        observed_matrix[source_context][source_context] = (max(occurences.T[source_context])) - observed_matrix[source_context].sum()
    observed_matrix = observed_matrix.fillna(0)
    return observed_matrix

#index=target, coln and source-seems wrong?
def matri64x4_to_64x64(matrix):
    newmatrix = pd.DataFrame(0, columns=CONTEXTS, index=CONTEXTS)
    for source_context in CONTEXTS:
        for base in set(BASES) - {source_context[1]}:
            target_context = source_context[0] + base + source_context[2]
            newmatrix.loc[target_context,source_context] = matrix.loc[source_context,base]

    return newmatrix

def matri64x64_to_64x4(matrix):
    matrix = pd.DataFrame(0, columns=BASES, index=CONTEXTS)
    for source_context in CONTEXTS:
        for base in set(BASES) - {source_context[1]}:
            target_context = source_context[0] + base + source_context[2]
            matrix[base][source_context] = matrix[source_context][target_context]

    return matrix


def get_cupy_occs(occurences):
    df = pd.DataFrame({c: {c2:occurences[c2[1]][c] for c2 in CONTEXTS} for c in CONTEXTS})
    for c in CONTEXTS:
        for c2 in CONTEXTS:
            if c == c2:
                df[c][c2] = max(occurences.T[c])
    return cupy.array(df.T, dtype=np.double)

def get_gene_name(line):

    attrs = line.split(";")[:-1]
    for attr in attrs:
        attr = attr.split()
        if attr[0] == "gene_name":
            return re.sub('\"', '', attr[1])

    return 'Not_a_gene'

def get_rev_comp(sequence):
    rev_comp_dict = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    rev_comp = ''
    for base in sequence:
        rev_comp += rev_comp_dict[base]
    return rev_comp[::-1]

def matrix_to_series(matrix):
    series = matrix.stack()
    series.index = series.index.map(lambda s: '->'.join(map(str, s)))
    return series

def df_to_dics(df):
    dics = df.to_dict(orient='split')
    index = dics['index']
    columns = dics['columns']
    data = dics['data']
    dics = [{(i, j): data[i][j]} for i in index for j in columns]
    one_dic = {(i, j): data[i][j] for i in index for j in columns}
    return one_dic, dics

def fill_by_dics(df, list_of_dics):
    filled_df = df.copy()
    for s in list_of_dics:
        key = list(s.keys())[0]
        i = key[0]; j = key[1]
        filled_df.iloc[i, j] = s[key]
    return filled_df

class mutation:
    def __init__(self, tri, base):
        self.tri = tri
        self.base = base
        self.label = tri + '->' + base

    @staticmethod
    def list_to_str(muts):
        return f'({",".join([str(i) for i in muts])})'

    @staticmethod
    def str_to_list(string):
        string = string[1:-1]
        muts = []
        for label in string.split(','):
            tri, base = label.split('->')
            muts.append(mutation(tri, base))
        return muts

    @staticmethod
    def get_cpg_muts():
        return [mutation('CGA', 'A'),
                mutation('TCG', 'T'),
                mutation('CGC', 'A'),
                mutation('GCG', 'T'),
                mutation('CGG', 'A'),
                mutation('CCG', 'T'),
                mutation('CGT', 'A'),
                mutation('ACG', 'T')]

    def __eq__(self, other):
        return self.tri == other.tri and self.base == other.base

    def __hash__(self):
        return hash(self.label)

    def __str__(self):
        return self.label

    def __repr__(self):
        return self.label

def get_mut_obj_list():
    return [mutation(tri, base) for tri in CONTEXTS for base in BASES if base != tri[1]]
