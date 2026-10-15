'''
first calcalulate global matrix Q^T, then infer Q, M and compare with inputted M
script is to import pickles and infer a mutation matrix per gene
using observed muts
'''

from cuda_operations import get_transition_cpmatrix_from_rates

input_muts='test_pipeline/mu_sub_check/forward_sim_4_allele/whole_genome_dict_forward_sim_4_allele_unsmoothed_muts.pkl'
input_trinucs='test_pipeline/mu_sub_check/forward_sim_4_allele/whole_genome_dict_forward_sim_4_allele_unsmoothed_occs.pkl'
'''
format is dicts, where keys are chrs, entries are dfs
within dfs indexes are muts AAA->T, columns are simulated genes, (labelled by numbers 0,1..)
for occs, entries are triculetide counts in ancestor
for muts entries are observed subsitutions in branch'''
def find_global_Q_t():
    #load pickles and sum across all chrs and rows of dfs to get global observed subs and total inital occurences

def calculate_general_muts_matrix(pop_size, force=False):
    mutation_matrix_path = Path('test_pipeline/mu_sub_check/output/best_in_df_mutations_4_allele.csv')
    try:
        if force:
            raise FileNotFoundError
        muts_matrix = pd.read_csv(mutation_matrix_path, index_col=0, header=0, sep='\t')
    except FileNotFoundError:
        from cuda_operations import get_mut_matrix_from_sub_matrix

        sub_matrix = pd.read_csv('test_pipeline/mu_sub_check/output/best_in_df.csv', index_col=0, header=0, sep='\t')
        muts_matrix = get_mut_matrix_from_sub_matrix(sub_matrix, pop_size)

    muts_matrix.to_csv(mutation_matrix_path, sep='\t')

def global_recurrence_correction()
    sub_matrix = get_transition_cpmatrix_from_rates(
            rates, trinucs, generations=generations
        )
    sub_matrix.to_csv('files/best_in_df.csv', sep='\t')


def run_get_M()
    '''
    calcualte global mut matrix form recurrence corrected global sub matrix
    then compare with inputted matrix'''
    calculate_general_muts_matrix(
        population_size, force=args.correct_rates_for_recurrence
    )
    #run selectoin analyse gene

def run_get_gene_scaling(so,gene,nexp)
    #so = selection.operations
    #nexp comes from obsered subs in pickle for that gene
    #gene.nonsyn-sites comes from occurences in pickle (assume all are non syn)

    if not gene.nonsyn_sites:
        if gene.nobs == 0:
            return 0.0
        raise ValueError(
            f'No possible nonsynonymous sites were parsed for {gene.gene_name}'
        )
    if nexp <= 0:
        raise ValueError(
            f'Cannot calibrate a positive mutation rate from expected_n={nexp} '
            f'for {gene.gene_name}'
        )

    def neutral_error(log_scale):
        #no selection, rescale to match nexp calculated from cbase 
        try:
            expected = self._expected_four_allele_nonsyn(
                gene, 10 ** log_scale, 0.0
            )
        except ValueError:
            return np.inf
        return abs(expected - nexp)

    scale_solution = minimize_scalar(
        neutral_error,
        bounds=(np.log10(0.1), np.log10(5.0)),
        method='bounded',
        options={'xatol': 1e-10},
    )
    if not scale_solution.success or not np.isfinite(scale_solution.fun):
        raise RuntimeError(
            f'Could not rescale the four-allele mutation rates for {gene.gene_name}'
        )
    scale = 10 ** scale_solution.x
    return scale 
