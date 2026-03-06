import multiprocessing
import sys
import cupy
import pandas as pd
import essentials as es
from predictor import Predictor
import numpy as np
from graphpoint import GraphPointAbstract as GPA
from joblib import Parallel, delayed
from database_interface import DatabaseInterface

try:
    best_seed = pd.read_csv('files/best_in_df_mutations.csv', index_col=0, header=0, sep='\t')
    general_mut_matrix = pd.read_csv('files/best_in_df_mutations.csv', index_col=0, header=0, sep='\t')
    general_mut_matrix = cupy.array(general_mut_matrix).T
except FileNotFoundError: pass

predictor = Predictor()
database = DatabaseInterface()
generations = 160000

def get_error(initial_state, trasition_cpmatrix, observed_cpmatrix):

    arr_cupy = cupy.copy(trasition_cpmatrix)
    result = cupy.linalg.matrix_power(arr_cupy, generations)
    result = cupy.dot(result, initial_state)
    error_bases = cupy.subtract(result, observed_cpmatrix)
    error_bases = cupy.absolute(error_bases)
    #round to whole numbers
    #error_bases = cupy.around(error_bases, 0)

    return error_bases, error_bases.sum().sum()

def max_error_and_pos(error_cpmat):
    max_value = cupy.max(error_cpmat)
    max_pos = cupy.unravel_index(cupy.argmax(error_cpmat), error_cpmat.shape)
    #max_pos = cupy.argmax(error_cpmat)
    return max_value, max_pos

def max_errors_and_poss(error_cpmat):
    flattened = error_cpmat.flatten()
    max_values = cupy.sort(flattened)[::-1]
    max_poss = cupy.argsort(flattened)[::-1]
    max_poss = cupy.unravel_index(max_poss, error_cpmat.shape)
    #max_poss = cupy.argsort(error_cpmat)
    return max_values, max_poss

def approximate_sum(cpmat):
    approximated = cupy.copy(cpmat)
    #approximated = cupy.around(approximated, 6)
    # for each row, sum the row and subtract the difference from 1 from the largest value
    for row in range(approximated.shape[0]):
        approximated[row] = approximated[row] / approximated[row].sum()
    return approximated

def get_try_cpmatrix(guessed_cpmat, max_error_pos, ratio):
    try_cpmat = cupy.copy(guessed_cpmat)
    try_cpmat[max_error_pos] *= (ratio)
    try_cpmat[max_error_pos[0]] = cupy.divide(try_cpmat[max_error_pos[0]], cupy.sum(try_cpmat[max_error_pos[0]]))
    assert try_cpmat[2][15].get() == 0
    return try_cpmat

def get_expected_from_IT(initial_state, trasition_matrix):

    #TODO check occ max or each
    initial_state = cupy.array(es.matri64x4_to_64x64(initial_state))
    arr_cupy = cupy.copy(trasition_matrix)
    result = cupy.linalg.matrix_power(arr_cupy, generations)
    result = cupy.dot(result, initial_state)
    result = cupy.copy(result)
    result = es.matri64x4_to_64x64(result)
    return result

def get_number_of_mutations(occurences, trasition_matrix_original):
    trasition_matrix = cupy.copy(trasition_matrix_original)
    intial_state = cupy.array(occurences.values.flatten())
    #intial_state = cupy.array(occurences.max(axis=1))
    transition_matrix_dev = cupy.array(trasition_matrix)
    arr_cupy = cupy.copy(transition_matrix_dev)
    result = cupy.linalg.matrix_power(arr_cupy, generations)
    cupy.fill_diagonal(result, 0)


    result = cupy.dot(result, intial_state)
    mutations = result.get().sum().sum()
    return mutations

def get_number_of_mutations_maria(occurences, transition_matrix_original):
    """occurences: df index =trinucs, coln counts=trinuc counts in gene
       transition matrix = 64 by 64 matrix, indexes = numbers correpsomding to tirnucs eg AAA=0 ...TTT=64
      
    """
    P = cupy.array(transition_matrix_original)
    # ensure vector
    pi0 = cupy.array(occurences.values.flatten())
    #propagate generations
    Pt = cupy.linalg.matrix_power(P, generations)
    #consider only subs changing trinuc
    cupy.fill_diagonal(Pt, 0)
    expected_per_context = Pt.sum(axis=1)
    #expected syn subs giving starting composition of gene
    mutations = cupy.dot(pi0, expected_per_context)
    return float(mutations.get())


def get_number_of_mutations_og(occurences, trasition_matrix_original):
    # Computes the total expected number of mutations after a given number of generations
    # using a transition matrix. The function:
    # 1. Copies the provided transition matrix to GPU memory (CuPy).
    # 2. Initializes the starting state as the trinucs occurrence count per context
    #    in the anc seqn (taken across columns of the `occurences` matrix).
    # 3. Raises the transition matrix to the power of `generations`, modelling
    #    repeated application of the mutation process across generations.
    # 4. Sets the diagonal of the resulting matrix to zero so that self-transitions
    #    (no mutation events) are ignored.
    # 5. Multiplies the powered transition matrix by the initial state vector to
    #    obtain the number of subn expected in each state i by  s_i(1-Q_{ii}^t)
    #    here s_i is the intial frequency of trinucletide i
    # 6. Transfers the result back to CPU and sums all entries to obtain the total
    #    number of mutations.
    trasition_matrix = cupy.copy(trasition_matrix_original)
    intial_state = cupy.array(occurences)  
    transition_matrix_dev = cupy.array(trasition_matrix)
    arr_cupy = cupy.copy(transition_matrix_dev)
    result = cupy.linalg.matrix_power(arr_cupy, generations)
    cupy.fill_diagonal(result, 0)
    result = cupy.dot(result, intial_state)
    mutations = result.get().sum().sum()
    return mutations

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
    #sum axis depends on if index=source or target
    expected_per_context = Pt_syn.sum(axis=0)

    #expected syn subs giving starting composition of gene
    mutations = cupy.dot(pi0, expected_per_context)

    return float(mutations.get())

def check_get_number_of_mutations_matrix_orientation(occurences, trasition_matrix):
    ''' checks by checking the row by row matrix operation. break point on the for
        loop and check the matrix orientation by checking if the right values are
        being multiplied
    '''
    old = get_number_of_mutations(occurences, trasition_matrix)

    intial_state = cupy.array(occurences.max(axis=1))

    transition_matrix_dev = cupy.array(trasition_matrix)
    arr_cupy = cupy.copy(transition_matrix_dev)
    result = cupy.linalg.matrix_power(arr_cupy, generations)
    cupy.fill_diagonal(result, 0)

    qn = cupy.array([[0]*64]*64 , dtype=np.double)

    for c in es.CONTEXTS:
        q0_c = cupy.array([0]*64, dtype=np.double)
        q0_c[es.CONTEXTS.index(c)] = intial_state[es.CONTEXTS.index(c)]
        qn[es.CONTEXTS.index(c)] = cupy.dot(q0_c, result)

    mutations = qn.sum().sum()

    result = cupy.dot(result, intial_state)
    mutations = result.get().sum().sum()
    return mutations

def get_transition_cpmatrix(occurences, matrix):

    intial_state = cupy.array(es.matri64x4_to_64x64(occurences))
    guessed_matrix = cupy.array(best_seed)
    observed_matrix = cupy.array(es.get_observed_cpmatrix(matrix ,occurences))
    alpha = 0.99
    max_loop = 300000
    error = cupy.array([0]*max_loop)

    for i in range(max_loop):

        error_mat, errored_bases = get_error(intial_state, guessed_matrix, observed_matrix)
        error[i] = errored_bases

        _max_errors, max_error_poss = max_errors_and_poss(error_mat)

        for max_error_pos in zip(max_error_poss[0][:100], max_error_poss[1][:100]):

            if guessed_matrix[max_error_pos] == 0: continue
            if observed_matrix[max_error_pos] == 0: continue
            analyzed = False
            for ratio in 1+alpha, 1-alpha:
                try_matrix = get_try_cpmatrix(guessed_matrix, max_error_pos, ratio)
                #try_matrix = get_try_cpmatrix(intial_state, guessed_matrix, max_error_pos, ratio)
                new_error_cpmat, _ = get_error(intial_state, try_matrix, observed_matrix)
                new_error_cpmat = cupy.absolute(new_error_cpmat)

                if cupy.sum(new_error_cpmat) < cupy.sum(error_mat):

                    guessed_matrix = try_matrix
                    analyzed = True
                    break
            if analyzed: break

        if not analyzed:
            alpha *= 0.5
            #print('alpha reduced to', alpha)

        if i % 100 == 0: sys.stdout.write('\r' + str(i) + ' iterations')

        if alpha < 1e-7:
            #print('converged')
            break

def transition_matrix_scale(matrix_original, scale):
    #check if the matrix is the same as the general_mut_matrix
    matrix = cupy.copy(matrix_original)
    def scale_f(m):
        for row in range(m.shape[0]):
            m[row] = m[row] * scale
            m[row][row] = 0
            m[row][row] = 1 - cupy.sum(m[row])
        return m
    if cupy.all(matrix==general_mut_matrix):
        try: scaled_matrix = database.get_table_from_index(0, str(scale), True)
        except Exception as e:
            scaled_matrix = scale_f(matrix)
            #check if scaled matrix is of type cupy array
            if not type(scaled_matrix) == cupy.ndarray: scaled_matrix = cupy.array(scaled_matrix)
            scaled_matrix_pd = pd.DataFrame(scaled_matrix.get())
            #database.insert_table_into_database(scaled_matrix_pd, str(scale))
            database.insert_table_into_database(scaled_matrix_pd, 0, str(scale), True)

        if not type(scaled_matrix) == cupy.ndarray: scaled_matrix = cupy.array(scaled_matrix)
        return scaled_matrix

    else: return scale_f(matrix)

def get_sub_matrix_from_scaled_mut_matrix_database(mut_matrix_device, population_size, selection=0, scale=None):
    #the mut matrix is already scaled, the scale parameter is used for the database index
    if scale:
        try:
            scaled_matrix = database.get_table_from_index(selection, str(scale), False)
            if not type(scaled_matrix) == cupy.ndarray: scaled_matrix = cupy.array(scaled_matrix)
            return scaled_matrix

        except Exception as e:
            scaled_matrix = get_sub_matrix_from_mut_matrix_device(mut_matrix_device, population_size, selection)
            scaled_matrix_pd = pd.DataFrame(scaled_matrix.get())
            database.insert_table_into_database(scaled_matrix_pd,selection, str(scale), False)
            if not type(scaled_matrix) == cupy.ndarray: scaled_matrix = cupy.array(scaled_matrix)
            return scaled_matrix

    return get_sub_matrix_from_mut_matrix_device(mut_matrix_device, population_size, selection)

def get_sub_matrix_from_mut_matrix_device(mut_matrix_device, population_size, selection=0):

    mu_matrix_host = pd.DataFrame(mut_matrix_device.get())
    #sub_matrix_host is a zero matrix with the same shape as the mut_matrix_host
    return get_sub_matrix_from_mut_matrix_host(mu_matrix_host, population_size, selection)

def get_sub_matrix_from_mut_matrix_host(mu_matrix_host, population_size, selection=0):

    #sub_matrix_host is a zero matrix with the same shape as the mut_matrix_host
    sub_matrix_host = pd.DataFrame(np.zeros(mu_matrix_host.shape))

    mutations_dics, mutations_dics_list = es.df_to_dics(mu_matrix_host)

    def f (sub_dict):
        import warnings
        warnings.filterwarnings("ignore")
        sub = float(list(sub_dict.values())[0])
        key = list(sub_dict.keys())[0]
        if key[0] == key[1]: return {key:0}
        if sub == 0: return {key:0}
        if np.isnan(sub): return {key:0}

        mu = mutations_dics[(key[0], key[1])]
        backwards_mu = mutations_dics[(key[1], key[0])]
        ratio = 1 #TODO make ratio equals 1 when it's a TpG to C mutation
        point = GPA(mu, backwards_mu*ratio, population_size, selection_coeff=selection)
        try:
            sub_corrected_by_selection = predictor.get_sub_rate(point)
        except Exception as e:
            sub_corrected_by_selection = predictor.get_sub_rate_regression(point)
        if sub_corrected_by_selection in [np.inf, -np.inf, np.nan]: sub_corrected_by_selection = 0
        return {key:sub_corrected_by_selection}

    _, sub_corrected_by_selection = es.df_to_dics(sub_matrix_host)
    number_of_cores = multiprocessing.cpu_count()
    batch_size = int((64*64)/number_of_cores)
    analyzed_list = Parallel(n_jobs=-1, verbose=0, batch_size=batch_size)(delayed(f)(s) for s in mutations_dics_list)
    sub_matrix_host = es.fill_by_dics(sub_matrix_host, analyzed_list)
    for i in range(len(sub_matrix_host)): #refill the diagonal
            sub_matrix_host[i][i] = 1 - sub_matrix_host[i].sum()
    sub_matrix_device = cupy.array(sub_matrix_host)

    return sub_matrix_device

def get_mut_matrix_from_sub_matrix(sub_matrix, population_size, selection=0):
    sub_index = sub_matrix.index; sub_columns = sub_matrix.columns
    sub_matrix = pd.DataFrame(cupy.array(sub_matrix).get())
    mut_matrix = pd.DataFrame(np.zeros(sub_matrix.shape))

    mutations_dics, mutations_dics_list = es.df_to_dics(sub_matrix)

    def f (sub_dict):
        import warnings
        warnings.filterwarnings("ignore")
        sub = float(list(sub_dict.values())[0])
        key = list(sub_dict.keys())[0]
        if key[0] == key[1]: return {key:0}
        if sub == 0: return {key:0}
        if np.isnan(sub): return {key:0}

        #mu = mutations_dics[(key[0], key[1])]
        backwards_mu = mutations_dics[(key[1], key[0])]

        point = GPA(0, backwards_mu, population_size, selection_coeff=selection,  mean_sub_rate=sub)

        mu = predictor.get_mu(point)

        if mu in [np.inf, -np.inf, np.nan]: mu = 0
        return {key:mu}

    _, sub_corrected_by_selection = es.df_to_dics(mut_matrix)
    number_of_cores = multiprocessing.cpu_count()
    number_of_cores = 1
    batch_size = int((64*64)/number_of_cores)
    analyzed_list = Parallel(n_jobs=-1, verbose=0, batch_size=batch_size)(delayed(f)(s) for s in mutations_dics_list)
    mut_matrix = es.fill_by_dics(mut_matrix, analyzed_list)
    for i in range(len(mut_matrix)): #refill the diagonal
            mut_matrix[i][i] = 1 - mut_matrix[i].sum()
    mut_matrix = cupy.array(mut_matrix)
    mut_df = pd.DataFrame(mut_matrix.get(), index=sub_index, columns=sub_columns)
    return mut_df

def get_transition_matrix_for_observed_mutations(exp_muts_no, occurences,prob, population_size,  obs_ratio=1, selection=0):
    '''
    occurences is trinucs counts per gene from cbase?
    PROBELM here considering only expected non syn muts-so should only accept non syn? 
    rescales mut matrix per gene to sub matrix using N/pop gen work
    then rescales sub matrix by mu_g such that exp muts = occs*sub matrix*mu_g matches observed subs
    '''
    
    try:
        best_seed = pd.read_csv('files/best_in_df_mutations.csv', index_col=0, header=0, sep='\t')
        general_mut_matrix = pd.read_csv('files/best_in_df_mutations.csv', index_col=0, header=0, sep='\t')
        general_mut_matrix = cupy.array(general_mut_matrix).T
    except FileNotFoundError: pass


    #occurences = es.matri64x4_to_64x64(occurences.T)
    #intial_state = cupy.array(occurences.max(axis=1))

    def f(ratio):
        #error = lambda n: 0 if abs(n-exp_muts_no)<=0.05 else abs(n-exp_muts_no)/exp_muts_no
        error = lambda n: abs(n-exp_muts_no)
        #error = lambda n: abs(n-exp_muts_no)/exp_muts_no
        scaled_muts_matrix = transition_matrix_scale(general_mut_matrix, ratio)
        guessed_sub_matrix = get_sub_matrix_from_scaled_mut_matrix_database(scaled_muts_matrix, population_size, selection, scale=ratio)
        #calcs expected number of mutations given starting trinuc freqs and sub matrix Q and gene time (pie^Qt)
        #occuences is series with index trinucs
        pd.DataFrame(guessed_sub_matrix.get(),index=range(64),columns=range(64)).to_csv('test_subs.csv')
        
        number_of_mutations = get_number_of_mutations_syn(occurences, guessed_sub_matrix,prob)
        print(f'rescale expected {number_of_mutations}, cbase exp {exp_muts_no}')
        return float(error(number_of_mutations))

    one_ratio_sub_matrix = get_sub_matrix_from_scaled_mut_matrix_database(general_mut_matrix, population_size, selection, scale=1.0)
    one_ratio_muts = get_number_of_mutations_syn(occurences, one_ratio_sub_matrix,prob)
    print(f'one ratio muts {one_ratio_muts}')
    if one_ratio_muts > exp_muts_no:
        bounds = [1, 0.05]
    else:
        bounds = [1, 5]
    

    #tetsing maria
    ratio_test=predictor.minimize_by_search(f, bounds)
    print(ratio_test)
    best_ratio=float(ratio_test)
    '''
    tolerance = 0.05
    best_ratio =  float(predictor.minimize_by_search(f, bounds, negative=False,
                                                     breadth=5, max_depth=30,
                                                     nearest_value_tie_breaker=1, log=False,
                                                    first_breadth=100, tol=tolerance))
    '''
    best_mu_matrix = transition_matrix_scale(general_mut_matrix, best_ratio)
    #best_sub_matrix = get_sub_matrix_from_scaled_mut_matrix(best_mu_matrix, population_size, selection, scale=best_ratio)
    return best_mu_matrix, best_ratio
