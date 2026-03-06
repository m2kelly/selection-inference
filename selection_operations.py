import math
import subprocess
import time
import cupy
import pandas as pd
import essentials as es
import cuda_operations as co
import numpy as np
from joblib import Parallel, delayed

from predictor import Predictor
from graphpoint import GraphPointAbstract as GPA

BASES = ['A', 'C', 'G', 'T']
CONTEXTS = [f'{b1}{b2}{b3}' for b1 in BASES for b2 in BASES for b3 in BASES]
wfes_single = '/home/hossam26644/Documents/wfes/wfes2/bin/wfes_single'

class SelectionOperations:
    def __init__(self, selection_range=[0.0000001,0.0001], generations=160000,
                 population_size=None, transition_matrix='files/transition_matrix.csv',
                 syn_occs='files/occurences.csv'):
        self.selection_range = selection_range
        self.generations = generations
        self.population_size = population_size
        self.transition_matrix = transition_matrix
        self.syn_occs = syn_occs
        self.predictor = Predictor()
        #self.predictor.self_train()

    def run_analysis(self, gene):
        ''' function added by Hossameldin Loay to compute the DNobs by DNexp for a given gene
            Input:
                genename: name of the gene
                mobs: number of observed missense mutations
                kobs: number of observed nonsense mutations
                pofx_given_s: function to calculate the probability of a given
                            number of mutations based on the selected model, number of
                            synonymous mutations the ratio between the expected number of
                            missense or nonsense mutations and the number of expected
                            synonymous
                sobs: number of observed synonymous mutations
                L: idk but it looks like it is always one
                ratm: ratio between the expected number of missense and the number of expected
                    synonymous mutations
                ratk: ratio between the expected number of nonsense and the number of expected
                    synonymous mutations
                large_flag: 1 or 0
            Output: a string representing of \t separated values of the
                    genename, Dmobs, Dmexp, Dkobs, Dkexp, Dnobs/Dnexp
        '''

        expected_m = self.calc_expected_number_of_mutations(gene, gene.ratm)
        expected_k = self.calc_expected_number_of_mutations(gene, gene.ratk)
        expected_k = expected_k if not math.isnan(expected_k) else 0
        expected_n = expected_m + expected_k
        print(f'{gene.gene_name}')
        
        selection = self.get_selection(gene, expected_n)
        print(f'{gene.gene_name}: selection = {selection}, expected = {expected_n}, obs = {gene.nobs}')
        lowest_n, highest_n = self.get_percentiles(gene, gene.ratn)
        lowest_selection = self.get_selection(gene, highest_n)
        print(f'{gene.gene_name}: lowest = {lowest_selection}, expected = {highest_n}, obs = {gene.nobs}')
        highest_selection = self.get_selection(gene, lowest_n)
        print(f'{gene.gene_name}: highest = {highest_selection}, expected = {lowest_n}, obs = {gene.nobs}')

        gene.expected_selection = selection
        gene.max_possible_highest_selection = highest_selection
        gene.min_possible_highest_selection = lowest_selection
        gene.expected_n = expected_n
        gene.lowest_n = lowest_n
        gene.highest_n = highest_n


        print(f'{gene.gene_name}: lowest = {lowest_selection}, highest = {highest_selection}, expected = {selection}')

        return gene


    @staticmethod
    def calc_expected_number_of_mutations(gene, ratio, pofx_given_s=None):
        '''function to compute the expected number of mutations based on the pofx_given_s formula,
        The function keeps summing n*p till p is very little, or delta n*p is ~zero
        n here is a possible number of mutations, p is the probability of that number of
        mutations to happen
        pofx_given_s uses the number of observed syn mutations and the (ratio) of the type
        of the mutation to the expected number of synonymous mutations
        returns sum of n*p, for n->~inf
        '''
        if pofx_given_s is None: pofx_given_s = gene.pofx_given_s

        sumpvals = 0
        n = 0.0
        expected_no = 0
        while sumpvals < 1:
            pval = float(pofx_given_s(n, gene.sobs, 1, ratio, gene.large_flag))
            if (pval > 1) or (expected_no==(n*pval) and n != 0) or pval==0: break
            sumpvals += pval
            expected_no += n * pval
            n += 1.0
        return expected_no

    @staticmethod
    def get_percentiles(gene, ratio, pofx_given_s=None, percentiles=(0.05, 0.95)):
        if pofx_given_s is None: pofx_given_s = gene.pofx_given_s
        lower_percentile = min(percentiles); upper_percentile = max(percentiles)
        lowest_val = None; highest_val = None
        sumpvals = 0
        n = 0.0
        expected_no = 0
        while sumpvals < 1:
            pval = float(pofx_given_s(n, gene.sobs, 1, ratio, gene.large_flag))
            if pval==np.inf: pval = float(pofx_given_s(n, gene.sobs, 1, ratio, 1))
            if (pval > 1) or (expected_no==(n*pval) and n != 0) or pval==0: break

            if lowest_val is None and sumpvals + pval > lower_percentile:
                lowest_val = n
            if highest_val is None and sumpvals + pval > upper_percentile:
                highest_val = n-1

            sumpvals += pval
            expected_no += n * pval
            n += 1.0
        return lowest_val, highest_val

    def get_selection(self, gene, nexp):

        if gene.nobs == nexp: return 0
        #neutral run, s=0 to rescale mut matrix
        #minimises x  |sub=f(x*mu, N, s=0)-nexp| 
        occurences=gene.occs
        nonsyn_target=1-gene.syn_target
        syn_prob= es.matri64x4_to_64x64(gene.syn_target)
        nonsyn_prob=es.matri64x4_to_64x64(nonsyn_target)
        
        #trying scling with nexp only
        #mutations_matrix, best_ratio = self.get_transition_matrix(gene, nexp+gene.sobs) #consider all subs-do not restrict to syn or non syn
        mutations_matrix, best_ratio = self.get_transition_matrix(gene, gene.sobs,syn_prob)
        mutations_matrix_host = pd.DataFrame(mutations_matrix.get())
        
        
        "TO DO check wether to divide along axis or index"
        
        sub_matrix_no_s=co.get_sub_matrix_from_mut_matrix_host(mutations_matrix_host, population_size=self.population_size,
                                                            selection=0)
        expected_s =co.get_number_of_mutations_syn(occurences, sub_matrix_no_s,syn_prob)
        print(f'expected s {expected_s},real s {gene.sobs}')
        
        tolerance = 0.1
        obs_subs = gene.nobs 
        print(f'obs subs {obs_subs}')
        neg_direction = True if gene.nobs < nexp else False
        
        def error_function(selection):
            print(selection)
            try:

                sub_matrix = co.get_sub_matrix_from_mut_matrix_host(mutations_matrix_host, population_size=self.population_size,
                                                            selection=selection)
                #if have nan entries then conitnue
                #check if sub_matrix is an identity matrix
                #if sub_matrix is None or not cupy.isfinite(sub_matrix).all():
                    #return np.inf
            
                if not neg_direction and (cupy.all(cupy.eye(64) == sub_matrix)):
                    return np.inf
                #TO DO EDIT THESE FUNCTIONS
                
                expected_n = co.get_number_of_mutations_syn(occurences, sub_matrix,nonsyn_prob)
                if np.isnan(expected_n):
                    return np.inf
                

                expected_subs=expected_n 
                print(f'expected subs {expected_subs}')
                return -1 if (expected_subs <= 0.01 and obs_subs== 0) else abs(expected_subs-obs_subs)

            except OverflowError as e:
                return np.inf

        selection = self.predictor.minimize_by_search_selection(error_function, self.selection_range,neg_direction)

        return selection
    

    def get_selection_test_n_s(self, gene, nexp):

        if gene.nobs == nexp: return 0
        #neutral run, s=0 to rescale mut matrix
        #minimises x  |sub=f(x*mu, N, s=0)-nexp| 

        #maria to do-edit so outputs sub matrix without selection-for syn target
        print('getting rescaled mut matrix')
        mutations_matrix, best_ratio = self.get_transition_matrix(gene, nexp+gene.sobs) #consider all subs-do not restrict to syn or non syn
        mutations_matrix_host = pd.DataFrame(mutations_matrix.get())
        
        occurences=gene.occs
        nonsyn_target=1-gene.syn_target
        for i in range(64):
            nonsyn_target.loc[i,i]=0.0
        syn_prob= es.matri64x4_to_64x64(gene.syn_target)
        nonsyn_prob=es.matri64x4_to_64x64(nonsyn_target)
        
        "TO DO check wether to divide along axis or index"
        
        sub_matrix_no_s=co.get_sub_matrix_from_mut_matrix_host(mutations_matrix_host, population_size=self.population_size,
                                                            selection=0)
        expected_s =co.get_number_of_mutations_syn(occurences, sub_matrix_no_s,syn_prob)
        print(f'expected s {expected_s}')
        
        tolerance = 0.1
        obs_subs = gene.nobs +gene.sobs
        neg_direction = True if gene.nobs < nexp else False
        print(neg_direction)
        def error_function(selection):
            print(selection)
            try:

                sub_matrix = co.get_sub_matrix_from_mut_matrix_host(mutations_matrix_host, population_size=self.population_size,
                                                            selection=selection)
                print(sub_matrix) #if have nan entries then conitnue
                #check if sub_matrix is an identity matrix
                #if sub_matrix is None or not cupy.isfinite(sub_matrix).all():
                    #return np.inf
            
                if not neg_direction and (cupy.all(cupy.eye(64) == sub_matrix)):
                    return np.inf
                #TO DO EDIT THESE FUNCTIONS
                print('calculating expected_n')
                expected_n = co.get_number_of_mutations_syn(occurences, sub_matrix,nonsyn_prob)
                if np.isnan(expected_n):
                    return np.inf
                print(f'expected n {expected_n}')
                print('calculating expected_s')
                
                expected_subs=expected_n+expected_s
                return -1 if (expected_subs <= 0.01 and obs_subs== 0) else abs(expected_subs-obs_subs)

            except OverflowError as e:
                return np.inf

        selection = self.predictor.minimize_by_search_selection(error_function, self.selection_range,neg_direction)

        return selection
    

    def get_selection_copy(self, gene, nexp):

        if gene.nobs == nexp: return 0
        #neutral run, s=0 to rescale mut matrix
        #minimises x  |sub=f(x*mu, N, s=0)-nexp| 

        #maria to do-edit so outputs sub matrix without selection-for syn target
        print('getting rescaled mut matrix')
        print(f' s obs is {gene.sobs}')
        mutations_matrix, best_ratio = self.get_transition_matrix(gene, gene.sobs+nexp) #consider all subs-do not restrict to syn or non syn
        mutations_matrix_host = pd.DataFrame(mutations_matrix.get())
        print(mutations_matrix_host)
        sub_matrix_no_s=co.get_sub_matrix_from_mut_matrix_host(mutations_matrix_host, population_size=self.population_size,
                                                            selection=0)
        
        occurences=gene.occs
        nonsyn_target=1-gene.syn_target
        syn_prob= es.matri64x4_to_64x64(gene.syn_target)
        nonsyn_prob=es.matri64x4_to_64x64(nonsyn_target)
        
        "TO DO check wether to divide along axis or index"
        
        print(f'syn prob {syn_prob}')
        print(f'non syn prob {nonsyn_prob}')
        
        tolerance = 0.1
        neg_direction = True if gene.nobs < nexp else False
        def error_function(selection):
            try:

                sub_matrix = co.get_sub_matrix_from_mut_matrix_host(mutations_matrix_host, population_size=self.population_size,
                                                            selection=selection)
                print(sub_matrix) #if have nan entries then conitnue
                #check if sub_matrix is an identity matrix
                if not neg_direction and (cupy.all(cupy.eye(64) == sub_matrix)):
                    return np.inf
                if sub_matrix is None or not cupy.isfinite(sub_matrix).all():
                    return np.inf
                #TO DO EDIT THESE FUNCTIONS
                print('calculating expected_n')
                expected_n = co.get_number_of_mutations_syn(occurences, sub_matrix,nonsyn_prob)
                print(f'expected n {expected_n}')
                print('calculating expected_s')
                expected_s =co.get_number_of_mutations_syn(occurences, sub_matrix_no_s,syn_prob)
                print(f'expected n {expected_n}')
                return -1 if (expected_n <= 0.01 and (gene.nobs +gene.sobs)== 0) else expected_n+expected_s

            except OverflowError as e:
                return np.inf



        try:

            selection = self.predictor.binary_search(error_function, self.selection_range, gene.nobs+gene.sobs,
                                                    negative=neg_direction, max_depth=30, tol=tolerance)
        except Exception as e:
            print(e)
            if e.args[0] == 'did not converge':
                def error_grid_function(selection):
                    n = error_function(selection)
                    #0.6 is a hack to deal with the case where the expected number of mutations is 0
                    return 0.6 if (n <= tolerance and gene.nobs == 0) else abs(n-gene.nobs-gene.sobs)

                selection = self.predictor.minimize_by_search(error_grid_function, self.selection_range,
                                              negative=neg_direction, max_depth=30, breadth=5,
                                              nearest_value_tie_breaker=0, tol=tolerance, parallel=False)

        return selection
        '''
        else:
            raise Exception('not yet implemented')

            if get_subs_based_on_selection(nobs, gene_df, occs_df, syn_ratio, missense_ratio, max(selection_range)) > nexp:
                return 1
            else:
                def error_function(selection):
                    return get_subs_based_on_selection(nobs, gene_df, occs_df, syn_ratio, missense_ratio, selection) - nobs
                selection = predictor.minimize_by_search(error_function, selection_range, negative=False)
                return selection'''

    def get_transition_matrix(self, gene, nexp, prob,ratio=None):
        if ratio is None: ratio = gene.syn_ratio
        #syn_prob= es.matri64x4_to_64x64(gene.syn_target)
        mu_matrix, best_ratio = co.get_transition_matrix_for_observed_mutations(nexp, gene.occs, prob, population_size=self.population_size, selection=0)
        return mu_matrix, best_ratio

    def get_mutations_matrix(self, transition_matrix):
        #TODO: add the mutations matrix
        pass

    def get_subs_based_on_selection__(self, sub_matrix, gene):
        #TODO rempve this function
        #with pd.option_context('display.precision', 100):print(sub_matrix[0])

        sub_matrix_device = cupy.array(sub_matrix)
        corrected_t_n = cupy.linalg.matrix_power(sub_matrix_device, self.generations)
        corrected_t_n_cpu = corrected_t_n.get()
        corrected_t_n_cpu_pandas = pd.DataFrame(corrected_t_n_cpu)
        #with pd.option_context('display.precision', 100):print(corrected_t_n_cpu_pandas[0])

        cupy.fill_diagonal(corrected_t_n, 0)

        occs = cupy.array(gene.occs.max(axis=1))
        corrected_expected_df = cupy.dot(occs, corrected_t_n)

        return corrected_expected_df.get().sum().sum()

    def get_sub(self, point):
        command = f'{wfes_single} --fixation -N {point.pop_size} -s {point.selection_coeff*2} \
                    -v {point.mue_frwrd} -u {point.mue_bckwrds} -h 0.5'

        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        output = output.decode()
        lines = output.split('\n')
        for line in lines:
            if 'T_fix' in line:
                time = float(line.split('=')[1].strip())

        return 1/time

class Gene:
    ''' to be changed to a modified gene class'''
    def __init__(self, pickled_gene, large_flag=0, matrix=None, occs=None, syn_target=None,pofx_given_s=None):
        self.gene_name = pickled_gene['gene']
        self.sobs = int(pickled_gene["obs"][2])
        self.mobs = int(pickled_gene["obs"][0])
        self.kobs = int(pickled_gene["obs"][1])
        self.nobs = self.mobs + self.kobs

        self.sexp = pickled_gene["exp"][2]
        self.mexp = pickled_gene["exp"][0]
        self.kexp = pickled_gene["exp"][1]

        self.ratm = self.mexp/self.sexp
        self.ratk = self.kexp/self.sexp
        self.ratn = (self.kexp + self.mexp)/self.sexp
        self.syn_ratio = self.sobs/self.sexp
        self.large_flag = large_flag
        self.length = pickled_gene['len']

        self.matrix = matrix
        self.occs = occs
        self.syn_target = syn_target
        self.pofx_given_s = pofx_given_s

        self.expected_selection = None
        self.max_possible_highest_selection = None
        self.min_possible_highest_selection = None