import math
import subprocess
import time
import cupy
import pandas as pd
import essentials as es
import cuda_operations as co
import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import minimize_scalar, brentq


from predictor import Predictor
from graphpoint import GraphPointAbstract as GPA

BASES = ['A', 'C', 'G', 'T']
CONTEXTS = [f'{b1}{b2}{b3}' for b1 in BASES for b2 in BASES for b3 in BASES]
wfes_single = '/home/hossam26644/Documents/wfes/wfes2/bin/wfes_single'

class SelectionOperations:
    #nonsyn max 0.0001
    #nonsense max 0.01
    def __init__(self, selection_range=[0.0000001,0.001], generations=160000,
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
        #for speeding only run expected, not high,low intervals
        '''
        lowest_n, highest_n = self.get_percentiles(gene, gene.ratn)
        lowest_selection = self.get_selection(gene, highest_n)
        print(f'{gene.gene_name}: lowest = {lowest_selection}, expected = {highest_n}, obs = {gene.nobs}')
        highest_selection = self.get_selection(gene, lowest_n)
        print(f'{gene.gene_name}: highest = {highest_selection}, expected = {lowest_n}, obs = {gene.nobs}')
        '''
        #only if not running lower/upper limits
        highest_selection= selection
        lowest_selection=selection
        lowest_n=expected_n
        highest_n=expected_n


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
        if ratio == 0:
            return 0.0
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


    def get_selection_no_parallel(self,gene,nexp):
        '''
        alternative idea to avoid implementing the 4 allele maths with selection
        to apply to genes with strong selection so no recurrence or parallele mutations
        do not need to apply the 4 alelle maths correction or Q^T correction
        '''
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
        Ne = self.population_size
        if Ne is None or Ne <= 0:
            raise ValueError('population_size must be positive')

        
        #expected mutations rather than expected subs-just rescale with mu and do not do recurrence correction? 
        exp_muts = self._expected_four_allele_mutation_nonsyn(gene, scale,recurrence=True)
        if exp_muts <= 0:
            raise ValueError(
                f'Expected nonsynonymous mutations must be positive for '
                f'{gene.gene_name}: {exp_muts}'
            )
        w = gene.nobs / exp_muts
        if np.isclose(w, 1.0):
            return 0.0

        #compare to wright fisher diffusion approximations to sub rate with selection/sub rate without selection
        #2N(1-e**(-2s))/1-e**(-2Ns)
        from scipy.optimize import brentq

        def fixation_ratio(s):
            if s == 0:
                return 1.0 / Ne

            return 2*Ne*(np.expm1(-2*s)) / np.expm1(-2*Ne*s)

        def error(s):
            return fixation_ratio(s) - w

        lower, upper = self.selection_range

        if w < 1:
            lower, upper = -upper, -lower

        selection = brentq(error, lower, upper)

        return float(selection)

        


    def get_selection(self, gene, nexp):

        if gene.nobs == nexp: return 0
        if gene.nonsyn_sites is not None:
            #testing no parallel, no recurrence correction method 
            #return self.get_selection_no_parallel(gene,nexp)
            return self.get_selection_four_allele(gene, nexp)
        else:
            raise ValueError('did not supply trinucs for 4 allele model ')

        #neutral run, s=0 to rescale mut matrix
        #minimises x  |sub=f(x*mu, N, s=0)-nexp| 
        occurences=gene.occs
        nonsyn_target=1-gene.syn_target
        syn_prob= es.matri64x4_to_64x64(gene.syn_target)
        nonsyn_prob=es.matri64x4_to_64x64(nonsyn_target)
        

        #trying scling with nexp only
        #mutations_matrix, best_ratio = self.get_transition_matrix(gene, nexp+gene.sobs) #consider all subs-do not restrict to syn or non syn
        #mutations_matrix, best_ratio = self.get_transition_matrix(gene, gene.sobs,syn_prob)
        mutations_matrix, best_ratio = self.get_transition_matrix(gene,nexp,nonsyn_prob)
        #mutations_matrix_host = pd.DataFrame(mutations_matrix.get())
        
        
        "TO DO check wether to divide along axis or index"
        #index=target, columns=ref
        #host not given same output as database using inside get transition matrix
        #sub_matrix_no_s=co.get_sub_matrix_from_mut_matrix_host(mutations_matrix_host, population_size=self.population_size,selection=0)
        
        #pd.DataFrame(data=best_mu_matrix.get()).to_csv(f'test/mu_matrix_s_0')
        #sub_matrix_no_s=co.get_sub_matrix_from_mut_matrix_device(mutations_matrix, self.population_size, selection=0)
        #sub_matrix_no_s=co.get_sub_matrix_from_scaled_mut_matrix_database(mutations_matrix, population_size=self.population_size,selection=0,scale=best_ratio)
        #pd.DataFrame(data=sub_matrix_no_s.get()).to_csv(f'test/sub_matrix_s_0')
        #check and cols =ref trinucs, index =alt trinucs
        # not correctly rescaling based on best ratio 
        #expected_s =co.get_number_of_mutations_syn(occurences, sub_matrix_no_s,syn_prob)
        #print(f'expected s {expected_s},real s {gene.sobs}')
        
        tolerance = 0.1
        obs_subs = gene.nobs 
        print(f'obs subs {obs_subs}')
        neg_direction = True if gene.nobs < nexp else False
        
        def error_function(selection):
            print(selection)
            try:
                
                #problem with sub matrix host 
                sub_matrix=co.get_sub_matrix_from_mut_matrix_device(mutations_matrix, self.population_size, selection=selection)
                #sub_matrix=co.get_sub_matrix_from_scaled_mut_matrix_database(mutations_matrix, population_size=self.population_size,selection=selection,scale=best_ratio)
                #sub_matrix = co.get_sub_matrix_from_mut_matrix_host(mutations_matrix_host, population_size=self.population_size,selection=selection)
                #pd.DataFrame(data=sub_matrix.get()).to_csv(f'test/sub_matrix_s_{selection}')
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
        print(neg_direction)
        selection = self.predictor.minimize_by_search_selection(error_function, self.selection_range,neg_direction)

        return selection

    def _get_four_allele_mutation_matrix(self, left, right, scale):
        if not hasattr(self, 'general_mut_matrix'):
            self.general_mut_matrix = pd.read_csv(
                'files/best_in_df_mutations_4_allele.csv',
                index_col=0,
                sep='\t',
            )
        

        matrix = {base: {alt: 0.0 for alt in BASES} for base in BASES}
        for ref in BASES:
            source_context = left + ref + right
            for alt in BASES:
                if alt == ref:
                    continue
                target_context = left + alt + right
                matrix[ref][alt] = (
                    float(self.general_mut_matrix.loc[source_context, target_context])
                    * scale
                )
        return matrix
    

    def _expected_four_allele_mutation_nonsyn(self, gene, scale,recurrence=False):
        '''
        Count sites ending in a nonsynonymous state after the generation interval.
        #rescale non digagonal entries with scale, recalculate dignaonal =1-sum 9row)
        #multply be nonsyn trinuc->alt counts *generations
        #sum to get expected number of muts
        '''
        if not hasattr(self, 'general_mut_matrix'):
            self.general_mut_matrix = pd.read_csv(
                'files/best_in_df_mutations_4_allele.csv',
                index_col=0,
                sep='\t',
            )
        expected = 0.0
        if not recurrence:
            for site in gene.nonsyn_sites:
                trinucleotide = site['trinucleotide']
                count = float(site['count'])
                for alt in site['nonsyn_alt_bases']:
                    target = trinucleotide[0] + alt + trinucleotide[2]
                    expected += (
                        count
                        * float(self.general_mut_matrix.loc[trinucleotide, target])
                        * scale
                        * self.generations
                    )
            return expected
        
        matrix_cache = {}
        transition_cache = {}
        endpoint_cache = {}
        grouped_sites = {}
        for site in gene.nonsyn_sites:
            trinucleotide = site['trinucleotide']
            ref = site['reference_base']
            nonsyn_alts = tuple(site['nonsyn_alt_bases'])
            left, right = trinucleotide[0], trinucleotide[2]
            context_key = (left, right)
            site_key = (context_key, ref, nonsyn_alts)
            grouped_sites[site_key] = (
                grouped_sites.get(site_key, 0.0) + float(site['count'])
            )

        #calculate sub rates on trinuc groups =per group
        expected = cupy.asarray(0.0, dtype=cupy.float64)
        for (context_key, ref, nonsyn_alts), count in grouped_sites.items():
            if context_key not in matrix_cache:
                left, right = context_key
                matrix_cache[context_key] = self._get_four_allele_mutation_matrix(
                    left, right, scale
                )
            left, right = context_key
            trinucleotide = left + ref + right

            # With neutrality, all nonsynonymous classes sharing a flank context
            # have the same rate and transition matrices, irrespective of which
            # alternate alleles are counted as nonsynonymous.
            
            transition_key = (context_key, ref, nonsyn_alts)
            endpoint_key = transition_key

            if transition_key not in transition_cache:
                
                rates = matrix_cache[context_key]

                rate_matrix = np.array(
                    [
                        [rates[source][target] for target in BASES]
                        for source in BASES
                    ],
                    dtype=float,
                )
                if not np.isfinite(rate_matrix).all() or (rate_matrix < 0).any():
                    raise ValueError(
                        f'Invalid substitution rates for {trinucleotide}, '
                        f'{ref}, nonsynonymous alts {nonsyn_alts}'
                    )

                #4x4 transition matrix under config
                transition_matrix = rate_matrix.copy()
                np.fill_diagonal(transition_matrix, 0.0)
                for source_index, source in enumerate(BASES):
                    outgoing_rate = transition_matrix[source_index].sum()
                    if outgoing_rate > 1.0:
                        raise ValueError(
                            f'Outgoing substitution rate exceeds one per generation '
                            f'for {trinucleotide} {source}: {outgoing_rate}'
                        )
                    transition_matrix[source_index, source_index] = 1.0 - outgoing_rate

                P = cupy.asarray(transition_matrix, dtype=cupy.float64)
                transition_cache[transition_key] = cupy.linalg.matrix_power(
                    P, self.generations
                )

            if endpoint_key not in endpoint_cache:
                initial_counts = cupy.zeros(len(BASES), dtype=cupy.float64)
                initial_counts[BASES.index(ref)] = 1.0
                endpoint_cache[endpoint_key] = cupy.dot(
                    initial_counts, transition_cache[transition_key]
                )
            #only add porbability of the nonsyn subs to happen
            endpoint_probabilities = endpoint_cache[endpoint_key]
            nonsyn_indices = [BASES.index(alt) for alt in nonsyn_alts]
            expected += count * cupy.sum(endpoint_probabilities[nonsyn_indices])
        return float(expected.get())




    def _expected_four_allele_nonsyn(self, gene, scale, selection):
        '''
        Count sites ending in a nonsynonymous state after the generation interval.
        Each reference/context/nonsynonymous-alternate class has its own 4x4
        transition matrix because its fitness assignments depend on that class.
        '''
        if self.population_size is None:
            raise ValueError('population_size is required for four-allele inference')

        point = GPA(0, 0, self.population_size)
        matrix_cache = {}
        transition_cache = {}
        endpoint_cache = {}
        grouped_sites = {}
        for site in gene.nonsyn_sites:
            trinucleotide = site['trinucleotide']
            ref = site['reference_base']
            nonsyn_alts = tuple(site['nonsyn_alt_bases'])
            left, right = trinucleotide[0], trinucleotide[2]
            context_key = (left, right)
            site_key = (context_key, ref, nonsyn_alts)
            grouped_sites[site_key] = (
                grouped_sites.get(site_key, 0.0) + float(site['count'])
            )

        #calculate sub rates on trinuc groups =per group
        expected = cupy.asarray(0.0, dtype=cupy.float64)
        for (context_key, ref, nonsyn_alts), count in grouped_sites.items():
            if context_key not in matrix_cache:
                left, right = context_key
                matrix_cache[context_key] = self._get_four_allele_mutation_matrix(
                    left, right, scale
                )
            left, right = context_key
            trinucleotide = left + ref + right

            # With neutrality, all nonsynonymous classes sharing a flank context
            # have the same rate and transition matrices, irrespective of which
            # alternate alleles are counted as nonsynonymous.
            if selection == 0:
                transition_key = context_key
                endpoint_key = (context_key, ref)
            else:
                transition_key = (context_key, ref, nonsyn_alts, selection)
                endpoint_key = transition_key

            if transition_key not in transition_cache:
                fitness = {base: 1.0 for base in BASES}
                for alt in nonsyn_alts:
                    fitness[alt] = 1.0 + selection
                rates = self.predictor.get_sub_rate_diffusion_4_allele_renewal_selection(
                    point,
                    matrix_cache[context_key],
                    fitness_matrix=fitness,
                )

                rate_matrix = np.array(
                    [
                        [rates[source][target] for target in BASES]
                        for source in BASES
                    ],
                    dtype=float,
                )
                if not np.isfinite(rate_matrix).all() or (rate_matrix < 0).any():
                    
                    print(
                        f'Invalid substitution rates for {trinucleotide}, '
                        f'{ref}, nonsynonymous alts {nonsyn_alts}'
                    )
                    return np.inf

                #4x4 transition matrix under config
                transition_matrix = rate_matrix.copy()
                np.fill_diagonal(transition_matrix, 0.0)
                for source_index, source in enumerate(BASES):
                    outgoing_rate = transition_matrix[source_index].sum()
                    if outgoing_rate > 1.0:
                        raise ValueError(
                            f'Outgoing substitution rate exceeds one per generation '
                            f'for {trinucleotide} {source}: {outgoing_rate}'
                        )
                    transition_matrix[source_index, source_index] = 1.0 - outgoing_rate

                P = cupy.asarray(transition_matrix, dtype=cupy.float64)
                transition_cache[transition_key] = cupy.linalg.matrix_power(
                    P, self.generations
                )

            if endpoint_key not in endpoint_cache:
                initial_counts = cupy.zeros(len(BASES), dtype=cupy.float64)
                initial_counts[BASES.index(ref)] = 1.0
                endpoint_cache[endpoint_key] = cupy.dot(
                    initial_counts, transition_cache[transition_key]
                )
            #only add porbability of the nonsyn subs to happen
            endpoint_probabilities = endpoint_cache[endpoint_key]
            nonsyn_indices = [BASES.index(alt) for alt in nonsyn_alts]
            expected += count * cupy.sum(endpoint_probabilities[nonsyn_indices])
        return float(expected.get())

    def get_selection_four_allele(self, gene, nexp):
        if not gene.nonsyn_sites:
            
            raise ValueError(
               
                f'No possible nonsynonymous sites were parsed for {gene.gene_name}'
            )
        #if gene.nobs == 0:
            #return 0.0
        if nexp <= 0 or np.isnan(nexp):
            raise ValueError(
                f'Cannot calibrate a positive mutation rate from expected_n={nexp} '
                f'for {gene.gene_name}'
            )
        
        print(gene.nobs,nexp)

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

        print(f'{gene.gene_name} scale={scale}')
        
        lower, upper = self.selection_range
        if gene.nobs < nexp:
            lower, upper = -upper, -lower

        tolerance=max((upper - lower) * 1e-5, 1e-8)
        #tolerance=max((upper - lower) * 1e-8, 1e-14)

        if gene.nobs == 0:

            if tolerance <= 0:
                raise ValueError("Tolerance must be positive")

            def expected_n(s):
                try:
                    exp=self._expected_four_allele_nonsyn(gene, scale, s)
                    print(s,exp)
                    return exp
                except (ValueError, OverflowError):
                    return np.inf

            # Check neutrality first
            if expected_n(0.0) <= tolerance:
                return 0.0

            # Find a finite expected value below tolerance
            # Search from strongest negative selection towards zero
            s_values = np.logspace(lower, 0.0, 100)

            strongest_valid_s = None

            for s in s_values:
                exp = expected_n(s)

                if np.isfinite(exp) and exp <= tolerance:
                    strongest_valid_s = s
                    break

            if strongest_valid_s is None:
                raise ValueError(
                    f'Cannot reach tolerance={tolerance} for {gene.gene_name}'
                )

            # Find weakest negative selection satisfying tolerance
            solution = brentq(
                lambda s: expected_n(s) - tolerance,
                strongest_valid_s,
                0.0,
                xtol=1e-10,
            )

            return float(solution)

        #minimize error between n obs and E[n|s]
        lower, upper = self.selection_range
        if gene.nobs < nexp:
            lower, upper = -upper, -lower

        def selection_error(selection):
            try:
                expected = self._expected_four_allele_nonsyn(
                    gene, scale, selection
                )
            except ValueError:
                return np.inf
            print(f'{gene.gene_name}: selection = {selection}, expected = {expected}, obs = {gene.nobs}')
            return abs(expected - gene.nobs)

        selection_solution = minimize_scalar(
            selection_error,
            bounds=(lower, upper),
            method='bounded',
            options={'xatol': tolerance},
        )
        if not selection_solution.success or not np.isfinite(selection_solution.fun):
            raise RuntimeError(
                f'Could not infer four-allele selection for {gene.gene_name}'
            )
        return float(selection_solution.x)
    



    def get_transition_matrix(self, gene, nexp, prob=False,ratio=None):
        if ratio is None: ratio = 1
        #syn_prob= es.matri64x4_to_64x64(gene.syn_target)
        mu_matrix, best_ratio = co.get_transition_matrix_for_observed_mutations(nexp, gene.occs,prob, population_size=self.population_size, selection=0)
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
    def __init__(self, pickled_gene, large_flag=0, matrix=None, occs=None, syn_target=None,pofx_given_s=None,nonsyn_sites=None):
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
        self.nonsyn_sites = nonsyn_sites

        self.expected_selection = None
        self.max_possible_highest_selection = None
        self.min_possible_highest_selection = None
