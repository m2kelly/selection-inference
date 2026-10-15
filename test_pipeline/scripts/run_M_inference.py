#!/usr/bin/env python3
'''Test recovery of the global mutation matrix from the simulation, so testing recurrence correction 
and inversion of sub=f(mu,N) in get_mu 
TO RUN: python -m test_pipeline.scripts.run_M_inference.py'''


from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

import essentials as es
from cuda_operations import (
    get_mut_matrix_from_sub_matrix,
    get_transition_cpmatrix_from_rates,
    get_transition_cpmatrix_from_rates_greedy
)


DEFAULT_DIR = Path('test_pipeline/mu_sub_check/forward_sim_no_mu')
input_muts = DEFAULT_DIR / 'whole_genome_dict_forward_sim_no_mu_unsmoothed_muts.pkl'
input_trinucs = DEFAULT_DIR / 'whole_genome_dict_forward_sim_no_mu_unsmoothed_occs.pkl'


def load_simulation(muts_path=input_muts, trinucs_path=input_trinucs):
    '''load pickles'''
    with open(muts_path, 'rb') as handle:
        muts = pickle.load(handle)
    with open(trinucs_path, 'rb') as handle:
        trinucs = pickle.load(handle)
    if set(muts) != set(trinucs):
        raise ValueError('mutation and occurrence pickles have different chromosomes')
    return muts, trinucs


def find_global_Q_t(muts_dict, trinucs_dict):
    '''Return observed endpoint rates and starting trinucleotide counts.'''
    first = muts_dict[next(iter(muts_dict))]
    mutation_totals = pd.Series(0.0, index=first.index)
    occurrence_totals = pd.Series(0.0, index=es.CONTEXTS)
    for chrom in muts_dict:
        mutation_totals = mutation_totals.add(
            muts_dict[chrom].sum(axis=1), fill_value=0.0
        )
        occ = trinucs_dict[chrom]
        for context in es.CONTEXTS:
            rows = [f'{context}->{alt}' for alt in es.BASES if alt != context[1]]
            context_occs = occ.loc[rows]
            if not np.allclose(
                context_occs.to_numpy(), context_occs.iloc[0].to_numpy()
            ):
                raise ValueError(
                    f'occurrences disagree between alternate alleles for {context}'
                )
            occurrence_totals.loc[context] += context_occs.iloc[0].sum()

    rates = pd.DataFrame(0.0, index=es.CONTEXTS, columns=es.BASES)
    for mutation, count in mutation_totals.items():
        context, alt = mutation.split('->')
        rates.loc[context, alt] = count / occurrence_totals.loc[context]
    return rates, occurrence_totals.to_frame('count')


def calculate_general_muts_matrix(sub_matrix, pop_size, output_path):
    '''input recurrence corrected sub matrix per generation Q, and infer mut matrix'''
    muts_matrix = get_mut_matrix_from_sub_matrix(sub_matrix, pop_size)
    muts_matrix.to_csv(output_path, sep='\t')
    return muts_matrix


def global_recurrence_correction(rates, trinucs, generations, output_path):
    '''input Q^T and intial trinuc counts, output per generation sub matrix Q'''
    #test with greedy and without 
    sub_matrix = get_transition_cpmatrix_from_rates_greedy(
        rates, trinucs, generations=generations
    )
    sub_matrix.to_csv(output_path, sep='\t')
    return sub_matrix


def load_simulated_mutation_matrix(simulation_dir, truth_matrix=None):
    '''
    convert from input .pkl format into 64x64 rate dfs'''
    if truth_matrix is not None and truth_matrix.is_file():
        return pd.read_csv(truth_matrix, index_col=0, sep='\t')
    audit = pd.read_csv(simulation_dir / 'mut_sub_rates.tsv')
    first_mu = audit.loc[audit.mu_index == audit.mu_index.min()]
    matrix = pd.DataFrame(0.0, index=es.CONTEXTS, columns=es.CONTEXTS)
    for row in first_mu.itertuples(index=False):
        context, alt = row.mutation.split('->')
        target = context[0] + alt + context[2]
        matrix.loc[context, target] = row.mutation_rate_per_generation / row.mu_scale
    for context in es.CONTEXTS:
        matrix.loc[context, context] = 1.0 - matrix.loc[context].sum()
    return matrix


def off_diagonal_values(matrix):
    '''return values as nparrary for comparisons'''
    return np.asarray([
        float(matrix.loc[context, context[0] + alt + context[2]])
        for context in es.CONTEXTS for alt in es.BASES if alt != context[1]
    ])


def compare_matrices(inferred, simulated):
    '''summary statistics about inputted mut matrix in simulations vs inferred matrix here'''
    inferred_values = off_diagonal_values(inferred)
    simulated_values = off_diagonal_values(simulated)
    nonzero = simulated_values > 0
    relative = np.abs(
        inferred_values[nonzero] - simulated_values[nonzero]
    ) / simulated_values[nonzero]
    return {
        'matrix_rmse': np.sqrt(np.mean((inferred_values - simulated_values) ** 2)),
        'matrix_median_relative_error': np.median(relative),
        'matrix_max_relative_error': np.max(relative),
        'matrix_scale_ratio': (
            np.dot(inferred_values, simulated_values)
            / np.dot(simulated_values, simulated_values)
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--simulation-dir', type=Path, default=DEFAULT_DIR)
    parser.add_argument(
        '--output-dir', type=Path,
        default=DEFAULT_DIR / 'inference_64_64_recurrence',
    )
    parser.add_argument('--truth-matrix', type=Path)
    parser.add_argument('--population-size', type=int, default=200000)
    parser.add_argument('--generations', type=int, default=160000)
    return parser.parse_args()


def main():
    args = parse_args()
    muts, occs = load_simulation(
        args.simulation_dir / input_muts.name,
        args.simulation_dir / input_trinucs.name,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rates, trinucs = find_global_Q_t(muts, occs)
    sub_matrix = global_recurrence_correction(
        rates, trinucs, args.generations,
        args.output_dir / 'inferred_substitution_matrix.tsv',
    )
    inferred = calculate_general_muts_matrix(
        sub_matrix, args.population_size,
        args.output_dir / 'inferred_mutation_matrix.tsv',
    )
    truth = load_simulated_mutation_matrix(
        args.simulation_dir, args.truth_matrix
    )
    print('Global matrix inference complete')
    for name, value in compare_matrices(inferred, truth).items():
        print(f'{name}: {value:.12g}')


if __name__ == '__main__':
    main()
