#!/usr/bin/env python3
'''Test recovery of the simulated per-gene mutation-rate scalings.
by default loads t he mu matrix from files-that was used as input into the simulations
then finds scalings relative to this
TO RUN: python -m test_pipeline.scripts.run_gene_mu_inference.py
'''

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

import essentials as es
from selection_operations import SelectionOperations


DEFAULT_DIR = Path('test_pipeline/mu_sub_check/forward_sim_4_allele')
input_muts = DEFAULT_DIR / 'whole_genome_dict_forward_sim_4_allele_unsmoothed_muts.pkl'
input_trinucs = DEFAULT_DIR / 'whole_genome_dict_forward_sim_4_allele_unsmoothed_occs.pkl'


def load_simulation(muts_path=input_muts, trinucs_path=input_trinucs):
    with open(muts_path, 'rb') as handle:
        muts = pickle.load(handle)
    with open(trinucs_path, 'rb') as handle:
        trinucs = pickle.load(handle)
    if set(muts) != set(trinucs):
        raise ValueError('mutation and occurrence pickles have different chromosomes')
    return muts, trinucs


def load_simulated_mutation_matrix(simulation_dir, truth_matrix=None):
    '''
    load mu matrix inputted into simulations'''
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


def make_gene(chrom, column, muts, occs):
    '''makes a fake gene from forward simulations trinucs in anc and observed subs
    assumes for now that all the simulated sites are nonsyn '''

    nonsyn_sites = []
    for context in es.CONTEXTS:
        alts = ''.join(alt for alt in es.BASES if alt != context[1])
        nonsyn_sites.append({
            'trinucleotide': context,
            'reference_base': context[1],
            'nonsyn_alt_bases': alts,
            'count': float(occs.loc[f'{context}->{alts[0]}', column]),
        })
    return SimpleNamespace(
        gene_name=f'{chrom}:{column}',
        nonsyn_sites=nonsyn_sites,
        nobs=float(muts[column].sum()),
    )


def run_get_gene_scaling(so, gene, nexp, scale_bounds=(1e-4, 5.0)):
    '''
    finds the per gene scaling by trying rescaling of mut matrix M
    for earch rescaling recalculate Q, Q^T and expected observed subs from Q^T 
    and inital states, then chose mu scaling which minimises difference from nexp'''
    if not gene.nonsyn_sites:
        if gene.nobs == 0:
            return 0.0
        raise ValueError(f'No sites were parsed for {gene.gene_name}')
    if nexp < 0:
        raise ValueError(f'Observed substitutions cannot be negative: {nexp}')

    def neutral_error(log_scale):
        try:
            expected = so._expected_four_allele_nonsyn(
                gene, 10 ** log_scale, 0.0
            )
        except ValueError:
            return np.inf
        return abs(expected - nexp)

    solution = minimize_scalar(
        neutral_error,
        bounds=tuple(np.log10(scale_bounds)),
        method='bounded',
        options={'xatol': 1e-8},
    )
    if not solution.success or not np.isfinite(solution.fun):
        raise RuntimeError(
            f'Could not infer mutation-rate scale for {gene.gene_name}'
        )
    return float(10 ** solution.x)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--simulation-dir', type=Path, default=DEFAULT_DIR)
    parser.add_argument(
        '--output-dir', type=Path,
        default=DEFAULT_DIR / 'inference_4_4_recurrence',
    )
    parser.add_argument('--truth-matrix', type=Path)
    parser.add_argument('--population-size', type=int, default=200000)
    parser.add_argument('--generations', type=int, default=160000)
    parser.add_argument(
        '--max-bins', type=int, default=10,
        help='number of bins to test; 0 means all',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    muts, occs = load_simulation(
        args.simulation_dir / input_muts.name,
        args.simulation_dir / input_trinucs.name,
    )
    metadata = pd.read_csv(
        args.simulation_dir / 'bin_metadata.tsv', sep='\t'
    ).set_index(['chr', 'bin'])
    metadata=metadata.iloc[:100]
    truth = load_simulated_mutation_matrix(
        args.simulation_dir, args.truth_matrix
    )
    operations = SelectionOperations(
        generations=args.generations, population_size=args.population_size
    )
    operations.general_mut_matrix = truth

    rows = []
    limit = None if args.max_bins == 0 else args.max_bins
    for number, ((chrom, bin_id), truth_row) in enumerate(metadata.iterrows()):
        if limit is not None and number >= limit:
            break
        gene = make_gene(chrom, bin_id, muts[chrom], occs[chrom])
        inferred_scale = run_get_gene_scaling(operations, gene, gene.nobs)
        rows.append({
            'chr': chrom,
            'bin': bin_id,
            'simulated_mu_scale': truth_row.mu_scale,
            'inferred_mu_scale': inferred_scale,
            'observed_substitutions': gene.nobs,
        })
        print(
            f'{gene.gene_name}: simulated={truth_row.mu_scale:.8g}, '
            f'inferred={inferred_scale:.8g}'
        )

    results = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(
        args.output_dir / 'per_bin_scale_recovery.tsv', sep='\t', index=False
    )
    if not results.empty:
        correlation = results[
            ['simulated_mu_scale', 'inferred_mu_scale']
        ].corr().iloc[0, 1]
        print(f'per_bin_scale_correlation: {correlation:.12g}')
    print(f'Outputs written to {args.output_dir}')


if __name__ == '__main__':
    main()
