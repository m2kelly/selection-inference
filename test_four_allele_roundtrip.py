#!/usr/bin/env python3
"""Test recovery of a known selection coefficient from synthetic expected counts.

The targets file is the TSV/TSV.GZ summary produced by ``parse_genes_pkl.py``.
This is a deterministic model round trip, not a test with Poisson-sampled counts.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from selection_operations import SelectionOperations


DEFAULT_TARGET_COLUMNS = {
    "gene",
    "trinucleotide",
    "reference_base",
    "nonsyn_alt_bases",
    "count",
}


def parse_nonsyn_alt_bases(value: object) -> str:
    text = str(value)
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        parsed = text

    bases = "".join(parsed) if isinstance(parsed, (tuple, list)) else str(parsed)
    if not bases or any(base not in "ACGT" for base in bases):
        raise ValueError(f"invalid nonsynonymous alternate-base list: {value!r}")
    return bases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--targets",
        type=Path,
        default=Path("files/four_allele_site_summary.tsv.gz"),
        help="parsed site summary TSV/TSV.GZ from parse_genes_pkl.py",
    )
    parser.add_argument("--gene", help="gene name in the parsed site summary")
    parser.add_argument("--selection", "-s", type=float, required=True)
    parser.add_argument("--population-size", "-N", type=int, default=200000)
    parser.add_argument(
        "--mutation-matrix",
        type=Path,
        default=Path("files/best_in_df_mutations_4_allele.csv"),
        help="global 64x64 four-allele mutation matrix",
    )
    parser.add_argument(
        "--selection-range",
        type=float,
        nargs=2,
        default=(1e-7, 1e-4),
        metavar=("MIN", "MAX"),
        help="absolute selection bounds used by the pipeline (default: 1e-7 1e-4)",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=160000,
        help="number of generations, matching the pipeline default",
    )
    parser.add_argument(
        "--relative-tolerance",
        type=float,
        default=0.01,
        help="maximum relative error allowed for recovery (default: 0.01)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.population_size <= 0 or args.generations <= 0:
        raise ValueError("population size and generations must be positive")

    lower, upper = args.selection_range
    if lower <= 0 or upper < lower:
        raise ValueError("selection bounds must satisfy 0 < MIN <= MAX")
    if args.relative_tolerance < 0:
        raise ValueError("relative tolerance must be non-negative")
    if not lower <= abs(args.selection) <= upper and args.selection != 0:
        raise ValueError(
            f"abs(selection) must be within the selection range [{lower}, {upper}]"
        )
    if not args.targets.is_file():
        raise FileNotFoundError(f"parsed targets file not found: {args.targets}")
    if not args.mutation_matrix.is_file():
        raise FileNotFoundError(
            f"four-allele mutation matrix not found: {args.mutation_matrix}"
        )

    summary = pd.read_csv(args.targets, sep="\t")
    missing = DEFAULT_TARGET_COLUMNS - set(summary.columns)
    if missing:
        raise ValueError(
            f"parsed targets file is missing columns: {sorted(missing)}"
        )

    gene_rows = summary.loc[summary["gene"] == args.gene]
    if gene_rows.empty:
        raise ValueError(f"no parsed nonsynonymous targets found for {args.gene}")

    nonsyn_sites = [
        {
            "trinucleotide": str(row.trinucleotide),
            "reference_base": str(row.reference_base),
            "nonsyn_alt_bases": parse_nonsyn_alt_bases(row.nonsyn_alt_bases),
            "count": int(row.count),
        }
        for row in gene_rows.itertuples(index=False)
    ]

    operations = SelectionOperations(
        selection_range=[lower, upper],
        generations=args.generations,
        population_size=args.population_size,
    )
    operations.general_mut_matrix = pd.read_csv(
        args.mutation_matrix, index_col=0, sep="\t"
    )
    gene = SimpleNamespace(
        gene_name=args.gene,
        nonsyn_sites=nonsyn_sites,
        nobs=0.0,
    )

    neutral_expected = operations._expected_four_allele_nonsyn(
        gene, scale=1.0, selection=0.0
    )
    selected_expected = operations._expected_four_allele_nonsyn(
        gene, scale=1.0, selection=args.selection
    )
    if not pd.notna(neutral_expected) or neutral_expected <= 0:
        raise ValueError(
            f"invalid neutral expected count for {args.gene}: {neutral_expected}"
        )
    if not pd.notna(selected_expected) or selected_expected < 0:
        raise ValueError(
            f"invalid selected expected count for {args.gene}: {selected_expected}"
        )

    #framework to test selection inference recovery 
    gene.nobs = selected_expected
    recovered = operations.get_selection(gene, neutral_expected)
    print(f"gene: {args.gene}")
    print(f"population_size: {args.population_size}")
    print(f"generations: {args.generations}")
    print(f"nonsynonymous target groups: {len(nonsyn_sites)}")
    print(f"neutral expected substitutions: {neutral_expected:.12g}")
    print(f"selected expected substitutions: {selected_expected:.12g}")
    print(f"input selection: {args.selection:.12g}")
    print(f"recovered selection: {recovered:.12g}")
    absolute_error = abs(recovered - args.selection)
    relative_error = (
        absolute_error / abs(args.selection)
        if args.selection != 0
        else absolute_error
    )
    print(f"absolute error: {absolute_error:.12g}")
    print(f"relative error: {relative_error:.2%}")
    passed = relative_error <= args.relative_tolerance
    print(
        f"round-trip: {'PASS' if passed else 'FAIL'} "
        f"(tolerance {args.relative_tolerance:.2%})"
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
