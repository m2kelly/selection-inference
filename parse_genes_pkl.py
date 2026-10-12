#!/usr/bin/env python3
"""Classify the three possible alternate alleles at every coding site.

The input is the ``genes.pkl`` written by CBaSE-is. Each valid row contains a
strand-normalized pentamer; its middle three bases are the reference codon.
For each codon position this script classifies all alternate bases and counts
each distinct nonsynonymous pattern by gene and reference trinucleotide.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, TextIO

import pandas as pd


BASES = ("A", "C", "G", "T")
SUMMARY_COLUMNS = (
    "gene", "trinucleotide", "reference_base", "n_nonsyn",
    "nonsyn_alt_bases", "syn_alt_bases", "count",
)
SITE_COLUMNS = (
    "gene", "codon_ref_index", "codon", "pentamer", "codon_base_index",
    "trinucleotide", "reference_base", "A_effect", "C_effect", "G_effect",
    "T_effect",
)


def load_codon_table(path: Path) -> dict[str, str]:
    with path.open() as handle:
        table = json.load(handle)
    expected = {a + b + c for a in BASES for b in BASES for c in BASES}
    if set(table) != expected:
        raise ValueError(f"{path} is not a complete 64-codon table")
    return table


def mutation_type(
    ref_codon: str, alt_codon: str, codon_table: Mapping[str, str]
) -> str:
    """Match ``CBaSE.parameters.analysis.seq.Seq.get_mutation_type``."""
    if codon_table[alt_codon] == "Stop" or codon_table[ref_codon] == "Stop":
        return "nonsense"
    if codon_table[ref_codon] == codon_table[alt_codon]:
        return "coding-synon"
    return "missense"


def classify_pentamer(
    pentamer: str,
    codon_table: Mapping[str, str],
    codon: str | None = None,
) -> list[dict[str, object]]:
    """Return allele effects for the three coding sites in one pentamer.

    Effects are ``current``, ``syn``, or ``nonsyn``. Both CBaSE missense and
    nonsense classes are collapsed to ``nonsyn`` for the four-allele model.
    """
    pentamer = pentamer.upper()
    if len(pentamer) != 5 or any(base not in BASES for base in pentamer):
        raise ValueError(f"invalid CBaSE pentamer: {pentamer!r}")

    ref_codon = pentamer[1:4]
    if codon is not None and codon.upper() != ref_codon:
        raise ValueError(
            f"codon {codon!r} does not match middle bases of {pentamer!r}"
        )

    sites = []
    for codon_index, ref_base in enumerate(ref_codon):
        effects = {}
        mutation_types = {}
        for alt_base in BASES:
            if alt_base == ref_base:
                effects[alt_base] = "current"
                continue
            alt_codon = (
                ref_codon[:codon_index]
                + alt_base
                + ref_codon[codon_index + 1 :]
            )
            kind = mutation_type(ref_codon, alt_codon, codon_table)
            mutation_types[alt_base] = kind
            effects[alt_base] = "syn" if kind == "coding-synon" else "nonsyn"

        nonsyn_alts = tuple(base for base in BASES if effects[base] == "nonsyn")
        syn_alts = tuple(base for base in BASES if effects[base] == "syn")
        sites.append({
            "codon_base_index": codon_index,
            "trinucleotide": pentamer[codon_index : codon_index + 3],
            "reference_base": ref_base,
            "effects": effects,
            "mutation_types": mutation_types,
            "nonsyn_alt_bases": nonsyn_alts,
            "syn_alt_bases": syn_alts,
            "n_nonsyn": len(nonsyn_alts),
        })
    return sites


def _open_site_output(path: Path) -> tuple[TextIO, csv.DictWriter]:
    handle = (
        gzip.open(path, "wt", newline="")
        if path.suffix == ".gz"
        else path.open("w", newline="")
    )
    writer = csv.DictWriter(handle, fieldnames=SITE_COLUMNS, delimiter="\t")
    writer.writeheader()
    return handle, writer


def summarize_genes(
    frames: Iterable[pd.DataFrame],
    codon_table: Mapping[str, str],
    site_output: Path | None = None,
    genes: set[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Summarize sites having at least one nonsynonymous alternate allele.

    The optional detailed TSV is streamed so the full human pickle does not
    require retaining tens of millions of site rows in memory.
    """
    counts: Counter[tuple[object, ...]] = Counter()
    stats = Counter(frames=0, input_rows=0, valid_codons=0, skipped_rows=0)
    classifications = {
        a + b + c + d + e: classify_pentamer(
            a + b + c + d + e, codon_table
        )
        for a in BASES
        for b in BASES
        for c in BASES
        for d in BASES
        for e in BASES
    }
    site_handle = None
    site_writer = None
    if site_output is not None:
        site_output.parent.mkdir(parents=True, exist_ok=True)
        site_handle, site_writer = _open_site_output(site_output)

    try:
        for frame in frames:
            stats["frames"] += 1
            required = {"gene_name", "base_ref_indices", "contexts"}
            missing = required - set(frame.columns)
            if missing:
                raise ValueError(f"genes.pkl frame is missing columns: {sorted(missing)}")

            for row in frame.itertuples(index=False):
                stats["input_rows"] += 1
                gene = getattr(row, "gene_name")
                context = getattr(row, "contexts")
                if not isinstance(gene, str) or not isinstance(context, str):
                    stats["skipped_rows"] += 1
                    continue
                if genes is not None and gene not in genes:
                    continue

                pentamer = context.upper()
                if len(pentamer) != 5 or any(base not in BASES for base in pentamer):
                    stats["skipped_rows"] += 1
                    continue
                stats["valid_codons"] += 1

                codon = pentamer[1:4]
                for site in classifications[pentamer]:
                    if site["n_nonsyn"] == 0:
                        stats["all_syn_sites_omitted"] += 1
                        continue

                    nonsyn = "".join(site["nonsyn_alt_bases"])
                    syn = "".join(site["syn_alt_bases"])
                    key = (
                        gene, site["trinucleotide"], site["reference_base"],
                        site["n_nonsyn"], nonsyn, syn,
                    )
                    counts[key] += 1
                    stats["recorded_sites"] += 1

                    if site_writer is not None:
                        effects = site["effects"]
                        site_writer.writerow({
                            "gene": gene,
                            "codon_ref_index": getattr(row, "base_ref_indices"),
                            "codon": codon,
                            "pentamer": pentamer,
                            "codon_base_index": site["codon_base_index"],
                            "trinucleotide": site["trinucleotide"],
                            "reference_base": site["reference_base"],
                            **{f"{base}_effect": effects[base] for base in BASES},
                        })
    finally:
        if site_handle is not None:
            site_handle.close()

    rows = [(*key, count) for key, count in counts.items()]
    summary = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    if not summary.empty:
        summary.sort_values(
            ["gene", "trinucleotide", "n_nonsyn", "nonsyn_alt_bases"],
            inplace=True,
            ignore_index=True,
        )
    return summary, dict(stats)


def write_summary(summary: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    compression = "gzip" if path.suffix == ".gz" else None
    summary.to_csv(path, sep="\t", index=False, compression=compression)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("genes_pickle", type=Path, help="CBaSE-is genes.pkl")
    parser.add_argument(
        "--codon-table", type=Path,
        default=Path(__file__).parent / "Auxiliary" / "codon_table.json",
    )
    parser.add_argument(
        "--summary", type=Path, default=Path("four_allele_site_summary.tsv.gz"),
        help="per-gene summary TSV (gzip when the name ends in .gz)",
    )
    parser.add_argument(
        "--sites", type=Path,
        help="optional detailed site TSV (gzip when the name ends in .gz)",
    )
    parser.add_argument(
        "--gene", action="append", dest="genes",
        help="only process this gene; may be supplied more than once",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    codon_table = load_codon_table(args.codon_table)
    with args.genes_pickle.open("rb") as handle:
        frames = pickle.load(handle)
    summary, stats = summarize_genes(
        frames,
        codon_table,
        site_output=args.sites,
        genes=set(args.genes) if args.genes else None,
    )
    write_summary(summary, args.summary)
    print(json.dumps({**stats, "summary_rows": len(summary)}, sort_keys=True))


if __name__ == "__main__":
    main()
