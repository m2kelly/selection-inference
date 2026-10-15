import os
import pickle
import numpy as np
import pandas as pd
import cupy as cp
from predictor import Predictor
from graphpoint import GraphPointAbstract as GPA
#from NeParallel.pop_size.graphpoint import GraphPointAbstract as GPA

#python -m test_pipeline.scripts.1_simulate_wg.py
'''
idea is to simulate a full genome, by simulating different mutation rates per bin
then can run full N inference on output (so in pickle dicitonary of dfs format)

load mutation transition matrix from eg human-chimp
then apply N transformation to transition matrix
NOTE: apply transformation per trinucleotide category rather than per mut channel,
as pop genetics N correction only makes sense on this scale
then redivide into mut categories using the orginal ratio between the alts per category




'''



# -----------------------------
# User parameters
# -----------------------------
N_e = int(2e5)
generations = 160_000
bin_size = 100_000
n_bins = 20_000
n_chrs = 22

#mus = np.linspace(0.01, 2.0, num=100)
#mus = np.geomspace(0.01, 2.0, num=100)
u = np.linspace(0, 1, 100)
min=0.001
max=2.5

#min=0.1
#max=3

mus = min + (max - min) * u**5

mus=100*[1]
print(mus)
dir='/home/dweghorngroup/selection_per_gene_4_allele_/test_pipeline/mu_sub_check'
mut_matrix_file = "/home/dweghorngroup/selection_per_gene_final_step_intron_all_occ_fixed/files/best_in_df_mutations_4_allele.csv"
name="forward_sim_no_mu"


out_dir = f"{dir}/{name}"
os.makedirs(out_dir, exist_ok=True)
muts_out = os.path.join(out_dir, f"whole_genome_dict_{name}_unsmoothed_muts.pkl")
occs_out = os.path.join(out_dir, f"whole_genome_dict_{name}_unsmoothed_occs.pkl")
metadata_out = os.path.join(out_dir, "bin_metadata.tsv")
subrates_out=os.path.join(out_dir, "mut_sub_rates.tsv")
seed = 123
rng = np.random.default_rng(seed)


# -----------------------------
# Definitions
# -----------------------------
BASES = ["A", "C", "G", "T"]
CONTEXTS = [a + b + c for a in BASES for b in BASES for c in BASES]
MUTS = [
    tri + "->" + alt
    for tri in CONTEXTS
    for alt in BASES
    if alt != tri[1]
]

mut_to_tri = np.array([m.split("->")[0] for m in MUTS])
mut_to_alt = np.array([m.split("->")[1] for m in MUTS])

ctx_to_idx = {tri: i for i, tri in enumerate(CONTEXTS)}
mut_source_idx = np.array([ctx_to_idx[tri] for tri in mut_to_tri])

predictor = Predictor()


# -----------------------------
# Build base transition matrix
# -----------------------------
def load_base_transition_matrix(mut_matrix_file):
    P_df = pd.read_csv(mut_matrix_file, sep="\t", index_col=0)
    print(P_df.loc['CAT']['CCT'])
    print(P_df.loc['CCT']['CAT'])
    P_df = P_df.astype(float)

    # Force expected order
    P_df = P_df.loc[CONTEXTS, CONTEXTS]

    P = P_df.to_numpy(copy=True)

    # Off-diagonal entries are mutation-rate/flux entries
    np.fill_diagonal(P, 0.0)

    row_sums = P.sum(axis=1)
    if np.any(row_sums > 1.0):
        bad = np.where(row_sums > 1.0)[0]
        raise ValueError(
            f"Base off-diagonal row sums exceed 1. "
            f"Example bad contexts: {[CONTEXTS[i] for i in bad[:10]]}. "
            f"Max row sum={row_sums.max()}"
        )

    np.fill_diagonal(P, 1.0 - row_sums)

    if not np.allclose(P.sum(axis=1), 1.0):
        raise ValueError("Rows of base P do not sum to 1.")

    return P


# -----------------------------
# Population-genetic correction
# -----------------------------
def make_graphpoint(mu_fwrd, mu_bwrd, N, selection=0.0):
    """
    Handles either GPA(mu_fwrd, mu_bwrd, pop_size=N, s=...)
    or GPA(mu_fwrd, mu_bwrd, N, selection).
    """
    try:
        return GPA(mu_fwrd, mu_bwrd, pop_size=N, s=selection)
    except TypeError:
        try:
            return GPA(mu_fwrd, mu_bwrd, pop_size=N)
        except TypeError:
            return GPA(mu_fwrd, mu_bwrd, N, selection)

def build_substitution_corrected_P_by_trinuc(
    P_base,
    mu_scale,
    N,
    predictor,
    reverse_mode="weighted",
):
    """
    Population-genetic correction at the trinuc/site level.

    For each source trinuc i:

        mu_fwrd_total = sum_j P_base[i, j] * mu_scale

    Then apply population-genetic correction once to the total any-alt rate:

        sub_total_i = predictor.get_sub_rate(
            GPA(mu_fwrd_total, mu_bwrd_effective, N, 0)
        )

    Then redistribute the corrected total rate across outgoing channels using
    the original mutation spectrum:

        P_corr[i, j] = sub_total_i * P_base[i, j] / sum_j P_base[i, j]

    This avoids applying population-genetic correction separately to each
    mutation channel.
    """

    P_off_base = P_base.copy().astype(float)
    np.fill_diagonal(P_off_base, 0.0)

    n = P_off_base.shape[0]
    P_off = np.zeros_like(P_off_base, dtype=float)

    for i in range(n):
        row = P_off_base[i].copy()
        row_sum = row.sum()

        if row_sum <= 0:
            continue

        # Original relative spectrum from this trinuc to each alt
        channel_weights = row / row_sum

        # Total forward mutation rate for this physical trinuc/site class
        mu_fwrd_total = row_sum * mu_scale

        if reverse_mode == "weighted":
            # Effective backward rate from the mixture of alt states back to i.
            # If i mutates mostly to j, then j->i matters more.
            mu_bwrd_effective = np.sum(
                channel_weights * P_off_base[:, i]
            ) * mu_scale

        elif reverse_mode == "incoming_sum":
            # More aggressive: total incoming mutation rate into i
            mu_bwrd_effective = P_off_base[:, i].sum() * mu_scale

        elif reverse_mode == "same":
            # Symmetric simplification
            mu_bwrd_effective = mu_fwrd_total

        else:
            raise ValueError(
                "reverse_mode must be 'weighted', 'incoming_sum', or 'same'"
            )

        point = make_graphpoint(
            mu_fwrd_total,
            mu_bwrd_effective,
            N,
            selection=0.0,
        )

        sub_total = predictor.get_sub_rate(point)

        if not np.isfinite(sub_total):
            raise ValueError(
                f"Non-finite substitution rate for row {i}: "
                f"mu_fwrd_total={mu_fwrd_total}, "
                f"mu_bwrd_effective={mu_bwrd_effective}, "
                f"sub_total={sub_total}"
            )

        if sub_total < 0:
            raise ValueError(
                f"Negative substitution rate for row {i}: {sub_total}"
            )

        # Redistribute corrected total substitution rate across alt channels
        P_off[i] = sub_total * channel_weights

    np.fill_diagonal(P_off, 0.0)

    row_sums = P_off.sum(axis=1)

    if np.any(row_sums > 1.0):
        bad = np.where(row_sums > 1.0)[0]
        raise ValueError(
            f"Corrected off-diagonal row sums exceed 1 for mu={mu_scale}. "
            f"Example bad rows: {bad[:10]}, max row sum={row_sums.max()}"
        )

    P_corr = P_off.copy()
    np.fill_diagonal(P_corr, 1.0 - row_sums)

    # numerical cleanup
    P_corr[P_corr < 0] = 0.0
    P_corr = P_corr / P_corr.sum(axis=1, keepdims=True)

    return P_corr

def build_substitution_corrected_P_2_allele(P_base, mu_scale, N, predictor):
    """
    Scale off-diagonal mutation rates by mu_scale, then convert each
    mutation rate to a substitution rate using predictor.get_sub_rate.

    For each i -> j:
        mu_fwrd = P_base[i, j] * mu_scale
        mu_bwrd = P_base[j, i] * mu_scale
        P_corr[i, j] = predictor.get_sub_rate(GPA(mu_fwrd, mu_bwrd, N, 0))
    """
    P_off_base = P_base.copy()
    np.fill_diagonal(P_off_base, 0.0)

    n = P_off_base.shape[0]
    P_off = np.zeros_like(P_off_base, dtype=float)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            mu_fwrd = P_off_base[i, j] * mu_scale
            mu_bwrd = P_off_base[j, i] * mu_scale

            if mu_fwrd <= 0:
                continue

            point = make_graphpoint(mu_fwrd, mu_bwrd, N, selection=0.0)
            P_off[i, j] = predictor.get_sub_rate(point)

    np.fill_diagonal(P_off, 0.0)

    row_sums = P_off.sum(axis=1)
    if np.any(row_sums > 1.0):
        bad = np.where(row_sums > 1.0)[0]
        raise ValueError(
            f"Corrected off-diagonal row sums exceed 1 for mu={mu_scale}. "
            f"Example bad contexts: {[CONTEXTS[i] for i in bad[:10]]}. "
            f"Max row sum={row_sums.max()}"
        )

    P_corr = P_off.copy()
    np.fill_diagonal(P_corr, 1.0 - row_sums)

    # numerical cleanup
    P_corr[P_corr < 0] = 0.0
    P_corr = P_corr / P_corr.sum(axis=1, keepdims=True)

    return P_corr

def build_substitution_corrected_P(P_base, mu_scale, N, predictor):
    """
    Convert the 64x64 trinucleotide mutation matrix to substitution rates
    with the four-allele renewal model.

    Trinucleotides with the same left and right flanks form a 4x4 block,
    whose rows and columns are the central bases A/C/G/T.  Each of the 16
    blocks is passed to the predictor once, and the returned substitution
    rates are placed back into their original positions in a 64x64 matrix.
    """
    P_off_base = np.asarray(P_base, dtype=float).copy()
    if P_off_base.shape != (len(CONTEXTS), len(CONTEXTS)):
        raise ValueError(
            f"P_base must have shape {(len(CONTEXTS), len(CONTEXTS))}; "
            f"got {P_off_base.shape}"
        )
    np.fill_diagonal(P_off_base, 0.0)

    P_off = np.zeros_like(P_off_base, dtype=float)
    context_to_index = {context: i for i, context in enumerate(CONTEXTS)}
    point = make_graphpoint(0.0, 0.0, N, selection=0.0)

    # A four-allele block only describes central-base changes.  Catch an
    # incompatible input matrix instead of silently discarding flank changes.
    central_change_mask = np.zeros_like(P_off_base, dtype=bool)
    for left in BASES:
        for right in BASES:
            block_indices = [context_to_index[f"{left}{base}{right}"] for base in BASES]
            central_change_mask[np.ix_(block_indices, block_indices)] = True
    np.fill_diagonal(central_change_mask, False)
    unexpected = (P_off_base != 0.0) & ~central_change_mask
    if np.any(unexpected):
        i, j = np.argwhere(unexpected)[0]
        raise ValueError(
            "The four-allele block model only supports central-base changes, "
            f"but P_base contains {CONTEXTS[i]} -> {CONTEXTS[j]} = "
            f"{P_off_base[i, j]}"
        )

    for left in BASES:
        for right in BASES:
            mu_matrix = {
                ref: {
                    alt: (
                        0.0
                        if ref == alt
                        else P_off_base[
                            context_to_index[f"{left}{ref}{right}"],
                            context_to_index[f"{left}{alt}{right}"],
                        ] * mu_scale
                    )
                    for alt in BASES
                }
                for ref in BASES
            }

            sub_matrix = predictor.get_sub_rate_diffusion_4_allele_renewal(
                point,
                mu_matrix,test=True
            )

            for ref in BASES:
                source = context_to_index[f"{left}{ref}{right}"]
                for alt in BASES:
                    if ref == alt:
                        continue
                    target = context_to_index[f"{left}{alt}{right}"]
                    sub_rate = float(sub_matrix[ref][alt])
                    if not np.isfinite(sub_rate) or sub_rate < 0.0:
                        raise ValueError(
                            f"Invalid substitution rate for "
                            f"{left}{ref}{right}->{alt}: {sub_rate}"
                        )
                    P_off[source, target] = sub_rate

    np.fill_diagonal(P_off, 0.0)

    row_sums = P_off.sum(axis=1)
    if np.any(row_sums > 1.0):
        bad = np.where(row_sums > 1.0)[0]
        raise ValueError(
            f"Corrected off-diagonal row sums exceed 1 for mu={mu_scale}. "
            f"Example bad contexts: {[CONTEXTS[i] for i in bad[:10]]}. "
            f"Max row sum={row_sums.max()}"
        )

    P_corr = P_off.copy()
    np.fill_diagonal(P_corr, 1.0 - row_sums)

    # numerical cleanup
    P_corr[P_corr < 0] = 0.0
    P_corr = P_corr / P_corr.sum(axis=1, keepdims=True)

    return P_corr


# -----------------------------
# Random inital trinuc occurrences per bin
# -----------------------------
def random_trinuc_probs(rng):
    """
    Random but more realistic than uniform:
    - varying GC content per bin
    - CpG depletion
    """
    gc = np.clip(rng.normal(0.41, 0.06), 0.25, 0.65)

    base_p = {
        "A": (1.0 - gc) / 2.0,
        "T": (1.0 - gc) / 2.0,
        "C": gc / 2.0,
        "G": gc / 2.0,
    }

    probs = []
    cpg_depletion = np.clip(rng.lognormal(mean=np.log(0.25), sigma=0.25), 0.05, 0.8)

    for tri in CONTEXTS:
        p = base_p[tri[0]] * base_p[tri[1]] * base_p[tri[2]]

        # Deplete trinucs containing CpG dinucleotide
        if "CG" in tri:
            p *= cpg_depletion

        probs.append(p)

    probs = np.array(probs, dtype=float)
    probs = probs / probs.sum()
    return probs


def random_q0_for_bin(rng, bin_size):
    probs = random_trinuc_probs(rng)
    return rng.multinomial(bin_size, probs)


# -----------------------------
# Convert P^t to observed mutation counts
# -----------------------------
def sample_observed_mut_counts(q0, P_t, rng, stochastic=True):
    """
    Returns observed substitutions indexed by MUTS.

    If stochastic=True:
        for each source trinuc, sample final targets from multinomial(q0[source], P_t[source])
    If stochastic=False:
        use expected counts q0[source] * P_t[source, target]

    Output:
        np.array shape (len(MUTS),)
    """
    out = np.zeros(len(MUTS), dtype=float)

    if stochastic:
        final_counts = np.zeros_like(P_t, dtype=float)

        for i in range(len(CONTEXTS)):
            if q0[i] <= 0:
                continue

            row = P_t[i].copy()
            row[row < 0] = 0.0
            row = row / row.sum()

            final_counts[i] = rng.multinomial(int(q0[i]), row)

    else:
        final_counts = q0[:, None] * P_t

    for k, (tri, alt) in enumerate(zip(mut_to_tri, mut_to_alt)):
        source_i = ctx_to_idx[tri]
        target_tri = tri[0] + alt + tri[2]
        target_j = ctx_to_idx[target_tri]

        out[k] = final_counts[source_i, target_j]

    return out


def make_occ_counts_for_bin(q0):
    """
    Occurrence count for XXX->Y is the number of starting XXX trinucs.
    Same value repeated for the three possible alts.
    """
    return q0[mut_source_idx].astype(float)

def matrix_to_mutation_vector(matrix):
    """
    Extract central-base changes from a 64x64 matrix.

    Output order is exactly MUTS:
        AAA->C, AAA->G, AAA->T, AAC->A, ...
    """
    values = []

    for mutation in MUTS:
        source, alt = mutation.split("->")
        target = source[0] + alt + source[2]

        source_idx = ctx_to_idx[source]
        target_idx = ctx_to_idx[target]

        values.append(matrix[source_idx, target_idx])

    return np.asarray(values, dtype=float)

# -----------------------------
# Precompute P^generations for each mu
# -----------------------------
print("Loading base transition matrix")
P_base = load_base_transition_matrix(mut_matrix_file)
#print(P_base)

print("Precomputing P^generations for each mu")
P_t_cache = {}
P_cache={}

for mu_idx, mu_scale in enumerate(mus):
    print(f"  mu {mu_idx + 1}/{len(mus)}: {mu_scale:.4g}")

    #running on 4 alelle model per mut category
    P_corr = build_substitution_corrected_P(
        P_base=P_base,
        mu_scale=mu_scale,
        N=N_e,
        predictor=predictor)
    P_cache[mu_idx]=P_corr.copy()
    P_gpu = cp.asarray(P_corr, dtype=cp.float32)
    P_t = cp.linalg.matrix_power(P_gpu, int(generations))
    P_t = cp.asnumpy(P_t).astype(float)

    # numerical cleanup
    P_t[P_t < 0] = 0.0
    P_t = P_t / P_t.sum(axis=1, keepdims=True)

    P_t_cache[mu_idx] = P_t


#------
# Save the calculated mu idx, mu and sub rates
#-------
#convert subs and muts to vectors, order of entries is order of muts
rows = []

base_mutation_vector = matrix_to_mutation_vector(P_base)

for mu_idx, mu_scale in enumerate(mus):
    mutation_vector = base_mutation_vector * mu_scale
    substitution_vector = matrix_to_mutation_vector(P_cache[mu_idx])
    substitution_T_vector = matrix_to_mutation_vector(P_t_cache[mu_idx])

    for mutation, mutation_rate, substitution_rate, substitution_T_rate in zip(
        MUTS,
        mutation_vector,
        substitution_vector,
        substitution_T_vector
    ):
        rows.append(
            {
                "mu_index": mu_idx,
                "mu_scale": mu_scale,
                "mutation": mutation,
                "mutation_rate_per_generation": mutation_rate,
                "substitution_rate_per_generation": substitution_rate,
                "substitution_rate_after_generations": substitution_T_rate,
                "generations": generations,
                "N_e": N_e,
            }
        )

subrates_df = pd.DataFrame(rows)
subrates_df.to_csv(subrates_out, index=False)


# -----------------------------
# Assign bins to chromosomes and mu values
# -----------------------------
all_bins = np.arange(n_bins)
bins_by_chr = np.array_split(all_bins, n_chrs)

# Use the 100 mu values repeatedly across bins, shuffled
bin_mu_idx = np.arange(n_bins) % len(mus)
rng.shuffle(bin_mu_idx)




# -----------------------------
# Generate dictionaries
# -----------------------------
muts_dict = {}
occ_dict = {}
metadata_rows = []

print("Generating simulated bins")

for chr_i, bin_ids in enumerate(bins_by_chr, start=1):
    chr_name = f"chr{chr_i}"
    print(f"  {chr_name}: {len(bin_ids)} bins")

    muts_arr = np.zeros((len(MUTS), len(bin_ids)), dtype=np.float32)
    occ_arr = np.zeros((len(MUTS), len(bin_ids)), dtype=np.float32)

    for local_col, global_bin in enumerate(bin_ids):
        mu_idx = int(bin_mu_idx[global_bin])
        mu_scale = float(mus[mu_idx])

        q0 = random_q0_for_bin(rng, bin_size)

        observed = sample_observed_mut_counts(
            q0=q0,
            P_t=P_t_cache[mu_idx],
            rng=rng,
            stochastic=False,
        )

        occs = make_occ_counts_for_bin(q0)

        muts_arr[:, local_col] = observed
        occ_arr[:, local_col] = occs

        metadata_rows.append(
            {
                "chr": chr_name,
                "bin": int(global_bin),
                "mu_idx": mu_idx,
                "mu_scale": mu_scale,
                "N_e": N_e,
                "generations": generations,
                "bin_size": bin_size,
            }
        )

    muts_dict[chr_name] = pd.DataFrame(
        muts_arr,
        index=MUTS,
        columns=bin_ids,
    )

    occ_dict[chr_name] = pd.DataFrame(
        occ_arr,
        index=MUTS,
        columns=bin_ids,
    )


# -----------------------------
# Save outputs
# -----------------------------
print("Saving outputs")

with open(muts_out, "wb") as f:
    pickle.dump(muts_dict, f)

with open(occs_out, "wb") as f:
    pickle.dump(occ_dict, f)

metadata = pd.DataFrame(metadata_rows)
metadata.to_csv(metadata_out, sep="\t", index=False)

print(f"Saved mutation counts to: {muts_out}")
print(f"Saved occurrence counts to: {occs_out}")
print(f"Saved bin metadata to: {metadata_out}")
