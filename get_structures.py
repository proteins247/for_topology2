#!/usr/bin/env python
"""
get_structures.py

This script is only provided as reference for how data.csv was
generated. Accompanying data is not present in this repository.

objectives of this script:
1. Provide PDB files
2. Provide csv file
   - Link each gene to structure file
   - Provide details on each gene
   - csv file should have as many rows as there are codons x genes

Note, PDB numbering is not guaranteed to be good
https://proteopedia.org/wiki/index.php/Unusual_sequence_numbering

"""
import sys
import argparse
import collections
import gzip
import glob
import pickle as pkl
import pandas as pd
import numpy as np
np.random.seed(23)
import scipy.signal
import scipy.stats
from IPython import start_ipython
import shutil

from Bio import pairwise2
import Bio.PDB
import Bio.PDB.Polypeptide
PDB_PARSER = Bio.PDB.PDBParser()
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning
warnings.simplefilter('ignore', PDBConstructionWarning)

# Will's gene enrichment data; 511 entries
ENRICHMENT = "../jacobs2017evidence/gene_enrichment+free_energy_v2-full_free_energy.pkl"

# My database of genes and corresponding structures
#   Has all E. coli genes that have structures.
STRUCTURES = "../2020-02__slow_codons_and_structure2/3_genes_and_downloaded_structures_best.pkl"

STRUCTURE_DIR = "../2020-02__slow_codons_and_structure2/structures/"  # already downloaded
NEW_STRUCTURE_DIR = "structures/"  # will copy to this destination


def expand_df(df, column):
    """Take a column of lists; give each value a row in the dataframe.

    """
    expanded2 = pd.DataFrame({
        col: np.repeat(df[col].values, df[column].str.len())
        for col in df.columns.drop(column)}
    ).assign(**{column: list(np.concatenate(df[column].values))})
    return expanded2


# def align_residues_to_sequence(residues, aaseq):
#     """Return list of (resid, residue) where resid is the position of
#     residue in aaseq.

#     # Offset: add offset to actual PDB resid to get sequence resid
#     """
#     peptide = Bio.PDB.Polypeptide.Polypeptide(residues)
#     peptide_seq = str(peptide.get_sequence())
#     # returns a list of alignments, take top
#     alignment = pairwise2.align.globalms(aaseq, peptide_seq, 2, -1, -1, -0.1,
#                                          penalize_end_gaps=False)[0]
#     # (sorry, i reversed aaseq and peptide_seq arguments)

#     # Would be good to check that alignment is good
#     score = alignment[2]
#     if score / len(aaseq) < 0.85:
#         print("WARNING: score / len(aaseq) < 0.85")

#     # the premise is that there are no gaps for aaseq, and that
#     # aaseq[0] is 1.
#     aligned = []                # to be returned
#     aaseq_aligned = alignment[0]
#     peptide_seq_aligned = alignment[1]
#     aaseq_index = 1
#     peptide_seq_index = 0
#     for letter_a, letter_p in zip(aaseq_aligned, peptide_seq_aligned):
#         if letter_a == letter_p:
#             aligned.append((aaseq_index, residues[peptide_seq_index]))
#             aaseq_index += 1
#             peptide_seq_index += 1
#         elif letter_a == '-':
#             # print("Insertion?", residues[0].full_id)
#             peptide_seq_index += 1
#         elif letter_p == '-':
#             aaseq_index += 1
#         elif letter_a != letter_p:
#             # Due to mutation?
#             aaseq_index += 1
#             peptide_seq_index += 1
#         else:
#             raise Exception
#     # offset = aligned[0][0] - aligned[0][1].id[1]
#     # offset2 = aligned[-1][0] - aligned[-1][1].id[1]
#     # if offset != offset2:
#     #     print("Mismatched offsets", aligned[0][1].full_id)
#     return aligned


# def find_enriched_regions(name, enrichment, enrichment_threshold,
#                           pvalue_threshold, excluded_length=80):
#     """
#     For now, just find the first peak
#     """
#     data = enrichment.loc[name].query("center > %d" % excluded_length)
#     peaks = scipy.signal.find_peaks(data["enriched"], enrichment_threshold)[0]
#     # Note that peaks are >= enrichment_threshold
#     qualified_peaks = []
#     for peak in peaks:
#         region = slice(peak - 5, peak + 6)
#         pvalues = data["p_value"].values[region]
#         if np.any(pvalues < pvalue_threshold):
#             qualified_peaks.append(peak)
#     return data["center"].values[qualified_peaks]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ipython', action='store_true',
                        help="start ipython at the end of the script")
    # parser.add_argument("--enrichment-threshold", type=float, default=0.75)
    # parser.add_argument("--pvalue-threshold", type=float, default=0.01)
    # parser.add_argument("--unextruded-length", type=int, default=30)
    # parser.add_argument("--nterm-exclusion", type=int, default=80)
    # parser.add_argument("--random-count", type=int, default=5)
    # parser.add_argument("--co-exclusion", type=int, default=1)
    return parser.parse_args()


def main(args):
    # Get genes and file paths
    names = pd.read_pickle(STRUCTURES)

    # We don't need the SCOP data for domain annotation
    names = names.groupby(level='name').head(1)
    names = names.drop(columns=['domain', 'pdbid', 'chain'])

    # Will's enrichment data
    enrichment = pd.read_pickle(ENRICHMENT)
    # This dataframe is gene name, codon position, enrichment,
    #   pvalue, free energy, free energy constrained
    # Now, Will used different names sometimes
    # in particular, we want to change
    #   'bioD' -> 'bioD1'
    #   'umpH' -> 'nagD'
    enrichment = enrichment.set_index("name", drop=False)
    enrichment.loc['bioD', "name"] = 'bioD1'
    enrichment.loc['umpH', "name"] = 'nagD'
    # weirdly, renaming here also alters the index?
    enrichment = enrichment.reset_index(drop=True)

    # Sometimes Will used different gene names. We have a
    #  synonyms column
    names['all_names'] = [[name] + synonyms for name, synonyms in
                          zip(names['name'], names['synonyms'])]
    expanded = expand_df(names, 'all_names')
    expanded = expanded.set_index(["all_names", "name"], drop=False)

    # Now for some reason, there are duplicates due to synonyms
    # I'm going to take care of the 2 that conflict
    # fabG, mgsA
    expanded = expanded.drop([('fabG', 'accC'), ('mgsA', 'rarA')])

    # Keep only those with enrichment data
    will_gene_names = [s.lower() for s in enrichment['name'].unique()]
    has_enrichment = [name.lower() in will_gene_names
                      for name in expanded.index.get_level_values('all_names')]
    expanded = expanded[has_enrichment].copy()
    # At 511 genes, exactly the number will has

    """
    At this point, we have two dataframes. `expanded` and `enrichment`
    `expanded` has 511 rows and 17 columns.
    `enrichment` has 138222 rows and 6 columns
    The index, 'all_names', in `expanded` matches up with the gene names in `enrichment`

    The goal now is to blow up expanded so it has 138222 rows, each
    row being an amino acid or codon

    """

    expanded_aaseq = list(''.join(expanded['aaseq'].values))
    seq_minus_stop = expanded['seq'].apply(lambda x: x[:-3])
    seq_joined = ''.join(seq_minus_stop)
    seq_in_codons = [seq_joined[i: i + 3] for i in range(0, len(seq_joined), 3)]
    resnum = np.concatenate([np.arange(1, l + 1) for l in expanded['aalen']])
    expanded2 = pd.DataFrame({
        col: np.repeat(expanded[col].values, expanded['aalen'])
        for col in expanded.columns.drop(['aaseq', 'seq'])})
    expanded2 = expanded2.assign(
        aaseq=expanded_aaseq, codon=seq_in_codons, resnum=resnum)

    merged = pd.merge(
        enrichment, expanded2, how='left', left_on=['name', 'position'],
        right_on=['all_names', 'resnum'])

    to_keep = merged['position'] != 0
    merged = merged[to_keep]
    merged = merged.drop(columns=['name_y', 'all_names', 'resnum'])
    merged = merged.rename(columns={'name_x': 'name'})
    for group in merged.groupby('name'):
        filename = group[1]['filename'].iloc[0]
        shutil.copy(STRUCTURE_DIR + filename, NEW_STRUCTURE_DIR)

    merged.to_csv("data.csv")

    if args.ipython:
        sys.argv = [sys.argv[0]]
        start_ipython(user_ns=dict(locals(), **globals()))
    return 0


if __name__ == "__main__":
    sys.exit(main(parse_args()))
