#!/usr/bin/env python
"""
find_cotranslational_folders.py

Examine `data.csv` dataframe, identifying proteins with a high
likelihood to fold cotranslationally.

`data.csv` contains free energy and rare codon enrichment data along
the sequences of 511 /E. coli/ genes. The free energy values show the
free energy of the folded state relative to the unfolded ensemble for
a nascent protein of a given length. i.e. `F_L` is the minimum free
energy when considering nascent chain lengths of 1 through `L` in the
protein's native conformation. The rare codon enrichment data
indicates the enrichment of evolutionarily-conserved rare codons in a
15-codon region centered at a particular site (`enriched`) as well as
the likelihood of such a level of rare codon enrichment occurring by
chance (`p_value`).

The identification of cotranslational folders is based on the
hypothesis that rare codon regions that occur ~30 codons downstream of
significant (1 kT) drops in free energy represent meaningful
translational pauses or slow-downs that allow the nascent chain to
partially fold. The ~30 codon separation accounts for translated
residues still conformationally restricted within the ribosome's exit
tunnel. The actual separation used in Jacobs's paper is a range of
20-60 codons.

Thus, this script finds genes with such occurrences of rare codons and
free energy drops. The `--p-value` argument is a filter; only sites
with rare codon enrichment p-values below the specified threshold will
be considered.

author : Victor Zhao

"""
import sys
import argparse

import numpy as np
import pandas as pd

DATAFILE = "data.csv"


def find_enriched_regions(subdf, p_value_threshold,
                          enrichment_threshold=0.75, excluded_length=80):
    """Find locations where there are conserved rare codons.

    This is a function for `GroupBy.apply` that takes a dataframe with
    a `F_L` column.
    """
    locations = np.where((subdf['enriched'] > enrichment_threshold)
                         & (subdf['p_value'] < p_value_threshold))[0]
    # print(type(locations))
    # print(locations, subdf['position'])
    positions = subdf['position'].values[locations]
    p_values = subdf['p_value'].values[locations]
    p_values = p_values[positions > excluded_length]
    positions = positions[positions > excluded_length]
    return list(zip(positions, p_values))


def find_free_energy_drops(subdf):
    """Find locations where F_L drops more than 1 kT.

    Function finds drops between "plateaus." A plateau is when F_L has
    the same value for more than 1 length.

    This is a function for `GroupBy.apply` that takes a dataframe with
    a `F_L` column.
    """
    drops = []
    free_energy = subdf['F_L']
    difference = np.diff(free_energy)
    drop_total = 0
    for position, delta in zip(subdf['position'][1:], difference):
        if delta != 0:
            drop_total += delta
        else:
            if drop_total < -1:
                drops.append((position, drop_total))
                drop_total = 0
            else:
                drop_total = 0
    return drops


def has_cotranslational_signature(drops, rare_codons, separation=range(20, 61)):
    output = ""
    for drop in drops:
        drop_position = drop[0]
        qualifying_rare = [r for r in rare_codons
                           if (r[0] - drop_position) in separation]
        if qualifying_rare:
            output += "  drop, {}, with downstream rare codons {}\n".format(
                drop, qualifying_rare)
    return output


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--p-value', type=float, default=0.001,
                        help="Threshold for rare codon enrichment "
                        "p-value.")
    return parser.parse_args()


def main(args):
    data = pd.read_csv(DATAFILE)
    drops = data.groupby('name').apply(find_free_energy_drops)
    enriched = data.groupby('name').apply(
        find_enriched_regions, args.p_value)
    for (gene, drops, rare_codons) in zip(drops.index, drops, enriched):
        if drops and rare_codons:
            output = has_cotranslational_signature(drops, rare_codons)
            if output:
                print("Gene {}".format(gene))
                print(output)
    return


if __name__ == "__main__":
    sys.exit(main(parse_args()))
