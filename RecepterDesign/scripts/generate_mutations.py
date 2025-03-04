#!/usr/bin/env python3
# scripts/generate_mutations.py - Generate receptor mutations for screening

import os
import sys
import argparse
import random
import pandas as pd
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MutationGenerator")

# Amino acid properties
AA_PROPERTIES = {
    'hydrophobic': ['A', 'I', 'L', 'M', 'F', 'V', 'P', 'G'],
    'polar': ['S', 'T', 'Y', 'C', 'N', 'Q'],
    'charged': ['D', 'E', 'K', 'R', 'H'],
    'aromatic': ['F', 'Y', 'W'],
    'small': ['A', 'G', 'S']
}

# Amino acid groups for conservative mutations
AA_GROUPS = {
    'A': ['G', 'S', 'V'],
    'R': ['K', 'H', 'Q'],
    'N': ['Q', 'D', 'E', 'H'],
    'D': ['E', 'N', 'Q'],
    'C': ['S'],
    'E': ['D', 'Q', 'K'],
    'Q': ['N', 'E', 'K', 'R'],
    'G': ['A', 'S'],
    'H': ['R', 'K', 'N', 'Q'],
    'I': ['L', 'V', 'M'],
    'L': ['I', 'V', 'M', 'F'],
    'K': ['R', 'Q', 'E'],
    'M': ['L', 'I', 'V'],
    'F': ['Y', 'W', 'L'],
    'P': ['G', 'A'],
    'S': ['T', 'A', 'G'],
    'T': ['S'],
    'W': ['F', 'Y'],
    'Y': ['F', 'W'],
    'V': ['I', 'L', 'M', 'A']
}


def read_fasta(fasta_path):
    """Read protein sequence from FASTA file."""
    sequence = ""
    with open(fasta_path, 'r') as f:
        for line in f:
            if line.startswith('>'):
                continue
            sequence += line.strip()
    return sequence


def identify_functional_regions(sequence):
    """
    Identify functional regions in protein sequence.
    This is a simplified simulation - in reality would use domain prediction tools.
    """
    seq_length = len(sequence)

    # Simplified: Just identify random regions as functional
    # In a real implementation, would use domain databases and prediction tools

    # Simulate transmembrane domains (for receptors)
    tm_domains = []
    for i in range(3, 7):  # Typical number of TM domains in GPCRs
        start = random.randint(0, seq_length - 25)
        end = start + random.randint(20, 25)  # TM helices ~20-25 amino acids
        tm_domains.append((start, end, 'transmembrane'))

    # Simulate binding pocket residues
    binding_residues = []
    for _ in range(10):
        pos = random.randint(0, seq_length - 1)
        binding_residues.append((pos, pos + 1, 'binding_pocket'))

    # Combine all regions
    functional_regions = tm_domains + binding_residues

    # Sort by start position
    functional_regions.sort(key=lambda x: x[0])

    return functional_regions


def get_mutation_type(aa):
    """Determine the type of amino acid for targeted mutations."""
    for prop, aas in AA_PROPERTIES.items():
        if aa in aas:
            return prop
    return "unknown"


def generate_conservative_mutation(aa):
    """Generate a conservative mutation for given amino acid."""
    if aa in AA_GROUPS:
        return random.choice(AA_GROUPS[aa])
    # Fallback
    return random.choice([a for a in "ACDEFGHIKLMNPQRSTVWY" if a != aa])


def generate_nonconservative_mutation(aa):
    """Generate a non-conservative mutation for given amino acid."""
    aa_type = get_mutation_type(aa)

    # Get amino acids from different property groups
    other_types = [t for t in AA_PROPERTIES.keys() if t != aa_type]
    if not other_types:
        return random.choice([a for a in "ACDEFGHIKLMNPQRSTVWY" if a != aa])

    selected_type = random.choice(other_types)
    candidates = [a for a in AA_PROPERTIES[selected_type] if a != aa]

    if candidates:
        return random.choice(candidates)
    else:
        return random.choice([a for a in "ACDEFGHIKLMNPQRSTVWY" if a != aa])


def generate_mutations(sequence, max_mutations, functional_regions=None,
                       conservative_ratio=0.7, critical_mutation_ratio=0.3):
    """
    Generate a list of mutations for the sequence.

    Args:
        sequence: Protein sequence
        max_mutations: Maximum number of mutations to generate
        functional_regions: List of functional regions (start, end, type)
        conservative_ratio: Ratio of conservative mutations
        critical_mutation_ratio: Ratio of mutations in critical regions

    Returns:
        list: List of mutation strings (e.g., 'A123B')
    """
    seq_length = len(sequence)
    mutations = []

    # Identify critical regions if not provided
    if functional_regions is None:
        functional_regions = identify_functional_regions(sequence)

    # Create a mapping of positions to region types
    position_to_region = {}
    for start, end, region_type in functional_regions:
        for pos in range(start, min(end, seq_length)):
            position_to_region[pos] = region_type

    # Determine how many mutations in critical vs non-critical regions
    critical_count = int(max_mutations * critical_mutation_ratio)
    non_critical_count = max_mutations - critical_count

    # Generate mutations in critical regions
    critical_positions = list(position_to_region.keys())
    if critical_positions and critical_count > 0:
        for _ in range(min(critical_count, len(critical_positions))):
            pos = random.choice(critical_positions)
            critical_positions.remove(pos)  # Avoid duplicate positions

            # Get original amino acid
            original_aa = sequence[pos]

            # Decide if conservative or non-conservative
            if random.random() < conservative_ratio:
                mutant_aa = generate_conservative_mutation(original_aa)
            else:
                mutant_aa = generate_nonconservative_mutation(original_aa)

            # Create mutation string (1-indexed for conventional notation)
            mutation = f"{original_aa}{pos + 1}{mutant_aa}"
            mutations.append((mutation, position_to_region.get(pos, "unknown")))

    # Generate mutations in non-critical regions
    non_critical_positions = [i for i in range(seq_length) if i not in position_to_region]
    if non_critical_positions and non_critical_count > 0:
        for _ in range(min(non_critical_count, len(non_critical_positions))):
            pos = random.choice(non_critical_positions)
            non_critical_positions.remove(pos)  # Avoid duplicate positions

            # Get original amino acid
            original_aa = sequence[pos]

            # Decide if conservative or non-conservative
            if random.random() < conservative_ratio:
                mutant_aa = generate_conservative_mutation(original_aa)
            else:
                mutant_aa = generate_nonconservative_mutation(original_aa)

            # Create mutation string (1-indexed for conventional notation)
            mutation = f"{original_aa}{pos + 1}{mutant_aa}"
            mutations.append((mutation, "non_critical"))

    return mutations


def main():
    parser = argparse.ArgumentParser(description='Generate receptor mutations for screening')
    parser.add_argument('--structure', required=True, help='Path to structure PDB file')
    parser.add_argument('--sequence', required=True, help='Path to sequence FASTA file')
    parser.add_argument('--output', required=True, help='Path to output mutations CSV')
    parser.add_argument('--max_mutations', type=int, default=50, help='Maximum number of mutations to generate')
    args = parser.parse_args()

    # Read protein sequence
    try:
        sequence = read_fasta(args.sequence)
        logger.info(f"Read sequence of length {len(sequence)} from {args.sequence}")
    except Exception as e:
        logger.error(f"Failed to read sequence: {e}")
        return 1

    # Identify functional regions
    # For a full implementation, would use structure analysis here
    functional_regions = identify_functional_regions(sequence)
    logger.info(f"Identified {len(functional_regions)} functional regions")

    # Generate mutations
    mutations = generate_mutations(
        sequence,
        args.max_mutations,
        functional_regions=functional_regions
    )
    logger.info(f"Generated {len(mutations)} mutations")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Save mutations to CSV
    df = pd.DataFrame(mutations, columns=['mutation', 'region_type'])
    df['wildtype_residue'] = df['mutation'].apply(lambda x: x[0])
    df['position'] = df['mutation'].apply(lambda x: int(''.join(c for c in x[1:-1] if c.isdigit())))
    df['mutant_residue'] = df['mutation'].apply(lambda x: x[-1])

    df.to_csv(args.output, index=False)
    logger.info(f"Saved mutations to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())