import numpy as np
import re
from collections import defaultdict


class CodonOptimizer:
    """Codon optimization for improved protein expression."""

    def __init__(self, host_organism="human", gc_content_range=(40, 60)):
        """
        Initialize codon optimizer.

        Args:
            host_organism: Target host organism for expression
            gc_content_range: Target GC content range (min, max)
        """
        self.host_organism = host_organism
        self.gc_content_range = gc_content_range

        # Load codon usage tables (frequencies)
        self.codon_usage = self._load_codon_usage(host_organism)

        # Define genetic code
        self.genetic_code = self._define_genetic_code()

        # Calculate relative adaptiveness
        self.relative_adaptiveness = self._calculate_relative_adaptiveness()

        # Common restriction sites to avoid
        self.restriction_sites = {
            'EcoRI': 'GAATTC',
            'BamHI': 'GGATCC',
            'HindIII': 'AAGCTT',
            'XhoI': 'CTCGAG',
            'NotI': 'GCGGCCGC',
            'XbaI': 'TCTAGA',
            'SacI': 'GAGCTC',
            'SalI': 'GTCGAC',
            'PstI': 'CTGCAG',
            'SmaI': 'CCCGGG'
        }

    def _load_codon_usage(self, organism):
        """Load codon usage table for specified organism."""
        # In a real implementation, this would load from a database
        # Here we'll use a simplified version for human cells
        if organism.lower() == "human":
            return {
                'UUU': 0.46, 'UUC': 0.54, 'UUA': 0.08, 'UUG': 0.13,
                'CUU': 0.13, 'CUC': 0.20, 'CUA': 0.07, 'CUG': 0.40,
                'AUU': 0.36, 'AUC': 0.47, 'AUA': 0.17, 'AUG': 1.00,
                'GUU': 0.18, 'GUC': 0.24, 'GUA': 0.12, 'GUG': 0.46,
                'UCU': 0.19, 'UCC': 0.22, 'UCA': 0.15, 'UCG': 0.05,
                'CCU': 0.29, 'CCC': 0.32, 'CCA': 0.28, 'CCG': 0.11,
                'ACU': 0.25, 'ACC': 0.36, 'ACA': 0.28, 'ACG': 0.11,
                'GCU': 0.26, 'GCC': 0.40, 'GCA': 0.23, 'GCG': 0.11,
                'UAU': 0.44, 'UAC': 0.56, 'UAA': 0.30, 'UAG': 0.24,
                'CAU': 0.42, 'CAC': 0.58, 'CAA': 0.27, 'CAG': 0.73,
                'AAU': 0.47, 'AAC': 0.53, 'AAA': 0.43, 'AAG': 0.57,
                'GAU': 0.46, 'GAC': 0.54, 'GAA': 0.42, 'GAG': 0.58,
                'UGU': 0.46, 'UGC': 0.54, 'UGA': 0.47, 'UGG': 1.00,
                'CGU': 0.08, 'CGC': 0.18, 'CGA': 0.11, 'CGG': 0.20,
                'AGU': 0.15, 'AGC': 0.24, 'AGA': 0.21, 'AGG': 0.21,
                'GGU': 0.16, 'GGC': 0.34, 'GGA': 0.25, 'GGG': 0.25
            }
        else:
            # Default to E. coli if organism not recognized
            return {
                'UUU': 0.58, 'UUC': 0.42, 'UUA': 0.14, 'UUG': 0.13,
                'CUU': 0.12, 'CUC': 0.10, 'CUA': 0.04, 'CUG': 0.47,
                'AUU': 0.49, 'AUC': 0.39, 'AUA': 0.11, 'AUG': 1.00,
                'GUU': 0.28, 'GUC': 0.20, 'GUA': 0.17, 'GUG': 0.35,
                'UCU': 0.17, 'UCC': 0.15, 'UCA': 0.14, 'UCG': 0.14,
                'CCU': 0.18, 'CCC': 0.13, 'CCA': 0.20, 'CCG': 0.49,
                'ACU': 0.19, 'ACC': 0.40, 'ACA': 0.17, 'ACG': 0.25,
                'GCU': 0.18, 'GCC': 0.26, 'GCA': 0.23, 'GCG': 0.33,
                'UAU': 0.59, 'UAC': 0.41, 'UAA': 0.61, 'UAG': 0.09,
                'CAU': 0.57, 'CAC': 0.43, 'CAA': 0.34, 'CAG': 0.66,
                'AAU': 0.49, 'AAC': 0.51, 'AAA': 0.74, 'AAG': 0.26,
                'GAU': 0.63, 'GAC': 0.37, 'GAA': 0.68, 'GAG': 0.32,
                'UGU': 0.46, 'UGC': 0.54, 'UGA': 0.30, 'UGG': 1.00,
                'CGU': 0.36, 'CGC': 0.36, 'CGA': 0.07, 'CGG': 0.11,
                'AGU': 0.16, 'AGC': 0.25, 'AGA': 0.07, 'AGG': 0.04,
                'GGU': 0.35, 'GGC': 0.37, 'GGA': 0.13, 'GGG': 0.15
            }

    def _define_genetic_code(self):
        """Define the standard genetic code."""
        return {
            'UUU': 'F', 'UUC': 'F', 'UUA': 'L', 'UUG': 'L',
            'CUU': 'L', 'CUC': 'L', 'CUA': 'L', 'CUG': 'L',
            'AUU': 'I', 'AUC': 'I', 'AUA': 'I', 'AUG': 'M',
            'GUU': 'V', 'GUC': 'V', 'GUA': 'V', 'GUG': 'V',
            'UCU': 'S', 'UCC': 'S', 'UCA': 'S', 'UCG': 'S',
            'CCU': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
            'ACU': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
            'GCU': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
            'UAU': 'Y', 'UAC': 'Y', 'UAA': '*', 'UAG': '*',
            'CAU': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
            'AAU': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
            'GAU': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
            'UGU': 'C', 'UGC': 'C', 'UGA': '*', 'UGG': 'W',
            'CGU': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
            'AGU': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
            'GGU': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
        }

    def _calculate_relative_adaptiveness(self):
        """Calculate relative adaptiveness of each codon."""
        # Group codons by amino acid
        codons_by_aa = defaultdict(list)
        for codon, aa in self.genetic_code.items():
            codons_by_aa[aa].append(codon)

        # Calculate relative adaptiveness
        relative_adaptiveness = {}
        for aa, codons in codons_by_aa.items():
            # Skip stop codons
            if aa == '*':
                continue

            # Find max frequency for this amino acid
            max_freq = max(self.codon_usage[codon] for codon in codons)

            # Calculate relative adaptiveness
            for codon in codons:
                relative_adaptiveness[codon] = self.codon_usage[codon] / max_freq

        return relative_adaptiveness

    def calculate_cai(self, dna_sequence):
        """
        Calculate Codon Adaptation Index (CAI) for a DNA sequence.

        Args:
            dna_sequence: DNA sequence (string)

        Returns:
            float: CAI value (0-1, higher is better)
        """
        # Convert DNA to RNA
        rna_sequence = dna_sequence.upper().replace('T', 'U')

        # Extract codons
        codons = [rna_sequence[i:i + 3] for i in range(0, len(rna_sequence), 3)]

        # Remove incomplete final codon if present
        if len(codons[-1]) < 3:
            codons = codons[:-1]

        # Calculate CAI
        values = []
        for codon in codons:
            # Skip if not a valid codon or if it's a stop codon
            if codon not in self.genetic_code or self.genetic_code[codon] == '*':
                continue

            values.append(self.relative_adaptiveness[codon])

        # Geometric mean of values
        if not values:
            return 0

        return np.exp(np.mean(np.log(values)))

    def calculate_gc_content(self, dna_sequence):
        """
        Calculate GC content of a DNA sequence.

        Args:
            dna_sequence: DNA sequence (string)

        Returns:
            float: GC content percentage
        """
        sequence = dna_sequence.upper()
        gc_count = sequence.count('G') + sequence.count('C')
        return gc_count / len(sequence) * 100 if sequence else 0

    def check_restriction_sites(self, dna_sequence):
        """
        Check for restriction sites in a DNA sequence.

        Args:
            dna_sequence: DNA sequence (string)

        Returns:
            list: Found restriction sites
        """
        sequence = dna_sequence.upper()
        found_sites = []

        for enzyme, site in self.restriction_sites.items():
            if site in sequence:
                found_sites.append(enzyme)

        return found_sites

    def optimize_sequence(self, protein_sequence, avoid_sites=True, max_iterations=100):
        """
        Optimize DNA sequence for given protein sequence.

        Args:
            protein_sequence: Amino acid sequence (string)
            avoid_sites: Whether to avoid restriction sites
            max_iterations: Maximum optimization iterations

        Returns:
            tuple: (Optimized DNA sequence, CAI, GC content)
        """
        # Map amino acids to their possible codons
        aa_to_codons = defaultdict(list)
        for codon, aa in self.genetic_code.items():
            aa_to_codons[aa].append(codon)

        # Initial sequence using most frequent codons
        optimized_rna = ''
        for aa in protein_sequence.upper():
            # Skip invalid amino acids
            if aa not in aa_to_codons:
                continue

            # Get codons for this amino acid
            possible_codons = aa_to_codons[aa]

            # Select codon with highest frequency
            best_codon = max(possible_codons, key=lambda c: self.codon_usage[c])
            optimized_rna += best_codon

        # Convert to DNA
        optimized_dna = optimized_rna.replace('U', 'T')

        # Check properties
        cai = self.calculate_cai(optimized_dna)
        gc_content = self.calculate_gc_content(optimized_dna)
        restriction_sites = self.check_restriction_sites(optimized_dna) if avoid_sites else []

        # Iterative optimization if needed
        iteration = 0
        while iteration < max_iterations:
            # Stop if all criteria are met
            if (self.gc_content_range[0] <= gc_content <= self.gc_content_range[1] and
                    cai >= 0.8 and
                    (not avoid_sites or not restriction_sites)):
                break

            # Make a random improvement
            improved = False

            # Try to improve GC content if outside range
            if gc_content < self.gc_content_range[0] or gc_content > self.gc_content_range[1]:
                improved = self._adjust_gc_content(optimized_dna, protein_sequence, gc_content)

            # Try to remove restriction sites
            elif avoid_sites and restriction_sites:
                improved = self._remove_restriction_sites(optimized_dna, protein_sequence, restriction_sites)

            # Try to improve CAI if below threshold
            elif cai < 0.8:
                improved = self._improve_cai(optimized_dna, protein_sequence, cai)

            # Break if no improvement was made
            if not improved:
                break

            # Update properties
            optimized_dna = improved
            cai = self.calculate_cai(optimized_dna)
            gc_content = self.calculate_gc_content(optimized_dna)
            restriction_sites = self.check_restriction_sites(optimized_dna) if avoid_sites else []

            iteration += 1

        return optimized_dna, cai, gc_content, restriction_sites

    def _adjust_gc_content(self, dna_sequence, protein_sequence, current_gc):
        """Adjust GC content of DNA sequence."""
        # Convert DNA to codons
        dna = dna_sequence.upper()
        codons = [dna[i:i + 3] for i in range(0, len(dna), 3)]

        # Determine whether to increase or decrease GC
        increase_gc = current_gc < self.gc_content_range[0]

        # Randomly select a position to modify
        positions = list(range(len(codons)))
        np.random.shuffle(positions)

        for pos in positions:
            codon = codons[pos]
            aa = self.genetic_code[codon.replace('T', 'U')]

            # Get alternative codons for this amino acid
            alt_codons = [c.replace('U', 'T') for c in [codon for codon, amino in self.genetic_code.items() if amino == aa]]

            # Calculate GC content of each alternative
            gc_contents = [(c.count('G') + c.count('C')) / 3 for c in alt_codons]

            # Sort by GC content (ascending or descending)
            sorted_codons = sorted(zip(alt_codons, gc_contents),
                                   key=lambda x: x[1],
                                   reverse=increase_gc)

            # Find a better codon
            for new_codon, new_gc in sorted_codons:
                if new_codon != codon:
                    # Only accept if it improves GC content in the right direction
                    if (increase_gc and new_gc > codon.count('G') + codon.count('C') / 3) or \
                            (not increase_gc and new_gc < codon.count('G') + codon.count('C') / 3):
                        # Create new sequence
                        new_sequence = dna[:pos * 3] + new_codon + dna[(pos + 1) * 3:]
                        return new_sequence

        return False

    def _remove_restriction_sites(self, dna_sequence, protein_sequence, sites):
        """Remove restriction sites from DNA sequence."""
        dna = dna_sequence.upper()

        # For each restriction site
        for enzyme in sites:
            site = self.restriction_sites[enzyme]

            # Find all occurrences
            matches = [m.start() for m in re.finditer(site, dna)]

            for match in matches:
                # Determine affected codons
                start_codon = match // 3
                end_codon = (match + len(site) - 1) // 3

                # Try to modify each affected codon
                for pos in range(start_codon, end_codon + 1):
                    if pos >= len(protein_sequence):
                        continue

                    codon = dna[pos * 3:pos * 3 + 3]
                    aa = self.genetic_code[codon.replace('T', 'U')]

                    # Get alternative codons
                    alt_codons = [c.replace('U', 'T') for c in aa_to_codons[aa]]

                    # Try each alternative
                    for new_codon in alt_codons:
                        if new_codon == codon:
                            continue

                        # Create new sequence
                        new_sequence = dna[:pos * 3] + new_codon + dna[(pos + 1) * 3:]

                        # Check if site is removed
                        if self.restriction_sites[enzyme] not in new_sequence:
                            return new_sequence

        return False

    def _improve_cai(self, dna_sequence, protein_sequence, current_cai):
        """Improve CAI of DNA sequence."""
        dna = dna_sequence.upper()
        codons = [dna[i:i + 3] for i in range(0, len(dna), 3)]

        # Identify codons with low adaptiveness
        rna_codons = [c.replace('T', 'U') for c in codons]
        adaptiveness = [self.relative_adaptiveness.get(c, 0) for c in rna_codons]

        # Sort positions by adaptiveness (ascending)
        positions = sorted(range(len(adaptiveness)), key=lambda i: adaptiveness[i])

        # Try to improve worst codons first
        for pos in positions:
            if pos >= len(protein_sequence):
                continue

            codon = codons[pos]
            aa = self.genetic_code[codon.replace('T', 'U')]

            # Get alternative codons
            alt_codons = [c.replace('U', 'T') for c in aa_to_codons[aa]]

            # Sort by adaptiveness (descending)
            sorted_codons = sorted(alt_codons,
                                   key=lambda c: self.relative_adaptiveness.get(c.replace('T', 'U'), 0),
                                   reverse=True)

            # Try each alternative
            for new_codon in sorted_codons:
                if new_codon == codon:
                    continue

                # Only accept if it improves adaptiveness
                if self.relative_adaptiveness.get(new_codon.replace('T', 'U'), 0) > \
                        self.relative_adaptiveness.get(codon.replace('T', 'U'), 0):
                    # Create new sequence
                    new_sequence = dna[:pos * 3] + new_codon + dna[(pos + 1) * 3:]
                    return new_sequence

        return False