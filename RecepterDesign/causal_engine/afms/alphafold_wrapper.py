import os
import subprocess
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


class AlphaFoldWrapper:
    """Wrapper for AlphaFold to predict mutant protein structures."""

    def __init__(self, alphafold_path, output_dir):
        """
        Initialize AlphaFold wrapper.

        Args:
            alphafold_path: Path to AlphaFold installation
            output_dir: Directory to store generated structures
        """
        self.alphafold_path = alphafold_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def generate_mutant_sequence(self, wildtype_seq, mutation):
        """
        Generate mutant sequence based on mutation description.

        Args:
            wildtype_seq: Original protein sequence
            mutation: Mutation in format "A123B" (original -> position -> mutant)

        Returns:
            str: Mutated sequence
        """
        wt_aa = mutation[0]
        position = int(mutation[1:-1]) - 1  # Convert to 0-indexed
        mut_aa = mutation[-1]

        # Verify wildtype matches expected residue
        if wildtype_seq[position] != wt_aa:
            raise ValueError(f"Wildtype residue at position {position + 1} is {wildtype_seq[position]}, not {wt_aa}")

        # Create mutated sequence
        mutated_seq = wildtype_seq[:position] + mut_aa + wildtype_seq[position + 1:]
        return mutated_seq

    def predict_structure(self, sequence, name):
        """
        Run AlphaFold to predict protein structure.

        Args:
            sequence: Protein sequence
            name: Identifier for the structure

        Returns:
            str: Path to output PDB file
        """
        # Create FASTA file for input
        fasta_path = os.path.join(self.output_dir, f"{name}.fasta")
        with open(fasta_path, 'w') as f:
            f.write(f">{name}\n{sequence}\n")

        # Prepare AlphaFold command
        output_path = os.path.join(self.output_dir, name)
        os.makedirs(output_path, exist_ok=True)

        cmd = [
            self.alphafold_path,
            "--fasta_paths", fasta_path,
            "--output_dir", output_path,
            "--model_preset", "monomer",
            "--max_template_date", "2022-01-01",
            "--use_gpu_relax", "True"
        ]

        # Run AlphaFold (in real implementation, handle errors properly)
        try:
            subprocess.run(cmd, check=True)
            # Return path to the predicted structure (best model)
            return os.path.join(output_path, "ranked_0.pdb")
        except subprocess.CalledProcessError as e:
            print(f"AlphaFold prediction failed: {e}")
            return None

    def run_mutation_screening(self, wildtype_seq, mutations):
        """
        Screen multiple mutations using AlphaFold.

        Args:
            wildtype_seq: Original protein sequence
            mutations: List of mutations to screen

        Returns:
            dict: Mapping from mutation to predicted structure path
        """
        results = {}

        # Predict wildtype structure as reference
        wildtype_path = self.predict_structure(wildtype_seq, "wildtype")
        results["wildtype"] = wildtype_path

        # Predict each mutant structure
        for mutation in mutations:
            try:
                mutant_seq = self.generate_mutant_sequence(wildtype_seq, mutation)
                mutant_path = self.predict_structure(mutant_seq, mutation)
                results[mutation] = mutant_path
            except Exception as e:
                print(f"Failed to process mutation {mutation}: {e}")
                results[mutation] = None

        return results
