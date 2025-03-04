import os
import subprocess
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.PDB import PDBParser, PDBIO, Structure, Model, Chain, Residue
from Bio.PDB.Polypeptide import PPBuilder
from Bio.PDB.DSSP import dssp_dict_from_pdb_file
from typing import Dict, List, Optional, Tuple


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
        self.pdb_parser = PDBParser(QUIET=True)

    def generate_mutant_sequence(self, wildtype_seq: str, mutation: str) -> str:
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

    def predict_structure(self, sequence: str, name: str) -> Optional[str]:
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

        # Configure AlphaFold parameters for high accuracy
        cmd = [
            self.alphafold_path,
            "--fasta_paths", fasta_path,
            "--output_dir", output_path,
            "--model_preset", "monomer_ptm",  # Use pTM model for better accuracy
            "--max_template_date", "2022-01-01",
            "--db_preset", "full_dbs",  # Use full databases
            "--num_multimer_predictions_per_model", "5",  # Generate multiple predictions
            "--use_gpu_relax", "True",
            "--num_ensemble", "1",
            "--num_recycle", "3"  # More recycling for better accuracy
        ]

        try:
            subprocess.run(cmd, check=True)
            
            # Get all predicted models
            model_paths = [os.path.join(output_path, f"ranked_{i}.pdb") for i in range(5)]
            best_model = self._select_best_model(model_paths)
            
            return best_model

        except subprocess.CalledProcessError as e:
            print(f"AlphaFold prediction failed: {e}")
            return None

    def _select_best_model(self, model_paths: List[str]) -> str:
        """
        Select the best model based on structure quality metrics.

        Args:
            model_paths: List of paths to predicted structure models

        Returns:
            str: Path to the best model
        """
        best_score = float('-inf')
        best_model = None

        for model_path in model_paths:
            if os.path.exists(model_path):
                # Calculate structure quality metrics
                score = self._evaluate_structure(model_path)
                if score > best_score:
                    best_score = score
                    best_model = model_path

        return best_model

    def _evaluate_structure(self, pdb_path: str) -> float:
        """
        Evaluate structure quality using multiple metrics.

        Args:
            pdb_path: Path to PDB file

        Returns:
            float: Combined quality score
        """
        try:
            # Parse structure
            structure = self.pdb_parser.get_structure('model', pdb_path)
            
            # Calculate secondary structure content
            dssp_dict = dssp_dict_from_pdb_file(pdb_path)
            ss_content = self._calculate_ss_content(dssp_dict)
            
            # Calculate packing quality (simplified)
            packing_score = self._calculate_packing_quality(structure)
            
            # Calculate Ramachandran plot statistics (simplified)
            rama_score = self._calculate_rama_score(structure)
            
            # Combine scores (weights can be adjusted)
            total_score = (
                0.4 * ss_content +
                0.3 * packing_score +
                0.3 * rama_score
            )
            
            return total_score
            
        except Exception as e:
            print(f"Structure evaluation failed: {e}")
            return float('-inf')

    def _calculate_ss_content(self, dssp_dict: Dict) -> float:
        """
        Calculate secondary structure content score.
        """
        if not dssp_dict:
            return 0.0
            
        ss_counts = {'H': 0, 'E': 0, 'C': 0}  # Helix, Sheet, Coil
        total = 0
        
        for residue in dssp_dict.values():
            ss = residue[2]
            if ss in ['H', 'G', 'I']:  # All helical types
                ss_counts['H'] += 1
            elif ss in ['E', 'B']:  # All sheet types
                ss_counts['E'] += 1
            else:
                ss_counts['C'] += 1
            total += 1
            
        if total == 0:
            return 0.0
            
        # Calculate normalized score (prefer some secondary structure over all coil)
        helix_sheet_fraction = (ss_counts['H'] + ss_counts['E']) / total
        return min(helix_sheet_fraction * 2, 1.0)  # Scale to [0,1]

    def _calculate_packing_quality(self, structure: Structure.Structure) -> float:
        """
        Calculate structure packing quality score.
        """
        try:
            # Simple measure based on C-alpha distances
            ca_atoms = [atom for atom in structure.get_atoms() if atom.name == 'CA']
            if len(ca_atoms) < 2:
                return 0.0
                
            # Calculate average distance to nearby residues
            total_score = 0
            count = 0
            
            for i, ca1 in enumerate(ca_atoms):
                local_dists = []
                for j, ca2 in enumerate(ca_atoms):
                    if abs(i - j) > 1:  # Non-adjacent residues
                        dist = ca1 - ca2
                        if dist < 12.0:  # Only consider nearby residues
                            local_dists.append(dist)
                
                if local_dists:
                    # Score based on number and distribution of contacts
                    avg_dist = np.mean(local_dists)
                    score = np.exp(-(avg_dist - 7.0)**2 / 8.0)  # Gaussian around ideal distance
                    total_score += score
                    count += 1
            
            return total_score / count if count > 0 else 0.0
            
        except Exception:
            return 0.0

    def _calculate_rama_score(self, structure: Structure.Structure) -> float:
        """
        Calculate Ramachandran plot statistics score.
        """
        try:
            ppb = PPBuilder()
            phi_psi = []
            
            # Get phi/psi angles
            for pp in ppb.build_peptides(structure):
                angles = pp.get_phi_psi_list()
                phi_psi.extend(angles)
            
            if not phi_psi:
                return 0.0
                
            # Count angles in allowed regions (simplified)
            good_angles = 0
            total_angles = 0
            
            for phi, psi in phi_psi:
                if phi is not None and psi is not None:
                    # Very simplified Ramachandran plot regions
                    if (-180 <= phi <= 0 and -120 <= psi <= 30) or \
                       (-180 <= phi <= 0 and 120 <= psi <= 180) or \
                       (-180 <= phi <= -45 and -180 <= psi <= -120):
                        good_angles += 1
                    total_angles += 1
            
            return good_angles / total_angles if total_angles > 0 else 0.0
            
        except Exception:
            return 0.0

    def run_mutation_screening(self, wildtype_seq: str, mutations: List[str]) -> Dict[str, Optional[str]]:
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
