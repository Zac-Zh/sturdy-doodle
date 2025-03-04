import os
import subprocess
import numpy as np
import MDAnalysis as mda
from Bio.PDB import PDBParser


class GromacsAnalyzer:
    """Analyze protein structures and mutations using GROMACS."""

    def __init__(self, gromacs_path, work_dir):
        """
        Initialize GROMACS analyzer.

        Args:
            gromacs_path: Path to GROMACS executable
            work_dir: Working directory for GROMACS runs
        """
        self.gromacs_path = gromacs_path
        self.work_dir = work_dir
        os.makedirs(work_dir, exist_ok=True)

    def _prepare_simulation(self, pdb_path, output_name):
        """
        Prepare GROMACS simulation from PDB structure.

        Args:
            pdb_path: Path to input PDB file
            output_name: Base name for output files

        Returns:
            bool: Success status
        """
        try:
            # Convert PDB to GROMACS format
            pdb2gmx_cmd = [
                f"{self.gromacs_path}gmx", "pdb2gmx",
                "-f", pdb_path,
                "-o", f"{output_name}.gro",
                "-p", f"{output_name}.top",
                "-ff", "amber99sb-ildn",  # Force field
                "-water", "tip3p",
                "-ignh"  # Ignore hydrogen atoms
            ]
            subprocess.run(pdb2gmx_cmd, check=True, cwd=self.work_dir)

            # Define simulation box
            editconf_cmd = [
                f"{self.gromacs_path}gmx", "editconf",
                "-f", f"{output_name}.gro",
                "-o", f"{output_name}_box.gro",
                "-c",  # Center
                "-d", "1.0",  # Distance to box edge
                "-bt", "cubic"  # Box type
            ]
            subprocess.run(editconf_cmd, check=True, cwd=self.work_dir)

            # Solvate the box
            solvate_cmd = [
                f"{self.gromacs_path}gmx", "solvate",
                "-cp", f"{output_name}_box.gro",
                "-cs", "spc216.gro",
                "-o", f"{output_name}_solv.gro",
                "-p", f"{output_name}.top"
            ]
            subprocess.run(solvate_cmd, check=True, cwd=self.work_dir)

            # Add ions for neutralization
            # First create input file for genion
            with open(os.path.join(self.work_dir, "ions.mdp"), "w") as f:
                f.write("; minimal ions.mdp file\n")
                f.write("integrator = steep\n")
                f.write("nsteps = 0\n")
                f.write("cutoff-scheme = Verlet\n")

            # Generate preprocessed topology
            grompp_cmd = [
                f"{self.gromacs_path}gmx", "grompp",
                "-f", "ions.mdp",
                "-c", f"{output_name}_solv.gro",
                "-p", f"{output_name}.top",
                "-o", f"{output_name}_ions.tpr"
            ]
            subprocess.run(grompp_cmd, check=True, cwd=self.work_dir)

            # Add ions (automated selection of SOL group)
            genion_input = b"SOL\n"
            genion_cmd = [
                f"{self.gromacs_path}gmx", "genion",
                "-s", f"{output_name}_ions.tpr",
                "-o", f"{output_name}_ions.gro",
                "-p", f"{output_name}.top",
                "-pname", "NA", "-nname", "CL",
                "-neutral"
            ]
            subprocess.run(genion_cmd, input=genion_input, check=True, cwd=self.work_dir)

            return True

        except subprocess.CalledProcessError as e:
            print(f"GROMACS preparation failed: {e}")
            return False

    def calculate_binding_energy(self, complex_pdb, receptor_pdb, ligand_pdb):
        """
        Calculate binding free energy (ΔΔG) between receptor and ligand.

        Args:
            complex_pdb: Path to complex structure PDB
            receptor_pdb: Path to receptor structure PDB
            ligand_pdb: Path to ligand structure PDB

        Returns:
            float: Estimated binding free energy in kcal/mol
        """
        # This is a simplified implementation. A full implementation would:
        # 1. Run equilibration and production MD
        # 2. Use MM-PBSA or similar method for binding energy

        try:
            # Prepare each system
            self._prepare_simulation(complex_pdb, "complex")
            self._prepare_simulation(receptor_pdb, "receptor")
            self._prepare_simulation(ligand_pdb, "ligand")

            # Create energy minimization parameters
            with open(os.path.join(self.work_dir, "em.mdp"), "w") as f:
                f.write("; Energy minimization parameters\n")
                f.write("integrator = steep\n")
                f.write("nsteps = 1000\n")
                f.write("cutoff-scheme = Verlet\n")
                f.write("rcoulomb = 1.0\n")
                f.write("rvdw = 1.0\n")
                f.write("emtol = 1000.0\n")
                f.write("emstep = 0.01\n")

            # For complex, receptor, and ligand systems:
            systems = ["complex", "receptor", "ligand"]
            energies = {}

            for system in systems:
                # Run energy minimization
                grompp_cmd = [
                    f"{self.gromacs_path}gmx", "grompp",
                    "-f", "em.mdp",
                    "-c", f"{system}_ions.gro",
                    "-p", f"{system}.top",
                    "-o", f"{system}_em.tpr"
                ]
                subprocess.run(grompp_cmd, check=True, cwd=self.work_dir)

                mdrun_cmd = [
                    f"{self.gromacs_path}gmx", "mdrun",
                    "-v",
                    "-s", f"{system}_em.tpr",
                    "-deffnm", f"{system}_em"
                ]
                subprocess.run(mdrun_cmd, check=True, cwd=self.work_dir)

                # Extract energy
                energy_cmd = [
                    f"{self.gromacs_path}gmx", "energy",
                    "-f", f"{system}_em.edr",
                    "-o", f"{system}_energy.xvg"
                ]
                energy_input = b"10\n0\n"  # Select potential energy, then exit
                subprocess.run(energy_cmd, input=energy_input, check=True, cwd=self.work_dir)

                # Parse energy file (simple approach)
                energy_file = os.path.join(self.work_dir, f"{system}_energy.xvg")
                with open(energy_file, 'r') as f:
                    lines = f.readlines()

                # Get final energy value (last non-comment line)
                energy_lines = [line for line in lines if not line.startswith("#") and not line.startswith("@")]
                final_energy = float(energy_lines[-1].split()[1])
                energies[system] = final_energy

            # Calculate binding energy: ΔG = G_complex - (G_receptor + G_ligand)
            binding_energy = energies["complex"] - (energies["receptor"] + energies["ligand"])

            # Convert from kJ/mol to kcal/mol
            binding_energy_kcal = binding_energy / 4.184

            return binding_energy_kcal

        except Exception as e:
            print(f"Binding energy calculation failed: {e}")
            return None

    def calculate_rmsf(self, pdb_path, traj_path=None):
        """
        Calculate Root Mean Square Fluctuation (RMSF) for structure stability.

        Args:
            pdb_path: Path to structure PDB
            traj_path: Optional path to trajectory file

        Returns:
            float: Average RMSF value in Angstroms
        """
        try:
            if not traj_path:
                # If no trajectory provided, run a short MD to generate one
                # Prepare system
                base_name = os.path.basename(pdb_path).split('.')[0]
                success = self._prepare_simulation(pdb_path, base_name)
                if not success:
                    return None

                # Create MD parameters for a very short run
                with open(os.path.join(self.work_dir, "md.mdp"), "w") as f:
                    f.write("; Short MD parameters\n")
                    f.write("integrator = md\n")
                    f.write("nsteps = 5000\n")  # 10 ps at 2 fs timestep
                    f.write("dt = 0.002\n")
                    f.write("cutoff-scheme = Verlet\n")
                    f.write("rcoulomb = 1.0\n")
                    f.write("rvdw = 1.0\n")
                    f.write("tcoupl = V-rescale\n")
                    f.write("tc-grps = Protein Non-Protein\n")
                    f.write("tau_t = 0.1 0.1\n")
                    f.write("ref_t = 300 300\n")
                    f.write("gen_vel = yes\n")
                    f.write("gen_temp = 300\n")
                    f.write("nstxout = 100\n")  # Save coordinates every 0.2 ps

                # Run MD
                grompp_cmd = [
                    f"{self.gromacs_path}gmx", "grompp",
                    "-f", "md.mdp",
                    "-c", f"{base_name}_ions.gro",
                    "-p", f"{base_name}.top",
                    "-o", f"{base_name}_md.tpr"
                ]
                subprocess.run(grompp_cmd, check=True, cwd=self.work_dir)

                mdrun_cmd = [
                    f"{self.gromacs_path}gmx", "mdrun",
                    "-v",
                    "-s", f"{base_name}_md.tpr",
                    "-deffnm", f"{base_name}_md"
                ]
                subprocess.run(mdrun_cmd, check=True, cwd=self.work_dir)

                traj_path = os.path.join(self.work_dir, f"{base_name}_md.xtc")

            # Calculate RMSF
            rmsf_cmd = [
                f"{self.gromacs_path}gmx", "rmsf",
                "-s", f"{base_name}_md.tpr",
                "-f", traj_path,
                "-o", f"{base_name}_rmsf.xvg",
                "-res"  # Per-residue RMSF
            ]
            rmsf_input = b"1\n"  # Select backbone group
            subprocess.run(rmsf_cmd, input=rmsf_input, check=True, cwd=self.work_dir)

            # Parse RMSF file
            rmsf_file = os.path.join(self.work_dir, f"{base_name}_rmsf.xvg")
            rmsf_values = []

            with open(rmsf_file, 'r') as f:
                for line in f:
                    if not line.startswith("#") and not line.startswith("@"):
                        parts = line.strip().split()
                        rmsf_values.append(float(parts[1]))

            # Calculate average RMSF
            avg_rmsf = np.mean(rmsf_values)
            return avg_rmsf

        except Exception as e:
            print(f"RMSF calculation failed: {e}")
            return None

    def screen_mutations(self, wildtype_pdb, mutant_pdbs, ligand_pdb=None):
        """
        Screen mutations for stability and binding.

        Args:
            wildtype_pdb: Path to wildtype structure
            mutant_pdbs: Dict mapping mutation IDs to PDB paths
            ligand_pdb: Optional path to ligand for binding calculations

        Returns:
            dict: Results with ΔΔG and RMSF for each mutation
        """
        results = {}

        # Calculate wildtype properties
        wt_rmsf = self.calculate_rmsf(wildtype_pdb)
        wt_binding = None
        if ligand_pdb:
            # For binding calculations, we would need complex structures
            # This is simplified
            wt_binding = 0.0

        # For each mutant
        for mutation, pdb_path in mutant_pdbs.items():
            if pdb_path is None:
                results[mutation] = {"valid": False, "reason": "Structure prediction failed"}
                continue

            # Calculate RMSF for stability
            mut_rmsf = self.calculate_rmsf(pdb_path)

            # Calculate binding energy change if ligand provided
            mut_binding = None
            delta_binding = None
            if ligand_pdb:
                # Simplified - would need complex structures
                mut_binding = 0.0
                delta_binding = mut_binding - wt_binding

            # Store results
            results[mutation] = {
                "valid": True,
                "rmsf": mut_rmsf,
                "delta_rmsf": mut_rmsf - wt_rmsf if mut_rmsf and wt_rmsf else None,
                "binding_energy": mut_binding,
                "delta_binding": delta_binding,
                "passes_filter": (
                        mut_rmsf is not None and
                        mut_rmsf < 1.5 and  # RMSF stability criterion
                        (delta_binding is None or delta_binding > 2.0)  # Binding energy criterion
                )
            }

        return results
