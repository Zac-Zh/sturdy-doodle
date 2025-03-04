#!/usr/bin/env python3
# download_data.py - Script to download and prepare data for receptor design

import os
import sys
import requests
import gzip
import shutil
import json
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DataDownloader")

# UniProt IDs for different receptor families
RECEPTOR_DATA = {
    "gpcr_class_a": [
        {"id": "P07550", "name": "ADRB2_HUMAN", "description": "Beta-2 adrenergic receptor"},
        {"id": "P08172", "name": "ACM2_HUMAN", "description": "Muscarinic acetylcholine receptor M2"},
        {"id": "P35462", "name": "D2DR_HUMAN", "description": "D(2) dopamine receptor"}
    ],
    "cytokine_receptors": [
        {"id": "P08887", "name": "IL6R_HUMAN", "description": "Interleukin-6 receptor subunit alpha"},
        {"id": "P16871", "name": "IL7R_HUMAN", "description": "Interleukin-7 receptor subunit alpha"},
        {"id": "P40189", "name": "IL6RB_HUMAN", "description": "Interleukin-6 receptor subunit beta"}
    ],
    "ion_channels": [
        {"id": "P16389", "name": "KCNA2_HUMAN", "description": "Potassium voltage-gated channel subfamily A member 2"},
        {"id": "P22460", "name": "KCNA5_HUMAN", "description": "Potassium voltage-gated channel subfamily A member 5"},
        {"id": "P22459", "name": "KCNA4_HUMAN", "description": "Potassium voltage-gated channel subfamily A member 4"}
    ]
}

# Example causal relationships for expert knowledge
EXPERT_KNOWLEDGE = {
    "causal_relationships": [
        {
            "source": "helix_orientation",
            "target": "binding_pocket_size",
            "confidence": 0.8,
            "evidence": "Literature consensus"
        },
        {
            "source": "loop_flexibility",
            "target": "ligand_binding_rate",
            "confidence": 0.6,
            "evidence": "Molecular dynamics studies"
        },
        {
            "source": "transmembrane_domain_hydrophobicity",
            "target": "membrane_insertion_efficiency",
            "confidence": 0.75,
            "evidence": "Experimental data from mutagenesis"
        },
        {
            "source": "disulfide_bond_formation",
            "target": "protein_stability",
            "confidence": 0.85,
            "evidence": "Structural analysis"
        },
        {
            "source": "glycosylation_site_accessibility",
            "target": "protein_folding_rate",
            "confidence": 0.5,
            "evidence": "Preliminary experimental findings"
        }
    ]
}


def ensure_directories():
    """Ensure all necessary directories exist."""
    directories = [
        "data/sequences",
        "data/expert_knowledge",
        "output/receptor_design",
        "output/logs"
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")


def download_uniprot_sequence(uniprot_id):
    """Download protein sequence from UniProt."""
    url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download sequence for {uniprot_id}: {e}")
        # Provide a placeholder sequence if download fails
        return f">{uniprot_id} Placeholder sequence\nMGTLEIRPASVSGWGVLLLLLLVLVVVVAVLGSSRPHYPRGWSLPDGPRDVVYAGLHGVLRGG"


def create_fasta_files():
    """Create FASTA files for receptor families."""
    for family, receptors in RECEPTOR_DATA.items():
        filename = f"data/sequences/{family}.fasta"
        with open(filename, 'w') as f:
            for receptor in receptors:
                sequence_data = download_uniprot_sequence(receptor["id"])
                f.write(sequence_data)
                if not sequence_data.endswith('\n'):
                    f.write('\n')

        logger.info(f"Created FASTA file: {filename}")


def create_expert_knowledge():
    """Create expert knowledge JSON file."""
    filename = "data/expert_knowledge/receptor_domain_causal_priors.json"
    with open(filename, 'w') as f:
        json.dump(EXPERT_KNOWLEDGE, f, indent=2)

    logger.info(f"Created expert knowledge file: {filename}")


def create_placeholder_config():
    """Create a placeholder configuration file if it doesn't exist."""
    if os.path.exists("config.yaml"):
        logger.info("Config file already exists, skipping")
        return

    config_content = """# Configuration for Receptor Design Pipeline
experiment_name: "receptorgan_pipeline_v1"
output_dir: "output/receptor_design"
random_seed: 42

# Receptor specifications
receptors:
  sequences:
    - "data/sequences/gpcr_class_a.fasta"
    - "data/sequences/cytokine_receptors.fasta"
    - "data/sequences/ion_channels.fasta"
  families:
    - "GPCR"
    - "Cytokine"
    - "IonChannel"
  target_functions:
    - "antagonist_il6"
    - "agonist_vegfr"
    - "inhibitor_tnf"

# External tools settings - MODIFY THESE PATHS FOR YOUR SYSTEM
tools:
  alphafold_path: "/path/to/alphafold"
  gromacs_path: "/path/to/gromacs"
"""

    with open("config.yaml", 'w') as f:
        f.write(config_content)

    logger.info("Created placeholder config.yaml file")


def main():
    """Main function to download and prepare data."""
    logger.info("Starting data download and preparation")

    # Create necessary directories
    ensure_directories()

    # Create receptor sequence FASTA files
    create_fasta_files()

    # Create expert knowledge file
    create_expert_knowledge()

    # Create placeholder config
    create_placeholder_config()

    logger.info("Data preparation completed successfully")


if __name__ == "__main__":
    main()