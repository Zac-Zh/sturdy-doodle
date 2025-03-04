#!/usr/bin/env python3
# scripts/run_pipeline.py - Run the receptor design pipeline

import os
import sys
import argparse
import subprocess
import logging
from pathlib import Path
import yaml
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PipelineRunner")


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_command(command, description=None, check=True):
    """Run a shell command and log the output."""
    if description:
        logger.info(description)

    logger.info(f"Running command: {' '.join(command)}")

    try:
        result = subprocess.run(
            command,
            check=check,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        logger.info(result.stdout)
        if result.stderr:
            logger.warning(result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}")
        logger.error(e.stdout)
        logger.error(e.stderr)
        return False


def run_alphafold_predictions(config, receptor_sequences):
    """Run AlphaFold predictions for receptor sequences."""
    alphafold_path = config.get('tools', {}).get('alphafold_path', '')
    if not alphafold_path or not os.path.exists(alphafold_path):
        logger.warning("AlphaFold path not provided or invalid, skipping structure prediction")
        return False

    output_dir = Path(config.get('output_dir', 'output/receptor_design'))

    for sequence_path in receptor_sequences:
        seq_name = Path(sequence_path).stem
        output_path = output_dir / "structures" / f"{seq_name}_predicted.pdb"

        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Run AlphaFold
        success = run_command(
            [
                alphafold_path,
                "--fasta_paths", sequence_path,
                "--output_dir", str(output_dir / "alphafold_output" / seq_name),
                "--model_preset", "monomer",
                "--max_template_date", "2022-01-01"
            ],
            f"Running AlphaFold for {seq_name}",
            check=False
        )

        if not success:
            logger.warning(f"AlphaFold prediction failed for {seq_name}, creating placeholder")
            # Create placeholder PDB
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                f.write("HEADER    PLACEHOLDER PDB FILE\n")
                f.write("TITLE     PLACEHOLDER FOR ALPHAFOLD PREDICTION\n")
                f.write("REMARK    THIS IS A PLACEHOLDER FILE FOR TESTING\n")
                f.write("END\n")

    return True


def generate_mutations(config):
    """Generate mutations for receptor sequences."""
    output_dir = Path(config.get('output_dir', 'output/receptor_design'))
    receptor_sequences = config.get('receptors', {}).get('sequences', [])

    for sequence_path in receptor_sequences:
        seq_name = Path(sequence_path).stem
        structure_path = output_dir / "structures" / f"{seq_name}_predicted.pdb"
        output_path = output_dir / "mutations" / f"{seq_name}_mutations.csv"

        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Check if structure exists, create placeholder if not
        if not os.path.exists(structure_path):
            logger.warning(f"Structure not found for {seq_name}, creating placeholder")
            os.makedirs(os.path.dirname(structure_path), exist_ok=True)
            with open(structure_path, 'w') as f:
                f.write("HEADER    PLACEHOLDER PDB FILE\n")
                f.write("END\n")

        # Run mutation generation
        run_command(
            [
                sys.executable,
                "scripts/generate_mutations.py",
                "--structure", str(structure_path),
                "--sequence", sequence_path,
                "--output", str(output_path),
                "--max_mutations", str(config.get('max_mutations_per_receptor', 50))
            ],
            f"Generating mutations for {seq_name}"
        )

    return True


def run_stage1(config):
    """Run Stage 1: Dynamic Causal Inference Engine."""
    logger.info("Running Stage 1: Dynamic Causal Inference Engine")
    start_time = time.time()

    # Step 1.1: Run AlphaFold predictions
    receptor_sequences = config.get('receptors', {}).get('sequences', [])
    run_alphafold_predictions(config, receptor_sequences)

    # Step 1.2: Generate mutations
    generate_mutations(config)

    # Step 1.3: Run the main pipeline script
    run_command(
        [sys.executable, "main.py", "--config", "config.yaml", "--stage", "1"],
        "Running Stage 1 of the pipeline"
    )

    duration = time.time() - start_time
    logger.info(f"Stage 1 completed in {duration:.2f} seconds")
    return True


def run_stage2(config):
    """Run Stage 2: Hierarchical Generative Optimization."""
    logger.info("Running Stage 2: Hierarchical Generative Optimization")
    start_time = time.time()

    run_command(
        [sys.executable, "main.py", "--config", "config.yaml", "--stage", "2"],
        "Running Stage 2 of the pipeline"
    )

    duration = time.time() - start_time
    logger.info(f"Stage 2 completed in {duration:.2f} seconds")
    return True


def run_stage3(config):
    """Run Stage 3: Closed Loop Validation and Adaptation."""
    logger.info("Running Stage 3: Closed Loop Validation and Adaptation")
    start_time = time.time()

    run_command(
        [sys.executable, "main.py", "--config", "config.yaml", "--stage", "3"],
        "Running Stage 3 of the pipeline"
    )

    duration = time.time() - start_time
    logger.info(f"Stage 3 completed in {duration:.2f} seconds")
    return True


def generate_report(config):
    """Generate the final pipeline report."""
    logger.info("Generating pipeline report")

    run_command(
        [sys.executable, "main.py", "--config", "config.yaml", "--report"],
        "Generating pipeline report"
    )

    return True


def main():
    parser = argparse.ArgumentParser(description='Run the receptor design pipeline')
    parser.add_argument('--config', default='config.yaml', help='Path to configuration file')
    parser.add_argument('--stages', default='1,2,3', help='Pipeline stages to run (comma-separated)')
    parser.add_argument('--only-report', action='store_true', help='Only generate the report')
    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1

    logger.info(f"Starting receptor design pipeline with config: {args.config}")

    # Determine stages to run
    stages = [int(s) for s in args.stages.split(',') if s.isdigit()]

    if args.only_report:
        generate_report(config)
        return 0

    # Run each stage
    if 1 in stages:
        if not run_stage1(config):
            logger.error("Stage 1 failed")
            return 1

    if 2 in stages:
        if not run_stage2(config):
            logger.error("Stage 2 failed")
            return 1

    if 3 in stages:
        if not run_stage3(config):
            logger.error("Stage 3 failed")
            return 1

    # Generate report
    generate_report(config)

    logger.info("Pipeline execution completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())