import os
import json
import datetime
import hashlib
import pandas as pd
import numpy as np
from pathlib import Path


class ExperimentTracker:
    """Track experiments and manage data versions for receptor design pipeline."""

    def __init__(self, base_dir="experiments", experiment_name=None):
        """
        Initialize experiment tracker.

        Args:
            base_dir: Base directory for experiments
            experiment_name: Optional experiment name (generated if None)
        """
        self.base_dir = Path(base_dir)

        # Generate experiment name if not provided
        if experiment_name is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"experiment_{timestamp}"

        self.experiment_name = experiment_name
        self.experiment_dir = self.base_dir / experiment_name

        # Create directories
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        (self.experiment_dir / "data").mkdir(exist_ok=True)
        (self.experiment_dir / "models").mkdir(exist_ok=True)
        (self.experiment_dir / "results").mkdir(exist_ok=True)
        (self.experiment_dir / "logs").mkdir(exist_ok=True)

        # Initialize metadata
        self.metadata = {
            "experiment_name": experiment_name,
            "created_at": datetime.datetime.now().isoformat(),
            "datasets": {},
            "models": {},
            "results": {},
            "runs": []
        }

        # Save initial metadata
        self._save_metadata()

        # Initialize current run
        self.current_run = None

    def start_run(self, run_name=None, description=None, params=None):
        """
        Start a new experimental run.

        Args:
            run_name: Run name (generated if None)
            description: Run description
            params: Run parameters

        Returns:
            str: Run name
        """
        # Generate run name if not provided
        if run_name is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"run_{timestamp}"

        # Create run directory
        run_dir = self.experiment_dir / "runs" / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        # Initialize run metadata
        self.current_run = {
            "run_name": run_name,
            "status": "running",
            "started_at": datetime.datetime.now().isoformat(),
            "completed_at": None,
            "description": description,
            "params": params or {},
            "metrics": {},
            "artifacts": [],
            "logs": []
        }

        # Add to metadata
        self.metadata["runs"].append(self.current_run)
        self._save_metadata()

        print(f"Started run: {run_name}")
        return run_name

    def end_run(self, status="completed"):
        """
        End the current run.

        Args:
            status: Run status (completed, failed, etc.)

        Returns:
            bool: Success flag
        """
        if self.current_run is None:
            print("No active run to end")
            return False

        # Update run metadata
        self.current_run["status"] = status
        self.current_run["completed_at"] = datetime.datetime.now().isoformat()

        # Save metadata
        self._save_metadata()

        print(f"Ended run: {self.current_run['run_name']} with status: {status}")
        self.current_run = None

        return True

    def log_metric(self, key, value, step=None):
        """
        Log a metric for the current run.

        Args:
            key: Metric name
            value: Metric value
            step: Optional step number

        Returns:
            bool: Success flag
        """
        if self.current_run is None:
            print("No active run for logging metrics")
            return False

        # Initialize metric entry if needed
        if key not in self.current_run["metrics"]:
            self.current_run["metrics"][key] = []

        # Add metric
        metric_entry = {
            "value": value,
            "timestamp": datetime.datetime.now().isoformat()
        }
        if step is not None:
            metric_entry["step"] = step

        self.current_run["metrics"][key].append(metric_entry)

        # Save metadata
        self._save_metadata()

        return True

    def log_artifact(self, artifact_path, description=None):
        """
        Log an artifact for the current run.

        Args:
            artifact_path: Path to artifact file
            description: Optional artifact description

        Returns:
            bool: Success flag
        """
        if self.current_run is None:
            print("No active run for logging artifacts")
            return False

        # Copy artifact to run directory
        artifact_file = Path(artifact_path)
        if not artifact_file.exists():
            print(f"Artifact file not found: {artifact_path}")
            return False

        # Calculate checksum
        checksum = self._calculate_file_hash(artifact_file)

        # Define artifact metadata
        artifact = {
            "name": artifact_file.name,
            "path": str(artifact_file),
            "description": description,
            "added_at": datetime.datetime.now().isoformat(),
            "checksum": checksum
        }

        # Add to run metadata
        self.current_run["artifacts"].append(artifact)

        # Save metadata
        self._save_metadata()

        return True

    def log_message(self, message, level="INFO"):
        """
        Log a message for the current run.

        Args:
            message: Log message
            level: Log level (INFO, WARNING, ERROR)

        Returns:
            bool: Success flag
        """
        if self.current_run is None:
            print("No active run for logging messages")
            return False

        # Create log entry
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "level": level,
            "message": message
        }

        # Add to run logs
        self.current_run["logs"].append(log_entry)

        # Save to log file
        run_log_file = self.experiment_dir / "logs" / f"{self.current_run['run_name']}.log"
        with open(run_log_file, "a") as f:
            f.write(f"[{log_entry['timestamp']}] {level}: {message}\n")

        # Save metadata
        self._save_metadata()

        return True

    def save_dataset(self, dataframe, name, description=None, version=None):
        """
        Save a dataset with versioning.

        Args:
            dataframe: Pandas DataFrame
            name: Dataset name
            description: Optional description
            version: Optional version string

        Returns:
            str: Dataset version
        """
        # Generate version if not provided
        if version is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            version = f"v_{timestamp}"

        # Create dataset directory
        dataset_dir = self.experiment_dir / "data" / name
        dataset_dir.mkdir(exist_ok=True)

        # Save dataset
        file_path = dataset_dir / f"{version}.csv"
        dataframe.to_csv(file_path, index=False)

        # Calculate checksum
        checksum = self._calculate_file_hash(file_path)

        # Create dataset metadata
        dataset_meta = {
            "name": name,
            "version": version,
            "description": description,
            "created_at": datetime.datetime.now().isoformat(),
            "shape": list(dataframe.shape),
            "columns": list(dataframe.columns),
            "path": str(file_path),
            "checksum": checksum
        }

        # Update experiment metadata
        if name not in self.metadata["datasets"]:
            self.metadata["datasets"][name] = []

        self.metadata["datasets"][name].append(dataset_meta)

        # Save metadata
        self._save_metadata()

        # Add to current run if active
        if self.current_run is not None:
            self.log_artifact(file_path, f"Dataset: {name} version: {version}")

        return version

    def load_dataset(self, name, version=None):
        """
        Load a dataset by name and version.

        Args:
            name: Dataset name
            version: Dataset version (latest if None)

        Returns:
            pandas.DataFrame: Loaded dataset
        """
        # Check if dataset exists
        if name not in self.metadata["datasets"]:
            print(f"Dataset not found: {name}")
            return None

        # Get versions
        versions = self.metadata["datasets"][name]

        # Use latest version if not specified
        if version is None:
            version_meta = versions[-1]
        else:
            # Find requested version
            version_meta = next((v for v in versions if v["version"] == version), None)
            if version_meta is None:
                print(f"Version not found: {version} for dataset: {name}")
                return None

        # Load dataset
        file_path = version_meta["path"]

        # Verify checksum
        current_checksum = self._calculate_file_hash(file_path)
        if current_checksum != version_meta["checksum"]:
            print(f"Warning: Checksum mismatch for dataset: {name} version: {version_meta['version']}")

        # Load dataframe
        return pd.read_csv(file_path)

    def save_model(self, model_path, name, description=None, version=None, metrics=None):
        """
        Save a model with versioning.

        Args:
            model_path: Path to model file
            name: Model name
            description: Optional description
            version: Optional version string
            metrics: Optional performance metrics

        Returns:
            str: Model version
        """
        # Generate version if not provided
        if version is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            version = f"v_{timestamp}"

        # Create model directory
        model_dir = self.experiment_dir / "models" / name
        model_dir.mkdir(exist_ok=True)

        # Copy model file
        src_path = Path(model_path)
        if not src_path.exists():
            print(f"Model file not found: {model_path}")
            return None

        dst_path = model_dir / f"{version}{src_path.suffix}"
        import shutil
        shutil.copy2(src_path, dst_path)

        # Calculate checksum
        checksum = self._calculate_file_hash(dst_path)

        # Create model metadata
        model_meta = {
            "name": name,
            "version": version,
            "description": description,
            "created_at": datetime.datetime.now().isoformat(),
            "path": str(dst_path),
            "metrics": metrics or {},
            "checksum": checksum
        }

        # Update experiment metadata
        if name not in self.metadata["models"]:
            self.metadata["models"][name] = []

        self.metadata["models"][name].append(model_meta)

        # Save metadata
        self._save_metadata()

        # Add to current run if active
        if self.current_run is not None:
            self.log_artifact(dst_path, f"Model: {name} version: {version}")

        return version

    def get_model_path(self, name, version=None):
        """
        Get path to a model by name and version.

        Args:
            name: Model name
            version: Model version (latest if None)

        Returns:
            str: Path to model file
        """
        # Check if model exists
        if name not in self.metadata["models"]:
            print(f"Model not found: {name}")
            return None

        # Get versions
        versions = self.metadata["models"][name]

        # Use latest version if not specified
        if version is None:
            version_meta = versions[-1]
        else:
            # Find requested version
            version_meta = next((v for v in versions if v["version"] == version), None)
            if version_meta is None:
                print(f"Version not found: {version} for model: {name}")
                return None

        # Return model path
        return version_meta["path"]

    def save_result(self, result_data, name, description=None, version=None):
        """
        Save an experiment result.

        Args:
            result_data: Result data (dict)
            name: Result name
            description: Optional description
            version: Optional version string

        Returns:
            str: Result version
        """
        # Generate version if not provided
        if version is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            version = f"v_{timestamp}"

        # Create result directory
        result_dir = self.experiment_dir / "results" / name
        result_dir.mkdir(exist_ok=True)

        # Save result
        file_path = result_dir / f"{version}.json"
        with open(file_path, "w") as f:
            json.dump(result_data, f, indent=2)

        # Create result metadata
        result_meta = {
            "name": name,
            "version": version,
            "description": description,
            "created_at": datetime.datetime.now().isoformat(),
            "path": str(file_path)
        }

        # Update experiment metadata
        if name not in self.metadata["results"]:
            self.metadata["results"][name] = []

        self.metadata["results"][name].append(result_meta)

        # Save metadata
        self._save_metadata()

        # Add to current run if active
        if self.current_run is not None:
            self.log_artifact(file_path, f"Result: {name} version: {version}")

        return version

    def load_result(self, name, version=None):
        """
        Load a result by name and version.

        Args:
            name: Result name
            version: Result version (latest if None)

        Returns:
            dict: Result data
        """
        # Check if result exists
        if name not in self.metadata["results"]:
            print(f"Result not found: {name}")
            return None

        # Get versions
        versions = self.metadata["results"][name]

        # Use latest version if not specified
        if version is None:
            version_meta = versions[-1]
        else:
            # Find requested version
            version_meta = next((v for v in versions if v["version"] == version), None)
            if version_meta is None:
                print(f"Version not found: {version} for result: {name}")
                return None

        # Load result
        file_path = version_meta["path"]
        with open(file_path, "r") as f:
            return json.load(f)

    def get_run_summary(self, run_name=None):
        """
        Get summary of a run.

        Args:
            run_name: Run name (current run if None)

        Returns:
            dict: Run summary
        """
        if run_name is None:
            if self.current_run is None:
                print("No active run")
                return None
            run_meta = self.current_run
        else:
            # Find run by name
            run_meta = next((r for r in self.metadata["runs"] if r["run_name"] == run_name), None)
            if run_meta is None:
                print(f"Run not found: {run_name}")
                return None

        # Create summary
        summary = {
            "run_name": run_meta["run_name"],
            "status": run_meta["status"],
            "started_at": run_meta["started_at"],
            "completed_at": run_meta["completed_at"],
            "description": run_meta["description"],
            "params": run_meta["params"],
            "metrics": {},
            "artifacts_count": len(run_meta["artifacts"]),
            "logs_count": len(run_meta["logs"])
        }

        # Summarize metrics
        for key, values in run_meta["metrics"].items():
            if values:
                summary["metrics"][key] = {
                    "last": values[-1]["value"],
                    "min": min(v["value"] for v in values),
                    "max": max(v["value"] for v in values),
                    "mean": sum(v["value"] for v in values) / len(values)
                }

        return summary

    def compare_runs(self, run_names, metrics=None):
        """
        Compare multiple runs.

        Args:
            run_names: List of run names
            metrics: Optional list of metrics to compare

        Returns:
            pandas.DataFrame: Comparison table
        """
        # Find runs
        run_metas = []
        for name in run_names:
            run = next((r for r in self.metadata["runs"] if r["run_name"] == name), None)
            if run is None:
                print(f"Run not found: {name}")
                continue
            run_metas.append(run)

        if not run_metas:
            print("No valid runs to compare")
            return None

        # Collect all metrics if not specified
        if metrics is None:
            metrics = set()
            for run in run_metas:
                metrics.update(run["metrics"].keys())
            metrics = sorted(metrics)

        # Build comparison data
        data = []
        for run in run_metas:
            row = {
                "run_name": run["run_name"],
                "status": run["status"],
                "started_at": run["started_at"]
            }

            # Add metrics
            for metric in metrics:
                if metric in run["metrics"] and run["metrics"][metric]:
                    values = [v["value"] for v in run["metrics"][metric]]
                    row[f"{metric}_last"] = values[-1]
                    row[f"{metric}_min"] = min(values)
                    row[f"{metric}_max"] = max(values)
                    row[f"{metric}_mean"] = sum(values) / len(values)
                else:
                    row[f"{metric}_last"] = None
                    row[f"{metric}_min"] = None
                    row[f"{metric}_max"] = None
                    row[f"{metric}_mean"] = None

            data.append(row)

        return pd.DataFrame(data)

    def _save_metadata(self):
        """Save experiment metadata to file."""
        metadata_file = self.experiment_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            # Convert to serializable format
            metadata = {k: v for k, v in self.metadata.items()}
            json.dump(metadata, f, indent=2)

    def _calculate_file_hash(self, file_path):
        """Calculate SHA-256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            # Read in chunks to handle large files
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()