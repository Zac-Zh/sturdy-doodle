import os
import networkx as nx
import numpy as np
import json
import datetime
from collections import defaultdict

class CausalGraphVersionControl:
    """Version control for causal graph structures."""

    def __init__(self, base_dir="causal_graphs", initial_graph=None):
        """
        Initialize causal graph version control.

        Args:
            base_dir: Directory to store graph versions
            initial_graph: Optional initial graph
        """
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

        # Current graph
        self.current_graph = initial_graph if initial_graph else nx.DiGraph()

        # Version history
        self.versions = []
        self.current_version = 0

        # Data statistics by receptor family
        self.data_counts = defaultdict(int)

        # Save initial version if provided
        if initial_graph:
            self.commit("Initial graph structure")

    def commit(self, message):
        """
        Commit current graph state.

        Args:
            message: Commit message

        Returns:
            int: Version number
        """
        # Create version metadata
        timestamp = datetime.datetime.now().isoformat()
        version = {
            'version': self.current_version + 1,
            'timestamp': timestamp,
            'message': message,
            'node_count': self.current_graph.number_of_nodes(),
            'edge_count': self.current_graph.number_of_edges()
        }

        # Save graph to file
        graph_file = os.path.join(self.base_dir, f"graph_v{version['version']}.json")
        self._save_graph(graph_file)

        # Save version metadata
        self.versions.append(version)
        self.current_version += 1

        # Save version history
        history_file = os.path.join(self.base_dir, "version_history.json")
        with open(history_file, 'w') as f:
            json.dump(self.versions, f, indent=2)

        return self.current_version

    def checkout(self, version):
        """
        Checkout a specific graph version.

        Args:
            version: Version number

        Returns:
            bool: Success flag
        """
        if version < 1 or version > self.current_version:
            print(f"Error: Version {version} does not exist")
            return False

        # Load graph from file
        graph_file = os.path.join(self.base_dir, f"graph_v{version}.json")
        self.current_graph = self._load_graph(graph_file)

        print(f"Checked out version {version}: {self.versions[version - 1]['message']}")
        return True

    def add_evidence(self, mutation, phenotype, receptor_family, confidence=0.5):
        """
        Add experimental evidence to graph.

        Args:
            mutation: Mutation description
            phenotype: Observed phenotype
            receptor_family: Receptor family
            confidence: Confidence in evidence

        Returns:
            bool: Whether graph was updated
        """
        # Add nodes if they don't exist
        if mutation not in self.current_graph:
            self.current_graph.add_node(mutation, type='mutation')

        if phenotype not in self.current_graph:
            self.current_graph.add_node(phenotype, type='phenotype')

        # Update edge
        if self.current_graph.has_edge(mutation, phenotype):
            # Update existing edge
            edge_data = self.current_graph[mutation][phenotype]
            old_confidence = edge_data.get('confidence', 0.5)
            old_evidence = edge_data.get('evidence_count', 0)

            # Update with new evidence (weighted average)
            new_confidence = (old_confidence * old_evidence + confidence) / (old_evidence + 1)
            edge_data['confidence'] = new_confidence
            edge_data['evidence_count'] = old_evidence + 1
            edge_data['last_update'] = datetime.datetime.now().isoformat()

            was_updated = abs(new_confidence - old_confidence) > 0.1
        else:
            # Add new edge
            self.current_graph.add_edge(mutation, phenotype,
                                        confidence=confidence,
                                        evidence_count=1,
                                        receptor_family=receptor_family,
                                        last_update=datetime.datetime.now().isoformat())
            was_updated = True

        # Update data count for this receptor family
        self.data_counts[receptor_family] += 1

        # Check if we should create a new branch for this family
        if self.data_counts[receptor_family] >= 100 and was_updated:
            self._create_family_branch(receptor_family)

        return was_updated

    def _create_family_branch(self, receptor_family):
        """Create a new branch for a receptor family."""
        # Commit current state
        self.commit(f"Pre-branch state for {receptor_family}")

        # Extract subgraph for this family
        family_edges = [(u, v) for u, v, d in self.current_graph.edges(data=True)
                        if d.get('receptor_family') == receptor_family]
        family_nodes = set()
        for u, v in family_edges:
            family_nodes.add(u)
            family_nodes.add(v)

        family_subgraph = self.current_graph.subgraph(family_nodes).copy()

        # Save as a specialized branch
        branch_dir = os.path.join(self.base_dir, f"branch_{receptor_family}")
        os.makedirs(branch_dir, exist_ok=True)

        # Save initial branch state
        branch_file = os.path.join(branch_dir, "graph_v1.json")
        self._save_graph_object(family_subgraph, branch_file)

        # Create branch metadata
        branch_meta = {
            'name': receptor_family,
            'created_from_version': self.current_version,
            'creation_date': datetime.datetime.now().isoformat(),
            'node_count': family_subgraph.number_of_nodes(),
            'edge_count': family_subgraph.number_of_edges()
        }

        # Save branch metadata
        meta_file = os.path.join(branch_dir, "branch_info.json")
        with open(meta_file, 'w') as f:
            json.dump(branch_meta, f, indent=2)

        print(f"Created new branch for receptor family: {receptor_family}")

    def _save_graph(self, filename):
        """Save current graph to file."""
        self._save_graph_object(self.current_graph, filename)

    def _save_graph_object(self, graph, filename):
        """Save a graph object to file."""
        # Convert to serializable format
        data = {
            'nodes': [],
            'edges': []
        }

        # Add nodes with attributes
        for node, attrs in graph.nodes(data=True):
            node_data = {'id': node}
            node_data.update(attrs)
            data['nodes'].append(node_data)

        # Add edges with attributes
        for src, dst, attrs in graph.edges(data=True):
            edge_data = {'source': src, 'target': dst}
            edge_data.update(attrs)
            data['edges'].append(edge_data)

        # Save to file
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

    def _load_graph(self, filename):
        """Load graph from file."""
        with open(filename, 'r') as f:
            data = json.load(f)

        # Create new graph
        graph = nx.DiGraph()

        # Add nodes
        for node_data in data['nodes']:
            node_id = node_data.pop('id')
            graph.add_node(node_id, **node_data)

        # Add edges
        for edge_data in data['edges']:
            source = edge_data.pop('source')
            target = edge_data.pop('target')
            graph.add_edge(source, target, **edge_data)

        return graph

    def get_version_history(self):
        """Get version history."""
        return self.versions

    def get_graph_summary(self):
        """Get summary of current graph."""
        # Node types
        node_types = {}
        for node, attrs in self.current_graph.nodes(data=True):
            node_type = attrs.get('type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1

        # Edge statistics
        edge_count = self.current_graph.number_of_edges()

        # Confidence distribution
        confidences = [d.get('confidence', 0) for _, _, d in self.current_graph.edges(data=True)]
        avg_confidence = np.mean(confidences) if confidences else 0

        # Receptor family distribution
        families = {}
        for _, _, d in self.current_graph.edges(data=True):
            family = d.get('receptor_family', 'unknown')
            families[family] = families.get(family, 0) + 1

        return {
            'version': self.current_version,
            'node_count': self.current_graph.number_of_nodes(),
            'node_types': node_types,
            'edge_count': edge_count,
            'avg_confidence': avg_confidence,
            'receptor_families': families,
            'data_counts': dict(self.data_counts)
        }

    def merge_branch(self, receptor_family):
        """
        Merge a receptor family branch into the main graph.

        Args:
            receptor_family: Receptor family to merge

        Returns:
            bool: Success flag
        """
        branch_dir = os.path.join(self.base_dir, f"branch_{receptor_family}")

        # Check if branch exists
        if not os.path.exists(branch_dir):
            print(f"Error: Branch for {receptor_family} does not exist")
            return False

        # Get latest version in branch
        versions = [f for f in os.listdir(branch_dir) if f.startswith("graph_v")]
        if not versions:
            print(f"Error: No versions found in branch {receptor_family}")
            return False

        versions.sort(key=lambda x: int(x.split("_v")[1].split(".")[0]))
        latest = versions[-1]

        # Load branch graph
        branch_file = os.path.join(branch_dir, latest)
        branch_graph = self._load_graph(branch_file)

        # Merge into main graph
        for node, attrs in branch_graph.nodes(data=True):
            if node not in self.current_graph:
                self.current_graph.add_node(node, **attrs)

        for src, dst, attrs in branch_graph.edges(data=True):
            if not self.current_graph.has_edge(src, dst):
                self.current_graph.add_edge(src, dst, **attrs)
            else:
                # Update with branch data if confidence is higher
                main_conf = self.current_graph[src][dst].get('confidence', 0)
                branch_conf = attrs.get('confidence', 0)

                if branch_conf > main_conf:
                    self.current_graph[src][dst].update(attrs)

        # Commit merged state
        self.commit(f"Merged branch {receptor_family}")
        return True