# causal_engine/dpmm/bayesian_graph.py
import numpy as np
import networkx as nx
from scipy.stats import dirichlet
import pymc3 as pm
from causal_engine.dpmm.complexity_penalty import ComplexityPenalizer


class CausalGraphOptimizer:
    """Implements dynamic causal graph structure optimization using Dirichlet Process Mixture Model."""

    def __init__(self, prior_graph=None, alpha=1.0, lambda_complexity=0.1):
        """
        Initialize the causal graph optimizer.

        Args:
            prior_graph: NetworkX DiGraph with expert knowledge
            alpha: Concentration parameter for Dirichlet process
            lambda_complexity: Penalty coefficient for graph complexity
        """
        self.prior_graph = prior_graph if prior_graph else nx.DiGraph()
        self.current_graph = self.prior_graph.copy()
        self.alpha = alpha
        self.lambda_complexity = lambda_complexity
        self.edge_confidence = {}

        # Initialize edge confidences based on expert knowledge
        for u, v, data in self.prior_graph.edges(data=True):
            self.edge_confidence[(u, v)] = data.get('confidence', 0.5)

    def _calculate_complexity(self, graph):
        """Calculate graph complexity as a function of edges and nodes."""
        n_nodes = graph.number_of_nodes()
        n_edges = graph.number_of_edges()
        # Penalize dense graphs more heavily
        return n_edges / (n_nodes * (n_nodes - 1) / 2) if n_nodes > 1 else 0

    def _likelihood(self, graph, data):
        """Calculate likelihood of data given graph structure."""
        # Simplified likelihood calculation
        log_likelihood = 0

        # For each causal relationship in the data
        for source, target, effect in data:
            if graph.has_edge(source, target):
                # If the edge exists in our graph, add to likelihood
                log_likelihood += np.log(0.8)  # Simplified probability of observing effect
            else:
                # If edge doesn't exist but we observe an effect, penalize
                log_likelihood += np.log(0.2)

        return np.exp(log_likelihood)

    def update_with_experiment(self, exp_data):
        """
        Update causal graph with new experimental data.

        Args:
            exp_data: List of tuples (source_node, target_node, effect_strength)
        """
        possible_graphs = self._generate_candidate_graphs()
        posterior_probs = []

        for graph in possible_graphs:
            # Calculate posterior following Bayes rule with complexity penalty
            likelihood = self._likelihood(graph, exp_data)
            prior = self._calculate_prior(graph)
            complexity = self._calculate_complexity(graph)

            # P(G|D) ∝ P(D|G) · P(G) · e^(-λ·Complexity(G))
            posterior = likelihood * prior * np.exp(-self.lambda_complexity * complexity)
            posterior_probs.append(posterior)

        # Normalize posterior probabilities
        posterior_probs = np.array(posterior_probs)
        posterior_probs = posterior_probs / np.sum(posterior_probs)

        # Select graph with highest posterior probability
        best_graph_idx = np.argmax(posterior_probs)
        self.current_graph = possible_graphs[best_graph_idx]

        # Update edge confidences
        self._update_edge_confidences(exp_data)

        return self.current_graph

    def _generate_candidate_graphs(self):
        """Generate candidate graph structures by adding/removing edges."""
        candidates = [self.current_graph.copy()]

        # Add potential new edges
        for node1 in self.current_graph.nodes():
            for node2 in self.current_graph.nodes():
                if node1 != node2 and not self.current_graph.has_edge(node1, node2):
                    g = self.current_graph.copy()
                    g.add_edge(node1, node2, confidence=0.1)  # Low initial confidence
                    candidates.append(g)

        # Remove existing edges with low confidence
        for u, v in list(self.current_graph.edges()):
            if self.edge_confidence.get((u, v), 0) < 0.3:
                g = self.current_graph.copy()
                g.remove_edge(u, v)
                candidates.append(g)

        return candidates

    def _calculate_prior(self, graph):
        """Calculate prior probability of graph based on expert knowledge."""
        prior = 1.0

        # Compare with prior graph (expert knowledge)
        for u, v in self.prior_graph.edges():
            confidence = self.edge_confidence.get((u, v), 0.5)
            if graph.has_edge(u, v):
                prior *= confidence
            else:
                prior *= (1 - confidence)

        return prior

    def _update_edge_confidences(self, exp_data):
        """Update edge confidence levels based on experimental data."""
        for source, target, effect_strength in exp_data:
            if self.current_graph.has_edge(source, target):
                # Increase confidence based on effect strength
                current_conf = self.edge_confidence.get((source, target), 0.5)
                # Bayesian update of confidence
                new_conf = (current_conf + effect_strength) / (1 + abs(effect_strength))
                self.edge_confidence[(source, target)] = new_conf

                # Update edge attribute in graph
                self.current_graph[source][target]['confidence'] = new_conf


