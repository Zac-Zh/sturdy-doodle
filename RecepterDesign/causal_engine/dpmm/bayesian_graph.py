# causal_engine/dpmm/bayesian_graph.py
import numpy as np
import networkx as nx
from scipy.stats import dirichlet
import pymc3 as pm
from causal_engine.dpmm.complexity_penalty import ComplexityPenalizer

# causal_engine/dpmm/bayesian_graph.py
import numpy as np
import networkx as nx
from causal_engine.dpmm.complexity_penalty import ComplexityPenalizer


class CausalGraphOptimizer:
    """Implements dynamic causal graph structure optimization using Bayesian updates."""

    def __init__(self, prior_graph=None, alpha=1.0, lambda_complexity=0.1):
        """
        Initialize the causal graph optimizer.

        Args:
            prior_graph: NetworkX DiGraph with expert knowledge
            alpha: Concentration parameter for prior strength
            lambda_complexity: Penalty coefficient for graph complexity
        """
        self.prior_graph = prior_graph if prior_graph else nx.DiGraph()
        self.current_graph = self.prior_graph.copy()
        self.alpha = alpha
        self.lambda_complexity = lambda_complexity
        self.edge_confidence = {}
        self.complexity_penalizer = ComplexityPenalizer(lambda_complexity)

        # Initialize edge confidences based on expert knowledge
        for u, v, data in self.prior_graph.edges(data=True):
            self.edge_confidence[(u, v)] = data.get('confidence', 0.5)

    def _calculate_complexity(self, graph):
        """Calculate graph complexity as a function of edges and nodes."""
        return self.complexity_penalizer.calculate(graph)

    def _likelihood(self, graph, data):
        """
        Calculate likelihood of data given graph structure.
        
        Args:
            graph: NetworkX DiGraph representing causal structure
            data: List of tuples (source_node, target_node, effect_strength)
            
        Returns:
            float: Likelihood score
        """
        log_likelihood = 0

        # For each causal relationship in the data
        for source, target, effect_strength in data:
            if graph.has_edge(source, target):
                # If the edge exists, likelihood depends on effect strength
                # Higher effect strength gives higher likelihood
                edge_conf = graph[source][target].get('confidence', 0.5)
                log_likelihood += np.log(edge_conf * (0.5 + abs(effect_strength)/2))
            else:
                # If edge doesn't exist but we observe an effect, penalize based on strength
                log_likelihood += np.log(0.1 + 0.1 * (1 - abs(effect_strength)))

        return np.exp(log_likelihood)

    def update_with_experiment(self, exp_data):
        """
        Update causal graph with new experimental data.

        Args:
            exp_data: List of tuples (source_node, target_node, effect_strength)
            
        Returns:
            NetworkX DiGraph: Updated causal graph
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
        if np.sum(posterior_probs) > 0:  # Avoid division by zero
            posterior_probs = posterior_probs / np.sum(posterior_probs)
            
            # Select graph with highest posterior probability
            best_graph_idx = np.argmax(posterior_probs)
            self.current_graph = possible_graphs[best_graph_idx]

        # Update edge confidences
        self._update_edge_confidences(exp_data)

        return self.current_graph

    def _generate_candidate_graphs(self):
        """
        Generate candidate graph structures by adding/removing edges.
        
        Returns:
            list: List of candidate NetworkX DiGraph objects
        """
        candidates = [self.current_graph.copy()]

        # Add potential new edges
        for node1 in self.current_graph.nodes():
            for node2 in self.current_graph.nodes():
                if node1 != node2 and not self.current_graph.has_edge(node1, node2):
                    # Skip if this would create a cycle
                    if not self._would_create_cycle(node1, node2):
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
    
    def _would_create_cycle(self, source, target):
        """
        Check if adding an edge would create a cycle in the graph.
        
        Args:
            source: Source node
            target: Target node
            
        Returns:
            bool: True if adding edge would create cycle, False otherwise
        """
        # If there's already a path from target to source, adding source->target creates a cycle
        return nx.has_path(self.current_graph, target, source)

    def _calculate_prior(self, graph):
        """
        Calculate prior probability of graph based on expert knowledge.
        
        Args:
            graph: NetworkX DiGraph to calculate prior for
            
        Returns:
            float: Prior probability
        """
        prior = 1.0

        # Compare with prior graph (expert knowledge)
        for u, v in self.prior_graph.edges():
            confidence = self.edge_confidence.get((u, v), 0.5)
            if graph.has_edge(u, v):
                prior *= confidence
            else:
                prior *= (1 - confidence)

        # Add prior for non-expert edges (discourage spurious edges)
        alpha_penalty = self.alpha * 0.1  # Parameter controlling how much to penalize extra edges
        for u, v in graph.edges():
            if not self.prior_graph.has_edge(u, v):
                prior *= alpha_penalty

        return prior

    def _update_edge_confidences(self, exp_data):
        """
        Update edge confidence levels based on experimental data.
        
        Args:
            exp_data: List of tuples (source_node, target_node, effect_strength)
        """
        for source, target, effect_strength in exp_data:
            if self.current_graph.has_edge(source, target):
                # Increase confidence based on effect strength
                current_conf = self.edge_confidence.get((source, target), 0.5)
                # Bayesian update of confidence
                new_conf = (current_conf + abs(effect_strength)) / (1 + abs(effect_strength))
                self.edge_confidence[(source, target)] = new_conf

                # Update edge attribute in graph
                self.current_graph[source][target]['confidence'] = new_conf

    def get_edge_confidences(self):
        """
        Return dictionary of all edge confidences.
        
        Returns:
            dict: Dictionary mapping edge tuples to confidence values
        """
        return self.edge_confidence
