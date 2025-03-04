import numpy as np
class ComplexityPenalizer:
    """Implements complexity penalty for causal graph optimization."""

    def __init__(self, lambda_param=0.1):
        """
        Initialize complexity penalizer.

        Args:
            lambda_param: Weight of complexity penalty
        """
        self.lambda_param = lambda_param

    def calculate_penalty(self, graph):
        """
        Calculate complexity penalty for a given graph.

        Args:
            graph: NetworkX DiGraph

        Returns:
            float: Penalty value
        """
        # Calculate based on edge density and node degree distribution
        n_nodes = graph.number_of_nodes()
        if n_nodes <= 1:
            return 0

        # Edge density component
        max_edges = n_nodes * (n_nodes - 1)
        edge_density = graph.number_of_edges() / max_edges if max_edges > 0 else 0

        # Node degree entropy component
        degrees = [d for _, d in graph.degree()]
        degree_sum = sum(degrees)
        if degree_sum == 0:
            degree_entropy = 0
        else:
            # Normalized degree distribution
            degree_probs = [d / degree_sum for d in degrees if d > 0]
            # Shannon entropy of degree distribution
            degree_entropy = -sum(p * np.log(p) for p in degree_probs)

        # Combined penalty (higher entropy and density = higher complexity)
        penalty = self.lambda_param * (edge_density + 0.5 * degree_entropy)

        return penalty

    def apply_penalty(self, graph_posterior):
        """
        Apply complexity penalty to graph posterior probability.

        Args:
            graph_posterior: Original posterior probability

        Returns:
            float: Penalized posterior probability
        """
        penalty = self.calculate_penalty(graph_posterior['graph'])
        return graph_posterior['probability'] * np.exp(-penalty)
