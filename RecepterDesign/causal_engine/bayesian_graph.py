"""
Re-export CausalGraphOptimizer from the dpmm submodule
"""

# Simply re-export the class from the actual module location
try:
    from causal_engine.dpmm.bayesian_graph import CausalGraphOptimizer
except ImportError:
    # Fallback implementation if the real one isn't available
    import networkx as nx
    import logging
    
    logger = logging.getLogger(__name__)
    
    class CausalGraphOptimizer:
        """
        Fallback implementation of the CausalGraphOptimizer.
        """
        
        def __init__(self, lambda_complexity=0.1):
            """Initialize the optimizer."""
            self.lambda_complexity = lambda_complexity
            logger.info(f"Using fallback CausalGraphOptimizer with lambda_complexity={lambda_complexity}")
        
        def update_with_experiment(self, exp_data):
            """Mock implementation that returns a simple graph."""
            graph = nx.DiGraph()
            
            # Add some nodes and edges
            graph.add_node("binding_domain")
            graph.add_node("stability")
            graph.add_node("expression") 
            graph.add_node("function")
            
            graph.add_edge("binding_domain", "function", confidence=0.85)
            graph.add_edge("stability", "expression", confidence=0.7)
            graph.add_edge("expression", "function", confidence=0.6)
            
            logger.info(f"Created fallback causal graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
            return graph