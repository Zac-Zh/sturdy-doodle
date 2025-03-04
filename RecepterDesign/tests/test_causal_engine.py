import unittest
import networkx as nx
from causal_engine.dpmm.bayesian_graph import CausalGraphOptimizer
from causal_engine.dpmm.complexity_penalty import ComplexityPenalizer

class TestCausalGraphOptimizer(unittest.TestCase):
    def setUp(self):
        self.prior_graph = nx.DiGraph()
        self.prior_graph.add_edge('A', 'B', confidence=0.7)
        self.prior_graph.add_edge('B', 'C', confidence=0.6)
        self.optimizer = CausalGraphOptimizer(prior_graph=self.prior_graph)

    def test_initialization(self):
        self.assertEqual(self.optimizer.current_graph.number_of_edges(), 2)
        self.assertEqual(self.optimizer.edge_confidence[('A', 'B')], 0.7)
        self.assertEqual(self.optimizer.edge_confidence[('B', 'C')], 0.6)

    def test_update_with_experiment(self):
        exp_data = [('A', 'B', 0.8), ('B', 'C', 0.7), ('A', 'C', 0.5)]
        updated_graph = self.optimizer.update_with_experiment(exp_data)
        self.assertGreater(updated_graph.number_of_edges(), 2)
        self.assertGreater(self.optimizer.edge_confidence[('A', 'B')], 0.7)

    def test_complexity_penalty(self):
        penalizer = ComplexityPenalizer()
        penalty = penalizer.calculate_penalty(self.prior_graph)
        self.assertGreater(penalty, 0)

if __name__ == '__main__':
    unittest.main()