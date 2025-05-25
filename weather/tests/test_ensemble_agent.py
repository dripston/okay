import unittest
import numpy as np
import pandas as pd
from agents.ensemble_agent import MetaEnsembleAgent, HierarchicalEnsemble

class TestMetaEnsembleAgent(unittest.TestCase):
    def setUp(self):
        self.agent = MetaEnsembleAgent()
        self.sample_data = self._generate_sample_data()
        
    def _generate_sample_data(self):
        dates = pd.date_range(start='2023-01-01', periods=60, freq='D')
        return pd.DataFrame({
            'tavg': np.random.normal(25, 5, 60),
            'tmax': np.random.normal(30, 5, 60),
            'tmin': np.random.normal(20, 5, 60),
            'prcp': np.random.exponential(1, 60),
            'wspd': np.random.normal(10, 3, 60),
            'pres': np.random.normal(1013, 5, 60)
        }, index=dates)
    
    def test_model_initialization(self):
        self.assertIsNotNone(self.agent.base_models)
        self.assertIsNotNone(self.agent.meta_learner)
        self.assertIsNotNone(self.agent.uncertainty_estimator)
    
    def test_prediction_shape(self):
        self.agent.train(self.sample_data, self.sample_data['tavg'])
        pred = self.agent.predict(self.sample_data)
        self.assertIsInstance(pred, dict)
        self.assertIn('prediction', pred)
        self.assertIn('confidence', pred)
        
    def test_uncertainty_estimation(self):
        self.agent.train(self.sample_data, self.sample_data['tavg'])
        pred = self.agent.predict(self.sample_data)
        self.assertGreater(pred['confidence'], 0)
        self.assertLess(pred['confidence'], 1)