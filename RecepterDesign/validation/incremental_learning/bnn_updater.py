import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.nn import PyroModule, PyroSample
from scipy.stats import norm

class BayesianNeuralNetwork(PyroModule):
    """Bayesian Neural Network for uncertainty-aware causal learning."""

    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        """
        Initialize BNN.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            output_dim: Output dimension
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # First layer weights and biases with priors
        self.fc1_weight = PyroSample(
            dist.Normal(0., 1.).expand([hidden_dim, input_dim]).to_event(2)
        )
        self.fc1_bias = PyroSample(
            dist.Normal(0., 1.).expand([hidden_dim]).to_event(1)
        )

        # Second layer weights and biases with priors
        self.fc2_weight = PyroSample(
            dist.Normal(0., 1.).expand([output_dim, hidden_dim]).to_event(2)
        )
        self.fc2_bias = PyroSample(
            dist.Normal(0., 1.).expand([output_dim]).to_event(1)
        )

    def forward(self, x, y=None):
        """
        Forward pass of BNN with uncertainty.

        Args:
            x: Input tensor
            y: Optional target tensor

        Returns:
            dist.Normal: Distribution over outputs
        """
        # First hidden layer
        x = F.linear(x, self.fc1_weight, self.fc1_bias)
        x = F.relu(x)

        # Output layer
        mean = F.linear(x, self.fc2_weight, self.fc2_bias)

        # Observation model
        sigma = pyro.sample("sigma", dist.Gamma(1.0, 1.0))

        # Output distribution
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mean, sigma), obs=y)

        return mean


class CausalBNNUpdater:
    """Bayesian Neural Network-based causal model updater."""

    def __init__(self, num_features, hidden_dim=64, num_samples=1000, learning_rate=0.01):
        """
        Initialize causal BNN updater.

        Args:
            num_features: Number of input features
            hidden_dim: Hidden dimension
            num_samples: Number of posterior samples
            learning_rate: Learning rate for SVI
        """
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_samples = num_samples
        self.learning_rate = learning_rate

        # Initialize BNN
        self.model = BayesianNeuralNetwork(num_features, hidden_dim)

        # Setup optimizer and inference
        self.optimizer = optim.Adam({"lr": learning_rate})
        self.svi = SVI(self.model, self._guide, self.optimizer, loss=Trace_ELBO())

        # Storage for learned causal relationships
        self.causal_effects = {}
        self.effect_uncertainties = {}

    def _guide(self, x, y=None):
        """Variational guide for Bayesian inference."""
        # Layer 1 variational parameters
        fc1w_loc = pyro.param("fc1w_loc", torch.randn(self.hidden_dim, self.num_features))
        fc1w_scale = pyro.param("fc1w_scale", torch.ones(self.hidden_dim, self.num_features),
                                constraint=dist.constraints.positive)
        fc1b_loc = pyro.param("fc1b_loc", torch.randn(self.hidden_dim))
        fc1b_scale = pyro.param("fc1b_scale", torch.ones(self.hidden_dim),
                                constraint=dist.constraints.positive)

        # Layer 2 variational parameters
        fc2w_loc = pyro.param("fc2w_loc", torch.randn(1, self.hidden_dim))
        fc2w_scale = pyro.param("fc2w_scale", torch.ones(1, self.hidden_dim),
                                constraint=dist.constraints.positive)
        fc2b_loc = pyro.param("fc2b_loc", torch.randn(1))
        fc2b_scale = pyro.param("fc2b_scale", torch.ones(1),
                                constraint=dist.constraints.positive)

        # Sample variational parameters
        pyro.sample("fc1_weight", dist.Normal(fc1w_loc, fc1w_scale).to_event(2))
        pyro.sample("fc1_bias", dist.Normal(fc1b_loc, fc1b_scale).to_event(1))
        pyro.sample("fc2_weight", dist.Normal(fc2w_loc, fc2w_scale).to_event(2))
        pyro.sample("fc2_bias", dist.Normal(fc2b_loc, fc2b_scale).to_event(1))

        # Observation noise
        sigma_loc = pyro.param("sigma_loc", torch.tensor(1.0),
                               constraint=dist.constraints.positive)
        pyro.sample("sigma", dist.Delta(sigma_loc))

    def update_with_experiment(self, feature_vector, target, experiment_id, num_epochs=1000):
        """
        Update causal model with experimental data.

        Args:
            feature_vector: Input feature vector
            target: Target outcome
            experiment_id: Unique experiment identifier
            num_epochs: Number of training epochs

        Returns:
            dict: Update statistics
        """
        # Convert inputs to tensors
        x = torch.tensor(feature_vector, dtype=torch.float32)
        y = torch.tensor(target, dtype=torch.float32)

        # Train BNN
        losses = []
        for epoch in range(num_epochs):
            loss = self.svi.step(x, y)
            losses.append(loss)

            # Early stopping if converged
            if epoch > 100 and abs(losses[-1] - losses[-2]) < 1e-5:
                break

        # Get posterior predictive distribution
        posterior_samples = []
        for _ in range(self.num_samples):
            # Forward pass to get prediction
            pred = self.model(x)
            posterior_samples.append(pred.detach().numpy())

        posterior_samples = np.array(posterior_samples)

        # Calculate mean effect and uncertainty
        mean_effect = np.mean(posterior_samples)
        effect_std = np.std(posterior_samples)

        # Compute p-value (simplified)
        p_value = 2 * (1 - norm.cdf(abs(mean_effect) / effect_std))

        # Store results
        self.causal_effects[experiment_id] = mean_effect
        self.effect_uncertainties[experiment_id] = effect_std

        # Only update causal edge if effect is significant
        is_significant = p_value < 0.01

        return {
            'experiment_id': experiment_id,
            'mean_effect': mean_effect,
            'effect_std': effect_std,
            'p_value': p_value,
            'is_significant': is_significant,
            'training_loss': losses[-1]
        }

    def predict(self, feature_vector, num_samples=100):
        """
        Make prediction with uncertainty.

        Args:
            feature_vector: Input feature vector
            num_samples: Number of samples for uncertainty estimation

        Returns:
            tuple: (Mean prediction, Standard deviation)
        """
        x = torch.tensor(feature_vector, dtype=torch.float32)

        # Generate samples from posterior
        predictions = []
        for _ in range(num_samples):
            pred = self.model(x)
            predictions.append(pred.detach().numpy())

        predictions = np.array(predictions)

        # Return mean and standard deviation
        return np.mean(predictions, axis=0), np.std(predictions, axis=0)
