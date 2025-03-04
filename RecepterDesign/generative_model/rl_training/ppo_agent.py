import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque


class TrajectoryBuffer:
    """Buffer for storing trajectories."""

    def __init__(self, gamma=0.99):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.returns = []
        self.advantages = []
        self.gamma = gamma
        self.lam = 0.95  # GAE-Lambda parameter

    def add(self, state, action, reward, value, log_prob):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)

    def compute_returns_and_advantages(self, last_value):
        # Compute GAE
        next_value = last_value
        next_advantage = 0
        for i in reversed(range(len(self.rewards))):
            delta = self.rewards[i] + self.gamma * next_value - self.values[i]
            next_advantage = delta + self.gamma * self.lam * next_advantage
            next_value = self.values[i]
            self.advantages.insert(0, next_advantage)
            self.returns.insert(0, next_advantage + self.values[i])

        # Normalize advantages
        self.advantages = torch.tensor(self.advantages)
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get_batch(self, batch_size=None):
        batch_size = batch_size or len(self.states)
        indices = np.random.permutation(len(self.states))[:batch_size]
        return (
            [self.states[i] for i in indices],
            [self.actions[i] for i in indices],
            torch.tensor([self.returns[i] for i in indices]),
            torch.tensor([self.advantages[i] for i in indices]),
            torch.tensor([self.log_probs[i] for i in indices])
        )

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.returns.clear()
        self.advantages.clear()


class PPOAgent:
    """Proximal Policy Optimization agent for training GAN generators."""

    def __init__(self, generator, discriminator, lr=3e-4, gamma=0.99, clip_ratio=0.2,
                 target_kl=0.01, value_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5,
                 update_epochs=4, mini_batch_size=64):
        """
        Initialize PPO agent.

        Args:
            generator: Generator model
            discriminator: Discriminator model
            lr: Learning rate
            gamma: Discount factor
            clip_ratio: PPO clip ratio
            target_kl: Target KL divergence
            value_coef: Value loss coefficient
            entropy_coef: Entropy coefficient
            max_grad_norm: Maximum gradient norm
            update_epochs: Number of PPO update epochs
            mini_batch_size: Mini-batch size for updates
        """
        self.generator = generator
        self.discriminator = discriminator
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.mini_batch_size = mini_batch_size

        # Value function (critic)
        self.value_net = nn.Sequential(
            nn.Linear(generator.d_model, generator.d_model // 2),
            nn.ReLU(),
            nn.Linear(generator.d_model // 2, 1)
        )

        # Create optimizers
        self.generator_optimizer = optim.Adam(generator.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)

        # Buffer for storing trajectories
        self.buffer = TrajectoryBuffer(gamma)

    def select_action(self, state, deterministic=False):
        """
        Select action based on current state.

        Args:
            state: Current state
            deterministic: Whether to select deterministically

        Returns:
            tuple: (Action, Log probability, Value)
        """
        with torch.no_grad():
            # Get action distribution from generator
            atom_logits, positions, bonds, torsion_angles = self.generator(
                state['template_atoms'],
                state['template_positions'],
                state['target_function']
            )

            # Calculate action probabilities
            atom_probs = F.softmax(atom_logits, dim=-1)

            # Calculate value (critic)
            value = self.value_net(state['target_function']).squeeze(-1)

            if deterministic:
                # Select most likely atom types
                atom_types = torch.argmax(atom_probs, dim=-1)
            else:
                # Sample atom types from distribution
                atom_dist = torch.distributions.Categorical(atom_probs)
                atom_types = atom_dist.sample()

            # Calculate log probability of selected action
            log_prob = self._compute_action_log_prob(atom_probs, atom_types, positions, bonds, torsion_angles)

        return {
            'atom_types': atom_types,
            'positions': positions,
            'bonds': bonds,
            'torsion_angles': torsion_angles
        }, log_prob, value

    def update(self):
        """
        Update policy and value networks using PPO.

        Returns:
            dict: Training statistics
        """
        # Compute returns and advantages
        with torch.no_grad():
            last_value = self.value_net(self.buffer.states[-1]['target_function']).squeeze(-1)
        self.buffer.compute_returns_and_advantages(last_value)

        # PPO update epochs
        for _ in range(self.update_epochs):
            # Process minibatches
            for states, actions, returns, advantages, old_log_probs in self.buffer.get_batch(self.mini_batch_size):
                # Get current action distributions
                atom_logits, positions, bonds, torsion_angles = self.generator(
                    torch.stack([s['template_atoms'] for s in states]),
                    torch.stack([s['template_positions'] for s in states]),
                    torch.stack([s['target_function'] for s in states])
                )

                # Calculate current log probabilities and entropy
                atom_probs = F.softmax(atom_logits, dim=-1)
                current_log_probs = self._compute_action_log_prob(
                    atom_probs,
                    torch.stack([a['atom_types'] for a in actions]),
                    positions,
                    bonds,
                    torsion_angles
                )
                entropy = -torch.mean(torch.sum(atom_probs * torch.log(atom_probs + 1e-10), dim=-1))

                # Calculate ratios and surrogate objectives
                ratios = torch.exp(current_log_probs - old_log_probs)
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Calculate value loss
                current_values = self.value_net(torch.stack([s['target_function'] for s in states])).squeeze(-1)
                value_loss = F.mse_loss(current_values, returns)

                # Combined loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                # Update networks
                self.generator_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                loss.backward()

                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)

                self.generator_optimizer.step()
                self.value_optimizer.step()

                # Calculate approximate KL divergence
                with torch.no_grad():
                    kl = ((old_log_probs - current_log_probs) * ratios).mean()
                    if kl > self.target_kl * 1.5:
                        break

        # Clear buffer after update
        self.buffer.clear()

        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'kl': kl.item()
        }

    def _compute_action_log_prob(self, atom_probs, atom_types, positions, bonds, torsion_angles):
        """
        Compute log probability of selected actions.

        Args:
            atom_probs: Probabilities for atom types
            atom_types: Selected atom types
            positions: Generated positions
            bonds: Generated bonds
            torsion_angles: Generated torsion angles

        Returns:
            torch.Tensor: Log probability of actions
        """
        # Log probability of atom types
        atom_log_probs = torch.log(torch.gather(atom_probs, -1, atom_types.unsqueeze(-1)).squeeze(-1))

        # Add position and bond contributions (assuming Gaussian distributions)
        pos_log_probs = -0.5 * torch.sum(positions ** 2, dim=-1)  # Simplified
        bond_log_probs = -0.5 * torch.sum(bonds ** 2, dim=-1)  # Simplified
        torsion_log_probs = -0.5 * torch.sum(torsion_angles ** 2, dim=-1)  # Simplified

        # Combine all components
        total_log_prob = atom_log_probs + pos_log_probs + bond_log_probs + torsion_log_probs

        return total_log_prob
