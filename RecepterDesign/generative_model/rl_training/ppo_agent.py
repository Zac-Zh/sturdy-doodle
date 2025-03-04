import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque


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
            atom_logits, positions, _ = self.generator(state['template_atoms'],
                                                       state['template_positions'],
                                                       state['target_function'])

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
            log_prob = self._compute_action_log_prob(atom_probs, atom_types, positions)

        return {
            'atom_types': atom_types,
            'positions': positions
        }, log_prob, value

    def update(self):
        """
        Update policy and value networks using PPO.

        Returns:
            dict: Training statistics
        """
        # Get data from buffer
        states, actions, old_log_probs, rewards, dones, values = self.buffer.get()

        # Compute advantages and returns
        returns, advantages = self._compute_gae(rewards, values, dones)

        # PPO update
        policy_loss_epoch = 0
        value_loss_epoch = 0
        entropy_epoch = 0
        kl_epoch = 0

        # Multiple update epochs
        for _ in range(self.update_epochs):
            # Generate random mini-batch indices
            indices = np.random.permutation(len(states))

            # Update in mini-batches
            for start_idx in range(0, len(states), self.mini_batch_size):
                end_idx = start_idx + self.mini_batch_size
                batch_indices = indices[start_idx:end_idx]

                # Get mini-batch
                mb_states = {k: v[batch_indices] for k, v in states.items()}
                mb_actions = {k: v[batch_indices] for k, v in actions.items()}
                mb_old_log_probs = old_log_probs[batch_indices]
                mb_advantages = advantages[batch_indices]
                mb_returns = returns[batch_indices]

                # Get current log probs and values
                atom_logits, positions, _ = self.generator(mb_states['template_atoms'],
                                                           mb_states['template_positions'],
                                                           mb_states['target_function'])
                atom_probs = F.softmax(atom_logits, dim=-1)

                # Calculate log probability of actions
                curr_log_probs = self._compute_action_log_prob(atom_probs,
                                                               mb_actions['atom_types'],
                                                               mb_actions['positions'])

                # Calculate entropy
                entropy = self._compute_entropy(atom_probs)

                # Calculate value predictions
                values = self.value_net(mb_states['target_function']).squeeze(-1)

                # Calculate ratios and surrogate objectives
                ratios = torch.exp(curr_log_probs - mb_old_log_probs)
                surr1 = ratios * mb_advantages
                surr2 = torch.clamp(ratios, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * mb_advantages

                # Calculate policy loss, value loss, and entropy
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, mb_returns)
                entropy_loss = -entropy.mean()

                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                # Calculate approximate KL divergence
                approx_kl = ((ratios - 1) - torch.log(ratios)).mean().item()

                # Early stopping based on KL
                if approx_kl > 1.5 * self.target_kl:
                    break

                # Update generator
                self.generator_optimizer.zero_grad()
                policy_loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.generator.parameters(), self.max_grad_norm)
                self.generator_optimizer.step()

                # Update value function
                self.value_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
                self.value_optimizer.step()

                # Accumulate statistics
                policy_loss_epoch += policy_loss.item()
                value_loss_epoch += value_loss.item()
                entropy_epoch += entropy.mean().item()
                kl_epoch += approx_kl

        # Clear buffer after update
        self.buffer.clear()

        # Return average statistics
        num_updates = self.update_epochs
        return {
            'policy_loss': policy_loss_epoch / num_updates,
            'value_loss': value_loss_epoch / num_updates,
            'entropy': entropy_epoch / num_updates,
            'approx_kl': kl_epoch / num_updates
        }

    def _compute_action_log_prob(self, atom_probs, atom_types, positions):
        """Compute log probability of action."""
        # Log probability of atom types
        batch_size, seq_len, num_atom_types = atom_probs.shape
        indices = atom_types.unsqueeze(-1)
        selected_probs = torch.gather(atom_probs, -1, indices).squeeze(-1)
        log_probs = torch.log(selected_probs + 1e-8)

        # Sum log probs over sequence length
        total_log_prob = log_probs.sum(dim=-1)

        return total_log_prob

    def _compute_entropy(self, atom_probs):
        """Compute entropy of action distribution."""
        # Entropy of categorical distribution
        entropy = -torch.sum(atom_probs * torch.log(atom_probs + 1e-8), dim=-1)

        # Sum entropy over sequence length
        total_entropy = entropy.sum(dim=-1)

        return total_entropy

    def _compute_gae(self, rewards, values, dones, last_value=0, gae_lambda=0.95):
        """Compute Generalized Advantage Estimation."""
        advantages = torch.zeros_like(rewards)
        last_gae_lam = 0

        # Reverse iteration for efficient GAE calculation
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            last_gae_lam = delta + self.gamma * gae_lambda * (1 - dones[t]) * last_gae_lam
            advantages[t] = last_gae_lam

        # Calculate returns
        returns = advantages + values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return returns, advantages


class TrajectoryBuffer:
    """Buffer for storing trajectories."""

    def __init__(self, gamma=0.99, buffer_size=1000):
        """
        Initialize trajectory buffer.

        Args:
            gamma: Discount factor
            buffer_size: Maximum buffer size
        """
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.clear()

    def add(self, state, action, reward, next_state, done, log_prob, value):
        """
        Add trajectory step to buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Done flag
            log_prob: Log probability of action
            value: Value prediction
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def clear(self):
        """Clear buffer."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def get(self):
        """
        Get data from buffer.

        Returns:
            tuple: (States, Actions, Log probs, Rewards, Dones, Values)
        """
        # Convert lists to appropriate format
        # States and actions may contain dictionaries
        states = {k: torch.stack([s[k] for s in self.states]) for k in self.states[0].keys()}
        actions = {k: torch.stack([a[k] for a in self.actions]) for k in self.actions[0].keys()}

        # Convert other lists to tensors
        rewards = torch.tensor(self.rewards, dtype=torch.float32)
        dones = torch.tensor(self.dones, dtype=torch.float32)
        log_probs = torch.stack(self.log_probs)
        values = torch.stack(self.values)

        return states, actions, log_probs, rewards, dones, values

    def size(self):
        """Get current buffer size."""
        return len(self.rewards)
