import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import optparse
import pickle

import memory as mem
from feedforward import Feedforward

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(1)

class BatchRenorm(torch.jit.ScriptModule):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-3,
        momentum: float = 0.01,
        affine: bool = True,
    ):
        super().__init__()
        self.register_buffer(
            "running_mean", torch.zeros(num_features, dtype=torch.float)
        )
        self.register_buffer(
            "running_std", torch.ones(num_features, dtype=torch.float)
        )
        self.register_buffer(
            "num_batches_tracked", torch.tensor(0, dtype=torch.long)
        )
        self.weight = torch.nn.Parameter(
            torch.ones(num_features, dtype=torch.float)
        )
        self.bias = torch.nn.Parameter(
            torch.zeros(num_features, dtype=torch.float)
        )
        self.affine = affine
        self.eps = eps
        self.step = 0
        self.momentum = momentum

    def _check_input_dim(self, x: torch.Tensor) -> None:
        raise NotImplementedError()  # pragma: no cover

    @property
    def rmax(self) -> torch.Tensor:
        return (2 / 35000 * self.num_batches_tracked + 25 / 35).clamp_(
            1.0, 3.0
        )

    @property
    def dmax(self) -> torch.Tensor:
        return (5 / 20000 * self.num_batches_tracked - 25 / 20).clamp_(
            0.0, 5.0
        )

    def forward(self, x: torch.Tensor, mask = None) -> torch.Tensor:
        '''
        Mask is a boolean tensor used for indexing, where True values are padded
        i.e for 3D input, mask should be of shape (batch_size, seq_len)
        mask is used to prevent padded values from affecting the batch statistics
        '''
        self._check_input_dim(x)
        if x.dim() > 2:
            x = x.transpose(1, -1)
        if self.training:
            dims = [i for i in range(x.dim() - 1)]
            if mask is not None:
                z = x[~mask]
                batch_mean = z.mean(0)
                batch_std = z.std(0, unbiased=False) + self.eps
            else:
                batch_mean = x.mean(dims)
                batch_std = x.std(dims, unbiased=False) + self.eps

            r = (
                batch_std.detach() / self.running_std.view_as(batch_std)
            ).clamp_(1 / self.rmax, self.rmax)
            d = (
                (batch_mean.detach() - self.running_mean.view_as(batch_mean))
                / self.running_std.view_as(batch_std)
            ).clamp_(-self.dmax, self.dmax)
            x = (x - batch_mean) / batch_std * r + d
            self.running_mean += self.momentum * (
                batch_mean.detach() - self.running_mean
            )
            self.running_std += self.momentum * (
                batch_std.detach() - self.running_std
            )
            self.num_batches_tracked += 1
        else:
            x = (x - self.running_mean) / self.running_std
        if self.affine:
            x = self.weight * x + self.bias
        if x.dim() > 2:
            x = x.transpose(1, -1)
        return x


class BatchRenorm1d(BatchRenorm):
    def _check_input_dim(self, x: torch.Tensor) -> None:
        if x.dim() not in [2, 3]:
            raise ValueError("expected 2D or 3D input (got {x.dim()}D input)")


class BatchRenorm2d(BatchRenorm):
    def _check_input_dim(self, x: torch.Tensor) -> None:
        if x.dim() != 4:
            raise ValueError("expected 4D input (got {x.dim()}D input)")


class BatchRenorm3d(BatchRenorm):
    def _check_input_dim(self, x: torch.Tensor) -> None:
        if x.dim() != 5:
            raise ValueError("expected 5D input (got {x.dim()}D input)")

class UnsupportedSpace(Exception):
    """Exception raised when the Sensor or Action space are not compatible."""
    def __init__(self, message="Unsupported Space"):
        self.message = message
        super().__init__(self.message)

class QFunction(Feedforward):
    def __init__(self, observation_dim, action_dim, hidden_sizes=[100,100],
                 learning_rate = 0.0002):
        super().__init__(input_size=observation_dim + action_dim, hidden_sizes=hidden_sizes,
                         output_size=1)
        self.bns = nn.ModuleList([
            BatchRenorm1d(size) for size in hidden_sizes
        ])
        self.activations = nn.ModuleList([
            nn.ReLU() for size in hidden_sizes
        ])
        self.optimizer=torch.optim.Adam(self.parameters(),
                                        lr=learning_rate,
                                        eps=0.000001)
        self.loss = torch.nn.SmoothL1Loss()

    def forward(self, x):
        for layer,activation_fun, bn in zip(self.layers, self.activations, self.bns):
            x = activation_fun(bn(layer(x)))
        if self.output_activation is not None:
            return self.output_activation(self.readout(x))
        else:
            return self.readout(x)

    def fit(self, observations, actions, targets): # all arguments should be torch tensors
        self.train() # put model in training mode
        self.optimizer.zero_grad()
        # Forward pass

        pred = self.Q_value(observations,actions)
        # Compute Loss
        loss = self.loss(pred, targets)

        # Backward pass
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def Q_value(self, observations, actions):
        return self.forward(torch.hstack([observations,actions]))

class Policy(Feedforward):
    def __init__(self, observation_dim, action_dim, hidden_sizes=[100, 100], learning_rate=0.0003):
        super().__init__(input_size=observation_dim, hidden_sizes=hidden_sizes, output_size=action_dim)
        self.bns = nn.ModuleList([
            BatchRenorm1d(size) for size in hidden_sizes
        ])
        self.activations = nn.ModuleList([
            nn.ReLU() for size in hidden_sizes
        ])
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.log_std_min = -20
        self.log_std_max = 2

    def forward(self, x):
        for layer,activation_fun, bn in zip(self.layers, self.activations, self.bns):
            x = activation_fun(bn(layer(x)))
        if self.output_activation is not None:
            return self.output_activation(self.readout(x))
        else:
            return self.readout(x)

    def get_action(self, observation):
        mean, log_std = self.forward(observation).chunk(2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        x = dist.rsample()
        action = torch.tanh(x)

        # Account for tanh squashing in log prob
        log_prob = dist.log_prob(x)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)

        return action, log_prob

    def fit(self, observations, log_probs, q_values):
        self.train()
        self.optimizer.zero_grad()
        policy_loss = (log_probs - q_values).mean()
        policy_loss.backward()
        self.optimizer.step()
        return policy_loss.item()

class SACAgent:
    def __init__(self, observation_space, action_space, **userconfig):
        if not isinstance(observation_space, spaces.Box):
            raise UnsupportedSpace(f'Observation space {observation_space} incompatible (Require: Box)')
        if not isinstance(action_space, spaces.Box):
            raise UnsupportedSpace(f'Action space {action_space} incompatible (Require: Box)')

        self._observation_space = observation_space
        self._obs_dim = observation_space.shape[0]
        self._action_space = action_space
        self._action_dim = action_space.shape[0]
        self._config = {
            "discount": 0.95,
            "tau": 0.005,
            "learning_rate_actor": 0.00001,
            "learning_rate_critic": 0.0001,
            "hidden_sizes_actor": [128, 128],
            "hidden_sizes_critic": [128, 128, 64],
            "entropy_coeff": 0.2,
            "batch_size": 256,
            "buffer_size": int(1e6),
        }
        self._config.update(userconfig)

        # Replay buffer
        self.buffer = mem.Memory(max_size=self._config["buffer_size"])

        # Networks
        self.policy = Policy(observation_dim=self._obs_dim, action_dim=2 * self._action_dim,
                             hidden_sizes=self._config["hidden_sizes_actor"], learning_rate=self._config["learning_rate_actor"])
        self.q1 = QFunction(self._obs_dim, self._action_dim, self._config["hidden_sizes_critic"],
                            self._config["learning_rate_critic"])
        self.q2 = QFunction(self._obs_dim, self._action_dim, self._config["hidden_sizes_critic"],
                            self._config["learning_rate_critic"])
        print(self.q1)
        print(self.q2)
        print(self.policy)
        self.target_entropy = -np.prod(action_space.shape)
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)
        self._copy_nets()

    def _copy_nets(self):
        pass

    def act(self, observation):
        with torch.no_grad():
            observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
            action, _ = self.policy.get_action(observation)
        action = action.squeeze(0).numpy()
        action = self._action_space.low + (action + 1.0) / 2.0 * (self._action_space.high - self._action_space.low)
        return action

    def store_transition(self, transition):
        self.buffer.add_transition(transition)

    def state(self):
        return {
            "policy": self.policy.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict()
        }

    def restore_state(self, state):
        self.policy.load_state_dict(state["policy"])
        self.q1.load_state_dict(state["q1"])
        self.q2.load_state_dict(state["q2"])
        self._copy_nets()  # Ensure target networks are properly synchronized

    def reset(self):
        pass  # SAC does not use action noise; no reset needed

    def train(self, iter_fit=256):
        self.q1.train()
        self.q2.train()
        self.policy.train()
        losses = []
        for _ in range(iter_fit):
            # Sample batch from replay buffer
            batch = self.buffer.sample(self._config["batch_size"])
            observations, actions, rewards, next_obs, dones = map(
                lambda x: torch.tensor(np.array(x), dtype=torch.float32), zip(*batch)
            )

            # Calculate Q-targets
            with torch.no_grad():
                # Get next actions and their log probs from current policy
                next_actions, next_log_probs = self.policy.get_action(next_obs)

                # Concatenate states and actions for CrossQ
                combined_obs = torch.cat([observations, next_obs], dim=0)
                combined_actions = torch.cat([actions, next_actions], dim=0)

                # Get Q-values using target networks with combined batch
                target_q1 = self.q1.Q_value(combined_obs, combined_actions)
                target_q2 = self.q2.Q_value(combined_obs, combined_actions)

                # Split back to current and next
                _, next_q1 = torch.chunk(target_q1, 2, dim=0)
                _, next_q2 = torch.chunk(target_q2, 2, dim=0)

                # Take minimum Q-value for robustness
                min_next_q = torch.min(next_q1, next_q2)

                alpha = self._config["entropy_coeff"]
                q_target = rewards.unsqueeze(-1) + \
                           (1 - dones.unsqueeze(-1)) * self._config["discount"] * \
                           (min_next_q - alpha * next_log_probs)

            # Update Q-functions
            # First Q-function
            q1_all = self.q1.Q_value(combined_obs, combined_actions)
            current_q1, _ = torch.chunk(q1_all, 2, dim=0)
            q1_loss = self.q1.loss(current_q1, q_target.detach())
            self.q1.optimizer.zero_grad()
            q1_loss.backward()
            self.q1.optimizer.step()

            # Second Q-function
            q2_all = self.q2.Q_value(combined_obs, combined_actions)
            current_q2, _ = torch.chunk(q2_all, 2, dim=0)
            q2_loss = self.q2.loss(current_q2, q_target.detach())
            self.q2.optimizer.zero_grad()
            q2_loss.backward()
            self.q2.optimizer.step()

            # Update policy
            # Get actions and log probs from current policy
            new_actions, log_probs = self.policy.get_action(observations)

            # Calculate Q-values for new actions
            q1_all = self.q1.Q_value(torch.concat([observations, next_obs]),
                                     torch.concat([actions, next_actions]))
            _, q1_new = torch.chunk(q1_all, 2)
            q2_all = self.q2.Q_value(torch.concat([observations, next_obs]),
                                     torch.concat([actions, next_actions]))
            _, q2_new = torch.chunk(q2_all, 2)
            min_q_new = torch.min(q1_new, q2_new)

            # Calculate policy loss (negative for gradient ascent)
            policy_loss = (alpha * log_probs - min_q_new).mean()

            # Update policy
            self.policy.optimizer.zero_grad()
            policy_loss.backward()
            self.policy.optimizer.step()

            # Store losses for logging
            losses.append((q1_loss.item(), q2_loss.item(), policy_loss.item()))
        self.q1.eval()
        self.q2.eval()
        self.policy.eval()
        return losses

    def _soft_update(self, source, target):
        pass

    def reset(self):
        pass


def main():
    optParser = optparse.OptionParser()
    optParser.add_option('-e', '--env', action='store', type='string',
                         dest='env_name', default="Pendulum-v1",
                         help='Environment (default %default)')
    optParser.add_option('-a', '--agent', action='store', type='string',
                         dest='agent', default="DDPG",
                         help='Agent type (DDPG or SAC, default %default)')
    optParser.add_option('-n', '--eps', action='store', type='float',
                         dest='eps', default=0.1,
                         help='Policy noise (default %default)')
    optParser.add_option('-t', '--train', action='store', type='int',
                         dest='train', default=32,
                         help='Number of training batches per episode (default %default)')
    optParser.add_option('-l', '--lr', action='store', type='float',
                         dest='lr', default=0.0001,
                         help='Learning rate for actor/policy (default %default)')
    optParser.add_option('-m', '--maxepisodes', action='store', type='float',
                         dest='max_episodes', default=2000,
                         help='Number of episodes (default %default)')
    optParser.add_option('-u', '--update', action='store', type='float',
                         dest='update_every', default=100,
                         help='Number of episodes between target network updates (default %default)')
    optParser.add_option('-s', '--seed', action='store', type='int',
                         dest='seed', default=None,
                         help='Random seed (default %default)')
    opts, args = optParser.parse_args()

    # Set environment
    env_name = opts.env_name
    env = gym.make(env_name)
    render = False
    max_episodes = opts.max_episodes
    max_timesteps = 2000
    train_iter = opts.train
    eps = opts.eps
    lr = opts.lr
    random_seed = opts.seed

    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    # Initialize agent
    if opts.agent == "DDPG":
        agent = DDPGAgent(env.observation_space, env.action_space,
                          eps=eps, learning_rate_actor=lr,
                          update_target_every=opts.update_every)
    elif opts.agent == "SAC":
        agent = SACAgent(env.observation_space, env.action_space,
                         learning_rate_actor=lr)
    else:
        raise ValueError(f"Unsupported agent type: {opts.agent}")

    # Logging variables
    rewards = []
    lengths = []
    losses = []
    timestep = 0

    def save_statistics():
        with open(f"./results/{opts.agent}_{env_name}-eps{eps}-t{train_iter}-l{lr}-s{random_seed}-stat.pkl", 'wb') as f:
            pickle.dump({"rewards": rewards, "lengths": lengths, "eps": eps, "train": train_iter,
                         "lr": lr, "update_every": opts.update_every, "losses": losses}, f)

    # Training loop
    for i_episode in range(1, max_episodes + 1):
        ob, _info = env.reset()
        agent.reset()
        total_reward = 0
        for t in range(max_timesteps):
            timestep += 1
            done = False
            a = agent.act(ob)
            (ob_new, reward, done, trunc, _info) = env.step(a)
            total_reward += reward
            agent.store_transition((ob, a, reward, ob_new, done))
            ob = ob_new
            if done or trunc:
                break

        losses.extend(agent.train(train_iter))

        rewards.append(total_reward)
        lengths.append(t)

        # Save every 500 episodes
        if i_episode % 500 == 0:
            print("########## Saving a checkpoint... ##########")
            torch.save(agent.state(), f'./results/{opts.agent}_{env_name}_{i_episode}-eps{eps}-t{train_iter}-l{lr}-s{random_seed}.pth')
            save_statistics()

        # Logging
        if i_episode % 20 == 0:
            avg_reward = np.mean(rewards[-20:])
            avg_length = int(np.mean(lengths[-20:]))
            print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, avg_reward))
    save_statistics()

if __name__ == '__main__':
    main()