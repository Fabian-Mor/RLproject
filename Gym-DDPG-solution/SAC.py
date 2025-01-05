import torch
import numpy as np
from torch.distributions import MultivariateNormal
from torch import nn
import gymnasium as gym
from gymnasium import spaces
import optparse
import pickle

class QFunction(nn.Module):
    def __init__(self, input_size, action_dim, hidden_sizes, learning_rate=3e-4):
        super().__init__()
        layers = []
        last_size = input_size + action_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(last_size, size))
            layers.append(nn.ReLU())
            last_size = size
        layers.append(nn.Linear(last_size, 1))
        self.network = nn.Sequential(*layers)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, obs, action):
        return self.network(torch.cat([obs, action], dim=-1))

    def fit(self, obs, action, target):
        q_value = self.forward(obs, action)
        loss = nn.MSELoss()(q_value, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()


class StochasticPolicy(nn.Module):
    def __init__(self, input_size, hidden_sizes, action_dim, learning_rate=3e-4):
        super().__init__()
        layers = []
        last_size = input_size
        for size in hidden_sizes:
            layers.append(nn.Linear(last_size, size))
            layers.append(nn.ReLU())
            last_size = size
        layers.append(nn.Linear(last_size, action_dim * 2))  # Mean and log_std
        self.network = nn.Sequential(*layers)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, obs):
        x = self.network(obs)
        mean, log_std = torch.chunk(x, 2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)  # Clamp for numerical stability
        std = torch.exp(log_std)
        return mean, std

    def sample(self, obs):
        mean, std = self.forward(obs)
        dist = MultivariateNormal(mean, torch.diag_embed(std))
        action = dist.rsample()  # Reparameterization trick
        log_prob = dist.log_prob(action)
        return action, log_prob


class SACAgent:
    def __init__(self, obs_dim, action_dim, hidden_sizes_actor, hidden_sizes_critic,
                 learning_rate_actor=3e-4, learning_rate_critic=3e-4, alpha=0.2, discount=0.99):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.discount = discount
        self.alpha = alpha

        # Policy network
        self.policy = StochasticPolicy(obs_dim, hidden_sizes_actor, action_dim, learning_rate_actor)

        # Q-functions
        self.Q1 = QFunction(obs_dim, action_dim, hidden_sizes_critic, learning_rate_critic)
        self.Q2 = QFunction(obs_dim, action_dim, hidden_sizes_critic, learning_rate_critic)
        self.Q1_target = QFunction(obs_dim, action_dim, hidden_sizes_critic, learning_rate_critic)
        self.Q2_target = QFunction(obs_dim, action_dim, hidden_sizes_critic, learning_rate_critic)

        # Copy weights to target networks
        self._copy_nets()

        # Entropy adjustment (optional)
        self.target_entropy = -float(action_dim)
        self.log_alpha = torch.tensor([np.log(alpha)], requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=learning_rate_actor)

    def _copy_nets(self):
        self.Q1_target.load_state_dict(self.Q1.state_dict())
        self.Q2_target.load_state_dict(self.Q2.state_dict())

    def train(self, buffer, batch_size=256, iter_fit=32):
        losses = []
        for _ in range(iter_fit):
            data = buffer.sample(batch_size)
            obs, actions, rewards, next_obs, dones = data

            obs = torch.tensor(obs, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.float32)
            rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1)
            next_obs = torch.tensor(next_obs, dtype=torch.float32)
            dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(-1)

            # Critic Loss
            with torch.no_grad():
                next_action, next_log_prob = self.policy.sample(next_obs)
                target_q1 = self.Q1_target(next_obs, next_action)
                target_q2 = self.Q2_target(next_obs, next_action)
                target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob.unsqueeze(-1)
                td_target = rewards + self.discount * (1.0 - dones) * target_q

            q1_loss = self.Q1.fit(obs, actions, td_target)
            q2_loss = self.Q2.fit(obs, actions, td_target)

            # Actor Loss
            new_action, log_prob = self.policy.sample(obs)
            q1_value = self.Q1(obs, new_action)
            q2_value = self.Q2(obs, new_action)
            actor_loss = (self.alpha * log_prob - torch.min(q1_value, q2_value)).mean()

            self.policy.optimizer.zero_grad()
            actor_loss.backward()
            self.policy.optimizer.step()

            # Entropy Adjustment
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp().item()

            losses.append((q1_loss, q2_loss, actor_loss.item(), alpha_loss.item()))

        # Update target networks
        self._copy_nets()
        return losses


class ReplayBuffer:
    def __init__(self, max_size, obs_dim, action_dim):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.obs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(max_size, dtype=np.float32)
        self.next_obs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.dones = np.zeros(max_size, dtype=np.float32)

    def store(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idxs = np.random.choice(self.size, batch_size, replace=False)
        return (
            self.obs[idxs],
            self.actions[idxs],
            self.rewards[idxs],
            self.next_obs[idxs],
            self.dones[idxs]
        )


def main():
    optParser = optparse.OptionParser()
    optParser.add_option('-e', '--env', action='store', type='string',
                         dest='env_name', default="Pendulum-v1",
                         help='Environment (default %default)')
    optParser.add_option('-t', '--train', action='store', type='int',
                         dest='train', default=32,
                         help='number of training batches per episode (default %default)')
    optParser.add_option('-l', '--lr', action='store', type='float',
                         dest='lr', default=0.0003,
                         help='learning rate for policy and Q-networks (default %default)')
    optParser.add_option('-m', '--maxepisodes', action='store', type='float',
                         dest='max_episodes', default=2000,
                         help='number of episodes (default %default)')
    optParser.add_option('-s', '--seed', action='store', type='int',
                         dest='seed', default=None,
                         help='random seed (default %default)')
    opts, args = optParser.parse_args()

    env_name = opts.env_name
    env = gym.make(env_name)
    render = False
    max_episodes = opts.max_episodes
    max_timesteps = 2000
    train_iter = opts.train
    lr = opts.lr
    random_seed = opts.seed
    hidden_sizes_actor = [256, 256]
    hidden_sizes_critic = [256, 256]
    alpha = 0.2
    discount = 0.99

    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    agent = SACAgent(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        hidden_sizes_actor=hidden_sizes_actor,
        hidden_sizes_critic=hidden_sizes_critic,
        learning_rate_actor=lr,
        learning_rate_critic=lr,
        alpha=alpha,
        discount=discount
    )

    replay_buffer = ReplayBuffer(max_size=1000000, obs_dim=env.observation_space.shape[0],
                                 action_dim=env.action_space.shape[0])

    rewards = []
    lengths = []
    losses = []

    for i_episode in range(1, max_episodes + 1):
        ob, _ = env.reset()
        total_reward = 0
        for t in range(max_timesteps):
            if render:
                env.render()

            ob_tensor = torch.tensor(ob, dtype=torch.float32).unsqueeze(0)
            action, _ = agent.policy.sample(ob_tensor)
            action = action.detach().numpy()[0]
            next_ob, reward, done, trunc, _ = env.step(action)
            replay_buffer.store(ob, action, reward, next_ob, done)

            ob = next_ob
            total_reward += reward

            if replay_buffer.size > 1000:
                loss = agent.train(replay_buffer, batch_size=256, iter_fit=train_iter)
                losses.extend(loss)

            if done or trunc:
                break

        rewards.append(total_reward)
        lengths.append(t)

        if i_episode % 500 == 0:
            torch.save(agent.policy.state_dict(),
                       f'./results/SAC_{env_name}_{i_episode}-t{train_iter}-l{lr}-s{random_seed}.pth')
            with open(f"./results/SAC_{env_name}_statistics.pkl", 'wb') as f:
                pickle.dump({"rewards": rewards, "lengths": lengths, "losses": losses}, f)

        if i_episode % 20 == 0:
            avg_reward = np.mean(rewards[-20:])
            avg_length = int(np.mean(lengths[-20:]))
            print(f'Episode {i_episode} \t avg length: {avg_length} \t reward: {avg_reward}')

    env.close()


if __name__ == '__main__':
    main()


