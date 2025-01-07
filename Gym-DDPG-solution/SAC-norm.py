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
            nn.BatchNorm1d(size) for size in hidden_sizes
        ])
        self.optimizer=torch.optim.Adam(self.parameters(),
                                        lr=learning_rate,
                                        eps=0.000001)
        self.loss = torch.nn.SmoothL1Loss()

    def forward(self, x):
        for layer, bn, activation in zip(self.layers[:-1], self.bns, self.activations[:-1]):
            x = layer(x)
            # Apply BatchNorm before activation
            if x.shape[0] > 1:  # Only apply BatchNorm for batch size > 1
                x = bn(x)
            x = activation(x)
        return self.layers[-1](x)

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
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.log_std_min = -20
        self.log_std_max = 2

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

        self.q1_target = QFunction(self._obs_dim, self._action_dim, self._config["hidden_sizes_critic"], learning_rate=0)
        self.q2_target = QFunction(self._obs_dim, self._action_dim, self._config["hidden_sizes_critic"], learning_rate=0)
        print(self.q1)
        print(self.q2)
        print(self.policy)
        self.target_entropy = -np.prod(action_space.shape)
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)
        self._copy_nets()

    def _copy_nets(self):
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

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
            "q2": self.q2.state_dict(),
            "q1_target": self.q1_target.state_dict(),
            "q2_target": self.q2_target.state_dict()
        }

    def restore_state(self, state):
        self.policy.load_state_dict(state["policy"])
        self.q1.load_state_dict(state["q1"])
        self.q2.load_state_dict(state["q2"])
        self.q1_target.load_state_dict(state["q1_target"])
        self.q2_target.load_state_dict(state["q2_target"])
        self._copy_nets()  # Ensure target networks are properly synchronized

    def reset(self):
        pass  # SAC does not use action noise; no reset needed

    def train(self, iter_fit=256):
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

                # Get Q-values for next states from both target networks
                q1_all = self.q1_target.Q_value(torch.concat([observations, next_obs]),
                                                torch.concat([actions, next_actions]))
                q2_all = self.q2_target.Q_value(torch.concat([observations, next_obs]),
                                                torch.concat([actions, next_actions]))
                q1, q1_next = torch.chunk(q1_all, 2)
                q2, q2_next = torch.chunk(q2_all, 2)

                # Take minimum Q-value for robustness
                min_q_next = torch.min(q1_next, q2_next)

                # Calculate entropy term
                alpha = self._config["entropy_coeff"]
                entropy_term = alpha * next_log_probs

                # Compute targets for Q-functions
                q_target = rewards.unsqueeze(-1) + \
                           self._config["discount"] * (1 - dones.unsqueeze(-1)) * \
                           (min_q_next - entropy_term)

            # Update Q-functions
            # First Q-function
            q1_all = self.q1.Q_value(torch.concat([observations, next_obs]),
                                     torch.concat([actions, next_actions]))
            current_q1, _ = torch.chunk(q1_all, 2)
            q1_loss = self.q1.loss(current_q1, q_target.detach())
            self.q1.optimizer.zero_grad()
            q1_loss.backward()
            self.q1.optimizer.step()

            # Second Q-function
            q2_all = self.q2.Q_value(torch.concat([observations, next_obs]),
                                     torch.concat([actions, next_actions]))
            current_q2, _ = torch.chunk(q2_all, 2)
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

            self._soft_update(self.q1, self.q1_target)
            self._soft_update(self.q2, self.q2_target)

            # Store losses for logging
            losses.append((q1_loss.item(), q2_loss.item(), policy_loss.item()))

        return losses

    def _soft_update(self, source, target):
        tau = self._config["tau"]
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

    def reset(self):
        pass


class OUNoise():
    def __init__(self, shape, theta: float = 0.15, dt: float = 1e-2):
        self._shape = shape
        self._theta = theta
        self._dt = dt
        self.noise_prev = np.zeros(self._shape)
        self.reset()

    def __call__(self) -> np.ndarray:
        noise = (
            self.noise_prev
            + self._theta * ( - self.noise_prev) * self._dt
            + np.sqrt(self._dt) * np.random.normal(size=self._shape)
        )
        self.noise_prev = noise
        return noise

    def reset(self) -> None:
        self.noise_prev = np.zeros(self._shape)


class DDPGAgent(object):
    """
    Agent implementing Q-learning with NN function approximation.
    """
    def __init__(self, observation_space, action_space, **userconfig):

        if not isinstance(observation_space, spaces.box.Box):
            raise UnsupportedSpace('Observation space {} incompatible ' \
                                   'with {}. (Require: Box)'.format(observation_space, self))
        if not isinstance(action_space, spaces.box.Box):
            raise UnsupportedSpace('Action space {} incompatible with {}.' \
                                   ' (Require Box)'.format(action_space, self))

        self._observation_space = observation_space
        self._obs_dim=self._observation_space.shape[0]
        self._action_space = action_space
        self._action_n = action_space.shape[0]
        self._config = {
            "eps": 0.1,            # Epsilon: noise strength to add to policy
            "discount": 0.95,
            "buffer_size": int(1e6),
            "batch_size": 128,
            "learning_rate_actor": 0.00001,
            "learning_rate_critic": 0.0001,
            "hidden_sizes_actor": [128,128],
            "hidden_sizes_critic": [128,128,64],
            "update_target_every": 100,
            "use_target_net": True
        }
        self._config.update(userconfig)
        self._eps = self._config['eps']

        self.action_noise = OUNoise((self._action_n))

        self.buffer = mem.Memory(max_size=self._config["buffer_size"])

        # Q Network
        self.Q = QFunction(observation_dim=self._obs_dim,
                           action_dim=self._action_n,
                           hidden_sizes= self._config["hidden_sizes_critic"],
                           learning_rate = self._config["learning_rate_critic"])
        # target Q Network
        self.Q_target = QFunction(observation_dim=self._obs_dim,
                                  action_dim=self._action_n,
                                  hidden_sizes= self._config["hidden_sizes_critic"],
                                  learning_rate = 0)

        self.policy = Feedforward(input_size=self._obs_dim,
                                  hidden_sizes= self._config["hidden_sizes_actor"],
                                  output_size=self._action_n,
                                  activation_fun = torch.nn.ReLU(),
                                  output_activation = torch.nn.Tanh())
        self.policy_target = Feedforward(input_size=self._obs_dim,
                                         hidden_sizes= self._config["hidden_sizes_actor"],
                                         output_size=self._action_n,
                                         activation_fun = torch.nn.ReLU(),
                                         output_activation = torch.nn.Tanh())
        print(self.Q)
        print(self.Q_target)
        print(self.policy)
        print(self.policy_target)
        self._copy_nets()

        self.optimizer=torch.optim.Adam(self.policy.parameters(),
                                        lr=self._config["learning_rate_actor"],
                                        eps=0.000001)
        self.train_iter = 0

    def _copy_nets(self):
        self.Q_target.load_state_dict(self.Q.state_dict())
        self.policy_target.load_state_dict(self.policy.state_dict())

    def act(self, observation, eps=None):
        if eps is None:
            eps = self._eps
        #
        action = self.policy.predict(observation) + eps*self.action_noise()  # action in -1 to 1 (+ noise)
        action = self._action_space.low + (action + 1.0) / 2.0 * (self._action_space.high - self._action_space.low)
        return action

    def store_transition(self, transition):
        self.buffer.add_transition(transition)

    def state(self):
        return (self.Q.state_dict(), self.policy.state_dict())

    def restore_state(self, state):
        self.Q.load_state_dict(state[0])
        self.policy.load_state_dict(state[1])
        self._copy_nets()

    def reset(self):
        self.action_noise.reset()

    def train(self, iter_fit=32):
        to_torch = lambda x: torch.from_numpy(x.astype(np.float32))
        losses = []
        self.train_iter+=1
        if self._config["use_target_net"] and self.train_iter % self._config["update_target_every"] == 0:
            self._copy_nets()
        for i in range(iter_fit):

            # sample from the replay buffer
            data=self.buffer.sample(batch=self._config['batch_size'])
            s = to_torch(np.stack(data[:,0])) # s_t
            a = to_torch(np.stack(data[:,1])) # a_t
            rew = to_torch(np.stack(data[:,2])[:,None]) # rew  (batchsize,1)
            s_prime = to_torch(np.stack(data[:,3])) # s_t+1
            done = to_torch(np.stack(data[:,4])[:,None]) # done signal  (batchsize,1)

            if self._config["use_target_net"]:
                q_prime = self.Q_target.Q_value(s_prime, self.policy_target.forward(s_prime))
            else:
                q_prime = self.Q.Q_value(s_prime, self.policy.forward(s_prime))
            # target
            gamma=self._config['discount']
            td_target = rew + gamma * (1.0-done) * q_prime

            # optimize the Q objective
            fit_loss = self.Q.fit(s, a, td_target)

            # optimize actor objective
            self.optimizer.zero_grad()
            q = self.Q.Q_value(s, self.policy.forward(s))
            actor_loss = -torch.mean(q)
            actor_loss.backward()
            self.optimizer.step()

            losses.append((fit_loss, actor_loss.item()))

        return losses


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