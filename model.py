import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from batch_renorm import BatchRenorm1d

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, batch_norm=False, layer_norm=False, skip_connection=False, droQ=False, redQ=False, crossq=False):
        super(QNetwork, self).__init__()
        self.skip_connection = skip_connection
        self.droQ = droQ
        self.redQ = redQ
        self.crossq = crossq
        if droQ:
            self.dropout = nn.Dropout(p=0.01)

            self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
            self.layer_norm1 = nn.LayerNorm(hidden_dim)
            self.linear2 = nn.Linear(hidden_dim, hidden_dim)
            self.layer_norm2 = nn.LayerNorm(hidden_dim)
            self.linear3 = nn.Linear(hidden_dim, 1)

            self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
            self.layer_norm3 = nn.LayerNorm(hidden_dim)
            self.linear5 = nn.Linear(hidden_dim, hidden_dim)
            self.layer_norm4 = nn.LayerNorm(hidden_dim)
            self.linear6 = nn.Linear(hidden_dim, 1)

        elif redQ:
            self.ensemble_size = 10
            self.q_networks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(num_inputs + num_actions, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1)
                ) for _ in range(self.ensemble_size)
            ])

        elif crossq:
            hidden_dim = 2048
            self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
            self.linear2 = nn.Linear(hidden_dim, hidden_dim)
            self.linear3 = nn.Linear(hidden_dim, 1)
            self.bn1 = BatchRenorm1d(hidden_dim)
            self.bn2 = BatchRenorm1d(hidden_dim)

            self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
            self.linear5 = nn.Linear(hidden_dim, hidden_dim)
            self.linear6 = nn.Linear(hidden_dim, 1)
            self.bn3 = BatchRenorm1d(hidden_dim)
            self.bn4 = BatchRenorm1d(hidden_dim)


        else:
            if skip_connection:
                self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
                self.linear2 = nn.Linear(hidden_dim, hidden_dim)
                self.linear3 = nn.Linear(hidden_dim, hidden_dim)
                self.linear4 = nn.Linear(hidden_dim, hidden_dim)
                self.linear5 = nn.Linear(hidden_dim, 1)

                # Q2 architecture
                self.linear6 = nn.Linear(num_inputs + num_actions, hidden_dim)
                self.linear7 = nn.Linear(hidden_dim, hidden_dim)
                self.linear8 = nn.Linear(hidden_dim, hidden_dim)
                self.linear9 = nn.Linear(hidden_dim, hidden_dim)
                self.linear10 = nn.Linear(hidden_dim, 1)
            else:
                self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
                self.linear2 = nn.Linear(hidden_dim, hidden_dim)
                self.linear3 = nn.Linear(hidden_dim, 1)

                # Q2 architecture
                self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
                self.linear5 = nn.Linear(hidden_dim, hidden_dim)
                self.linear6 = nn.Linear(hidden_dim, 1)

            if batch_norm:
                self.bn1 = BatchRenorm1d(hidden_dim)
                self.bn2 = BatchRenorm1d(hidden_dim)
                self.bn3 = BatchRenorm1d(hidden_dim)
                self.bn4 = BatchRenorm1d(hidden_dim)
            elif layer_norm:
                self.bn1 = nn.LayerNorm(hidden_dim)
                self.bn2 = nn.LayerNorm(hidden_dim)
                self.bn3 = nn.LayerNorm(hidden_dim)
                self.bn4 = nn.LayerNorm(hidden_dim)
            else:
                self.bn1 = nn.Identity()
                self.bn2 = nn.Identity()
                self.bn3 = nn.Identity()
                self.bn4 = nn.Identity()

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)

        if self.droQ:
            x1 = self.linear1(xu)
            x1 = self.dropout(x1)
            x1 = self.layer_norm1(x1)
            x1 = F.relu(x1)
            x1 = self.linear2(x1)
            x1 = self.dropout(x1)
            x1 = self.layer_norm2(x1)
            x1 = F.relu(x1)
            x1 = self.linear3(x1)

            x2 = self.linear4(xu)
            x2 = self.dropout(x2)
            x2 = self.layer_norm3(x2)
            x2 = F.relu(x2)
            x2 = self.linear5(x2)
            x2 = self.dropout(x2)
            x2 = self.layer_norm4(x2)
            x2 = F.relu(x2)
            x2 = self.linear6(x2)

            return x1, x2

        elif self.redQ:
            q_values = [q_net(xu) for q_net in self.q_networks]
            q_values = torch.cat(q_values, dim=1)
            return q_values

        elif self.crossq:
            x1 = F.relu(self.bn1(self.linear1(xu)))
            x1 = F.relu(self.bn2(self.linear2(x1)))
            x1 = self.linear3(x1)

            x2 = F.relu(self.bn3(self.linear4(xu)))
            x2 = F.relu(self.bn4(self.linear5(x2)))
            x2 = self.linear6(x2)
            return x1, x2


        if self.skip_connection:
            x1_1 = F.relu(self.linear1(xu))
            x1_2 = F.relu(x1_1 + self.linear2(self.bn1(x1_1)))
            x1_3 = F.relu(self.linear3(x1_2))
            x1_4 = F.relu(x1_3 + self.linear4(self.bn2(x1_3)))
            x1 = self.linear5(x1_4)

            x2_1 = F.relu(self.linear6(xu))
            x2_2 = F.relu(x2_1 + self.linear7(self.bn3(x2_1)))
            x2_3 = F.relu(self.linear8(x2_2))
            x2_4 = F.relu(x2_3 + self.linear9(self.bn4(x2_3)))
            x2 = self.linear10(x2_4)

        else:
            x1 = F.relu(self.bn1(self.linear1(xu)))
            x1 = F.relu(self.bn2(self.linear2(x1)))
            x1 = self.linear3(x1)

            x2 = F.relu(self.bn3(self.linear4(xu)))
            x2 = F.relu(self.bn4(self.linear5(x2)))
            x2 = self.linear6(x2)

        return x1, x2


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)
