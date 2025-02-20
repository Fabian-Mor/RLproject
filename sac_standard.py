import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils import soft_update, hard_update
from model import GaussianPolicy, QNetwork, DeterministicPolicy
from replay_memory import ReplayMemory
import numpy as np

class SAC(object):
    def __init__(self, num_inputs, action_space, args):

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = torch.device("cuda" if args.cuda else "cpu")
        self.batch_norm = args.batch_norm
        self.layer_norm = args.layer_norm
        self.skip_connection = args.skip_connection
        self.droQ = args.droQ
        self.evaluate=False
        self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size, batch_norm=args.batch_norm,
                               layer_norm=args.layer_norm, skip_connection=args.skip_connection, droQ=args.droQ).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)
        self.use_target = args.use_target
        if args.use_target:
            self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size, batch_norm=args.batch_norm,
                                          layer_norm=args.layer_norm, skip_connection=args.skip_connection, droQ=args.droQ).to(self.device)
            hard_update(self.critic_target, self.critic)
        self.memory = ReplayMemory(args.replay_size, 1)
        self.batch_size = args.batch_size

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

    def act(self, state):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if self.evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def set_eval_mode(self):
        self.evaluate=True
        self.policy.eval()
        self.critic.eval()
        if self.use_target:
            self.critic_target.eval()

    def set_training_mode(self):
        self.evaluate=False
        self.policy.train()
        self.critic.train()
        self.critic_target.train()

    def train(self, updates, iter_fit=32):
        losses = []
        for i in range(iter_fit):
            # Sample a batch from memory
            state_batch, action_batch, reward_batch, next_state_batch, mask_batch = self.memory.sample(batch_size=self.batch_size)

            state_batch = torch.FloatTensor(state_batch).to(self.device)
            next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
            action_batch = torch.FloatTensor(action_batch).to(self.device)
            reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
            mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

            with torch.no_grad():
                next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
                if self.batch_norm:
                    # use the trick of the paper crossq
                    combined_obs = torch.cat([state_batch, next_state_batch], dim=0)
                    combined_actions = torch.cat([action_batch, next_state_action], dim=0)
                    if self.use_target:
                        qf1_next_target_all, qf2_next_target_all = self.critic_target(combined_obs, combined_actions)
                    else:
                        qf1_next_target_all, qf2_next_target_all = self.critic(combined_obs, combined_actions)
                    _, qf1_next_target = torch.chunk(qf1_next_target_all, 2, dim=0)
                    _, qf2_next_target = torch.chunk(qf2_next_target_all, 2, dim=0)
                else:
                    if self.use_target:
                        qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
                    else:
                        qf1_next_target, qf2_next_target = self.critic(next_state_batch, next_state_action)

                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
                next_q_value = reward_batch + self.gamma * (min_qf_next_target)
            qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
            qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
            qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
            qf_loss = qf1_loss + qf2_loss

            self.critic_optim.zero_grad()
            qf_loss.backward()
            self.critic_optim.step()

            pi, log_pi, _ = self.policy.sample(state_batch)

            qf1_pi, qf2_pi = self.critic(state_batch, pi)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)

            policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

            self.policy_optim.zero_grad()
            policy_loss.backward()
            self.policy_optim.step()

            if self.automatic_entropy_tuning:
                alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

                self.alpha_optim.zero_grad()
                alpha_loss.backward()
                self.alpha_optim.step()

                self.alpha = self.log_alpha.exp()
            if self.use_target and updates % self.target_update_interval == 0:
                soft_update(self.critic_target, self.critic, self.tau)

            losses.append((qf1_loss.item(), qf2_loss.item(), policy_loss.item()))
        return losses


    # Save model parameters
    def state(self, env_name, suffix="", ckpt_path=None):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        if ckpt_path is None:
            ckpt_path = "checkpoints/sac_checkpoint_{}_{}".format(env_name, suffix)
        print('Saving models to {}'.format(ckpt_path))
        if self.use_target:
            torch.save({'policy_state_dict': self.policy.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'critic_target_state_dict': self.critic_target.state_dict(),
                    'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                    'policy_optimizer_state_dict': self.policy_optim.state_dict()}, ckpt_path)
        else:
            torch.save({'policy_state_dict': self.policy.state_dict(),
                        'critic_state_dict': self.critic.state_dict(),
                        'critic_optimizer_state_dict': self.critic_optim.state_dict(),
                        'policy_optimizer_state_dict': self.policy_optim.state_dict()}, ckpt_path)

    # Load model parameters
    def restore_state(self, ckpt_path, evaluate=False):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            if self.use_target:
                self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            self.critic_optim.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            self.policy_optim.load_state_dict(checkpoint['policy_optimizer_state_dict'])

            if evaluate:
                self.evaluate = True
                self.policy.eval()
                self.critic.eval()
                if self.use_target:
                    self.critic_target.eval()
            else:
                self.policy.train()
                self.critic.train()
                self.critic_target.train()

    def reset(self):
        self.memory.reset()

    def store_transition(self, transition):
        self.memory.push(transition)
