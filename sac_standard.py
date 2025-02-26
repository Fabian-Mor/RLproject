import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils import soft_update, hard_update
from model import GaussianPolicy, QNetwork, DeterministicPolicy
from replay_memory import ReplayMemory
import numpy as np

# this file is adjusted from https://github.com/pranz24/pytorch-soft-actor-critic

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
        self.redQ = args.redQ
        self.crossQ = args.crossq
        self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size, batch_norm=args.batch_norm,
                               layer_norm=args.layer_norm, skip_connection=args.skip_connection, droQ=args.droQ, redQ=args.redQ, crossq=args.crossq).to(device=self.device)
        if self.crossQ:
            self.critic_optim = Adam(self.critic.parameters(), lr=args.lr, betas=(0.5, 0.999))
        else:
            self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)
        self.use_target = args.use_target
        if args.use_target and not args.crossq:
            self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size, batch_norm=args.batch_norm,
                                          layer_norm=args.layer_norm, skip_connection=args.skip_connection, droQ=args.droQ, redQ=args.redQ).to(self.device)
            hard_update(self.critic_target, self.critic)
        self.memory = ReplayMemory(args.replay_size, 1)
        self.batch_size = args.batch_size
        self.updates = 0
        if self.redQ or self.droQ:
            self.policy_update_interval = 20
        elif self.crossQ:
            self.policy_update_interval = 3
        else:
            self.policy_update_interval = 1

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            if self.crossQ:
                self.policy_optim = Adam(self.policy.parameters(), lr=args.lr, betas=(0.5, 0.999))
            else:
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

    def train(self, iter_fit=32):
        losses = []
        for i in range(iter_fit):
            # Sample a batch from memory
            state_batch, action_batch, reward_batch, next_state_batch, mask_batch = self.memory.sample(
                batch_size=self.batch_size)

            state_batch = torch.FloatTensor(state_batch).to(self.device)
            next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
            action_batch = torch.FloatTensor(action_batch).to(self.device)
            reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
            mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

            with torch.no_grad():

                next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
                if self.crossQ:
                    qf1_next_target_all, qf2_next_target_all = self.critic(
                        torch.cat([state_batch, next_state_batch], dim=0),
                        torch.cat([action_batch, next_state_action], dim=0)
                    )
                    qf1_next_target, qf2_next_target = qf1_next_target_all[state_batch.shape[0]:], qf2_next_target_all[
                                                                                                   state_batch.shape[
                                                                                                       0]:]
                if not self.crossQ:
                    if self.use_target:
                        qf_next_target_all = self.critic_target(next_state_batch, next_state_action)
                    else:
                        qf_next_target_all = self.critic(next_state_batch, next_state_action)

                    if self.critic.redQ:
                        indices = torch.randperm(10)[:2]
                        qf1_next_target, qf2_next_target = qf_next_target_all[:, indices[0]], qf_next_target_all[:,
                                                                                              indices[1]]
                        qf1_next_target = qf1_next_target.unsqueeze(1)
                        qf2_next_target = qf2_next_target.unsqueeze(1)
                    else:
                        qf1_next_target, qf2_next_target = qf_next_target_all

                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
                next_q_value = reward_batch + self.gamma * (min_qf_next_target - self.alpha * next_state_log_pi)

            qf_all = self.critic(state_batch, action_batch)
            if self.critic.redQ:
                indices = torch.randperm(10)[:2]
                qf1, qf2 = qf_all[:, indices[0]].unsqueeze(-1), qf_all[:, indices[1]].unsqueeze(-1)
            else:
                qf1, qf2 = qf_all

            qf1_loss = F.mse_loss(qf1, next_q_value)
            qf2_loss = F.mse_loss(qf2, next_q_value)
            qf_loss = qf1_loss + qf2_loss


            self.critic_optim.zero_grad()
            qf_loss.backward()
            self.critic_optim.step()

            if self.updates % self.policy_update_interval:
                pi, log_pi, _ = self.policy.sample(state_batch)

                qf_pi_all = self.critic(state_batch, pi)
                if self.critic.redQ:
                    qf1_pi, qf2_pi = qf_pi_all[:, indices[0]], qf_pi_all[:, indices[1]]
                else:
                    qf1_pi, qf2_pi = qf_pi_all

                min_qf_pi = torch.min(qf1_pi, qf2_pi)

                policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

                self.policy_optim.zero_grad()
                policy_loss.backward()
                self.policy_optim.step()

                if self.automatic_entropy_tuning:
                    alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
                    self.alpha_optim.zero_grad()
                    alpha_loss.backward()
                    self.alpha_optim.step()
                    self.alpha = self.log_alpha.exp()

            if self.use_target and self.updates % self.target_update_interval == 0:
                soft_update(self.critic_target, self.critic, self.tau)

            # losses.append((qf1_loss.item(), qf2_loss.item(), policy_loss.item()))
            self.updates += 1
        return # losses

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
