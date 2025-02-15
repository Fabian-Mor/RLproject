import copy
import random

import numpy as np
import hockey.hockey_env as h_env
import gymnasium as gym
from importlib import reload
import optparse
from types import SimpleNamespace
import torch
from sac_standard import SAC


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(1)


# TODO: create szenario to train agent to not only go for balls in a straight line

def training(env, max_episodes, max_timesteps, agent, player2, train_iter):
    rewards = []
    losses = []
    timestep = 0
    for i_episode in range(1, max_episodes + 1):
        ob1, _info = env.reset()
        ob2 = env.obs_agent_two()
        total_reward = 0
        side_a = random.choice([True, False])
        # print("side_a", side_a)
        for t in range(max_timesteps):
            timestep += 1
            if side_a:
                a1 = agent.act(ob1)
                a2 = player2.act(ob2)
            else:
                a1 = player2.act(ob1)
                a2 = agent.act(ob2)

            (ob1_new, reward, done, trunc, _info) = env.step(np.hstack([a1, a2]))
            ob2_new = env.obs_agent_two()
            if side_a:
                # print(ob1, a1, reward, ob1_new, done)
                agent.store_transition((ob1, a1, reward, ob1_new, done))
            else:
                reward = env.get_reward_agent_two(env.get_info_agent_two())
                # print(ob2, a2, reward, ob2_new, done)
                agent.store_transition((ob2, a2, reward, ob2_new, done))
            total_reward += reward
            ob2 = ob2_new
            ob1 = ob1_new
            if done or trunc: break
        losses.extend(agent.train(train_iter))
        rewards.append(total_reward)

        if i_episode % 20 == 0:
            avg_reward = np.mean(rewards[-20:])
            print('Episode {} \t reward: {}'.format(i_episode, avg_reward))


def show_progress(env, max_timesteps, agent, show_runs, opponent=None):
    for _ in range(show_runs):
        ob1, info = env.reset()
        ob2 = env.obs_agent_two()
        for _ in range(max_timesteps):
            env.render()
            a1 = agent.act(ob1)
            if opponent is not None:
                a2 = opponent.act(ob2)
            else:
                a2 = [0, 0., 0, 0]
            obs, r, d, t, info = env.step(np.hstack([a1, a2]))
            ob1 = obs
            ob2 = env.obs_agent_two()
            if d or t: break
        print(info)
    env.close()


def main():
    optParser = optparse.OptionParser()
    optParser.add_option('-e', '--env', action='store', type='string',
                         dest='env_name', default="Pendulum-v1",
                         help='Environment (default %default)')
    optParser.add_option('-a', '--agent', action='store', type='string',
                         dest='agent', default="SAC",
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
                         dest='max_episodes', default=1000,
                         help='Number of episodes (default %default)')
    optParser.add_option('-u', '--update', action='store', type='float',
                         dest='update_every', default=100,
                         help='Number of episodes between target network updates (default %default)')
    optParser.add_option('-s', '--seed', action='store', type='int',
                         dest='seed', default=None,
                         help='Random seed (default %default)')
    opts, args = optParser.parse_args()

    args = SimpleNamespace(
        lr=0.0003,
        policy="Gaussian",
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        batch_size=128,
        automatic_entropy_tuning=True,
        hidden_size=256,
        target_update_interval=1,
        replay_size=100000000,
        cuda=False,
        use_target=True,
        batch_norm=False,
        layer_norm=False,
        skip_connection=False,
    )
    max_episodes = opts.max_episodes
    max_timesteps = 100000
    train_iter = opts.train
    eps = opts.eps
    lr = opts.lr
    random_seed = opts.seed
    train = True

    np.set_printoptions(suppress=True)
    # TODO: maybe include random agent to train shooting and defense in the beginning

    if train:
        reload(h_env)
        basic_env = h_env.HockeyEnv()
        agent = SAC(basic_env.observation_space.shape[0], basic_env.action_space, args)
        player2 = h_env.BasicOpponent(weak=False)

        # train shooting
        env = h_env.HockeyEnv(mode=h_env.Mode.TRAIN_DEFENSE)
        training(env, max_episodes, max_timesteps, agent, player2, train_iter)
        show_progress(env, max_timesteps, agent, 10)

        # train defense
        env = h_env.HockeyEnv(mode=h_env.Mode.TRAIN_SHOOTING)
        training(env, max_episodes, max_timesteps, agent, player2, train_iter)
        show_progress(env, max_timesteps, agent, 10)

        # train against weak
        env = h_env.HockeyEnv()
        training(env, max_episodes, max_timesteps, agent, player2, train_iter)

        # train against strong
        player2 = h_env.BasicOpponent(weak=False)
        training(env, max_episodes, max_timesteps, agent, player2, train_iter)

        agent.state("hockey", suffix="target")
        show_progress(env, max_timesteps, agent, 10, opponent=player2)

    else:
        args1 = copy.deepcopy(args)
        args1.use_target = True
        args1.batch_norm = False
        args1.layer_norm = False
        args1.skip_connection = False

        args2 = copy.deepcopy(args)
        args2.use_target = False
        args2.batch_norm = False
        args2.layer_norm = True
        args2.skip_connection = True

        reload(h_env)
        basic_env = h_env.HockeyEnv()
        player2 = h_env.BasicOpponent(weak=False)
        agent1 = SAC(basic_env.observation_space.shape[0], basic_env.action_space, args1)
        agent1.restore_state("checkpoints/sac_checkpoint_hockey_target", True)
        agent2 = SAC(basic_env.observation_space.shape[0], basic_env.action_space, args2)
        agent2.restore_state("checkpoints/sac_checkpoint_hockey_layer_skip", True)
        show_progress(basic_env, max_timesteps, agent2, 10, opponent=agent1)


if __name__ == '__main__':
    main()