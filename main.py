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
                         dest='max_episodes', default=2000,
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
        batch_norm=False
    )
    max_episodes = opts.max_episodes
    max_timesteps = 100000
    train_iter = opts.train
    eps = opts.eps
    lr = opts.lr
    random_seed = opts.seed

    np.set_printoptions(suppress=True)
    reload(h_env)

    # train defense
    env = h_env.HockeyEnv(mode=h_env.Mode.TRAIN_DEFENSE)
    agent = SAC(env.observation_space.shape[0], env.action_space, args)
    player2 = h_env.BasicOpponent(weak=False)
    o, info = env.reset()

    rewards = []
    losses = []
    timestep = 0
    for i_episode in range(1, max_episodes + 1):
        ob, _info = env.reset()
        total_reward = 0
        for t in range(max_timesteps):
            timestep += 1
            done = False
            a1 = agent.act(ob)
            a2 = [0,0.,0,0]
            (ob_new, reward, done, trunc, _info) = env.step(np.hstack([a1, a2]))
            agent.store_transition((ob, a1, reward, ob_new, done))
            total_reward += reward
            ob = ob_new
            if done or trunc: break
        losses.extend(agent.train(train_iter))
        rewards.append(total_reward)

    rewards = []
    lengths = []
    losses = []
    timestep = 0

    # train against agent
    env = h_env.HockeyEnv()
    for i_episode in range(1, max_episodes + 1):
        ob, _info = env.reset()
        obs_agent2 = env.obs_agent_two()
        agent.reset()
        total_reward = 0
        for t in range(max_timesteps):
            timestep += 1
            done = False
            a = agent.act(ob)
            a2 = player2.act(obs_agent2)
            (ob_new, reward, done, trunc, _info) = env.step(np.hstack([a, a2]))
            agent.store_transition((ob, a, reward, ob_new, done))
            total_reward += reward
            ob = ob_new
            obs_agent2 = env.obs_agent_two()
            if done or trunc:
                break

        losses.extend(agent.train(train_iter))

        rewards.append(total_reward)
        lengths.append(t)

        if i_episode % 20 == 0:
            avg_reward = np.mean(rewards[-20:])
            avg_length = int(np.mean(lengths[-20:]))
            print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, avg_reward))


    o, info = env.reset()
    _ = env.render()
    obs_buffer = []
    reward_buffer = []
    obs, info = env.reset()
    obs_agent2 = env.obs_agent_two()
    for _ in range(251):
        env.render()
        a1 = agent.act(obs)
        a2 = player2.act(obs_agent2)
        obs, r, d, t, info = env.step(np.hstack([a1, a2]))
        obs_buffer.append(obs)
        reward_buffer.append(r)
        obs_agent2 = env.obs_agent_two()
        if d or t: break
    obs_buffer = np.asarray(obs_buffer)
    reward_buffer = np.asarray(reward_buffer)
    print(np.mean(obs_buffer,axis=0))
    print(np.std(obs_buffer,axis=0))

if __name__ == '__main__':
    main()