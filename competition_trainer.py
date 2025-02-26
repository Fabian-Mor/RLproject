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
from random_agent import RandomAgent


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(1)



def training(env, max_episodes, max_timesteps, agent, player2, train_iter, warmup=False):
    rewards = []
    timestep = 0
    for i_episode in range(1, max_episodes + 1):
        ob1, _info = env.reset()
        ob2 = env.obs_agent_two()
        total_reward = 0
        for t in range(max_timesteps):
            timestep += 1
            a1 = agent.act(ob1)
            a2 = player2.act(ob2)

            (ob1_new, reward, done, trunc, _info) = env.step(np.hstack([a1, a2]))
            ob2_new = env.obs_agent_two()
            reward *= 10
            reward -= 1/50
            reward += 0.1 * (_info["reward_closeness_to_puck"] + _info["reward_touch_puck"])
            agent.store_transition((ob1, a1, reward, ob1_new, done))
            total_reward += reward
            ob2 = ob2_new
            ob1 = ob1_new
            if done or trunc:
                break

        agent.train(iter_fit=train_iter)
        rewards.append(total_reward)

        if i_episode % 20 == 0:
            avg_reward = np.mean(rewards[-20:])
            print('Episode {} \t reward: {}'.format(i_episode, avg_reward))
            if warmup and avg_reward > 0:
                print('ending warmup after {} episodes'.format(i_episode))
                return


def show_progress(env, max_timesteps, agent, show_runs, opponent=None, verbose=True, render=False):
    winner = []
    for _ in range(show_runs):
        ob1, info = env.reset()
        ob2 = env.obs_agent_two()
        for _ in range(max_timesteps):
            if render:
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
        winner.append(info["winner"])
    winner = np.array(winner)
    if verbose:
        print(np.mean(winner))
        print("score", np.count_nonzero(winner == 1), np.count_nonzero(winner == -1))
        print("draws: ", np.count_nonzero(winner == 0))
    env.close()
    return np.mean(winner)


def weighted_random_choice(lst):
    weights = np.arange(1, len(lst) + 1)
    index = np.random.choice(len(lst), p=weights / weights.sum())
    return lst[index]


def main():
    args = SimpleNamespace(
        lr=0.001, # 0.0003
        policy="Gaussian",
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        batch_size=256, # 512
        automatic_entropy_tuning=True,
        hidden_size=256,
        target_update_interval=1,
        replay_size=1000000,
        cuda=False,
        use_target=True,
        batch_norm=False,
        layer_norm=False,
        skip_connection=False,
        droQ=True,
        redQ=False,
        crossq=False,
    )
    max_timesteps = 100000
    if args.redQ or args.droQ:
        train_iter = 20
    else:
        train_iter = 1

    args1 = copy.deepcopy(args)
    args1.use_target = True
    args1.droQ = False
    args1.redQ = False
    args1.crossq = False

    args2 = copy.deepcopy(args)
    args2.use_target = False
    args2.droQ = False
    args2.redQ = False
    args2.crossq = True

    args3 = copy.deepcopy(args)
    args3.use_target = True
    args3.droQ = True
    args3.redQ = False
    args3.crossq = False

    args4 = copy.deepcopy(args)
    args4.use_target = True
    args4.droQ = False
    args4.redQ = True
    args4.crossq = False

    model_name = "droQ_competition_v6"
    load_model = "droq_competition_16000"
    self_play_episodes = 50001
    episodes_per_agent = 1
    add_to_self_play_episodes = 10000

    reload(h_env)
    basic_env = h_env.HockeyEnv()
    action_space = h_env.HockeyEnv_BasicOpponent().action_space
    np.set_printoptions(suppress=True)



    print("Starting self-play")
    # self training
    agent = SAC(basic_env.observation_space.shape[0], action_space, args3)
    agent.restore_state(ckpt_path=f"checkpoints/self_train_checkpoints/{load_model}")

    droQ_agents = [SAC(basic_env.observation_space.shape[0], action_space, args3) for _ in range(11)]
    redQagents = [SAC(basic_env.observation_space.shape[0], action_space, args4) for _ in range(2)]
    sacagents = [SAC(basic_env.observation_space.shape[0], action_space, args1) for _ in range(2)]
    crossQagents = [SAC(basic_env.observation_space.shape[0], action_space, args2) for _ in range(2)]
    self_play_opponents = droQ_agents + redQagents + sacagents + crossQagents
    names = [
        "droq_competition_16000", "droq_competition_30000", "droq_competition_34000", "droq_competition_20000", "droq_competition_6000",
        "droq_warmup_26000", "droq_34000", "droq_warmup_34000", "droq_26000", "droq_30000", "droq_warmup_10000",
             "redq_20000", "redq_6000",
             "sac_20000", "sac_30000",
             "crossq_16000", "crossq_20000"]
    for agent, name in zip(self_play_opponents, names):
        agent.restore_state(ckpt_path=f"checkpoints/self_train_checkpoints/{name}", evaluate=True)
        score = show_progress(basic_env, max_timesteps, agent, 100, opponent=h_env.BasicOpponent(weak=False),
                              verbose=True)
        print(score)
    basic_opponents = [h_env.BasicOpponent(weak=True), h_env.BasicOpponent(weak=False)]
    print(len(self_play_opponents), len(basic_opponents))

    acc = add_to_self_play_episodes
    train_basic = True


    agent = SAC(basic_env.observation_space.shape[0], action_space, args3)
    agent.restore_state(ckpt_path=f"checkpoints/self_train_checkpoints/{load_model}")

    for i in range(0, self_play_episodes, episodes_per_agent):
        if i % 100 == 0:
            score = show_progress(basic_env, max_timesteps, agent, 100, opponent=h_env.BasicOpponent(weak=False), verbose=False)
            if score > 0.8:
                train_basic = False
            else:
                train_basic = True
            print(f"Episode: {i}: {score}, {train_basic}")
        if train_basic:
            opp = np.random.choice(basic_opponents)
        else:
            opp = np.random.choice(self_play_opponents)

        training(basic_env, episodes_per_agent, max_timesteps, agent, opp, train_iter)
        if i >= acc:
            print(show_progress(basic_env, max_timesteps, agent, 100, opponent=h_env.BasicOpponent(weak=True)))
            print(show_progress(basic_env, max_timesteps, agent, 100, opponent=h_env.BasicOpponent(weak=False)))
            agent.state("hockey", ckpt_path=f"checkpoints/self_train_checkpoints/{model_name}_{i}")
            agent_self_play = SAC(basic_env.observation_space.shape[0], action_space, args)
            agent_self_play.restore_state(f"checkpoints/self_train_checkpoints/{model_name}_{i}", evaluate=True)
            self_play_opponents.append(agent_self_play)
            acc += add_to_self_play_episodes

    agent.state("hockey", suffix=f"{model_name}")


if __name__ == '__main__':
    main()