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
import time


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(1)



def training(env, max_episodes, max_timesteps, agent, player2, train_iter, warmup=False):
    rewards = []
    timestep = 0
    steps = 0
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
            steps += 1

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
        droQ=False,
        redQ=False,
        crossq=False,
    )
    max_timesteps = 100000
    if args.redQ or args.droQ:
        train_iter = 20
    else:
        train_iter = 1

    stability_test = False
    stability_runs = 10

    train = True
    model_name = "sac"
    load_model = ""
    warmup = True
    warmup_max_episodes = 1000
    self_play = True
    warmup_episodes = 1500 # 500 # standard 1000
    basic_episodes = 0 # 500 # standard 10000
    self_play_episodes = 27001
    episodes_per_agent = 1
    add_to_self_play_episodes = 3000

    reload(h_env)
    basic_env = h_env.HockeyEnv()
    action_space = h_env.HockeyEnv_BasicOpponent().action_space
    np.set_printoptions(suppress=True)


    if stability_test:
        for i in range(stability_runs):
            agent = SAC(basic_env.observation_space.shape[0], action_space, args)
            player2 = h_env.BasicOpponent(weak=True)
            if warmup:
                # train shooting
                env = h_env.HockeyEnv(mode=h_env.Mode.TRAIN_SHOOTING)
                training(env, warmup_max_episodes, max_timesteps, agent, player2, train_iter, warmup=True)
                # train defense
                env = h_env.HockeyEnv(mode=h_env.Mode.TRAIN_DEFENSE)
                training(env, warmup_max_episodes, max_timesteps, agent, player2, train_iter, warmup=True)

            # train against weak
            env = h_env.HockeyEnv()
            training(env, basic_episodes, max_timesteps, agent, player2, train_iter)

            # train against strong
            player2 = h_env.BasicOpponent(weak=False)
            training(env, basic_episodes, max_timesteps, agent, player2, train_iter)

            agent.state("hockey", ckpt_path=f"checkpoints/stability_test/{model_name}/run_{i}")


    if train:
        agent = SAC(basic_env.observation_space.shape[0], action_space, args)
        if load_model != "":
            agent.restore_state(load_model, evaluate=False)
        random_player = RandomAgent(action_space)
        player2 = h_env.BasicOpponent(weak=True)
        basic_opponents = [h_env.BasicOpponent(weak=True), h_env.BasicOpponent(weak=False)]

        agent.state("hockey", ckpt_path=f"checkpoints/self_train_checkpoints/{model_name}_0")
        if warmup:
            # train shooting
            env = h_env.HockeyEnv(mode=h_env.Mode.TRAIN_SHOOTING)
            training(env, warmup_episodes, max_timesteps, agent, random_player, train_iter, warmup=False)
            # train defense
            env = h_env.HockeyEnv(mode=h_env.Mode.TRAIN_DEFENSE)
            training(env, warmup_episodes, max_timesteps, agent, random_player, train_iter, warmup=False)

        env = h_env.HockeyEnv()
        if self_play:
            start_time = time.time()
            print("Starting self-play")
            # self training
            agent.state("hockey", ckpt_path=f"checkpoints/self_train_checkpoints/{model_name}_3000")
            agent_self_play = SAC(basic_env.observation_space.shape[0], action_space, args)
            agent_self_play.restore_state(f"checkpoints/self_train_checkpoints/{model_name}_3000", evaluate=True)
            self_play_opponents = [agent_self_play]
            acc = add_to_self_play_episodes
            train_basic = True
            train_weak = True
            for i in range(0, self_play_episodes, episodes_per_agent):
                if i % 100 == 0:
                    score = show_progress(env, max_timesteps, agent, 100, opponent=h_env.BasicOpponent(weak=False), verbose=False)
                    score_weak = show_progress(env, max_timesteps, agent, 100, opponent=h_env.BasicOpponent(weak=True),
                                          verbose=False)
                    if score_weak > 0:
                        train_weak = False
                    else:
                        train_weak = True
                    if score > 0:
                        train_basic = False
                    else:
                        train_basic = True
                    print(f"Episode: {i}: {score}, {score_weak}")
                if train_weak:
                    opp = h_env.BasicOpponent(weak=True)
                elif train_basic:
                    opp = np.random.choice(basic_opponents)
                else:
                    opp = weighted_random_choice(self_play_opponents)

                training(env, episodes_per_agent, max_timesteps, agent, opp, train_iter)
                if i >= acc:
                    print(show_progress(env, max_timesteps, agent, 100, opponent=h_env.BasicOpponent(weak=True)))
                    print(show_progress(env, max_timesteps, agent, 100, opponent=h_env.BasicOpponent(weak=False)))
                    print(time.time() - start_time)
                    agent.state("hockey", ckpt_path=f"checkpoints/self_train_checkpoints/{model_name}_{i+3000}")
                    agent_self_play = SAC(basic_env.observation_space.shape[0], action_space, args)
                    agent_self_play.restore_state(f"checkpoints/self_train_checkpoints/{model_name}_{i+3000}", evaluate=True)
                    self_play_opponents.append(agent_self_play)
                    acc += add_to_self_play_episodes

        agent.state("hockey", suffix=f"{model_name}")
        show_progress(env, max_timesteps, agent, 100, opponent=player2)

    else:
        args1 = copy.deepcopy(args)
        args1.use_target = True
        args1.batch_norm = False
        args1.layer_norm = False
        args1.skip_connection = False
        args1.droQ = False

        args2 = copy.deepcopy(args)
        args2.use_target = False
        args2.batch_norm = True
        args2.layer_norm = False
        args2.skip_connection = False
        args2.droQ = False

        args3 = copy.deepcopy(args)
        args3.use_target = True
        args3.batch_norm = False
        args3.layer_norm = True
        args3.skip_connection = True
        args3.droQ = False

        args4 = copy.deepcopy(args)
        args4.use_target = True
        args4.batch_norm = False
        args4.layer_norm = False
        args4.skip_connection = False
        args4.droQ = True

        player2 = h_env.BasicOpponent(weak=False)
        random_player = RandomAgent(action_space)


        droq0 = SAC(basic_env.observation_space.shape[0], action_space, args4)
        droq0.restore_state("checkpoints/sac_checkpoint_hockey_droQ", True)
        droq1 = SAC(basic_env.observation_space.shape[0], action_space, args4)
        droq1.restore_state("checkpoints/sac_checkpoint_hockey_droQ_new_reward", True)


        show_progress(basic_env, max_timesteps, droq0, 100, opponent=droq1)
        show_progress(basic_env, max_timesteps, droq0, 100, opponent=player2)
        show_progress(basic_env, max_timesteps, droq1, 100, opponent=player2)

        basic = SAC(basic_env.observation_space.shape[0], action_space, args1)
        basic.restore_state("checkpoints/stability_test/sac/run_0", True)
        crossq = SAC(basic_env.observation_space.shape[0], action_space, args2)
        crossq.restore_state("checkpoints/stability_test/crossq/run_0", True)
        deep = SAC(basic_env.observation_space.shape[0], action_space, args3)
        deep.restore_state("checkpoints/stability_test/deep/run_0", True)
        droq = SAC(basic_env.observation_space.shape[0], action_space, args4)
        droq.restore_state("checkpoints/stability_test/droq/run_0", True)

        print("sac vs strong")
        show_progress(basic_env, max_timesteps, basic, 100, opponent=player2)

        print("crossq vs strong")
        show_progress(basic_env, max_timesteps, crossq, 100, opponent=player2)

        print("deep vs strong")
        show_progress(basic_env, max_timesteps, deep, 100, opponent=player2)

        print("droq vs strong")
        show_progress(basic_env, max_timesteps, droq, 100, opponent=player2)


if __name__ == '__main__':
    main()