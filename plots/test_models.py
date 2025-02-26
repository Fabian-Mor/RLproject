import copy
from types import SimpleNamespace
import numpy as np
import itertools
import glicko2
import hockey.hockey_env as h_env
from importlib import reload

from main import show_progress
from sac_standard import SAC

args = SimpleNamespace(
        lr=0.0001,
        policy="Gaussian",
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        batch_size=1024,
        automatic_entropy_tuning=True,
        hidden_size=256,
        target_update_interval=1,
        replay_size=1000000,
        cuda=False,

        batch_norm = False,
        layer_norm = False,
        skip_connection = False,
    )


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


reload(h_env)
env = h_env.HockeyEnv()
action_space = h_env.HockeyEnv_BasicOpponent().action_space


agent1 = SAC(env.observation_space.shape[0], action_space, args3)
agent2 = SAC(env.observation_space.shape[0], action_space, args3)

agent3 = SAC(env.observation_space.shape[0], action_space, args4)
agent1.restore_state(ckpt_path="../checkpoints/self_train_checkpoints/droq_competition_16000", evaluate=False)
agent2.restore_state(ckpt_path="../checkpoints/self_train_checkpoints/droq_competition_26000", evaluate=True) #32000
agent3.restore_state(ckpt_path="../checkpoints/self_train_checkpoints/redq_20000", evaluate=True)
basic = h_env.BasicOpponent(weak=False)

# show_progress(env, 100000, agent2, 100, opponent=basic, render=False)
show_progress(env, 100000, agent1, 100, opponent=agent2, render=False)
# show_progress(env, 100000, agent1, 100, opponent=agent3, render=False)
# show_progress(env, 100000, agent2, 100, opponent=agent3, render=False)

