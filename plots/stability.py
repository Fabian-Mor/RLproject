import copy
from types import SimpleNamespace
import hockey.hockey_env as h_env
from sac_standard import SAC
from main import show_progress
import numpy as np
from importlib import reload
import matplotlib.pyplot as plt


args = SimpleNamespace(
        lr=0.0001,
        policy="Gaussian",
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        batch_size=256,
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
args2.batch_norm = True#
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
basic_env = h_env.HockeyEnv()
action_space = h_env.HockeyEnv_BasicOpponent().action_space
np.set_printoptions(suppress=True)
opponent = h_env.BasicOpponent(weak=False)
args_list = [args1, args2, args3, args4]

names = ["sac", "crossq", "droq", "redq"]
all_scores = {name: [] for name in names}
for name, arg in zip(names, args_list):
    path = f"../checkpoints/stability_test/{name}/"
    scores = []
    for i in range(10):
        agent = SAC(basic_env.observation_space.shape[0], action_space, arg)
        agent.restore_state(path + f"run_{i}", evaluate=True)
        score = show_progress(basic_env, 100000, agent, 100, opponent=opponent, verbose=False)
        scores.append(score)
    all_scores[name] = scores

plt.rcParams.update({'font.size': 20})  # Increase global font size

plt.figure(figsize=(12, 7), dpi=300)

# Box plot
plt.boxplot(all_scores.values(), labels=names, showmeans=False)

# Scatter individual scores
for i, name in enumerate(names):
    y_vals = all_scores[name]
    x_vals = np.full(len(y_vals), i + 1)  # Boxplot indexes start at 1
    plt.scatter(x_vals, y_vals, color='blue', alpha=0.6, label="Individual Scores" if i == 0 else "")

plt.xlabel("Agent Type")
plt.ylabel("Expected Score")
plt.ylim(-1, 1)
plt.title("Expected Scores against the strong opponent")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()
