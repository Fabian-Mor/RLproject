import copy
from types import SimpleNamespace
import numpy as np
import itertools
import glicko2
import hockey.hockey_env as h_env
from importlib import reload
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
    )


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


reload(h_env)
env = h_env.HockeyEnv()
action_space = h_env.HockeyEnv_BasicOpponent().action_space
# Define initial Glicko-2 parameters
INITIAL_RATING = 1500
INITIAL_RD = 350  # Rating deviation
INITIAL_VOLATILITY = 0.06  # Default value


# Define a class for agents with Glicko-2 ratings
class Agent:
    def __init__(self, name, path="", args=None):
        self.name = name
        if name == 'weak':
            self.agent = h_env.BasicOpponent(weak=True)
        elif name == 'strong':
            self.agent = h_env.BasicOpponent(weak=False)
        else:
            self.agent = SAC(env.observation_space.shape[0], action_space, args)
            self.agent.restore_state(path, evaluate=True)
        self.rating = glicko2.Player(rating=INITIAL_RATING, rd=INITIAL_RD, vol=INITIAL_VOLATILITY)

    def act(self, obs):
        return self.agent.act(obs)


# Function to simulate a match and return the result (1 win, 0.5 draw, 0 loss)
def play_match(agent1, agent2, max_timesteps=100000):
    """Simulate a match between two agents and return match result (1 win, 0.5 draw, 0 loss)."""
    ob1, info = env.reset()
    ob2 = env.obs_agent_two()
    for _ in range(max_timesteps):
        a1 = agent1.act(ob1)
        a2 = agent2.act(ob2)
        obs, r, d, t, info = env.step(np.hstack([a1, a2]))
        ob1 = obs
        ob2 = env.obs_agent_two()
        if d or t:
            break
    winner = info['winner']
    if winner == 1:
        return 1, 0
    elif winner == 0:
        return 0.5, 0.5
    else:
        return 0, 1


# Function to run a round-robin tournament
def run_tournament(agents, threshold=1.0, max_iterations=1000000):
    """Run a round-robin tournament where every agent plays against every other agent."""
    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        prev_ratings = {agent.name: agent.rating.rating for agent in agents}

        for agent1, agent2 in itertools.combinations(agents, 2):
            result1, result2 = play_match(agent1, agent2)
            # Update ratings
            agent1.rating.update_player([agent2.rating.rating], [agent2.rating.rd], [result1])
            agent2.rating.update_player([agent1.rating.rating], [agent1.rating.rd], [result2])

        # Compute max rating change
        max_change = max(abs(agent.rating.rating - prev_ratings[agent.name]) for agent in agents)

        # Stop if rating change is below the threshold
        if max_change < threshold:
            break


agents = [
    Agent("strong"),
    Agent("basic", path="../checkpoints/sac_checkpoint_hockey_basic_new_reward", args=args1),
    Agent("crossq", path="../checkpoints/sac_checkpoint_hockey_crossq_new_reward", args=args2),
    Agent("deep", path="../checkpoints/sac_checkpoint_hockey_deep_new_reward", args=args3),
    Agent("droq", path="../checkpoints/sac_checkpoint_hockey_droq_new_reward", args=args4),
]

# Run tournament
run_tournament(agents)

# Print final ratings
print("\nFinal Ratings:")
for agent in agents:
    print(f"{agent.name}: {agent.rating.rating:.2f}, RD: {agent.rating.rd:.2f}")

