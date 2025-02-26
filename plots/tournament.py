import copy
from types import SimpleNamespace
import numpy as np
import itertools
import glicko2
import hockey.hockey_env as h_env
from importlib import reload
import matplotlib.pyplot as plt
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
def run_tournament(agents, threshold=1.0, max_iterations=5):
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


base_agents = [Agent("weak"), Agent("strong")]
all_agents = [agent for agent in base_agents]

numbers = [0, 3000, 6000, 9000, 12000, 15000, 18000, 21000, 24000, 27000, 30000] #, 44000, 46000, 50000, 54000, 56000, 60000, 64000, 66000, 70000, 74000, 76000, 80000, 86000
agents = {}
for number in numbers:
    agents_number = [
            Agent(f"sac 20", path=f"../checkpoints/self_train_checkpoints/sac_self_20_{number}", args=args1),
            Agent(f"crossq 20", path=f"../checkpoints/self_train_checkpoints/crossq_self_20_{number}", args=args2),
            Agent(f"droq 20", path=f"../checkpoints/self_train_checkpoints/droq_self_{number}", args=args3),
            Agent(f"redq 20", path=f"../checkpoints/self_train_checkpoints/redq_self_{number}", args=args4),
        ]
    all_agents += agents_number
    agents[number] = agents_number

total_time = {"crossq": 8974/30000, "redq": 11633/30000, "droq": 8334/30000, "sac": 4216/30000}
run_tournament(all_agents)


# Print final ratings
print("\nFinal Ratings:")
for agent in sorted(all_agents, key=lambda agent: agent.rating.rating, reverse=True):
    print(f"{agent.name}: {agent.rating.rating:.2f}, RD: {agent.rating.rd:.2f}")
# Extract Elo ratings over time


agent_names = {agent.name for agents_list in agents.values() for agent in agents_list}
elo_history = {name: [] for name in agent_names}
rd_history = {name: [] for name in agent_names}

time_steps = numbers  # The time steps from your code

time_scaled = {name: [t * total_time[name] for t in time_steps] for name in total_time}

for number in time_steps:
    for agent in agents[number]:
        elo_history[agent.name].append((number, agent.rating.rating))
        rd_history[agent.name].append((number, agent.rating.rd))

# Plot Elo rating evolution with 95% confidence intervals
plt.rcParams.update({'font.size': 20})  # Increase global font size

plt.figure(figsize=(14, 7), dpi=300)
for agent_name, data in elo_history.items():
    data.sort()
    times, elos = zip(*data)
    rd_data = rd_history[agent_name]
    _, rds = zip(*rd_data)
    lower_bound = np.array(elos) - np.array(rds)
    upper_bound = np.array(elos) + np.array(rds)

    plt.plot(times, elos, label=agent_name)
    plt.fill_between(times, lower_bound, upper_bound, alpha=0.1)

# Differentiate weak and strong agents with dashed lines and colors
for base_agent in base_agents:
    base_elo = base_agent.rating.rating
    base_rd = base_agent.rating.rd
    lower_bound = base_elo - base_rd
    upper_bound = base_elo + base_rd
    plt.axhline(y=base_elo, linestyle="dashed", linewidth=2, label=base_agent.name, color="black" if base_agent.name == "weak" else "purple")
    plt.fill_between(time_steps, lower_bound, upper_bound, alpha=0.1, color="gray" if base_agent.name == "weak" else "pink")

plt.xlabel("Episodes")  # Moves x-label downward
plt.ylabel("Elo Rating")
plt.xlim(0, 30000)
plt.title("Evolution of Elo Ratings Over Episodes")
plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
plt.grid(True)

plt.tight_layout()
plt.show()

# Second plot with time scaling
plt.figure(figsize=(12, 7), dpi=300)
for agent_name, data in elo_history.items():
    data.sort()
    times = time_scaled.get(agent_name, time_steps)
    elos = [elo for _, elo in data]
    rd_data = rd_history[agent_name]
    _, rds = zip(*rd_data)
    lower_bound = np.array(elos) - np.array(rds)
    upper_bound = np.array(elos) + np.array(rds)

    plt.plot(times, elos, label=agent_name)
    plt.fill_between(times, lower_bound, upper_bound, alpha=0.1)

# Differentiate weak and strong agents with dashed lines and colors
for base_agent in base_agents:
    base_elo = base_agent.rating.rating
    base_rd = base_agent.rating.rd
    lower_bound = base_elo - base_rd
    upper_bound = base_elo + base_rd
    plt.axhline(y=base_elo, linestyle="dashed", linewidth=2, label=base_agent.name, color="black" if base_agent.name == "weak" else "purple")
    plt.fill_between(time_steps, lower_bound, upper_bound, alpha=0.1, color="gray" if base_agent.name == "weak" else "pink")

plt.xlim(0, 11633)
plt.xlabel("Time in seconds")
plt.ylabel("Elo Rating")
plt.title("Evolution of Elo Ratings Over Time")
plt.legend()
plt.grid(True)
plt.show()
