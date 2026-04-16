from toolkit import test_and_visualize_agents, create_env_and_train_agents
from stable_baselines3.common.vec_env import DummyVecEnv
from main import StockTradingEnv
import sys
from pathlib import Path

# Get the absolute path to the directory containing 'data'
# This assumes your current script is in a folder at the same level as 'data'
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

# Now you can import using the folder name as the module
from data.data_engineer import load_split_data

test_data = load_split_data('test')
training_data = load_split_data('train')
validation_data = load_split_data('validate')

# Create the environment and train the agents
threshold = 0.1
total_timesteps = 10000
train_env, val_env, ppo_agent, a2c_agent, ddpg_agent, sac_agent, td3_agent, ensemble_agent = \
  create_env_and_train_agents(training_data, validation_data, total_timesteps, threshold)

n_tests = 1000
agents = {
    'PPO Agent': ppo_agent,
    'A2C Agent': a2c_agent,
    'DDPG Agent': ddpg_agent,
    'SAC Agent': sac_agent,
    'TD3 Agent': td3_agent,
    'Ensemble Agent': ensemble_agent
}

test_and_visualize_agents(train_env, agents, training_data, n_tests=n_tests)

test_env = DummyVecEnv([lambda: StockTradingEnv(test_data)])
test_and_visualize_agents(test_env, agents, test_data, n_tests=n_tests)