import sys
import gym
import moenvs
sys.path.append('./')
sys.path.append('../')
from tqdm import tqdm
import lib

import lib.common_ptan as ptan
import numpy as np
name = "Walker2d_MO_TD3_HER"
args = lib.utilities.settings.HYPERPARAMS[name]
env_main = gym.make(args.scenario_name)
env_main.seed(args.seed)
test_env_main = gym.make(args.scenario_name)
test_env_main.seed(args.seed)

def make_env(scenario_name, seed):
    def thunk():
        env = gym.make(scenario_name)
        env.seed = seed
        return env
    return thunk
all_seeds = [0,1,2,3]
envs = gym.vector.SyncVectorEnv([make_env(args.scenario_name, i) for i in all_seeds])
# make_env2 =  lambda: gym.make("Pendulum-v1", g=9.81)
# envs = gym.vector.SyncVectorEnv([make_env2() for i in all_seeds])
print(envs.action_space)
print(envs.single_action_space)
    #       args.obs_shape = env_main.observation_space.shape[0]
    # args.action_shape = env_main.action_space.shape[0]
    # args.reward_size = len(env_main.reward_space)
    # args.max_action = env_main.action_space.high
    # args.max_episode_len = env_main._max_episode_steps)
print(envs.action_space.sample())
envs.reset()
envs.step(envs.action_space.sample())
x = np.loadtxt("PD-MORL/interp_objs_walker2d.txt",delimiter=",")