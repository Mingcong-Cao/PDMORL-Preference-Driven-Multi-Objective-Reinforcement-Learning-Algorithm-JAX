from __future__ import absolute_import, division, print_function
import sys
import time


import torch
import random
import torch.optim as optim
import torch.multiprocessing as mp
import os

sys.path.append('../')
sys.path.append('./')
from tqdm import tqdm
import lib.utilities

import lib.common_ptan as ptan

import numpy as np

import gym
import moenvs
import itertools
from collections import namedtuple
from torch.utils.tensorboard import SummaryWriter
#jax related
import chex
import flashbax as fbx
from lib.models.networks_jax import Actor, Critic
import jax, flax, optax
import jax.numpy as jnp
from jax import jit
from flax.training.train_state import TrainState
from typing import Any, Callable, Sequence, Optional
from functools import partial
from flax import struct


class TrainState_actor(TrainState):
    target_params: flax.core.FrozenDict

class TrainState_critic(TrainState):
    target_params: flax.core.FrozenDict
    Q1: Callable = struct.field(pytree_node=False)

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state','terminal', 'preference', 'step_idx', 'p_id'])    

def generate_w_batch_test(step_size, reward_size):
    mesh_array = []
    step_size = step_size
    for i in range(reward_size):
        mesh_array.append(np.arange(0,1+step_size, step_size))
        
    w_batch_test = np.array(list(itertools.product(*mesh_array)))
    w_batch_test = w_batch_test[w_batch_test.sum(axis=1) == 1,:]
    w_batch_test = np.unique(w_batch_test,axis =0)
    
    return w_batch_test

# Evaluate agent for X episodes and returns average reward
def eval_agent(test_env, get_action, actor_state, args, preference, key, eval_episodes=1):
    
    avg_reward = np.zeros((eval_episodes,))
    avg_multi_obj = np.zeros((eval_episodes,args.reward_size))
    for eval_ep in range(eval_episodes):
        test_env.seed(eval_ep*10)
        test_env.action_space.seed(eval_ep*10)  
        # reset the environment
        state = test_env.reset()
        terminal = False
        tot_rewards = 0
        multi_obj = 0
        cnt = 0
        # Interact with the environment

        while not terminal:
            # if hasattr(agent, 'deterministic'):
            #     action = agent(state, preference, deterministic = True)
            # else:
            #     action = agent(state, preference)
            preference_ep, action, key = get_action(deterministic = True, actor_state =actor_state, states = state, preference = preference, key = key)
            next_state, reward, terminal, info = test_env.step(action)
            tot_rewards += np.dot(preference, reward)
            multi_obj +=  reward#
            state = next_state
            cnt += 1
        avg_reward[eval_ep]= tot_rewards
        avg_multi_obj[eval_ep]= multi_obj
    return avg_reward, avg_multi_obj, key

Queue_obj = namedtuple('Queue_obj', ['avg_reward', 'avg_multi_obj','process_ID'])


def generate_training_functions(
    action_shape: int,
    expl_noise: float,
    max_action: np.ndarray,
):  
    @partial(jit, static_argnums=(0,))  
    def get_action(
        deterministic: bool,
        actor_state: TrainState,
        states: np.ndarray,
        preference: np.ndarray,
        key: jnp.ndarray,
    ):
        if not deterministic:
            key, noise_key_ac, noise_key_pref = jax.random.split(key, 3)
            actions = actor_state.apply_fn(actor_state.params, states.reshape((1, np.prod(states.shape))), preference.reshape((1, np.prod(preference.shape)))).flatten()
            # TD3 Exploration
            preference = preference + (jax.random.normal(noise_key_pref, (preference.shape))*0.05).clip(-0.05, 0.05)
            preference = jnp.abs(preference)/jnp.linalg.norm(preference, ord = 1)
            print(actions)
            actions = actions + jax.random.normal(noise_key_ac, (action_shape,)) * jnp.sqrt(max_action * expl_noise)
            actions = actions.clip(-max_action,max_action)
        else:
            actions = actor_state.apply_fn(actor_state.params, states.reshape((1, np.prod(states.shape))), preference.reshape((1, np.prod(preference.shape)))).flatten()
        return preference, actions, key

    return get_action

def generate_learning_functions(
    policy_noise: float,
    noise_clip: float,
    max_action: np.ndarray,
    gamma: float,
    tau: float
):
    # @partial(jit, static_argnums=(0,))  
    @jax.jit
    def learn_critic(
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        terminals: np.ndarray,
        next_states: np.ndarray,
        w_batch: np.ndarray,
        actor_state: TrainState_actor,
        critic_state: TrainState_critic,
        key: jnp.ndarray,            
    ):
        #calculate the target Q values
        key, noise_key = jax.random.split(key, 2)
        noise = (jax.random.uniform(noise_key, actions.shape)*policy_noise).clip(-noise_clip, noise_clip)
        next_action_batch = (actor_state.apply_fn(actor_state.target_params, next_states, w_batch) + noise).clip(-max_action, max_action)
        target_Q1, target_Q2 = critic_state.apply_fn(critic_state.target_params, next_states, w_batch, next_action_batch)
        wTauQ1 = jax.lax.batch_matmul(jnp.expand_dims(w_batch, 1), jnp.expand_dims(target_Q1, 2)).squeeze(1)
        wTauQ2 = jax.lax.batch_matmul(jnp.expand_dims(w_batch, 1), jnp.expand_dims(target_Q2, 2)).squeeze(1)
        # print(target_Q1.shape, wTauQ1.shape)
        Tau_Q = jnp.where(wTauQ1 < wTauQ2, target_Q1, target_Q2)
        target_Q = gamma * jnp.multiply(Tau_Q, 1 - terminals.reshape(-1,1)) + rewards
        # print(Tau_Q.shape)
        
        def calc_loss(params):
            #originally we use smooth_l1_loss in pytorch with beta = 1.
            #here we replace it with huber loss with del = 1, which is identical
            current_Q1, current_Q2 = critic_state.apply_fn(params, states, w_batch, actions)
            return (optax.huber_loss(current_Q1, target_Q) + optax.huber_loss(current_Q2, target_Q)).mean(), (current_Q1.mean(), current_Q2.mean())
        
        (critic_loss, (current_Q1_mean, current_Q2_mean)), grads = jax.value_and_grad(calc_loss, has_aux=True)(critic_state.params)
        critic_state = critic_state.apply_gradients(grads=grads)
        print(target_Q.shape)

        return critic_state, critic_loss, current_Q1_mean, current_Q2_mean, key
    
    @jax.jit
    def learn_actor(
        states: np.ndarray,
        w_batch: np.ndarray,
        actor_state: TrainState_actor,
        critic_state: TrainState_critic,
        key: jnp.ndarray,            
    ):
        def actor_loss(params):
            Q1_critic = critic_state.apply_fn(critic_state.params, states, w_batch, actor_state.apply_fn(params, states, w_batch),  method = critic_state.Q1)
            wQ1 = jax.lax.batch_matmul(jnp.expand_dims(w_batch, 1), jnp.expand_dims(Q1_critic, 2)).squeeze(1)
            return (-wQ1).mean()
        #below in this function not modified yet
        actor_loss_value, grads = jax.value_and_grad(actor_loss)(actor_state.params)
        actor_state = actor_state.apply_gradients(grads=grads)
        actor_state = actor_state.replace(
            target_params = optax.incremental_update(actor_state.params, actor_state.target_params, tau)
        )
        critic_state = critic_state.replace(
            target_params = optax.incremental_update(critic_state.params, critic_state.target_params, tau)
        )
        return actor_state, critic_state, actor_loss_value
    return learn_critic, learn_actor

def child_process(preference, p_id, train_queue):

    start_time = time.time()
    args = lib.utilities.settings.HYPERPARAMS["Walker2d_MO_TD3_HER_Key"]
    args.p_id = p_id
    
    track = True
    writer = None
    if track:
        import wandb
        project_name = "Walker2d" 
        entity = None
        args = lib.utilities.settings.HYPERPARAMS["Walker2d_MO_TD3_HER_Key"]
        wandb.init(
                project=project_name,
                entity= entity,
                sync_tensorboard=True,
                config=vars(args),
                name= f"TD3_HER_KEY__{args.scenario_name}__seed{args.seed}__pid{args.p_id}__{int(time.time())}",
                monitor_gym=True,
                save_code=True,
            )
        writer = SummaryWriter(f"TD3_HER_KEY__jax__{args.scenario_name}__seed{args.seed}__pid{args.p_id}__{int(time.time())}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

    #we don't need device for jax
    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device
    # setup the environment

    torch.manual_seed(p_id*args.seed)
    random.seed(p_id*args.seed)
    np.random.seed(p_id*args.seed)
    os.environ['PYTHONHASHSEED'] = str(p_id*args.seed)
    torch.backends.cudnn.deterministic = True
    env = gym.make(args.scenario_name)
    env.seed(p_id*args.seed)
    env = gym.make(args.scenario_name)
    test_env = gym.make(args.scenario_name) 

    #Initialize environment related arguments
    args.obs_shape = env.observation_space.shape[0]
    args.action_shape = env.action_space.shape[0]
    args.reward_size = len(env.reward_space)
    args.max_action = env.action_space.high
    args.max_episode_len = env._max_episode_steps
        
    
    # Initialize critic and actor networks
    # actor = lib.models.networks.Actor(args)
    # critic = lib.models.networks.Critic(args)
    actor = Actor(
        state_size = args.obs_shape,
        action_size = args.action_shape,
        reward_size = args.reward_size,
        _layer_N = args.layer_N_actor,
        hidden_size = args.hidden_size,
        max_action = jnp.array(args.max_action)
    )
    critic = Critic(
        state_size = args.obs_shape,
        action_size = args.action_shape,
        reward_size = args.reward_size,
        _layer_N = args.layer_N_actor,
        hidden_size = args.hidden_size,
        max_action = jnp.array(args.max_action)
    )

    #initialize the train state that contains all parameters
    #generate random keys for jax
    key = jax.random.PRNGKey(args.seed)
    key, actor_key, critic_key = jax.random.split(key, 3)
    #dummy variables to initialze
    state_dum = jnp.ones((1,args.obs_shape))
    preference_dum = jnp.ones((1,args.reward_size))
    action_dum = jnp.ones((1,args.action_shape))
    actor_state = TrainState_actor.create(
        apply_fn = actor.apply,
        params =actor.init(actor_key, state_dum, preference_dum),
        target_params = actor.init(actor_key, state_dum, preference_dum),
        tx=optax.chain(optax.clip_by_global_norm(max_norm = 100), optax.adam(learning_rate=args.lr_actor)),
    )

    critic_state = TrainState_critic.create(
        apply_fn = critic.apply,
        Q1 = critic.Q1,
        params = critic.init(critic_key, state_dum, preference_dum, action_dum),
        target_params= critic.init(critic_key, state_dum, preference_dum, action_dum),
        tx=optax.chain(optax.clip_by_global_norm(max_norm = 100), optax.adam(learning_rate=args.lr_critic)),
    )

    #Edit the neural network model name
    args.name_model = args.name + " -Key"
    
    #Initialize Replay Buffer
    replay_buffer = ptan.experience_jax.ExperienceReplayBuffer(buffer_size = args.replay_size)
    
    # Main Loop
    done_episodes = 0
    max_metric = 0
    max_multi_obj_reward = 0
    # Only print the progress of first child process
    if p_id == 0:
        disable = False
    else:
        disable  = True

    history, cur_rewards = [], 0
    iter_idx = 0
    global_steps = 0
    s = env.reset() 
    if args.reward_size > 1:
        cur_rewards = np.zeros((args.reward_size)) 
        cur_steps = 0
    else:
        cur_rewards = 0
        cur_steps = 0
    total_rewards = []
    total_steps = []

    #initialize functions
    get_action = generate_training_functions(
                                action_shape = args.action_shape,
                                expl_noise =args.expl_noise,
                                max_action = args.max_action,
    )
    learn_critic, learn_actor = generate_learning_functions(
        policy_noise = args.policy_noise,
        noise_clip = args.noise_clip,
        max_action = args.max_action,
        gamma = args.gamma, 
        tau = args.tau,
    )

    start_time = time.time()
    for ts in tqdm(range(0, args.time_steps), disable=disable): #iterate through the fixed number of timesteps

        preference_ep = preference
        if global_steps < args.start_timesteps:
            action = env.action_space.sample()
        else:
            preference_ep, action, key = get_action(deterministic = False, actor_state =actor_state, states = s, preference = preference, key = key)
        
        preference_ep = np.array(preference_ep)
        action = np.array(action)
        s_next, r, done, info = env.step(np.array(action))
        this_experience = Experience(state=s, action=action, reward=r, next_state = s_next, terminal = done, preference = preference_ep, step_idx=cur_steps, p_id=p_id)
        #flashbax buffer
        # buffer_state = replay_buffer.add(buffer_state, this_experience)

        cur_steps += 1
        cur_rewards += r
        global_steps += 1
        s = s_next
        
        #add to buffer
        replay_buffer.add_one(this_experience)

        if done:
            s = env.reset()
            if args.reward_size > 1:
                total_rewards.append(cur_rewards)
                cur_rewards = np.zeros((args.reward_size)) 
                total_steps.append(cur_steps)
                cur_steps = 0
            else:
                total_rewards.append(cur_rewards)
                total_steps.append(cur_steps)
                cur_rewards = 0
                cur_steps = 0


        if len(replay_buffer.buffer) < 2*args.batch_size:
                continue 
        
        # Learn from the minibatch 

        batch = replay_buffer.sample(args.batch_size) 
        # # @partial(jit, static_argnums=(0,)) 
        # def unpack_batch(batch_size, batch):
        #     states, actions, rewards, terminals, next_states, preferences = [],[],[],[],[],[]
        #     for i in range(batch_size):
        #         exp = batch[i]
        #         states.append(jnp.array(exp.state))
        #         actions.append(exp.action)
        #         rewards.append(exp.reward)
        #         next_states.append(exp.next_state)
        #         terminals.append(exp.terminal)
        #         preferences.append(exp.preference)
        #     return jnp.array(states, copy=False,dtype=jnp.float32), jnp.array(actions,dtype=jnp.float32), \
        #            jnp.array(rewards, dtype=jnp.float32), \
        #            jnp.array(terminals, dtype=jnp.uint8), \
        #            jnp.array(next_states, copy=False,dtype=jnp.float32),\
        #            jnp.array(preferences, dtype=jnp.float32, copy=False)
        def unpack_batch(batch):
            states, actions, rewards, terminals, next_states, preferences = [],[],[],[],[],[]
            for _, exp in enumerate(batch):
                states.append(np.array(exp.state))
                actions.append(exp.action)
                rewards.append(exp.reward)
                next_states.append(exp.next_state)
                terminals.append(exp.terminal)
                preferences.append(exp.preference)
            return np.array(states, copy=False,dtype=np.float32), np.array(actions,dtype=np.float32), \
                   np.array(rewards, dtype=np.float32), \
                   np.array(terminals, dtype=np.uint8), \
                   np.array(next_states, copy=False,dtype=np.float32),\
                   np.array(preferences, dtype=np.float32, copy=False)

        #states = unpack_batch(args.batch_size, args.obs_shape, batch)
        states, actions, rewards, terminals, next_states, w_batch = unpack_batch(batch)
        critic_state, critic_loss, current_Q1_mean, current_Q2_mean, key = learn_critic(
            states = states,
            actions = actions,
            rewards = rewards,
            terminals = terminals,
            next_states = next_states,
            w_batch = w_batch,
            actor_state = actor_state,
            critic_state = critic_state,
            key = key,           
        )

        if global_steps % args.policy_freq == 0:
            actor_state, critic_state, actor_loss = learn_actor(
            states = states,
            w_batch = w_batch,
            actor_state= actor_state,
            critic_state = critic_state,
            key = key            
            )
            if global_steps % 5000 == 0:
                print(f"pref: {preference_ep}critic_l: {critic_loss} Q1: {current_Q1_mean} Q2: {current_Q2_mean} actor_l: {actor_loss}")

        if global_steps % 1000 == 0:
            #wrtie results
            if writer:
                now_time = time.time()
                writer.add_scalar("charts/SPS", ts/(now_time - start_time), global_steps)

    
        # new_rewards = exp_source.pop_total_rewards()

        if total_rewards:
            total_rewards = []
            done_episodes += 1
            #Evaluate agent
            if done_episodes % args.eval_freq == 0:
                avg_reward, avg_multi_obj, key = eval_agent(test_env,get_action, actor_state, args, preference, key, eval_episodes = 10)
                avg_reward_mean = avg_reward.mean()
                avg_reward_std = avg_reward.std()
                avg_multi_obj_mean = avg_multi_obj.mean(axis=0)
                avg_multi_obj_std = avg_multi_obj.std(axis=0)
                if max_metric <= avg_reward_mean:
                    max_metric = avg_reward_mean
                    max_multi_obj_reward = avg_multi_obj_mean
                    queue_obj = Queue_obj(avg_reward = max_metric, avg_multi_obj = max_multi_obj_reward, process_ID = p_id)
                print("\n---------------------------------------")
                print(f"Process {args.p_id} - Evaluation Episode {done_episodes}: Reward: {np.round(avg_reward_mean,2)}(std: +-{avg_reward_std}), Multi-Objective Reward: {avg_multi_obj_mean}(std: +-{avg_multi_obj_std}), Max_Metric: {max_multi_obj_reward}, Preference: {preference}")
                print("---------------------------------------")
                if writer:
                    writer.add_scalar("charts/reward_mean", avg_reward_mean, global_steps)
    
    
    avg_reward, avg_multi_obj, key = eval_agent(test_env, get_action, actor_state, args, preference, key, eval_episodes = 10)
    avg_reward_mean = avg_reward.mean()
    avg_reward_std = avg_reward.std()
    avg_multi_obj_mean = avg_multi_obj.mean(axis=0)
    avg_multi_obj_std = avg_multi_obj.std(axis=0)
    if max_metric <= avg_reward_mean:
        max_metric = avg_reward_mean
        max_multi_obj_reward = avg_multi_obj_mean
        queue_obj = Queue_obj(avg_reward = max_metric, avg_multi_obj = max_multi_obj_reward, process_ID = p_id)
    print("\n---------------------------------------")
    print(f"Process {args.p_id} - Evaluation Episode {done_episodes}: Reward: {np.round(avg_reward_mean,2)}(std: +-{avg_reward_std}), Multi-Objective Reward: {avg_multi_obj_mean}(std: +-{avg_multi_obj_std}), Max_Metric: {max_multi_obj_reward}, Preference: {preference}")
    print("---------------------------------------")
    print("Done in %d steps and %d episodes!" % (ts, done_episodes))
    print("Time Consumed")
    print("%0.2f minutes" % ((time.time() - start_time)/60))
    train_queue.put(queue_obj)

def main_parallel(process_count, reward_size):
    mp.set_start_method('spawn')
    os.environ['OMP_NUM_THREADS'] = "1"
    w_batch = generate_w_batch_test(0.001, reward_size) # w step size and number of objectives
    obj_num = process_count
    idx = np.round(np.linspace(0, len(w_batch)-1, num=obj_num)).astype(int)
    preference_array = w_batch[idx]
    torch.set_num_threads(process_count)
    # Initialize child processes
    train_queue_list = []
    data_proc_list = []
    for id in range(process_count):
        train_queue = mp.Queue(maxsize=1)
        data_proc = mp.Process(target=child_process,
                               args=(preference_array[id],id,train_queue))
        data_proc.start()
        train_queue_list.append(train_queue)
        data_proc_list.append(data_proc)
    for id in range(process_count):
        data_proc_list[id].join()
    return train_queue_list, data_proc_list

if __name__ == "__main__":

    # process_count = 3 # Number of key preferences
    #set the memory allocation for jax to avoid OOM error
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".25"
    process_count = 3 # setting processes num to only 1, debug use only
    reward_size = 2 # 2 objective problem
    train_queue_list, data_proc_list = main_parallel(process_count, reward_size)
    results = np.zeros((process_count,reward_size))
    
    for id in range(process_count):
        result = train_queue_list[id].get()
        results[id] = result[1]

    for p in data_proc_list:
        p.terminate()
        p.join() 

    f = open('interp_objs_walker2d_jax.txt', 'w')
    for t in results:
        line = ','.join(str(x) for x in t)
        f.write(line +' \n')
    f.close()

    
