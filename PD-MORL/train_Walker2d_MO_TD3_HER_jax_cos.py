from __future__ import absolute_import, division, print_function
import sys
# importing time module
import time

from tensorboardX import SummaryWriter
from datetime import datetime
import torch
import random
import torch.optim as optim
import torch.multiprocessing as mp
import os

sys.path.append('./')
sys.path.append('../')
from tqdm import tqdm
import lib

import lib.common_ptan as ptan
from lib.utilities import MORL_utils_jax

import numpy as np
import gym
import moenvs
from scipy.interpolate import RBFInterpolator

from collections import namedtuple, deque
import copy

#jax packages
from lib.models.networks_jax import Actor, Critic
import jax, flax, optax
import jax.numpy as jnp
from jax import jit
from flax.training.train_state import TrainState
from typing import Any, Callable, Sequence, Optional
from functools import partial
from flax import struct
from datetime import datetime

jax.default_device(jax.devices()[1])
Queue_obj = namedtuple('Queue_obj', ['ep_samples', 'time_step','ep_cnt','process_ID'])
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state','terminal', 'preference', 'step_idx'])    

class TrainState_actor(TrainState):
    target_params: flax.core.FrozenDict

class TrainState_critic(TrainState):
    target_params: flax.core.FrozenDict
    Q1: Callable = struct.field(pytree_node=False)


def cosine_similarity(
    x1: jnp.ndarray,
    x2: jnp.ndarray,
    eps: float = 1e-8,
):
    #compute the cosine similarity as defined in pytorch https://pytorch.org/docs/stable/generated/torch.nn.CosineSimilarity.html
    #x1, x2 should be 2d arrays
    #return a 1-d array with cosine similarities between two corresponding elements in x1, x2
    dotted = jax.lax.batch_matmul(jnp.expand_dims(x1, 1), jnp.expand_dims(x2, 2)).squeeze()
    # print("dotted",dotted)
    norm1 = jnp.linalg.norm(x1, axis = 1, ord = 2)
    norm2 = jnp.linalg.norm(x2, axis = 1,ord = 2)
    # print("norm",norm1*norm2)
    similarity = dotted / jnp.maximum(norm1*norm2, eps)
    return similarity


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
            # TD3 Exploration do not change preference here
            # preference = preference + (jax.random.normal(noise_key_pref, (preference.shape))*0.05).clip(-0.05, 0.05)
            # preference = jnp.abs(preference)/jnp.linalg.norm(preference, ord = 1)
            # print(actions)
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
    tau: float,
    actor_loss_coeff: float,
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
        w_batch_interp: np.ndarray,
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
            #try different angle term here
            angle_term_1 = 45 * cosine_similarity(w_batch_interp, current_Q1) #rad to degree
            angle_term_2 = 45 * cosine_similarity(w_batch_interp, current_Q2)
            return (optax.huber_loss(current_Q1, target_Q) + optax.huber_loss(current_Q2, target_Q)).mean() \
                 - angle_term_1.mean() - angle_term_2.mean() , (current_Q1.mean(), current_Q2.mean(), angle_term_1.mean(), angle_term_2.mean())
        
        (critic_loss, (current_Q1_mean, current_Q2_mean, angle_term_1_mean, angle_term_2_mean)), grads = jax.value_and_grad(calc_loss, has_aux=True)(critic_state.params)
        critic_state = critic_state.apply_gradients(grads=grads)
        print(target_Q.shape)
        global_norm = optax.global_norm(grads)
        return critic_state, critic_loss, current_Q1_mean, current_Q2_mean, angle_term_1_mean, angle_term_2_mean, global_norm, grads, key
    
    @jax.jit
    def learn_actor(
        states: np.ndarray,
        w_batch: np.ndarray,
        w_batch_interp: np.ndarray,
        actor_state: TrainState_actor,
        critic_state: TrainState_critic,
        key: jnp.ndarray,            
    ):
        def actor_loss(params):
            Q1_critic = critic_state.apply_fn(critic_state.params, states, w_batch, actor_state.apply_fn(params, states, w_batch),  method = critic_state.Q1)
            wQ1 = jax.lax.batch_matmul(jnp.expand_dims(w_batch, 1), jnp.expand_dims(Q1_critic, 2)).squeeze(1)
            angle_term = 45 * cosine_similarity(w_batch_interp, Q1_critic)
            return (-wQ1).mean() - actor_loss_coeff * angle_term.mean(), angle_term.mean()
        #below in this function not modified yet
        (actor_loss_value, angle_term_mean), grads = jax.value_and_grad(actor_loss, has_aux=True)(actor_state.params)
        actor_state = actor_state.apply_gradients(grads=grads)
        global_norm = optax.global_norm(grads)
        actor_state = actor_state.replace(
            target_params = optax.incremental_update(actor_state.params, actor_state.target_params, tau)
        )
        critic_state = critic_state.replace(
            target_params = optax.incremental_update(critic_state.params, critic_state.target_params, tau)
        )
        return actor_state, critic_state, actor_loss_value,angle_term_mean,  global_norm
    return learn_critic, learn_actor

class SyncVectorEnv():

    def __init__(self, func_list):
        self.envs_list = []
        self.num_env = len(func_list)
        for i in func_list:
            self.envs_list.append(i())

    def reset(self):
        s = []
        for i in self.envs_list:
            s.append(i.reset())
        return np.array(s)
    
    def step_subenv(self, subenv_num, action):
        return self.envs_list[subenv_num].step(action)
    
    def reset_subenv(self, subenv_num):
        return self.envs_list[subenv_num].reset()
    
    def sample_action(self):
        actions = []
        for i in self.envs_list:
            actions.append(i.action_space.sample())
        return np.array(actions)

if __name__ == "__main__":
    # mp.set_start_method('spawn')
    # os.environ['OMP_NUM_THREADS'] = "1"
    start_time = time.time()
    name = "Walker2d_MO_TD3_HER"
    args = lib.utilities.settings.HYPERPARAMS[name]
    args.plot_name  = name
    PROCESSES_COUNT = args.process_count
    # torch.set_num_threads(PROCESSES_COUNT)

    #TODO: need to modify/this is just for test purpose
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".1"
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    # os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.backends.cudnn.deterministic = True
    
    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device
    # setup the environment
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
    # PROCESSES_COUNT = 1
    #envs = gym.vector.AsyncVectorEnv([make_env(args.scenario_name, i*args.seed) for i in range(PROCESSES_COUNT)], context = 'spawn')
    envs = SyncVectorEnv([make_env(args.scenario_name, i*args.seed) for i in range(PROCESSES_COUNT)])
    # envs = gym.vector.make(args.scenario_name, asynchronous=True, num_envs=PROCESSES_COUNT)
    #Initialize environment related arguments
    args.obs_shape = env_main.observation_space.shape[0]
    args.action_shape = env_main.action_space.shape[0]
    args.reward_size = len(env_main.reward_space)
    args.max_action = env_main.action_space.high
    args.max_episode_len = env_main._max_episode_steps
          
    
    #Writer object for Tensorboard
    log_dir = "runs/" + datetime.now().strftime("%d %b %Y, %H-%M") + ' MO_TD3_HER_JAX '+args.scenario_name
    writer = SummaryWriter(log_dir, comment = ' TD3_HER_JAX'+args.scenario_name)
    # #Initialize the networks
    #actor = lib.models.networks.Actor(args)
    # critic = lib.models.networks.Critic(args)
    # writer = None

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
    key_arr = jax.random.split(key, (PROCESSES_COUNT, ))
    #dummy variables for initialze
    state_dum = jnp.ones((1,args.obs_shape))
    preference_dum = jnp.ones((1,args.reward_size))
    action_dum = jnp.ones((1,args.action_shape))

    #initialize the states
    actor_state = TrainState_actor.create(
        apply_fn = actor.apply,
        params =actor.init(actor_key, state_dum, preference_dum),
        target_params = actor.init(actor_key, state_dum, preference_dum),
        #trying adaptive clipping
        # tx = optax.chain(optax.adaptive_grad_clip(clipping = 0.01), optax.adam(learning_rate=args.lr_actor))
        tx=optax.chain(optax.clip_by_global_norm(max_norm = 10.0), optax.adam(learning_rate=args.lr_actor)),
    )

    critic_state = TrainState_critic.create(
        apply_fn = critic.apply,
        Q1 = critic.Q1,
        params = critic.init(critic_key, state_dum, preference_dum, action_dum),
        target_params= critic.init(critic_key, state_dum, preference_dum, action_dum),
        #tx = optax.chain(optax.adaptive_grad_clip(clipping = 0.01), optax.adam(learning_rate=args.lr_critic))
        tx=optax.chain(optax.clip_by_global_norm(max_norm = 10.0), optax.adam(learning_rate=args.lr_critic)),
    )

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
        actor_loss_coeff = args.actor_loss_coeff,
    )

    get_action_batch = jax.vmap(get_action, in_axes=(None, None, 0, 0, 0))

    #Edit the neural network model name
    args.name_model = args.name

    #Load previously trained model
    if args.load_model == True:
        load_path = "Exps/{}/".format(name)
        model_actor = torch.load("{}{}.pkl".format(load_path,"{}_{}_{}".format(args.scenario_name, args.name_model,'final_actor'))) # Change the model name accordingly
        actor = lib.models.networks.Actor(args)
        actor.load_state_dict(model_actor)
        model_critic = torch.load("{}{}.pkl".format(load_path,"{}_{}_{}".format(args.scenario_name, args.name_model,'final_critic'))) # Change the model name accordingly
        critic = lib.models.networks.Critic(args)
        critic.load_state_dict(model_critic)
        
    # actor.share_memory()
    # critic.share_memory()

    #Initialize preference spaces
    w_batch_test = lib.utilities.MORL_utils.generate_w_batch_test(args, step_size = args.w_step_size)
    w_batch_eval = lib.utilities.MORL_utils.generate_w_batch_test(args, step_size = 0.005)
    w_batch_test_split = np.array_split(w_batch_test,PROCESSES_COUNT)

    #Initialize Experience Source and Replay Buffer
    # exp_source_main = ptan.experience.MORLExperienceSource(env_main, agent_main, args, steps_count=1)
    replay_buffer_main = ptan.experience_jax.ExperienceReplayBuffer_HER_MO(args)

    # Initialize Interpolator
    x = np.loadtxt("interp_objs_walker2d.txt",delimiter=",")
    x_unit = x/np.linalg.norm(x,ord=2,axis=1,keepdims=True)
    idx_w_batch = np.round(np.linspace(0, len(w_batch_test)-1, num=len(x))).astype(int)
    w_batch_interp = w_batch_test[idx_w_batch]
    interp = RBFInterpolator(w_batch_interp, x_unit, kernel= 'linear')
    # agent_main.interp = interp

    # Main Loop
    done_episodes = 0
    time_step = 0
    process_step_array = np.zeros((PROCESSES_COUNT,))
    process_episode_array = np.zeros((PROCESSES_COUNT,))
    eval_cnt = 1
    eval_cnt_ep = 1
    global_steps = 0

    preferences = np.zeros((PROCESSES_COUNT,args.reward_size))
    # reset_preferences = np.full(PROCESSES_COUNT, True)
    for i in range(PROCESSES_COUNT):
        #set initialpreferences
        w_batch_i = w_batch_test_split[i]
        preferences[i] = w_batch_i[np.random.randint(len(w_batch_i))]
    
    s_all = envs.reset()
    total_learning_steps = 0

    figure_dir = 'Figures/{}/{}'.format(args.scenario_name, log_dir.split('/')[1])
    if not os.path.exists(figure_dir):
        os.mkdir(figure_dir)
    try:
        for ts in tqdm(range(0, args.time_steps)): #iterate through the fixed number of timesteps

            if global_steps < args.start_timesteps:
                actions = envs.sample_action()
            else:        
                preference_ep, actions, key_arr = get_action_batch(False,actor_state, s_all, preferences, key_arr)
                actions = np.array(actions)
            
            s_next_all = []
            for i in range(PROCESSES_COUNT):
                #store in replay buffer
                #Check time step limit
                s_next, r, done, info = envs.step_subenv(i, actions[i])            
                this_experience = Experience(state=s_all[i], action=actions[i], reward=r, next_state = s_next, terminal = done, preference = preferences[i], step_idx=global_steps)
                replay_buffer_main.populate(this_experience)

                #check if an episode is done
                #if done reset the preference
                if done:
                    w_batch_i = w_batch_test_split[i]
                    preferences[i] = np.round(w_batch_i[np.random.randint(len(w_batch_i))], 3)
                    process_episode_array[i] += 1
                    s_next = envs.reset_subenv(i)
                s_next_all.append(s_next)

            global_steps = global_steps + 1
            s_all = np.array(s_next_all)
            if len(replay_buffer_main.buffer) < 2*args.batch_size*args.weight_num:
                continue 

            # Learn from the minibatch
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
            
            for i in range(PROCESSES_COUNT):
                batch = replay_buffer_main.sample(args.batch_size) 
                states, actions, rewards, terminals, next_states, w_batch = unpack_batch(batch)
                w_batch_interp_sample = interp(w_batch)
                critic_state, critic_loss, current_Q1_mean, current_Q2_mean, angle_term_1_mean, angle_term_2_mean, critic_norm, grads, key = learn_critic(
                    states = states,
                    actions = actions,
                    rewards = rewards,
                    terminals = terminals,
                    next_states = next_states,
                    w_batch = w_batch,
                    w_batch_interp = w_batch_interp_sample,
                    actor_state = actor_state,
                    critic_state = critic_state,
                    key = key,           
                )
                if total_learning_steps % args.policy_freq == 0:
                    actor_state, critic_state, actor_loss, angle_term, actor_norm = learn_actor(
                        states = states,
                        w_batch = w_batch,
                        w_batch_interp = w_batch_interp_sample,
                        actor_state= actor_state,
                        critic_state = critic_state,
                        key = key            
                    )
                if total_learning_steps % 10000 == 0:    
                    print(critic_loss, actor_loss, angle_term_1_mean, angle_term_2_mean)
                if (total_learning_steps % 1000) == 0:
                    writer.add_scalar('Loss/Actor_Loss'.format(), actor_loss, total_learning_steps)
                    writer.add_scalar('Loss/Critic_Loss'.format(), critic_loss, total_learning_steps)
                    writer.add_scalar('Loss/Angle_term_1'.format(), angle_term_1_mean, total_learning_steps)
                    writer.add_scalar('Loss/Angle_term_2'.format(), angle_term_2_mean, total_learning_steps)
                    writer.add_scalar('Loss/Angle_term'.format(), angle_term, total_learning_steps)
                    writer.add_scalar('Loss/critic_norm'.format(), critic_norm, total_learning_steps)
                    writer.add_scalar('Loss/actor_norm'.format(), actor_norm, total_learning_steps)
                total_learning_steps += 1
                       
            # Update Interpolator
            if (process_episode_array > (eval_cnt_ep)).all():
                eval_cnt_ep +=1
                x_tmp, key =  MORL_utils_jax.eval_agent_interp(test_env_main, w_batch_interp ,get_action, actor_state, args, key, eval_episodes = args.eval_episodes)
                # eval_agent_interp(test_env_main, agent_main, w_batch_interp, args, eval_episodes=args.eval_episodes)
                for obj_num in range(len(x)):
                    scalarized_obj_prev = np.dot(w_batch_interp[obj_num],x[obj_num])
                    scalarized_obj_current = np.dot(w_batch_interp[obj_num],x_tmp[obj_num])
                    if scalarized_obj_current > scalarized_obj_prev:
                        print(f"Previous Objective: {x[obj_num]}, Current Objective: {x_tmp[obj_num]}, Preference: {w_batch_interp[obj_num]}")
                        x[obj_num] = x_tmp[obj_num]
                x_unit = x/np.linalg.norm(x,ord=1,axis=1,keepdims=True)        
                interp = RBFInterpolator(w_batch_interp, x_unit, kernel= 'linear')
                # agent_main.interp = interp
            time_step = global_steps*PROCESSES_COUNT
            # # Evaluate agent
            # need to add this line 
            if (process_episode_array > (args.eval_freq*eval_cnt)).all():
                eval_cnt +=1
                hypervolume, sparsity, objs, key = MORL_utils_jax.eval_agent(test_env_main, w_batch_eval ,get_action, actor_state, args, key, eval_episodes = args.eval_episodes)
                file_ext='{}'.format(time_step)
                

                
                print(f"Time steps of Each Process: {process_step_array}, Episode Count of Each Process: {process_episode_array}")
                #Store episode results and write to tensorboard
                MORL_utils_jax.store_results( [], hypervolume, sparsity, time_step, writer, args)
                # lib.utilities.common_utils.save_model(actor, args, name = name, ext ='actor_{}'.format(time_step))
                # lib.utilities.common_utils.save_model(critic, args, name = name,ext ='critic_{}'.format(time_step))
                MORL_utils_jax.plot_objs(args,objs, figure_dir, ext='{}'.format(time_step))
                        
       
            
            
    finally:
        pass
        # for p in data_proc_list:
        #     p.terminate()
        #     p.join()      
    print(f"Total Number of Time Steps: {sum(process_step_array)}, Total Number of Episodes: {sum(process_episode_array)}")
    # #Evaluate the final agent
    # hypervolume, sparsity, objs = lib.utilities.MORL_utils.eval_agent(test_env_main, agent_main, w_batch_test, args, eval_episodes=args.eval_episodes)
    # lib.utilities.MORL_utils.store_results([], hypervolume, sparsity, time_step, writer, args)
    # lib.utilities.common_utils.save_model(actor, args, name = name,ext ='final_actor')
    # lib.utilities.common_utils.save_model(critic, args, name = name,ext ='final_critic')
    # lib.utilities.MORL_utils.plot_objs(args,objs,ext='final')
    print("Done in %d steps and %d episodes!" % (time_step, sum(process_episode_array)))
    print("Time Consumed")
    print("%0.2f minutes" % ((time.time() - start_time)/60))
    writer.close()