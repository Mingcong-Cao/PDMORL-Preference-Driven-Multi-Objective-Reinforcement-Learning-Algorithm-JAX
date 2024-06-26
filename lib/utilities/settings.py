from types import SimpleNamespace


HYPERPARAMS = {
       
    'DeepSeaTreasure_MO_DDQN_HER': SimpleNamespace(**{
        'scenario_name':         "dst-v0",
        'cuda':             False,
        'name':             'MO_DDQN_HER',
        'replay_size':      10000,
        'time_steps':       100000,
        'start_timesteps':  64,
        'w_step_size':      0.01,
        'weight_num':       3,
        'tau':              0.005,
        'epsilon_start':    0.8,
        'epsilon_final':    0.05,
        'epsilon_decay':    True,
        'lr':               3e-4,
        'gamma':            0.99,
        'batch_size':       32,
        'process_count':    10,
        'eval_freq':        200,
        'max_episode_len':  30,
        'layer_N':          3,
        'hidden_size':      256,
        'seed':             1
    }),
    
    'FruitTreeNavigation_MO_DDQN_HER': SimpleNamespace(**{
        'scenario_name':         "ftn-v0",
        'cuda':             False,
        'name':             'MO_DDQN_HER',
        'depth':            5,
        'replay_size':      10000,
        'time_steps':       100000,
        'start_timesteps':  64,
        'w_step_size':      0.05,
        'weight_num':       3,
        'tau':              0.005,
        'epsilon_start':    0.8,
        'epsilon_final':    0.05,
        'epsilon_decay':    True,
        'lr':               3e-4,
        'gamma':            0.99,
        'batch_size':       32,
        'process_count':    10,
        'eval_freq':        200,
        'max_episode_len':  7,
        'layer_N':          3,
        'hidden_size':      512,
        'seed':             1
    }),

        
    'Walker2d_MO_TD3_HER': SimpleNamespace(**{
        'scenario_name':         "MO-Walker2d-v2",
        'cuda':             False,
        'load_model':             False,
        'name':             'MO_TD3_HER',
        'replay_size':      2000000,
        'time_steps':       1000000,
        'start_timesteps':  10000,
        'w_step_size':      0.001,
        'weight_num':       3,
        'expl_noise':       0.1,
        'lr_actor':         3e-4,
        'lr_critic':         3e-4,
        'gamma':            0.995,
        'batch_size':       256,
        'process_count':    10,
        'eval_freq':        100,
        'tau':              0.005,
        'policy_noise':     0.2,
        'noise_clip':       0.5,
        'policy_freq':      10,
        'eval_episodes':     3,
        'max_episode_len':  500,
        'layer_N_critic':   1,
        'layer_N_actor':    1,
        'actor_loss_coeff':    10,
        'hidden_size':      400,
        'seed':             2
    }),

    'Walker2d_MO_TD3_HER_Key': SimpleNamespace(**{
        'scenario_name':         "MO-Walker2d-v2",
        'cuda':             False,
        'name':             'MO_TD3_HER',
        'replay_size':      500000,
        'time_steps':       2000000,
        'start_timesteps':  25000,
        'expl_noise':       0.1,
        'lr_actor':         3e-4,
        'lr_critic':         3e-4,
        'gamma':            0.99,
        'batch_size':       100,
        'eval_freq':        100,
        'tau':              0.005,
        'policy_noise':     0.2,
        'noise_clip':       0.5,
        'policy_freq':      2,
        'max_episode_len':  500,
        'layer_N_critic':   1,
        'layer_N_actor':    1,
        'hidden_size':      400,
        'seed':             2
    }),

    'HalfCheetah_MO_TD3_HER': SimpleNamespace(**{
        'scenario_name':         "MO-HalfCheetah-v2",
        'cuda':             False,
        'load_model':             False,
        'name':             'MO_TD3_HER',
        'replay_size':      2000000,
        'time_steps':       1000000,
        'start_timesteps':  10000,
        'w_step_size':      0.001,
        'weight_num':       3,
        'expl_noise':       0.1,
        'lr_actor':         3e-4,
        'lr_critic':         3e-4,
        'gamma':            0.995,
        'batch_size':       256,
        'process_count':    10,
        'eval_freq':        100,
        'tau':              0.005,
        'policy_noise':     0.2,
        'noise_clip':       0.5,
        'policy_freq':      10,
        'eval_episodes':     3,
        'max_episode_len':  500,
        'layer_N_critic':   1,
        'layer_N_actor':    1,
        'actor_loss_coeff':    10,
        'hidden_size':      400,
        'seed':             1
    }),

    'HalfCheetah_MO_TD3_HER_Key': SimpleNamespace(**{
        'scenario_name':         "MO-HalfCheetah-v2",
        'cuda':             False,
        'name':             'MO_TD3_HER',
        'replay_size':      500000,
        'time_steps':       2000000,
        'start_timesteps':  25000,
        'expl_noise':       0.1,
        'lr_actor':         3e-4,
        'lr_critic':         3e-4,
        'gamma':            0.99,
        'batch_size':       100,
        'eval_freq':        100,
        'tau':              0.005,
        'policy_noise':     0.2,
        'noise_clip':       0.5,
        'policy_freq':      2,
        'max_episode_len':  500,
        'layer_N_critic':   1,
        'layer_N_actor':    1,
        'hidden_size':      400,
        'seed':             1
    }),

    
    'Swimmer_MO_TD3_HER': SimpleNamespace(**{
        'scenario_name':         "MO-Swimmer-v2",
        'cuda':             False,
        'load_model':             False,
        'name':             'MO_TD3_HER',
        'replay_size':      2000000,
        'time_steps':       1000000,
        'start_timesteps':  10000,
        'w_step_size':      0.001,
        'weight_num':       3,
        'expl_noise':       0.1,
        'lr_actor':         3e-4,
        'lr_critic':         3e-4,
        'gamma':            0.995,
        'batch_size':       256,
        'process_count':    10,
        'eval_freq':        100,
        'tau':              0.005,
        'policy_noise':     0.2,
        'noise_clip':       0.5,
        'policy_freq':      10,
        'eval_episodes':     3,
        'max_episode_len':  500,
        'layer_N_critic':   1,
        'layer_N_actor':    1,
        'actor_loss_coeff':    10,
        'hidden_size':      400,
        'seed':             1
    }),

    'Swimmer_MO_TD3_HER_Key': SimpleNamespace(**{
        'scenario_name':         "MO-Swimmer-v2",
        'cuda':             False,
        'name':             'MO_TD3_HER',
        'replay_size':      1000000,
        'time_steps':       1000000,
        'start_timesteps':  10000,
        'expl_noise':       0.1,
        'lr_actor':         3e-4,
        'lr_critic':         3e-4,
        'gamma':            0.99,
        'batch_size':       100,
        'eval_freq':        100,
        'tau':              0.005,
        'policy_noise':     0.2,
        'noise_clip':       0.5,
        'policy_freq':      5,
        'max_episode_len':  500,
        'layer_N_critic':   1,
        'layer_N_actor':    1,
        'hidden_size':      400,
        'seed':             1
    }),


    'Ant_MO_TD3_HER': SimpleNamespace(**{
        'scenario_name':         "MO-Ant-v2",
        'cuda':             False,
        'load_model':             False,
        'name':             'MO_TD3_HER',
        'replay_size':      2000000,
        'time_steps':       1000000,
        'start_timesteps':  10000,
        'w_step_size':      0.001,
        'weight_num':       3,
        'expl_noise':       0.1,
        'lr_actor':         3e-4,
        'lr_critic':         3e-4,
        'gamma':            0.995,
        'batch_size':       256,
        'process_count':    10,
        'eval_freq':        100,
        'tau':              0.005,
        'policy_noise':     0.2,
        'noise_clip':       0.5,
        'policy_freq':      10,
        'eval_episodes':     3,
        'max_episode_len':  500,
        'layer_N_critic':   1,
        'layer_N_actor':    1,
        'actor_loss_coeff':    10,
        'hidden_size':      400,
        'seed':             1
    }),

    'Ant_MO_TD3_HER_Key': SimpleNamespace(**{
        'scenario_name':         "MO-Ant-v2",
        'cuda':             False,
        'name':             'MO_TD3_HER',
        'replay_size':      500000,
        'time_steps':       2000000,
        'start_timesteps':  25000,
        'expl_noise':       0.1,
        'lr_actor':         3e-4,
        'lr_critic':         3e-4,
        'gamma':            0.99,
        'batch_size':       100,
        'eval_freq':        100,
        'tau':              0.005,
        'policy_noise':     0.2,
        'noise_clip':       0.5,
        'policy_freq':      2,
        'max_episode_len':  500,
        'layer_N_critic':   1,
        'layer_N_actor':    1,
        'hidden_size':      400,
        'seed':             1
    }),

    'Hopper_MO_TD3_HER': SimpleNamespace(**{
        'scenario_name':         "MO-Hopper-v2",
        'cuda':             False,
        'load_model':             False,
        'name':             'MO_TD3_HER',
        'replay_size':      2000000,
        'time_steps':       1000000,
        'start_timesteps':  10000,
        'w_step_size':      0.001,
        'weight_num':       3,
        'expl_noise':       0.1,
        'lr_actor':         3e-4,
        'lr_critic':         3e-4,
        'gamma':            0.995,
        'batch_size':       256,
        'process_count':    10,
        'eval_freq':        100,
        'tau':              0.005,
        'policy_noise':     0.2,
        'noise_clip':       0.5,
        'policy_freq':      20,
        'eval_episodes':     3,
        'max_episode_len':  500,
        'layer_N_critic':   1,
        'layer_N_actor':    1,
        'actor_loss_coeff':    10,
        'hidden_size':      400,
        'seed':             1
    }),

    'Hopper_MO_TD3_HER_Key': SimpleNamespace(**{
        'scenario_name':         "MO-Hopper-v2",
        'cuda':             False,
        'name':             'MO_TD3_HER',
        'replay_size':      1000000,
        'time_steps':       2000000,
        'start_timesteps':  10000,
        'expl_noise':       0.1,
        'lr_actor':         3e-4,
        'lr_critic':         3e-4,
        'gamma':            0.99,
        'batch_size':       100,
        'eval_freq':        100,
        'tau':              0.005,
        'policy_noise':     0.2,
        'noise_clip':       0.5,
        'policy_freq':      10,
        'max_episode_len':  500,
        'layer_N_critic':   1,
        'layer_N_actor':    1,
        'hidden_size':      400,
        'seed':             1
    }),
    
}