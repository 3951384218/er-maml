# Inner loop parameters
    'inner_lr': 0.01,     
    'max_path_length': 150,
    'adapt_steps': 3,
    'adapt_batch_size': 10,  
    'ppo_epochs': 3,
    'ppo_clip_ratio': 0.2,
    # Outer loop parameters
    'meta_batch_size': 20,  # 'ways'
    'outer_lr': 0.1,        
    'backtrack_factor': 0.5,
    'ls_max_steps': 15,
    'max_kl': 0.01,
    # Common parameters
    'activation': 'relu',  # for MetaWorld use tanh, others relu
    'tau': 0.95,
    'gamma': 0.99,
    'fc_neurons': 100,
    # Other parameters
    'algo_name': 'ER-MAML',
    'device': 'cpu',  
    'num_iterations': 1000,
    'save_every': 100,
    'seed': 48,
    # Evo
    'sigma': 0.001,
    'temp': 0.05,
    'n_model': 2,
    'evo_lr': 0.01,
    # Grad norm     
    'norm_a': 0.1,
    'grad_rate': 0.001,

    'adapt_steps': 3,  # Number of steps to adapt to a new task
    'adapt_batch_size': 20,  # Number of shots per task
    'n_tasks': 10,  # Number of different tasks to evaluate on