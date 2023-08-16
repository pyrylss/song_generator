import torch

# Hyperparameters and Configurations
n_features = 348
n_heads = 6
n_layers = 6
dropout = 0.2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
block_size = 25 
batch_size = 6
learning_rate = 3e-4
eval_interval = 500
max_iters = 20000
eval_iters = 500
temperature = 1.0