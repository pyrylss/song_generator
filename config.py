import torch

# Hyperparameters and Configurations
n_features = 348
n_heads = 6
n_layers = 6
dropout = 0.2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
block_size = 256 
batch_size = 64
learning_rate = 3e-4
eval_interval = 500
max_iters = 20000
eval_iters = 200
temperature = 1.0