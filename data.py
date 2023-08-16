import os
import torch

dir_path = 'lyrics/'

text = ''
for filename in os.listdir(dir_path):
    if filename.endswith('.txt'):
        with open(os.path.join(dir_path, filename), 'r', encoding='utf-8') as f:
            text += f.read().lower()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {w:i for i,w in enumerate(chars)}
itos = {i:w for i,w in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join(itos[i] for i in l)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]