import torch
import torch.nn as nn
from torch.nn import functional as F
import string
import re
import os

# hyperparameters
n_features = 64
n_heads = 16
n_layers = 6
dropout=0.2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
block_size = 128
batch_size = 32
learning_rate = 1e-3
eval_interval = 100
max_iters = 101
eval_iters = 50

dir_path = 'lyrics/'
    
text = ''

# os.listdir(dir_path) returns a list of filenames in the directory
for filename in os.listdir(dir_path):
    # Check if the file is a .txt file
    if filename.endswith('.txt'):
        with open(os.path.join(dir_path, filename), 'r', encoding='utf-8') as f:
            text += f.read().lower() + '\n'

punctuation = string.punctuation.replace("'", "")
translator = str.maketrans('', '', punctuation)
text = re.sub(r'\(.*?\)', '', text)
text = text.translate(translator)
text = re.sub(r'\[.*?\]', '', text)
words = list(set(re.split(r' |\n', text)))
vocab_size = len(words)

stoi = {w:i for i,w in enumerate(words)} # mapping from words to integers
itos = {i:w for i,w in enumerate(words)} # mapping from integers to words
encode = lambda s: [stoi[c] for c in re.split(r' |\n', s)] # take a string, output a list of integers
decode = lambda l: ' '.join(itos[i] for i in l).replace(' \n ', '\n') # take a list of integers, output a string

data = torch.tensor(encode(text), dtype=torch.long)

n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split=='train' else val_data
    ix = torch.randint(len(data)-block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Block(nn.Module):
    
    def __init__(self, n_features, n_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(n_features, n_heads)
        self.ln1 = nn.LayerNorm(n_features)
        
        self.ff = nn.Sequential(
            nn.Linear(n_features, 4 * n_features),
            nn.ReLU(),
            nn.Linear(4 * n_features, n_features),
        )
        self.ln2 = nn.LayerNorm(n_features)
        
    def forward(self, x):
        x = x.permute(1, 0, 2)  # permute to match (seq_len, batch_size, embedding_dim)
        attn_output, _ = self.mha(self.ln1(x), self.ln1(x), self.ln1(x))  # attn_output has the same shape as x
        x = x + attn_output
        x = x.permute(1, 0, 2)  # permute back to (batch_size, seq_len, embedding_dim)
        x = x + self.ff(self.ln2(x))
        return x  

class SongGenerator(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.embedding = nn.Embedding(vocab_size, n_features)
        
        self.pos_encoding = nn.Embedding(block_size, n_features)
        self.blocks = nn.Sequential(*[Block(n_features, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_features) # final layer norm
        self.lm_head = nn.Linear(n_features, vocab_size)

        
    def forward(self, idx, targets=None):
        
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.embedding(idx) # (B,T,C)
        pos_emb = self.pos_encoding(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B, T, C)
        x = self.ln_f(x) # (B, T, C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


model = SongGenerator()
m = model.to(device)

optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

with open('eminem.txt', 'r', encoding='utf-8') as f:
    text_e = f.read().lower()

text_e = re.sub(r'\(.*?\)', '', text)
text_e = text.translate(translator)
words_e = list(set(text_e.split()))
vocab_size_e = len(words_e)



# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=100)[0].tolist()))