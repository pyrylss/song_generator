import torch
import torch.nn as nn
from torch.nn import functional as F
import string
import re
import os
from zipfile import ZipFile
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR

device = 'cuda' if torch.cuda.is_available() else 'cpu'

file_name = 'lyrics.zip'
dir_path = ''

with ZipFile(file_name, 'r') as zip:
    zip.extractall(path=dir_path)

text = ''

# os.listdir(dir_path) returns a list of filenames in the directory
for filename in os.listdir('./lyrics'):
    # Check if the file is a .txt file
    if filename.endswith('.txt'):
        with open(os.path.join('./lyrics', filename), 'r', encoding='utf-8') as f:
            text += f.read().lower()

# hyperparameters
n_features = 348
n_heads = 6
n_layers = 6
dropout = 0.2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
block_size = 256 # what is the maximum context length for predictions?
batch_size = 64 # how many independent sequences will we process in parallel?
learning_rate = 3e-4
eval_interval = 500
max_iters = 18000
eval_iters = 200
temperature = 1.0

chars = sorted(list(set(text)))
vocab_size = len(chars)
vocab_size


# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

data = torch.tensor(encode(text), dtype=torch.long)

n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

torch.manual_seed(1337)
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split=='train' else val_data
    ix = torch.randint(len(data)-block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

xb, yb = get_batch('train')

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

def subsequent_mask(sz):
    mask = (torch.tril(torch.ones(sz, sz)))
    mask = mask.masked_fill(mask[:sz, :sz] == 0, float('-inf'))
    return mask

mask = subsequent_mask(10)

class Block(nn.Module):

    def __init__(self, n_features, n_heads):
        super().__init__()
        self.mha = nn.MultiheadAttention(n_features, n_heads, dropout=dropout)
        self.ln1 = nn.LayerNorm(n_features)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)


        self.ff = nn.Sequential(
            nn.Linear(n_features, 4 * n_features),
            nn.ReLU(),
            nn.Linear(4 * n_features, n_features),
            nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(n_features)

        #self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))


    def forward(self, x, mask=None):

        if mask is not None:
            mask = mask.to(device)

        attn_in = x.permute(1, 0, 2) # (B, T, C) --> (T, B, C)
        attn_output, _ = self.mha(self.ln1(attn_in), self.ln1(attn_in), self.ln1(attn_in), attn_mask=mask)  # attn_output has the same shape as attn_in
        attn_output = attn_output.permute(1, 0, 2) # (T, B, C) --> (B, T, C)

        x = self.drop1(x) + attn_output
        x = self.drop2(x) + self.ff(self.ln2(x))
        return x

torch.manual_seed(1337)

class SongGenerator(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.embedding = nn.Embedding(vocab_size, n_features)

        self.pos_encoding = nn.Embedding(block_size, n_features)
        self.blocks = nn.Sequential(*[Block(n_features, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_features) # final layer norm
        self.lm_head = nn.Linear(n_features, vocab_size)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, idx, targets=None):

        B, T = idx.shape
        mask = subsequent_mask(T).to(device)

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.dropout1(self.embedding(idx)) # (B,T,C)
        pos_emb = self.pos_encoding(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb # (B,T,C)
        #x = self.blocks(x, mask=mask) # (B, T, C)
        for block in self.blocks:
            x = block(x, mask=mask)

        x = self.dropout2(self.ln_f(x)) # (B, T, C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=temperature):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            #logits = logits/temperature
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)

            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next.T), dim=1) # (B, T+1)
        return idx

model = SongGenerator()
model = model.to(device)

def main():
    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(context, max_new_tokens=100)[0].tolist()))

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5, verbose=True)
    #scheduler =

    best_val_loss = float('inf')  # start with a high value
    no_improve_counter = 0
    patience = 3

    for iter in range(max_iters):
        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            #scheduler.step(losses['val'])
            context = torch.tensor([[stoi['\n']]], dtype=torch.long, device=device)
            print(decode(model.generate(context, max_new_tokens=100)[0].tolist()))

            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                no_improve_counter = 0
            else:
                no_improve_counter += 1
                if no_improve_counter > patience:
                    print("Early stopping!")
                    break

        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # after 25 000 epochs, train loss: 1.25, val loss: 1.29

    # generate from the model
    context = torch.tensor([[stoi['\n']]], dtype=torch.long, device=device)
    print(decode(model.generate(context, max_new_tokens=1000)[0].tolist()))

    # save the model before finetuning
    torch.save(model.state_dict(), 'songGen.pth')

    output = decode(model.generate(context, max_new_tokens=2000)[0].tolist())
    with open('model_output.txt', 'w') as f:
        f.write(output)

    file_name = 'eminem.txt'

    with open('./eminem.txt', 'r') as f:
        text_e = f.read().lower()

    data_e = torch.tensor(encode(text_e), dtype=torch.long)
    print(data_e.shape, data_e.dtype)

    n = int(0.9*len(data_e))
    train_data = data_e[:n]
    val_data = data_e[n:]
    print(len(train_data), len(val_data))

    new_learning_rate = 1e-5
    optimizer = torch.optim.AdamW(model.parameters(), lr=new_learning_rate)

    for iter in range(1000):
        # every once in a while evaluate the loss on train and val sets
        if iter % 99 == 0:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            #scheduler.step(losses['val'])
            context = torch.tensor([[stoi['\n']]], dtype=torch.long, device=device)
            print(decode(model.generate(context, max_new_tokens=100)[0].tolist()))

        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # after 1000 iters
    context = torch.tensor([[stoi['\n']]], dtype=torch.long, device=device)
    print(decode(model.generate(context, max_new_tokens=1000)[0].tolist()))

    # after 2300 iters
    context = torch.tensor([[stoi['\n']]], dtype=torch.long, device=device)
    print(decode(model.generate(context, max_new_tokens=1000)[0].tolist()))

    output_e = decode(model.generate(context, max_new_tokens=2500)[0].tolist())
    with open('model_output_eminem.txt', 'w') as f:
        f.write(output_e)

if __name__ == "__main__":
    main()
