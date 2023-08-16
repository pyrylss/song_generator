import torch
import torch.nn as nn
from torch.nn import functional as F
from config import *
from dataclasses import dataclass

def subsequent_mask(sz):
    mask = (torch.tril(torch.ones(sz, sz)))
    mask = mask.masked_fill(mask[:sz, :sz] == 0, float('-inf'))
    return mask

class Block(nn.Module):
    
    def __init__(self, n_features, n_heads, dropout, device):
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
        
    def forward(self, x, mask=None):

        if mask is not None:
            mask = mask.to(device)
        attn_in = x.permute(1, 0, 2) # (B, T, C) --> (T, B, C)
        attn_output, _ = self.mha(self.ln1(attn_in), self.ln1(attn_in), self.ln1(attn_in), attn_mask=mask)  # attn_output has the same shape as x
        attn_output = attn_output.permute(1, 0, 2) # (T, B, C) --> (B, T, C)

        x = self.drop1(x) + attn_output
        x = self.drop2(x) + self.ff(self.ln2(x))
        return x

class SongGenerator(nn.Module):

    def __init__(self, vocab_size, n_features, block_size, n_layers, n_heads, dropout, device):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.embedding = nn.Embedding(vocab_size, n_features)
        
        self.pos_encoding = nn.Embedding(block_size, n_features)
        self.blocks = nn.Sequential(*[Block(n_features, n_heads, dropout, device) for _ in range(n_layers)])
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
        #x = self.blocks(x) # (B, T, C)
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