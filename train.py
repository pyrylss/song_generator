import torch
from torch.optim.lr_scheduler import StepLR
import torch
from torch.optim.lr_scheduler import StepLR
from config import *
from data import *
from model import SongGenerator
from utils import get_batch, estimate_loss

m = SongGenerator(vocab_size, n_features, block_size, n_layers, n_heads, dropout, device).to(device)

optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
#scheduler = StepLR(optimizer, step_size=2000, gamma=0.5)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5, verbose=True)

best_val_loss = float('inf')  # start with a high value
no_improve_counter = 0
patience = 3

for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters-1:
        losses = estimate_loss(m)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        #scheduler.step(losses['val'])

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
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    #scheduler.step()

# generate from the model
context = torch.tensor([[stoi['\n']]], dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=1000)[0].tolist()))

# save the model
torch.save(m.state_dict(), 'songGen.pth')
