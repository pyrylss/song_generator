import torch
from songgen import SongGenerator, stoi, decode  # Assuming songgen.py contains these

# Initialize device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize model
model = SongGenerator()
model = model.to(device)

# Load the pre-trained model
model_path = 'songGen.pth'
model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))

# Set model to evaluation mode
model.eval()

# Generate text
context = torch.tensor([[stoi['\n']]], dtype=torch.long, device=device)
generated_text = decode(model.generate(context, max_new_tokens=500)[0].tolist())

# Print generated text
print(generated_text)