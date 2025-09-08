import torch
import matplotlib.pyplot as plt
from torch import nn, Tensor
import wandb as wnb
import os
from data_loader import DataLoader
from tqdm import trange
from pathlib import Path
import numpy as np

class DiscreteFlowHistory(nn.Module):
    def __init__(self, seq_len: int = 7, his_len: int = 5, h: int = 32, v: int = 128, aug_state_dim: int = 0, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.seq_len = seq_len
        self.his_len = his_len
        self.h = h
        self.v = v
        self.aug_state_dim = aug_state_dim
        self.n_heads = n_heads
        self.n_layers = n_layers

        # Token embedding
        self.embed = nn.Embedding(v, h)
        
        # Contact number embedding
        self.hist_embed = nn.Embedding(v, h)
        self.hist_score = nn.Linear(h, 1)         # scores per history step
        # self.attn_drop  = nn.Dropout(p=0.1)       # optional
        
        # Positional encoding (so attention knows order)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, h))

        # Transformer encoder (multi-head self-attention)
        encoder_layer = nn.TransformerEncoderLayer(d_model=h, nhead=n_heads, dim_feedforward=4*h, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Projection for time + augmented state
        self.time_proj = nn.Linear(1, h)
        self.aug_proj = nn.Linear(aug_state_dim, h)

        # Output layer: predict logits per token
        self.fc_out = nn.Linear(h, v)

    def forward(self, x_t_c: Tensor, t: Tensor, aug_state: Tensor, contact_nums_history: Tensor) -> Tensor:
        # x_t_c: [batch, seq_len]
        B, L = x_t_c.shape
        H = self.his_len

        # Embed tokens + add positional encoding
        token_emb = self.embed(x_t_c)  # [B, L, h]
        token_emb = token_emb + self.pos_embed[:, :L, :]

        # Project time and aug_state to same size and add as conditioning
        t_emb = self.time_proj(t[:, None]).unsqueeze(1)   # [B, 1, h]
        aug_emb = self.aug_proj(aug_state).unsqueeze(1)   # [B, 1, h]
        
        # History embedding
        if contact_nums_history.dim() == 2:
            hist_ids = contact_nums_history.view(B, H, L)     # [B, H, L]
        else:
            hist_ids = contact_nums_history                   # [B, H, L]
            assert hist_ids.shape[1:] == (H, L)
        
        hist_vecs = self.hist_embed(hist_ids)                 # [B, H, L, h]
        # hist_emb  = hist_vecs.mean(dim=1)    
        scores = self.hist_score(hist_vecs).squeeze(-1)
        weights = torch.softmax(scores, dim=1)                   # softmax over H
        # weights = self.attn_drop(weights)
        # weighted sum over H → [B, L, h]
        hist_emb = (weights.unsqueeze(-1) * hist_vecs).sum(dim=1)
                
        # Broadcast conditioning to sequence length
        x = token_emb + hist_emb + t_emb + aug_emb

        # Run through transformer encoder
        enc = self.encoder(x)  # [B, L, h]

        # Predict logits for each position
        logits = self.fc_out(enc)  # [B, L, v]

        return logits

if __name__ == "__main__":    
    # Load the dataset
    file_name = "dataset_batch_1_1000eps"
    dir_name = "data-link7-2-contact_v3"
    d_loader = DataLoader(file_name, dir_name)
    
    # Get the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize the model
    seq_len = 7
    his_len = 100
    max_contact_per_pos = 3
    vocab_size = max_contact_per_pos + 1
    
    # Num epochs
    num_epochs = 10000
    batch_size = 512
    
    # Get sample data to determine dimensions
    _, aug_state_sample, _, _, \
    _, aug_state_sample_history, _, _, \
    = d_loader.sample_contact_ids_robot_state_history(10, history_len=his_len)
    aug_state_sample = np.array(aug_state_sample)
    aug_state_dim = aug_state_sample.shape[1] if aug_state_sample.ndim > 1 else 1
    aug_state_sample_history = np.array(aug_state_sample_history)
    aug_state_history_dim = aug_state_sample_history.shape[1] if aug_state_sample_history.ndim > 1 else 1
    aug_state_dim = aug_state_dim + aug_state_history_dim
    print(f"Augmented state dimension: {aug_state_dim}")
    
    model = DiscreteFlowHistory(
        seq_len=seq_len,
        his_len=his_len,
        v=vocab_size,
        aug_state_dim=aug_state_dim
    ).to(device)

    # Initialize the optimizer
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in trange(num_epochs, desc="Training High Level Discrete Flow"):
        # Sample data from loader
        _, aug_state, c_pos, contact_nums, \
        _, aug_state_history, c_pos_history, contact_nums_history \
        = d_loader.sample_contact_ids_robot_state_history(batch_size, history_len=his_len, data_slice="train")
        aug_state_aug_history = np.concatenate([aug_state, aug_state_history], axis=-1)
        # Convert all data to tensors on the correct device
        contact_nums = torch.tensor(contact_nums, device=device, dtype=torch.long)
        # aug_state = torch.tensor(aug_state, device=device, dtype=torch.float32)
        contact_nums_history = torch.tensor(contact_nums_history, device=device, dtype=torch.long)
        aug_state_aug_history = torch.tensor(aug_state_aug_history, device=device, dtype=torch.float32)
        x_1_c = torch.clamp(contact_nums, max=max_contact_per_pos)
        x_0_c = torch.randint(low=0, high=max_contact_per_pos, size=(batch_size, seq_len), device=device)

        # Random interpolation
        t = torch.rand(batch_size, device=device)
        x_t_c = torch.where(torch.rand((batch_size, seq_len), device=device) < t[:, None], x_1_c, x_0_c)

        # Forward pass
        logits = model(x_t_c=x_t_c, t=t, aug_state=aug_state_aug_history, contact_nums_history=contact_nums_history)

        # Compute loss
        loss = nn.functional.cross_entropy(
            logits.view(-1, vocab_size),
            x_1_c.view(-1)
        )
        
        # Backward pass
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    print("Training completed!")
    
    # Sampling
    num_samples = 1000
    x_t_c = torch.zeros((num_samples, seq_len), dtype=torch.long, device=device)

    # Get new data for sampling
    _, aug_state, c_pos, contact_nums, \
    _, aug_state_history, c_pos_history, contact_nums_history \
    = d_loader.sample_contact_ids_robot_state_history(num_samples, history_len=his_len, data_slice="validate")
    aug_state_aug_history = np.concatenate([aug_state, aug_state_history], axis=-1)
    aug_state_aug_history = torch.tensor(aug_state_aug_history, device=device, dtype=torch.float32)
    contact_nums_history = torch.tensor(contact_nums_history, device=device, dtype=torch.long)
    
    # TODO: change contact numbers to the correct shape
    contact_nums = torch.tensor(contact_nums, device=device, dtype=torch.long)
    x_1_c = torch.clamp(contact_nums, max=max_contact_per_pos)

    t = 0.0  # Initialize t properly
    h_c = 0.1
    results = []
    
    with torch.no_grad():  # No gradients needed for sampling
        while t < 1.0 - 1e-3:
            # Forward pass
            t_tensor = torch.full((num_samples,), t, device=device)
            logits = model(x_t_c=x_t_c, t=t_tensor, aug_state=aug_state_aug_history, contact_nums_history=contact_nums_history)
            p1 = torch.softmax(logits, dim=-1)
            
            # Sample next edit operation
            h = min(h_c, 1.0 - t)
            one_hot_x_t = nn.functional.one_hot(x_t_c, vocab_size).float()
            u = (p1 - one_hot_x_t) / (1.0 - t)
            x_t_c = torch.distributions.Categorical(probs=one_hot_x_t + h * u).sample()            
            
            # Update time
            t += h
            
            results.append((x_t_c.cpu().numpy(), t))
        
    print("Sampling completed!")
    
    # Analyze results
    final_contacts = results[-1][0]

    # Compare the final results with x_1_c
    num_correct_total = final_contacts == x_1_c.cpu().numpy()
    num_correct_total = np.all(num_correct_total, axis=1)  # Check if all positions match
    rate = num_correct_total.sum() / num_samples
    
    # Initial results with x_1_c
    initial_contacts = results[0][0]
    num_correct_total = initial_contacts == x_1_c.cpu().numpy()
    num_correct_total = np.all(num_correct_total, axis=1)  # Check if all positions match
    rate_init = num_correct_total.sum() / num_samples
    print(f"Comparison with x_1_c: init, {rate_init}, final, {rate}")