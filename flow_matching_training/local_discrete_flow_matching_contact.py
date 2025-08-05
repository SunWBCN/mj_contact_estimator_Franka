import torch
import matplotlib.pyplot as plt
from torch import nn, Tensor
from sklearn.datasets import make_moons
import wandb as wnb
import yaml
import numpy as np
import os
from data_loader import DataLoader
from discrete_flow_matching_moon import chi_square_test

class DiscreteFlowLocal(nn.Module):
    def __init__(self, num_offsets: int  = 21, dim: int = 1, h: int = 128, v: int = 128):
        super().__init__()
        self.num_offsets = num_offsets # Store num_offsets
        self.embed = nn.Embedding(v, h)
        self.net = nn.Sequential(
            nn.Linear(dim * h + 1, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, dim * num_offsets)
        )

    def forward(self, x_t: Tensor, t: Tensor, aug_state: Tensor = None) -> Tensor:
        # Reshape the output to [batch_size, dim, num_offsets]
        if aug_state is not None:
            input_tensor = torch.cat((t[:, None], self.embed(x_t), aug_state), -1)
        else:
            input_tensor = torch.cat((t[:, None], self.embed(x_t)), -1)
        
        return self.net(input_tensor).reshape(list(x_t.shape) + [self.num_offsets])

class RescheduleDiscreteFlowLocal(nn.Module):
    def __init__(self, num_offsets: int  = 21, dim: int = 1, h: int = 128, v: int = 128):
        super().__init__()
        self.num_offsets = num_offsets # Store num_offsets
        self.embed = nn.Embedding(v, h)
        self.net = nn.Sequential(
            nn.Linear(dim * h + 1, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, dim * num_offsets)
        )

    def forward(self, x_t: Tensor, t: Tensor, aug_state: Tensor = None) -> Tensor:
        # Reshape the output to [batch_size, dim, num_offsets]
        if aug_state is not None:
            input_tensor = torch.cat((t[:, None], self.embed(x_t), aug_state), -1)
        else:
            input_tensor = torch.cat((t[:, None], self.embed(x_t)), -1)
        return self.net(input_tensor).reshape(list(x_t.shape) + [self.num_offsets])
    
def generate_closest_indices_table(num_rows=128, num_closest=21):
    table = torch.zeros((num_rows, num_closest), dtype=torch.long)
    for i in range(num_rows):
        indices = torch.arange(num_rows)
        distances = torch.abs(indices - i)
        closest = torch.argsort(distances)[:num_closest]
        table[i] = closest
    return table

def find_nearest_neibor_indexes(x_1: Tensor, x_t: Tensor, table: Tensor) -> Tensor:
    x_1_flat = x_1
    x_t_flat = x_t
    row_candidates = table[x_t_flat]  # shape: (N, K)
    
    # 2. Compute distances to x_1_flat (broadcast)
    x_1_expanded = x_1_flat.unsqueeze(1)  # shape: (N, 1)
    distances = torch.abs(row_candidates - x_1_expanded)  # shape: (N, K)
    
    # 3. Get closest index per row
    indexes = torch.argmin(distances, dim=1)  # shape: (N,)
    
    # 4. Get the target values from table using advanced indexing
    targets = row_candidates[torch.arange(len(x_t_flat)), indexes]  # shape: (N,)
    return indexes.reshape(list(x_t.shape)), targets.reshape(list(x_t.shape))

def get_nearest_neighbors(x_t: Tensor, table: Tensor) -> Tensor:
    x_t_flat = x_t
    row_candidates = table[x_t_flat]  # shape: (N, K)
    return row_candidates

if __name__ == "__main__":    
    # load hyperparameters from yaml file
    with open("config_contact.yaml", "r") as f:
        config = yaml.safe_load(f)
    vocab_size = config.get("vocab_size", 128)
    num_nearest_neibor = config.get("num_nearest_neibor", 41)
    num_offsets = num_nearest_neibor
    batch_size = config.get("batch_size")
    num_epochs = config.get("num_epochs")
    lr = config.get("learning_rate")
    reschedule = config.get("reschedule")
    enforce = config.get("enforce")
    temp_anneling = config.get("temp_anneling")
    temp = config.get("temperature")
    div = config.get("div", 10.0)
    sample_step = config.get("sample_step")
    gap = (num_nearest_neibor - 1) // 2
    use_wnb = config.get("use_wnb")
    max_contact_id = 1000  # Maximum contact ID for scaling    
    table = generate_closest_indices_table(num_rows=vocab_size, num_closest=num_nearest_neibor)
    condition = True
    condition_str = "cond" if condition else ""
    reschedule_str = "R" if reschedule else ""
    d_loader = DataLoader("dataset_1000eps_1_contact")
    
    if reschedule:
        model = RescheduleDiscreteFlowLocal(num_offsets=num_offsets, v=vocab_size)
    else:
        model = DiscreteFlowLocal(num_offsets=num_offsets, v=vocab_size)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    file_name = f"local_discrete_flow_contact{num_epochs}_Vacob{vocab_size}_NN{num_nearest_neibor}_{condition_str}_{reschedule}.pth"
    file_path = f"models/{file_name}"
    if os.path.exists(file_path):
        print(f"Loading model from {file_path}")
        model.load_state_dict(torch.load(file_path))
    else:
        print(f"Model file {file_path} does not exist. Starting training from scratch.")

    reschedule_str = "Reschedule" if reschedule else ""
    enforce_str = "Enforce" if enforce else ""
    if use_wnb:
        wnb.init(project="Contact Force Estimation", name=f"{enforce_str}{reschedule_str}LocalDiscreteFlowMatchingMoon")
        wnb.watch(model, log="all")
        wnb.config.update(config)

    for epoch in range(num_epochs):
        if not condition:
            x_1 = d_loader.sample_contact_ids(batch_size, noise=0.05, max_contact_id=max_contact_id)
            x_0 = torch.randint(low=0, high=vocab_size, size=(batch_size, ))
        else:
            x_0, x_1 = d_loader.sample_contact_condition_ids(batch_size, noise=0.05, max_contact_id=max_contact_id)

        if not reschedule:
            # Original implementation: map the global 
            t = torch.rand(batch_size) 
            x_t = torch.where(torch.rand(batch_size) <  t, x_1, x_0) 
            
            # compute the difference that we wanna predict
            x_1_nearest_idx, targets_ = find_nearest_neibor_indexes(x_1, x_t, table)
        else:            
            # adapt the sampling time
            stepsize = torch.abs(x_1 - x_0) // gap + 1
            max_stepsize = torch.max(stepsize)
            lower_bd = 1 - 1 / max_stepsize * stepsize
            upper_bd = 1 - 1 / max_stepsize * (stepsize - 1)
            t = torch.rand(batch_size) * (upper_bd - lower_bd) + lower_bd
            t_flat = t
            
            # generate a boltzmann distribution, using the distance between the nearest neighbors to the target
            # then sample from the distribution to get x_t
            x_0_nearest_neighbors = get_nearest_neighbors(x_0, table) # shape: (batch_size*2, num_nearest_neibor)
            
            # compute the distances to the target
            distances = torch.abs(x_1.unsqueeze(1) - x_0_nearest_neighbors)  # shape: (batch_size*2, num_nearest_neibor)
    
            # retrieve the positions where t is 1
            t_is_one_positions = torch.where(t_flat == 1.0)
            
            # Avoid division by zero and NaN values
            if torch.any(t_flat == 1.0):
                t_flat = torch.where(t_flat == 1.0, torch.ones_like(t_flat) * 0.99, t_flat)
            
            # compute the Boltzmann distribution
            expont = - distances
            divider = temp * (1 - t_flat) / t_flat  # Avoid division by zero   
            divider = divider.unsqueeze(1)  # Change from (batch_size,) to (batch_size, 1)
            expont = expont / divider            
            expont_max = torch.max(expont, dim=1, keepdim=True)[0]
            expont = expont - expont_max
            
            boltzmann_dist = torch.exp(expont) / torch.sum(torch.exp(expont), dim=1, keepdim=True)
            # for i in range(boltzmann_dist.shape[0]):
            #     print(torch.sum(boltzmann_dist[i]))
            sampled_nn_indices = torch.distributions.Categorical(probs=boltzmann_dist).sample()
            x_t = x_0_nearest_neighbors[torch.arange(batch_size), sampled_nn_indices]
            # fill the positions where t is 1 with the target
            if enforce:
                x_t[t_is_one_positions] = x_1[t_is_one_positions]
            x_1_nearest_idx, targets_ = find_nearest_neibor_indexes(x_1, x_t, table)

        logits = model(x_t, t)
        loss = nn.functional.cross_entropy(logits, x_1_nearest_idx).mean()
        optim.zero_grad()
        loss.backward()
        optim.step()
        if use_wnb:
            wnb.log({"loss": loss.item()})
            wnb.log({"learning_rate": optim.param_groups[0]['lr']})
            
    # Check the existence of the model file, if not, save the model
    if not os.path.exists(file_path):
        print(f"Saving model to {file_path}")
        torch.save(model.state_dict(), file_path)
    else:
        print(f"Model file {file_path} already exists. Skipping save.")
            
    # Sampling
    num_samples = 500
    if not condition:
        x_t = torch.randint(low=0, high=vocab_size, size=(num_samples, ))
    else:
        x_t, _ = d_loader.sample_contact_condition_ids(num_samples, noise=0.0, max_contact_id=max_contact_id)
    t = 0.0
    results = [(x_t, t)]
    results_x_t = []
    h_c = sample_step / div
    while t < 1.0 - 1e-3:
        # Get predicted probabilities over offsets
        p1 = torch.softmax(model(x_t, torch.ones(num_samples) * t), dim=-1)

        # Sample an offset index for each element and dimension
        sampled_nn_indices = torch.distributions.Categorical(probs=p1).sample()

        # Find the closest indices in the table
        x_t = table[x_t, sampled_nn_indices]

        # Clip x_t to stay within the valid vocabulary range
        x_t = torch.clip(x_t, min=0, max=vocab_size - 1).long()

        h = min(h_c, 1.0 - t)
        t += h
        results.append((x_t, t))
        results_x_t.append(x_t)

    # visualize the results in wandb
    results = torch.stack(results_x_t)
    results = results.flatten()
    if not condition:
        dset_x1 = d_loader.sample_contact_ids(len(results), noise=0.0, max_contact_id=max_contact_id)
    else:
        _, dset_x1 = d_loader.sample_contact_condition_ids(len(results), noise=0.0, max_contact_id=max_contact_id)
    chi2, p = chi_square_test(dset_x1.numpy(), results.numpy())
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(results.numpy(), bins=50, label='Sampled Contact IDs')
    plt.hist(dset_x1.numpy(), bins=50, label='Original Contact IDs')
    plt.title('Distribution of Contact IDs')
    plt.xlabel('Contact ID')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.show()