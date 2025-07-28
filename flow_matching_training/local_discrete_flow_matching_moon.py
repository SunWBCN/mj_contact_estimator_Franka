import torch
import matplotlib.pyplot as plt
from torch import nn, Tensor
from sklearn.datasets import make_moons
import wandb as wnb

class DiscreteFlowLocal(nn.Module):
    def __init__(self, num_offsets: int  = 21, dim: int = 2, h: int = 128, v: int = 128):
        super().__init__()
        self.num_offsets = num_offsets # Store num_offsets
        self.embed = nn.Embedding(v, h)
        self.net = nn.Sequential(
            nn.Linear(dim * h + 1, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, dim * num_offsets)
        )

    def forward(self, x_t: Tensor, t: Tensor) -> Tensor:
        # Reshape the output to [batch_size, dim, num_offsets]
        return self.net(torch.cat((t[:, None], self.embed(x_t).flatten(1, 2)), -1)).reshape(list(x_t.shape) + [self.num_offsets])

class RescheduleDiscreteFlowLocal(nn.Module):
    def __init__(self, num_offsets: int  = 21, dim: int = 2, h: int = 128, v: int = 128):
        super().__init__()
        self.num_offsets = num_offsets # Store num_offsets
        self.embed = nn.Embedding(v, h)
        self.net = nn.Sequential(
            nn.Linear(dim * h + 2, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, dim * num_offsets)
        )

    def forward(self, x_t: Tensor, t: Tensor) -> Tensor:
        # Reshape the output to [batch_size, dim, num_offsets]
        return self.net(torch.cat((t, self.embed(x_t).flatten(1, 2)), -1)).reshape(list(x_t.shape) + [self.num_offsets])

def generate_closest_indices_table(num_rows=128, num_closest=21):
    table = torch.zeros((num_rows, num_closest), dtype=torch.long)
    for i in range(num_rows):
        indices = torch.arange(num_rows)
        distances = torch.abs(indices - i)
        closest = torch.argsort(distances)[:num_closest]
        table[i] = closest
    return table

def find_nearest_neibor_indexes(x_1: Tensor, x_t: Tensor, table: Tensor) -> Tensor:
    x_1_flat = x_1.flatten(0, 1)
    x_t_flat = x_t.flatten(0, 1)
    row_candidates = table[x_t_flat]  # shape: (N, K)
    
    # 2. Compute distances to x_1_flat (broadcast)
    x_1_expanded = x_1_flat.unsqueeze(1)  # shape: (N, 1)
    distances = torch.abs(row_candidates - x_1_expanded)  # shape: (N, K)
    
    # 3. Get closest index per row
    indexes = torch.argmin(distances, dim=1)  # shape: (N,)
    
    # 4. Get the target values from table using advanced indexing
    targets = row_candidates[torch.arange(len(x_t_flat)), indexes]  # shape: (N,)
    return indexes.reshape(list(x_t.shape)), targets.reshape(list(x_t.shape))

if __name__ == "__main__":
    vocab_size = 128
    num_nearest_neibor = 61

    table = generate_closest_indices_table(num_rows=vocab_size, num_closest=num_nearest_neibor)
    
    # Training a model
    batch_size = 256
    vocab_size = 128
    num_offsets = num_nearest_neibor
    gap = (num_nearest_neibor - 1) // 2
    num_epochs = 10000
    lr = 0.001

    reschedule = False
    if reschedule:
        model = RescheduleDiscreteFlowLocal(num_offsets=num_offsets, v=vocab_size)
    else:
        model = DiscreteFlowLocal(num_offsets=num_offsets, v=vocab_size)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    reschedule_str = "Reschedule" if reschedule else ""
    wnb.init(project="Contact Force Estimation", name=f"{reschedule_str}LocalDiscreteFlowMatchingMoon")
    wnb.watch(model, log="all")
    wnb.config.update({
        "batch_size": batch_size,
        "vocab_size": vocab_size,
        "learning_rate": lr,
        "num_epochs": num_epochs,
        "reschedule": reschedule,
    })

    for epoch in range(num_epochs):
        x_1 = Tensor(make_moons(batch_size, noise=0.05)[0])
        x_1 = torch.round(torch.clip(x_1 * 35 + 50, min=0.0, max=vocab_size - 1)).long()

        x_0 = torch.randint(low=0, high=vocab_size, size=(batch_size, 2))

        if not reschedule:
            # Original implementation: map the global 
            t = torch.rand(batch_size) 
            x_t = torch.where(torch.rand(batch_size, 2) <  t[:, None], x_1, x_0) 
            
            # compute the difference that we wanna predict
            x_1_nearest_idx, targets_ = find_nearest_neibor_indexes(x_1, x_t, table)
        else:
            # convert the sampling time and target into the local version
            sampling_interval = 1 / (torch.abs(x_1 - x_0) // gap + 1)
            t = torch.rand(batch_size, 2) * sampling_interval
            x_1_nearest_idx, x_1_reachable = find_nearest_neibor_indexes(x_1, x_0, table)
            x_t = torch.where(torch.rand(batch_size, 2) * sampling_interval < t, x_1_reachable, x_0)

        logits = model(x_t, t)
        loss = nn.functional.cross_entropy(logits.flatten(0, 1), x_1_nearest_idx.flatten(0, 1)).mean()
        optim.zero_grad()
        loss.backward()
        optim.step()
        wnb.log({"loss": loss.item()})
        wnb.log({"learning_rate": optim.param_groups[0]['lr']})
            
    # Sampling
    x_t = torch.randint(low=0, high=vocab_size, size=(200, 2))
    t = 0.0
    results = [(x_t, t)]
    div = 90
    h_c = 0.1 / div
    while t < 1.0 - 1e-3:
        # Get predicted probabilities over offsets
        if not reschedule:
            p1 = torch.softmax(model(x_t, torch.ones(200) * t), dim=-1)
        else:
            p1 = torch.softmax(model(x_t, torch.ones(200, 2) * t), dim=-1)

        # Sample an offset index for each element and dimension
        sampled_nn_indices = torch.distributions.Categorical(probs=p1).sample()

        # Find the closest indices in the table
        x_t = table[x_t, sampled_nn_indices]

        # Clip x_t to stay within the valid vocabulary range
        x_t = torch.clip(x_t, min=0, max=vocab_size - 1).long()

        h = min(h_c, 1.0 - t)
        t += h
        results.append((x_t, t))

    # visualize the results in wandb

    fig, axes = plt.subplots(1, len(results) // div + 1, figsize=(15, 2), sharex=True, sharey=True)

    count = 0
    for (x_t, t) in results:
        if count % div == 0:
            ax = axes[count // div]
            ax.scatter(x_t.detach()[:, 0], x_t.detach()[:, 1], s=10)
            ax.set_title(f't={t:.1f}')
        count += 1
    wnb.log({"sampled_moons": wnb.Image(fig)})
    wnb.finish()

    plt.tight_layout()
    plt.show()