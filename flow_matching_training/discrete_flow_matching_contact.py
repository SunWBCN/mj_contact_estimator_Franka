# %pip install torch
import torch
import matplotlib.pyplot as plt
from torch import nn, Tensor
import wandb as wnb
import numpy as np
import os
from discrete_flow_matching_moon import chi_square_test
from data_loader import DataLoader

class DiscreteFlow(nn.Module):
    def __init__(self, dim: int = 1, h: int = 128, v: int = 128):
        super().__init__()
        self.v = v
        self.embed = nn.Embedding(v, h)
        self.net = nn.Sequential(
            nn.Linear(dim * h + 1, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, dim * v))

    def forward(self, x_t: Tensor, t: Tensor, aug_state: Tensor = None) -> Tensor:
        if aug_state is not None:
            input_tensor = torch.cat((t[:, None], self.embed(x_t), aug_state), -1)
        else:
            input_tensor = torch.cat((t[:, None], self.embed(x_t)), -1)
        return self.net(input_tensor).reshape(list(x_t.shape) + [self.v])
    
if __name__ == "__main__":
    # Training
    batch_size = 512
    vocab_size = 4096 #128
    lr = 0.001
    num_epochs = 1000
    max_contact_id = 4000  # Maximum contact ID for scaling
    if vocab_size < max_contact_id:
        print(f"Warning: vocab_size {vocab_size} is less than max_contact_id {max_contact_id}. This may lead to unexpected behavior.")
        exit(1)
    model = DiscreteFlow(v=vocab_size, dim=1)
    use_wnb = False
    condition_previous_contact = False  # Whether to condition on the previous contact ID
    d_loader = DataLoader("dataset_1000eps_1_contact")
    
    # load model if exists
    conditional_previous_contact_str = "cond_prev_contact" if condition_previous_contact else ""
    file_name = f"discrete_flow_contact{num_epochs}{conditional_previous_contact_str}Vocab{vocab_size}.pth"
    file_path = f"models/{file_name}"
    if os.path.exists(file_path):
        print(f"Loading model from {file_path}")
        model.load_state_dict(torch.load(file_path))
    else:
        print(f"Model file {file_path} does not exist. Starting training from scratch.")
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    if use_wnb:
        wnb.init(project="Contact Force Estimation", name=f"DiscreteFlowMatchingContact{conditional_previous_contact_str}Vocab{vocab_size}")
        wnb.watch(model, log="all")
        
        # log hyperparameters
        wnb.config.update({
            "batch_size": batch_size,
            "vocab_size": vocab_size,
            "learning_rate": lr,
            "num_epochs": num_epochs,
        })
    if not os.path.exists(file_name):
        for epoch in range(num_epochs):
            if not condition_previous_contact:
                x_1 = d_loader.sample_contact_ids(batch_size, noise=0.05, max_contact_id=max_contact_id)
                x_0 = torch.randint(low=0, high=vocab_size, size=(batch_size, ))
            else:
                x_0, x_1 = d_loader.sample_contact_condition_ids(batch_size, noise=0.05, max_contact_id=max_contact_id)

            t = torch.rand(batch_size)
            x_t = torch.where(torch.rand(batch_size) < t, x_1, x_0)

            logits = model(x_t, t)
            loss = nn.functional.cross_entropy(logits, x_1).mean()
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
    if not condition_previous_contact:
        x_t = torch.randint(low=0, high=vocab_size, size=(num_samples, ))
    else:
        x_t, _ = d_loader.sample_contact_condition_ids(num_samples, noise=0.0, max_contact_id=max_contact_id)
    t = 0.0
    div = 1
    h_c = 0.1 / div
    results = [(x_t, t)]
    results_x_t = []
    while t < 1.0 - 1e-3:
        p1 = torch.softmax(model(x_t, torch.ones(num_samples) * t), dim=-1)
        h = min(h_c, 1.0 - t)
        one_hot_x_t = nn.functional.one_hot(x_t, vocab_size).float()
        u = (p1 - one_hot_x_t) / (1.0 - t)
        x_t = torch.distributions.Categorical(probs=one_hot_x_t + h * u).sample()
        t += h
        results.append((x_t, t))
        results_x_t.append(x_t)

    # Compare the bar plot of the sampled contact IDs and the original contact IDs
    results_x_t = torch.stack(results_x_t)
    results_x_t = results_x_t.flatten()
    if not condition_previous_contact:
        dset_x1 = d_loader.sample_contact_ids(len(results_x_t), noise=0.0, max_contact_id=max_contact_id)
    else:
        x_0, dset_x1 = d_loader.sample_contact_condition_ids(len(results_x_t), noise=0.0, max_contact_id=max_contact_id)
    
    # table = [results_x_t.numpy(), dset_x1.numpy()]
    chi2, p = chi_square_test(dset_x1.numpy(), results_x_t.numpy())
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(results_x_t.numpy(), bins=50, label='Sampled Contact IDs')
    plt.hist(dset_x1.numpy(), bins=50, label='Original Contact IDs')
    plt.title('Distribution of Contact IDs')
    plt.xlabel('Contact ID')
    plt.ylabel('Frequency')
    plt.legend()    
    plt.tight_layout()
    plt.show()
