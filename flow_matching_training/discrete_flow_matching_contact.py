# %pip install torch
import torch
import matplotlib.pyplot as plt
from torch import nn, Tensor
import wandb as wnb
import numpy as np
import os
from discrete_flow_matching_moon import chi_square_test
from data_loader import DataLoader
from tqdm import trange
from pathlib import Path

class DiscreteFlow(nn.Module):
    def __init__(self, dim: int = 1, h: int = 16, v: int = 128, aug_state_dim: int = 0, nn_neibors: int = 20):
        super().__init__()
        self.v = v
        self.h = h
        self.embed = nn.Embedding(v, h)
        self.k = nn_neibors
        # input_size = dim * h + 1 + aug_state_dim  # t + embedding + aug_state
        input_size = nn_neibors * 3 + 1 * 3 + 1 + h + aug_state_dim  # Assuming dim * h = 64 for contact IDs
        self.net = nn.Sequential(
            nn.Linear(input_size, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, h), nn.ELU(),
            nn.Linear(h, dim * v))

    def embed_contact_ids(self, x_t: Tensor) -> Tensor:
        contact_pos = d_loader.retreive_contact_pos_from_ids_tensor(x_t).view(-1, 3).to(device)  # Assuming x_t is a tensor of contact IDs
        nn_contact_pos = d_loader.retreive_nn_neibors_from_ids_tensor(x_t, k=self.k).view(-1, self.k * 3).to(device) # Assuming each contact position has 3 dimensions
        contact_link_id = d_loader.retreive_link_ids_from_ids_tensor(x_t).view(-1, 1).to(device)
        contact_link_id_embed = self.embed(contact_link_id).view(-1, self.h)
        return torch.cat((contact_pos, contact_link_id_embed, nn_contact_pos), dim=-1)

    def forward(self, x_t: Tensor, t: Tensor, aug_state: Tensor = None) -> Tensor:
        if aug_state is not None:
            input_tensor = torch.cat((t[:, None], self.embed_contact_ids(x_t), aug_state), -1)
        else:
            input_tensor = torch.cat((t[:, None], self.embed_contact_ids(x_t)), -1)
        return self.net(input_tensor).reshape(list(x_t.shape) + [self.v])
    
def compute_vacob_size(min_contact_id: int, max_contact_id: int) -> int:
    """
    Compute the vocabulary size based on the range of contact IDs.
    """
    max_contact_id_reduced =  max_contact_id - min_contact_id
    for i in range(1, 20):
        if 2 ** i > max_contact_id_reduced:
            return 2 ** i
    
if __name__ == "__main__":
    # Load the dataset
    d_loader = DataLoader("dataset_batch_1_1000eps")
    
    # Get the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Training
    batch_size = 4096
    lr = 0.001
    num_epochs = 10
    max_contact_id = d_loader.max_contact_id  # Maximum contact ID for scaling
    vocab_size = int(max_contact_id) + 1 
    # vocab_size = compute_vacob_size(d_loader.min_contact_id, max_contact_id)
    if vocab_size < max_contact_id - d_loader.min_contact_id:
        print(f"Warning: vocab_size {vocab_size} is less than max_contact_id {max_contact_id}. This may lead to unexpected behavior.")
        exit(1)
    use_wnb = False
    condition_previous_contact = False  # Whether to condition on the previous contact ID
    use_aug_state = True
        
    if use_aug_state:
        aug_state_dim = d_loader.joint_pos_data.shape[1] + d_loader.joint_vel_data.shape[1] + d_loader.joint_tau_cmd_data.shape[1] + d_loader.joint_tau_ext_gt_data.shape[1]
    else:
        aug_state_dim = 0
    model = DiscreteFlow(v=vocab_size, dim=1, aug_state_dim=aug_state_dim)
    model = model.to(device)

    # load model if exists
    cond_prev_con_str = "cond_prev_contact" if condition_previous_contact else ""
    aug_state_str = "aug_state" if use_aug_state else ""
    file_name = f"discrete_flow_contact{num_epochs}{cond_prev_con_str}{aug_state_str}Vocab{vocab_size}.pth"
    dir_path = (Path(__file__).resolve().parent / "models").as_posix()
    file_path = f"{dir_path}/{file_name}"
    if os.path.exists(file_path):
        print(f"Loading model from {file_path}")
        model.load_state_dict(torch.load(file_path))
    else:
        print(f"Model file {file_path} does not exist. Starting training from scratch.")
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    if use_wnb:
        wnb.init(project="Contact Force Estimation", name=f"DFMContact{cond_prev_con_str}{aug_state_str}Vocab{vocab_size}")
        wnb.watch(model, log="all")
        
        # log hyperparameters
        wnb.config.update({
            "batch_size": batch_size,
            "vocab_size": vocab_size,
            "learning_rate": lr,
            "num_epochs": num_epochs,
        })
    if not os.path.exists(file_path):
        print("========================== ENTERING TRAINING LOOP ========================== ")
        for epoch in trange(num_epochs):
            if not condition_previous_contact:
                x_1, aug_state, c_pos = d_loader.sample_contact_ids_robot_state(batch_size)
                x_1 = x_1.to(device)
                x_0 = torch.randint(low=0, high=vocab_size, size=(batch_size, ), device=device)
            else:
                raise NotImplementedError("Conditioning on previous contact ID is not implemented yet.")

            t = torch.rand(batch_size, device=device)  # Random time step between 0 and 1
            x_t = torch.where(torch.rand(batch_size, device=device) < t, x_1, x_0)

            if use_aug_state:
                aug_state = aug_state.view(batch_size, -1).to(device)
                logits = model(x_t=x_t, t=t, aug_state=aug_state)
            else:
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
    num_samples = 1000
    if not condition_previous_contact:
        x_t = torch.randint(low=0, high=vocab_size, size=(num_samples, ))
        x_1, aug_state, c_pos = d_loader.sample_contact_ids_robot_state(batch_size=num_samples)
    else:
        raise NotImplementedError("Conditioning on previous contact ID is not implemented yet.")
    x_t = x_t.to(device)
    aug_state = aug_state.to(device)
    x_1 = x_1.to(device)
    t = 0.0
    div = 1
    h_c = 0.1 / div
    results = [(x_t, t)]
    results_x_t = []
    while t < 1.0 - 1e-3:
        if not use_aug_state:
            p1 = torch.softmax(model(x_t, torch.ones(num_samples, device=device) * t), dim=-1)
        else:
            p1 = torch.softmax(model(x_t, torch.ones(num_samples, device=device) * t, aug_state=aug_state), dim=-1)
        h = min(h_c, 1.0 - t)
        one_hot_x_t = nn.functional.one_hot(x_t, vocab_size).float()
        u = (p1 - one_hot_x_t) / (1.0 - t)
        x_t = torch.distributions.Categorical(probs=one_hot_x_t + h * u).sample()
        t += h
        results.append((x_t, t))
        results_x_t.append(x_t)

    # Compare the bar plot of the sampled contact IDs and the original contact IDs
    final_results_x_t = torch.stack([results_x_t[-1]])
    final_results_x_t = final_results_x_t.flatten()
    init_results_x_t = torch.stack([results_x_t[0]])
    init_results_x_t = init_results_x_t.flatten()
    
    # Compute the prediction error, sums up the total number of predicted contact IDs
    correct_predictions = (final_results_x_t == x_1).sum().item()
    total_predictions = final_results_x_t.shape[0]
    accuracy = correct_predictions / total_predictions

    corrent_predictions_init = (init_results_x_t == x_1).sum().item()
    total_predictions_init = init_results_x_t.shape[0]
    accuracy_init = corrent_predictions_init / total_predictions_init
    print(f"Initial accuracy of contact IDs: {accuracy_init * 100:.2f}%")
    print(f"Accuracy of sampled contact IDs: {accuracy * 100:.2f}%")
    
    # table = [final_results_x_t.numpy(), dset_x1.numpy()]
    x_1 = x_1.cpu()
    final_results_x_t = final_results_x_t.cpu()
    chi2, p = chi_square_test(x_1.numpy(), final_results_x_t.numpy())
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(final_results_x_t.numpy(), bins=50, label='Sampled Contact IDs', density=True)
    plt.hist(x_1.numpy(), bins=50, label='Original Contact IDs', density=True)
    plt.title('Distribution of Contact IDs')
    plt.xlabel('Contact ID')
    plt.ylabel('Frequency')
    plt.legend()    
    plt.tight_layout()
    plt.show()
