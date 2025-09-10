# %pip install torch
import torch
import matplotlib.pyplot as plt
from torch import nn, Tensor
import wandb as wnb
import os
from data_loader import DataLoader
from tqdm import trange
from pathlib import Path
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# You choose these IDs; example uses “append-on-top” (no shift of real IDs):
# real tokens: 0..V_real-1, EOS = V_real, PAD = V_real+1
class DiscreteFlowSeqHistory(nn.Module):
    def __init__(self,
                 V_total: int,
                 h: int = 64,
                 aug_state_dim: int = 0,
                 n_heads: int = 4,
                 n_layers: int = 2,
                 use_token_feats: bool = True,     # per-token numeric feats (e.g., xyz) + link ids
                 token_feat_dim: int = 0,          # numeric feats per token (e.g., 3 for xyz)
                 num_links: int | None = None,     # set if you pass link ids per token
                 use_token_hist: bool = False,     # per-token ID history [B,H,C]
                 use_state_hist: bool = False,     # global state history [B,H,D]
                 hist_len: int = 100,
                 max_c_num: int = 10,
                 use_c_pos: bool = True,     # contact position history [B,C*3]
                 PAD_ID: int = None):              # set to V_real
        super().__init__()
        assert PAD_ID is not None
        self.h        = h
        self.V_total  = V_total
        self.PAD_ID   = PAD_ID
        self.use_token_feats = use_token_feats
        self.token_feat_dim  = token_feat_dim
        self.use_token_hist  = use_token_hist
        self.use_state_hist  = use_state_hist
        self.has_link_ids    = num_links is not None
        self.use_c_pos = use_c_pos
        self.max_c_num = max_c_num

        # Token embedding (PAD row will be zeroed by padding_idx)
        self.token_embed = nn.Embedding(V_total, h, padding_idx=PAD_ID)

        # Transformer over the sequence (batch_first=True)
        enc_layer = nn.TransformerEncoderLayer(d_model=h, nhead=n_heads,
                                               dim_feedforward=4*h, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # Sinusoidal positional encoding (created on the fly)
        # (no learnable params; tiny helper below)
        # ---- Conditioning: time + current state ----
        self.time_proj     = nn.Linear(1, h)
        self.aug_now_proj  = nn.Linear(aug_state_dim, h) if aug_state_dim > 0 else None

        # Optional per-token features
        if use_token_feats and token_feat_dim > 0:
            self.token_feat_proj = nn.Linear(token_feat_dim, h)
        if use_token_feats and self.has_link_ids:
            self.link_embed = nn.Embedding(num_links, h)

        # Optional per-token ID history pooling (attention over H)
        if use_token_hist:
            self.hist_token_embed = nn.Embedding(V_total, h, padding_idx=PAD_ID)
            self.hist_token_score = nn.Linear(h, 1)

        # Optional global state history pooling (attention over H)
        if use_state_hist and aug_state_dim > 0:
            self.state_hist_proj  = nn.Linear(aug_state_dim, h)
            self.state_hist_score = nn.Linear(h, 1)
            self.state_hist_gate  = nn.Parameter(torch.tensor(1.0))

        # Optional contact position pooling
        if use_c_pos:
            self.c_pos_proj = nn.Linear(3 * max_c_num, h)
            self.c_pos_score = nn.Linear(h, 1)

        # Output head
        self.fc_out = nn.Linear(h, V_total)

    # --- sinusoidal positions ---
    def sinusoidal_pos(self, L: int, device) -> torch.Tensor:
        d = self.h
        pos = torch.arange(L, device=device, dtype=torch.float32).unsqueeze(1)     # [L,1]
        i   = torch.arange(d, device=device, dtype=torch.float32).unsqueeze(0)     # [1,d]
        denom = torch.pow(10000.0, (i//2)*2.0/d)
        emb = pos / denom
        emb[:, 0::2] = torch.sin(emb[:, 0::2])
        emb[:, 1::2] = torch.cos(emb[:, 1::2])
        return emb.unsqueeze(0)   # [1, L, h]

    # --- per-token history pooling over H (masked by pad_mask over C) ---
    def pool_token_hist(self, contact_ids_hist: torch.LongTensor,
                        pad_mask: torch.BoolTensor) -> torch.Tensor:
        # contact_ids_hist: [B, H, C] (model IDs incl PAD/EOS)
        B, H, C = contact_ids_hist.shape
        vecs = self.hist_token_embed(contact_ids_hist)             # [B,H,C,h]

        # kill entire columns where the current sequence position is PAD
        vecs = vecs.masked_fill(pad_mask.unsqueeze(1).unsqueeze(-1), 0.0)

        scores = self.hist_token_score(vecs).squeeze(-1)           # [B,H,C]
        scores = scores.masked_fill(pad_mask.unsqueeze(1), -1e9)
        weights = torch.softmax(scores, dim=1)                     # [B,H,C]
        pooled  = (weights.unsqueeze(-1) * vecs).sum(dim=1)        # [B,C,h]
        pooled  = pooled * (~pad_mask).unsqueeze(-1).float()
        return pooled

    # --- global state history pooling over H ---
    def pool_state_hist(self, aug_state_hist: torch.Tensor) -> torch.Tensor:
        # aug_state_hist: [B, H, D]
        sh = self.state_hist_proj(aug_state_hist)                  # [B,H,h]
        scores  = self.state_hist_score(sh).squeeze(-1)            # [B,H]
        weights = torch.softmax(scores, dim=1)                     # [B,H]
        pooled  = (weights.unsqueeze(-1) * sh).sum(dim=1)          # [B,h]
        return self.state_hist_gate * pooled                       # [B,h]

    def forward(self,
                x_t: torch.LongTensor,        # [B, C] (model IDs incl PAD/EOS)
                t: torch.FloatTensor,             # [B]
                aug_state: torch.FloatTensor | None,   # [B, D] or None
                pad_mask: torch.BoolTensor,       # [B, C]  True=PAD
                token_feats: torch.FloatTensor | None = None,      # [B, C, F] (numeric), optional
                token_link_ids: torch.LongTensor | None = None,     # [B, C] (ints), optional
                contact_ids_hist: torch.LongTensor | None = None,   # [B, H, C] (model IDs), optional
                c_pos: torch.FloatTensor | None = None,     # [B, 3 * C], optional
                # aug_state_hist: torch.FloatTensor | None = None     # [B, H, D], optional
                ) -> torch.Tensor:
        B, C = x_t.shape
        device = x_t.device

        # base token + sinusoidal pos
        tok = self.token_embed(x_t)                            # [B,C,h]
        # pos = self.sinusoidal_pos(C, device).expand(B, -1, -1)     # [B,C,h]
        # tok = tok + pos

        # add optional per-token numeric feats
        if self.use_token_feats and token_feats is not None and token_feats.shape[-1] > 0:
            feat_emb = self.token_feat_proj(token_feats)           # [B,C,h]
            tok = tok + feat_emb

        # add optional per-token link embeddings
        if self.use_token_feats and self.has_link_ids and (token_link_ids is not None):
            link_emb = self.link_embed(token_link_ids.clamp_min(0))# [B,C,h]
            tok = tok + link_emb

        # per-token ID history
        if self.use_token_hist and (contact_ids_hist is not None):
            tok = tok + self.pool_token_hist(contact_ids_hist, pad_mask)   # [B,C,h]

        # global conditioning: time + current state
        x = tok + self.time_proj(t[:, None]).unsqueeze(1)          # [B,1,h] -> broadcast
        if self.aug_now_proj is not None and (aug_state is not None):
            x = x + self.aug_now_proj(aug_state).unsqueeze(1)  # [B,1,h]

        # # global state history
        # if self.use_state_hist and (aug_state_hist is not None):
        #     glob = self.pool_state_hist(aug_state_hist).unsqueeze(1)   # [B,1,h]
        #     x = x + glob

        # contact position conditioning
        if self.use_c_pos and c_pos is not None:
            # c_pos should be [B, C*3] -> project to [B, h] then broadcast to [B, C, h]
            # c_pos = (c_pos - c_pos.mean(dim=1, keepdim=True)) / (c_pos.std(dim=1, keepdim=True) + 1e-6)
            c_pos_emb = self.c_pos_proj(c_pos)  # [B, h]
            c_pos_emb = c_pos_emb.unsqueeze(1)  # [B, 1, h] 
            x = x + c_pos_emb  # Broadcasting: [B, C, h] + [B, 1, h] -> [B, C, h]

        # self-attention across sequence, masking PAD positions
        enc = self.encoder(x, src_key_padding_mask=pad_mask)       # [B,C,h]

        # per-position logits over full vocab (incl PAD/EOS; you’ll block PAD at sampling)
        logits = self.fc_out(enc)                                   # [B,C,V_total]
        return logits

def contactN2linkid(contact_num):
    linkids = []
    for i in range(len(contact_num)):
        c_n = contact_num[i]
        linkid = np.zeros(10, dtype=int)
        for j in range(len(c_n)):
            n_j = c_n[j]
            for n in range(n_j):
                linkid[n] = j
        linkids.append(linkid)
    linkids = np.array(linkids)
    return linkids

def get_samples(batch_size, history_len, data_slice="train"):
    # retrieve samples from dataset, note that each variable is a numpy array
    contact_id, aug_state, c_pos, contact_nums, \
    contact_id_history, aug_state_history, c_pos_history, contact_nums_history \
    = d_loader.sample_contact_ids_robot_state_history(batch_size, history_len=history_len, data_slice=data_slice)    
    
    # augmented state features
    aug_state_aug_history = np.concatenate([aug_state, aug_state_history], axis=-1)
    aug_state_aug_history = torch.tensor(aug_state_aug_history, device=device, dtype=torch.float32)
    aug_state = torch.tensor(aug_state, device=device, dtype=torch.float32)
    
    # padding mask
    pad_mask = contact_id == PAD_ID
    pad_mask = pad_mask.to(device)

    # linkid embeddings
    linkids = contactN2linkid(contact_nums)
    linkids = torch.tensor(linkids, device=device, dtype=torch.long)
    return contact_id, aug_state, aug_state_aug_history, c_pos, contact_nums, pad_mask, linkids

def compute_cpos_particles(c_pos, pad_mask, x_t, sample_size):
    # Replace your training section starting from the c_pos_x_t_tmp line
    # instead of current contact positions
    c_pos_x_t_tmp = torch.zeros_like(c_pos, device=device, dtype=torch.float32)
    pos_pad = pad_mask.unsqueeze(-1).expand(-1, -1, 3)  # Shape: [B, C, 3]
    pos_pad = pos_pad.reshape(sample_size, -1)

    # Get positions for non-PAD contact IDs
    non_pad_positions = torch.where(~pad_mask)
    non_pad_contact_ids = x_t[non_pad_positions].cpu().numpy().tolist()
    retrieved_positions = d_loader.retrieve_contact_pos_from_ids(non_pad_contact_ids)
    retrieved_tensor = torch.tensor(retrieved_positions, device=device, dtype=torch.float32)
    c_pos_x_t_tmp[~pos_pad] = retrieved_tensor.flatten()
    return c_pos_x_t_tmp

if __name__ == "__main__":
    # Make sure JAX uses CPU
    os.environ["JAX_PLATFORM_NAME"] = "cpu"

    # Load the dataset
    file_name = "dataset_batch_1_20eps"
    dir_name = "data-link7-2-contact_v3"
    d_loader = DataLoader(file_name, dir_name)
    
    # Get the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Training
    batch_size = 256
    lr = 0.001
    num_epochs = 1000
    max_contact_id = d_loader.max_contact_id
    vocab_size = int(max_contact_id) + 1 
    PAD_ID = vocab_size
    V_total = vocab_size + 1  # total vocab size including PAD
    PAD_POS_VALUE = 0.0

    # Hyperflag
    use_wnb = False

    # History len
    his_len = 100    
    
    _, aug_state_sample, _, _, \
    _, aug_state_sample_history, _, _, \
    = d_loader.sample_contact_ids_robot_state_history(10, history_len=his_len)
    aug_state_sample = np.array(aug_state_sample)
    aug_state_dim = aug_state_sample.shape[1] if aug_state_sample.ndim > 1 else 1
    aug_state_sample_history = np.array(aug_state_sample_history)
    aug_state_history_dim = aug_state_sample_history.shape[1] if aug_state_sample_history.ndim > 1 else 1
    aug_state_dim = aug_state_dim + aug_state_history_dim
    print(f"Augmented state dimension: {aug_state_dim}")

    # Define model    
    model = DiscreteFlowSeqHistory(V_total=V_total, aug_state_dim=aug_state_dim, PAD_ID=PAD_ID)
    model = model.to(device)
    
    # load the model if it exists
    model_dir = (Path(__file__).resolve().parent / "models").as_posix()
    model_name = "low_level_discrete_flow_2contacts"
    model_path = f"{model_dir}/{model_name}.pth"

    if use_wnb:
        wnb.init(project="Contact Force Estimation IIWA", name=f"{model_name}")
        wnb.watch(model, log="all")
    
    # TODO: generate proper sampling space for each link
    sampling_space = d_loader.data_dict["global_start_end_indices"]
    link_7_sampling_space = sampling_space[-1]
    if os.path.exists(model_path):
        print(f"========================== LOADING MODEL FROM {model_path} ==========================")
        model.load_state_dict(torch.load(model_path))
    else:
        # if the model doesn't exist, start the training
        optim = torch.optim.Adam(model.parameters(), lr=lr)
        print("========================== ENTERING TRAINING LOOP ========================== ")
        for epoch in trange(num_epochs):
            # clean cpu cache
            torch.cuda.empty_cache()
            contact_id, aug_state, aug_state_aug_history, c_pos, \
            contact_nums, pad_mask, link_ids = \
            get_samples(batch_size, his_len, data_slice="train")

            x_1 = contact_id
            x_1 = x_1.to(device)

            x_0 = torch.randint(low=link_7_sampling_space[0], high=link_7_sampling_space[1], size=x_1.shape, device=device)

            # fill x_0 with <PAD_ID>
            x_0[pad_mask] = PAD_ID
            t = torch.rand(batch_size, device=device)  # Random time step between 0 and 1
            # create x_t by sampling from x_1 and x_0
            x_t = torch.where(torch.rand(batch_size, x_1.shape[1], device=device) < t[:, None], x_1, x_0)

            # retrieve the contact positions for x_t
            c_pos_x_t = compute_cpos_particles(c_pos, pad_mask, x_t, batch_size)
            
            # sample also contact positions history, retreive the pad_mask for it and fill with all 0
            logits = model(x_t=x_t, t=t, aug_state=aug_state_aug_history, pad_mask=pad_mask)

            # what is the loss
            loss = nn.functional.cross_entropy(
                logits.view(-1, V_total), 
                x_1.view(-1)).mean()
            
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
            
            if use_wnb:
                wnb.log({"loss": loss.item()})
                wnb.log({"learning_rate": optim.param_groups[0]['lr']})
            
        # save the model
        torch.save(model.state_dict(), model_path)

    num_test = 100
    correct_total = 0
    avg_err = 0.0
    vis_result = True
    mode = "validate"
    min_val = int(link_7_sampling_space[0])
    max_val = int(link_7_sampling_space[1]) - 1
    
    # Sampling, test with 100 samples
    for i in range(num_test):
        num_samples = 400
        contact_id, aug_state, aug_state_aug_history, c_pos, \
        contact_nums, pad_mask, link_ids = \
        get_samples(num_samples, his_len, data_slice=mode)

        x_1 = contact_id
        x_1 = x_1.to(device)

        # get the first one
        random_idx = np.random.randint(0, contact_id.shape[0])
        x_1_0 = x_1[random_idx]
        aug_state_aug_history_0 = aug_state_aug_history[random_idx]

        # repeat it such that it has the same size as x_1
        x_1 = x_1_0.unsqueeze(0).repeat(x_1.shape[0], 1)
        aug_state_aug_history = aug_state_aug_history_0.unsqueeze(0).repeat(aug_state_aug_history.shape[0], 1)

        # retrieve the ground truth contact positions
        x_1_0_mask = x_1_0 == PAD_ID
        ground_truth_contact_positions = d_loader.retrieve_contact_pos_from_ids(
            x_1_0[~x_1_0_mask].cpu().numpy().tolist()
        )
        x_1_mask = x_1 == PAD_ID
        n_contacts = len(x_1_0[~x_1_0_mask])
        print("Number of contacts:", n_contacts)
        if n_contacts == 0:
            continue

        # generate random x_t
        x_t = torch.randint(low=link_7_sampling_space[0], high=link_7_sampling_space[1], size=(contact_id.shape[0], contact_id.shape[1]), device=device)
        x_t[x_1_mask] = PAD_ID
        pad_mask = x_1_mask

        t = 0.0
        div = 2
        h_c = 0.1 / div
        results = [(x_t.clone(), t)]
        results_x_t = [x_t.clone()]
        
        x_t_c_pos = d_loader.retrieve_contact_pos_from_ids(
            x_t[~pad_mask].cpu().numpy().tolist()
        )
        x_t_c_pos = x_t_c_pos.reshape(num_samples, n_contacts, 3)
        results_c_pos = [x_t_c_pos.copy()]

        # Retrieve the contact positions for x_t
        c_pos_x_t = compute_cpos_particles(c_pos, pad_mask, x_t, num_samples)

        print("Starting sampling...")
        with torch.no_grad():
            while t < 1.0 - 1e-3:
                t_tensor = torch.full((num_samples,), t, device=device)
                logits = model(x_t=x_t, t=t_tensor, aug_state=aug_state_aug_history, pad_mask=pad_mask)

                # CRITICAL: Mask PAD token logits to prevent predicting PAD where it shouldn't be
                logits[:, :, PAD_ID] = torch.where(
                    pad_mask,  # Where current positions are PAD
                    logits[:, :, PAD_ID],  # Keep original logits (allow PAD prediction)
                    torch.full_like(logits[:, :, PAD_ID], -float('inf'))  # Block PAD prediction
                )
                
                p1 = torch.softmax(logits, dim=-1)  # Shape: (num_samples, seq_len, V_total)
                h = min(h_c, 1.0 - t)
                
                # One-hot encoding with correct vocabulary size
                one_hot_x_t = nn.functional.one_hot(x_t, num_classes=V_total).float()
                
                # Flow update rule with numerical stability
                u = (p1 - one_hot_x_t) / (1.0 - t + 1e-8)
                
                # Update probabilities
                updated_probs = one_hot_x_t + h * u
                updated_probs = torch.clamp(updated_probs, min=1e-8)
                updated_probs = updated_probs / updated_probs.sum(dim=-1, keepdim=True)
                
                # Sample new tokens
                x_t_new = torch.distributions.Categorical(probs=updated_probs).sample()
                
                # limit the sampling to the link 7 sampling space
                x_t_new = torch.clamp(x_t_new, min=min_val, max=max_val)

                # Preserve existing PAD tokens (crucial!)
                x_t = torch.where(pad_mask, PAD_ID, x_t_new)
                
                t += h
                results.append((x_t.clone(), t))
                results_x_t.append(x_t.clone())
                
                x_t_c_pos = d_loader.retrieve_contact_pos_from_ids(
                    x_t[~pad_mask].cpu().numpy().tolist()
                )
                x_t_c_pos = x_t_c_pos.reshape(num_samples, n_contacts, 3)
                results_c_pos.append(x_t_c_pos)
                if len(results_x_t) % 5 == 0:
                    print(f"Sampling step {len(results_x_t)}, t={t:.3f}")

        # retrieve the contact positions
        init_predict = results_x_t[0]
        final_predict = results_x_t[-1]

        correct_init = (init_predict == x_1) & (~pad_mask)
        accuracy_init = correct_init.float().mean().item()
        # print(f"Initial predicted accuracy: {accuracy_init:.4f}")
        
        # compute the predicted accuracy
        correct = (final_predict == x_1) & (~pad_mask)
        accuracy = correct.float().mean().item()
        # print(f"Final predicted accuracy: {accuracy:.4f}")
        
        # get the feasible ids
        feasible_ids = final_predict[~pad_mask]
        feasible_ids = feasible_ids.cpu().numpy().tolist()
        
        # sample contact positions from feasible_ids
        final_contact_positions = d_loader.retrieve_contact_pos_from_ids(feasible_ids)
        
        # get the initial feasible ids
        initial_feasible_ids = init_predict[~pad_mask]
        initial_feasible_ids = initial_feasible_ids.cpu().numpy().tolist()
        initial_contact_positions = d_loader.retrieve_contact_pos_from_ids(initial_feasible_ids)
        
        # get the ground truth feasible ids
        ground_truth_feasible_ids = x_1[~pad_mask]
        ground_truth_feasible_ids = ground_truth_feasible_ids.cpu().numpy().tolist()
        ground_truth_contact_positions = d_loader.retrieve_contact_pos_from_ids(ground_truth_feasible_ids)

        # compute the l2 distances
        final_l2_distances = np.linalg.norm(final_contact_positions - ground_truth_contact_positions, axis=-1)
        initial_l2_distances = np.linalg.norm(initial_contact_positions - ground_truth_contact_positions, axis=-1)

        # mode of the predicted contact positions
        from scipy import stats
        final_contact_positions_ = final_contact_positions.reshape(num_samples, n_contacts, 3)
        ground_truth_contact_positions_ = ground_truth_contact_positions.reshape(num_samples, n_contacts, 3)
        final_contact_positions_mode, mode_counts = stats.mode(final_contact_positions_, axis=0, keepdims=False)
        final_contact_positions_mean = final_contact_positions_.mean(axis=0)

        # select the second frequency mode, first remove the final contact positions mode
        mode_mask = final_contact_positions_ != final_contact_positions_mode
        final_contact_positions_remove_mode = final_contact_positions_[mode_mask]
        find_contact_positions_second_mode, second_mode_counts = stats.mode(final_contact_positions_remove_mode.reshape(-1, 3), axis=0, keepdims=False)

        # Compare the mode and ground truth
        print("Ground Truth Contact Positions:", ground_truth_contact_positions_[0])
        print("Final Contact Positions Mode:", final_contact_positions_mode)
        print("Second frequent Contact Positions:", find_contact_positions_second_mode)
        print("Final Contact Positions Mean:", final_contact_positions_mean)
        print("Mode Counts:", mode_counts, "Second Mode Counts:", second_mode_counts, " Total Counts: ", num_samples)
        print("ERROR mode to ground truth:", np.mean(final_contact_positions_mode - ground_truth_contact_positions_))
        final_modes_mode_num = [stats.mode(final_contact_positions_[:, j, :], axis=0) for j in range(n_contacts)]

        # Global mode
        final_contact_positions_flat_ = final_contact_positions_.reshape(-1, 3)
        final_contact_positions_mode_global, mode_counts_global = stats.mode(final_contact_positions_flat_, axis=0, keepdims=False)
        mode_mask_global = final_contact_positions_flat_ != final_contact_positions_mode_global
        final_contact_positions_remove_mode_global = final_contact_positions_flat_[mode_mask_global]
        find_contact_positions_second_mode_global, second_mode_counts_global = stats.mode(final_contact_positions_remove_mode_global.reshape(-1, 3), axis=0, keepdims=False)
        print("Global Mode:", final_contact_positions_mode_global, "Counts:", mode_counts_global)
        print("Global Second frequent Contact Positions:", find_contact_positions_second_mode_global, "Counts:", second_mode_counts_global)
                    
        # compute the predicted accuracy
        if final_contact_positions_mode_global in ground_truth_contact_positions_[0] and \
           find_contact_positions_second_mode_global in ground_truth_contact_positions_[0]:
            correct_total += 1

        if vis_result:
            # visualize the evolution of contact positions in 3D
            num_figs = min(len(results), 11) # Limit to 11 subplots for readability
            fig = plt.figure(figsize=(20, 10))

            # Create a grid layout for 3D subplots
            rows = 2
            cols = (num_figs + 1) // 2

            for i, (x_t, t) in enumerate(results[:]):
                if i % div == 0:
                    ax = fig.add_subplot(rows, cols, i//div+1, projection='3d')
                    
                    for j in range(n_contacts):
                        results_c_pos_dis = results_c_pos[i][:, j, :]
                        
                        # Plot predicted contact positions (blue points)
                        ax.scatter(
                            results_c_pos_dis[:, 0], 
                            results_c_pos_dis[:, 1], 
                            results_c_pos_dis[:, 2],
                            s=15, alpha=0.6, color='blue', marker='o', label='Predicted' if j == 0 else ""
                        )
                        
                        # Plot ground truth contact position (red cross)
                        ax.scatter(
                            ground_truth_contact_positions[j, 0], 
                            ground_truth_contact_positions[j, 1], 
                            ground_truth_contact_positions[j, 2],
                            s=100, color='red', marker='x', linewidths=3, label='Ground Truth' if j == 0 else ""
                        )
                    
                    ax.set_title(f"t={t:.2f}", fontsize=12, fontweight='bold')
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y') 
                    ax.set_zlabel('Z')
                    
                    # Set equal aspect ratio for better visualization
                    max_range = np.array([
                        results_c_pos_dis[:, 0].max() - results_c_pos_dis[:, 0].min(),
                        results_c_pos_dis[:, 1].max() - results_c_pos_dis[:, 1].min(),
                        results_c_pos_dis[:, 2].max() - results_c_pos_dis[:, 2].min()
                    ]).max() / 2.0
                    
                    mid_x = (results_c_pos_dis[:, 0].max() + results_c_pos_dis[:, 0].min()) * 0.5
                    mid_y = (results_c_pos_dis[:, 1].max() + results_c_pos_dis[:, 1].min()) * 0.5
                    mid_z = (results_c_pos_dis[:, 2].max() + results_c_pos_dis[:, 2].min()) * 0.5
                    
                    ax.set_xlim(mid_x - max_range, mid_x + max_range)
                    ax.set_ylim(mid_y - max_range, mid_y + max_range)
                    ax.set_zlim(mid_z - max_range, mid_z + max_range)
                    
                    if i == 0:  # Add legend to first subplot
                        ax.legend()
            
            plt.suptitle('Evolution of Contact Positions During Flow Matching', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.show()
    print(f"Overall accuracy of mode prediction: {correct_total}/{num_test} = {correct_total/num_test:.4f}")