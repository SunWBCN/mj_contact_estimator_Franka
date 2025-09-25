# %pip install torch
import torch
import matplotlib.pyplot as plt
from torch import nn, Tensor
import wandb as wnb
import os
from tqdm import trange
from pathlib import Path
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from h5_torch_dataset import H5Dataset, make_dataloaders
import h5py
import time
from data_loader import DataLoader

# Add the missing DiscreteFlowSeqHistory class
class DiscreteFlowSeqHistory(nn.Module):
    def __init__(self,
                 V_total,
                 h: int = 64,
                 num_links: int = None,
                 token_feat_dim: int = 0, 
                 n_heads: int = 4,
                 n_layers: int = 2,
                 aug_state_dim: int = 0,
                 max_c_num: int = 10,
                 use_c_pos: bool = False,
                 use_concat: bool = False,
                 use_token_feats: bool = True,
                 use_state_history: bool = False,
                 PAD_ID: int = None):
        super().__init__()
        self.use_c_pos = use_c_pos
        self.use_concat = use_concat
        self.use_token_feats = use_token_feats
        self.has_link_ids = num_links is not None
        self.use_state_history = use_state_history

        # Calculate total embedding dimension
        total_dim = h  # Base token embedding
        if use_concat:
            if token_feat_dim > 0 and use_token_feats:
                total_dim += h  # Add h for token features
            if num_links is not None:
                total_dim += h  # Add h for link embeddings
        self.total_embedding_dim = total_dim
        
        # Base embeddings
        self.token_embed = nn.Embedding(V_total, h, padding_idx=PAD_ID)
        
        if token_feat_dim > 0 and use_token_feats:
            self.token_feat_proj = nn.Linear(token_feat_dim, h)
        if num_links is not None:
            self.link_embed = nn.Embedding(num_links, h)
        
        # Transformer with correct dimension
        enc_layer = nn.TransformerEncoderLayer(
            d_model=total_dim,
            nhead=n_heads,
            dim_feedforward=4*total_dim,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        
        # Projections with correct dimensions
        self.time_proj = nn.Linear(1, total_dim)
        self.aug_now_proj = nn.Linear(aug_state_dim, total_dim) if aug_state_dim > 0 else None
        self.c_pos_proj = nn.Linear(3 * max_c_num, total_dim) if use_c_pos else None
        
        # Output projection
        self.fc_out = nn.Linear(total_dim, V_total)

        # History attention components
        if use_state_history:
            self.state_hist_attention = nn.MultiheadAttention(
                embed_dim=aug_state_dim,
                num_heads=n_heads,
                batch_first=True
            )
            self.state_hist_proj = nn.Linear(aug_state_dim, total_dim)
            self.state_hist_gate = nn.Parameter(torch.tensor(0.5))

    def forward(self, x_t, t, aug_state, pad_mask, token_feats=None, 
                token_link_ids=None, c_pos=None, aug_state_history=None, **kwargs):
        B, C = x_t.shape
        device = x_t.device
        
        if self.use_concat:
            # Concatenation approach
            embeddings_list = []
            
            # Base token embedding
            tok = self.token_embed(x_t)  # [B,C,h]
            embeddings_list.append(tok)
            
            # Token features
            if self.use_token_feats and token_feats is not None and token_feats.shape[-1] > 0:
                feat_emb = self.token_feat_proj(token_feats)  # [B,C,h]
                embeddings_list.append(feat_emb)
            
            # Link embeddings  
            if self.has_link_ids and token_link_ids is not None:
                link_emb = self.link_embed(token_link_ids.clamp_min(0))  # [B,C,h]
                embeddings_list.append(link_emb)
            
            # Concatenate all embeddings
            x = torch.cat(embeddings_list, dim=-1)  # [B,C,total_embedding_dim]
            
        else:
            # Original summation approach
            tok = self.token_embed(x_t)  # [B,C,h]
            if self.use_token_feats and token_feats is not None:
                feat_emb = self.token_feat_proj(token_feats)
                tok = tok + feat_emb
            if self.has_link_ids and token_link_ids is not None:
                link_emb = self.link_embed(token_link_ids.clamp_min(0))
                tok = tok + link_emb
            x = tok  # [B,C,h]
        
        # Add time and state conditioning
        x = x + self.time_proj(t[:, None]).unsqueeze(1)  # [B,1,total_embedding_dim]
        if self.aug_now_proj is not None and aug_state is not None:
            x = x + self.aug_now_proj(aug_state).unsqueeze(1)  # [B,1,total_embedding_dim]
        
        # History attention
        if self.use_state_history and aug_state_history is not None:
            state_hist_emb = self.attend_state_history(aug_state_history)  # [B,total_embedding_dim]
            x = x + state_hist_emb.unsqueeze(1)  # [B,1,total_embedding_dim] -> [B,C,total_embedding_dim]

        # Contact position conditioning
        if self.use_c_pos and c_pos is not None:
            c_pos_emb = self.c_pos_proj(c_pos).unsqueeze(1)  # [B,1,total_embedding_dim]
            x = x + c_pos_emb
        
        # Transformer encoder
        enc = self.encoder(x, src_key_padding_mask=pad_mask)  # [B,C,total_embedding_dim]
        
        # Output logits
        logits = self.fc_out(enc)  # [B,C,V_total]
        return logits
    
    def attend_state_history(self, aug_state_history):
        """
        Attend over robot state history
        Input: aug_state_history [B, H, aug_state_dim]
        Output: [B, total_embedding_dim] - aggregated history features
        """
        B, H, d = aug_state_history.shape
        
        # Self-attention over history sequence
        attended_history, _ = self.state_hist_attention(
            query=aug_state_history,      # [B, H, d]
            key=aug_state_history,        # [B, H, d] 
            value=aug_state_history       # [B, H, d]
        )  # Output: [B, H, d]
        
        # Aggregate over history dimension
        pooled_history = attended_history.mean(dim=1)  # [B, d]
        
        # Project to model dimension and gate
        projected = self.state_hist_proj(pooled_history)  # [B, total_embedding_dim]
        return self.state_hist_gate * projected

def ss2links(sampling_space, robot_name="kuka_iiwa"):
    if robot_name == "kuka_iiwa":
        num_links = 7
        link_meshes = [1, 2, 1, 2, 1, 2, 1]
        assert len(link_meshes) == num_links, "Number of links does not match length of link_meshes"
        link_ranges = []
        for i in range(num_links):
            mesh_count = link_meshes[i]
            start_i = sum(link_meshes[:i])
            end_i = start_i + mesh_count
            start_idx = sampling_space[start_i][0]
            end_idx = sampling_space[end_i - 1][1]
            link_ranges.append((start_idx, end_idx))
    else:
        raise NotImplementedError("Only kuka_iiwa is implemented")
    return link_ranges

# Contact position retrieval from H5 dataset
class ContactPositionRetriever:
    def __init__(self, h5_path):
        self.h5_path = h5_path
        # Load mesh data for position lookup
        with h5py.File(h5_path, 'r') as f:
            # Assuming you have a mapping from contact IDs to positions in the H5 file
            if 'contact_id_to_position' in f:
                self.id_to_pos = dict(f['contact_id_to_position'][:])
            else:
                # Fallback: create dummy positions
                self.id_to_pos = {}
                print("Warning: No contact_id_to_position mapping found, using dummy positions")
    
    def retrieve_contact_pos_from_ids(self, contact_ids):
        """Retrieve contact positions from contact IDs"""
        positions = []
        for cid in contact_ids:
            if cid in self.id_to_pos:
                positions.append(self.id_to_pos[cid])
            else:
                # Dummy position if not found
                positions.append([0.0, 0.0, 0.0])
        return np.array(positions)

def generate_random_samples_ultra_fast(link_ids, x_1, ss_links):
    """Ultra-fast fully vectorized version - no loops, no .item() calls"""
    batch_size, seq_len = x_1.shape
    device = x_1.device
    
    # Initialize output
    random_samples = torch.zeros_like(x_1)
    
    # Create mask for valid positions (avoid .sum().item() - expensive!)
    valid_mask = link_ids >= 0
    
    if valid_mask.any():
        # Get all valid link IDs at once (no loops!)
        valid_link_ids = link_ids[valid_mask]
        
        # Vectorized range lookup
        starts = ss_links[valid_link_ids, 0]
        ends = ss_links[valid_link_ids, 1]
        ranges = ends - starts
        
        # Generate all random numbers in one operation
        randoms = torch.rand(len(valid_link_ids), device=device)
        random_values = starts + (randoms * ranges.float()).long()
        
        # Assign back to output (vectorized)
        random_samples[valid_mask] = random_values
    
    return random_samples

if __name__ == "__main__":
    # Define number of contacts and link names
    num_contacts = 3
    link_names = "link7"
    traj_mode = False  # Set to True for trajectory mode, False otherwise

    # Data loader setup
    file_name = "dataset_batch_1_200eps"
    dir_name = f"data-{link_names}-{num_contacts}contact_100_v5"

    # Setup H5 dataset
    h5_path = Path(__file__).resolve().parent / "dataset" / dir_name / f"{file_name}.h5"
    if not h5_path.exists():
        print(f"H5 file not found at {h5_path}")
        print("Please run the preprocessing script first:")
        print(f"python preprocess_to_hdf5.py --file_name {file_name} --dir_name {dir_name}")
        exit(1)

    # Get the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    file_name = "dataset_batch_1_200eps"
    dir_name = f"data-{link_names}-{num_contacts}contact_100_v5"
    d_loader = DataLoader(file_name, dir_name, only_mesh=True)

    # Training parameters
    batch_size = 64
    lr = 0.001
    num_epochs = 100
    
    # Get max_contact_id from H5 file
    with h5py.File(h5_path, 'r') as f:
        max_contact_id = int(f.attrs['max_contact_id'])
        # Fix: Read sampling space from mappings dataset, not attributes
        sampling_space = f["mappings"]["global_start_end_indices"][:]
    
    vocab_size = int(max_contact_id) + 1 
    PAD_ID = vocab_size
    V_total = vocab_size + 1  # total vocab size including PAD
    PAD_POS_VALUE = 0.0

    # Hyperflag
    use_wnb = False

    # History len
    his_len = 100    
    
    # Create H5 dataloaders
    train_loader, test_loader, val_loader = make_dataloaders(
        h5_path,
        batch_size=batch_size,
        history_len=his_len,
        history_layout="stack",  # or "flat"
        num_workers=16,
        train_range=(1000, 1800),  # Specify ranges explicitly
        test_range=(1800, 2000),
        val_range=(3000, 4000),
        traj_mode=traj_mode
    )
    
    print(f"Created dataloaders:")
    print(f"  Train: {len(train_loader)} batches")
    print(f"  Test: {len(test_loader)} batches")
    print(f"  Val: {len(val_loader)} batches") 

    # Get augmented state dimension from a sample batch
    print("Fetching a sample batch to determine augmented state dimension...")
    sample_batch = next(iter(train_loader))
    aug_state_dim = sample_batch["aug_state"].shape[1]
    print(f"Augmented state dimension: {aug_state_dim}")

    # Convert sampling space and create link ranges
    ss_links = ss2links(sampling_space, robot_name="kuka_iiwa")
    ss_links_tensor = torch.as_tensor(ss_links, device=device, dtype=torch.long)
    print(f"Link sampling ranges: {ss_links}")

    # Define model    
    model = DiscreteFlowSeqHistory(
        V_total=V_total, 
        aug_state_dim=aug_state_dim, 
        PAD_ID=PAD_ID, 
        num_links=7,
        use_token_feats=False, 
        use_concat=True, 
        use_state_history=True
    )
    model = model.to(device)
    
    # Load the model if it exists
    model_dir = (Path(__file__).resolve().parent / "models").as_posix()
    model_name = f"low_level_discrete_flow_{link_names}_{num_contacts}contacts_h5"
    model_path = f"{model_dir}/{model_name}.pth"

    if use_wnb:
        wnb.init(project="Contact Force Estimation IIWA", name=f"{model_name}")
        wnb.watch(model, log="all")
    
    if os.path.exists(model_path):
        print(f"========================== LOADING MODEL FROM {model_path} ==========================")
        model.load_state_dict(torch.load(model_path))
    else:
        # Training loop with H5 dataset
        optim = torch.optim.Adam(model.parameters(), lr=lr)
        print("========================== ENTERING TRAINING LOOP WITH H5 DATASET ========================== ")
        
        for epoch in trange(num_epochs):
            # # Clean GPU cache
            # import time
            # if torch.cuda.is_available():
            #     torch.cuda.empty_cache()

            epoch_loss = 0.0
            num_batches = 0
            
            before_time = time.time()
            for batch_idx, batch in enumerate(train_loader):
                import time
                
                # 1. Data loading timing
                data_start = time.time()
                # if batch_idx == 0:
                #     print(data_start - before_time, "seconds since last batch end")

                contact_id = batch["contact_ids"].to(device)
                aug_state = batch["aug_state"].to(device)
                c_pos = batch["contact_positions"].to(device)
                contact_nums = batch["contact_nums"].to(device)
                aug_history = batch["hist_aug_state"].to(device) if "hist_aug_state" in batch else None
                pad_mask = batch["pad_mask"].to(device)
                link_ids = batch["link_ids"].to(device)
                data_end = time.time()
                
                # 2. Data preparation timing
                prep_start = time.time()
                current_batch_size = contact_id.shape[0]
                x_1 = contact_id
                
                # Generate random samples for flow matching
                x_0 = generate_random_samples_ultra_fast(link_ids, x_1, ss_links_tensor)
                x_0[pad_mask] = PAD_ID
                
                t = torch.rand(current_batch_size, device=device)
                x_t = torch.where(torch.rand(current_batch_size, x_1.shape[1], device=device) < t[:, None], x_1, x_0)
                prep_end = time.time()
                
                # 3. Forward pass timing
                forward_start = time.time()
                logits = model(
                    x_t=x_t, 
                    t=t, 
                    aug_state=aug_state,
                    pad_mask=pad_mask, 
                    token_link_ids=link_ids,
                    aug_state_history=aug_history
                )
                forward_end = time.time()
                
                # 4. Loss computation timing
                loss_start = time.time()
                loss = nn.functional.cross_entropy(
                    logits.view(-1, V_total), 
                    x_1.view(-1)).mean()
                loss_end = time.time()
                
                # 5. Backward pass timing
                backward_start = time.time()
                optim.zero_grad()
                loss.backward()
                optim.step()
                backward_end = time.time()
                
                total_time = backward_end - data_start
                epoch_loss += loss.item()
                num_batches += 1
            if epoch % 100 == 0:
                print(epoch_loss / num_batches)
    
        # save the model
        torch.save(model.state_dict(), model_path)

    # Evaluation loop with H5 dataset
    print("Starting evaluation...")
    num_test = 100  # Reduced for testing
    correct_total = 0
    num_samples = 100  # Reduced for testing
    vis_result = True

    # Create test dataset using H5Dataset directly
    test_ds = H5Dataset(
        h5_path, 
        split="test",  # Use test split for testing
        history_len=his_len, 
        history_layout="stack",
        traj_mode=traj_mode,
    )
    val_ds = H5Dataset(
        h5_path,
        split="val",  # Use val split for validation
        history_len=his_len,
        history_layout="stack",
        traj_mode=traj_mode,
    )

    for i in range(min(num_test, len(test_ds))):

        # Get single sample and expand for ensemble
        sample = test_ds[i]
        
        # Convert single sample to batch format for model
        contact_id = sample["contact_ids"].unsqueeze(0).to(device)  # [1, K]
        aug_state = sample["aug_state"].unsqueeze(0).to(device)     # [1, 28]
        aug_history = sample["hist_aug_state"].unsqueeze(0).to(device) if "hist_aug_state" in sample else None  # [1, H, 28]
        contact_nums = sample["contact_nums"].unsqueeze(0).cpu().numpy()  # [1, 7]
        link_ids = sample["link_ids"].unsqueeze(0).to(device)  # [1, K]
        pad_mask = sample["pad_mask"].unsqueeze(0).to(device)  # 

        x_1_0 = contact_id[0]  # [K]
        aug_state_0 = aug_state[0]  # [28]
        aug_history_0 = aug_history[0] if aug_history is not None else None  # [H, 28]
        
        # Generate link IDs
        link_ids_0 = torch.tensor(link_ids[0], device=device, dtype=torch.long)

        # Repeat for ensemble sampling
        x_1 = x_1_0.unsqueeze(0).repeat(num_samples, 1)
        aug_state = aug_state_0.unsqueeze(0).repeat(num_samples, 1)
        link_ids = link_ids_0.unsqueeze(0).repeat(num_samples, 1)
        if aug_history_0 is not None:
            aug_history = aug_history_0.unsqueeze(0).repeat(num_samples, 1, 1)
        
        # Check if sample has valid contacts
        x_1_0_mask = x_1_0 == PAD_ID
        if x_1_0_mask.all():  # Skip if all PAD
            continue
            
        x_1_mask = x_1 == PAD_ID
        n_contacts = len(x_1_0[~x_1_0_mask])
        print(f"Test {i}: Number of contacts: {n_contacts}")
        if n_contacts == 0:
            continue

        # Generate random x_t
        x_t = generate_random_samples_ultra_fast(link_ids, x_1, ss_links_tensor)
        x_t[x_1_mask] = PAD_ID
        pad_mask = x_1_mask

        t = 0.0
        div = 1
        h_c = 0.1 / div
        results_x_t = [x_t.clone()]
        x_t_c_pos = d_loader.retrieve_contact_pos_from_ids(
            x_t[~pad_mask].cpu().numpy().tolist()
        )
        results_c_pos = [x_t_c_pos.copy()]
        results = [(x_t.clone(), t)]

        print("Starting sampling...")
        import time
        st = time.time()
        with torch.no_grad():
            while t < 1.0 - 1e-3:
                t_tensor = torch.full((num_samples,), t, device=device)
                logits = model(
                    x_t=x_t, 
                    t=t_tensor, 
                    aug_state=aug_state,
                    pad_mask=pad_mask, 
                    token_link_ids=link_ids,
                    aug_state_history=aug_history
                )

                # CRITICAL: Mask PAD token logits
                logits[:, :, PAD_ID] = torch.where(
                    pad_mask,
                    logits[:, :, PAD_ID],
                    torch.full_like(logits[:, :, PAD_ID], -float('inf'))
                )
                
                p1 = torch.softmax(logits, dim=-1)
                h = min(h_c, 1.0 - t)
                
                one_hot_x_t = nn.functional.one_hot(x_t, num_classes=V_total).float()
                u = (p1 - one_hot_x_t) / (1.0 - t + 1e-8)
                updated_probs = one_hot_x_t + h * u
                updated_probs = torch.clamp(updated_probs, min=1e-8)
                updated_probs = updated_probs / updated_probs.sum(dim=-1, keepdim=True)
                
                x_t_new = torch.distributions.Categorical(probs=updated_probs).sample()
                x_t = torch.where(pad_mask, PAD_ID, x_t_new)
                
                t += h
                results.append((x_t.clone(), t))
                results_x_t.append(x_t.clone())
                
                x_t_c_pos = d_loader.retrieve_contact_pos_from_ids(
                    x_t[~pad_mask].cpu().numpy().tolist()
                )
                results_c_pos.append(x_t_c_pos)

        et = time.time()
        print(et-st, "sampling time===========")

        # retrieve the contact positions
        init_predict = results_x_t[0]
        final_predict = results_x_t[-1]
        
        # get the feasible ids
        feasible_ids = final_predict[~pad_mask]
        feasible_ids = feasible_ids.cpu().numpy().tolist()
        
        # sample contact positions from feasible_ids
        final_contact_positions = d_loader.retrieve_contact_pos_from_ids(feasible_ids)
        
        # get the initial feasible ids
        initial_feasible_ids = init_predict[~pad_mask]
        initial_feasible_ids = initial_feasible_ids.cpu().numpy().tolist()
        initial_contact_positions = d_loader.retrieve_contact_pos_from_ids(initial_feasible_ids)
        
        # get the ground truth feasible ids, take the unique one
        ground_truth_feasible_ids = x_1[0][~pad_mask[0]]
        ground_truth_feasible_ids = ground_truth_feasible_ids.cpu().numpy().tolist()
        ground_truth_contact_positions = d_loader.retrieve_contact_pos_from_ids(ground_truth_feasible_ids)
        
        # compare the mode and ground truth
        print("Ground Truth Contact Positions:", ground_truth_contact_positions)

        # global mode
        unique_cpos, counts = np.unique(final_contact_positions, axis=0, return_counts=True)
        check_num = 0
        for (cpos, count) in zip(unique_cpos, counts):
            if cpos in ground_truth_contact_positions:
                check_num += 1
            print(f"Predicted Contact Position: {cpos}, Count: {count}")

        if check_num == n_contacts:
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
                        results_c_pos_dis = results_c_pos[i].reshape(-1, n_contacts, 3)[:, j, :]
                        
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
                    
                    if i == 0:  # Add legend to first subplot
                        ax.legend()
            
            plt.suptitle('Evolution of Contact Positions During Flow Matching', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.show()
    print(f"Overall accuracy of mode prediction for testing: {correct_total}/{num_test} = {correct_total/num_test:.4f}")

