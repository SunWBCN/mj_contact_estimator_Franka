# h5_torch_dataset.py
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, get_worker_info
from pathlib import Path
from typing import Tuple, Optional, Dict

_link_names_ = ["link1", "link2", "link3", "link4", "link5", "link6", "link7"]
_mesh_names_ = ["link_1", "link_2_orange", "link_2_grey", "link_3",
                "link_4_orange", "link_4_grey", "link_5",
                "link_6_orange", "link_6_grey", "link_7"]

def _open_h5(h5_path: str) -> h5py.File:
    # swmr=True allows concurrent read-only access across workers
    return h5py.File(h5_path, "r", swmr=True)

class H5Dataset(Dataset):
    """
    PyTorch Dataset that streams one time-step sample from HDF5.

    Returns per __getitem__ (no history):
      contact_ids:         LongTensor [K]
      aug_state:           FloatTensor [28]
      contact_positions:   FloatTensor [3K]  (matches your original shape)
      contact_nums:        IntTensor   [7]

    If history_len > 0, also returns:
      hist_contact_ids:        LongTensor [H*K]  (flattened; set history_layout='stack' to get [H,K])
      hist_aug_state:          FloatTensor [H*28]
      hist_contact_positions:  FloatTensor [H*3K]
      hist_contact_nums:       IntTensor   [H*7]

    Notes:
    - Only contact_nums history is zeroed for pre-episode indices (to match your original code).
    - HDF5 file is opened lazily per worker process for safe multi-worker loading.
    """
    def __init__(
        self,
        h5_path: str,
        split: str = "train",
        train_range: Optional[Tuple[int, int]] = (1000, 1800),
        test_range: Optional[Tuple[int, int]] = (1800, 2000),
        val_range: Optional[Tuple[int, int]] = (3000, 4000),
        traj_mode: bool = False,  # unused for API compatibility
        history_len: int = 0,
        history_layout: str = "flat",  # 'flat' -> [H*K], 'stack' -> [H,K]
        robot_name: str = "kuka_iiwa",
        device: Optional[torch.device] = None,  # unused for API compatibility
    ):
        if robot_name != "kuka_iiwa":
            raise ValueError(f"Unsupported robot: {robot_name}")
        self.device = device

        super().__init__()
        self.h5_path = str(h5_path)
        if not Path(self.h5_path).exists():
            raise FileNotFoundError(f"HDF5 not found: {self.h5_path}")
        # File handle is created lazily (per worker)
        self._h5: Optional[h5py.File] = None

        # Read small attrs/stats once (from a temporary handle)
        with _open_h5(self.h5_path) as f:
            self.T = int(f.attrs["total_steps"])
            if not traj_mode:
                self.ep_len = int(f.attrs["ep_len"]) * 2 # assume that all the episodes has
                                                         # the same length and multiply by 2 to
                                                         # combine the static and dynamic phases
                self.eps = int(f.attrs["eps"]) // 2      # same here
            else:
                self.ep_len = int(f.attrs["ep_len"])
                self.eps = int(f.attrs["eps"])
            self.K = int(f.attrs["max_num_contacts"])
            self.padding_id = int(f.attrs["padding_id"])

            s = f["stats"]
            self._jp_mean = s["joint_pos_mean"][:]
            self._jp_std  = s["joint_pos_std"][:]
            self._jv_mean = s["joint_vel_mean"][:]
            self._jv_std  = s["joint_vel_std"][:]
            self._jc_mean = s["joint_tau_cmd_mean"][:]
            self._jc_std  = s["joint_tau_cmd_std"][:]
            self._je_mean = s["joint_tau_ext_gt_mean"][:]
            self._je_std  = s["joint_tau_ext_gt_std"][:]

        # choose index window
        if traj_mode:
            print("=================== Prepare for trajectory mode")
            if split not in ("train", "test", "val"):
                raise ValueError(f"Invalid split: {split}")
            else:
                if split == "train":
                    self.valid_range = np.arange(*train_range)
                elif split == "test":
                    self.valid_range = np.arange(*test_range)
                else:
                    self.valid_range = np.arange(*val_range)
        else:
            print("=================== Prepare for non-trajectory mode")
            if split not in ("train", "test", "val"):
                raise ValueError(f"Invalid split: {split}")
            else:
                if split == "train" or split == "test":
                    eps_range = [0, int(self.eps*0.8)]
                else:
                    eps_range = [int(self.eps*0.8), self.eps]
                valid_range = []
                dynamic_phase_start = self.ep_len // 2
                for i in range(*eps_range):
                    if split == "train":
                        valid_range.extend(np.arange(i*self.ep_len + dynamic_phase_start, i*self.ep_len + int(dynamic_phase_start*1.8)))
                    elif split == "test":
                        valid_range.extend(np.arange(i*self.ep_len + int(dynamic_phase_start*1.8), (i+1)*self.ep_len))
                    else:
                        valid_range.extend(np.arange(i*self.ep_len + dynamic_phase_start, (i+1)*self.ep_len))
                self.valid_range = np.array(valid_range, dtype=np.int64)
        self.history_len = int(history_len)
        assert history_layout in ("flat", "stack")
        self.history_layout = history_layout

    # --- h5 accessors (per-worker lazy open) ---
    @property
    def f(self) -> h5py.File:
        # reopen in each worker if needed
        if self._h5 is None or not self._h5.id.valid:
            self._h5 = _open_h5(self.h5_path)
        return self._h5

    # Datasets (resolved on first use)
    @property
    def ds_ids(self): return self.f["global_contact_ids"]         # (T,K)
    @property
    def ds_pos(self): return self.f["contact_positions"]          # (T,3K)
    @property
    def ds_cnt(self): return self.f["feasible_contact_num"]       # (T,7)
    @property
    def ds_jp(self):  return self.f["joint_pos"]                  # (T,7)
    @property
    def ds_jv(self):  return self.f["joint_vel"]
    @property
    def ds_jc(self):  return self.f["joint_tau_cmd"]
    @property
    def ds_je(self):  return self.f["joint_tau_ext_gt"]
    @property
    def ds_li(self):  return self.f["link_ids"]                   # (T,K)

    # mappings & distance tables for helper ops
    @property
    def m(self):      return self.f["mappings"]
    @property
    def dgrp(self):   return self.f["distance_tables"]

    def __len__(self) -> int:
        return len(self.valid_range)

    def _abs_index(self, i: int) -> int:
        return self.valid_range[i]

    def _make_aug(self, idx_abs: int) -> np.ndarray:
        jp = (self.ds_jp[idx_abs] - self._jp_mean) / self._jp_std
        jv = (self.ds_jv[idx_abs] - self._jv_mean) / self._jv_std
        jc = (self.ds_jc[idx_abs] - self._jc_mean) / self._jc_std
        je = (self.ds_je[idx_abs] - self._je_mean) / self._je_std
        return np.concatenate([jp, jv, jc, je], axis=-1).astype(np.float32)  # (28,)

    def _history_indices(self, idx_abs: int):
        """
        Returns:
          hist_abs: [H] absolute indices for history (clamped within episode to >=0)
          invalid:  [H] boolean mask where the unclipped local index was < 0
        """
        if self.history_len <= 0:
            return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=bool)

        local = idx_abs % self.ep_len
        ep    = idx_abs // self.ep_len
        raw_hist_local = np.array([local - i for i in range(1, self.history_len + 1)], dtype=np.int64)
        invalid = raw_hist_local < 0
        hist_local = np.clip(raw_hist_local, 0, self.ep_len - 1)
        hist_abs = ep * self.ep_len + hist_local

        # Get unique indices for HDF5 access
        unique_indices, inverse_indices = np.unique(hist_abs, return_inverse=True)
    
        # hist_abs_sorted = np.sort(hist_abs)
        return hist_abs, invalid, unique_indices, inverse_indices

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        # st = time.time()
        idx_abs = self._abs_index(i)

        ids = self.ds_ids[idx_abs]          # (K,)
        pos = self.ds_pos[idx_abs]          # (3K,)
        cnt = self.ds_cnt[idx_abs]          # (7,)
        
        aug = self._make_aug(idx_abs)       # (28,)

        # print(f"{i} - getitem 00: {time.time()-st}")

        sample = {
            "contact_ids":        torch.as_tensor(ids, dtype=torch.long),
            "aug_state":          torch.as_tensor(aug, dtype=torch.float32),
            "contact_positions":  torch.as_tensor(pos, dtype=torch.float32),
            "contact_nums":       torch.as_tensor(cnt, dtype=torch.int32),
        }

        # add pad mask 
        pad_mask = (ids == self.padding_id)
        sample["pad_mask"] = torch.as_tensor(pad_mask, dtype=torch.bool)

        # also update the link_ids
        link_ids = self.ds_li[idx_abs]     # (K,)
        sample["link_ids"] = torch.as_tensor(link_ids, dtype=torch.long)

        if self.history_len > 0:
            hist_abs, invalid, unique_indices, inverse_indices = self._history_indices(idx_abs)
            # print(f"{i} - getitem 01: {time.time()-st}")

            # Access HDF5 with unique indices only
            ids_h_unique = self.ds_ids[unique_indices]         # (n_unique, K)
            pos_h_unique = self.ds_pos[unique_indices]         # (n_unique, 3K)
            cnt_h_unique = self.ds_cnt[unique_indices]         # (n_unique, 7)

            # aug_h_unique = np.stack([self._make_aug(h) for h in unique_indices], axis=0)  # (n_unique, 28)

            jp = (self.ds_jp[unique_indices] - self._jp_mean) / self._jp_std
            jv = (self.ds_jv[unique_indices] - self._jv_mean) / self._jv_std
            jc = (self.ds_jc[unique_indices] - self._jc_mean) / self._jc_std
            je = (self.ds_je[unique_indices] - self._je_mean) / self._je_std
            aug_h_unique = np.concatenate([jp, jv, jc, je], axis=-1).astype(np.float32)  # (28,)

            # print(f"{i} - getitem 02: {time.time()-st}")


            # Expand back to full history using inverse indices
            ids_h = ids_h_unique[inverse_indices]              # (H, K)
            pos_h = pos_h_unique[inverse_indices]              # (H, 3K)
            cnt_h = cnt_h_unique[inverse_indices]              # (H, 7)
            aug_h = aug_h_unique[inverse_indices]    

            # match your original: zero ONLY contact_nums where history was pre-episode
            if invalid.any():
                cnt_h[invalid] = 0

            if self.history_layout == "flat":
                sample.update({
                    "hist_contact_ids":        torch.as_tensor(ids_h.reshape(-1), dtype=torch.long),      # [H*K]
                    "hist_aug_state":          torch.as_tensor(aug_h.reshape(-1), dtype=torch.float32),    # [H*28]
                    "hist_contact_positions":  torch.as_tensor(pos_h.reshape(-1), dtype=torch.float32),    # [H*3K]
                    "hist_contact_nums":       torch.as_tensor(cnt_h.reshape(-1), dtype=torch.int32),      # [H*7]
                })
            else:  # 'stack'
                sample.update({
                    "hist_contact_ids":        torch.as_tensor(ids_h, dtype=torch.long),      # [H,K]
                    "hist_aug_state":          torch.as_tensor(aug_h, dtype=torch.float32),    # [H,28]
                    "hist_contact_positions":  torch.as_tensor(pos_h, dtype=torch.float32),    # [H,3K]
                    "hist_contact_nums":       torch.as_tensor(cnt_h, dtype=torch.int32),      # [H,7]
                })
        
        # print(f"{i} - getitem TOTAL: {time.time()-st}")
        return sample

    # -------- Neighbor/mapping helpers (same interface as your previous code) --------
    def global_ids_to_local_ids(self, global_ids: np.ndarray) -> np.ndarray:
        return self.m["globalid2localid"][global_ids]

    def global2mesh_name(self, global_ids: np.ndarray) -> np.ndarray:
        arr = self.m["global_mesh_names"][global_ids]
        return np.array([x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x) for x in arr])

    def global2link_name(self, global_ids: np.ndarray) -> np.ndarray:
        arr = self.m["globalid2linkname"][global_ids]
        return np.array([x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x) for x in arr])

    def local2global_id(self, mesh_names: np.ndarray, local_idxes: np.ndarray) -> np.ndarray:
        start_end = self.m["global_start_end_indices"][:]  # [n_mesh,2]
        name2id = {n: i for i, n in enumerate(_mesh_names_)}
        mesh_ids = np.vectorize(name2id.get)(mesh_names)
        starts = start_end[mesh_ids, 0]
        return starts + local_idxes

    def retrieve_contact_pos_from_ids(self, contact_ids: np.ndarray) -> np.ndarray:
        return self.m["global_face_center_list"][contact_ids]

    def retreive_nn_neibors(self, contact_ids: np.ndarray, k: int = 20):
        """
        Returns nearest_contact_ids [N,k], geodesic_distances [N,k], same-mesh.
        """
        local_ids = self.global_ids_to_local_ids(contact_ids)
        mesh_names = self.global2mesh_name(contact_ids)

        out_ids, out_d = [], []
        for loc, mn in zip(local_ids, mesh_names):
            row = self.dgrp[mn][loc]            # 1D distances
            idxs = np.argsort(row)[1:k+1]      # skip self
            out_ids.append(self.local2global_id(np.array([mn]*k), idxs))
            out_d.append(row[idxs])
        return np.stack(out_ids, 0), np.stack(out_d, 0)

    def sorted_neibors_source_target(self, source_contact_id: int, target_contact_id: int):
        nn_ids, _ = self.retreive_nn_neibors(np.array([source_contact_id]), k= self.dgrp[self.global2mesh_name(np.array([source_contact_id]))[0]].shape[0]-1)
        nn_ids = nn_ids[0]
        target_local = self.global_ids_to_local_ids(np.array([target_contact_id]))[0]
        mesh_name = self.global2mesh_name(np.array([source_contact_id]))[0]
        dist_row = self.dgrp[mesh_name][target_local]
        nn_local = self.global_ids_to_local_ids(nn_ids)
        dists = dist_row[nn_local]
        order = np.argsort(dists)
        return order, dists[order], nn_ids[order]

    def search_closest_neibor_within_link(self, source_contact_id: int, target_contact_id: int):
        if source_contact_id == target_contact_id:
            return source_contact_id, 0.0, 0
        idxs, dists, ids_sorted = self.sorted_neibors_source_target(source_contact_id, target_contact_id)
        return ids_sorted[0], float(dists[0]), int(idxs[0])

    def search_closest_neibors(self, source_contact_ids: np.ndarray, target_contact_ids: np.ndarray):
        src_links = self.global2link_name(source_contact_ids)
        tgt_links = self.global2link_name(target_contact_ids)
        out_id, out_d, out_i = [], [], []
        for s, t, sl, tl in zip(source_contact_ids, target_contact_ids, src_links, tgt_links):
            if sl == tl:
                cid, d, i = self.search_closest_neibor_within_link(int(s), int(t))
                out_id.append(cid); out_d.append(d); out_i.append(i)
        return np.array(out_id), np.array(out_d), np.array(out_i)

def make_dataloaders(
    h5_path: str,
    batch_size: int = 512,
    history_len: int = 0,
    history_layout: str = "flat",
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    train_range: Tuple[int, int] = (1000, 1800),
    test_range: Tuple[int, int]   = (1800, 2000),
    val_range: Tuple[int, int]    = (3000, 4000),
    robot_name: str = "kuka_iiwa",
    traj_mode: bool = True,  # unused for API compatibility
):
    train_ds = H5Dataset(h5_path, split="train", traj_mode=traj_mode,
                            train_range=train_range, test_range=test_range, val_range=val_range,
                            history_len=history_len, history_layout=history_layout,
                            robot_name=robot_name)
    test_ds   = H5Dataset(h5_path, split="test", traj_mode=traj_mode,
                            train_range=train_range, test_range=test_range, val_range=val_range,
                            history_len=history_len, history_layout=history_layout,
                            robot_name=robot_name)
    val_ds   = H5Dataset(h5_path, split="val", traj_mode=traj_mode,
                            train_range=train_range, test_range=test_range, val_range=val_range,
                            history_len=history_len, history_layout=history_layout,
                            robot_name=robot_name)

    # Default collate works (tensors are 1D per sample; it stacks to [B, ...])
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=(num_workers > 0), drop_last=True,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )
    test_dl = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=(num_workers > 0), drop_last=False
    )
    val_dl = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=(num_workers > 0), drop_last=False
    )
    return train_dl, test_dl, val_dl

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

def generate_random_samples(link_ids, x_1, ss_links):
    random_samples = torch.zeros_like(x_1)
    for i in range(x_1.shape[0]):
        num_c = (link_ids[i] >= 0).sum().item()
        for j in range(num_c):
            link_id = link_ids[i, j].item()
            link_range = ss_links[link_id]
            random_samples[i, j] = torch.randint(low=link_range[0], high=link_range[1], size=(1,))
    return random_samples

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
    # Tiny smoke test
    from pathlib import Path
    num_contacts = 3
    h5 = Path(__file__).parent / "dataset" / f"data-link7-{num_contacts}contact_100_v5" / "dataset_batch_1_200eps.h5"

    max_contact_id = 53642
    PAD_ID = max_contact_id + 1

    import time
    start_time = time.time()
    # train_dl, val_dl = make_dataloaders(h5, batch_size=256, history_len=100, history_layout="stack", num_workers=2)
    train_dl, test_dl, val_dl = make_dataloaders(
        h5, batch_size=20, history_len=32, history_layout="stack", num_workers=8,
        prefetch_factor=4, traj_mode=False
        )
    print(len(train_dl.dataset), len(test_dl.dataset), len(val_dl.dataset))
    print("=================================================================")
    end_time = time.time()
    print(f"Dataloaders created in {end_time - start_time:.4f} seconds")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # get sampling space from h5
    with h5py.File(h5, 'r') as f:
        sampling_space = f["mappings"]["global_start_end_indices"][:]
        ss_links = ss2links(sampling_space, robot_name="kuka_iiwa")
        ss_links_tensor = torch.as_tensor(ss_links, device=device, dtype=torch.long)

    print(f"number of batches in an epoch: {len(train_dl)}")

    for epoch in range(1000):
        st_epoch = time.time()
        st_batch = time.time()
        for b_idx, batch in enumerate(train_dl):
            print(f"{epoch:04d} - {b_idx:03d} loaded in: {time.time() - st_batch}")
            
            # start_time = time.time()
            # contact_id = batch["contact_ids"].to(device)
            # aug_state = batch["aug_state"].to(device)
            # c_pos = batch["contact_positions"].to(device)
            # contact_nums = batch["contact_nums"].to(device)
            # aug_state = batch["aug_state"].to(device)
            # aug_history = batch["hist_aug_state"].to(device) if "hist_aug_state" in batch else None
            # pad_mask = batch["pad_mask"].to(device)
            # link_ids = batch["link_ids"].to(device)
            # end_time = time.time()
            # # print(f"Batch moved to {device} in {end_time - start_time:.4f} seconds")

            # x_1 = contact_id
            # start_time = time.time()
            # x_0 = generate_random_samples_ultra_fast(link_ids, x_1, ss_links_tensor)
            # end_time = time.time()
            # # print(f"Random samples generated in {end_time - start_time:.4f} seconds")

            # start_time = time.time()
            # x_0 = generate_random_samples(link_ids, x_1, ss_links)
            # end_time = time.time()
            # # print(f"Random samples (original) generated in {end_time - start_time:.4f} seconds")

            # time.sleep(1)
            st_batch = time.time()
        
        print(f"\n{epoch:04d} epoch took: {time.time()-st_epoch}")