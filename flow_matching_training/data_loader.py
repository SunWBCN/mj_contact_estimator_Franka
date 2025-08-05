import numpy as np
import torch
from torch import Tensor

def normalize_contact_ids(contact_ids: Tensor, max_contact_id: int = 1000) -> Tensor:
    contact_ids_max = contact_ids.max().item()
    contact_ids_min = contact_ids.min().item()
    contact_ids = contact_ids - contact_ids_min  # Normalize to start from 0
    contact_ids = contact_ids / contact_ids.max() * max_contact_id  # Scale to [0, max_contact_id]
    return np.round(contact_ids).astype(int)

class DataLoader:
    def __init__(self, file_name, limited_dim: int = -1):
        self.data = np.load(f"../dataset_mujoco/{file_name}.npz")
        self._prepare_global_ids()
        self._prepare_joint_data(limited_dim)
    
    def _prepare_global_ids(self):
        contacts = self.data["contacts"]
        contact_global_ids = contacts[:, :, :, 13]
        global_contact_ids = contact_global_ids[:, 0, 0]  # (N,), assume only one contact per timestep
        self.global_contact_ids = global_contact_ids
    
    def _prepare_joint_data(self, limited_dim: int = -1):
        # flatten joint data except for the first dimension
        self.joint_pos_data = self.data["joint_pos"].reshape(self.data["joint_pos"].shape[0], -1)
        self.joint_vel_data = self.data["joint_vel"].reshape(self.data["joint_vel"].shape[0], -1)
        self.joint_tau_cmd_data = self.data["joint_tau_cmd"].reshape(self.data["joint_tau_cmd"].shape[0], -1)
        self.joint_tau_ext_gt_data = self.data["joint_tau_ext_gt"].reshape(self.data["joint_tau_ext_gt"].shape[0], -1)
    
        if limited_dim > 0:
            num_joints = 7
            self.joint_pos_data = self.joint_pos_data[:, :limited_dim * num_joints]
            self.joint_vel_data = self.joint_vel_data[:, :limited_dim * num_joints]
            self.joint_tau_cmd_data = self.joint_tau_cmd_data[:, :limited_dim * num_joints]
            self.joint_tau_ext_gt_data = self.joint_tau_ext_gt_data[:, :limited_dim * num_joints]
    
    def get_joint_data(self):
        return self.data["joint_pos"], self.data["joint_vel"], self.data["joint_tau_cmd"], self.data["joint_tau_ext_gt"]
    
    def get_contacts(self):
        return self.data["contacts"]
    
    def sample_contact_ids(self, batch_size: int, noise: float = 0.05, max_contact_id: int = 1000) -> Tensor:
        contact_ids = self.global_contact_ids
        contact_ids_max = contact_ids.max().item()
        contact_ids = normalize_contact_ids(contact_ids, max_contact_id)
        if noise > 0:
            noise_tensor = np.random.randn(contact_ids.shape[0]) * noise
            contact_ids = contact_ids / 100 + noise_tensor
            contact_ids = contact_ids * 100
            contact_ids = np.round(contact_ids).astype(int)  # Changed to .astype(int) for compatibility
            contact_ids = np.clip(contact_ids, 0, contact_ids_max)
        random_indices = np.random.randint(0, len(contact_ids), batch_size)
        contact_ids = contact_ids[random_indices]
        contact_ids = torch.tensor(contact_ids, dtype=torch.long)
        return contact_ids
    
    def sample_contact_condition_ids(self, batch_size: int, noise: float = 0.05, max_contact_id: int = 1000) -> Tensor:
        contact_ids = self.global_contact_ids
        contact_ids_max = contact_ids.max().item()
        contact_ids = normalize_contact_ids(contact_ids, max_contact_id)

        # Create pairs of contact IDs
        contact_ids_x_0 = contact_ids[: -1]
        contact_ids_x_1 = contact_ids[1:]
        if noise > 0:
            noise_tensor = np.random.randn(contact_ids_x_0.shape[0]) * noise
            contact_ids_x_0 = contact_ids_x_0 / 100 + noise_tensor
            contact_ids_x_1 = contact_ids_x_1 / 100 + noise_tensor
            contact_ids_x_0 = np.round(contact_ids_x_0).astype(int)
            contact_ids_x_1 = np.round(contact_ids_x_1).astype(int)  # Changed to .astype(int) for compatibility
            contact_ids_x_0 = np.clip(contact_ids_x_0, 0, contact_ids_max)
            contact_ids_x_1 = np.clip(contact_ids_x_1, 0, contact_ids_max)
        random_indices = np.random.randint(0, len(contact_ids_x_0), batch_size)
        contact_ids_x_0 = contact_ids_x_0[random_indices]
        contact_ids_x_1 = contact_ids_x_1[random_indices]
        x_0 = contact_ids_x_0
        x_1 = contact_ids_x_1
        x_0 = torch.tensor(x_0, dtype=torch.long)
        x_1 = torch.tensor(x_1, dtype=torch.long)
        return x_0, x_1
    
    def sample_contact_ids_robot_state(self, batch_size: int, noise: float = 0.05, max_contact_id: int = 1000) -> Tensor:
        contact_ids = self.global_contact_ids
        contact_ids_max = contact_ids.max().item()
        contact_ids = normalize_contact_ids(contact_ids, max_contact_id)
        if noise > 0:
            noise_tensor = np.random.randn(contact_ids.shape[0]) * noise
            contact_ids = contact_ids / 100 + noise_tensor
            contact_ids = contact_ids * 100
            contact_ids = np.round(contact_ids).astype(int)  # Changed to .astype(int) for compatibility
            contact_ids = np.clip(contact_ids, 0, contact_ids_max)
        random_indices = np.random.randint(0, len(contact_ids), batch_size)
        contact_ids = contact_ids[random_indices]
        contact_ids = torch.tensor(contact_ids, dtype=torch.long)
        joint_pos = self.joint_pos_data[random_indices]
        joint_vel = self.joint_vel_data[random_indices]
        joint_tau_cmd = self.joint_tau_cmd_data[random_indices]
        joint_tau_ext_gt = self.joint_tau_ext_gt_data[random_indices]
        joint_pos = torch.tensor(joint_pos, dtype=torch.float32)
        joint_vel = torch.tensor(joint_vel, dtype=torch.float32)
        joint_tau_cmd = torch.tensor(joint_tau_cmd, dtype=torch.float32)
        joint_tau_ext_gt = torch.tensor(joint_tau_ext_gt, dtype=torch.float32)
        aug_state = torch.cat((joint_pos, joint_vel, joint_tau_cmd, joint_tau_ext_gt), dim=-1)
        return contact_ids, aug_state
    
    def sample_contact_ids_robot_state_condition(self, batch_size: int, noise: float = 0.05, max_contact_id: int = 1000) -> Tensor:
        contact_ids = self.global_contact_ids
        contact_ids_max = contact_ids.max().item()
        contact_ids = normalize_contact_ids(contact_ids, max_contact_id)
        # Create pairs of contact IDs
        contact_ids_x_0 = contact_ids[: -1]
        contact_ids_x_1 = contact_ids[1:]
        if noise > 0:
            noise_tensor = np.random.randn(contact_ids_x_0.shape[0]) * noise
            contact_ids_x_0 = contact_ids_x_0 / 100 + noise_tensor
            contact_ids_x_1 = contact_ids_x_1 / 100 + noise_tensor
            contact_ids_x_0 = np.round(contact_ids_x_0).astype(int)
            contact_ids_x_1 = np.round(contact_ids_x_1).astype(int)  # Changed to .astype(int) for compatibility
            contact_ids_x_0 = np.clip(contact_ids_x_0, 0, contact_ids_max)
            contact_ids_x_1 = np.clip(contact_ids_x_1, 0, contact_ids_max)
        random_indices = np.random.randint(0, len(contact_ids_x_0), batch_size)
        contact_ids_x_0 = contact_ids_x_0[random_indices]
        contact_ids_x_1 = contact_ids_x_1[random_indices]
        x_0 = contact_ids_x_0
        x_1 = contact_ids_x_1
        x_0 = torch.tensor(x_0, dtype=torch.long)
        x_1 = torch.tensor(x_1, dtype=torch.long)
        joint_pos = self.joint_pos_data[random_indices]
        joint_vel = self.joint_vel_data[random_indices]
        joint_tau_cmd = self.joint_tau_cmd_data[random_indices]
        joint_tau_ext_gt = self.joint_tau_ext_gt_data[random_indices]
        joint_pos = torch.tensor(joint_pos, dtype=torch.float32)
        joint_vel = torch.tensor(joint_vel, dtype=torch.float32)
        joint_tau_cmd = torch.tensor(joint_tau_cmd, dtype=torch.float32)
        joint_tau_ext_gt = torch.tensor(joint_tau_ext_gt, dtype=torch.float32)
        aug_state = torch.cat((joint_pos, joint_vel, joint_tau_cmd, joint_tau_ext_gt), dim=-1)
        return x_0, x_1, aug_state
        
if __name__ == "__main__":
    file_name = "dataset_1000eps_1_contact"
    d_loader = DataLoader(file_name, 100)
    c_ids, aug_state = d_loader.sample_contact_ids_robot_state(batch_size=10, noise=0.05, max_contact_id=1000)
    c_ids_0, c_ids_1, aug_state_cond = d_loader.sample_contact_ids_robot_state_condition(batch_size=10, noise=0.05, max_contact_id=1000)
    print(c_ids_0.shape, c_ids_1.shape, aug_state.shape)