import numpy as np
import torch
from torch import Tensor
from pathlib import Path

_link_names_ = ["link1", "link2", "link3", "link4", "link5", "link6", "link7"]
class DataLoader:
    def __init__(self, file_name, limited_dim: int = -1, robot_name: str = "kuka_iiwa_14"):
        data_path = (Path(__file__).resolve().parent / ".." / "dataset_mujoco").as_posix()
        self.data = np.load(f"{data_path}/{file_name}.npz")
        self._prepare_contact_states()
        self._prepare_joint_data(limited_dim)
        self._prepare_mesh_data(robot_name)
        self._prepare_distance_table()
        
    def _prepare_mesh_data(self, robot_name: str = "kuka_iiwa_14"):
        xml_path = (Path(__file__).resolve().parent / ".." / f"{robot_name}/mesh_data").as_posix()
        self.data_dict = np.load(f"{xml_path}/mesh_data_dict.npy", allow_pickle=True).item()
        print("Mesh data loaded from mesh_data.npy")
    
    def _prepare_contact_states(self):
        contacts = self.data["contacts"]
        # currently only prepare for the first contact
        contact_global_ids = contacts[:, :, 0, 13].reshape(-1, )
        self.global_contact_ids = contact_global_ids
        contact_positions = contacts[:, :, 0, :3]
        contact_forces = contacts[:, :, 0, 3:6]
        surface_normals = contacts[:, :, 0, 6:9]
        contact_positions = contact_positions.reshape(-1, 3)
        contact_forces = contact_forces.reshape(-1, 3)
        surface_normals = surface_normals.reshape(-1, 3)

        self.contact_forces = contact_forces
        self.surface_normals = surface_normals
        self.contact_positions = contact_positions

        self.max_contact_id = contact_global_ids.max().item()
        self.min_contact_id = contact_global_ids.min().item()
    
    def _prepare_joint_data(self, limited_dim: int = -1):
        # flatten joint data except for the first dimension
        self.joint_pos_data = self.data["joint_pos"].reshape(-1, 7)
        self.joint_vel_data = self.data["joint_vel"].reshape(-1, 7)
        self.joint_tau_cmd_data = self.data["joint_tau_cmd"].reshape(-1, 7)
        self.joint_tau_ext_gt_data = self.data["joint_tau_ext_gt"].reshape(-1, 7)
    
        # compute the mean and variance of the dataset for normalization in sampling
        self.joint_pos_mean = self.joint_pos_data.mean(axis=0)
        self.joint_pos_std = self.joint_pos_data.std(axis=0)
        self.joint_vel_mean = self.joint_vel_data.mean(axis=0)
        self.joint_vel_std = self.joint_vel_data.std(axis=0)
        self.joint_tau_cmd_mean = self.joint_tau_cmd_data.mean(axis=0)
        self.joint_tau_cmd_std = self.joint_tau_cmd_data.std(axis=0)
        self.joint_tau_ext_gt_mean = self.joint_tau_ext_gt_data.mean(axis=0)
        self.joint_tau_ext_gt_std = self.joint_tau_ext_gt_data.std(axis=0)

    def _prepare_distance_table(self):
        # TODO: rewrite loading the distance table.
        # Get the mesh names 
        mesh_names = ["link_1", "link_2_orange", "link_2_grey", "link_3", "link_4_orange", "link_4_grey", "link_5",
                      "link_6_orange", "link_6_grey", "link_7"]
        
        # Load the table for mesh data
        distance_tables = {}
        dir_name = f"{Path(__file__).resolve().parent}/../kuka_iiwa_14/mesh_geodesic"
        for mesh_name in mesh_names:
            sliced_file_path = f"{dir_name}/distance_{mesh_name}.npz"
            distance_table = np.load(sliced_file_path)["distances_faces_sliced"]
            distance_tables[mesh_name] = distance_table
        self.distance_tables = distance_tables
        
    def get_distance_table(self, mesh_names: str):
        result = list(map(self.distance_tables.get, mesh_names))
        return result
    
    def get_joint_data(self):
        return self.data["joint_pos"], self.data["joint_vel"], self.data["joint_tau_cmd"], self.data["joint_tau_ext_gt"]
    
    def get_contacts(self):
        return self.data["contacts"]
    
    def sample_contact_ids_robot_state(self, batch_size: int) -> Tensor:
        contact_ids = self.global_contact_ids
        # contact_ids_max = contact_ids.max().item()
        # contact_ids_min = contact_ids.min().item()
        # contact_ids = contact_ids - contact_ids_min  # Normalize to start from 0
        
        random_indices = np.random.randint(0, len(contact_ids), batch_size)
        contact_ids = contact_ids[random_indices]
        contact_ids = torch.tensor(contact_ids, dtype=torch.long)
        
        joint_pos = self.joint_pos_data[random_indices]
        joint_vel = self.joint_vel_data[random_indices]
        joint_tau_cmd = self.joint_tau_cmd_data[random_indices]
        joint_tau_ext_gt = self.joint_tau_ext_gt_data[random_indices]
        
        # normalize the joint states with the mean and variance computed from the dataset
        joint_pos = (joint_pos - self.joint_pos_mean) / self.joint_pos_std
        joint_vel = (joint_vel - self.joint_vel_mean) / self.joint_vel_std
        joint_tau_cmd = (joint_tau_cmd - self.joint_tau_cmd_mean) / self.joint_tau_cmd_std
        joint_tau_ext_gt = (joint_tau_ext_gt - self.joint_tau_ext_gt_mean) / self.joint_tau_ext_gt_std
        
        joint_pos = torch.tensor(joint_pos, dtype=torch.float32)
        joint_vel = torch.tensor(joint_vel, dtype=torch.float32)
        joint_tau_cmd = torch.tensor(joint_tau_cmd, dtype=torch.float32)
        joint_tau_ext_gt = torch.tensor(joint_tau_ext_gt, dtype=torch.float32)
        aug_state = torch.cat((joint_pos, joint_vel, joint_tau_cmd, joint_tau_ext_gt), dim=-1)
        
        # corresponding contact positions
        contact_positions = self.contact_positions[random_indices]
        contact_forces = self.contact_forces[random_indices]
        surface_normals = self.surface_normals[random_indices]
        contact_positions = torch.tensor(contact_positions, dtype=torch.float32)
        contact_forces = torch.tensor(contact_forces, dtype=torch.float32)
        surface_normals = torch.tensor(surface_normals, dtype=torch.float32)
        return contact_ids, aug_state, contact_positions
        
    def retreive_nn_neibors(self, contact_ids: np.ndarray, k: int = 20) -> np.ndarray:
        """
        Retrieve the nearest k neighbors for each contact ID.
        """
        contact_ids = self.recover_contact_ids(contact_ids)
        
        local_contact_ids = self.global_ids_to_local_ids(contact_ids)
        link_names, mesh_ids, mesh_names = self.global_ids_to_link_mesh_names(contact_ids)
        
        # Retrieve the distance table for the corresponding mesh
        distance_tables = self.get_distance_table(mesh_names)
        
        # Extract the nearest k contact IDs based on the distance table
        nearest_local_contact_ids = []
        geodesic_distances = []
        for distance_table, local_contact_id in zip(distance_tables, local_contact_ids):
            selected_distance_table = distance_table[local_contact_id]
            nearest_local_contact_id = np.argsort(selected_distance_table)[1:k+1]
            nearest_local_contact_ids.append(nearest_local_contact_id)
            geodesic_distances.append(selected_distance_table[nearest_local_contact_id])
            
        geodesic_distances = np.array(geodesic_distances)
        nearest_local_contact_ids = np.array(nearest_local_contact_ids)
        nearest_local_contact_ids = nearest_local_contact_ids.flatten()
        # Extend the link_names to match the number of nearest_local_contact_ids
        link_names = np.repeat(link_names, k)
        nearest_contact_positions = self.get_data_link_names(link_names, nearest_local_contact_ids)[0]
        
        # Reshape to match the number of contact IDs and k neighbors
        nearest_contact_positions = nearest_contact_positions.reshape(-1, k * 3)        
        return nearest_contact_positions, geodesic_distances

    def recover_contact_ids(self, contact_ids: Tensor) -> Tensor:
        contact_ids_max = self.global_contact_ids.max().item()
        contact_ids_min = self.global_contact_ids.min().item()
        # contact_ids = contact_ids + contact_ids_min  # Recover to original range
        # convert to integers
        return contact_ids
    
    # Helper functions
    def global2link_name(self, global_ids: np.ndarray):
        mapping = self.data_dict["globalid2linkname"]
        return mapping[global_ids]
    
    def global2mesh_id(self, global_ids: np.ndarray):
        mapping = self.data_dict["global_mesh_ids"]
        return mapping[global_ids]
    
    def global2mesh_name(self, global_ids: np.ndarray):
        mapping = self.data_dict["global_mesh_names"]
        return mapping[global_ids]
    
    def retrieve_contact_pos_from_ids(self, contact_ids: np.ndarray) -> np.ndarray:
        return self.get_data(contact_ids)[0]

    def retreive_contact_pos_from_ids_tensor(self, contact_ids: Tensor) -> Tensor:
        contact_ids = contact_ids.cpu().numpy()
        contact_positions = self.retrieve_contact_pos_from_ids(contact_ids)
        return torch.tensor(contact_positions, dtype=torch.float32)

    def retreive_nn_neibors_from_ids_tensor(self, contact_ids: Tensor, k: int = 20) -> Tensor:
        contact_ids = contact_ids.cpu().numpy()
        nearest_contact_positions = self.retreive_nn_neibors(contact_ids, k)
        return torch.tensor(nearest_contact_positions, dtype=torch.float32)

    def retreive_link_ids_from_ids_tensor(self, contact_ids: Tensor) -> Tensor:
        contact_ids = contact_ids.cpu().numpy()
        link_names, mesh_ids, mesh_names = self.global_ids_to_link_mesh_names(contact_ids)
        link_ids = np.array([_link_names_.index(name) for name in link_names])
        return torch.tensor(link_ids, dtype=torch.long)

    def global_ids_to_local_ids(self, global_ids: np.ndarray) -> np.ndarray:
        """
        Convert global contact IDs to local contact IDs.
        """
        mapping = self.data_dict["globalid2localid"]
        local_ids = mapping[global_ids]
        return local_ids

    def global_ids_to_link_mesh_names(self, global_idxs: np.ndarray) -> np.ndarray:
        """
        Convert global contact IDs to local mesh names.
        """
        link_names = self.global2link_name(global_idxs)
        mesh_ids = self.global2mesh_id(global_idxs)
        mesh_names = self.global2mesh_name(global_idxs)
        return link_names, mesh_ids, mesh_names
    
    def get_data(self, global_idxs: np.ndarray) -> tuple:
        """
        Get the data corresponding to the global indices in numpy format.
        """
        face_center_list = self.data_dict["global_face_center_list"][global_idxs]
        normal_list = self.data_dict["global_normal_list"][global_idxs]
        rot_mat_list = self.data_dict["global_rot_mat_list"][global_idxs]
        face_vertices_list = self.data_dict["global_face_vertices_list"][global_idxs]
        geom_ids = self.data_dict["global_geom_ids"][global_idxs]
        link_names = self.global2link_name(global_idxs)
        mesh_ids = np.array([self.data_dict["body_names_mapping"][name]["mesh_id"] for name in link_names])
        mesh_names = np.array([self.data_dict["body_names_mapping"][name]["mesh_name"] for name in link_names])
        return face_center_list, normal_list, rot_mat_list, face_vertices_list, geom_ids, link_names, mesh_ids, mesh_names
    
    def get_data_link_names(self, link_names: list, local_idxes: np.ndarray):
        # TODO: fix it
        link_name_ids = []
        for link_name in link_names:
            link_name_ids.append(_link_names_.index(link_name))
        
        start_end_indices = np.array(self.data_dict["global_start_end_indices"])
        starting_indices = start_end_indices[link_name_ids, 0]
        global_idxs = starting_indices + local_idxes
        face_center_list, normal_list, rot_mat_list, face_vertices_list, geom_ids, \
        link_names, mesh_ids, mesh_names = self.get_data(global_idxs)
        return face_center_list, normal_list, rot_mat_list, face_vertices_list, geom_ids

if __name__ == "__main__":
    file_name = "dataset_batch_1_1000eps"
    d_loader = DataLoader(file_name)
    print(d_loader.max_contact_id, d_loader.min_contact_id)
    batch_size = 1000
    c_ids, aug_state, contact_positions = d_loader.sample_contact_ids_robot_state(batch_size=batch_size)
    print(f"Sampled contact ID min: {c_ids.min().item()}, max: {c_ids.max().item()}")
    print(c_ids.shape, aug_state.shape, contact_positions.shape)
    
    # retrieve nearest neighbors
    nearest_contact_positions, geodesic_distances = d_loader.retreive_nn_neibors(c_ids.numpy(), k=20)
    print(f"Nearest contact positions shape: {nearest_contact_positions.shape}")