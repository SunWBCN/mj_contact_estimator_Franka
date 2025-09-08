import numpy as np
import torch
from torch import Tensor
from pathlib import Path
import os

_link_names_ = ["link1", "link2", "link3", "link4", "link5", "link6", "link7"]
_mesh_names_ = ["link_1", "link_2_orange", "link_2_grey", "link_3", "link_4_orange", "link_4_grey", "link_5",
                "link_6_orange", "link_6_grey", "link_7"]
class DataLoader:
    def __init__(self, file_name, dir_name, limited_dim: int = -1, robot_name: str = "kuka_iiwa_14", nn_num: int = 20,
                 max_num_contacts: int = 10):
        data_path = (Path(__file__).resolve().parent / "dataset" / f"{dir_name}").as_posix()
        self.preprocessed_contact_data_path = f"{data_path}/preprocessed_{file_name}.npz"
        self.data = np.load(f"{data_path}/{file_name}.npz")
        self.robot_name = robot_name
        if self.robot_name == "kuka_iiwa_14":
            self.num_joints = 7
            self.max_contact_id = 53642 # 53643 feasible contact positions
        else:
            raise ValueError(f"Number of joints for robot {self.robot_name} is not implemented.")
        self.nn_num = nn_num
        self.max_num_contacts = max_num_contacts
        self._prepare_mesh_data(robot_name)
        self._prepare_contact_states()
        self._prepare_joint_data(limited_dim)
        self._prepare_distance_table()
        
    def _prepare_mesh_data(self, robot_name: str = "kuka_iiwa_14"):
        xml_path = (Path(__file__).resolve().parent / ".." / f"{robot_name}/mesh_data").as_posix()
        self.data_dict = np.load(f"{xml_path}/mesh_data_dict.npy", allow_pickle=True).item()
        print("Mesh data loaded from mesh_data.npy")
    
    def _prepare_contact_states(self):
        if self.preprocessed_contact_data_path is None or not os.path.exists(self.preprocessed_contact_data_path):
            print("=================== No preprocessed data found. Preparing contact states")
            # TODO: implement contact state preparation for the case when number of contacts is variable
            contacts = self.data["contacts"]
            feasible_contact_positions = []
            feasible_contact_forces = []
            feasible_surface_normals = []
            feasible_contact_num = []
            feasible_contact_ids = []
                    
            self.total_steps = contacts.shape[0] * contacts.shape[1]
            self.eps = contacts.shape[0]
            self.ep_len = contacts.shape[1]

            # compute the feasible contact states
            for ep in contacts:
                for step in ep:
                    c_num = np.zeros(self.num_joints, dtype=int)
                    tmp_contact_pos = []
                    tmp_contact_force = []
                    tmp_surface_normal = []
                    tmp_contact_ids = []
                    for c in step:
                        if np.isnan(c[0]):
                            continue
                        else:
                            tmp_contact_pos.extend(c[:3].copy())
                            tmp_contact_force.extend(c[3:6].copy())
                            tmp_surface_normal.extend(c[6:9].copy())
                            contact_id = int(c[13])
                            tmp_contact_ids.extend([contact_id])
                            link_names, _, _, = self.global_ids_to_link_mesh_names(np.array([contact_id]))
                            link_id = _link_names_.index(link_names[0])
                            c_num[link_id] += 1
                    if len(tmp_contact_pos) != 0:
                        tmp_contact_pos = np.array(tmp_contact_pos)
                        tmp_contact_force = np.array(tmp_contact_force)
                        tmp_surface_normal = np.array(tmp_surface_normal)
                        tmp_contact_ids = np.array(tmp_contact_ids)
                        feasible_contact_positions.append(tmp_contact_pos.copy())
                        feasible_contact_forces.append(tmp_contact_force.copy())
                        feasible_surface_normals.append(tmp_surface_normal.copy())
                        feasible_contact_ids.append(tmp_contact_ids.copy())
                    feasible_contact_num.append(c_num)

            # replace all Nan with a padding ID
            padding_id = self.max_contact_id + 1
            contacts = np.nan_to_num(contacts, nan=padding_id)
            # extract contact ids, positions, forces and normals
            contact_global_ids = contacts[:, :, :, 13].astype(int)
            contact_global_positions = contacts[:, :, :, :3]
            contact_global_forces = contacts[:, :, :, 3:6]
            contact_global_normals = contacts[:, :, :, 6:9]
            
            # Add Padding
            # if contact_global_ids's maximum dimension is smaller than self.max_num_contacts
            # extend it with all padding_id, to match the max_num_contacts, the same as contact
            # positions, contact forces and contact normals
            if contact_global_ids.shape[-1] < self.max_num_contacts:
                padding = np.full((contact_global_ids.shape[0], contact_global_ids.shape[1], self.max_num_contacts - contact_global_ids.shape[-1]), padding_id, dtype=int)
                contact_global_ids = np.concatenate((contact_global_ids, padding), axis=2)
                padding = np.full((contact_global_positions.shape[0], contact_global_positions.shape[1], self.max_num_contacts - contact_global_positions.shape[2], 3), padding_id, dtype=float)
                contact_global_positions = np.concatenate((contact_global_positions, padding), axis=2)
                padding = np.full((contact_global_forces.shape[0], contact_global_forces.shape[1], self.max_num_contacts - contact_global_forces.shape[2], 3), padding_id, dtype=float)
                contact_global_forces = np.concatenate((contact_global_forces, padding), axis=2)
                padding = np.full((contact_global_normals.shape[0], contact_global_normals.shape[1], self.max_num_contacts - contact_global_normals.shape[2], 3), padding_id, dtype=float)
                contact_global_normals = np.concatenate((contact_global_normals, padding), axis=2)
            self.global_contact_ids = contact_global_ids.reshape(-1, self.max_num_contacts)
            self.contact_positions = contact_global_positions.reshape(-1, 3 * self.max_num_contacts)
            self.contact_forces = contact_global_forces.reshape(-1, 3 * self.max_num_contacts)
            self.surface_normals = contact_global_normals.reshape(-1, 3 * self.max_num_contacts)

            # define the overall padding for contact ids, contact positions, forces and normals
            self.global_contact_ids_paddings = self.global_contact_ids == padding_id
            self.contact_positions_paddings = self.contact_positions == padding_id
            self.contact_forces_paddings = self.contact_forces == padding_id
            self.surface_normals_paddings = self.surface_normals == padding_id

            feasible_contact_positions = np.array(feasible_contact_positions, dtype=object)
            feasible_contact_forces = np.array(feasible_contact_forces, dtype=object)
            feasible_surface_normals = np.array(feasible_surface_normals, dtype=object)
            feasible_contact_num = np.array(feasible_contact_num, dtype=np.int32)
            feasible_contact_ids = np.array(feasible_contact_ids, dtype=object)

            self.feasible_global_contact_ids = feasible_contact_ids
            self.feasible_global_contact_positions = feasible_contact_positions
            self.feasible_global_contact_forces = feasible_contact_forces
            self.feasible_global_surface_normals = feasible_surface_normals
            self.feasible_global_contact_num = feasible_contact_num
            print("=================== Prepared contact states finished")
            print("=================== Saving the preprocessed data")
            np.savez_compressed(self.preprocessed_contact_data_path,
                                 global_contact_ids=self.global_contact_ids,
                                 contact_positions=self.contact_positions,
                                 contact_forces=self.contact_forces,
                                 surface_normals=self.surface_normals,
                                 feasible_contact_ids=self.feasible_global_contact_ids,
                                 feasible_contact_positions=self.feasible_global_contact_positions,
                                 feasible_contact_forces=self.feasible_global_contact_forces,
                                 feasible_surface_normals=self.feasible_global_surface_normals,
                                 feasible_contact_num=self.feasible_global_contact_num,
                                 total_steps=self.total_steps,
                                 eps=self.eps,
                                 ep_len=self.ep_len
                                 )
            print("=================== Saved the preprocessed data")
        else:
            print("=================== Loading the preprocessed data")
            data = np.load(self.preprocessed_contact_data_path, allow_pickle=True)
            self.global_contact_ids = data["global_contact_ids"]
            self.contact_positions = data["contact_positions"]
            self.contact_forces = data["contact_forces"]
            self.surface_normals = data["surface_normals"]
            self.feasible_global_contact_ids = data["feasible_contact_ids"]
            self.feasible_global_contact_positions = data["feasible_contact_positions"]
            self.feasible_global_contact_forces = data["feasible_contact_forces"]
            self.feasible_global_surface_normals = data["feasible_surface_normals"]
            self.feasible_global_contact_num = data["feasible_contact_num"]
            self.total_steps = int(data["total_steps"])
            self.eps = int(data["eps"])
            self.ep_len = int(data["ep_len"])
            print("=================== Loaded the preprocessed data")
        self.training_steps = [0, self.total_steps // 10 * 9]  # take 9/10 data as training dataset
        self.validation_steps = [self.total_steps // 10 * 9, self.total_steps]  # take 1/10 data as validation dataset
        print(self.training_steps, self.validation_steps)
        print("=================== Prepared training and validation steps finished")

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
        print("=================== Prepared joint data finished")

    def _prepare_distance_table(self):
        if self.robot_name == "kuka_iiwa_14":
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
        else:
            raise ValueError(f"Distance table for robot {self.robot_name} is not implemented.")
        print("=================== Prepared distance table finished")

    def get_distance_table(self, mesh_names: str):
        result = list(map(self.distance_tables.get, mesh_names))
        return result
    
    def get_joint_data(self):
        return self.data["joint_pos"], self.data["joint_vel"], self.data["joint_tau_cmd"], self.data["joint_tau_ext_gt"]
    
    def get_contacts(self):
        return self.data["contacts"]
    
    def get_data_from_dset(self, random_indices) -> Tensor:
        contact_ids = self.global_contact_ids
        # assert len(contact_ids) == self.total_steps, f"Expected {self.total_steps} contact ids, but got {len(contact_ids)}"

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
        contact_nums = self.feasible_global_contact_num[random_indices]
        contact_positions = torch.tensor(contact_positions, dtype=torch.float32)
        contact_forces = torch.tensor(contact_forces, dtype=torch.float32)
        surface_normals = torch.tensor(surface_normals, dtype=torch.float32)
        contact_nums = torch.tensor(contact_nums, dtype=torch.int32)

        return contact_ids, aug_state, contact_positions, contact_nums
        
    def global2local_step(self, global_steps):
        local_steps = global_steps % self.ep_len
        local_eps = global_steps // self.ep_len
        return local_steps, local_eps
        
    def local2global_step(self, local_eps, local_steps):
        global_steps = local_steps + local_eps * self.ep_len
        return global_steps

    # Function only for training high level
    def sample_contact_ids_robot_state(self, batch_size: int) -> Tensor:
        random_indices = np.random.randint(0, self.total_steps, batch_size)
        return self.get_data_from_dset(random_indices)

    # Function only for training high level
    def sample_contact_ids_robot_state_history(self, batch_size: int, history_len: int = 5, data_slice: str = "train") -> Tensor:
        if data_slice == "train":
            global_random_indices = np.random.randint(self.training_steps[0], self.training_steps[1], batch_size)
        elif data_slice == "validate":
            global_random_indices = np.random.randint(self.validation_steps[0], self.validation_steps[1], batch_size)
        else:
            raise ValueError(f"Unknown data_slice: {data_slice}")

        # retrieve the current one
        contact_ids, aug_state, contact_positions, contact_nums = self.get_data_from_dset(global_random_indices)

        # retrieve the data from history
        local_indices, local_eps = self.global2local_step(global_random_indices)

        # compute history indices
        history_indices = [local_indices - i for i in range(1, history_len+1)]
        # record the indices in the history data where indices are equal or smaller than 0
        history_indices_smaller_then_zero = [torch.tensor(indices) < 0 for indices in history_indices]
        history_indices = [torch.clamp(torch.tensor(indices), min=0) for indices in history_indices]
        history_indices = torch.stack(history_indices, dim=1).reshape(-1)
        
        # expand the dimension of local_eps the same as history indices
        local_eps = [torch.tensor(local_eps) for _ in range(history_len)]
        local_eps = torch.stack(local_eps, dim=1).reshape(-1)
        global_history_indices = self.local2global_step(local_eps, history_indices)
        contact_ids_history, aug_state_history, contact_positions_history, contact_nums_history = self.get_data_from_dset(global_history_indices)
        
        # special treatment for history of contact numbers, if the index is equal or smaller than 0 before
        # set the corresponding contact number to all zero
        history_indices_smaller_then_zero = torch.stack(history_indices_smaller_then_zero, dim=1).reshape(-1)
        contact_nums_history[history_indices_smaller_then_zero] = 0
        contact_ids_history = contact_ids_history.reshape(batch_size, -1)
        aug_state_history = aug_state_history.reshape(batch_size, -1)
        contact_positions_history = contact_positions_history.reshape(batch_size, -1)
        contact_nums_history = contact_nums_history.reshape(batch_size, -1)
        return contact_ids, aug_state, contact_positions, contact_nums, \
               contact_ids_history, aug_state_history, contact_positions_history, \
               contact_nums_history
              
    # Function only for low level training 
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
        neighbor_mesh_names = []
        for distance_table, local_contact_id, mesh_name in zip(distance_tables, local_contact_ids, mesh_names):
            selected_distance_table = distance_table[local_contact_id]
            nearest_local_contact_id = np.argsort(selected_distance_table)[1:k+1]
            nearest_local_contact_ids.append(nearest_local_contact_id)
            geodesic_distances.append(selected_distance_table[nearest_local_contact_id])
            neighbor_mesh_names.append(np.repeat(mesh_name, k))
            
        geodesic_distances = np.array(geodesic_distances)
        nearest_local_contact_ids = np.array(nearest_local_contact_ids)
        nearest_local_contact_ids = nearest_local_contact_ids.flatten()
        neighbor_mesh_names = np.array(neighbor_mesh_names).flatten()
        
        nearest_contact_ids = self.local2global_id(neighbor_mesh_names, nearest_local_contact_ids)
        return nearest_contact_ids, geodesic_distances

    def recover_contact_ids(self, contact_ids: Tensor) -> Tensor:
        contact_ids_max = self.global_contact_ids.max().item()
        contact_ids_min = self.global_contact_ids.min().item()
        # contact_ids = contact_ids + contact_ids_min  # Recover to original range
        # convert to integers
        return contact_ids

    def sorted_neibors_source_target(self, source_contact_id: np.ndarray, target_contact_id: np.ndarray) -> tuple:
        nearest_contact_ids, source_geodesic_distances = \
        self.retreive_nn_neibors(source_contact_id, self.nn_num)
        target_local_id = self.global_ids_to_local_ids(np.array([target_contact_id]))[0]
        mesh_name = self.global_ids_to_link_mesh_names([source_contact_id])[0]
        distance_table = self.get_distance_table(mesh_name)
        nearest_local_id = self.global_ids_to_local_ids(nearest_contact_ids)
        
        geodesic_distances = distance_table[target_local_id, nearest_local_id]
        sorted_indexes = np.argsort(geodesic_distances)
        sorted_geodesic_distances = geodesic_distances[sorted_indexes]
        sorted_nearest_contact_ids = nearest_contact_ids[sorted_indexes]
        return sorted_indexes, sorted_geodesic_distances, sorted_nearest_contact_ids

    def search_closest_neibor_within_link(self, source_contact_id: int, target_contact_id: int) -> tuple:
        """
        Search the closest neighbor within the same link.
            Outpout: 
                closest_neibor_id: global id for the closest neibor from source to target
                geodesic_distance: the geodesic distance 
        """
        if source_contact_id == target_contact_id:
            geodesic_distance = 0.0
            closest_neibor_id = source_contact_id
            index = 0
        else:
            sorted_indexes, sorted_geodesic_distances, sorted_nearest_contact_ids = self.sorted_neibors_source_target(source_contact_id, target_contact_id)
            index = sorted_indexes[0]
            geodesic_distance = sorted_geodesic_distances[0]
            closest_neibor_id = sorted_nearest_contact_ids[0]
        return closest_neibor_id, geodesic_distance, index

    def search_closest_neibors(self, source_contact_ids: np.ndarray, target_contact_ids: np.ndarray) -> tuple:
        source_link_names = self.global2link_name(source_contact_ids)
        target_link_names = self.global2link_name(target_contact_ids)
        
        closest_neibor_ids, geodesic_distances, indexes = [], [], []
        for i in range(len(source_contact_ids)):
            source_link_name = source_link_names[i]
            target_link_name = target_link_names[i]
            if source_link_name == target_link_name:
                closest_neibor_id, geodesic_distance, index = self.search_closest_neibor_within_link(source_contact_ids[i], target_contact_ids[i])
                closest_neibor_ids.append(closest_neibor_id)
                geodesic_distances.append(geodesic_distance)
                indexes.append(index)
            else:
                pass
        return np.array(closest_neibor_ids), np.array(geodesic_distances), np.array(indexes)

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

    def retreive_nn_contact_pos_from_ids_tensor(self, contact_ids: Tensor, k: int) -> Tensor:
        contact_ids = contact_ids.cpu().numpy()
        nearest_global_ids, geodesic_distances = \
        self.retreive_nn_neibors(contact_ids, k)
        nearest_contact_positions = self.retrieve_contact_pos_from_ids(nearest_global_ids)
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
        link_names, mesh_ids, mesh_names = \
        self.global_ids_to_link_mesh_names(global_idxs)
        return face_center_list, normal_list, rot_mat_list, face_vertices_list, geom_ids, \
               link_names, mesh_ids, mesh_names
    
    def local2global_id(self, mesh_names: list, local_idxes: np.ndarray):
        mesh_name_ids = []
        for mesh_name in mesh_names:
            mesh_name_ids.append(_mesh_names_.index(mesh_name))
        mesh_name_ids = np.array(mesh_name_ids)
        start_end_indices = np.array(self.data_dict["global_start_end_indices"])
        starting_indices = start_end_indices[mesh_name_ids, 0]
        global_idxs = starting_indices + local_idxes
        return global_idxs

if __name__ == "__main__":
    file_name = "dataset_batch_1_1000eps"
    dir_name = "data-link7-2-contact_v3"
    d_loader = DataLoader(file_name, dir_name)
    batch_size = 1000
    c_ids, aug_state, contact_positions, contact_nums = d_loader.sample_contact_ids_robot_state(batch_size=batch_size)
    print(f"Sampled contact ID min: {c_ids.min().item()}, max: {c_ids.max().item()}")
    print(c_ids.shape, aug_state.shape, contact_positions.shape)
    
    c_ids, aug_state, contact_positions, contact_nums, \
    c_ids_history, aug_state_history, contact_positions_history, contact_nums_history = \
        d_loader.sample_contact_ids_robot_state_history(batch_size=100000, history_len=5)
    
    # # visualize the dataset in 3d
    # padding_id = d_loader.max_contact_id + 1
    # contact_positions_mask = contact_positions == padding_id
    # contact_positions = contact_positions[~contact_positions_mask].reshape(-1, 3)
    # import matplotlib.pyplot as plt
    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(contact_positions[:, 0], contact_positions[:, 1], contact_positions[:, 2])
    # plt.show()

    # visualize the sampling space
    start_end_idx = d_loader.data_dict["global_start_end_indices"]
    print(start_end_idx)
    link_7_idx = start_end_idx[-1]
    x_t = np.random.randint(low=link_7_idx[0], high=link_7_idx[1], size=(100000,))
    c_pos = d_loader.retrieve_contact_pos_from_ids(x_t)

    # visualize contact positions in 3d
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(c_pos[:, 0], c_pos[:, 1], c_pos[:, 2])
    plt.show()

    # # TODO: debug for retrieve nearest neighbors
    # nearest_contact_positions, geodesic_distances = d_loader.retreive_nn_neibors(c_ids.numpy(), k=20)
    # print(f"Nearest contact positions shape: {nearest_contact_positions.shape}")
    # print(geodesic_distances)