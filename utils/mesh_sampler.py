import mujoco
import mujoco.viewer
import numpy as np
import trimesh
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import cdist
import os
import jax
from jax import numpy as jnp
jax.config.update("jax_enable_x64", True)

def get_geom_ids_using_mesh(model, mesh_name):
    # Get the mesh ID from the name
    try:
        mesh_id = model.mesh(mesh_name).id
    except KeyError:
        raise ValueError(f"Mesh name '{mesh_name}' not found in model.")

    # Find all geoms using this mesh
    geom_ids = []
    for geom_id in range(model.ngeom):
        if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH:
            if model.geom_dataid[geom_id] == mesh_id:
                geom_ids.append(geom_id)
    return geom_ids

def compute_arrow_rotation(d, pos_center, pos_ref):
    z = d / np.linalg.norm(d)
    y = pos_ref - pos_center
    y = y / np.linalg.norm(y)
    x = np.cross(y, z)
    x = x / np.linalg.norm(x)
    return np.column_stack((x, y, z))  # shape (3, 3)

def compute_equivalent_wrench(wrench, com_point):
    """
    Calculate the equivalent wrench at the center of mass (COM) point. Assume
    the wrench and distance from the CoM to acting point are in the same frame,
    e.g., world frame.
    Args:
        wrench: Wrench vector (6,).
        com_point: Position vector (3,) representing the center of mass point.
    """
    force = wrench[:3]
    torque = wrench[3:]
    
    # Calculate the equivalent wrench at the center of mass
    equivalent_force = force
    equivalent_torque = torque + np.cross(com_point, force)
    return np.concatenate((equivalent_force, equivalent_torque))

def batchwise_nearest(queries, references):
    # queries: (B, D)
    # references: (N, D)
    # returns: indices of closest reference for each query
    # dists = np.linalg.norm(queries[:, None, :] - references[None, :, :], axis=2)  # (B, N)
    dists = cdist(queries, references, 'euclidean')  # (B, N)
    nearest_indices = np.argmin(dists, axis=1)  # (B,)
    return nearest_indices

@jax.jit
def batchwise_nearest_jax(queries, references):
    # queries: (B, D)
    # references: (N, D)
    # returns: indices of closest reference for each query
    dists = jnp.linalg.norm(queries[:, None, :] - references[None, :, :], axis=2)  # (B, N)
    # dists = jax.scipy.spatial.distance.cdist(queries, references, 'euclidean')  # (B, N)
    nearest_indices = jnp.argmin(dists, axis=1)  # (B,)
    return nearest_indices

@jax.jit
def slice_with_indices(array, indices):
    return jax.vmap(lambda i: array[i])(indices)

from typing import List
def assign_indices_to_names(lengths: jnp.ndarray, indices: jnp.ndarray, names: List[str]) -> List[str]:
    # Compute start and end of each segment
    segment_starts = jnp.cumsum(jnp.concatenate([jnp.array([0]), lengths[:-1]]))
    segment_ends = jnp.cumsum(lengths)

    # Determine segment id for each index
    in_segment = (indices[None, :] >= segment_starts[:, None]) & \
                 (indices[None, :] < segment_ends[:, None])
    segment_ids = jnp.argmax(in_segment, axis=0)

    # Map segment ids to names (names is a Python list, use take or list comprehension)
    assigned_names = [names[int(seg_id)] for seg_id in segment_ids]
    return assigned_names

mesh_names_2_body_names = {
    "link_7": "link7",
    "link_6_orange": "link6",
    "link_6_grey": "link6",
    "link_5": "link5",
    "link_4_orange": "link4",
    "link_4_grey": "link4",
    "link_3": "link3",
    "link_2_orange": "link2",
    "link_2_grey": "link2",
    "link_1": "link1"
}   
# _link_names_ = ["link1", "link2", "link3", "link4", "link5", "link6", "link7"]

class MeshSampler:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, init_mesh_data: bool = True, robot_name: str = "kuka_iiwa_14"):
        self.model = model
        self.data = data
        self.mesh_names = [model.mesh(i).name for i in range(model.nmesh)]
        self.geom_names = [model.geom(i).name for i in range(model.ngeom)]
        # self.mesh_id 
        self.robot_name = robot_name
        self._load_feasible_region(robot_name)
        self._load_boundary_region(robot_name)
        if init_mesh_data:
            self._init_mesh_data()
        else:
            xml_path = (Path(__file__).resolve().parent / ".." / f"{robot_name}/mesh_data").as_posix()
            self.data_dict = np.load(f"{xml_path}/mesh_data_dict.npy", allow_pickle=True).item()
            print("Mesh data loaded from mesh_data.npy")
            for mesh_name in self.mesh_names:
                if mesh_name not in self.data_dict:
                    print(f"Mesh {mesh_name} not found in loaded data.")

    def _load_feasible_region(self, robot_name: str = "kuka_iiwa_14"):
        """
        Load feasible region indices for each mesh.
        """
        index_path = (Path(__file__).resolve().parent / ".." / f"{robot_name}/mesh_data").as_posix()
        if robot_name == "kuka_iiwa_14":
            feasible_mesh_names = ["link_1", "link_2_grey", "link_2_orange", "link_3", "link_4_grey",
                    "link_4_orange", "link_5", "link_6_grey", "link_6_orange", "link_7"]
            self.feasible_region_idxes = {}
            for mesh_name in feasible_mesh_names:
                file_name = f"{index_path}/{mesh_name}_valid_face_indices.npy"
                if os.path.exists(file_name):
                    self.feasible_region_idxes[mesh_name] = np.load(f"{index_path}/{mesh_name}_valid_face_indices.npy")
                else:
                    self.feasible_region_idxes[mesh_name] = [0, -1]
        else:
            raise NotImplementedError(f"Robot {robot_name} not implemented for loading feasible regions.")

    def _load_boundary_region(self, robot_name: str = "kuka_iiwa_14"):
        """
        Load boundary region indices for each mesh.
        """
        index_path = (Path(__file__).resolve().parent / ".." / f"{robot_name}/mesh_data").as_posix()
        if robot_name == "kuka_iiwa_14":
            feasible_mesh_names = ["link_1", "link_2_grey", "link_2_orange", "link_3", "link_4_grey",
                    "link_4_orange", "link_5", "link_6_grey", "link_6_orange", "link_7"]
            self.boundary_region_idxes = {}
            for mesh_name in feasible_mesh_names:
                self.boundary_region_idxes[mesh_name] = np.load(f"{index_path}/{mesh_name}_non_manifold_face_indices.npy")
        else:
            raise NotImplementedError(f"Robot {robot_name} not implemented for loading boundary regions.")

    def _skip_mesh(self, mesh_name: str) -> bool:
        """
        Determine whether to skip a mesh based on its name.
        """
        if self.robot_name == "kuka_iiwa_14":
            return mesh_name == "band" or mesh_name == "kuka" or mesh_name == "link_0" or mesh_name == "link_7"
        else:
            raise NotImplementedError(f"Robot {self.robot_name} not implemented for skipping meshes.")

    def _check_body_name(self, body_name: str):
        """
        Check if the body name is valid.
        """
        if self.robot_name == "kuka_iiwa_14":
            valid_body_names = ["link_1", "link_2", "link_3", "link_4", "link_5", "link_6"] # we would not allow to sample from link 7
            return body_name in valid_body_names
        else:
            raise NotImplementedError(f"Robot {self.robot_name} not implemented for checking body names.")

    def _init_mesh_data(self):
        """
        Initialize mesh data for visualization.
        """
        # Get vertex indices
        print("Number of meshes:", self.model.nmesh)
        print("Number of vertices:", len(self.model.mesh_vertadr))

        self.data_dict = {}
        global_geom_ids = []
        global_normal_list = []
        global_face_center_list = []
        global_face_vertices_list = []
        global_rot_mat_list = []
        global_vis_face_list = []
        globalid2linkname = []
        globalid2geomname = []
        globalid2localid = []
        global_num_samples = []
        global_mesh_ids = []
        global_mesh_names = []
        record_mesh_names = []
        
        for i in range(self.model.nmesh):
            geom_ids = []
            normal_list = []
            face_center_list = []
            face_vertices_list = []
            rot_mat_list = []
            vis_face_list = []
            
            v_start = self.model.mesh_vertadr[i]
            v_count = self.model.mesh_vertnum[i]
            vertices = self.model.mesh_vert[v_start : v_start+v_count].reshape(-1, 3)
            
            f_start = self.model.mesh_faceadr[i]
            f_count = self.model.mesh_facenum[i]
            faces = self.model.mesh_face[f_start : f_start+f_count].reshape(-1, 3)
                                                        
            # retrieve the name of the mesh
            mesh_name = self.model.mesh(i).name
            geom_ids = get_geom_ids_using_mesh(self.model, mesh_name)
            geom_names = [self.model.geom(i).name for i in geom_ids]
            mesh_ids = [self.model.geom_dataid[i] for i in geom_ids]
            mesh_names = [self.mesh_names[i] for i in mesh_ids if i != -1]
            
            # preprocessing the valid meshes
            print(mesh_name, geom_ids, geom_names, mesh_ids, mesh_names)
            
            # skip the meshes that are not able to acts as a contact surface
            if self._skip_mesh(mesh_name):
                continue
            
            if mesh_name in self.feasible_region_idxes.keys():
                indexes = self.feasible_region_idxes[mesh_name]
                if indexes[0] == 0 and indexes[1] == -1:
                    faces_ = faces
                else:
                    faces_ = faces[indexes]
                print("================", mesh_name, "================")
                print(f"Number of faces: {len(faces_)}")
                print(f"Number of faces before: {len(faces)}")
                        
            data_dict_mesh_i = {}
            for face_i in range(len(faces)):
                face = faces[face_i]
                face_vertices = vertices[face]
                face_vertices_list.append(face_vertices)
                for i_ in range(3):
                    vis_face_list.append(face_vertices[i_])
                A, B, C = face_vertices[0], face_vertices[1], face_vertices[2]
                face_center = (A + B + C) / 3
                u = B - A   
                v = C - A
                normal = np.cross(u, v)
                normal = normal / np.linalg.norm(normal)
                normal = -normal # by default pointing inwards
                normal_list.append(normal)
                face_center_list.append(face_center)
                rot_mat = compute_arrow_rotation(normal, face_center, A) 
                rot_mat_list.append(rot_mat)
            data_dict_mesh_i["vis_face_list"] = np.array(vis_face_list)
            data_dict_mesh_i["normal_list"] = np.array(normal_list)
            data_dict_mesh_i["face_center_list"] = np.array(face_center_list)
            data_dict_mesh_i["rot_mat_list"] = np.array(rot_mat_list)
            data_dict_mesh_i["geom_ids"] = geom_ids
            data_dict_mesh_i["face_vertices_list"] = np.array(face_vertices_list)
            data_dict_mesh_i["num_faces"] = f_count
            data_dict_mesh_i["num_vertices"] = v_count
            data_dict_mesh_i["geom_names"] = geom_names
            data_dict_mesh_i["geom_ids"] = geom_ids
            data_dict_mesh_i["mesh_ids"] = mesh_ids
            data_dict_mesh_i["mesh_names"] = mesh_names
            data_dict_mesh_i["vis_face_list_jax"] = jnp.array(vis_face_list)
            data_dict_mesh_i["normal_list_jax"] = jnp.array(normal_list)
            data_dict_mesh_i["face_center_list_jax"] = jnp.array(face_center_list)
            data_dict_mesh_i["rot_mat_list_jax"] = jnp.array(rot_mat_list)
            data_dict_mesh_i["face_vertices_list_jax"] = jnp.array(face_vertices_list)
            self.data_dict[mesh_name] = data_dict_mesh_i
            record_mesh_names.append(mesh_name)

            if mesh_name in self.feasible_region_idxes.keys():
                global_geom_ids.extend(geom_ids * len(face_center_list))
                global_mesh_ids.extend(mesh_ids * len(face_center_list)) # Note that all mesh_ids is is 
                                                                         # a list with only one element,
                                                                         # if the robot changes, we need to check
                global_mesh_names.extend(mesh_names * len(face_center_list))
                global_normal_list.extend(normal_list)
                global_face_center_list.extend(face_center_list)
                global_face_vertices_list.extend(face_vertices_list)
                global_rot_mat_list.extend(rot_mat_list)
                global_vis_face_list.extend(vis_face_list)
                
                linkname = mesh_names_2_body_names[mesh_name]
                globalid2linkname.extend([linkname] * len(face_center_list))
                globalid2geomname.extend(geom_names * len(face_center_list))
                globalid2localid.extend(list(range(len(face_center_list))))
                global_num_samples.append(len(face_center_list))
                
        # Convert lists to numpy arrays
        global_geom_ids = np.array(global_geom_ids)
        global_mesh_ids = np.array(global_mesh_ids)
        global_mesh_names = np.array(global_mesh_names)
        global_normal_list = np.array(global_normal_list)
        global_face_center_list = np.array(global_face_center_list)
        global_face_vertices_list = np.array(global_face_vertices_list)
        global_rot_mat_list = np.array(global_rot_mat_list)
        global_vis_face_list = np.array(global_vis_face_list)
        
        # Convert numpy arrays to JAX arrays
        global_geom_ids_jax = jnp.array(global_geom_ids)
        global_mesh_ids_jax = jnp.array(global_mesh_ids)
        global_normal_list_jax = jnp.array(global_normal_list)
        global_face_center_list_jax = jnp.array(global_face_center_list)
        global_face_vertices_list_jax = jnp.array(global_face_vertices_list)
        global_rot_mat_list_jax = jnp.array(global_rot_mat_list)
        global_vis_face_list_jax = jnp.array(global_vis_face_list)
        
        globalid2linkname = np.array(globalid2linkname)
        globalid2geomname = np.array(globalid2geomname)
        globalid2localid = jnp.array(globalid2localid)
        
        global_num_samples = jnp.array(global_num_samples)
        global_starting_indices = jnp.cumsum(global_num_samples)[:-1]
        global_starting_indices = jnp.concatenate((jnp.array([0]), global_starting_indices))
        global_start_end_indices = jnp.stack((global_starting_indices, jnp.cumsum(global_num_samples)), axis=1)
        
        # check the data_dict
        assert len(global_start_end_indices) == len(record_mesh_names), f"Global: {len(global_start_end_indices)} and Recorded {len(record_mesh_names)} Mismatch in mesh names and start-end indices"
        for i in range(len(record_mesh_names)):
            mesh_name_ = record_mesh_names[i]
            self.data_dict[mesh_name_]["global_start_end_indices"] = global_start_end_indices[i]
        
        # Add global arrays to the data_dict
        self.data_dict["global_geom_ids"] = global_geom_ids
        self.data_dict["global_mesh_ids"] = global_mesh_ids
        self.data_dict["global_mesh_names"] = global_mesh_names
        self.data_dict["global_normal_list"] = global_normal_list
        self.data_dict["global_face_center_list"] = global_face_center_list
        self.data_dict["global_face_vertices_list"] = global_face_vertices_list
        self.data_dict["global_rot_mat_list"] = global_rot_mat_list
        self.data_dict["global_vis_face_list"] = global_vis_face_list
        self.data_dict["global_geom_ids_jax"] = global_geom_ids_jax
        self.data_dict["global_mesh_ids_jax"] = global_mesh_ids_jax
        self.data_dict["global_normal_list_jax"] = global_normal_list_jax
        self.data_dict["global_face_center_list_jax"] = global_face_center_list_jax
        self.data_dict["global_face_vertices_list_jax"] = global_face_vertices_list_jax
        self.data_dict["global_rot_mat_list_jax"] = global_rot_mat_list_jax
        self.data_dict["global_vis_face_list_jax"] = global_vis_face_list_jax
        self.data_dict["globalid2linkname"] = globalid2linkname
        self.data_dict["globalid2geomname"] = globalid2geomname
        self.data_dict["globalid2localid"] = globalid2localid
        self.data_dict["global_num_samples"] = global_num_samples
        self.data_dict["global_start_end_indices"] = global_start_end_indices
            
        # Generate a mapping from body names to mesh and geom names
        # TODO: Check the geometric id and mesh id for the whole dataset
        # a single body can include multiple meshes, so we need to account for that
        body_names = [self.model.body(i).name for i in range(self.model.nbody) if "link" in self.model.body(i).name]
        body_names_mapping = {}
        # mesh_names_mapping = {}
        for body_name in body_names:
            body_id = self.model.body(body_name).id
            geom_ids = [i for i in range(self.model.ngeom) if self.model.geom_bodyid[i] == body_id]
            if len(geom_ids) == 0:
                raise ValueError(f"No geoms found for body '{body_name}'")
            mesh_ids = [self.model.geom_dataid[i] for i in geom_ids]
            mesh_names = [self.mesh_names[i] for i in mesh_ids if i != -1]
            if len(mesh_names) == 0:
                raise ValueError(f"No meshes found for body '{body_name}'")    
            # mesh_name = mesh_names
            geom_names = [self.model.geom(i).name for i in geom_ids]
            
            # Remove the kuka, and band meshes
            if body_name == "link3" or body_name == "link5":
                mesh_names = [mesh_names[0]]
                mesh_ids = [mesh_ids[0]]
                geom_ids = [geom_ids[0]]

            body_names_mapping[body_name] = {
                "mesh_name": mesh_names,
                "mesh_id": mesh_ids,
                "geom_id": geom_ids,
            }
        self.data_dict["body_names_mapping"] = body_names_mapping

        xml_path = (Path(__file__).resolve().parent / ".." / f"{self.robot_name}/mesh_data").as_posix()
        file_name = f"{xml_path}/mesh_data_dict.npy"
        np.save(file_name, self.data_dict)
        print(f"Mesh data saved to {file_name}")

    def global2local_id(self, global_ids: jnp.ndarray):
        mapping = self.data_dict["globalid2localid"]
        return slice_with_indices(mapping, global_ids)

    def global2link_name(self, global_ids: jnp.ndarray):
        mapping = self.data_dict["globalid2linkname"]
        return mapping[global_ids]
    
    def global2geom_name(self, global_ids: jnp.ndarray):
        mapping = self.data_dict["globalid2geomname"]
        return mapping[global_ids]
            
    def check_degenerate_faces(self, vertices, faces):
        degenerate_faces = []
        
        for i, face in enumerate(faces):
            # Check for duplicate vertices in face
            if len(set(face)) < 3:
                degenerate_faces.append(i)
                continue
                
            # Check for zero-area triangles
            v0, v1, v2 = vertices[face]
            edge1 = v1 - v0
            edge2 = v2 - v0
            cross = np.cross(edge1, edge2)
            area = np.linalg.norm(cross) / 2
            
            if area < 1e-8:  # Very small area threshold
                degenerate_faces.append(i)
        
        return degenerate_faces

    def visualize_mesh(self, vertices: np.ndarray, faces: np.ndarray, faces_start: int = 10, faces_end: int = -1):
        degenerate_faces = self.check_degenerate_faces(vertices, faces)
        print(f"Degenerate faces: {len(degenerate_faces)}")
        
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        faces = faces[faces_start:faces_end] if faces_end != -1 else faces[faces_start:]
        print(faces.shape)
        for face_i in range(len(faces)):
            face = faces[face_i]
            face_vertices = vertices[face]
            face_mesh = trimesh.Trimesh(vertices=face_vertices, faces=[[0, 1, 2]], process=False)
            face_mesh.visual.vertex_colors = [0, 255, 0, 255]  # Green color
            mesh += face_mesh    
        mesh.show()

    def visualize_mesh_indexes(self, vertices: np.ndarray, faces: np.ndarray, faces_indexes: np.ndarray = None):
        print(vertices.shape, faces.shape, "SHAPE")
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        
        if faces_indexes is None or (len(faces_indexes) == 2):
            faces_indexes = np.arange(len(faces))
        
        for face_i in faces_indexes:
            face = faces[face_i]
            face_vertices = vertices[face]
            face_mesh = trimesh.Trimesh(vertices=face_vertices, faces=[[0, 1, 2]], process=False)
            face_mesh.visual.vertex_colors = [0, 255, 0, 255]
            mesh += face_mesh
        mesh.show()

    def visualize_pos_sphere(self, mesh_id: int, geoms_pos_local: np.ndarray = None):
        radius = 0.004
        rgba = np.array([1.0, 0.0, 0.0, 1.0])
        mesh_name_i = self.mesh_names[mesh_id]
    
        viewer = mujoco.viewer.launch_passive(self.model, self.data)
        scene = viewer.user_scn
        ngeom_init = scene.ngeom
    
        def reset_scene(scene):
            scene.ngeom = ngeom_init
        
        if geoms_pos_local is None:
            vis_face_list = self.data_dict[mesh_name_i]["vis_face_list"]
            geoms_pos_local = [vis_face_list[0]]
        
        geom_ids = self.data_dict[mesh_name_i]["geom_ids"]
        geom_id = geom_ids[0]

        def update_scene(scene, geom_origin_pos_world, geom_origin_mat_world, geom_pos_local):
            geom_pos_world = geom_origin_pos_world + geom_origin_mat_world @ geom_pos_local

            scene.ngeom += 1  # increment ngeom
            # Create a new geom as a sphere
            mujoco.mjv_initGeom(scene.geoms[scene.ngeom - 1],
                                mujoco.mjtGeom.mjGEOM_SPHERE, size=np.array([radius, 0., 0.]),
                                pos=geom_pos_world, mat=geom_origin_mat_world.flatten(), rgba=rgba.astype(np.float32))
        while viewer.is_running():
            mujoco.mj_forward(self.model, self.data)
            geom_origin_pos_world = self.data.geom_xpos[geom_id]        # shape (3,)
            geom_origin_mat_world = self.data.geom_xmat[geom_id].reshape(3, 3)  # shape (3,3)
            
            mujoco.mj_step(self.model, self.data)
            reset_scene(scene)
            for i in range(len(geoms_pos_local)):
                geom_pos_local = geoms_pos_local[i]
                update_scene(scene, geom_origin_pos_world, geom_origin_mat_world, geom_pos_local)
            viewer.sync()
    
    def visualize_normal_arrow(self, mesh_id: int, geom_id: int, faces_center_local: np.ndarray = None, arrows_rot_mat_local: np.ndarray = None):
        viewer = mujoco.viewer.launch_passive(self.model, self.data)
        scene = viewer.user_scn
        ngeom_init = scene.ngeom
        rgba = np.array([1.0, 0.0, 0.0, 1.0])
        mesh_name_i = self.mesh_names[mesh_id]
        # geom_ids = self.data_dict[mesh_name_i]["geom_ids"]
        # geom_id = geom_ids[0] # TODO: handle multiple geoms, note that meshes "band", "kuka" are used by link3 and link5
        
        if faces_center_local is None or arrows_rot_mat_local is None:
            rot_mat_list = self.data_dict[mesh_name_i]["rot_mat_list"]
            face_center_list = self.data_dict[mesh_name_i]["face_center_list"]
            faces_center_local = [face_center_list[0]]
            arrows_rot_mat_local = [rot_mat_list[0]]
        
        def reset_scene(scene):
            scene.ngeom = ngeom_init
            
        def update_scene(scene, geom_origin_pos_world, geom_origin_mat_world, face_center_local, arrow_rot_mat_local):            
            # Create a new geom as an arrow
            arrow_length = 0.2
            arrow_radius = 0.01 * arrow_length
            arrow_half_length = 0.5 * arrow_length
            size_array = np.array([arrow_radius, arrow_radius, arrow_half_length],
                                dtype=np.float32)
            arrow_mat_world = geom_origin_mat_world @ arrow_rot_mat_local
            
            face_center_world = geom_origin_pos_world + geom_origin_mat_world @ face_center_local
            
            scene.ngeom += 1  # increment ngeom
            # Compute the rotation matrix that is orthogonal to the rotation matrix
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom - 1],
                mujoco.mjtGeom.mjGEOM_ARROW,
                size=size_array,
                pos=face_center_world,
                mat=arrow_mat_world.flatten(),
                rgba=rgba.astype(np.float32),
            )
                    
        while viewer.is_running():
            # TODO: visualize the foces and normals in mujoco    
            mujoco.mj_forward(self.model, self.data)
            geom_origin_pos_world = self.data.geom_xpos[geom_id]        # shape (3,)
            geom_origin_mat_world = self.data.geom_xmat[geom_id].reshape(3, 3)  # shape (3,3)
            
            mujoco.mj_step(self.model, self.data)
            reset_scene(scene)
            for i in range(len(faces_center_local)):
                face_center_local = faces_center_local[i]
                arrow_rot_mat_local = arrows_rot_mat_local[i]
                update_scene(scene, geom_origin_pos_world, geom_origin_mat_world, face_center_local, arrow_rot_mat_local)
            viewer.sync()
    
    def sample_pos_normal(self, mesh_name: str, num_samples: int = 3):
        """
        Sample a position and normal vector from the mesh.
        
        Args:
            mesh_name: Name of the mesh to sample from.
        
        Returns:
            pos: Sampled position (3D vector).
            normal: Sampled normal vector (3D vector).
        """
        mesh_data = self.data_dict[mesh_name]
        face_center_list = mesh_data["face_center_list"]
        normal_list = mesh_data["normal_list"]
        rot_mat_list = mesh_data["rot_mat_list"]
        face_vertices_list = mesh_data["face_vertices_list"]
        
        # TODO: could use a faster sampling method, right now is around 0.0002s
        idxs = np.random.choice(len(face_center_list), num_samples, replace=False)
        # idxs = np.random.permutation(len(face_center_list))[:num_samples]
        face_center_select = face_center_list[idxs]
        normal_select = normal_list[idxs]
        rot_mat_select = rot_mat_list[idxs]
        face_vertices_select = face_vertices_list[idxs]
        return face_center_select, normal_select, rot_mat_select, face_vertices_select

    def sample_pos_normal_jax(self, mesh_name: str, num_samples: int = 3, key: jax.random.PRNGKey = jax.random.PRNGKey(0)):
        """
        Sample a position and normal vector from the mesh using JAX.
        
        Args:
            mesh_name: Name of the mesh to sample from.
        
        Returns:
            pos: Sampled position (3D vector).
            normal: Sampled normal vector (3D vector).
        """
        mesh_data = self.data_dict[mesh_name]
        face_center_list = mesh_data["face_center_list_jax"]
        normal_list = mesh_data["normal_list_jax"]
        rot_mat_list = mesh_data["rot_mat_list_jax"]
        face_vertices_list = mesh_data["face_vertices_list_jax"]

        # Sample indices
        idxs = jax.random.choice(key, len(face_center_list), shape=(num_samples,), replace=False)
        
        # Select sampled data
        face_center_select = face_center_list[idxs]
        normal_select = normal_list[idxs]
        rot_mat_select = rot_mat_list[idxs]
        face_vertices_select = face_vertices_list[idxs]
        
        return face_center_select, normal_select, rot_mat_select, face_vertices_select
    
    def sample_body_pos_normal_jax(self, body_name: str, num_samples: int = 3, key: jax.random.PRNGKey = jax.random.PRNGKey(0)):
        """
        Sample a position and normal vector from the body using JAX.
        
        Args:
            body_name: Name of the body to sample from.
        
        Returns:
            pos: Sampled position (3D vector).
            normal: Sampled normal vector (3D vector).
        """
        # TODO: add random indexing mechanism to deal with multiple meshes on the same body.
        n_meshes = len(self.data_dict["body_names_mapping"][body_name]["mesh_name"])
        if n_meshes > 1:
            idx = np.random.randint(0, n_meshes)
        else:
            idx = 0
        mesh_name = self.data_dict["body_names_mapping"][body_name]["mesh_name"][idx]
        mesh_id = self.data_dict["body_names_mapping"][body_name]["mesh_id"][idx]
        geom_id = self.data_dict["body_names_mapping"][body_name]["geom_id"][idx]        
        
        face_center_select, normal_select, rot_mat_select, face_vertices_select = self.sample_pos_normal_jax(mesh_name, num_samples, key)
        return mesh_id, geom_id, face_center_select, normal_select, rot_mat_select, face_vertices_select

    def update_sampling_space_global(self, body_names: list):
        """
        Update the sampling space with the global indices of the face centers
        """
        feasible_idxes = self.compute_feasible_idxes(body_names)
        self.feasible_idxes = feasible_idxes

    def compute_feasible_idxes(self, body_names: list):
        # find the corresponding mesh names
        mesh_names = []
        for body_name in body_names:
            if body_name not in self.data_dict["body_names_mapping"]:
                raise ValueError(f"Body name '{body_name}' not found in the data dictionary.")
            mesh_names.extend(self.data_dict["body_names_mapping"][body_name]["mesh_name"])
    
        feasible_idxes = []
        for mesh_name in mesh_names:
            # TODO: retreive mesh ids corresponding to the body_id
            start = self.data_dict[mesh_name]["global_start_end_indices"][0]
            end = self.data_dict[mesh_name]["global_start_end_indices"][1]
            feasible_idxes.extend(jnp.arange(start, end).tolist())
        feasible_idxes = jnp.array(feasible_idxes)
        return feasible_idxes

    def sample_indexes_global(self, sample_body_names: None, num_samples: int = 3, key: jax.random.PRNGKey = jax.random.PRNGKey(0)):
        """
        Sample indices from the global search space.
        
        Args:
            sample_body_names: List of body names to sample from.
            num_samples: Number of samples to draw.
        
        Returns:
            idxs: Sampled indices.
        """
        if sample_body_names is not None:
            feasible_idxes = self.feasible_idxes
        else:
            # if no specific body names are provided, sample from the entire search space
            feasible_idxes = jnp.arange(self.data_dict["global_face_center_list_jax"].shape[0])
        idxes = jax.random.choice(key, feasible_idxes, shape=(num_samples,), replace=False)
        return idxes
    
    def sample_indexes_global_from_body(self, body_name: str, num_samples: int = 3, key: jax.random.PRNGKey = jax.random.PRNGKey(0)):
        body_names = [body_name]
        feasible_idxes = self.compute_feasible_idxes(body_names)
        idxes = jax.random.choice(key, feasible_idxes, shape=(num_samples,), replace=False)
        return idxes
    
    def get_data(self, global_idxs: jnp.ndarray):
        """
        Get the data corresponding to the global indices.
        """
        face_center_list = slice_with_indices(self.data_dict["global_face_center_list_jax"], global_idxs)
        normal_list = slice_with_indices(self.data_dict["global_normal_list_jax"], global_idxs)
        rot_mat_list = slice_with_indices(self.data_dict["global_rot_mat_list_jax"], global_idxs)
        face_vertices_list = slice_with_indices(self.data_dict["global_face_vertices_list_jax"], global_idxs)
        geom_ids = slice_with_indices(self.data_dict["global_geom_ids"], global_idxs)
        link_names = self.global2link_name(global_idxs)
        
        # TODO: use the global list to get mesh ids instead of the mapping! Note that link2, link4 and link6 has two meshes
        # mesh_ids = jnp.array([self.data_dict["body_names_mapping"][name]["mesh_id"] for name in link_names])
        mesh_ids = slice_with_indices(self.data_dict["global_mesh_ids"], global_idxs)
        return face_center_list, normal_list, rot_mat_list, face_vertices_list, geom_ids, link_names, mesh_ids
        
    def get_data_numpy(self, global_idxs: np.ndarray):
        """
        Get the data corresponding to the global indices in numpy format.
        """
        face_center_list = self.data_dict["global_face_center_list"][global_idxs]
        normal_list = self.data_dict["global_normal_list"][global_idxs]
        rot_mat_list = self.data_dict["global_rot_mat_list"][global_idxs]
        face_vertices_list = self.data_dict["global_face_vertices_list"][global_idxs]
        geom_ids = self.data_dict["global_geom_ids"][global_idxs]
        link_names = self.global2link_name(global_idxs)
        
        # TODO: use the global list to get mesh ids instead of the mapping! Note that link2, link4 and link6 has two meshes
        # mesh_ids = np.array([self.data_dict["body_names_mapping"][name]["mesh_id"] for name in link_names])
        mesh_ids = self.data_dict["global_mesh_ids"][global_idxs]
        return face_center_list, normal_list, rot_mat_list, face_vertices_list, geom_ids, link_names, mesh_ids

    def sample_indexes(self, num_samples: int = 3, key: jax.random.PRNGKey = jax.random.PRNGKey(0)):
        """
        Sample indices from the search space.
        
        Args:
            num_samples: Number of samples to draw.
        
        Returns:
            idxs: Sampled indices.
        """
        face_center_list = self.data_dict["search_space"]["face_center_list"]
        return jax.random.choice(key, len(face_center_list), shape=(num_samples,), replace=False)

    def sample_body_pos_normal(self, body_name: str, num_samples: int = 3):
        """
        Sample a position and normal vector from the body.
        
        Args:
            body_name: Name of the body to sample from.
        
        Returns:
            pos: Sampled position (3D vector).
            normal: Sampled normal vector (3D vector).
        """
        # check if the body_name is valid
        if not self._check_body_name(body_name):
            raise ValueError(f"Invalid body name: {body_name}")
        n_mesh = len(self.data_dict["body_names_mapping"][body_name]["mesh_name"])
        if n_mesh > 1:
            idx = np.random.randint(0, n_mesh)
        else:
            idx = 0
        mesh_name = self.data_dict["body_names_mapping"][body_name]["mesh_name"][idx]
        mesh_id = self.data_dict["body_names_mapping"][body_name]["mesh_id"][idx]
        geom_id = self.data_dict["body_names_mapping"][body_name]["geom_id"][idx]
        face_center_select, normal_select, rot_mat_select, face_vertices_select = self.sample_pos_normal(mesh_name, num_samples)
        return mesh_id, geom_id, face_center_select, normal_select, rot_mat_select, face_vertices_select
        
    def compute_equivalent_wrenches_multibody(self, contact_poss_geom: list, rots_mat_contact_geom: list, normal_vecs_geom: list, sample_body_names: list, geom_ids: list, ext_f_norm: float):
        model = self.model
        data = self.data
        equi_ext_fs = []
        equi_ext_f_poss = []
        equi_ext_wrenchs = []
        rot_mats_contact_world = []
        contact_poss_world = []
        contact_poss_com = []
        rot_mats_contact_com = []
        jacobian_contacts = []
        jacobian_bodies = []
        ext_wrenches = []
        for i in range(len(rots_mat_contact_geom)):
            geom_id = geom_ids[i]
            sample_body_name = sample_body_names[i]
            geom_pos_world = data.geom_xpos[geom_id]        # shape (3,)
            rot_mat_geom_world = data.geom_xmat[geom_id].reshape(3, 3)    
            com_pos_world = data.xpos[model.body(sample_body_name).id]
            rot_mat_com_world = data.xmat[model.body(sample_body_name).id].reshape(3, 3)
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{sample_body_name}_dummy_site1") # TODO: change the hardcode case to fit the number of contact points
            rot_mat_contact_geom = rots_mat_contact_geom[i]
            rot_mat_contact_world = rot_mat_geom_world @ rot_mat_contact_geom
            contact_pos_geom = contact_poss_geom[i]
            contact_pos_world = geom_pos_world + rot_mat_geom_world @ contact_pos_geom
            contact_pos_com = rot_mat_com_world.T @ (contact_pos_world - com_pos_world)
            rot_mat_contact_com = rot_mat_com_world.T @ rot_mat_contact_world

            # Retrieve the normal vector
            normal_vec_geom = normal_vecs_geom[i]
            normal_vec_world = rot_mat_geom_world @ normal_vec_geom

            # Applied force
            ext_f_geom = normal_vec_geom * ext_f_norm
            ext_f = normal_vec_world * ext_f_norm
            ext_wrench = np.concatenate((ext_f, np.zeros(3)))
            com_to_contact_world = contact_pos_world - com_pos_world
            equi_ext_wrench = compute_equivalent_wrench(ext_wrench, com_to_contact_world)
            ext_wrenches.append(ext_wrench)
            
            # Append the data.
            equi_ext_fs.append(ext_f)
            equi_ext_f_poss.append(com_pos_world)
            equi_ext_wrenchs.append(equi_ext_wrench)
            rot_mats_contact_world.append(rot_mat_contact_world)
            contact_poss_world.append(contact_pos_world)
            contact_poss_com.append(contact_pos_com)
            rot_mats_contact_com.append(rot_mat_contact_com)
            
            # Update the site position and quaternion for computing jacobian
            model.site_pos[site_id] = contact_pos_com
            rot = R.from_matrix(rot_mat_contact_com)
            quat_mujoco = rot.as_quat(scalar_first=True)
            model.site_quat[site_id] = quat_mujoco
            mujoco.mj_forward(model, data) # update kinematics
            jac_site_contact = np.zeros((6, model.nv))
            mujoco.mj_jacSite(model, data, jac_site_contact[:3], jac_site_contact[3:], site_id)
            jacobian_contacts.append(jac_site_contact)    
            jac_body = np.zeros((6, model.nv))
            mujoco.mj_jacBody(model, data, jac_body[:3], jac_body[3:], model.body(f"{sample_body_name}").id)
            jacobian_bodies.append(jac_body)
            
        return equi_ext_f_poss, equi_ext_wrenchs, jacobian_bodies, contact_poss_world, ext_wrenches, jacobian_contacts
        
    def compute_equivalent_wrenches(self, contact_poss_geom: list, rots_mat_contact_geom: list, normal_vecs_geom: list, sample_body_name: str, geom_id: int, ext_f_norm: float):
        model = self.model
        data = self.data
        geom_pos_world = data.geom_xpos[geom_id]        # shape (3,)
        rot_mat_geom_world = data.geom_xmat[geom_id].reshape(3, 3)    
        com_pos_world = data.xpos[model.body(sample_body_name).id]
        rot_mat_com_world = data.xmat[model.body(sample_body_name).id].reshape(3, 3)
        equi_ext_fs = []
        equi_ext_f_poss = []
        equi_ext_wrenchs = []
        rot_mats_contact_world = []
        contact_poss_world = []
        contact_poss_com = []
        rot_mats_contact_com = []
        jacobian_contacts = []
        jacobian_bodies = []
        ext_wrenches = []
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{sample_body_name}_dummy_site1") # TODO: change the hardcode case to fit the number of contact points
        for i in range(len(rots_mat_contact_geom)):
            rot_mat_contact_geom = rots_mat_contact_geom[i]
            rot_mat_contact_world = rot_mat_geom_world @ rot_mat_contact_geom
            contact_pos_geom = contact_poss_geom[i]
            contact_pos_world = geom_pos_world + rot_mat_geom_world @ contact_pos_geom
            contact_pos_com = rot_mat_com_world.T @ (contact_pos_world - com_pos_world)
            rot_mat_contact_com = rot_mat_com_world.T @ rot_mat_contact_world

            # Retrieve the normal vector
            normal_vec_geom = normal_vecs_geom[i]
            normal_vec_world = rot_mat_geom_world @ normal_vec_geom

            # Applied force
            ext_f_geom = normal_vec_geom * ext_f_norm
            ext_f = normal_vec_world * ext_f_norm
            ext_wrench = np.concatenate((ext_f, np.zeros(3)))
            com_to_contact_world = contact_pos_world - com_pos_world
            equi_ext_wrench = compute_equivalent_wrench(ext_wrench, com_to_contact_world)
            ext_wrenches.append(ext_wrench)
            
            # Append the data.
            equi_ext_fs.append(ext_f)
            equi_ext_f_poss.append(com_pos_world)
            equi_ext_wrenchs.append(equi_ext_wrench)
            rot_mats_contact_world.append(rot_mat_contact_world)
            contact_poss_world.append(contact_pos_world)
            contact_poss_com.append(contact_pos_com)
            rot_mats_contact_com.append(rot_mat_contact_com)
            
            # Update the site position and quaternion for computing jacobian
            model.site_pos[site_id] = contact_pos_com
            rot = R.from_matrix(rot_mat_contact_com)
            quat_mujoco = rot.as_quat(scalar_first=True)
            model.site_quat[site_id] = quat_mujoco
            mujoco.mj_forward(model, data) # update kinematics
            jac_site_contact = np.zeros((6, model.nv))
            mujoco.mj_jacSite(model, data, jac_site_contact[:3], jac_site_contact[3:], site_id)
            jacobian_contacts.append(jac_site_contact)    
            jac_body = np.zeros((6, model.nv))
            mujoco.mj_jacBody(model, data, jac_body[:3], jac_body[3:], model.body(f"{sample_body_name}").id)
            jacobian_bodies.append(jac_body)            
        return equi_ext_f_poss, equi_ext_wrenchs, jacobian_bodies, contact_poss_world, ext_wrenches, jacobian_contacts
    
    def update_model_data(self, model, data):
        """
        Update the model and data with the current mesh data.
        This is useful when the model or data has been modified.
        """
        self.model = model
        self.data = data
        
    def find_nearest_indexes(self, positions: jnp.ndarray, body_names: list):
        """
        Find the nearest indexes of the positions in the search space.
        
        Args:
            positions: Positions to find the nearest indexes for.
            body_names: List of body names to sample from.
        
        Returns:
            nearest_indexes: Nearest indexes of the positions in the search space.
        """
        feasible_idxes = self.feasible_idxes
        if body_names is not None:
            faces_search_space = slice_with_indices(self.data_dict["global_face_center_list_jax"], feasible_idxes)
        else:
            faces_search_space = self.data_dict["global_face_center_list_jax"]
        indices = batchwise_nearest_jax(positions, faces_search_space) # note that here the indices are relative to the search space
        
        # convert the indices to global indices
        nearest_indexes = feasible_idxes[indices]
        return nearest_indexes
        
from collections import defaultdict
"""
Code refer to 
https://stackoverflow.com/questions/76435070/how-do-i-use-python-trimesh-to-get-boundary-vertex-indices
"""
def boundary(mesh, close_paths=True):   
    # Set of all edges and of boundary edges (those that appear only once).
    edge_set = set()
    boundary_edges = set()
    edge_to_faces = defaultdict(list)  # Maps edge to faces that contain it

    # Build edge-to-face mapping
    for face_idx, face in enumerate(mesh.faces):
        # Get edges of this face
        edges = [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])]
        for edge in edges:
            # Sort edge to ensure consistent ordering
            sorted_edge = tuple(sorted(edge))
            edge_to_faces[sorted_edge].append(face_idx)

    # Iterate over all edges, as tuples in the form (i, j) (sorted with i < j to remove ambiguities).
    for e in map(tuple, mesh.edges_sorted):
        if e not in edge_set:
            edge_set.add(e)
            boundary_edges.add(e)
        elif e in boundary_edges:
            boundary_edges.remove(e)
        else:
            raise RuntimeError(f"The mesh is not a manifold: edge {e} appears more than twice.")

    # Given all boundary vertices, we create a simple dictionary that tells who are their neighbours.
    neighbours = defaultdict(lambda: [])
    for v1, v2 in boundary_edges:
        neighbours[v1].append(v2)
        neighbours[v2].append(v1)

    # We now look for all boundary paths by "extracting" one loop at a time.
    boundary_paths = []
    boundary_indices = []
    boundary_face_indices = []

    while len(boundary_edges) > 0:
        # Given the set of remaining boundary edges, get one of them and use it to start the current boundary path.
        v_previous, v_current = next(iter(boundary_edges))
        boundary_vertices = [v_previous]
        boundary_faces = []

        # Keep iterating until we close the current boundary curve
        while v_current != boundary_vertices[0]:
            # We grow the path by adding the vertex "v_current".
            boundary_vertices.append(v_current)

            # Find faces adjacent to this boundary edge
            edge = tuple(sorted([v_previous, v_current]))
            adjacent_faces = edge_to_faces.get(edge, [])
            boundary_faces.extend(adjacent_faces)

            # We now check which is the next vertex to visit.
            v1, v2 = neighbours[v_current]
            if v1 != v_previous:
                v_current, v_previous = v1, v_current
            elif v2 != v_previous:
                v_current, v_previous = v2, v_current
            else:
                raise RuntimeError(f"Next vertices to visit ({v1=}, {v2=}) are both equal to {v_previous=}.")

        # Handle the closing edge
        if len(boundary_vertices) > 1:
            edge = tuple(sorted([boundary_vertices[-1], boundary_vertices[0]]))
            adjacent_faces = edge_to_faces.get(edge, [])
            boundary_faces.extend(adjacent_faces)

        # Close the path (by repeating the first vertex) if needed.
        if close_paths:
            boundary_vertices.append(boundary_vertices[0])

        # Store the vertex indices and face indices before converting to coordinates
        boundary_indices.append(boundary_vertices.copy())
        boundary_face_indices.append(list(set(boundary_faces)))  # Remove duplicates

        # "Convert" the vertices from indices to actual Cartesian coordinates.
        boundary_paths.append(mesh.vertices[boundary_vertices])

        # Remove all boundary edges that were added to the last path.
        boundary_edges = set(e for e in boundary_edges if e[0] not in boundary_vertices)

    # Return the list of boundary paths, vertex indices, and face indices.
    return boundary_paths, boundary_indices, boundary_face_indices
        
if __name__ == "__main__":
    # Example usage
    robot_name = "kuka_iiwa_14"
    xml_path = (Path(__file__).resolve().parent / ".." / f"{robot_name}/scene.xml").as_posix()
    model = mujoco.MjModel.from_xml_path(f"{xml_path}")
    data = mujoco.MjData(model)
    mesh_sampler = MeshSampler(model, data, init_mesh_data=True)
    vis_mesh_body_name = "link6"
    n_mesh = len(mesh_sampler.data_dict["body_names_mapping"][vis_mesh_body_name]["mesh_name"])
    if n_mesh > 1:
        idx = np.random.randint(0, n_mesh)
    else:
        idx = 0
    print(n_mesh, "NMESH")
    mesh_id = mesh_sampler.data_dict["body_names_mapping"][vis_mesh_body_name]["mesh_id"][0]
    mesh_name = mesh_sampler.data_dict["body_names_mapping"][vis_mesh_body_name]["mesh_name"][0]

    # visualize mesh
    v_start = model.mesh_vertadr[mesh_id]
    v_count = model.mesh_vertnum[mesh_id]
    vertices = model.mesh_vert[v_start : v_start+v_count].reshape(-1, 3)
    
    f_start = model.mesh_faceadr[mesh_id]
    f_count = model.mesh_facenum[mesh_id]
    faces = model.mesh_face[f_start : f_start+f_count].reshape(-1, 3)
    
    mesh_sampler.visualize_mesh(vertices=vertices, faces=faces, faces_start=0, faces_end=800)
    
    indexes = mesh_sampler.feasible_region_idxes[mesh_name]
    mesh_sampler.visualize_mesh_indexes(vertices=vertices, faces=faces, faces_indexes=indexes)

    mesh_id, geom_id, faces_center_local, normals_local, rot_mats, face_vertices_select = mesh_sampler.sample_body_pos_normal("link6", num_samples=5)
    mesh_sampler.visualize_normal_arrow(mesh_id, geom_id, faces_center_local, rot_mats)