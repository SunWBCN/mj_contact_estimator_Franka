import mujoco
import mujoco.viewer
import numpy as np
import trimesh
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import cdist

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

class MeshSampler:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, init_mesh_data: bool = True, robot_name: str = "kuka_iiwa_14"):
        self.model = model
        self.data = data
        self.mesh_names = [model.mesh(i).name for i in range(model.nmesh)]
        self.geom_names = [model.geom(i).name for i in range(model.ngeom)]
        self.robot_name = robot_name
        if init_mesh_data:
            self._init_mesh_data()
        else:
            xml_path = (Path(__file__).resolve().parent / ".." / f"{robot_name}/mesh_data").as_posix()
            self.data_dict = np.load(f"{xml_path}/mesh_data_dict.npy", allow_pickle=True).item()
            print("Mesh data loaded from mesh_data.npy")
            for mesh_name in self.mesh_names:
                if mesh_name not in self.data_dict:
                    print(f"Mesh {mesh_name} not found in loaded data.")
    
    def _init_mesh_data(self):
        """
        Initialize mesh data for visualization.
        """
        # Get vertex indices
        print("Number of meshes:", self.model.nmesh)
        print("Number of vertices:", len(self.model.mesh_vertadr))

        self.data_dict = {}
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
        
            # skip the meshes that are not able to acts as a contact surface
            if mesh_name == "band" or mesh_name == "kuka":
                continue
        
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
            self.data_dict[mesh_name] = data_dict_mesh_i
            
        # Generate a mapping from body names to mesh and geom names
        body_names = [self.model.body(i).name for i in range(self.model.nbody) if "link" in self.model.body(i).name]
        body_names_mapping = {}
        for body_name in body_names:
            body_id = self.model.body(body_name).id
            geom_ids = [i for i in range(self.model.ngeom) if self.model.geom_bodyid[i] == body_id]
            if len(geom_ids) == 0:
                raise ValueError(f"No geoms found for body '{body_name}'")
            mesh_ids = [self.model.geom_dataid[i] for i in geom_ids]
            mesh_names = [self.mesh_names[i] for i in mesh_ids if i != -1]
            if len(mesh_names) == 0:
                raise ValueError(f"No meshes found for body '{body_name}'")     
            mesh_name = mesh_names[0]        
            mesh_id = mesh_ids[0]
            geom_id = geom_ids[0]    
            
            body_names_mapping[body_name] = {
                "mesh_name": mesh_name,
                "mesh_id": mesh_id,
                "geom_id": geom_id,
                "mesh_names": mesh_names,
                "mesh_ids": mesh_ids,
                "geom_names": self.data_dict[mesh_name]["geom_names"],
                "geom_ids": self.data_dict[mesh_name]["geom_ids"]
            }
        self.data_dict["body_names_mapping"] = body_names_mapping

        xml_path = (Path(__file__).resolve().parent / ".." / f"{self.robot_name}/mesh_data").as_posix()
        file_name = f"{xml_path}/mesh_data_dict.npy"
        np.save(file_name, self.data_dict)
        print(f"Mesh data saved to {file_name}")
            
    def visualize_mesh(self, mesh_id: int, num_faces_to_show: int = 10):
        v_start = self.model.mesh_vertadr[mesh_id]
        v_count = self.model.mesh_vertnum[mesh_id]
        vertices = self.model.mesh_vert[v_start : v_start+v_count].reshape(-1, 3)
        
        f_start = self.model.mesh_faceadr[mesh_id]
        f_count = self.model.mesh_facenum[mesh_id]
        faces = self.model.mesh_face[f_start : f_start+f_count].reshape(-1, 3)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

        for face_i in range(num_faces_to_show):
            face = faces[face_i]
            face_vertices = vertices[face]
            face_mesh = trimesh.Trimesh(vertices=face_vertices, faces=[[0, 1, 2]], process=False)
            face_mesh.visual.vertex_colors = [0, 255, 0, 255]  # Green color
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

    def sample_body_pos_normal(self, body_name: str, num_samples: int = 3):
        """
        Sample a position and normal vector from the body.
        
        Args:
            body_name: Name of the body to sample from.
        
        Returns:
            pos: Sampled position (3D vector).
            normal: Sampled normal vector (3D vector).
        """
        # body_id = self.model.body(body_name).id
        # geom_ids = [i for i in range(self.model.ngeom) if self.model.geom_bodyid[i] == body_id]
        # if len(geom_ids) == 0:
        #     raise ValueError(f"No geoms found for body '{body_name}'")
        # mesh_ids = [self.model.geom_dataid[i] for i in geom_ids]
        # mesh_names = [self.mesh_names[i] for i in mesh_ids if i != -1]
        # if len(mesh_names) == 0:
        #     raise ValueError(f"No meshes found for body '{body_name}'")     
        # mesh_name = mesh_names[0]        
        # mesh_id = mesh_ids[0]
        # geom_id = geom_ids[0]
        mesh_name = self.data_dict["body_names_mapping"][body_name]["mesh_name"]
        mesh_id = self.data_dict["body_names_mapping"][body_name]["mesh_id"]
        geom_id = self.data_dict["body_names_mapping"][body_name]["geom_id"]
        face_center_select, normal_select, rot_mat_select, face_vertices_select = self.sample_pos_normal(mesh_name, num_samples)
        return mesh_id, geom_id, face_center_select, normal_select, rot_mat_select, face_vertices_select
        
    def compute_equivalent_wrenches(self, contact_poss_geom: list, rots_mat_contact_geom: list, normal_vecs_geom: list, sample_body_name: str, geom_id: int, ext_f_norm: float):
        model = self.model
        data = self.data
        mujoco.mj_forward(model, data)
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
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{sample_body_name}_dummy_site1") # TODO: change the hardcode case to fit the number of contact points
            model.site_pos[site_id] = contact_pos_com
            rot = R.from_matrix(rot_mat_contact_com)
            quat_mujoco = rot.as_quat(scalar_first=True)
            model.site_quat[site_id] = quat_mujoco
            jac_site_contact = np.zeros((6, model.nv))
            mujoco.mj_jacSite(model, data, jac_site_contact[:3], jac_site_contact[3:], site_id)
            jacobian_contacts.append(jac_site_contact)
            
            jac_body = np.zeros((6, model.nv))
            mujoco.mj_jacBody(model, data, jac_body[:3], jac_body[3:], model.body(f"{sample_body_name}").id)
            jacobian_bodies.append(jac_body)
            
            # print(jac_site_contact.T @ ext_wrench - jac_body.T @ equi_ext_wrench) # Check the euqivalent wrench is correct or not
            # print("========================")
            
        return equi_ext_f_poss, equi_ext_wrenchs, jacobian_bodies, contact_poss_world, ext_wrenches, jacobian_contacts

    def retrieve_data_dict(self, body_name: str):
        # body_id = self.model.body(body_name).id
        # geom_ids = [i for i in range(self.model.ngeom) if self.model.geom_bodyid[i] == body_id]
        # if len(geom_ids) == 0:
        #     raise ValueError(f"No geoms found for body '{body_name}'")
        # mesh_ids = [self.model.geom_dataid[i] for i in geom_ids]
        # mesh_names = [self.mesh_names[i] for i in mesh_ids if i != -1]
        # if len(mesh_names) == 0:
        #     raise ValueError(f"No meshes found for body '{body_name}'")     
        # mesh_name = mesh_names[0]  
        mesh_name = self.data_dict["body_names_mapping"][body_name]["mesh_name"] 
        data_dict = self.data_dict[mesh_name]
        return data_dict

    def compute_nearest_position(self, position: np.ndarray, sample_body_name: str):
        """
            Compute the nearest position on the mesh to the given position.
        """
        data_dict = self.retrieve_data_dict(sample_body_name)
        faces_center_local = data_dict["face_center_list"]
        
        # Compute the nearest position on the mesh to the given position
        nearest_pos = None
        min_dist = float("inf")
        for i in range(len(faces_center_local)):
            face_center = faces_center_local[i]
            dist = np.linalg.norm(position - face_center)
            if dist < min_dist:
                min_dist = dist
                nearest_pos = face_center
        rot_mat_select = data_dict["rot_mat_list"][i]
        face_vertices_select = data_dict["face_vertices_list"][i]
        normal_select = data_dict["normal_list"][i]
        return nearest_pos, normal_select, rot_mat_select, face_vertices_select
    
    def compute_nearest_positions(self, positions: np.ndarray, sample_body_name: str):
        data_dict = self.retrieve_data_dict(sample_body_name)
        faces_center_local = data_dict["face_center_list"]
        indices = batchwise_nearest(positions, faces_center_local)
        nearest_positions = faces_center_local[indices]
        normals = data_dict["normal_list"][indices]
        rot_mats = data_dict["rot_mat_list"][indices]
        faces_vertices_select = data_dict["face_vertices_list"][indices]
        return nearest_positions, normals, rot_mats, faces_vertices_select
        
if __name__ == "__main__":
    # Example usage
    robot_name = "kuka_iiwa_14"
    xml_path = (Path(__file__).resolve().parent / ".." / f"{robot_name}/scene.xml").as_posix()
    model = mujoco.MjModel.from_xml_path(f"{xml_path}")
    data = mujoco.MjData(model)
    mesh_sampler = MeshSampler(model, data, init_mesh_data=False)
    mesh_id, geom_id, faces_center_local, normals_local, rot_mats, face_vertices_select = mesh_sampler.sample_body_pos_normal("link7", num_samples=5)
    print(faces_center_local)
    mesh_sampler.visualize_normal_arrow(mesh_id, geom_id, faces_center_local, rot_mats)
