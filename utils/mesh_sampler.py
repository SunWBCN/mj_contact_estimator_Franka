import mujoco
import mujoco.viewer
import numpy as np
import trimesh

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
            file_directory = f"../{robot_name}/mesh_data"
            self.data_dict = np.load(f"{file_directory}/mesh_data_dict.npy", allow_pickle=True).item()
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
                normal_list.append(normal)
                face_center_list.append(face_center)
                rot_mat = compute_arrow_rotation(normal, face_center, A)
                rot_mat_list.append(rot_mat)
            data_dict_mesh_i["vis_face_list"] = vis_face_list
            data_dict_mesh_i["normal_list"] = normal_list
            data_dict_mesh_i["face_center_list"] = face_center_list
            data_dict_mesh_i["rot_mat_list"] = rot_mat_list
            data_dict_mesh_i["geom_ids"] = geom_ids
            data_dict_mesh_i["face_vertices_list"] = face_vertices_list
            data_dict_mesh_i["num_faces"] = f_count
            data_dict_mesh_i["num_vertices"] = v_count
            data_dict_mesh_i["geom_names"] = geom_names
            data_dict_mesh_i["geom_ids"] = geom_ids
            self.data_dict[mesh_name] = data_dict_mesh_i
        file_directory = f"{self.robot_name}/mesh_data"
        file_name = f"{file_directory}/mesh_data_dict.npy"
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
        
        idxs = np.random.choice(len(face_center_list), num_samples, replace=False)
        face_center_select = np.array(face_center_list)[idxs]
        normal_select = np.array(normal_list)[idxs]
        rot_mat_select = np.array(rot_mat_list)[idxs]
        face_vertices_select = np.array(face_vertices_list)[idxs]
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
        face_center_select, normal_select, rot_mat_select, face_vertices_select = self.sample_pos_normal(mesh_name, num_samples)
        return mesh_id, geom_id, face_center_select, normal_select, rot_mat_select, face_vertices_select
        
if __name__ == "__main__":
    # Example usage
    model = mujoco.MjModel.from_xml_path("../kuka_iiwa_14/scene.xml")
    data = mujoco.MjData(model)
    mesh_sampler = MeshSampler(model, data, init_mesh_data=False)
    mesh_id, geom_id, faces_center_local, normals_local, rot_mats, face_vertices_select = mesh_sampler.sample_body_pos_normal("link6", num_samples=5)
    print(faces_center_local)
    mesh_sampler.visualize_normal_arrow(mesh_id, geom_id, faces_center_local, rot_mats)
