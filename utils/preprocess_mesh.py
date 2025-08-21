import numpy as np
import trimesh
from pathlib import Path
import mujoco
import pygeodesic.geodesic as geodesic
from tqdm import trange

def visualize_mesh_indexes(vertices: np.ndarray, faces: np.ndarray, faces_indexes: np.ndarray = None):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    
    if faces_indexes is None:
        faces_indexes = np.arange(len(faces))

    faces_selected = faces[faces_indexes]
    for face in faces_selected:
        face_vertices = vertices[face]
        face_mesh = trimesh.Trimesh(vertices=face_vertices, faces=[[0, 1, 2]], process=False)
        face_mesh.visual.vertex_colors = [0, 255, 0, 255]
        mesh += face_mesh
    mesh.show()
    
def find_faces_from_vertices_vectorized(faces_array: np.ndarray, target_vertex_indices: np.ndarray):
    """
    Vectorized version for better performance with large meshes
    """
    # Create boolean mask for target vertices
    target_mask = np.isin(faces_array, target_vertex_indices)
    
    # Find faces that contain at least one target vertex
    face_mask = np.any(target_mask, axis=1)
    face_indices = np.where(face_mask)[0]
    
    return face_indices
    
def check_common_entries_numpy(list1, list2):
    """Check if two arrays have common entries"""
    common = np.intersect1d(list1, list2)
    has_common = len(common) > 0
    return has_common, common

def compute_face_centroids(vertices, faces):
    return vertices[faces].mean(axis=1)  # shape (n_faces, 3)

def test_if_geodesic_computable(points, faces, valid_vertex_indices):
    geoalg = geodesic.PyGeodesicAlgorithmExact(points, faces)
    invalid_vertex_indexes = []
    for i in trange(len(points)):
        sourceIndex = np.array([i], dtype=np.int64)
        distances_, best_source_ = geoalg.geodesicDistances(sourceIndex)
        sliced_distances = distances_[valid_vertex_indices]
        if np.any(sliced_distances == np.inf):
            invalid_vertex_indexes.extend(sourceIndex)
    if len(invalid_vertex_indexes) > 0:
        return False, invalid_vertex_indexes
    else:
        return True, []

def test_only_if_geodesic_computable(points, faces, valid_face_indices, valid_vertex_indices):
    geoalg = geodesic.PyGeodesicAlgorithmExact(points, faces)
    for sourceFaceIndex in trange(len(faces), desc="Testing geodesic computability"):
        if sourceFaceIndex not in valid_face_indices:
            continue
        face = faces[sourceFaceIndex]
        for vertexIndex in face:
            sourceIndex = np.array([vertexIndex], dtype=np.int64)
            distances_, best_source_ = geoalg.geodesicDistances(sourceIndex)
            slices_distances = distances_[valid_vertex_indices]
            if np.any(slices_distances == np.inf):
                return False
    return True

if __name__ == "__main__":
    # Read the text.txt file and retreive vertex information
    robot_name = "kuka_iiwa_14"
    xml_path_single_mesh = "/home/junninghuang/Desktop/Code/contact_estimation/blender/single_mesh.xml"

    model = mujoco.MjModel.from_xml_path(f"{xml_path_single_mesh}")
    mesh_name = model.mesh(0).name
    print(mesh_name)
    mesh_id = 0
    v_start = model.mesh_vertadr[mesh_id]
    v_count = model.mesh_vertnum[mesh_id]
    vertices = model.mesh_vert[v_start : v_start+v_count].reshape(-1, 3)

    f_start = model.mesh_faceadr[mesh_id]
    f_count = model.mesh_facenum[mesh_id]
    faces = model.mesh_face[f_start : f_start+f_count].reshape(-1, 3)

    idx_path = f"{Path(__file__).resolve().parent}/../kuka_iiwa_14/mesh_data/{mesh_name}_mujoco_cached_del_ind.npy"
    if idx_path is None or not Path(idx_path).exists():
        vertex_indices = np.array([], dtype=np.int64)
        face_indices = np.array([], dtype=np.int64)
    else:
        vertex_indices = np.load(idx_path)
        face_indices = find_faces_from_vertices_vectorized(faces, vertex_indices)
        print(len(vertex_indices), "vertices to be deleted")
    # show the faces indexes that need to be deleted
    # visualize_mesh_indexes(vertices=vertices, faces=faces, faces_indexes=face_indices)
    
    # exclude the face_indices that need to be deleted
    total_indices = np.arange(len(faces))
    valid_face_indices = np.setdiff1d(total_indices, face_indices)
    print(len(valid_face_indices), "valid faces after deletion")
    # visualize_mesh_indexes(vertices=vertices, faces=faces, faces_indexes=valid_face_indices)
    valid_vertex_indices = np.setdiff1d(np.arange(len(vertices)), vertex_indices)    

    is_comp, invalid_vertex_indexes = test_if_geodesic_computable(vertices, faces, valid_vertex_indices)
    print("Is geodesic computable:", is_comp, invalid_vertex_indexes, len(invalid_vertex_indexes))

    # remove also the not computable face indexes in valid_face_indices
    valid_vertex_indices = np.setdiff1d(valid_vertex_indices, invalid_vertex_indexes)
    invalid_face_indices = find_faces_from_vertices_vectorized(faces, invalid_vertex_indexes)
    print(len(valid_face_indices), "Before filtering, Face")
    valid_face_indices = np.setdiff1d(valid_face_indices, invalid_face_indices)
    print(len(valid_face_indices), "After filtering, Face")
    
    is_comp = test_only_if_geodesic_computable(vertices, faces, valid_face_indices, valid_vertex_indices)
    if not is_comp:
        raise ValueError(f"Geodesic distance computation is not valid for the mesh {mesh_name}, invalid face indexes: {invalid_face_indices}, invalid vertex indexes: {invalid_vertex_indexes}")
    else:
        print("Geodesic distance computation is valid")

    # visualize it
    visualize_mesh_indexes(vertices=vertices, faces=faces, faces_indexes=valid_face_indices)
    idx_path = f"{Path(__file__).resolve().parent}/../kuka_iiwa_14/mesh_data/{mesh_name}_mujoco_cached_nom_ind.npy"
    nom_vertex_indices = np.load(idx_path)
    for vertex_index in vertex_indices:
        if vertex_index in nom_vertex_indices:
            print(f"Vertex {vertex_index} is in the non-manifold vertex indices, but it should be deleted.")
            exit(0)

    face_indices = find_faces_from_vertices_vectorized(faces, nom_vertex_indices)

    # Remove face indices that are not in the valid face indices
    nom_face_indices = np.intersect1d(face_indices, valid_face_indices)

    # show the faces indexes that need to be deleted
    visualize_mesh_indexes(vertices=vertices, faces=faces, faces_indexes=nom_face_indices)

    if np.any(np.isin(invalid_face_indices, nom_face_indices)) or np.any(np.isin(invalid_face_indices, valid_face_indices)):
        raise ValueError("Invalid face indexes found in non-manifold face indices or valid face indices")

    # save the valid face indices
    dir_path = f"{Path(__file__).resolve().parent}/../kuka_iiwa_14/mesh_data"
    np.save(f"{dir_path}/{mesh_name}_valid_vertex_indices.npy", valid_vertex_indices)
    np.save(f"{dir_path}/{mesh_name}_valid_face_indices.npy", valid_face_indices)
    np.save(f"{dir_path}/{mesh_name}_non_manifold_face_indices.npy", nom_face_indices)
    np.save(f"{dir_path}/{mesh_name}_non_manifold_vertex_indices.npy", nom_vertex_indices)