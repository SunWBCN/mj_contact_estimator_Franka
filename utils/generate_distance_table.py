# Imports
import pygeodesic.geodesic as geodesic
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from pathlib import Path
import argparse
from tqdm import trange
import time

def compute_face_centroids(vertices, faces):
    return vertices[faces].mean(axis=1)  # shape (n_faces, 3)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
    parser.add_argument('--index', type=int, default=9, help='Index of the mesh to process')
    args = parser.parse_args()
    
    # Read the mesh to get the points and faces of the mesh
    mesh_names = ["link_1", "link_2_orange", "link_2_grey", "link_3", "link_4_orange", "link_4_grey", "link_5", 
                  "link_6_orange", "link_6_grey", "link_7"]
    file_names = [(Path(__file__).resolve().parent / ".." / f"kuka_iiwa_14/assets/{mesh_name}_mujoco_cached.obj").as_posix() for mesh_name in mesh_names]

    # Read the mesh and compute the geodesic distances
    i = args.index
    mesh_name = mesh_names[i]
    mesh = trimesh.load(file_names[i], process=False) # keep the original order of the vertices and
                                                      # faces for geodesic computation
    print(f"============ load file name: {file_names[i]}")
    points = mesh.vertices
    faces = mesh.faces
    geoalg = geodesic.PyGeodesicAlgorithmExact(points, faces)
    print(f"number of faces: {faces.shape[0]}, {len(faces)}")

    # Compute the centroids of the faces and find the closest vertex to each centroid
    centroids = compute_face_centroids(points, faces)
    tree = cKDTree(points)
    _, closest_vertex_ids = tree.query(centroids)  # shape (n_faces,)
    face_to_vertex = closest_vertex_ids  # length = number of faces
    
    # Pre-compute KDTree for face centroids for faster path-to-face mapping
    centroids_tree = cKDTree(centroids)
    
    # Find the faces that share the same vertex
    # This is not necessary for the geodesic distance computation, but can be useful for
    # converting the distance between faces and vertices to the distance between faces and faces
    vertex_to_faces = {}
    for face_index, vertex_index in enumerate(face_to_vertex):
        if vertex_index not in vertex_to_faces:
            vertex_to_faces[vertex_index] = []
        vertex_to_faces[vertex_index].append(face_index)
    
    distances = np.zeros(shape=(len(faces), len(faces)), dtype=np.float64)  # shape (n_faces, n_faces)     
    distances = np.empty_like(distances)  # Initialize with zeros
    distances.fill(np.inf)  # Fill with infinity to indicate uncomputed distances

    # load valid face indexes
    valid_face_indices_npy = (Path(__file__).resolve().parent / ".." / f"kuka_iiwa_14/mesh_data/{mesh_name}_valid_face_indices.npy").as_posix()
    valid_face_indices = np.load(valid_face_indices_npy)
    if valid_face_indices.shape[0] == 2 and valid_face_indices[0] == 0 and valid_face_indices[1] == -1:
        valid_face_indices = np.arange(faces.shape[0])  # all faces are valid

    start_time = time.time()
    for sourceFaceIndex in trange(len(faces)):
        if sourceFaceIndex not in valid_face_indices:
            continue
        sourceIndex = face_to_vertex[sourceFaceIndex]
        # Compute geodesic distances
        sourceIndex = np.array([sourceIndex], dtype=np.int64)
        distances_, best_source_ = geoalg.geodesicDistances(sourceIndex)
        assert np.sum(np.array(distances_) == 0) == 1, f"Distances for face {sourceFaceIndex} contain more than one zero"
        
        # map vertices to faces
        for vertexIndex, distance in enumerate(distances_):
            # Find the faces that share the same vertex
            if vertexIndex in vertex_to_faces:
                for targetFaceIndex in vertex_to_faces[vertexIndex]:
                    distances[sourceFaceIndex, targetFaceIndex] = distance
    sliced_distances = distances[np.ix_(valid_face_indices, valid_face_indices)]
    
    # check if all distance are finite and only one zero for each row
    for i_ in range(len(sliced_distances)):
        assert np.isfinite(sliced_distances[i_]).all(), f"Distances for face {i_} contain non-finite values"

    # save the distance matrixes into a file
    np.savez_compressed(f"../kuka_iiwa_14/mesh_geodesic/distance_{mesh_name}.npz",
                        distances=distances,
                        sliced_distances=sliced_distances,
                        )

    end_time = time.time()
    print(f"Time taken to compute geodesic distances for {mesh_name}: {end_time - start_time} seconds")
