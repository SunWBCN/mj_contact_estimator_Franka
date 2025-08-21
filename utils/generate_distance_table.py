# Imports
import pygeodesic.geodesic as geodesic
import numpy as np
import trimesh
from pathlib import Path
import argparse
from tqdm import trange
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
    parser.add_argument('--index', type=int, default=6, help='Index of the mesh to process')
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
    
    distances_faces = np.ones(shape=(len(faces), len(faces)), dtype=np.float64) * -1  # shape (n_faces, n_faces)     
    distances = np.ones(shape=(len(points), len(points)), dtype=np.float64) * -1  # shape (n_points, n_points)

    # load valid face indexes
    valid_face_indices_npy = (Path(__file__).resolve().parent / ".." / f"kuka_iiwa_14/mesh_data/{mesh_name}_valid_face_indices.npy").as_posix()
    valid_face_indices = np.load(valid_face_indices_npy)
    if valid_face_indices.shape[0] == 2 and valid_face_indices[0] == 0 and valid_face_indices[1] == -1:
        valid_face_indices = np.arange(faces.shape[0])  # all faces are valid

    valid_vertex_indices_npy = (Path(__file__).resolve().parent / ".." / f"kuka_iiwa_14/mesh_data/{mesh_name}_valid_vertex_indices.npy").as_posix()
    valid_vertex_indices = np.load(valid_vertex_indices_npy)
    if valid_vertex_indices.shape[0] == 2 and valid_vertex_indices[0] == 0 and valid_vertex_indices[1] == -1:
        valid_vertex_indices = np.arange(points.shape[0])  # all vertices are valid

    start_time = time.time()
    for i in trange(len(points), desc=f"Computing geodesic distances for {mesh_name}"):
        sourceIndex = np.array([i], dtype=np.int64)
        distances_, best_source_ = geoalg.geodesicDistances(sourceIndex)
        distances[i, :] = distances_
    sliced_distances = distances[np.ix_(valid_vertex_indices, valid_vertex_indices)]
    if np.any(sliced_distances == np.inf):
        raise ValueError(f"Geodesic distance for vertices contains inf values, indicating disconnected components")
    
    face_dis_repre = "min"
    for sourceFaceIndex in valid_face_indices:
        for targetFaceIndex in valid_face_indices:
            face_source = faces[sourceFaceIndex]
            face_target = faces[targetFaceIndex]
            if sourceFaceIndex == targetFaceIndex:
                distances_faces[sourceFaceIndex, targetFaceIndex] = 0.0
            else:
                distances_c = []
                for v_source in face_source:
                    for v_target in face_target:
                        distances_c.append(distances[v_source, v_target])
                if face_dis_repre == "min":
                    distances_faces[sourceFaceIndex, targetFaceIndex] = np.min(distances_c)
                elif face_dis_repre == "mean":
                    distances_faces[sourceFaceIndex, targetFaceIndex] = np.mean(distances_c)
                    
    sliced_distances = distances[np.ix_(valid_face_indices, valid_face_indices)]
    if np.any(sliced_distances == np.inf):
        raise ValueError(f"Geodesic distance for faces contains inf values, indicating disconnected components")
    if np.any(sliced_distances == -1):
        raise ValueError(f"Geodesic distance for faces contains -1 values, indicating uncomputed distances")

    # save the distance matrixes into a file
    np.savez_compressed(f"../kuka_iiwa_14/mesh_geodesic/distance_{mesh_name}.npz",
                        distances=distances,
                        sliced_distances=sliced_distances,
                        )

    end_time = time.time()
    print(f"Time taken to compute geodesic distances for {mesh_name}: {end_time - start_time} seconds")
