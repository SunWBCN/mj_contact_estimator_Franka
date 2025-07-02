# Imports
import pygeodesic
import pygeodesic.geodesic as geodesic
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from pathlib import Path
import argparse

def compute_face_centroids(vertices, faces):
    return vertices[faces].mean(axis=1)  # shape (n_faces, 3)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
    parser.add_argument('--index', type=int, default=0, help='Index of the mesh to process')
    args = parser.parse_args()
    
    # Read the mesh to get the points and faces of the mesh
    mesh_names = ["link_1", "link_2_orange", "link_2_grey", "link_3", "link_4_orange", "link_4_grey", "link_5", 
                  "link_6_orange", "link_6_grey", "link_7"]
    file_names = [(Path(__file__).resolve().parent / ".." / f"kuka_iiwa_14/assets/{mesh_name}.obj").as_posix() for mesh_name in mesh_names]

    # Read the mesh and compute the geodesic distances
    i = args.index
    mesh = trimesh.load(file_names[i], process=False, maintain_order=True) # keep the original order of the vertices and
                                                                            # faces for geodesic computation
    # if not mesh.is_watertight:
    #     print(f"Mesh {file_names[i]} is not watertight. Skipping distance computation.")
    #     break
    points = mesh.vertices
    faces = mesh.faces
    geoalg = geodesic.PyGeodesicAlgorithmExact(points, faces)

    # Compute the centroids of the faces and find the closest vertex to each centroid
    centroids = compute_face_centroids(points, faces)
    tree = cKDTree(points)
    _, closest_vertex_ids = tree.query(centroids)  # shape (n_faces,)
    face_to_vertex = closest_vertex_ids  # length = number of faces
    
    # Find the faces that share the same vertex
    # This is not necessary for the geodesic distance computation, but can be useful for
    # converting the distance between faces and vertices to the distance between faces and faces
    vertex_to_faces = {}
    for face_index, vertex_index in enumerate(face_to_vertex):
        if vertex_index not in vertex_to_faces:
            vertex_to_faces[vertex_index] = []
        vertex_to_faces[vertex_index].append(face_index)
    
    distances = np.zeros(shape=(len(faces), len(faces)), dtype=np.float64)  # shape (n_faces, n_faces)            
    
    for sourceFaceIndex in range(faces.shape[0]):
        sourceIndex = face_to_vertex[sourceFaceIndex]
        sourceIndices = np.array([sourceIndex])
        distances_, _ = geoalg.geodesicDistances(sourceIndices) # Note that the distances_ is a 1D array
                                                                # from the source face to all vertices      
        print(sourceFaceIndex, np.sum(distances_ == 0.0))
        for vertexIndex, distance in enumerate(distances_):
            # Find the faces that share the same vertex
            if vertexIndex in vertex_to_faces:
                for targetFaceIndex in vertex_to_faces[vertexIndex]:
                    distances[sourceFaceIndex, targetFaceIndex] = distance
                    
    # save the distance matrixes into a file
    mesh_name = mesh_names[i]
    np.savez_compressed(f"distance_{mesh_name}.npz", distances=distances, face_to_vertex=face_to_vertex)
