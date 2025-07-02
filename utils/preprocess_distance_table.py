import trimesh
from pathlib import Path
import numpy as np

feasible_region_idxes = {"link_7": [[988, -1]], "link_6_orange": [[0, -1]], "link_6_grey": [[0, -1]],
                         "link_5": [[800, 1800], [2000, -1]],
                         "link_4_orange": [[0, -1]], "link_4_grey": [[0, -1]], "link_3": [[675, 5979], [6236, -1]],
                         "link_2_orange": [[0, -1]], "link_2_grey": [[0, -1]],
                         "link_1": [[400, 1000], [1000, 8120], [8451, -1]]}

if __name__ == "__main__":
    # Read the mesh to get the points and faces of the mesh
    mesh_names = ["link_1", "link_2_orange", "link_2_grey", "link_3", "link_4_orange", "link_4_grey", "link_5", 
                  "link_6_orange", "link_6_grey", "link_7"]
    fill_zero = False
    save_valid_region = True
    if fill_zero:
        distance_table_names = [f"distance_{mesh_names[i]}.npz" for i in range(10)]
        for i in range(10):
            file_path = f"{Path(__file__).resolve().parent}/{distance_table_names[i]}"
            distance_table = np.load(file_path)["distances"]
            
            # check if the distance table is a valid numpy array
            for j in range(len(distance_table)):
                num_zero_before = np.sum(distance_table[j, :] == 0.0)
                indexes = np.argsort(distance_table[j, :])[:20]
                values = distance_table[j, indexes]
                # find the smallest value but not equal to 0.0 in values
                min_value = np.min(values[values > 0.0])
                threshold = 0.8 * min_value
                # find the indexes that are zero in values
                zero_indexes = np.where(distance_table[j, :] == 0.0)[0]
                for zero_index in zero_indexes:
                    if zero_index != j:
                        distance_table[j, zero_index] = threshold
                num_zero = np.sum(distance_table[j, :] == 0.0)
                print(f"Row {j}: {num_zero_before} zeros before, {num_zero} zeros after, ")
                
            # Print out all the indexes that are zero 
            for j in range(len(distance_table)):
                zero_indexes = np.where(distance_table[j, :] == 0.0)[0]
                if len(zero_indexes) > 1:
                    assert False, f"Row {j} has more than one zero: {zero_indexes}"
                if zero_indexes[0] != j:
                    assert False, f"Row {j} has zero at index {zero_indexes[0]}, which is not the same as the row index."
                print(j, zero_indexes)
            
            # Save the modified distance table
            np.savez(file_path, distances=distance_table)
    
    # # compare the shape of the distance table with the number of faces in the mesh
    # import trimesh
    # for i in range(10):
    #     file_path = f"{Path(__file__).resolve().parent}/../kuka_iiwa_14/mesh_geodesic/distance_{mesh_names[i]}.npz"
    #     distance_table = np.load(file_path)["distances"]
    #     mesh = trimesh.load(f"{Path(__file__).resolve().parent}/../kuka_iiwa_14/assets/{mesh_names[i]}.obj", process=False)
    #     print(f"Mesh {mesh_names[i]}: distance table shape {distance_table.shape}, mesh faces {mesh.faces.shape[0]}")
    
    if save_valid_region:
        for i in range(len(mesh_names)):
            mesh_name = mesh_names[i]
            file_path = f"{Path(__file__).resolve().parent}/../kuka_iiwa_14/mesh_geodesic/distance_{mesh_name}.npz"
            distance_table = np.load(file_path)["distances"]
            faces_number = distance_table.shape[0]
            
            idxes = feasible_region_idxes[mesh_name]
            indexes = []
            for idx in idxes:
                if idx[1] == -1:
                    indexes.extend(range(idx[0], faces_number))
                else:
                    indexes.extend(range(idx[0], idx[1]))
            indexes = np.array(indexes)
            
            sliced_distance_table = distance_table[np.ix_(indexes, indexes)]
            
            # Save the sliced distance table
            sliced_file_path = f"{Path(__file__).resolve().parent}/../kuka_iiwa_14/mesh_geodesic/sliced_distance_{mesh_name}.npz"
            np.savez(sliced_file_path, distances=sliced_distance_table)