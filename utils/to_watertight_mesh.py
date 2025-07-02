import trimesh
from pathlib import Path
import mujoco
from mesh_sampler import MeshSampler
import pymeshlab
    
if __name__ == "__main__":
    from pymeshlab.pmeshlab import PercentageValue

    ms = pymeshlab.MeshSet()
    robot_name = "kuka_iiwa_14"
    mesh_names = ["link_1", "link_2_orange", "link_2_grey", "link_3", "link_4_orange", "link_4_grey", "link_5", 
                  "link_6_orange", "link_6_grey", "link_7"]
    for link_id in range(len(mesh_names)):
        mesh_path = (Path(__file__).resolve().parent / ".." / f"{robot_name}/assets/{mesh_names[link_id]}.obj").as_posix()
        ms.load_new_mesh(mesh_path)
        
        # Optional cleanup: remove disconnected parts and fill small holes
        ms.meshing_remove_connected_component_by_diameter(
            mincomponentdiag=PercentageValue(0.1)
        )    
        ms.meshing_close_holes(maxholesize=100)
        ms.meshing_remove_duplicate_faces()
        ms.meshing_remove_duplicate_vertices()
        ms.meshing_remove_null_faces()
        ms.meshing_remove_connected_component_by_face_number(mincomponentsize=100)

        # === Remeshing ===
        # Perform isotropic explicit remeshing to make triangle sizes more uniform
        # targetlen is the approximate edge length you want (depends on mesh scale)
        mesh_name = mesh_names[link_id]
        if mesh_name == "link_3" or mesh_name == "link_5" or mesh_name == "link_7":
            ms.meshing_isotropic_explicit_remeshing(targetlen=PercentageValue(0.4), iterations=2)
        else:
            continue

        # Save result
        mesh_name = mesh_names[link_id]
        save_name = f"watertight_{mesh_name}.obj"
        ms.save_current_mesh(f'{save_name}')
        
        # Load the mesh and visualize it
        mesh = trimesh.load(save_name, process=False)
        # mesh.show()