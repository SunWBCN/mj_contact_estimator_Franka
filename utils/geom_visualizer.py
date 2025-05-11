import numpy as np
import mujoco

def compute_arrow_rotation(d, pos_center, pos_ref):
    z = d / np.linalg.norm(d)
    y = pos_ref - pos_center
    y = y / np.linalg.norm(y)
    x = np.cross(y, z)
    x = x / np.linalg.norm(x)
    return np.column_stack((x, y, z))  # shape (3, 3)

def reset_scene(scene, ngeom_init):
    scene.ngeom = ngeom_init

def visualize_normal_arrow(scene, arrows_pos_world: list, arrows_vec_world: list, arrows_pos_ref_world: list = None, arrow_length: float = 0.2):
    rgba = np.array([1.0, 0.0, 0.0, 1.0])
        
    def update_scene(scene, arrow_pos_world, arrow_mat_world):            
        # Create a new geom as an arrow
        arrow_radius = 0.01 * arrow_length
        arrow_half_length = 0.5 * arrow_length
        size_array = np.array([arrow_radius, arrow_radius, arrow_half_length],
                            dtype=np.float32)
        
        scene.ngeom += 1  # increment ngeom
        # Compute the rotation matrix that is orthogonal to the rotation matrix
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom - 1],
            mujoco.mjtGeom.mjGEOM_ARROW,
            size=size_array,
            pos=arrow_pos_world,
            mat=arrow_mat_world.flatten(),
            rgba=rgba.astype(np.float32),
        )
                
    for i in range(len(arrows_pos_world)):
        arrow_pos_world = arrows_pos_world[i]
        arrow_vec_world = arrows_vec_world[i]
        if arrows_pos_ref_world is not None:
            arrow_pos_ref_world = arrows_pos_ref_world[i]
        else:
            # Perturb the arrow position to create a reference point if not provided
            arrow_pos_ref_world = arrow_pos_world + 0.01 * arrow_pos_world
        arrow_mat_world = compute_arrow_rotation(arrow_vec_world, arrow_pos_world, arrow_pos_ref_world)
        update_scene(scene, arrow_pos_world, arrow_mat_world)