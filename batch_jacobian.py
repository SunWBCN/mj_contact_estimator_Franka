import mujoco
from utils.mesh_sampler import MeshSampler
from pathlib import Path
from jax.scipy.spatial.transform import Rotation as R
import jax
import jax.numpy as jnp
from mujoco import mjx
    
"""
Batch computing the contact position and rotation matrix in the CoM frame.
"""

@jax.jit
def compute_contact_pos_rot_com(rot_mat_contact_geom, contact_pos_geom, rot_mat_geom_world, rot_mat_com_world):
    """
    Compute the contact position and rotation matrix in the center of mass (COM) frame.
    """
    rot_mat_contact_world = jnp.dot(rot_mat_geom_world, rot_mat_contact_geom)
    contact_pos_world = geom_pos_world + jnp.dot(rot_mat_geom_world, contact_pos_geom)
    contact_pos_com = jnp.dot(rot_mat_com_world.T, (contact_pos_world - com_pos_world))
    rot_mat_contact_com = jnp.dot(rot_mat_com_world.T, rot_mat_contact_world)
    return contact_pos_com, rot_mat_contact_com

@jax.jit
def compute_batch_contact_pos_rot_com(rot_mat_contact_geoms, contact_pos_geoms, rot_mat_geom_world, rot_mat_com_world):
    contact_poss_coms, rot_mats_contact_com = jax.vmap(compute_contact_pos_rot_com, in_axes=(0, 0, None, None))(rot_mat_contact_geoms, contact_pos_geoms, rot_mat_geom_world, rot_mat_com_world)
    return contact_poss_coms, rot_mats_contact_com
    
"""
Batch converting the rotation matrix to quaternion.
"""    
@jax.jit
def rotation_matrix_to_quaternion(rot_mat):
    rot = R.from_matrix(rot_mat)
    quat = rot.as_quat(scalar_first=True)
    return quat

@jax.jit
def compute_batch_rotation_matrix_to_quaternion(rot_mats_contact_com):
    quats = jax.vmap(rotation_matrix_to_quaternion)(rot_mats_contact_com)
    return quats
    
"""
Batch updating the site position and quaternion for computing jacobian.
"""
@jax.jit
def update_site_pos_quat(mjx_model, site_ids, contact_poss_coms, quats):
    # model.site_pos[site_ids, :] = contact_poss_coms
    # model.site_quat[site_ids, :] = quats
    site_pos = mjx_model.site_pos.at[site_ids].set(contact_poss_coms)
    site_quat = mjx_model.site_quat.at[site_ids].set(quats)
    mjx_model = mjx_model.replace(site_pos=site_pos, site_quat=site_quat)
    return mjx_model

"""
Batch computing the jacobian for the contact points.
"""
@jax.jit
def site_fn(mjx_model, qpos, site_id):
    """
    Compute the position of a site given qpos and site_id.
    """
    data_template = mjx.make_data(mjx_model)
    data = data_template.replace(qpos=qpos)
    data = mjx.forward(mjx_model, mjx_data)
    return data.site_xpos[site_id]  # (3,)

# Compute the Jacobian for a single site
@jax.jit
def compute_batch_site_jac(mjx_model, qpos, site_ids):
    jac_fn = jax.jacobian(site_fn, argnums=1)
    return jax.vmap(jac_fn, in_axes=(None, None, 0))(mjx_model, qpos, site_ids)

@jax.jit
def compute_batch_site_jac_pipeline(rot_mats_contact_geom, contact_poss_geom, rot_mat_geom_world,
                                    rot_mat_com_world, mjx_model, qpos, site_ids):
    """
    Compute the batch site Jacobian using the provided rotation matrices and contact positions.
    """
    contact_poss_coms, rot_mats_contact_com = compute_batch_contact_pos_rot_com(rot_mats_contact_geom, contact_poss_geom, rot_mat_geom_world, rot_mat_com_world)
    quats = compute_batch_rotation_matrix_to_quaternion(rot_mats_contact_com)
    mjx_model = update_site_pos_quat(mjx_model, site_ids, contact_poss_coms, quats)
    jacobians = compute_batch_site_jac(mjx_model, qpos, site_ids)
    return jacobians, contact_poss_coms, rot_mats_contact_com, quats

if __name__ == "__main__":
    robot_name = "kuka_iiwa_14"
    xml_path = (Path(__file__).resolve().parent / f"{robot_name}/scene.xml").as_posix()
    model = mujoco.MjModel.from_xml_path(f"{xml_path}")
    data = mujoco.MjData(model)

    mujoco.mj_resetData(model, data)
    mjx_model = mjx.put_model(model)
    mjx_data = mjx.put_data(model, data)

    print(mjx_data.time)
            
    mesh_sampler = MeshSampler(model, data, init_mesh_data=False)
    sample_body_name = "link7"
    mesh_id, geom_id, contact_poss_geom, normal_vecs_geom, rot_mats_contact_geom, face_vertices_select = mesh_sampler.sample_body_pos_normal(sample_body_name, num_samples=100)
    ext_f_norm = 5.0

    mujoco.mj_forward(model, data)
    geom_pos_world = data.geom_xpos[geom_id]        # shape (3,)
    rot_mat_geom_world = data.geom_xmat[geom_id].reshape(3, 3)    
    com_pos_world = data.xpos[model.body(sample_body_name).id]
    rot_mat_com_world = data.xmat[model.body(sample_body_name).id].reshape(3, 3)

    contact_poss_geom = jnp.array(contact_poss_geom)  # shape (n_particles, 3)
    rot_mats_contact_geom = jnp.array(rot_mats_contact_geom)  # shape (n_particles, 3, 3)
    rot_mat_geom_world = jnp.array(rot_mat_geom_world)  # shape (3, 3)
    rot_mat_com_world = jnp.array(rot_mat_com_world)  # shape (3, 3)
    com_pos_world = jnp.array(com_pos_world)  # shape (3,)

    for i in range(len(rot_mats_contact_geom)):
        rot_mat_contact_geom = rot_mats_contact_geom[i]
        rot_mat_contact_world = rot_mat_geom_world @ rot_mat_contact_geom
        contact_pos_geom = contact_poss_geom[i]
        contact_pos_world = geom_pos_world + rot_mat_geom_world @ contact_pos_geom
        contact_pos_com = rot_mat_com_world.T @ (contact_pos_world - com_pos_world)
        rot_mat_contact_com = rot_mat_com_world.T @ rot_mat_contact_world
        
        # Update the site position and quaternion for computing jacobian
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{sample_body_name}_dummy_site{i}") # TODO: change the hardcode case to fit the number of contact points
        model.site_pos[site_id] = contact_pos_com
        rot = R.from_matrix(rot_mat_contact_com)
        quat = rot.as_quat(scalar_first=True)
        model.site_quat[site_id] = quat

    import time
    for i in range(3):
        start = time.time()
        contact_poss_coms, rot_mats_contact_com = compute_batch_contact_pos_rot_com(rot_mats_contact_geom, contact_poss_geom, rot_mat_geom_world, rot_mat_com_world)
        jax.block_until_ready(contact_poss_coms)
        print("Compute Batch Time:", time.time() - start)

    import time 
    for i in range(3):
        start = time.time()
        quats = compute_batch_rotation_matrix_to_quaternion(rot_mats_contact_com)
        jax.block_until_ready(quats)
        print("Time:", time.time() - start)
        
    site_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{sample_body_name}_dummy_site{i}") for i in range(len(rot_mats_contact_geom))]
    site_ids = jnp.array(site_ids)  # shape (n_particles, )
        
    import time
    for i in range(3):
        start = time.time()
        mjx_model = update_site_pos_quat(mjx_model, site_ids, contact_poss_coms, quats)
        jax.block_until_ready(mjx_model)
        print("Update Time:", time.time() - start)

    # # Check the updated site position and quaternion
    # for i in range(len(rot_mats_contact_geom)):
    #     print(np.allclose(mjx_model.site_pos[site_ids[i]], contact_poss_coms[i]))
    #     print(np.allclose(mjx_model.site_quat[site_ids[i]], quats[i]))

    # Example usage
    qpos = jnp.zeros(model.nq)  # Example qpos

    # Timed runs
    import time
    for i in range(5):
        start = time.time()
        jacobians = compute_batch_site_jac(mjx_model, qpos, site_ids)
        jax.block_until_ready(jacobians)
        print(f"[{i}] Time: {time.time() - start:.4f}s")

    print("Jacobian shape:", jacobians.shape)  # (100, 3, nq)
    
    # Time runs
    for i in range(5):
        start = time.time()
        jacobians, contact_poss_coms, rot_mats_contact_com, quats = \
        compute_batch_site_jac_pipeline(rot_mats_contact_geom, contact_poss_geom, rot_mat_geom_world,
                                        rot_mat_com_world, mjx_model, qpos, site_ids)
        jax.block_until_ready(jacobians)
        print(f"[{i}] Time: {time.time() - start:.4f}s")
