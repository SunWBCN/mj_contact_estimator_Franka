
import jax
from mujoco import mjx
import jax.numpy as jnp

# Jit functions 
@jax.jit
def update_ctrl_mjx(mjx_data, tau):
    mjx_data = mjx_data.replace(ctrl=tau)
    return mjx_data

@jax.jit
def apply_wrench_mjx(mjx_data, wrench, body_id):
    """
    Apply a custom wrench to the specified body name using JAX.

    Args:
        mjx_model: JAX MuJoCo model
        mjx_data: JAX MuJoCo data
        wrench: Wrench vector to be applied (6D)
        body_name: Body name to which the wrenches will be applied
    """
    xfrc_applied = mjx_data.xfrc_applied.at[body_id].set(wrench)
    mjx_data = mjx_data.replace(xfrc_applied=xfrc_applied)
    return mjx_data

def warmup_jit(mjx_model, mjx_data, model, data, tau=jnp.zeros(7), target_body_name="link7"):
    mjx_data = update_ctrl_mjx(mjx_data, tau)
    mjx_data = jit_step(mjx_model, mjx_data)
    mjx_data = jit_inverse(mjx_model, mjx_data)
    mjx_data = jit_forward(mjx_model, mjx_data)
    body_id = model.body(target_body_name).id
    mjx_data = apply_wrench_mjx(mjx_data, jnp.zeros(6), body_id)
    return mjx_data

@jax.jit
def get_contact_pos_rot(mjx_data, geom_id, sample_body_id):
    # Retrieve rotation matrix and position of the center of mass (COM)
    geom_pos_world = mjx_data.geom_xpos[geom_id]  # shape (3,)
    rot_mat_geom_world = mjx_data.geom_xmat[geom_id].reshape(3, 3)
    com_pos_world = mjx_data.xpos[sample_body_id]
    rot_mat_com_world = mjx_data.xmat[sample_body_id].reshape(3, 3)
    return geom_pos_world, rot_mat_geom_world, com_pos_world, rot_mat_com_world

@jax.jit
def get_batch_contact_pos_rot(mjx_data, geom_ids, sample_body_ids):
    return jax.vmap(get_contact_pos_rot, in_axes=(None, 0, 0))(mjx_data, geom_ids, sample_body_ids)

jit_step = jax.jit(mjx.step)
jit_inverse = jax.jit(mjx.inverse)
jit_forward = jax.jit(mjx.forward)

if __name__ == "__main__":
    import mujoco
    
    
    # Load the model and data.
    model = mujoco.MjModel.from_xml_path("../kuka_iiwa_14/scene.xml")
    data = mujoco.MjData(model)
    # Reset the model and data.
    mujoco.mj_resetData(model, data)
    
    # Initial joint configuration saved as a keyframe in the XML file.
    key_name = "home"
    key_id = model.key(key_name).id
    
    # Reset the simulation.
    mujoco.mj_resetDataKeyframe(model, data, key_id)

    # Update model and data.
    mujoco.mj_forward(model, data)

    # Initialize the JAX model and data for mjx.
    mjx_model = mjx.put_model(model)
    mjx_data = mjx.put_data(model, data)
    
    # Warmup the JAX model and data.
    geom_id = 60
    sample_body_name = "link7"
    sample_body_id = model.body(sample_body_name).id
    geom_poss_world, rot_mats_geom_world, com_poss_world, rot_mats_com_world \
    = get_batch_contact_pos_rot(mjx_data, jnp.array([geom_id, geom_id, geom_id]), jnp.array([sample_body_id, sample_body_id, sample_body_id]))
    
    print(geom_poss_world.shape)  # Should be (3, 3)
    print(rot_mats_geom_world.shape)  # Should be (3, 3,