
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

def warmup_jit(mjx_model, mjx_data, model, data, tau=jnp.zeros(7), sample_body_name="link7"):
    mjx_data = update_ctrl_mjx(mjx_data, tau)
    mjx_data = jit_step(mjx_model, mjx_data)
    mjx_data = jit_inverse(mjx_model, mjx_data)
    mjx_data = jit_forward(mjx_model, mjx_data)
    body_id = model.body(sample_body_name).id
    mjx_data = apply_wrench_mjx(mjx_data, jnp.zeros(6), body_id)
    return mjx_data

jit_step = jax.jit(mjx.step)
jit_inverse = jax.jit(mjx.inverse)
jit_forward = jax.jit(mjx.forward)

