from jaxopt import OSQP
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

# Generate the QP problem with a pyramid constraint to approximate the friction cone
# The QP is a least square problem with a linear inequality constraint
mu = 1.0
G = jnp.array([
    [-0.0, -0.0, -1.0],   # -f_z <= 0  --> f_z >= 0
    [ 1.0,  0.0, -mu],    #  f_x - mu f_z <= 0
    [-1.0,  0.0, -mu],    # -f_x - mu f_z <= 0
    [ 0.0,  1.0, -mu],    #  f_y - mu f_z <= 0
    [ 0.0, -1.0, -mu],    # -f_y - mu f_z <= 0
])
h = jnp.zeros((5,))
G = jax.device_put(G, device=jax.devices()[0])
h = jax.device_put(h, device=jax.devices()[0])

@jax.jit
def solve_single_qp(Jc, gt_ext_tau):
    """
    Solve a single QP problem using OSQP.
    """
    Q = jnp.dot(Jc, Jc.T)
    c = -jnp.dot(Jc, gt_ext_tau)
    qp_solver = OSQP()
    res = qp_solver.run(params_obj=(Q, c), params_eq=None, params_ineq=(G, h))
    return res.params.primal, res.state.error

# Batch QP solver
@jax.jit
def solve_batch_qp(Jc_batch, gt_ext_tau):
    return jax.vmap(solve_single_qp, in_axes=(0, None))(Jc_batch, gt_ext_tau)

if __name__ == "__main__":
    import numpy as np
    data_dict = np.load("./utils/data.npz")

    # load data and conert them into jax array
    gt_ext_taus = []
    Jcs = []
    gt_ext_wrenches = []
    npart = 20
    for i in range(100, 100+npart):
        index = i
        Jc = data_dict["jacs_contact"][index]
        ext_tau = data_dict["gt_ext_taus"][index]
        ext_wrench = data_dict["gt_ext_wrenches"][index]
        gt_ext_tau = Jc.T @ ext_wrench
        gt_ext_taus.append(gt_ext_tau)
        gt_ext_wrenches.append(ext_wrench)
        
        # Solve the QP problem
        Jc = Jc[:3, :]
        Jcs.append(Jc)
        
        # Solve the QP problem
        param, error = solve_single_qp(Jc, gt_ext_tau)
        # print("Contact forces:", param)
        # print("Objective value:", error)
        # print("Ground Truth External force:", ext_wrench)

    Jcs = jnp.array(Jcs)
    gt_ext_taus = jnp.array(gt_ext_taus)
    gt_ext_wrenches = jnp.array(gt_ext_wrenches)
    Jcs = jax.device_put(Jcs, device=jax.devices()[0])
    gt_ext_taus = jax.device_put(gt_ext_taus, device=jax.devices()[0])
    gt_ext_wrenches = jax.device_put(gt_ext_wrenches, device=jax.devices()[0])

    import time
    # repeat Jcs[0] 10 times to create a batch of Jc
    # Jcs_batch = jnp.repeat(Jcs[0][None, :], 100, axis=0)
    Jcs_batch = Jcs
    for i in range(10):
        start = time.time()
        params, errors = solve_batch_qp(Jcs_batch, gt_ext_taus[0])
        jax.block_until_ready(params)
        print("Batch QP Time:", time.time() - start)
    print(params[:30])
    print(errors[:30])
    print(gt_ext_wrenches[0])
    # for i in range(30):
    #     param, error = solve_single_qp(Jcs[i], gt_ext_taus[0])
    #     jax.block_until_ready(param)
    #     print(param)
    #     print(error)