import mujoco
import numpy as np

def compute_gravity_forces(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """
    Compute generalized gravity forces g(q) using mujoco.mj_rne.
    
    Args:
        model: mujoco.MjModel
        data: mujoco.MjData (qpos should be valid)
    
    Returns:
        g: Gravity torque vector of shape (nv,)
    """
    nv = model.nv

    # Backup state
    qpos_backup = data.qpos.copy()
    qvel_backup = data.qvel.copy()
    qacc_backup = data.qacc.copy()

    # Set velocities and accelerations to zero
    data.qvel[:] = 0
    data.qacc[:] = 0

    # Ensure valid qpos
    mujoco.mj_forward(model, data)

    # Compute full inverse dynamics: gravity only when vel and acc = 0
    g = np.zeros(nv)
    mujoco.mj_rne(model, data, 0, g)  # 1 = include gravity
    g = data.qfrc_bias.copy()

    # Restore original state
    data.qpos[:] = qpos_backup
    data.qvel[:] = qvel_backup
    data.qacc[:] = qacc_backup
    mujoco.mj_forward(model, data)

    return g

def compute_joint_space_inertia_matrix(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """
    Compute the joint space inertia matrix M(q) using MuJoCo's mj_rne().

    Returns:
        M: Joint space inertia matrix of shape (nv, nv)
    """
    M = np.zeros((model.nv, model.nv))  # Pre-allocate the dense matrix
    mujoco.mj_forward(model, data) # Warning: don't forget to call mj_forward before mj_fullM
                                    # to update the dynamics state                      
    mujoco.mj_fullM(model, M, data.qM)
    return M

class mujocoDyn:
    """
    Class to compute generalized gravity forces and Coriolis matrix using MuJoCo.
    
    Args:
        model: mujoco.MjModel
        data: mujoco.MjData
    """
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        self.nv = model.nv

    def compute_all_forces(self):
        # TODO: Implement the function to compute the coriolis matrix
        """
        Compute all forces including gravity and Coriolis forces.
        
        Returns:
            g: Gravity torque vector of shape (nv,)
            M: Joint space inertia matrix of shape (nv, nv)
            C: Coriolis matrix of shape (nv, nv)
        """
        
        g = compute_gravity_forces(self.model, self.data)
        M = compute_joint_space_inertia_matrix(self.model, self.data)
        return g, M