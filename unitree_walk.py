from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
import mujoco

from quad_standalone.policy import Policy
from quad_standalone.viewer import MujocoViewer
import pinocchio as pino
from contact_estimator import high_gain_based_observer, kalman_disturbance_observer
from utils.mujoco_dyn import compute_gravity_forces

if __name__ == "__main__":
    # Constants
    CONTROL_FREQUENCY_IN_HZ = 50
    SIMULATION_FREQUENCY_IN_HZ = 200
    VELOCITY_COMMAND = (0.0, 0.0, 1.0)  # (x, y, yaw), min and max values are (-1.0, 1.0)
    P_GAIN = 20.0
    D_GAIN = 0.5
    ACTION_SCALING = 0.25
    MAX_JOINT_VELOCITIES = np.array([21.0] * 12)

    # Initialize the policy
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = Policy(num_inputs=48, num_actions=12).to(device)

    # Load the weights from the trained model
    dense_0_weights = np.load("quad_standalone/policy_weights/Dense_0_weights.npy").T
    dense_0_biases = np.load("quad_standalone/policy_weights/Dense_0_biases.npy")
    layer_norm_biases = np.load("quad_standalone/policy_weights/LayerNorm_0_biases.npy")
    dense_1_weights = np.load("quad_standalone/policy_weights/Dense_1_weights.npy").T
    dense_1_biases = np.load("quad_standalone/policy_weights/Dense_1_biases.npy")
    dense_2_weights = np.load("quad_standalone/policy_weights/Dense_2_weights.npy").T
    dense_2_biases = np.load("quad_standalone/policy_weights/Dense_2_biases.npy")
    dense_3_weights = np.load("quad_standalone/policy_weights/Dense_3_weights.npy").T
    dense_3_biases = np.load("quad_standalone/policy_weights/Dense_3_biases.npy")

    # Set the weights of the model
    policy.layer1.weight.data = torch.tensor(dense_0_weights, device=device)
    policy.layer1.bias.data = torch.tensor(dense_0_biases, device=device)
    policy.layer_norm.bias.data = torch.tensor(layer_norm_biases, device=device)
    policy.layer2.weight.data = torch.tensor(dense_1_weights, device=device)
    policy.layer2.bias.data = torch.tensor(dense_1_biases, device=device)
    policy.layer3.weight.data = torch.tensor(dense_2_weights, device=device)
    policy.layer3.bias.data = torch.tensor(dense_2_biases, device=device)
    policy.layer4.weight.data = torch.tensor(dense_3_weights, device=device)
    policy.layer4.bias.data = torch.tensor(dense_3_biases, device=device)

    # Initialize the Mujoco environment
    xml_path = (Path(__file__).resolve().parent / "quad_standalone" / "data" / "plane.xml").as_posix()
    model = mujoco.MjModel.from_xml_path(xml_path)
    model.opt.timestep = 1 / SIMULATION_FREQUENCY_IN_HZ
    data = mujoco.MjData(model)
    viewer = MujocoViewer(model, 1 / CONTROL_FREQUENCY_IN_HZ)
    nominal_joint_position = model.key_qpos[0, 7:]
    trunk_geom_names = ["trunk_1", "trunk_2", "trunk_3", "trunk_4", "trunk_5", "trunk_6", "trunk_7", "trunk_8"]
    trunk_geom_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name) for geom_name in trunk_geom_names]
    flip_obs_joint_idx_mask = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
    previous_action = np.zeros(12)
    
    dhdts_gt = []
    hgs_gt = []
    hgs_pred = []
    hg_pred = np.zeros(6)
    foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]

    pino_model = pino.buildModelFromMJCF(xml_path)
    pino_data = pino_model.createData()

    steps = 0

    # Initialize the contact estimator
    gm_estimator = high_gain_based_observer(1 / SIMULATION_FREQUENCY_IN_HZ, "hg", model.nv)
    gt_ext_taus = []
    est_ext_taus = []

    while True:
        # Generate observation
        qpos = (data.qpos[7:] - nominal_joint_position) / 3.14
        qpos = np.append(qpos[flip_obs_joint_idx_mask], 0.0)
        qvel = data.qvel[6:] / MAX_JOINT_VELOCITIES
        qvel = np.append(qvel[flip_obs_joint_idx_mask], 0.0)
        previous_action = previous_action / 3.14
        previous_action = np.append(previous_action[flip_obs_joint_idx_mask], 0.0)
        qpos_qvel_previous_action = np.vstack([qpos, qvel, previous_action]).T.flatten()
        angular_velocity = data.qvel[3:6] / 10.0
        goal_velocity = np.array(VELOCITY_COMMAND)
        orientation_quat = R.from_quat([data.qpos[4], data.qpos[5], data.qpos[6], data.qpos[3]])
        projected_gravity = orientation_quat.inv().apply(np.array([0.0, 0.0, -1.0]))
        observation = np.concatenate([qpos_qvel_previous_action, angular_velocity, goal_velocity, projected_gravity])

        # Get the action from the policy
        observation = torch.tensor(observation, dtype=torch.float32).to(device)
        action = policy(observation).detach().cpu().numpy()
        previous_action = action
        scaled_action = ACTION_SCALING * action
        target_joint_positions = nominal_joint_position + scaled_action
        torques = P_GAIN * (target_joint_positions - data.qpos[7:]) - D_GAIN * data.qvel[6:]

        # Step simulation
        data.ctrl = torques
        mujoco.mj_step(model, data, SIMULATION_FREQUENCY_IN_HZ // CONTROL_FREQUENCY_IN_HZ)
        data.qvel[6:] = np.clip(data.qvel[6:], -MAX_JOINT_VELOCITIES, MAX_JOINT_VELOCITIES)
        viewer.render(data)

        # Compute the dense mass matrix using MuJoCo
        M = np.zeros((model.nv, model.nv))  # Pre-allocate the dense matrix
        mujoco.mj_forward(model, data) # Warning: don't forget to call mj_forward before mj_fullM
                                        # to update the dynamics state                      
        mujoco.mj_fullM(model, M, data.qM)
        
        # Compute the Coriolis matrix with pinocchio
        C_pino = pino.computeCoriolisMatrix(pino_model, pino_data, data.qpos, data.qvel) 
        # Compute the Gravitry vector with MuJoCo
        g = compute_gravity_forces(model, data)

        # Update the generalized momentum observer
        gt_gm = M @ data.qvel
        tau_input = np.hstack([np.zeros(6), torques])
        est_ext_tau, est_gm = gm_estimator.update(data.qvel, M, C_pino, g, tau_input)

        # Compute the inverse dynamics torques.
        mujoco.mj_forward(model, data)
        mujoco.mj_inverse(model, data)

        # The total joint torques (including gravity, Coriolis, and external forces)
        tau_total = data.qfrc_inverse.copy()
        
        # Compute the joint space external torques
        gt_ext_tau = tau_total - tau_input
        gt_ext_taus.append(gt_ext_tau.copy())
        est_ext_taus.append(est_ext_tau.copy())

        # Reset simulation if the robot's trunk is in collision
        trunk_collision = False
        for con_i in range(0, data.ncon):
            con = data.contact[con_i]
            if con.geom1 in trunk_geom_ids or con.geom2 in trunk_geom_ids:
                trunk_collision = True
                break
        if trunk_collision:
            data.qpos = model.key_qpos[0]
            data.qvel = np.zeros_like(data.qvel)
            mujoco.mj_forward(model, data)
            previous_action = np.zeros(12)
            viewer.render(data)
        steps += 1
        if steps > 200:
            break
        
