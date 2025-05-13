import mujoco
import mujoco.viewer
import numpy as np
import time
import pinocchio as pino
from contact_estimator import high_gain_based_observer, kalman_disturbance_observer
from utils.mujoco_dyn import mujocoDyn
from external_wrench import WrenchApplier
from controller import cartesian_impedance_nullspace
from utils.geom_visualizer import visualize_normal_arrow, reset_scene, visualize_mat_arrows
from utils.mesh_sampler import MeshSampler

# Simulation timestep in seconds.
dt: float = 0.002

def main() -> None:
    assert mujoco.__version__ >= "3.1.0", "Please upgrade to mujoco 3.1.0 or later."

    # Load the model and data.
    model = mujoco.MjModel.from_xml_path("kuka_iiwa_14/scene.xml")
    data = mujoco.MjData(model)

    model.opt.timestep = dt
    
    # Load the model and data with Pinocchio
    pino_model = pino.buildModelFromMJCF("kuka_iiwa_14/iiwa14.xml")
    pino_data = pino_model.createData()

    # Get the dof and actuator ids for the joints we wish to control. These are copied
    # from the XML file. Feel free to comment out some joints to see the effect on
    # the controller.
    joint_names = [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
    ]
    # dof_ids = np.array([model.joint(name).id for name in joint_names])
    actuator_ids = np.array([model.actuator(name).id for name in joint_names])

    # Initial joint configuration saved as a keyframe in the XML file.
    key_name = "home"
    key_id = model.key(key_name).id

    # Initialize the contact estimator
    gm_estimator = high_gain_based_observer(dt, "hg", model.nv)
    # gm_estimator = kalman_disturbance_observer(dt, model.nv)

    # Initialize a buffer for visualization of the estimated external forces and generalized momentum.
    est_ext_wrenches = []
    est_gms = []
    gt_ext_wrenches = []
    gt_gms = []
    est_ext_taus = []
    gt_ext_taus = []
    computed_ext_wrenchs = []
    
    # Generate a wrench profile
    apply_body_names = ["link6"]
    wrench_applier = WrenchApplier(model, data, "sine", time_stop=10.0, dt=dt, body_names=apply_body_names)
    mujoco_dyn = mujocoDyn(model, data)

    # Set up the mesh sampler.
    mesh_sampler = MeshSampler(model, data, False, robot_name="kuka_iiwa_14")

    with mujoco.viewer.launch_passive(
        model=model,
        data=data,
        # show_left_ui=False,
        # show_right_ui=False,
    ) as viewer:
        scene = viewer.user_scn
        ngeom_init = scene.ngeom
        
        # Reset the simulation.
        mujoco.mj_resetDataKeyframe(model, data, key_id)

        # Reset the free camera.
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Enable site frame visualization.
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
        while viewer.is_running():
            step_start = time.time()

            # # Apply external wrench to the end-effector.
            # applied_external_wrench, applied_positions, is_end = wrench_applier.apply_predefined_wrench()
            # reset_scene(scene, ngeom_init)
            # applied_external_forces = [applied_external_wrench[key][: 3] for key in applied_external_wrench.keys()]
            # applied_positions = [applied_positions[key] for key in applied_external_wrench.keys()]
            # if not is_end and np.linalg.norm(applied_external_forces[0]) > 1e-8:
            #     visualize_normal_arrow(scene, applied_positions, applied_external_forces)
                
            # Sample a mesh point and visualize it.
            mesh_id, geom_id, faces_center_local, normals_local, rot_mats, face_vertices_select = mesh_sampler.sample_body_pos_normal("link6", num_samples=5)
            geom_origin_pos_world = data.geom_xpos[geom_id]        # shape (3,)
            geom_origin_mat_world = data.geom_xmat[geom_id].reshape(3, 3)  # shape (3,3)
            reset_scene(scene, ngeom_init)
            visualize_mat_arrows(scene, geom_origin_pos_world, geom_origin_mat_world, faces_center_local, rot_mats)
                        
            # Compute the Coriolis matrix with pinocchio TODO: implement the computation for Coriolis matrix with only mujoco
            C_pino = pino.computeCoriolisMatrix(pino_model, pino_data, data.qpos, data.qvel) 
            # Compute all dynamic matrixes with MuJoCo
            g, M = mujoco_dyn.compute_all_forces()
            
            # Compute the control law.
            tau = cartesian_impedance_nullspace(model, data, dt)
            
            # Update the generalized momentum observer
            gt_gm = M @ data.qvel
            est_ext_tau, est_gm = gm_estimator.update(data.qvel, M, C_pino, g, tau)
            
            # Print out the external forces on the end-effector.
            ee_body_name = "attachment"
            ee_body_id = model.body(ee_body_name).id
            ext_wrench = data.xfrc_applied[ee_body_id]

            # Compute the body jacobian for the end-effector.
            jac_body = np.zeros((6, model.nv))
            mujoco.mj_jacBody(model, data, jac_body[:3], jac_body[3:], ee_body_id)
            est_ext_wrench = np.linalg.pinv(jac_body.T) @ est_ext_tau

            # Append the estimated external forces and generalized momentum to the buffer.
            est_ext_wrenches.append(est_ext_wrench.copy())
            est_gms.append(est_gm.copy())
            gt_ext_wrenches.append(ext_wrench.copy())
            gt_gms.append(gt_gm.copy())

            # Set the control signal and step the simulation.
            np.clip(tau, *model.actuator_ctrlrange.T, out=tau)
            data.ctrl[actuator_ids] = tau[actuator_ids]
            mujoco.mj_step(model, data)
            
            # Compute the inverse dynamics torques.
            mujoco.mj_forward(model, data)
            mujoco.mj_inverse(model, data)

            # The total joint torques (including gravity, Coriolis, and external forces)
            tau_total = data.qfrc_inverse.copy()
            
            # Compute the joint space external torques
            gt_ext_tau = tau_total - tau
            gt_ext_taus.append(gt_ext_tau.copy())
            est_ext_taus.append(est_ext_tau.copy())
            computed_ext_wrench = np.linalg.pinv(jac_body.T) @ gt_ext_tau
            computed_ext_wrenchs.append(computed_ext_wrench.copy())

            viewer.sync()
            time_until_next_step = dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    # Visualize the estimated external forces and generalized momentum.
    est_ext_wrenches = np.array(est_ext_wrenches)
    est_gms = np.array(est_gms)
    est_ext_taus = np.array(est_ext_taus)
    gt_ext_wrenches = np.array(gt_ext_wrenches)
    gt_gms = np.array(gt_gms)
    gt_ext_taus = np.array(gt_ext_taus)
    computed_ext_wrenchs = np.array(computed_ext_wrenchs)
    
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10, 5))
    plot_param = 611
    axs_name = ["fx", "fy", "fz", "mx", "my", "mz"]
    t = np.arange(len(est_ext_wrenches)) * dt
    for i in range(6):
        ax = fig.add_subplot(plot_param)
        ax.plot(t, est_ext_wrenches[:, i], label=f"Estimated External {axs_name[i]}")
        ax.plot(t, gt_ext_wrenches[:, i], label=f"Ground Truth External {axs_name[i]}")
        ax.plot(t, computed_ext_wrenchs[:, i], label=f"Computed External Torque {axs_name[i]}")
        ax.legend()
        plot_param += 1

    fig = plt.figure(figsize=(10, 5))
    plot_param = 711
    axs_name = ["gm1", "gm2", "gm3", "gm4", "gm5", "gm6", "gm7"]
    for i in range(7):
        ax = fig.add_subplot(plot_param)
        ax.plot(t, est_gms[:, i], label=f"Estimated Generalized Momentum {axs_name[i]}")
        ax.plot(t, gt_gms[:, i], label=f"Ground Truth Generalized Momentum {axs_name[i]}")
        ax.legend()
        plot_param += 1
    
    fig = plt.figure(figsize=(10, 5))
    plot_param = 711
    axs_name = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
    for i in range(7):
        ax = fig.add_subplot(plot_param)
        ax.plot(t, est_ext_taus[:, i], label=f"Estimated External Torque {axs_name[i]}")
        ax.plot(t, gt_ext_taus[:, i], label=f"Ground Truth External Torque {axs_name[i]}")
        ax.legend()
        plot_param += 1

    plt.show()


if __name__ == "__main__":
    main()