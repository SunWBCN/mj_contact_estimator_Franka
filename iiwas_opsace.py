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
    # Reset the model and data.
    mujoco.mj_resetData(model, data)
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

    # Initialize a buffer for visualization of the estimated external forces and generalized momentum.
    est_ext_wrenches = []
    est_gms = []
    gt_ext_wrenches = []
    gt_gms = []
    est_ext_taus = []
    gt_ext_taus = []
    computed_gt_ext_taus = []
    qp_est_ext_wrenches = []
    jacs_site = []
    jacs_body = []
    jacs_contact = []
    gt_equi_ext_wrenches = []
    
    # Generate a wrench profile
    apply_body_names = ["attachment"]
    wrench_applier = WrenchApplier(model, data, "sine", time_stop=10.0, dt=dt, body_names=apply_body_names)
    mujoco_dyn = mujocoDyn(model, data)

    # Set up the mesh sampler, sample a mesh point.
    sample_body_name = "link7"
    mesh_sampler = MeshSampler(model, data, False, robot_name="kuka_iiwa_14")
    import time
    start_time = time.time()
    mesh_id, geom_id, contact_poss_geom, normal_vecs_geom, rots_mat_contact_geom, face_vertices_select = \
    mesh_sampler.sample_body_pos_normal(sample_body_name, num_samples=1)
    print("Time taken to sample mesh point:", time.time() - start_time)
    
    # Fixed value for testing
    geom_id = 60
    contact_poss_geom = np.array([np.array([ 0.00244227, 0.03933891, -0.02926769])])
    normal_vecs_geom = np.array([np.array([-0.75993156, -0.47094217, 0.44801494])])
    rots_mat_contact_geom = np.array([
                            np.array(
                                        [
                                        [-0.59587467, -0.25967506, -0.7599316],
                                        [0.78010535, -0.41187733, -0.4709422],
                                        [-0.19070667, -0.8734563, 0.44801497]
                                        ]
                                     )
                            ])
    ext_f_norm = 5.0
    
    applied_times = 0
    applied_ext_f = True
    applied_predefined_wrench = False

    # Settings for visualization
    arrow_length = 0.1 * ext_f_norm
    transparent = True
    if transparent:
        # Set transparency for all geometries
        for i in range(model.ngeom):
            model.geom_rgba[i, 3] = 0.5  # Set alpha (transparency) to 50%

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

        # Update model and data.
        mujoco.mj_forward(model, data)

        # Enable site frame visualization.
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
        
        while viewer.is_running():
            step_start = time.time()
            
            # Reset the visualization scene.
            reset_scene(scene, ngeom_init)
                
            # Retrieve the geometry position and rotation matrix.
            geom_pos_world = data.geom_xpos[geom_id]        # shape (3,)
            rot_mat_geom_world = data.geom_xmat[geom_id].reshape(3, 3)        # shape (3,3)
            visualize_mat_arrows(scene, geom_pos_world, rot_mat_geom_world, contact_poss_geom, rots_mat_contact_geom, arrow_length=arrow_length)

            # Compute the equivalent external forces to apply on the body.
            equi_ext_f_poss, equi_ext_wrenches, jacobian_bodies, contact_poss_world, ext_wrenches, jacobian_contacts = \
            mesh_sampler.compute_equivalent_wrenches(contact_poss_geom, rots_mat_contact_geom, normal_vecs_geom, sample_body_name, 
                                                        geom_id, ext_f_norm)

            # Visualize the equivalent external forces
            equi_ext_fs = np.array(equi_ext_wrenches)[:, :3]
            jac_body = jacobian_bodies[0]
            jacs_body.append(jac_body.copy())
            gt_equi_ext_wrenches.append(equi_ext_wrenches[0].copy())
            gt_ext_wrenches.append(ext_wrenches[0].copy())
            jac_contact = jacobian_contacts[0]
            jacs_contact.append(jac_contact.copy())
            
            visualize_normal_arrow(scene, equi_ext_f_poss, equi_ext_fs, rgba=np.array([0.0, 1.0, 0.0, 1.0]), arrow_length=arrow_length)

            # Apply the wrench profile to the specified body names.
            if applied_ext_f:
                if applied_predefined_wrench:
                    if applied_times < 10000:
                        wrench_applier.apply_predefined_wrench()
                else:
                    wrench_applier.apply_wrench(equi_ext_wrenches[0], sample_body_name)
                applied_times += 1
            
            # Compute the Coriolis matrix with pinocchio TODO: implement the computation for Coriolis matrix with only mujoco
            C_pino = pino.computeCoriolisMatrix(pino_model, pino_data, data.qpos, data.qvel)
            
            # Compute all dynamic matrixes with MuJoCo.
            g, M = mujoco_dyn.compute_all_forces()
            
            # Compute the control law.
            tau = cartesian_impedance_nullspace(model, data, dt)
            
            # Update the generalized momentum observer.
            gt_gm = M @ data.qvel
            est_ext_tau, est_gm = gm_estimator.update(data.qvel, M, C_pino, g, tau)

            # Append the estimated external forces and generalized momentum to the buffer.
            est_gms.append(est_gm.copy())
            gt_gms.append(gt_gm.copy())

            # Set the control signal and step the simulation.
            np.clip(tau, *model.actuator_ctrlrange.T, out=tau)
            data.ctrl[actuator_ids] = tau[actuator_ids]
            mujoco.mj_step(model, data)
            
            # Compute the inverse dynamics torques.
            # mujoco.mj_forward(model, data)
            mujoco.mj_inverse(model, data)

            # The total joint torques (including gravity, Coriolis, and external forces)
            tau_total = data.qfrc_inverse.copy()
            
            # Compute the joint space external torques
            gt_ext_tau = tau_total - tau
            gt_ext_taus.append(gt_ext_tau.copy())
            est_ext_taus.append(est_ext_tau.copy())

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
    qp_est_ext_wrenches = np.array(qp_est_ext_wrenches)
    computed_gt_ext_taus = np.array(computed_gt_ext_taus)
    gt_ext_wrenches = np.array(gt_ext_wrenches)
    
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10, 5))
    plot_param = 711
    axs_name = ["gm1", "gm2", "gm3", "gm4", "gm5", "gm6", "gm7"]
    t = np.arange(len(est_gms)) * dt
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
        # ax.plot(t, computed_gt_ext_taus[:, i], label=f"Computed External Torque {axs_name[i]}")
        ax.legend()
        plot_param += 1

    plt.show()

if __name__ == "__main__":
    main()