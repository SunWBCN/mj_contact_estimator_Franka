import mujoco
import numpy as np
import time
import pinocchio as pino
from contact_estimator import high_gain_based_observer, kalman_disturbance_observer
from utils.mujoco_dyn import mujocoDyn
from external_wrench import WrenchApplier
from controller import *
from utils.geom_visualizer import visualize_normal_arrow, reset_scene, visualize_mat_arrows
from utils.mesh_sampler import MeshSampler
from utils.robot_transform import *
from utils.qp_solver import BatchQPSolver, QPSolver, QPNonlinearSolver
from contact_particle_filter import ContactParticleFilter, ExplorationParticleFilter, cpf_step, cpf_step_explo, \
                                    visualize_particles, get_batch_contact_pos_rot
from mujoco import mjx
import jax
from jax import numpy as jnp
from utils.mujoco_viewer import MujocoViewer
from utils.mjx_functions import *
import matplotlib.pyplot as plt
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

def main() -> None:
    assert mujoco.__version__ >= "3.1.0", "Please upgrade to mujoco 3.1.0 or later."

    # Load the model and data.
    model = mujoco.MjModel.from_xml_path("kuka_iiwa_14/scene.xml")
    data = mujoco.MjData(model)
    # Reset the model and data.
    mujoco.mj_resetData(model, data)
    # Simulation timestep in seconds.
    dt: float = 0.002
    model.opt.timestep = dt
    
    # Initial joint configuration saved as a keyframe in the XML file.
    key_name = "home"
    key_id = model.key(key_name).id
    
    # Reset the simulation.
    mujoco.mj_resetDataKeyframe(model, data, key_id)

    # Update model and data.
    mujoco.mj_forward(model, data)
    
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
    actuator_ids = jnp.array([model.actuator(name).id for name in joint_names])

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
    target_body_name = "link7"
    mesh_sampler = MeshSampler(model, data, False, robot_name="kuka_iiwa_14")
    import time
    start_time = time.time()
    randomseed = np.random.randint(0, 10000)
    randomseed = 0
    mesh_id, geom_id, contact_poss_geom, normal_vecs_geom, rot_mats_contact_geom, face_vertices_select = \
    mesh_sampler.sample_body_pos_normal_jax(target_body_name, num_samples=1, key=jax.random.PRNGKey(randomseed))
    print("Time taken to sample mesh point:", time.time() - start_time)
    contact_particle_gt = [contact_poss_geom[0].copy(), normal_vecs_geom[0].copy(),
                           rot_mats_contact_geom[0].copy(), face_vertices_select[0].copy()]  # For testing the contact particle filter
    
    # Fixed value for testing
    contact_pos_target = contact_poss_geom[0]
    ext_f_norm = 20.0
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

    viewer = MujocoViewer(model, data)
    # Reset the free camera.
    mujoco.mjv_defaultFreeCamera(model, viewer.cam)
    scene = viewer.scn
    ngeom_init = scene.ngeom
    
    # Enable site frame visualization.
    viewer.vopt.frame = mujoco.mjtFrame.mjFRAME_SITE
    site_names = [f"{target_body_name}_dummy_site{i}" for i in range(1, 2)]
    for site_name in site_names:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id > -1:
            model.site_size[site_id] * 0.2
    
    # Initialize the JAX model and data for mjx.
    mjx_model = mjx.put_model(model)
    mjx_data = mjx.put_data(model, data)
    use_mjx = True
    
    # Prepare the jax model and contact particle filter
    if use_mjx:
        mjx_data = warmup_jit(mjx_model, mjx_data, model, data, target_body_name=target_body_name)
        data = mjx.get_data(model, mjx_data)
        # TODO: retrieve the site_ids after initialized the particles, the site ids are aligned with each particle
        # and use search_body_names instead of target_body_name
        search_body_names = [f"link{i}" for i in range(1, 8)]  # Search for particles in all links
        cpf = ExplorationParticleFilter(model=model, data=data, n_particles=100, robot_name="kuka_iiwa_14",
                                        search_body_names=search_body_names, 
                                        ext_f_norm=ext_f_norm,
                                        importance_distribution_noise=0.02, measurement_noise=0.02, 
                                        contact_particle_gt=contact_particle_gt,
                                        n_contacts=1)
        cpf.initialize_particles(init_body_name="link7")
    
        key = jax.random.PRNGKey(0)
        # Particle history for visualization
        particle_history = []
        average_errors = []
        gt_ext_tau = jnp.zeros(model.nv)
        update_gt_ext_tau = False
        mu = 200.0
        polyhedral_num = 4
        qp_solver = QPSolver(n_joints=model.nv, n_contacts=1, mu=mu, polyhedral_num=polyhedral_num)
        batch_qp_solver = BatchQPSolver(n_joints=model.nv, n_contacts=1, n_qps=cpf.n_particles, mu=mu,
                                        polyhedral_num=polyhedral_num)
        qp_nonlinear_solver = QPNonlinearSolver(n_joints=model.nv, n_contacts=1, mu=mu)
    
    carry = {"carry_particles": {}, "carry_mat_arrows": {}, "carry_normal_arrows": {}}
    
    while viewer.is_alive:
        step_start = time.time()
        
        # Reset the visualization scene.
        reset_scene(scene, ngeom_init)
            
        # Retrieve the geometry position and rotation matrix.
        geom_pos_world = data.geom_xpos[geom_id]        # shape (3,)
        rot_mat_geom_world = data.geom_xmat[geom_id].reshape(3, 3)        # shape (3,3)
        
        carry["carry_mat_arrows"]["geom_origin_pos_world"] = geom_pos_world
        carry["carry_mat_arrows"]["geom_origin_mat_world"] = rot_mat_geom_world
        carry["carry_mat_arrows"]["arrows_pos_local"] = contact_poss_geom
        carry["carry_mat_arrows"]["arrows_rot_mat_local"] = rot_mats_contact_geom
        
        # Compute the equivalent external forces to apply on the body.
        ## Add a function that computes the equivalent external forces with jax.numpy
        mesh_sampler.update_model_data(model, data)
        equi_ext_f_poss, equi_ext_wrenches, jacobian_bodies, contact_poss_world, ext_wrenches, jacobian_contacts = \
        mesh_sampler.compute_equivalent_wrenches(contact_poss_geom, rot_mats_contact_geom, normal_vecs_geom,
                                                 target_body_name, geom_id, ext_f_norm)

        # Visualize the equivalent external forces
        equi_ext_fs = np.array(equi_ext_wrenches)[:, :3]
        jac_body = jacobian_bodies[0]
        jacs_body.append(jac_body.copy())
        gt_equi_ext_wrenches.append(equi_ext_wrenches[0].copy())
        gt_ext_wrenches.append(ext_wrenches[0].copy())
        jac_contact = jacobian_contacts[0]
        jacs_contact.append(jac_contact.copy())
        
        carry["carry_normal_arrows"]["arrows_pos_world"] = equi_ext_f_poss
        carry["carry_normal_arrows"]["arrows_vec_world"] = equi_ext_fs

        # Apply the wrench profile to the specified body names.
        if not use_mjx:
            if applied_ext_f:
                if applied_predefined_wrench:
                    if applied_times < 10000:
                        wrench_applier.apply_predefined_wrench()
                else:
                    wrench_applier.apply_wrench(equi_ext_wrenches[0], target_body_name)
                applied_times += 1
        else:
            if applied_ext_f:
                body_id = model.body(target_body_name).id
                mjx_data = apply_wrench_mjx(mjx_data, equi_ext_wrenches[0], body_id)
        
        # Compute the Coriolis matrix with pinocchio TODO: implement the computation for Coriolis matrix with only mujoco
        C_pino = pino.computeCoriolisMatrix(pino_model, pino_data, data.qpos, data.qvel)
        
        # Compute all dynamic matrixes with MuJoCo.
        g, M = mujoco_dyn.compute_all_forces(model, data)
        
        # Compute the control law.
        tau = cartesian_impedance_nullspace_jax(model, data, dt)
        
        # Update the generalized momentum observer.
        gt_gm = M @ data.qvel
        est_ext_tau, est_gm = gm_estimator.update(data.qvel, M, C_pino, g, tau)

        # Append the estimated external forces and generalized momentum to the buffer.
        est_gms.append(est_gm.copy())
        gt_gms.append(gt_gm.copy())

        # Set the control signal and step the simulation.
        if not use_mjx:
            np.clip(tau, *model.actuator_ctrlrange.T, out=tau)
            data.ctrl[actuator_ids] = tau[actuator_ids]
            mujoco.mj_step(model, data)
        else:
            # jit functions
            tau = jnp.clip(tau, *model.actuator_ctrlrange.T)
            mjx_data = update_ctrl_mjx(mjx_data, tau)
            jax.block_until_ready(mjx_data)
            mjx_data = jit_step(mjx_model, mjx_data)
            jax.block_until_ready(mjx_data)
            mjx_data = jit_forward(mjx_model, mjx_data)
            jax.block_until_ready(mjx_data)
            data = mjx.get_data(model, mjx_data)
            jax.block_until_ready(data)
            mujoco.mj_forward(model, data)
            data_log = {"contact_pos_target": contact_pos_target, "jacobian_contact": jacobian_contacts[0], 
                        "ext_wrenches": ext_wrenches[0]}
            if update_gt_ext_tau:
                geom_poss_world, rot_mats_geom_world, com_poss_world, rot_mats_com_world \
                = cpf_step_explo(cpf, key, mjx_model, mjx_data, gt_ext_tau=gt_ext_tau, particle_history=particle_history,
                                 average_errors=average_errors, iters=10,
                                 data_log=data_log, batch_qp_solver=batch_qp_solver, polyhedral_num=polyhedral_num)            
            else:
                geom_poss_world, rot_mats_geom_world, com_poss_world, rot_mats_com_world \
                = get_batch_contact_pos_rot(mjx_data, cpf.geom_ids, cpf.particles_body_ids)
                
            particles_geom_pos_world = geom_poss_world
            particles_rot_mat_geom_world = rot_mats_geom_world
            carry["carry_particles"]["geom_origin_poss_world"] = particles_geom_pos_world # TODO: need to update in a batch manner
            carry["carry_particles"]["geom_origin_mats_world"] = particles_rot_mat_geom_world # TODO: need to update in a batch manner
            carry["carry_particles"]["particles_pos_geom"] = cpf.particles
            carry["carry_particles"]["particles_mat_geom"] = cpf.rots_mat_contact
                    
        # # Compute the inverse dynamics torques.
        # mujoco.mj_forward(model, data)
        # mujoco.mj_inverse(model, data)

        # # The total joint torques (including gravity, Coriolis, and external forces)
        # tau_total = data.qfrc_inverse.copy()
        # mjx_data = jit_inverse(mjx_model, mjx_data) if use_mjx else mujoco.mj_inverse(model, data)
        # tau_total = mjx_data.qfrc_inverse.copy() if use_mjx else data.qfrc_inverse.copy()
        
        # Compute the joint space external torques
        # gt_ext_tau_ = tau_total - tau
        gt_ext_tau = jnp.dot(jacobian_contacts[0].T, ext_wrenches[0])
        if not update_gt_ext_tau:
            update_gt_ext_tau = True
        gt_ext_taus.append(gt_ext_tau.copy())
        est_ext_taus.append(est_ext_tau.copy())

        viewer.update_data(data)
        viewer.render(carry=carry, vis_arrows=True, vis_particles=True)
        time_until_next_step = dt - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

    if use_mjx:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        visualize_particles(fig, axes, particle_history, average_errors, contact_pos_target)

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
    
    # # Save the data to a file
    # data_dict = {"gt_ext_wrenches": gt_ext_wrenches, "gt_ext_taus": gt_ext_taus, 
    #              "jacs_site": jacs_site, "jacs_site": jacs_site, "jacs_body": jacs_body,
    #              "jacs_contact": jacs_contact, "gt_ext_wrenches": gt_ext_wrenches,
    #              "gt_equi_ext_wrenches": gt_equi_ext_wrenches,}
    # np.savez("data.npz", **data_dict)
    # exit(0)
    
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