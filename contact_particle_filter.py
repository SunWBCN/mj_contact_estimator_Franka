import numpy as np
from utils import mesh_sampler
import jax.numpy as jnp
from mujoco import mjx
from pathlib import Path
import mujoco
from utils.batch_jacobian import compute_batch_site_jac_pipeline
import jax
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

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
    
@jax.jit
def slice_with_indices(array, indices):
    return jax.vmap(lambda i: array[i])(indices)

@jax.jit
def vec_world_coor(vec_local, rot_mat_local_world, pos_local_world):
    """
    Convert vectors from local coordinates to world coordinates.
    
    Args:
        vecs_local: Local vectors (3).
        rot_mat_local_world: Rotation matrix from local to world coordinates (3, 3).
        pos_local_world: Position vector in world coordinates (3,).
    
    Returns:
        vecs_world: Vectors in world coordinates (N, 3).
    """
    return pos_local_world + jnp.dot(rot_mat_local_world, vec_local)

@jax.jit
def batch_vecs_world_coor(vecs_local, rot_mats_local_world, poss_local_world):
    """
    Convert vectors from local coordinates to world coordinates in batch.
    
    Args:
        vecs_local: Local vectors (N, 3).
        rot_mat_local_world: Rotation matrix from local to world coordinates (3, 3).
        pos_local_world: Position vector in world coordinates (3,).
    
    Returns:
        vecs_world: Vectors in world coordinates (N, 3).
    """
    return jax.vmap(vec_world_coor, in_axes=(0, 0, 0))(vecs_local, rot_mats_local_world, poss_local_world)

def friction_cone_basis_jax(n, mu, k=4):
    n = n / jnp.linalg.norm(n)
    theta = jnp.arctan(mu)
    
    # Tangent vectors
    # if jnp.allclose(n[:2], 0):
    #     t1 = jnp.array([1, 0, 0])
    # else:
    t1 = jnp.array([-n[1], n[0], 0])
    t1 /= jnp.linalg.norm(t1)
    t2 = jnp.cross(n, t1)
    
    phi = jnp.linspace(0, 2 * jnp.pi, k, endpoint=False)

    # Compute basis vectors using broadcasting
    F_basis = jnp.cos(theta) * n[:, None] + jnp.sin(theta) * (
        jnp.cos(phi)[None, :] * t1[:, None] + jnp.sin(phi)[None, :] * t2[:, None]
    )
    return F_basis

friction_cone_basis_jax = jax.jit(friction_cone_basis_jax, static_argnums=(2,))

@jax.jit
def batch_friction_cone_basis_jax(ns, mu, k=4):
    """
    Batch version of friction_cone_basis_jax.
    
    Args:
        ns: Normals (N, 3).
        mu: Friction coefficient.
        k: Number of basis vectors.
    
    Returns:
        F_basis: Friction cone basis vectors (N, 3, k).
    """
    return jax.vmap(friction_cone_basis_jax, in_axes=(0, None, None))(ns, mu, k)

def batch_friction_cone_basis(normals, mu, k=8):
    """
    Compute the friction cone basis vectors for a batch of normal vectors.

    Args:
        normals (np.ndarray): Normal vectors (N, 3).
        mu (float): Friction coefficient.
        k (int): Number of basis vectors.

    Returns:
        np.ndarray: Friction cone basis vectors (N, k, 3).
    """
    # Normalize the normal vectors
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

    # Compute tangent vectors
    t1 = np.cross(normals, np.array([0, 0, 1]))
    t1 = t1 / np.linalg.norm(t1, axis=1, keepdims=True)
    t2 = np.cross(normals, t1)

    # Compute angles for basis vectors
    phi = np.linspace(0, 2 * np.pi, k, endpoint=False)

    # Compute basis vectors using broadcasting
    theta = np.arctan(mu)
    F_basis = (
        np.cos(theta) * normals[:, None, :]
        + np.sin(theta) * (
            np.cos(phi)[None, :, None] * t1[:, None, :]
            + np.sin(phi)[None, :, None] * t2[:, None, :]
        )
    )

    return F_basis

def compute_site_ids(model, particles_link_names, search_body_names):
    counts = {name: 2 for name in search_body_names}
    particles_site_ids = []
    for body_name in particles_link_names:
        particle_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{body_name}_dummy_site{counts[body_name]}")
        counts[body_name] += 1
        particles_site_ids.append(particle_site_id)
    particles_site_ids = jnp.array(particles_site_ids)
    return particles_site_ids
    

class ContactParticleFilter:
    def __init__(self, model, data, n_particles=1000, robot_name="kuka_iiwa_14", search_body_names=["link7"], ext_f_norm=5.0,
                 importance_distribution_noise=0.01, measurement_noise=0.01, contact_particle_gt=None):
        self.n_particles = n_particles
        self.sampler = mesh_sampler.MeshSampler(model, data, False, robot_name)
        self.search_body_names = search_body_names
        self.ext_f_norm = ext_f_norm
        self.importance_distribution_noise = importance_distribution_noise
        self.measurement_noise = measurement_noise
        self.model = model
        self.data = data
        self.initialize_particles(contact_particle_gt)
    
    def initialize_particles(self, contact_particle_gt=None, search_body_names=None):
        randomseed = np.random.randint(0, 1000000)
        key = jax.random.PRNGKey(randomseed)
        self.sampler.update_sampling_space_jax(self.search_body_names) # Don't forget to update the sampling space
        
        if search_body_names is None:
            search_body_names = self.search_body_names
        
        mesh_ids, geom_ids, contact_poss_geom, normal_vecs_geom, rots_mat_contact_geom, face_vertices_select, \
        particles_link_names = \
        self.sampler.sample_bodies_pos_normal_jax(body_names=search_body_names, num_samples=self.n_particles, key=key)
        
        self.particles = contact_poss_geom
        self.forces_normal = normal_vecs_geom * self.ext_f_norm
        self.rots_mat_contact = rots_mat_contact_geom
        self.face_vertices_select = face_vertices_select
        self.geom_ids = geom_ids
        self.weight = jnp.ones(self.n_particles) / self.n_particles
        
        self.particles_link_names = particles_link_names
        self.particles_site_ids = compute_site_ids(self.model, particles_link_names, search_body_names)
        self.particles_body_ids = jnp.array([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name) for name in particles_link_names])
        
        if contact_particle_gt is not None:
            contact_pos_geom_gt, normal_vec_geom_gt, rot_mat_contact_gt, face_vertice_select_gt = contact_particle_gt
            self.particles = self.particles.at[0].set(jnp.array(contact_pos_geom_gt))
            self.forces_normal = self.forces_normal.at[0].set(jnp.array(normal_vec_geom_gt * self.ext_f_norm))
            self.rots_mat_contact = self.rots_mat_contact.at[0].set(jnp.array(rot_mat_contact_gt))
            self.face_vertices_select = self.face_vertices_select.at[0].set(jnp.array(face_vertice_select_gt))
            
        self.particle_center = jnp.mean(self.particles, axis=0)
        return self.particles, self.forces_normal, self.rots_mat_contact, self.face_vertices_select, self.geom_ids, \
               self.particles_site_ids
        
    def predict_particles(self, key, search_body_names=None):
        if search_body_names is not None:
            self.sampler.update_sampling_space_jax(search_body_names)
            self.search_body_names = search_body_names
        
        perturbed_particles = self.particles + jax.random.normal(key=key, shape=self.particles.shape) * self.importance_distribution_noise
        # TODO: update the computed neartest position approach for multiple links and return the geom_ids and particles_site_ids
        nearest_positions, normals, rot_mats, face_vertices_select, geom_ids, particles_link_names \
        = self.sampler.compute_nearest_positions_bodies_jax(perturbed_particles, self.search_body_names)
        self.particles = nearest_positions
        self.forces_normal = normals * self.ext_f_norm
        self.rots_mat_contact = rot_mats
        self.face_vertices_select = face_vertices_select
        
        # TODO : update the geom_ids and particles_site_ids
        self.geom_ids = geom_ids
        self.particles_site_ids = compute_site_ids(self.model, particles_link_names, self.search_body_names)        
        self.particles_link_names = particles_link_names
        
        self.particles_body_ids = jnp.array([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name) for name in particles_link_names])
        return self.particles, self.forces_normal, self.rots_mat_contact, self.face_vertices_select, self.geom_ids

    def update_weights(self, epsilons):
        # Epsilon is the loss of a convex QP, use to compute the likelihood of the particles as weights
        # The loss is defined as the sum of square of the residuals
        weight_unnormalized = jnp.exp(-epsilons / (2 * self.measurement_noise**2))
        weight_normalized = weight_unnormalized / jnp.sum(weight_unnormalized)
        self.weight = weight_normalized
        
    def resample_particles(self, key, method='multinomial'):
        if method == "multinomial":
            # Resample the particles based on the weights
            indices = jax.random.choice(key, jnp.arange(self.n_particles), shape=(self.n_particles, ), p=self.weight)
            self.particles = slice_with_indices(self.particles, indices)
            self.forces_normal = slice_with_indices(self.forces_normal, indices)
            self.rots_mat_contact = slice_with_indices(self.rots_mat_contact, indices)
            self.face_vertices_select = slice_with_indices(self.face_vertices_select, indices)
            # TODO: update the geom_ids and particles_site_ids
            self.geom_ids = slice_with_indices(self.geom_ids, indices)
            self.particles_site_ids = slice_with_indices(self.particles_site_ids, indices)
            self.particles_body_ids = slice_with_indices(self.particles_body_ids, indices)
            self.weight = jnp.ones(self.n_particles) / self.n_particles
            self.particle_center = jnp.mean(self.particles, axis=0)
            self.particles_link_names = np.array(self.particles_link_names)[indices]
        else:
            raise ValueError("Unknown resampling method: {}".format(method))
        return self.particles, self.forces_normal, self.rots_mat_contact, self.face_vertices_select, self.geom_ids

def cpf_step(cpf, key, mjx_model, mjx_data, gt_ext_tau, batch_qp_solver, iters=20, particle_history=[], 
             average_errors=[], data_log=None, qp_loss=True, qp_solver=None, polyhedral_num=4):
    # TODO:batch computing the geom and com positions in th world frame
    qpos = mjx_data.qpos
    for i in range(iters):
        key, subkey = jax.random.split(key)
        # TODO: perturbed the particles and return geom_ids, site_ids of the particles
        particles, forces_normal, rot_mats_contact_geom, face_vertices_select, geom_ids = cpf.predict_particles(subkey) # The perturbations are not based on geodesic distances, so could jump from link7 to link6
        
        geom_poss_world, rot_mats_geom_world, com_poss_world, rot_mats_com_world \
        = get_batch_contact_pos_rot(mjx_data, cpf.geom_ids, cpf.particles_body_ids)
        
        # TODO: change the computation to use batch rot_mat_geom_world, geom_pos_pos_world, com_pos_world etc.
        jacobians, contact_poss_coms, rot_mats_contact_com, quats = \
        compute_batch_site_jac_pipeline(rot_mats_contact_geom, particles, rot_mats_geom_world,
                                        rot_mats_com_world, geom_poss_world, com_poss_world, 
                                        mjx_model, mjx_data, qpos, cpf.particles_site_ids)
        
        # wait till the jacobian is computed
        jax.block_until_ready(jacobians)
        
        # Compute the contact position basis
        # TODO: change the computation to use batch rot_mat_geom_world and geom_pos_world
        normal_vecs_world = batch_vecs_world_coor(forces_normal, rot_mats_geom_world, geom_poss_world)
        friction_cone_basises = batch_friction_cone_basis(normal_vecs_world, mu=batch_qp_solver.mu, k=polyhedral_num)
        
        if not qp_loss:
            contact_pos_target = data_log["contact_pos_target"]
            measurement_noise = cpf.measurement_noise
            contact_pos_target_measurement = contact_pos_target + jax.random.normal(key=key, shape=contact_pos_target.shape) * measurement_noise
            errors = jnp.linalg.norm(particles - contact_pos_target_measurement, axis=1)
        else:
            # params, errors = solve_batch_qp(jacobians, gt_ext_tau)
            if qp_solver is not None:
                params, errors = [], []
                for i in range(len(jacobians)):
                    param, error = qp_solver.solve(np.array(jacobians[i]), np.array(gt_ext_tau))
                    params.append(param)
                    errors.append(error)
                errors = jnp.array(errors).flatten()
            else:
                params, residual, errors = batch_qp_solver.solve(np.array(jacobians), np.array(gt_ext_tau),
                                                                 np.array(friction_cone_basises),)
        # # Print the corresponding errors to each link
        # link7_index = np.where(np.array(cpf.particles_link_names) == "link7")[0]
        # link6_index = np.where(np.array(cpf.particles_link_names) == "link6")[0]
        # link7_errors = errors[link7_index]
        # link6_errors = errors[link6_index]
        # print("Sorted Errors for link7:", jnp.sort(link7_errors))
        # print("Sorted Errors for link6:", jnp.sort(link6_errors))
        
        cpf.update_weights(errors)
        key, subkey = jax.random.split(key)        
        particles, forces_normal, rot_mats_contact_geom, face_vertices_select, geom_ids \
        = cpf.resample_particles(subkey, method='multinomial')
        
        particle_history.append(particles)
        average_errors.append(jnp.mean(errors))

    geom_poss_world, rot_mats_geom_world, com_poss_world, rot_mats_com_world \
    = get_batch_contact_pos_rot(mjx_data, cpf.geom_ids, cpf.particles_body_ids)
    return geom_poss_world, rot_mats_geom_world, com_poss_world, rot_mats_com_world
    
def visualize_particles(fig, axes, particle_history, average_errors, contact_pos_target):
    # Create animation
    particle_axes = axes[0]  # Top row for particle evolution
    error_ax = axes[1, 1]    # Bottom center for error evolution
    particle_center_ax = axes[1, 0] # Bottom left for error of particle center
    minimum_error_ax = axes[1, 2] # Bottom right for minimum error

    x_limit = 0.2
    y_limit = 0.2

    def update(frame):
        particles = particle_history[frame]
        particles_center = jnp.mean(jnp.array(particle_history[:frame + 1]), axis=1)
        for ax in particle_axes:
            ax.clear()

        # XY Plane
        particle_axes[0].scatter(particles[:, 0], particles[:, 1], c='blue', alpha=0.5)
        particle_axes[0].scatter(contact_pos_target[0], contact_pos_target[1], c='red', s=100)
        particle_axes[0].set_title(f"XY Plane (Iteration {frame+1})")
        particle_axes[0].set_xlabel("X")
        particle_axes[0].set_ylabel("Y")
        particle_axes[0].set_xlim(-x_limit, x_limit)
        particle_axes[0].set_ylim(-y_limit, y_limit)

        # XZ Plane
        particle_axes[1].scatter(particles[:, 0], particles[:, 2], c='blue', alpha=0.5)
        particle_axes[1].scatter(contact_pos_target[0], contact_pos_target[2], c='red', s=100)
        particle_axes[1].set_title(f"XZ Plane (Iteration {frame+1})")
        particle_axes[1].set_xlabel("X")
        particle_axes[1].set_ylabel("Z")
        particle_axes[1].set_xlim(-x_limit, x_limit)
        particle_axes[1].set_ylim(-y_limit, y_limit)

        # YZ Plane
        particle_axes[2].scatter(particles[:, 1], particles[:, 2], c='blue', alpha=0.5)
        particle_axes[2].scatter(contact_pos_target[1], contact_pos_target[2], c='red', s=100)
        particle_axes[2].set_title(f"YZ Plane (Iteration {frame+1})")
        particle_axes[2].set_xlabel("Y")
        particle_axes[2].set_ylabel("Z")
        particle_axes[2].set_xlim(-x_limit, x_limit)
        particle_axes[2].set_ylim(-y_limit, y_limit)

        # Update error plot
        error_ax.clear()
        error_ax.plot(range(1, frame + 2), average_errors[:frame + 1], marker='o', linestyle='-', color='blue')
        error_ax.set_title("Evolution of Average Errors")
        error_ax.set_xlabel("Iteration")
        error_ax.set_ylabel("Average Error")
        error_ax.grid(True)
        
        # Update the error plot for particle center
        particle_center_ax.clear()
        particles_center_error = jnp.linalg.norm(particles_center - contact_pos_target, axis=1)
        particle_center_ax.plot(range(1, frame + 2), particles_center_error, marker='o', linestyle='-', color='green')
        particle_center_ax.set_title("Error of Particle Center")
        particle_center_ax.set_xlabel("Iteration")
        particle_center_ax.set_ylabel("Error")
        particle_center_ax.set_ylim(0, 0.1)
        particle_center_ax.grid(True)
        
        # Update the minimum error plot
        minimum_error_ax.clear()
        min_error = jnp.min(jnp.array(average_errors[:frame + 1]))
        minimum_error_ax.plot(range(1, frame + 2), [min_error] * (frame + 1), marker='o', linestyle='-', color='orange')
        minimum_error_ax.set_title("Minimum Error")
        minimum_error_ax.set_xlabel("Iteration")
        minimum_error_ax.set_ylabel("Minimum Error")
        minimum_error_ax.set_ylim(0, 1)
        minimum_error_ax.grid(True)

    fps = 60
    interval = 1000 / fps  # milliseconds
    ani = FuncAnimation(fig, update, frames=len(particle_history), interval=interval, repeat=True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # TODO: Design test for the contact particle filter
    import numpy as np
    data_dict = np.load("./utils/data.npz")
    
    contact_poss_geom = jnp.array([jnp.array([ 0.00244227, 0.03933891, -0.02926769])])
    normal_vecs_geom = jnp.array([jnp.array([-0.75993156, -0.47094217, 0.44801494])])
    rots_mat_contact_geom = jnp.array([
                            jnp.array(
                                        [
                                        [-0.59587467, -0.25967506, -0.7599316],
                                        [0.78010535, -0.41187733, -0.4709422],
                                        [-0.19070667, -0.8734563, 0.44801497]
                                        ]
                                     )
                            ])
    contact_pos_target = contact_poss_geom[0]
    
    # load data and conert them into jax array
    gt_ext_taus = []
    Jcs = []
    for i in range(100, 200):
        index = i
        Jc = data_dict["jacs_contact"][index]
        ext_tau = data_dict["gt_ext_taus"][index]
        ext_wrench = data_dict["gt_ext_wrenches"][index]
        gt_ext_tau = Jc.T @ ext_wrench
        gt_ext_taus.append(gt_ext_tau)
        
        # Solve the QP problem
        Jc = Jc[:3, :]
        Jcs.append(Jc)
    
    # Convert to numpy arrays
    Jcs = jnp.array(Jcs)
    gt_ext_taus = jnp.array(gt_ext_taus)
    
    # TODO: contact particle filter initialization
    robot_name = "kuka_iiwa_14"
    sample_body_name = "link7"
    xml_path = (Path(__file__).resolve().parent / f"{robot_name}/scene.xml").as_posix()
    model = mujoco.MjModel.from_xml_path(f"{xml_path}")
    data = mujoco.MjData(model)

    mujoco.mj_resetData(model, data)
    mjx_model = mjx.put_model(model)
    mjx_data = mjx.put_data(model, data)
    
    cpf = ContactParticleFilter(model=model, data=data, n_particles=100, robot_name="kuka_iiwa_14", sample_body_name=sample_body_name, ext_f_norm=5.0,
                                importance_distribution_noise=0.005, measurement_noise=0.005)
    cpf.initialize_particles()
    iters = 20
    # qpos = jnp.zeros(model.nq)
    sample_body_name = cpf.sample_body_name
    site_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{sample_body_name}_dummy_site{i}") for i in range(cpf.n_particles)]
    site_ids = jnp.array(site_ids)  # shape (n_particles, )
    site_ids = jax.device_put(site_ids, device=jax.devices()[0])
    
    # update one step of mjx_model and mjx_data
    jit_step = jax.jit(mjx.step)
    mjx_data = jit_step(mjx_model, mjx_data)
    key = jax.random.PRNGKey(0)

    # Particle history for visualization
    particle_history = []
    average_errors = []

    # Run the CPF step
    for i in range(10):
        cpf_step(cpf, key, mjx_model, mjx_data, gt_ext_tau=gt_ext_taus[0], site_ids=site_ids, iters=iters, particle_history=particle_history, average_errors=average_errors)
    print(len(particle_history), len(average_errors))

    # Create combined figure for animation and error plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    visualize_particles(fig, axes, particle_history, average_errors, contact_pos_target)