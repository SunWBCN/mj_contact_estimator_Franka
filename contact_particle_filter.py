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

def get_data_cpf_set(cpf_set, mjx_data):
    geom_poss_tmp = []
    rot_mats_geom_tmp = []
    com_poss_tmp = []
    rot_mats_com_tmp = []
    for cpf in cpf_set:
        geom_ids = cpf.get_geom_ids()
        particles_body_ids = cpf.get_particles_body_ids()
        geom_poss_world, rot_mats_geom_world, com_poss_world, rot_mats_com_world \
        = get_batch_contact_pos_rot(mjx_data, geom_ids, particles_body_ids)
        geom_poss_tmp.append(geom_poss_world)
        rot_mats_geom_tmp.append(rot_mats_geom_world)
        com_poss_tmp.append(com_poss_world)
        rot_mats_com_tmp.append(rot_mats_com_world)
    geom_poss_world = jnp.concatenate(geom_poss_tmp, axis=0)
    rot_mats_geom_world = jnp.concatenate(rot_mats_geom_tmp, axis=0)
    com_poss_world = jnp.concatenate(com_poss_tmp, axis=0)
    rot_mats_com_world = jnp.concatenate(rot_mats_com_tmp, axis=0)
    return geom_poss_world, rot_mats_geom_world, com_poss_world, rot_mats_com_world

def compute_site_ids(model, particles_link_names, search_body_names):
    counts = {name: 2 for name in search_body_names}
    particles_site_ids = []
    for body_name in particles_link_names:
        particle_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{body_name}_dummy_site{counts[body_name]}")
        counts[body_name] += 1
        particles_site_ids.append(particle_site_id)
    particles_site_ids = jnp.array(particles_site_ids)
    return particles_site_ids

def rest_cpf_id(n_contacts, cur_idx):
    # Compute the remaining contact particle filter IDs
    rest_ids = []
    for i in range(n_contacts):
        if i != cur_idx:
            rest_ids.append(i)
    return jnp.array(rest_ids)

def get_cpf_pos(cpf_set):
    """
    Get the positions of the particles in the contact particle filter set.
    
    Args:
        cpf_set: List of ContactParticleFilter objects.
    
    Returns:
        jnp.ndarray: Positions of the particles in the contact particle filter set.
    """
    particles_positions = []
    for cpf in cpf_set:
        particles_positions.append(cpf.get_particles_positions())
    return jnp.concatenate(particles_positions, axis=0)

def get_cpf_rot(cpf_set):
    """
    Get the rotations of the particles in the contact particle filter set.
    
    Args:
        cpf_set: List of ContactParticleFilter objects.
    
    Returns:
        jnp.ndarray: Rotations of the particles in the contact particle filter set.
    """
    particles_rotations = []
    for cpf in cpf_set:
        particles_rotations.append(cpf.get_particles_rotations())
    return jnp.concatenate(particles_rotations, axis=0)

class ContactParticleFilter:
    def __init__(self, model, data, n_particles=1000, robot_name="kuka_iiwa_14", search_body_names=["link7"], ext_f_norm=5.0,
                 importance_distribution_noise=0.01, measurement_noise=0.01):
        self.n_particles = n_particles
        self.sampler = mesh_sampler.MeshSampler(model, data, False, robot_name)
        self.search_body_names = search_body_names
        self.ext_f_norm = ext_f_norm
        self.importance_distribution_noise = importance_distribution_noise
        self.measurement_noise = measurement_noise
        self.model = model
        self.data = data
    
    def initialize_particles(self, search_body_names=None):
        randomseed = np.random.randint(0, 1000000)
        key = jax.random.PRNGKey(randomseed)
        if search_body_names is None:
            search_body_names = self.search_body_names
        
        self.sampler.update_sampling_space_global(search_body_names)
        particles_indexes = self.sampler.sample_indexes_global(search_body_names, self.n_particles, key=key)
        self.particles_indexes = particles_indexes
        self.weight = jnp.ones(self.n_particles) / self.n_particles
        
    def predict_particles(self, key, search_body_names=None):
        if search_body_names is not None:
            # self.sampler.update_sampling_space_jax(search_body_names)
            # self.search_body_names = search_body_names
            self.sampler.update_sampling_space_global(search_body_names)
            self.search_body_names = search_body_names
        
        # TODO: new code base on global indexing
        particles = self.get_particles_positions()  # Get the current particles positions
        
        # perturb the particles
        perturbed_particles = particles + jax.random.normal(key=key, shape=particles.shape) * self.importance_distribution_noise
        
        # find the nearest positions indexes
        nearest_contact_indexes = self.sampler.find_nearest_indexes(perturbed_particles, self.search_body_names)
        
        # update the indexes of the particles
        self.particles_indexes = nearest_contact_indexes

    def update_weights(self, epsilons):
        # Epsilon is the loss of a convex QP, use to compute the likelihood of the particles as weights
        # The loss is defined as the sum of square of the residuals
        weight_unnormalized = jnp.exp(-epsilons / (2 * self.measurement_noise**2))
        weight_normalized = weight_unnormalized / jnp.sum(weight_unnormalized)
        self.weight = weight_normalized
        
    def resample_particles(self, key, method='multinomial'):
        if method == "multinomial":
            indices = jax.random.choice(key, self.particles_indexes, shape=(self.n_particles, ), p=self.weight)
            self.particles_indexes = indices
            self.weight = jnp.ones(self.n_particles) / self.n_particles
        else:
            raise ValueError("Unknown resampling method: {}".format(method))

    def get_particles_data(self):
        return self.get_particles_data_from_indexes(self.particles_indexes)

    def get_particles_data_from_indexes(self, particles_indexes):
        particles, forces_normal, rot_mats_contact_geom, face_vertices_select, geom_ids, particles_link_names, mesh_ids = \
        self.sampler.get_data(particles_indexes)
        particles_site_ids = compute_site_ids(self.model, particles_link_names, self.search_body_names)   
        particles_body_ids = jnp.array([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name) for name in particles_link_names])
        forces_normal = forces_normal * self.ext_f_norm
        return particles, forces_normal, rot_mats_contact_geom, face_vertices_select, geom_ids, particles_link_names, \
                particles_site_ids, particles_body_ids

    def get_particles_center_data(self):
        # TODO: make sure the data is correct
        particles = self.get_particles_positions()
        particles_center = jnp.mean(particles, axis=0)
        nearest_particles_indexes = self.sampler.find_nearest_indexes(jnp.array([particles_center]), self.search_body_names)
        return self.get_particles_data_from_indexes(nearest_particles_indexes)
    
    def get_most_probable_particle_data(self):
        # Get the index of the particle with the highest weight
        max_weight_index = jnp.argmax(self.weight)
        # Get the corresponding particle index
        most_probable_particle_index = self.particles_indexes[max_weight_index]
        # Retrieve the data for that particle
        return self.get_particles_data_from_indexes(jnp.array([most_probable_particle_index]))
    
    def get_random_particle_data(self):
        key = jax.random.PRNGKey(np.random.randint(0, 1000000))
        random_index = jax.random.choice(key, self.particles_indexes, shape=(1,), p=self.weight)
        return self.get_particles_data_from_indexes(random_index)

    def get_particles_positions(self):
        return self.get_particles_data()[0]
    
    def get_geom_ids(self):
        return self.get_particles_data()[4]

    def get_particles_link_names(self):
        return self.get_particles_data()[5]
    
    def get_particles_body_ids(self):
        return self.get_particles_data()[7]
    
    def get_particles_site_ids(self):
        return self.get_particles_data()[6]
    
    def get_particles_rotations(self):
        return self.get_particles_data()[2]
    
    def get_particles_forces(self):
        return self.get_particles_data()[1]

def cpf_step(cpf_set, key, mjx_model, mjx_data, gt_ext_tau, batch_qp_solver, iters=20, particle_history=[], 
             average_errors=[], data_log=None, qp_loss=True, qp_solver=None, polyhedral_num=4):
    qpos = mjx_data.qpos
    
    # Prepare the ground truth data for validation
    if data_log is not None:
        target_ext_wrenches = jnp.hstack(jnp.array([data_log["ext_wrenches"][i][:3] for i in range(batch_qp_solver.n_contacts)]))
        contact_pos_target = data_log["contact_pos_target"]
        target_jacobian_contact = data_log["target_jacobian_contact"]
        target_normal_vecs = data_log["target_normal_vecs_geom"]
        target_geom_ids = data_log["target_geom_ids"]
        target_body_names = data_log["target_body_names"]
        target_body_ids = jnp.array([mujoco.mj_name2id(cpf_set[0].model, mujoco.mjtObj.mjOBJ_BODY, name) for name in target_body_names])
        target_global_idxes = data_log["target_global_indexes"]

        # stack the target jacobian
        target_jacobian_contact = [target_jacobian[:3, :] for target_jacobian in target_jacobian_contact]
        target_jacobian_contact = jnp.concatenate(target_jacobian_contact, axis=0) # shape (n_contacts * 3, nv)
        # target_jacobian_contact = target_jacobian_contact[0]
        target_jacobian_contact = jnp.tile(target_jacobian_contact, (cpf_set[0].n_particles, 1, 1)) # shape (iters, n_contacts * 3, nv)

        target_geom_poss_world, target_rot_mats_geom_world, target_com_poss_world, target_rot_mats_com_world \
        = get_batch_contact_pos_rot(mjx_data, target_geom_ids, target_body_ids)
        target_normal_vecs_world = batch_vecs_world_coor(target_normal_vecs, target_rot_mats_geom_world, target_geom_poss_world)
        target_friction_cone_basises = batch_friction_cone_basis(target_normal_vecs_world, mu=batch_qp_solver.mu, k=polyhedral_num)
        # target_friction_cone_basises = jnp.tile(target_friction_cone_basises, (target_friction_cone_basises.shape[0], cpf_set[0].n_particles, 1, 1)) # shape (iters, n_contacts, k, 3)
        # repeat the target friction cone basises for each particle, so that it can be used in the optimization, the shape is (n_contacts, n_particles, k, 3)        
        target_friction_cone_basises = jnp.tile(target_friction_cone_basises[:, None, :, :], (1, cpf_set[0].n_particles, 1, 1)) # shape (n_contacts, n_particles, k, 3)

        # QP based loss
        if qp_solver is not None:
            target_jacobian_contact_ = target_jacobian_contact[0]
            target_friction_cone_basises_ = target_friction_cone_basises[:, 0, :, :]
            target_param_qp, target_residual_qp, target_errors = qp_solver.solve(np.array(target_jacobian_contact_), 
                                                                                   np.array(gt_ext_tau),
                                                                                   np.array(target_friction_cone_basises_)
                                                                                   )
        else:
            target_params, target_residual, target_errors = batch_qp_solver.solve(np.array(target_jacobian_contact),
                                                                                  np.array(gt_ext_tau),
                                                                                  np.array(target_friction_cone_basises))

    for i in range(iters):
        for j in range(len(cpf_set)):
            cpf = cpf_set[j]
            key, subkey = jax.random.split(key)
            cpf.predict_particles(subkey) # The perturbations are not based on geodesic distances, so could jump from link7 to link6
           
            # Retrieve cartesian space positions for jacobian computation
            particles, forces_normal, rots_mat_contact, face_vertices_select, geom_ids, particles_link_names, \
            particles_site_ids, particles_body_ids = cpf.get_particles_data()
            geom_poss_world, rot_mats_geom_world, com_poss_world, rot_mats_com_world \
            = get_batch_contact_pos_rot(mjx_data, geom_ids, particles_body_ids)
            
            # Compute the jacobians
            jacobians, contact_poss_coms, rot_mats_contact_com, quats = \
            compute_batch_site_jac_pipeline(rots_mat_contact, particles, rot_mats_geom_world,
                                            rot_mats_com_world, geom_poss_world, com_poss_world, 
                                            mjx_model, mjx_data, qpos, particles_site_ids)
            # wait till the jacobian is computed
            jax.block_until_ready(jacobians)
            
            # Compute the friction cone basis vectors
            normal_vecs_world = batch_vecs_world_coor(forces_normal, rot_mats_geom_world, geom_poss_world)
            
            # for the indexes of the contact particle filter that equal to the target contact indexes, set mu to 200.0
            # else set mu to 0.5
            friction_cone_basises = batch_friction_cone_basis(normal_vecs_world, mu=batch_qp_solver.mu, k=polyhedral_num)
            
            # TODO: Retrieve jacobian for the rest particle sets
            rest_ids = rest_cpf_id(len(cpf_set), j)
            if len(rest_ids) > 0:
                jacobians_rest_tmp = []
                friction_cone_basises_rest_tmp = []
                for k in rest_ids:
                    cpf_rest = cpf_set[k]
                    particles_rest, forces_normal_rest, rots_mat_contact_rest, face_vertices_select_rest, geom_ids_rest, \
                    particles_link_names_rest, particles_site_ids_rest, particles_body_ids_rest \
                    = cpf_rest.get_random_particle_data()
                    # particles_rest, forces_normal_rest, rots_mat_contact_rest, face_vertices_select_rest, geom_ids_rest, \
                    # particles_link_names_rest, particles_site_ids_rest, particles_body_ids_rest \
                    # = cpf_rest.get_most_probable_particle_data()

                    geom_poss_world_rest, rot_mats_geom_world_rest, com_poss_world_rest, rot_mats_com_world_rest \
                    = get_batch_contact_pos_rot(mjx_data, geom_ids_rest, particles_body_ids_rest)
                    jacobians_rest, contact_poss_coms_rest, rot_mats_contact_com_rest, quats_rest = \
                    compute_batch_site_jac_pipeline(rots_mat_contact_rest, particles_rest, rot_mats_geom_world_rest,
                                                    rot_mats_com_world_rest, geom_poss_world_rest, com_poss_world_rest, 
                                                    mjx_model, mjx_data, qpos, particles_site_ids_rest)
                    jax.block_until_ready(jacobians_rest)
                    normal_vecs_world_rest = batch_vecs_world_coor(forces_normal_rest, rot_mats_geom_world_rest, geom_poss_world_rest)
                    friction_cone_basises_rest = batch_friction_cone_basis(normal_vecs_world_rest, mu=batch_qp_solver.mu, k=polyhedral_num)
                    # Append the jacobians and friction cone basises for the rest particle sets
                    jacobians_rest_tmp.append(jacobians_rest)
                    friction_cone_basises_rest_tmp.append(friction_cone_basises_rest)
                    
                # Brocast the jacobians for the rest particle sets
                jacobians_rest = [jnp.tile(jacobian, (jacobians.shape[0], 1, 1)) for jacobian in jacobians_rest_tmp]
                jacobians_rest.insert(0, jacobians)  # Insert the current jacobian at the beginning
                
                # Concatenate the jacobians for the current particle set and the rest particle sets
                jacobians = jnp.concatenate(jacobians_rest, axis=1)

                # TODO: Remember to check the shape of friction_cone_basises
                friction_cone_basises_rest = [jnp.tile(friction_cone_basis, (friction_cone_basises.shape[0], 1, 1)) for friction_cone_basis in friction_cone_basises_rest_tmp]
                friction_cone_basises_rest.insert(0, friction_cone_basises)  #
                friction_cone_basises = jnp.stack(friction_cone_basises_rest, axis=0)  # shape (n_contacts, n_particles, k, 3)
            else:
                # Add an extra dimension to the friction cone basises to fit the optimization framework
                friction_cone_basises = friction_cone_basises.reshape((1, friction_cone_basises.shape[0], friction_cone_basises.shape[1], friction_cone_basises.shape[2]))
            
            if not qp_loss:
                contact_pos_target = data_log["contact_pos_target"][j]
                # measurement_noise = cpf.measurement_noise
                # contact_pos_target_measurement = contact_pos_target + jax.random.normal(key=key, shape=contact_pos_target.shape) * measurement_noise
                particle_positions = cpf.get_particles_positions()  # Get the current particles positions
                errors = jnp.linalg.norm(particle_positions - contact_pos_target, axis=1)
                errors = np.array(errors)
            else:
                # params, errors = solve_batch_qp(jacobians, gt_ext_tau)
                if qp_solver is not None:
                    params, errors = [], []
                    for i in range(len(jacobians)):
                        param, residual, error = qp_solver.solve(np.array(jacobians[i]), np.array(gt_ext_tau),
                                                       np.array(friction_cone_basises[:, i, :, :]))
                        params.append(param)
                        errors.append(error)
                    errors = np.array(errors)
                else:
                    params, residual, errors = batch_qp_solver.solve(np.array(jacobians), np.array(gt_ext_tau),
                                                                     np.array(friction_cone_basises),)
            # print(sorted(errors), target_errors)
            cpf.update_weights(errors)
            key, subkey = jax.random.split(key)        
            cpf.resample_particles(subkey, method='multinomial')
            
            particles = cpf.get_particles_positions()  # Get the current particles positions
            particle_history.append(particles)
            average_errors.append(jnp.mean(errors))

    # manage particle sets

    # geom_ids = cpf.get_geom_ids()
    # particles_body_ids = cpf.get_particles_body_ids()
    # geom_poss_world, rot_mats_geom_world, com_poss_world, rot_mats_com_world \
    # = get_batch_contact_pos_rot(mjx_data, geom_ids, particles_body_ids)
    geom_poss_world, rot_mats_geom_world, com_poss_world, rot_mats_com_world = get_data_cpf_set(cpf_set, mjx_data)
    return geom_poss_world, rot_mats_geom_world, com_poss_world, rot_mats_com_world
    
def visualize_particles(fig, axes, particle_history, average_errors, contact_pos_target):
    # Create animation
    particle_axes = axes[0]  # Top row for particle evolution
    error_ax = axes[1, 1]    # Bottom center for error evolution
    particle_center_ax = axes[1, 0] # Bottom left for error of particle center
    minimum_error_ax = axes[1, 2] # Bottom right for minimum error

    x_limit = 0.2
    y_limit = 0.2
    colors = ['red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']

    def update(frame):
        particles = particle_history[frame]
        particles_center = jnp.mean(jnp.array(particle_history[:frame + 1]), axis=1)
        for ax in particle_axes:
            ax.clear()

        # XY Plane
        particle_axes[0].scatter(particles[:, 0], particles[:, 1], c='blue', alpha=0.5)
        for i in range(len(contact_pos_target)):
            particle_axes[0].scatter(contact_pos_target[i][0], contact_pos_target[i][1], c=colors[i], s=100)
        particle_axes[0].set_title(f"XY Plane (Iteration {frame+1})")
        particle_axes[0].set_xlabel("X")
        particle_axes[0].set_ylabel("Y")
        particle_axes[0].set_xlim(-x_limit, x_limit)
        particle_axes[0].set_ylim(-y_limit, y_limit)

        # XZ Plane
        particle_axes[1].scatter(particles[:, 0], particles[:, 2], c='blue', alpha=0.5)
        for i in range(len(contact_pos_target)):
            particle_axes[1].scatter(contact_pos_target[i][0], contact_pos_target[i][2], c=colors[i], s=100)
        particle_axes[1].set_title(f"XZ Plane (Iteration {frame+1})")
        particle_axes[1].set_xlabel("X")
        particle_axes[1].set_ylabel("Z")
        particle_axes[1].set_xlim(-x_limit, x_limit)
        particle_axes[1].set_ylim(-y_limit, y_limit)

        # YZ Plane
        particle_axes[2].scatter(particles[:, 1], particles[:, 2], c='blue', alpha=0.5)
        for i in range(len(contact_pos_target)):
            particle_axes[2].scatter(contact_pos_target[i][1], contact_pos_target[i][2], c=colors[i], s=100)
        particle_axes[2].set_title(f"YZ Plane (Iteration {frame+1})")
        particle_axes[2].set_xlabel("Y")
        particle_axes[2].set_ylabel("Z")
        particle_axes[2].set_xlim(-x_limit, x_limit)
        particle_axes[2].set_ylim(-y_limit, y_limit)

        # # Update error plot
        # error_ax.clear()
        # error_ax.plot(range(1, frame + 2), average_errors[:frame + 1], marker='o', linestyle='-', color='blue')
        # error_ax.set_title("Evolution of Average Errors")
        # error_ax.set_xlabel("Iteration")
        # error_ax.set_ylabel("Average Error")
        # error_ax.grid(True)
        
        # # Update the error plot for particle center
        # particle_center_ax.clear()
        # particles_center_error = jnp.linalg.norm(particles_center - contact_pos_target, axis=1)
        # particle_center_ax.plot(range(1, frame + 2), particles_center_error, marker='o', linestyle='-', color='green')
        # particle_center_ax.set_title("Error of Particle Center")
        # particle_center_ax.set_xlabel("Iteration")
        # particle_center_ax.set_ylabel("Error")
        # particle_center_ax.set_ylim(0, 0.1)
        # particle_center_ax.grid(True)
        
        # # Update the minimum error plot
        # minimum_error_ax.clear()
        # min_error = jnp.min(jnp.array(average_errors[:frame + 1]))
        # minimum_error_ax.plot(range(1, frame + 2), [min_error] * (frame + 1), marker='o', linestyle='-', color='orange')
        # minimum_error_ax.set_title("Minimum Error")
        # minimum_error_ax.set_xlabel("Iteration")
        # minimum_error_ax.set_ylabel("Minimum Error")
        # minimum_error_ax.set_ylim(0, 1)
        # minimum_error_ax.grid(True)

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