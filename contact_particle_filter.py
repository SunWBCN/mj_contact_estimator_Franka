import numpy as np
from utils import mesh_sampler
import jax.numpy as jnp
from mujoco import mjx
from pathlib import Path
import mujoco
from batch_jacobian import compute_batch_site_jac_pipeline
from batch_qp import solve_batch_qp
import jax
import time
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

def get_contact_pos_rot(mjx_data, geom_id, sample_body_id):
    # Retrieve rotation matrix and position of the center of mass (COM)
    geom_pos_world = mjx_data.geom_xpos[geom_id]  # shape (3,)
    rot_mat_geom_world = mjx_data.geom_xmat[geom_id].reshape(3, 3)
    com_pos_world = mjx_data.xpos[sample_body_id]
    rot_mat_com_world = mjx_data.xmat[sample_body_id].reshape(3, 3)
    return geom_pos_world, rot_mat_geom_world, com_pos_world, rot_mat_com_world
    
@jax.jit
def slice_with_indices(array, indices):
    return jax.vmap(lambda i: array[i])(indices)

# TODO: convert the code to use jnp instead of np
class ContactParticleFilter:
    def __init__(self, model, data, n_particles=1000, robot_name="kuka_iiwa_14", sample_body_name="link7", ext_f_norm=5.0,
                 importance_distribution_noise=0.01, measurement_noise=0.01):
        self.n_particles = n_particles
        self.sampler = mesh_sampler.MeshSampler(model, data, False, robot_name)
        self.sample_body_name = sample_body_name
        self.ext_f_norm = ext_f_norm
        self.initialize_particles()
        self.importance_distribution_noise = importance_distribution_noise
        self.measurement_noise = measurement_noise
        self.model = model
        self.data = data
        self.sample_body_id = model.body(sample_body_name).id
    
    def initialize_particles(self):
        mesh_id, geom_id, contact_poss_geom, normal_vecs_geom, rots_mat_contact_geom, face_vertices_select = \
        self.sampler.sample_body_pos_normal_jax(self.sample_body_name, num_samples=self.n_particles)
        self.particles = contact_poss_geom
        self.forces_normal = normal_vecs_geom * self.ext_f_norm
        self.rots_mat_contact = rots_mat_contact_geom
        self.face_vertices_select = face_vertices_select
        self.geom_id = geom_id
        self.weight = jnp.ones(self.n_particles) / self.n_particles
        return self.particles, self.forces_normal, self.rots_mat_contact, self.face_vertices_select, self.geom_id
        
    def predict_particles(self, key):
        # pertubed the particles and project them to the contact surface
        perturbed_particles = self.particles + jax.random.normal(key=key, shape=self.particles.shape) * self.importance_distribution_noise
        nearest_positions, normals, rot_mats, face_vertices_select = self.sampler.compute_nearest_positions_jax(perturbed_particles, self.sample_body_name)
        self.particles = nearest_positions
        self.forces_normal = normals * self.ext_f_norm
        self.rots_mat_contact = rot_mats
        self.face_vertices_select = face_vertices_select
        return self.particles, self.forces_normal, self.rots_mat_contact, self.face_vertices_select, self.geom_id

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
            self.weight = jnp.ones(self.n_particles) / self.n_particles
        else:
            raise ValueError("Unknown resampling method: {}".format(method))
        return self.particles, self.forces_normal, self.rots_mat_contact, self.face_vertices_select, self.geom_id

def cpf_step(cpf, key, mjx_model, mjx_data, gt_ext_tau, site_ids, iters=20, particle_history=[], average_errors=[]):
    geom_pos_world, rot_mat_geom_world, com_pos_world, rot_mat_com_world = get_contact_pos_rot(mjx_data, cpf.geom_id,
                                                                                               cpf.sample_body_id)
    qpos = mjx_data.qpos
    for i in range(iters):
        key, subkey = jax.random.split(key)
        particles, forces_normal, rot_mats_contact_geom, face_vertices_select, geom_id = cpf.predict_particles(subkey)
        jacobians, contact_poss_coms, rot_mats_contact_com, quats = \
        compute_batch_site_jac_pipeline(rot_mats_contact_geom, particles, rot_mat_geom_world,
                                        rot_mat_com_world, geom_pos_world, com_pos_world, 
                                        mjx_model, mjx_data, qpos, site_ids)
        params, errors = solve_batch_qp(jacobians, gt_ext_tau)
        cpf.update_weights(errors)
        key, subkey = jax.random.split(key)
        particles, forces_normal, rot_mats_contact_geom, face_vertices_select, geom_id = cpf.resample_particles(subkey, method='multinomial')
        particle_history.append(particles)
        average_errors.append(jnp.mean(errors))
    
def visualize_particles(fig, axes, particle_history, average_errors, contact_pos_target):
    # Create animation
    particle_axes = axes[0]  # Top row for particle evolution
    error_ax = axes[1, 1]    # Bottom center for error evolution

    x_limit = 0.06
    y_limit = 0.06

    def update(frame):
        particles = particle_history[frame]
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