import numpy as np
from utils import mesh_sampler
import jax.numpy as jnp

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
    
    def initialize_particles(self):
        mesh_id, geom_id, contact_poss_geom, normal_vecs_geom, rots_mat_contact_geom, face_vertices_select = \
        self.sampler.sample_body_pos_normal(self.sample_body_name, num_samples=self.n_particles)
        self.particles = contact_poss_geom
        self.forces_normal = normal_vecs_geom * self.ext_f_norm
        self.rots_mat_contact = rots_mat_contact_geom
        self.face_vertices_select = face_vertices_select
        self.geom_id = geom_id
        self.weight = np.ones(self.n_particles) / self.n_particles
        return self.particles, self.forces_normal, self.rots_mat_contact, self.face_vertices_select, self.geom_id
        
    def predict_particles(self):
        # pertubed the particles and project them to the contact surface
        perturbed_particles = self.particles + np.random.normal(loc=0, scale=self.importance_distribution_noise, size=self.particles.shape)
        nearest_positions, normals, rot_mats, face_vertices_select = self.sampler.compute_nearest_positions(perturbed_particles, self.sample_body_name)
        self.particles = nearest_positions
        self.forces_normal = normals * self.ext_f_norm
        self.rots_mat_contact = rot_mats
        self.face_vertices_select = face_vertices_select
        return self.particles, self.forces_normal, self.rots_mat_contact, self.face_vertices_select, self.geom_id

    def update_weights(self, epsilons):
        # Epsilon is the loss of a convex QP, use to compute the likelihood of the particles as weights
        # The loss is defined as the sum of square of the residuals
        weight_unnormalized = np.exp(-epsilons / (2 * self.measurement_noise**2))
        weight_normalized = weight_unnormalized / np.sum(weight_unnormalized)
        self.weight = weight_normalized
        
    def resample_particles(self, method='multinomial'):
        if method == "multinomial":
            # Resample the particles based on the weights
            indices = np.random.choice(np.arange(self.n_particles), size=self.n_particles, p=self.weight)
            self.particles = self.particles[indices]
            self.forces_normal = self.forces_normal[indices]
            self.rots_mat_contact = self.rots_mat_contact[indices]
            self.face_vertices_select = self.face_vertices_select[indices]
            self.weight.fill(1.0 / self.n_particles)
        else:
            raise ValueError("Unknown resampling method: {}".format(method))
        return self.particles, self.forces_normal, self.rots_mat_contact, self.face_vertices_select, self.geom_id
    
if __name__ == "__main__":
    # Run test for the contact particle filter
    pass