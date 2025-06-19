import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import jax.numpy as jnp
import jax

def friction_cone_basis(n, mu, k=8):
    n = n / np.linalg.norm(n)
    theta = np.arctan(mu)
    
    # Tangent vectors
    if np.allclose(n[:2], 0):
        t1 = np.array([1, 0, 0])
    else:
        t1 = np.array([-n[1], n[0], 0])
        t1 /= np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    
    basis = []
    for i in range(k):
        phi = 2 * np.pi * i / k
        vec = np.cos(theta) * n + np.sin(theta) * (np.cos(phi) * t1 + np.sin(phi) * t2)
        basis.append(vec)
    F_basis = np.column_stack(basis)
    return F_basis

def friction_cone_basis_jax(n, mu, k=4):
    n = n / jnp.linalg.norm(n)
    theta = jnp.arctan(mu)
    
    # # Tangent vectors
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

if __name__ == "__main__":
    # Parameters
    n = np.array([1, 1, 1])
    mu = 0.5
    k = 4

    # Get basis vectors
    # F_basis = np.array(friction_cone_basis_jax(n, mu, k))
    F_basises = batch_friction_cone_basis(jnp.array([n, n, n, n, n]), mu, k)

    F_basis = np.array(F_basises[3]).T  # Convert JAX array to NumPy array

    # Plotting
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the normal vector (red)
    ax.quiver(0, 0, 0, n[0], n[1], n[2], color='r', length=1.0, normalize=True, label='Normal Vector')

    # Plot basis vectors (blue)
    for i in range(k):
        vec = F_basis[:, i]
        ax.quiver(0, 0, 0, vec[0], vec[1], vec[2], color='b', length=1.0, normalize=True, label=f'Basis Vector {i+1}')

    # Set equal aspect ratio
    ax.set_box_aspect([1,1,1])

    # Set limits
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([0, 1.2])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Friction Cone Basis Vectors (mu={mu})')

    ax.legend()
    plt.show()
