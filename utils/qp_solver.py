import cvxpy as cp
import numpy as np

class QPSolver:
    def __init__(self, njoints, n_contacts, mu=0.5):
        """
        Initialize the QP solver with parameterized variables.

        Args:
            njoints (int): Number of joints in the robot.
            n_contacts (int): Number of contact points.
            mu (float): Friction coefficient.
        """
        self.n_joints = njoints
        self.n_contacts = n_contacts
        self.mu = mu

        # Define the contact force variable
        self.f_c = cp.Variable(3 * self.n_contacts)  # [fx1, fy1, fz1, ...]

        # Define parameters for Jacobian and external torques
        self.Jc = cp.Parameter((3, self.n_joints))  # Jacobian matrix
        self.tau_ext = cp.Parameter(self.n_joints)  # External torques

        # Define the objective function
        self.objective = cp.Minimize(cp.sum_squares(self.Jc.T @ self.f_c - self.tau_ext))

        # Define the constraints
        self.constraints = []
        for i in range(self.n_contacts):
            fx = self.f_c[3 * i + 0]
            fy = self.f_c[3 * i + 1]
            fz = self.f_c[3 * i + 2]

            # Unilateral constraint
            self.constraints.append(fz >= 0)

            # Friction cone (pyramid approximation)
            self.constraints.append(cp.abs(fx) <= self.mu * fz)
            self.constraints.append(cp.abs(fy) <= self.mu * fz)

        # Define the problem
        self.prob = cp.Problem(self.objective, self.constraints)

    def solve(self, Jc, tau_ext):
        """
        Solve the QP problem with updated parameters.

        Args:
            Jc (np.ndarray): Jacobian matrix of size (3, n_joints).
            tau_ext (np.ndarray): External torques of size (n_joints,).

        Returns:
            np.ndarray: Estimated contact forces of size (3 * n_contacts,).
            float: Loss value of the optimization problem.
        """
        # Update the parameters
        self.Jc.value = Jc
        self.tau_ext.value = tau_ext

        # Solve the problem using OSQP
        self.prob.solve(solver=cp.OSQP, warm_start=True)

        # Return the results
        return self.f_c.value, self.objective.value
    
if __name__ == "__main__":
    import numpy as np
    data_dict = np.load("data.npz")
    print(data_dict.keys())

    index = 90
    Jc = data_dict["jacs_body"][index]
    # Jc = data_dict["jacs_site"][index]
    ext_tau = data_dict["gt_ext_taus"][index]
    ext_wrench = data_dict["gt_ext_wrenches"][index]
    gt_ext_tau = Jc.T @ ext_wrench
    print(gt_ext_tau, ext_tau, ext_wrench)

    # Parameters
    n_joints = 7
    n_contacts = 1
    mu = 0.5  # friction coefficient
    
    # Initialize the solver
    qp_solver = QPSolver(n_joints, n_contacts, mu)
    
    # Solve the QP problem
    Jc = Jc[:3, :]
    import time
    start_time = time.time()
    f_c, objective_value = qp_solver.solve(Jc, gt_ext_tau)
    end_time = time.time()
    print("Time taken to solve QP:", end_time - start_time)
    print("Contact forces:", f_c)
    print("Objective value:", objective_value)