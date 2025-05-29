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
    
class BatchQPSolver:
    def __init__(self, n_joints, n_contacts, n_qps, mu=0.5):
        """
        Initialize the batch QP solver.

        Args:
            n_joints (int): Number of joints in the robot.
            n_contacts (int): Number of contact points.
            n_qps (int): Number of QPs to solve simultaneously.
            mu (float): Friction coefficient.
        """
        self.n_joints = n_joints
        self.n_contacts = n_contacts
        self.n_qps = n_qps
        self.mu = mu

        # Define the contact force variable for all QPs
        self.f_c = cp.Variable((self.n_qps, 3 * self.n_contacts))  # [fx1, fy1, fz1, ...]

        # Define parameters for Jacobians and external torques
        self.Jc = cp.Parameter((self.n_qps, 3 * self.n_contacts, self.n_joints))  # Jacobian matrices for all QPs
        self.tau_ext = cp.Parameter((self.n_qps, self.n_joints))  # External torques for all QPs

        # Define the objective function (sum of squared residuals for all QPs)
        residuals = [self.Jc[i, :, :].T @ self.f_c[i, :] - self.tau_ext[i, :] for i in range(self.n_qps)]
        self.objective = cp.Minimize(cp.sum([cp.sum_squares(res) for res in residuals]))

        # Define the constraints for all QPs
        self.constraints = []
        for i in range(self.n_qps):
            for j in range(self.n_contacts):
                fx = self.f_c[i, 3 * j + 0]
                fy = self.f_c[i, 3 * j + 1]
                fz = self.f_c[i, 3 * j + 2]

                # Unilateral constraint
                self.constraints.append(fz >= 0)

                # Friction cone (pyramid approximation)
                self.constraints.append(cp.abs(fx) <= self.mu * fz)
                self.constraints.append(cp.abs(fy) <= self.mu * fz)

        # Define the problem
        self.prob = cp.Problem(self.objective, self.constraints)

    def solve(self, Jc_batch, tau_ext_batch):
        """
        Solve the batch QP problem with updated parameters.

        Args:
            Jc_batch (np.ndarray): Batch of Jacobian matrices of size (3, n_joints, n_qps).
            tau_ext_batch (np.ndarray): Batch of external torques of size (n_joints, n_qps).

        Returns:
            np.ndarray: Estimated contact forces of size (3 * n_contacts, n_qps).
            float: Loss value of the optimization problem.
        """
        # Update the parameters
        self.Jc.value = Jc_batch
        self.tau_ext.value = tau_ext_batch

        # Solve the problem
        self.prob.solve(solver=cp.OSQP, warm_start=True)

        # Return the results
        return self.f_c.value, self.objective.value
    
class BatchQPSolverMatrix:
    def __init__(self, n_joints, n_contacts, n_qps, mu=0.5):
        """
        Initialize the batch QP solver.

        Args:
            n_joints (int): Number of joints in the robot.
            n_contacts (int): Number of contact points.
            n_qps (int): Number of QPs to solve simultaneously.
            mu (float): Friction coefficient.
        """
        self.n_joints = n_joints
        self.n_contacts = n_contacts
        self.n_qps = n_qps
        self.mu = mu

        # Define the contact force variable for all QPs
        self.f_c = cp.Variable((self.n_qps * self.n_contacts * 3))  # [fx1, fy1, fz1, ...]

        # Define parameters for Jacobians and external torques
        self.Jc = cp.Parameter((self.n_qps * self.n_contacts * 3, self.n_qps * self.n_joints))  # Jacobian matrices for all QPs
        self.tau_ext = cp.Parameter(self.n_qps * self.n_joints)  # External torques for all QPs
        
        # Define the objective function (sum of squared residuals for all QPs)
        residuals = self.Jc.T @ self.f_c - self.tau_ext
        self.objective = cp.Minimize(cp.sum_squares(residuals))
        
        # Define the constraints for all QPs
        self.constraints = []
        for i in range(self.n_qps):
            start_idx = i * self.n_contacts * 3
            for j in range(self.n_contacts):
                fx = self.f_c[start_idx + 3 * j + 0]
                fy = self.f_c[start_idx + 3 * j + 1]
                fz = self.f_c[start_idx + 3 * j + 2]

                # Unilateral constraint
                self.constraints.append(fz >= 0)

                # Friction cone (pyramid approximation)
                self.constraints.append(cp.abs(fx) <= self.mu * fz)
                self.constraints.append(cp.abs(fy) <= self.mu * fz)
        
        # Define the problem
        self.prob = cp.Problem(self.objective, self.constraints)
                
    def solve(self, Jc_batch, tau_ext_batch):
        """
        Solve the batch QP problem with updated parameters.

        Args:
            Jc_batch (np.ndarray): Batch of Jacobian matrices of size (3, n_joints, n_qps).
            tau_ext_batch (np.ndarray): Batch of external torques of size (n_joints, n_qps).

        Returns:
            np.ndarray: Estimated contact forces of size (3 * n_contacts, n_qps).
            float: Loss value of the optimization problem.
        """
        # Update the parameters
        self.Jc.value = Jc_batch
        self.tau_ext.value = tau_ext_batch

        # Solve the problem
        self.prob.solve(solver=cp.OSQP, warm_start=True)

        # Return the results
        return self.f_c.value, self.objective.value
            
                
if __name__ == "__main__":
    import numpy as np
    data_dict = np.load("data.npz")
    print(data_dict.keys())
    import time
    start_time = time.time()
    
    # Parameters
    n_joints = 7
    n_contacts = 1
    mu = 0.5  # friction coefficient
    n_qps = 100
    
    # Initialize the solver
    qp_solver = QPSolver(n_joints, n_contacts, mu)
    gt_ext_taus = []
    fcs = []
    Jcs = []
    objective_values = []
    for i in range(100, 100 + n_qps):
        index = i
        Jc = data_dict["jacs_contact"][index]
        ext_tau = data_dict["gt_ext_taus"][index]
        ext_wrench = data_dict["gt_ext_wrenches"][index]
        gt_ext_tau = Jc.T @ ext_wrench
        gt_ext_taus.append(gt_ext_tau)
        
        # Solve the QP problem
        Jc = Jc[:3, :]
        Jcs.append(Jc)

        f_c, objective_value = qp_solver.solve(Jc, gt_ext_tau)
        objective_values.append(objective_value)
    print("Contact forces:", f_c)
    print("Ground Truth External force:", ext_wrench)
    print(np.mean(objective_values))
    end_time = time.time()
    print("Time taken to solve QP:", end_time - start_time)
    
    # Example data
    n_joints = 7
    n_contacts = 1
    mu = 0.5
    Jc_batch = np.array(Jcs)
    gt_ext_tau = np.array(gt_ext_taus)
    batch_qp_solver = BatchQPSolver(n_joints, n_contacts, n_qps, mu)
    
    for i in range(3):
        start_time = time.time()
        f_c_batch, loss = batch_qp_solver.solve(Jc_batch, gt_ext_tau)
        end_time = time.time()
        print("Batch QP solver time:", end_time - start_time)
        print(loss)
    print(f_c_batch)
    
    # Example data    
    Jc_batch = np.array(Jcs)
    gt_ext_tau = np.array(gt_ext_taus)
    from scipy.linalg import block_diag
    Jc_batch = block_diag(*Jc_batch)
    gt_ext_tau = gt_ext_tau.flatten()
    batch_qp_solver2 = BatchQPSolverMatrix(n_joints, n_contacts, n_qps, mu)
    print(Jc_batch.shape, gt_ext_tau.shape)
    
    for i in range(3):
        start_time = time.time()
        f_c_batch, loss = batch_qp_solver2.solve(Jc_batch, gt_ext_tau)
        end_time = time.time()
        print("Batch QP solver2 time:", end_time - start_time)
        # print(f_c_batch)
    # print(f_c_batch)