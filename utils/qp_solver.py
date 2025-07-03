import cvxpy as cp
import numpy as np

class QPNonlinearSolver:
    def __init__(self, n_joints, n_contacts, mu=0.5):
        """
        Initialize the QP solver with parameterized variables.

        Args:
            njoints (int): Number of joints in the robot.
            n_contacts (int): Number of contact points.
            mu (float): Friction coefficient.
        """
        self.n_joints = n_joints
        self.n_contacts = n_contacts
        self.mu = mu
        print(f"Initializing QP Solver with {n_joints} joints and {n_contacts} contacts, friction coefficient: {mu}")

        # Define the contact force variable
        self.f_c = cp.Variable(3 * self.n_contacts)  # [fx1, fy1, fz1, ...]

        # Define parameters for Jacobian and external torques
        self.Jc = cp.Parameter((3 * self.n_contacts, self.n_joints))  # Jacobian matrix
        self.tau_ext = cp.Parameter(self.n_joints)  # External torques

        # Define the objective function
        self.objective = cp.Minimize(cp.sum_squares(self.Jc.T @ self.f_c - self.tau_ext))

        # # # Define parameters for the quadratic cost function
        # self.Q = cp.Parameter((3 * self.n_contacts, 3 * self.n_contacts), PSD=True)  # Cost matrix
        # self.c = cp.Parameter(3 * self.n_contacts)  # Cost vector
        # self.objective = cp.Minimize(1/2 * cp.quad_form(self.f_c, self.Q) + self.c.T @ self.f_c)

        # Define the constraints
        self.constraints = []
        for i in range(self.n_contacts):
            fx = self.f_c[3 * i + 0]
            fy = self.f_c[3 * i + 1]
            fz = self.f_c[3 * i + 2]

            # Unilateral constraint
            self.constraints.append(fz >= 1e-3)

            # Friction cone (pyramid approximation)
            # self.constraints.append(cp.abs(fx) <= self.mu * fz)
            # self.constraints.append(cp.abs(fy) <= self.mu * fz)
            self.constraints.append(fx <= self.mu * fz)
            self.constraints.append(fy <= self.mu * fz)
            self.constraints.append(fx >= -self.mu * fz)
            self.constraints.append(fy >= -self.mu * fz)

            # # Friction cone (without pyramid approximation)
            # self.constraints.append(cp.norm(cp.vstack([fx, fy])) <= self.mu * fz)

        # Define the problem
        self.prob = cp.Problem(self.objective, self.constraints)

    def solve(self, Jc, tau_ext, verbose=False):
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

        # # Update the quadratic cost parameters
        # self.Q.value = Jc @ Jc.T + reg * np.eye(3 * self.n_contacts)
        # self.c.value = -Jc @ tau_ext
                    
        # Solve the problem using OSQP
        self.prob.solve(solver=cp.OSQP, warm_start=True, verbose=verbose)

        # Return the results
        return self.f_c.value, self.objective.value

class QPSolver:
    def __init__(self, n_joints, n_contacts, mu=0.5, polyhedral_num=4):
        """
        Initialize the QP solver with parameterized variables.

        Args:
            njoints (int): Number of joints in the robot.
            n_contacts (int): Number of contact points.
            mu (float): Friction coefficient.
        """
        self.n_joints = n_joints
        self.n_contacts = n_contacts
        self.mu = mu
        self.polyhedral_num = polyhedral_num
        print(f"Initializing QP Solver with {n_joints} joints and {n_contacts} contacts, friction coefficient: {mu}")

        # Define the contact force variable as linear combination of friction cone basis vectors
        self.alpha_c = cp.Variable(self.n_contacts * polyhedral_num)  # Coefficients for the friction cone basis vectors

        # Friction cone basis vectors
        self.F_basis = cp.Parameter((self.n_contacts * polyhedral_num, 3))

        # Define parameters for Jacobian and external torques
        self.Jc = cp.Parameter((3 * self.n_contacts, self.n_joints))  # Jacobian matrix
        self.tau_ext = cp.Parameter(self.n_joints)  # External torques

        # Define the objective function
        self.f_c = self.alpha_c @ self.F_basis  # Contact forces as a linear combination of basis vectors
        self.objective = cp.Minimize(cp.sum_squares(self.Jc.T @ self.f_c - self.tau_ext))
        
        # Define the constraints for the contact forces
        self.constraints = []
        for i in range(self.n_contacts):
            for j in range(polyhedral_num):
                alpha_idx = i * polyhedral_num + j
                self.constraints.append(self.alpha_c[alpha_idx] >= 0)

        # Define the problem
        self.prob = cp.Problem(self.objective, self.constraints)

    def solve(self, Jc, tau_ext, Fc_basis, verbose=False):
        """
        Solve the QP problem with updated parameters.

        Args:
            Jc (np.ndarray): Jacobian matrix of size (3, n_joints).
            tau_ext (np.ndarray): External torques of size (n_joints,).
            Fc_basis (np.ndarray): shape (n_contacts * polyhedral_num, 3) friction cone basis vectors.

        Returns:
            np.ndarray: Estimated contact forces of size (3 * n_contacts,).
            float: Loss value of the optimization problem.
        """
        # Update the parameters
        self.Jc.value = Jc
        self.tau_ext.value = tau_ext

        self.F_basis.value = Fc_basis
                    
        # Solve the problem using OSQP
        self.prob.solve(solver=cp.OSQP, warm_start=True, verbose=verbose)

        # Return the results
        return self.f_c.value, self.objective.value
    
class BatchNonlinerQPSolver:
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

        # # Define the contact force variable for all QPs
        # self.f_c = cp.Variable((self.n_qps, 3 * self.n_contacts))  # [fx1, fy1, fz1, ...]
        
        # You need to define as individual variables for each QP
        self.f_c = [cp.Variable(3 * self.n_contacts) for i in range(self.n_qps)]

        # Define parameters for Jacobians and external torques
        # self.Jc = cp.Parameter((self.n_qps, 3 * self.n_contacts, self.n_joints))  # Jacobian matrices for all QPs
        self.Jc = [cp.Parameter((3 * self.n_contacts, self.n_joints)) for _ in range(self.n_qps)]  # Individual Jacobian for each QP
        self.tau_ext = cp.Parameter((self.n_joints))  # External torques for all QPs

        # Define the objective function (sum of squared residuals for all QPs)
        residuals = [self.Jc[i].T @ self.f_c[i] - self.tau_ext for i in range(self.n_qps)]
        self.objective = cp.Minimize(cp.sum([cp.sum_squares(res) for res in residuals]))

        # Define the constraints for all QPs
        self.constraints = []
        for i in range(self.n_qps):
            for j in range(self.n_contacts):
                fx = self.f_c[i][3 * j + 0]
                fy = self.f_c[i][3 * j + 1]
                fz = self.f_c[i][3 * j + 2]

                # Unilateral constraint
                self.constraints.append(fz >= 0)

                # Friction cone (pyramid approximation)
                self.constraints.append(cp.abs(fx) <= self.mu * fz)
                self.constraints.append(cp.abs(fy) <= self.mu * fz)

        # Define the problem
        self.prob = cp.Problem(self.objective, self.constraints)

    def solve(self, Jc_batch, tau_ext):
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
        # self.Jc.value = Jc_batch
        for i in range(self.n_qps):
            self.Jc[i].value = Jc_batch[i, :, :]
        self.tau_ext.value = tau_ext

        # Solve the problem
        self.prob.solve(solver=cp.OSQP, warm_start=True)

        # Compute the residuals
        residuals = [self.Jc[i].value.T @ self.f_c[i].value - self.tau_ext.value for i in range(self.n_qps)]
        residuals = np.array(residuals)
        residuals = np.linalg.norm(residuals, axis=1)

        f_c_values = np.array([fc.value for fc in self.f_c])

        # Return the results
        return f_c_values, self.objective.value, residuals
                
class BatchQPSolver:
    def __init__(self, n_joints, n_contacts, n_qps, mu=0.5, polyhedral_num=4):
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
        self.polyhedral_num = polyhedral_num

        # Define the contact force variable for all QPs
        self.friction_cone_basis = [[cp.Parameter((self.polyhedral_num, 3)) for i in range(self.n_contacts)] for _ in range(self.n_qps)]
        self.alpha_c = [[cp.Variable(self.polyhedral_num) for i in range(self.n_contacts)] for _ in range(self.n_qps)]  # Coefficients for the friction cone basis vectors
        self.f_c = [cp.hstack([alpha_c @ frcition_cone_basis for alpha_c, frcition_cone_basis in zip(self.alpha_c[i], self.friction_cone_basis[i])]) for i in range(self.n_qps)]  # Contact forces as a linear combination of basis vectors

        # Define parameters for Jacobians and external torques
        # self.Jc = cp.Parameter((self.n_qps, 3 * self.n_contacts, self.n_joints))  # Jacobian matrices for all QPs
        self.Jc = [cp.Parameter((3 * self.n_contacts, self.n_joints)) for _ in range(self.n_qps)]  # Individual Jacobian for each QP
        self.tau_ext = cp.Parameter((self.n_joints))  # External torques for all QPs

        # Define the objective function (sum of squared residuals for all QPs)
        residuals = [self.Jc[i].T @ self.f_c[i] - self.tau_ext for i in range(self.n_qps)]
        self.objective = cp.Minimize(cp.sum([cp.sum_squares(res) for res in residuals]))

        # Define the constraints for all QPs
        self.constraints = []
        for i in range(self.n_qps):
            for j in range(self.n_contacts):
                for k in range(self.polyhedral_num):
                    self.constraints.append(self.alpha_c[i][j][k] >= 0)

        # Define the problem
        self.prob = cp.Problem(self.objective, self.constraints)

    def solve(self, Jc_batch, tau_ext, Friction_cone_basises):
        """
        Solve the batch QP problem with updated parameters.

        Args:
            Jc_batch (np.ndarray): Batch of Jacobian matrices of size (3*n_contacts, n_joints, n_qps).
            tau_ext_batch (np.ndarray): Batch of external torques of size (n_joints, n_qps).
            Friction_cone_basises (np.ndaaray): Batch of friction cone basis vectors of size (n_contacts, n_qps, polyhedral_num, 3).

        Returns:
            np.ndarray: Estimated contact forces of size (3 * n_contacts, n_qps).
            float: Loss value of the optimization problem.
        """
        # Update the parameters
        # self.Jc.value = Jc_batch
        for i in range(self.n_qps):
            self.Jc[i].value = Jc_batch[i, :, :]
        self.tau_ext.value = tau_ext
        for i in range(self.n_qps):
            for j in range(self.n_contacts):
                self.friction_cone_basis[i][j].value = Friction_cone_basises[j][i, :, :]

        # Solve the problem
        self.prob.solve(solver=cp.SCS, warm_start=True)

        # Compute the residuals
        residuals = [self.Jc[i].value.T @ self.f_c[i].value - self.tau_ext.value for i in range(self.n_qps)]
        residuals = np.array(residuals)
        residuals = np.linalg.norm(residuals, axis=1)

        f_c_values = np.array([fc.value for fc in self.f_c])

        # Return the results
        return f_c_values, self.objective.value, residuals
                
if __name__ == "__main__":
    import numpy as np
    from pathlib import Path
    data_path = Path(__file__).parent
    data_dict = np.load(f"{data_path}/data.npz")
    print(data_dict.keys())
    import time
    start_time = time.time()
    
    # Parameters
    n_joints = 7
    n_contacts = 1
    mu = 1.0  # friction coefficient
    n_qps = 100
    
    # Initialize the solver
    qp_solver = QPSolver(n_joints, n_contacts, mu)
    gt_ext_taus = []
    fcs = []
    Jcs = []
    objective_values = []
    ext_wrenches = []
    for i in range(100, 100 + n_qps):
        index = i
        Jc = data_dict["jacs_contact"][index]
        ext_tau = data_dict["gt_ext_taus"][index]
        ext_wrench = data_dict["gt_ext_wrenches"][index]
        gt_ext_tau = Jc.T @ ext_wrench
        gt_ext_taus.append(gt_ext_tau)
        ext_wrenches.append(ext_wrench)
        
        # Solve the QP problem
        Jc = Jc[:3, :]
        Jcs.append(Jc)

        f_c, objective_value = qp_solver.solve(Jc, gt_ext_tau)
        # print(f_c, "Contact forces")
        # print(ext_wrench, "Ground Truth External force")
        objective_values.append(objective_value)
        # print("Contact forces:", f_c)
        # print("Ground Truth External force:", ext_wrench)
        # print("")
    end_time = time.time()
    print("Time taken to solve QP:", end_time - start_time)
    
    # Test the qp solver for different
    start_time = time.time()
    qp_num = 30
    for i in range(qp_num):
        Jc = Jcs[i]
        gt_ext_tau = gt_ext_taus[0]
        f_c, objective_value = qp_solver.solve(Jc, gt_ext_tau)
        print(f"QP {i}: Contact forces: {f_c}, Objective value: {objective_value}")
    print("external wwrenches:", ext_wrenches[0])
    print(f"Time taken to solve {qp_num} individual QPs:", time.time() - start_time)
    
    # Example data
    n_joints = 7
    n_contacts = 1
    mu = 1.0
    Jc_batch = np.array(Jcs)
    gt_ext_taus = np.array(gt_ext_taus)
    ext_wrenches = np.array(ext_wrenches)
    test_idx = 0
    gt_ext_tau = gt_ext_taus[test_idx]  # Use the first one for batch processing
    ext_wrench = ext_wrenches[test_idx]
    
    batch_qp_solver = BatchQPSolver(n_joints, n_contacts, n_qps, mu)
    print(gt_ext_tau.shape, Jc_batch.shape, "SHAPE")
    
    for i in range(3):
        start_time = time.time()
        f_c_batch, loss, residuals = batch_qp_solver.solve(Jc_batch, gt_ext_tau)
        end_time = time.time()
        print("Batch QP solver time:", end_time - start_time)
    print(Jc_batch.shape)
    print(ext_wrench, f_c_batch[test_idx])
    print(residuals)
    idx = np.argmin(residuals)
    f_c_select = f_c_batch[idx]
    print(f_c_select)