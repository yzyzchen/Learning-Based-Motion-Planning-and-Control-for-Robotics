import control
import numpy as np
import scipy.linalg
import cvxpy as cp


class LinearMPC:

    def __init__(self, A, B, Q, R, horizon):
        self.dx = A.shape[0]
        self.du = B.shape[1]
        assert A.shape == (self.dx, self.dx)
        assert B.shape == (self.dx, self.du)
        self.H = horizon
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R

    def compute_SM(self):
        """
        Computes the S and M matrices as defined in the ipython notebook

        All the variables you need should be class member variables already

        Returns:
            S: np.array of shape (horizon * dx, horizon * du) S matrix
            M: np.array of shape (horizon * dx, dx) M matrix

        """
        A, B = self.A, self.B
        dx, du = self.dx, self.du
        A_zero, B_zero = np.zeros_like(A), np.zeros_like(B)
        S, M = np.empty((0, self.H * du)), np.empty((0, dx))
        # --- Your code here
        for i in range(self.H):
            S_row = np.empty((dx, 0))
            for j in range(self.H):
                # S[i * dx : (i + 1) * dx][j * dx : (j + 1) * du] = A**i @ B
                if i < j:
                    # np.concatenate((),)
                    S_row = np.hstack((S_row, B_zero))
                else:
                    S_row = np.hstack((S_row, np.linalg.matrix_power(A, i - j) @ B))
            S = np.vstack((S, S_row))
            M = np.vstack((M, np.linalg.matrix_power(A, i+1)))
        # ---
        return S, M

    def compute_Qbar_and_Rbar(self):
        Q_repeat = [self.Q] * self.H
        R_repeat = [self.R] * self.H
        return scipy.linalg.block_diag(*Q_repeat), scipy.linalg.block_diag(*R_repeat)

    def compute_finite_horizon_lqr_gain(self):
        """
            Compute the controller gain G0 for the finite-horizon LQR

        Returns:
            G0: np.array of shape (du, dx)

        """
        S, M = self.compute_SM()
        Qbar, Rbar = self.compute_Qbar_and_Rbar()
        G0 = None
        # --- Your code here
        G = -np.linalg.solve((S.T @ Qbar @ S + Rbar), (S.T @ Qbar @ M))
        G0 = G[:self.du,:]
        # ---
        return G0

    def compute_lqr_gain(self):
        """
            Compute controller gain G for infinite-horizon LQR
        Returns:
            Ginf: np.array of shape (du, dx)

        """
        Ginf = None
        theta_T_theta, _, _ = control.dare(self.A, self.B, self.Q, self.R)

        # --- Your code here
        G = -np.linalg.solve((self.B.T @ theta_T_theta @ self.B + self.R), (self.B.T @ theta_T_theta @ self.A))
        Ginf = G[:self.du,:]
        # ---
        return Ginf

    def lqr_box_constraints_qp_shooting(self, x0, u_min, u_max):
        """
            Solves the finite-horizon box-constrained LQR problem using Quadratic Programing with shooting

        Args:
            x0: np.array of shape (dx,) containing current state
            u_min: np.array of shape (du,), minimum control value
            u_max: np.array of shape (du,), maximum control value

        Returns:
            U: np.array of shape (horizon, du) containing sequence of optimal controls

        """

        S, M = self.compute_SM()
        Qbar, Rbar = self.compute_Qbar_and_Rbar()
        # --- Your code here
        U_max = u_max * np.ones(self.H)
        U_min = u_min * np.ones(self.H)
        U = cp.Variable(self.H)
        prob = cp.Problem(cp.Minimize(cp.quad_form(S @ U + M @ x0, Qbar) + cp.quad_form(U, Rbar)),
                          [U <= U_max,
                           U >= U_min])
        prob.solve()
        U = U.value
        # U = U.reshape(-1,1)
        # ---

        return U.reshape(-1,1)

    def lqr_box_constraints_qp_collocation(self, x0, u_min, u_max):
        """
            Solves the finite-horizon box-constrained LQR problem using Quadratic Programing
            with collocation

        Args:
            x0: np.array of shape (dx,) containing current state
            u_min: np.array of shape (du,), minimum control value
            u_max: np.array of shape (du,), maximum control value

        Returns:
            U: np.array of shape (horizon, du) containing sequence of optimal controls
            X: np.array of shape (horizon, dx) containing sequence of optimal states

        """

        # --- Your code here
        X, U = cp.Variable((self.H + 1, self.dx)), cp.Variable((self.H, self.du))
        # X.append(x0)
        cost = 0
        constraint = [X[0,:] == x0]
        for i in range(self.H):
            cost += cp.quad_form(X[i+1,:], self.Q)+cp.quad_form(U[i,:], self.R)
            constraint.append(X[i+1,:] == self.A @ X[i,:] + self.B @ U[i,:])
            constraint.append(U[i,:] <= u_max)
            constraint.append(U[i,:] >= u_min)
        prob = cp.Problem(cp.Minimize(cost), constraint)
        prob.solve()
        # ---

        return U.value, X.value[1:,:]
