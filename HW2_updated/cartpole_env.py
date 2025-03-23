import torch
import numpy as np
import pybullet as p
import pybullet_data as pd
from base_env import BaseEnv
import gym


class CartpoleEnv(BaseEnv):

    def __init__(self, *args, **kwargs):
        self.cartpole = None
        super().__init__(*args, **kwargs)

    def step(self, control):
        """
            Steps the simulation one timestep, applying the given force
        Args:
            control: np.array of shape (1,) representing the force to apply

        Returns:
            next_state: np.array of shape (4,) representing next cartpole state

        """
        p.setJointMotorControl2(self.cartpole, 0, p.TORQUE_CONTROL, force=control[0])
        p.stepSimulation()
        return self.get_state()

    def reset(self, state=None):
        """
            Resets the environment
        Args:
            state: np.array of shape (4,) representing cartpole state to reset to.
                   If None then state is randomly sampled
        """
        if state is not None:
            self.state = state
        else:
            self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        p.resetSimulation()
        p.setAdditionalSearchPath(pd.getDataPath())
        self.cartpole = p.loadURDF('cartpole.urdf')
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.dt)
        p.setRealTimeSimulation(0)
        p.changeDynamics(self.cartpole, 0, linearDamping=0, angularDamping=0,
                         lateralFriction=0, spinningFriction=0, rollingFriction=0)
        p.changeDynamics(self.cartpole, 1, linearDamping=0, angularDamping=0,
                         lateralFriction=0, spinningFriction=0, rollingFriction=0)
        p.changeDynamics(self.cartpole, -1, linearDamping=0, angularDamping=0,
                         lateralFriction=0, spinningFriction=0, rollingFriction=0)
        p.setJointMotorControl2(self.cartpole, 1, p.VELOCITY_CONTROL, force=0)
        p.setJointMotorControl2(self.cartpole, 0, p.VELOCITY_CONTROL, force=0)
        self.set_state(self.state)
        self._setup_camera()

    def get_state(self):
        """
            Gets the cartpole internal state

        Returns:
            state: np.array of shape (4,) representing cartpole state [x, theta, x_dot, theta_dot]

        """

        x, x_dot = p.getJointState(self.cartpole, 0)[0:2]
        theta, theta_dot = p.getJointState(self.cartpole, 1)[0:2]
        return np.array([x, theta, x_dot, theta_dot])

    def set_state(self, state):
        x, theta, x_dot, theta_dot = state
        p.resetJointState(self.cartpole, 0, targetValue=x, targetVelocity=x_dot)
        p.resetJointState(self.cartpole, 1, targetValue=theta, targetVelocity=theta_dot)

    def _get_action_space(self):
        action_space = gym.spaces.Box(low=-30, high=30)  # linear force # TODO: Verify that they are correct
        return action_space

    def _get_state_space(self):
        x_lims = [-5, 5]  # TODO: Verify that they are the correct limits
        theta_lims = [-np.pi, np.pi]
        x_dot_lims = [-10, 10]
        theta_dot_lims = [-5 * np.pi, 5 * np.pi]
        state_space = gym.spaces.Box(
            low=np.array([x_lims[0], theta_lims[0], x_dot_lims[0], theta_dot_lims[0]], dtype=np.float32),
            high=np.array([x_lims[1], theta_lims[1], x_dot_lims[1], theta_dot_lims[1]],
                          dtype=np.float32))  # linear force # TODO: Verify that they are correct
        return state_space

    def _setup_camera(self):
        self.render_h = 240
        self.render_w = 320
        base_pos = [0, 0, 0]
        cam_dist = 2
        cam_pitch = 0.3
        cam_yaw = 0
        self.view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=cam_dist,
            yaw=cam_yaw,
            pitch=cam_pitch,
            roll=0,
            upAxisIndex=2)
        self.proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                        aspect=self.render_w / self.render_h,
                                                        nearVal=0.1,
                                                        farVal=100.0)

    def linearize_numerical(self, state, control, eps=1e-3):
        """
            Linearizes cartpole dynamics around linearization point (state, control). Uses numerical differentiation
        Args:
            state: np.array of shape (4,) representing cartpole state
            control: np.array of shape (1,) representing the force to apply
            eps: Small change for computing numerical derivatives
        Returns:
            A: np.array of shape (4, 4) representing Jacobian df/dx for dynamics f
            B: np.array of shape (4, 1) representing Jacobian df/du for dynamics f
        """
        A, B = np.zeros((4, 4)), np.zeros((4, 1))
        # state = torch.tensor(state, dtype = torch.float32).unsqueeze(0)
        # control = torch.tensor(control, dtype = torch.float32).unsqueeze(0)
        # --- Your code here
        
        for i in range(len(state)):
            eps_m = np.zeros_like(state)
            eps_m[i] = eps
            f_max = self.dynamics(state + eps_m, control)
            f_min = self.dynamics(state - eps_m, control)
            A[:,i] = (f_max - f_min)/(2 * eps)
        # eps_m = torch.zeros_like(B)
        # for j in range(control.shape[0]):
        # eps_m = eps
        fu_max = self.dynamics(state, control + eps)
        fu_min = self.dynamics(state, control - eps)
        B = (fu_max - fu_min)/(2 * eps)
        # ---
        return A, B.reshape(-1,1)

    

def dynamics_analytic(state, action):

    """
        Computes x_t+1 = f(x_t, u_t) using analytic model of dynamics in Pytorch
        Should support batching
    Args:
        state: torch.tensor of shape (B, 4) representing the cartpole state
        action: torch.tensor of shape (B, 1) representing the force to apply

    Returns:
        next_state: torch.tensor of shape (B, 4) representing the next cartpole state

    """
    next_state = None
    dt = 0.05
    g = 9.81
    mc = 1
    mp = 0.1
    l = 0.5

    # --- Your code here
    # Unpack state components (Batch-wise)
    x, theta, x_dot, theta_dot = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
    # x, theta, x_dot, theta_dot = torch.chunk(state, chunks=4, dim = 1)
    # Compute angular acceleration (θ̈)
    theta_dotdot_num = g * torch.sin(theta) - torch.cos(theta) * (
        (action[:, 0] + mp * l * theta_dot**2 * torch.sin(theta)) / (mc + mp)
    )
    theta_dotdot_den = l * (4/3 - (mp * torch.cos(theta)**2) / (mc + mp))
    theta_dotdot = theta_dotdot_num / theta_dotdot_den

    # Compute linear acceleration (ẍ)
    x_dotdot = (action[:, 0] + mp * l * (theta_dot**2 * torch.sin(theta) - theta_dotdot * torch.cos(theta))) / (mc + mp)

    # Euler Integration
    x_dot_next = x_dot + dt * x_dotdot
    theta_dot_next = theta_dot + dt * theta_dotdot
    x_next = x + dt * x_dot_next
    theta_next = theta + dt * theta_dot_next

    # Stack results into `next_state`
    next_state = torch.stack((x_next, theta_next, x_dot_next, theta_dot_next), dim=1)  # Shape (B, 4)

    return next_state


def linearize_pytorch(state, control):
    """
        Linearizes cartpole dynamics around linearization point (state, control). Uses autograd of analytic dynamics
    Args:
        state: torch.tensor of shape (4,) representing cartpole state
        control: torch.tensor of shape (1,) representing the force to apply

    Returns:
        A: torch.tensor of shape (4, 4) representing Jacobian df/dx for dynamics f
        B: torch.tensor of shape (4, 1) representing Jacobian df/du for dynamics f

    """
    A, B = None, None
    # --- Your code here
    state = state.reshape(1,4)
    control = control.reshape(1,1)
    a = (state,control)

    jacobian = torch.autograd.functional.jacobian(dynamics_analytic,a)

    A = jacobian[0].reshape(4,4)
    B = jacobian[1].reshape(4,1)

    # ---
    return A, B
