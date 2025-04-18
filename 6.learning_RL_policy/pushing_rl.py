import numpy as np
from tqdm import tqdm

from gym.spaces import Box
# PPO Implementation reference:
# https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO.py

# DQN Implementation reference:
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html


from panda_pushing_env import TARGET_POSE, OBSTACLE_CENTRE, OBSTACLE_HALFDIMS, BOX_SIZE


class RandomPolicy(object):
    """
    A random policy for any environment.
    It has the same method as a stable-baselines3 policy object for compatibility.
    """

    def __init__(self, env):
        self.env = env

    def predict(self, state):
        # Random sample the env action space
        action = self.env.action_space.sample()  
        return action, None


def execute_policy(env, policy, num_steps=20):
    """
    Execute a given policy in the environment for a specified number of steps.
    
    Args:
        env: The PandaPushingEnv environment instance
        policy: A policy object (from stable-baselines3 or similar interface) 
               with predict() method
        num_steps: Maximum number of steps to execute (default: 20)
    
    Returns:
        states: List of states encountered during execution
        rewards: List of rewards received
        goal_reached: Boolean indicating if the target was reached
    
    Note:
        - The policy execution stops if either:
          1. num_steps is reached
          2. The environment signals done=True
          3. The goal position is reached (within BOX_SIZE distance)
        - The state space is the object's pose (x, y, θ)
    """
    states = []
    rewards = []
    goal_reached = False
    state = env.reset()
    states.append(state)
    goal_reached = False
    for i in tqdm(range(num_steps)):
        action_i, _states = policy.predict(state)
        state, reward, done, info = env.step(action_i)
        states.append(state)
        # Check if we have reached the goal
        end_pose = env.get_object_pos_planar()
        # Evaluate only position, not orientation
        goal_distance = np.linalg.norm(end_pose[:2] - TARGET_POSE[:2])  
        goal_reached = goal_distance < BOX_SIZE
        if done or goal_reached:
            break
    return states, rewards, goal_reached

def in_collision(state):
    max = OBSTACLE_CENTRE + OBSTACLE_HALFDIMS    
    min = OBSTACLE_CENTRE - OBSTACLE_HALFDIMS
    in_collision = np.all((state[:2] <= max[:2]) & (state[:2] >= min[:2]))
    return in_collision

def obstacle_free_pushing_reward_function_object_pose_space(state, action):
    """
    Define the reward function for the obstacle-free pushing task.
    
    The reward should encourage:
    1. Moving closer to the target position
    2. Reaching the target (with a significant bonus)
    3. Staying within the workspace bounds
    
    Args:
        state: numpy array [x, y, θ] representing object's pose
        action: numpy array [push_location, push_angle, push_length] 
               representing the action
    
    Returns:
        reward: Float value representing the reward
        
    Note:
        - The workspace limits are defined by space_limits
        - A state outside workspace bounds should receive a large negative reward
        - The target pose is available as TARGET_POSE
        - BOX_SIZE defines the threshold for reaching the target
    """
    reward = None
    SPACE_LIMITS = {
        'x': (0.0, 1.0),
        'y': (-0.5, 0.5),
        'theta': (-np.pi, np.pi)
    }
    # --- Your code here
    x, y, theta = state
    pos_dis = np.linalg.norm(TARGET_POSE[:2] - state[:2])
    ang_diff = np.abs(np.arctan2(np.sin(theta - TARGET_POSE[2]), np.cos(theta - TARGET_POSE[2])))
    # outside workspace: large negative reward
    if not (
        SPACE_LIMITS['x'][0] <= x <= SPACE_LIMITS['x'][1] and
        SPACE_LIMITS['y'][0] <= y <= SPACE_LIMITS['y'][1] and
        SPACE_LIMITS['theta'][0] <= theta <= SPACE_LIMITS['theta'][1]
    ): return -100
    # collision: large negative reward
    if in_collision(state): return -50
    # success: large positive reward
    if pos_dis < BOX_SIZE and ang_diff < np.deg2rad(10): return 100.0
    # else
    reward = -pos_dis - 0.1 * ang_diff
    # ---
    return reward
