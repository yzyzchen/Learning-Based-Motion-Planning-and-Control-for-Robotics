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
        action = self.env.action_space.sample()  # random sample the env action space
        return action, None


def execute_policy(env, policy, num_steps=20):
    """
    Given a policy and an environment, execute the policy for num_steps steps.
    The policy is a stable-baselines3 policy object.
    :param env:
    :param policy:
    :param num_steps:
    :return:
    """
    state = env.reset()
    states = [state]
    rewards = []
    goal_reached = False
    for i in range(num_steps):
        print(f"Step {i}: Executing action...")
        action, _ = policy.predict(state)
        next_state, reward, done, _ = env.step(action)
        states.append(next_state)
        rewards.append(reward)
        # check if goal reached
        if np.linalg.norm(next_state - TARGET_POSE) < BOX_SIZE:  # Use threshold instead of reward==0
            goal_reached = True
            print(f"Step {i}: Goal Reached!")
            break
        # check if finished exploring
        if done:
            print(f"Step {i}: Environment done. Resetting...")
            state = env.reset()
        else:
            state = next_state
    # ---
    return states, rewards, goal_reached




def obstacle_free_pushing_reward_function_object_pose_space(state, action):
    """
    Defines the state reward function for the action transition (prev_state, action, state)
    :param state: numpy array of shape (state_dim)
    :param action:numpy array of shape (action_dim)
    :return: reward value. <float>
    """
    # --- Your code here
    global previous_distance
    distance = np.linalg.norm(state - TARGET_POSE)
    reward = -distance
    # encourage progress bonus
    if 'previous_distance' in globals():
        reward += (previous_distance - distance) * 5
    # task completion bonus
    if distance < BOX_SIZE:
        reward += 10
    # action smoothing reward
    action_cost = np.linalg.norm(action)
    reward -= action_cost * 0.1
    # store each distance
    previous_distance = distance
    # ---
    return reward


def pushing_with_obstacles_reward_function_object_pose_space(state, action):
    """
    Defines the state reward function for the action transition (prev_state, action, state)
    :param state: numpy array of shape (state_dim)
    :param action:numpy array of shape (action_dim)
    :return: reward value. <float>
    """
    reward = None
    # --- Your code here
    global pre_obs_distance
    distance = np.linalg.norm(state - TARGET_POSE)
    reward = -distance
    # check collision
    reward -= int(in_collision(state)) * 100
    # encourage progress bonus
    if 'pre_obs_distance' in globals():
        reward += (pre_obs_distance - distance) * 5
    # task completion bonus
    if distance < BOX_SIZE:
        reward += 10
    # action smoothing reward
    action_cost = np.linalg.norm(action)
    reward -= action_cost * 0.1
    # store each distance
    pre_obs_distance = distance
    # ---
    return reward

# Ancillary functions
# --- Your code here
def in_collision(state):
    max = OBSTACLE_CENTRE + OBSTACLE_HALFDIMS    
    min = OBSTACLE_CENTRE - OBSTACLE_HALFDIMS
    in_collision = np.all((state <= max) & (state >= min))
    return in_collision
# ---