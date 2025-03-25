import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from panda_pushing_env import TARGET_POSE_FREE, TARGET_POSE_OBSTACLES, OBSTACLE_HALFDIMS, OBSTACLE_CENTRE, BOX_SIZE

TARGET_POSE_FREE_TENSOR = torch.as_tensor(TARGET_POSE_FREE, dtype=torch.float32)
TARGET_POSE_OBSTACLES_TENSOR = torch.as_tensor(TARGET_POSE_OBSTACLES, dtype=torch.float32)
OBSTACLE_CENTRE_TENSOR = torch.as_tensor(OBSTACLE_CENTRE, dtype=torch.float32)[:2]
OBSTACLE_HALFDIMS_TENSOR = torch.as_tensor(OBSTACLE_HALFDIMS, dtype=torch.float32)[:2]


def collect_data_random(env, num_trajectories=1000, trajectory_length=10):
    """
    Collect data from the provided environment using uniformly random exploration.
    :param env: Gym Environment instance.
    :param num_trajectories: <int> number of data to be collected.
    :param trajectory_length: <int> number of state transitions to be collected
    :return: collected data: List of dictionaries containing the state-action trajectories.
    Each trajectory dictionary should have the following structure:
        {'states': states,
        'actions': actions}
    where
        * states is a numpy array of shape (trajectory_length+1, state_size) containing the states [x_0, ...., x_T]
        * actions is a numpy array of shape (trajectory_length, actions_size) containing the actions [u_0, ...., u_{T-1}]
    Each trajectory is:
        x_0 -> u_0 -> x_1 -> u_1 -> .... -> x_{T-1} -> u_{T_1} -> x_{T}
        where x_0 is the state after resetting the environment with env.reset()
    All data elements must be encoded as np.float32.
    """
    collected_data = []
    # --- Your code here
    for _ in range(num_trajectories):
        # pos_0
        state = env.reset()
        state = np.array(state, dtype=np.float32)
        states = [state]
        actions = []
        for _ in range(trajectory_length):
            action = env.action_space.sample()
            action = np.array(action, dtype=np.float32)
            state, _, _, _ = env.step(action)
            state = np.array(state, dtype=np.float32)
            states.append(state)
            actions.append(action)
        trajectory = {
            'states': np.array(states, dtype=np.float32),
            'actions': np.array(actions, dtype = np.float32)
        }
        collected_data.append(trajectory)
    # ---
    return collected_data

# def collate_fn(batch):
#     state = torch.stack(item['state'] for item in batch)
#     action = torch.stack(item['action'] for item in batch)
#     next_state = torch.stack(item['next_state'] for item in batch)
#     return {'state':state,'action':action, 'next_state':next_state}



def process_data_single_step(collected_data, batch_size=500):
    """
    Process the collected data and returns a DataLoader for train and one for validation.
    The data provided is a list of trajectories (like collect_data_random output).
    Each DataLoader must load dictionary as {'state': x_t,
     'action': u_t,
     'next_state': x_{t+1},
    }
    where:
     x_t: torch.float32 tensor of shape (batch_size, state_size)
     u_t: torch.float32 tensor of shape (batch_size, action_size)
     x_{t+1}: torch.float32 tensor of shape (batch_size, state_size)

    The data should be split in a 80-20 training-validation split.
    :param collected_data:
    :param batch_size: <int> size of the loaded batch.
    :return:

    Hints:
     - Pytorch provides data tools for you such as Dataset and DataLoader and random_split
     - You should implement SingleStepDynamicsDataset below.
        This class extends pytorch Dataset class to have a custom data format.
    """
    # init
    data = SingleStepDynamicsDataset(collected_data)
    # split
    train_size = int(data.__len__() * 0.8)
    val_size = int(data.__len__() - train_size)
    train_data, val_data = torch.utils.data.random_split(data, [train_size, val_size])

    # train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    # val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    # ---
    return train_loader, val_loader
    # return train_data, val_data


def process_data_multiple_step(collected_data, batch_size=500, num_steps=4):
    """
    Process the collected data and returns a DataLoader for train and one for validation.
    The data provided is a list of trajectories (like collect_data_random output).
    Each DataLoader must load dictionary as
    {'state': x_t,
     'action': u_t, ..., u_{t+num_steps-1},
     'next_state': x_{t+1}, ... , x_{t+num_steps}
    }
    where:
     state: torch.float32 tensor of shape (batch_size, state_size)
     next_state: torch.float32 tensor of shape (batch_size, num_steps, action_size)
     action: torch.float32 tensor of shape (batch_size, num_steps, state_size)

    Each DataLoader must load dictionary dat
    The data should be split in a 80-20 training-validation split.
    :param collected_data:
    :param batch_size: <int> size of the loaded batch.
    :param num_steps: <int> number of steps to load the multistep data.
    :return:

    Hints:
     - Pytorch provides data tools for you such as Dataset and DataLoader and random_split
     - You should implement MultiStepDynamicsDataset below.
        This class extends pytorch Dataset class to have a custom data format.
    """
    # --- Your code here
    # init
    data = MultiStepDynamicsDataset(collected_data)
    # split
    train_size = int(data.__len__() * 0.8)
    val_size = int(data.__len__() - train_size)
    train_data, val_data = torch.utils.data.random_split(data, [train_size, val_size])
    # convert to dataloader
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False) 
    # ---
    return train_loader, val_loader


class SingleStepDynamicsDataset(Dataset):
    """
    Each data sample is a dictionary containing (x_t, u_t, x_{t+1}) in the form:
    {'state': x_t,
     'action': u_t,
     'next_state': x_{t+1},
    }
    where:
     x_t: torch.float32 tensor of shape (state_size,)
     u_t: torch.float32 tensor of shape (action_size,)
     x_{t+1}: torch.float32 tensor of shape (state_size,)
    """

    def __init__(self, collected_data):
        self.data = collected_data
        self.trajectory_length = self.data[0]['actions'].shape[0]

    def __len__(self):
        return len(self.data) * self.trajectory_length

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def __getitem__(self, item):
        """
        Return the data sample corresponding to the index <item>.
        :param item: <int> index of the data sample to produce.
            It can take any value in range 0 to self.__len__().
        :return: data sample corresponding to encoded as a dictionary with keys (state, action, next_state).
        The class description has more details about the format of this data sample.
        """
        # --- Your code here
        # flat indexes
        traj_idx = item // self.trajectory_length
        path_idx = item % self.trajectory_length
        # items
        state = self.data[traj_idx]['states'][path_idx]
        action = self.data[traj_idx]['actions'][path_idx]
        state_next = self.data[traj_idx]['states'][path_idx + 1]
        # Convert to torch.float32 tensors
        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.float32)
        state_next = torch.tensor(state_next, dtype=torch.float32)
        # store samples
        sample = {
            'state': state,
            'action': action,
            'next_state': state_next
        }
        # ---
        return sample


class MultiStepDynamicsDataset(Dataset):
    """
    Dataset containing multi-step dynamics data.

    Each data sample is a dictionary containing (state, action, next_state) in the form:
    {'state': x_t, -- initial state of the multipstep torch.float32 tensor of shape (state_size,)
     'action': [u_t,..., u_{t+num_steps-1}] -- actions applied in the muli-step.
                torch.float32 tensor of shape (num_steps, action_size)
     'next_state': [x_{t+1},..., x_{t+num_steps} ] -- next multiple steps for the num_steps next steps.
                torch.float32 tensor of shape (num_steps, state_size)
    }
    """

    def __init__(self, collected_data, num_steps=4):
        self.data = collected_data
        self.trajectory_length = self.data[0]['actions'].shape[0] - num_steps + 1
        self.num_steps = num_steps

    def __len__(self):
        return len(self.data) * (self.trajectory_length)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def __getitem__(self, item):
        """
        Return the data sample corresponding to the index <item>.
        :param item: <int> index of the data sample to produce.
            It can take any value in range 0 to self.__len__().
        :return: data sample corresponding to encoded as a dictionary with keys (state, action, next_state).
        The class description has more details about the format of this data sample.
        """
        # --- Your code here
        # define index
        traj_idx = item // self.trajectory_length
        path_idx = item % self.trajectory_length
        # items
        state = self.data[traj_idx]['states'][path_idx]
        action = self.data[traj_idx]['actions'][path_idx : path_idx + self.num_steps]
        state_next= self.data[traj_idx]['states'][path_idx + 1 : path_idx + self.num_steps + 1]
        # convert them
        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.float32)
        state_next = torch.tensor(state_next, dtype=torch.float32)
        # store return value
        sample = {
            'state': state,
            'action': action,
            'next_state': state_next
        }
        # ---
        return sample


class SE2PoseLoss(nn.Module):
    """
    Compute the SE2 pose loss based on the object dimensions (block_width, block_length).
    Need to take into consideration the different dimensions of pose and orientation to aggregate them.

    Given a SE(2) pose [x, y, theta], the pose loss can be computed as:
        se2_pose_loss = MSE(x_hat, x) + MSE(y_hat, y) + rg * MSE(theta_hat, theta)
    where rg is the radious of gyration of the object.
    For a planar rectangular object of width w and length l, the radius of gyration is defined as:
        rg = ((l^2 + w^2)/12)^{1/2}

    """

    def __init__(self, block_width, block_length):
        super().__init__()
        self.w = torch.tensor(block_width, dtype=torch.float32) 
        self.l = torch.tensor(block_length, dtype=torch.float32)
        self.rg = torch.sqrt((self.w**2 + self.l**2)/12)

    def forward(self, pose_pred, pose_target):
        # --- Your code here
        # mse_x = torch.mean(torch.square(pose_pred[...:0] - pose_target[...:0]))
        # mse_y = torch.mean(torch.square(pose_pred[...:1] - pose_target[...:1]))
        # mse_theta = self.rg * torch.mean(torch.square(pose_pred[...:2] - pose_target[...:2]))
        mse_x = torch.mean(torch.square(pose_pred[:, 0] - pose_target[:, 0]))  # Select x coordinate
        mse_y = torch.mean(torch.square(pose_pred[:, 1] - pose_target[:, 1]))  # Select y coordinate
        mse_theta = self.rg * torch.mean(torch.square(pose_pred[:, 2] - pose_target[:, 2]))  # Select theta
        se2_pose_loss = mse_x + mse_y + mse_theta
        # ---
        return se2_pose_loss


class SingleStepLoss(nn.Module):

    def __init__(self, loss_fn):
        super().__init__()
        self.loss = loss_fn

    def forward(self, model, state, action, target_state):
        """
        Compute the single step loss resultant of querying model with (state, action) and comparing the predictions with target_state.
        """
        # single_step_loss
        # --- Your code here
        state_pred = model(state, action)
        single_step_loss = self.loss(state_pred, target_state)
        # ---
        return single_step_loss


class MultiStepLoss(nn.Module):

    def __init__(self, loss_fn, discount=0.99):
        super().__init__()
        self.loss = loss_fn
        self.discount = discount

    def forward(self, model, state, actions, target_states):
        """
        Compute the multi-step loss resultant of multi-querying the model from (state, action) and comparing the predictions with targets.
        """
        multi_step_loss = 0
        steps = actions.shape[1]
        # --- Your code here
        for i in range(steps):
            state_pred = model(state, actions[:,i,:])
            discounted_loss = self.loss(state_pred, target_states[:,i,:]) * self.discount**i
            multi_step_loss += discounted_loss
            state = state_pred
        # ---
        return multi_step_loss


class AbsoluteDynamicsModel(nn.Module):
    """
    Model the absolute dynamics x_{t+1} = f(x_{t},a_{t})
    """

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.H = 100
        # --- Your code here
        self.input_dim = self.state_dim + self.action_dim
        self.linear1 = nn.Linear(self.input_dim, self.H)
        self.act1 = nn.ReLU()
        self.linear2 = nn.Linear(self.H, self.H)
        self.act2 = nn.ReLU()
        self.linear3 = nn.Linear(self.H, self.state_dim)
        # ---

    def forward(self, state, action):
        """
        Compute next_state resultant of applying the provided action to provided state
        :param state: torch tensor of shape (..., state_dim)
        :param action: torch tensor of shape (..., action_dim)
        :return: next_state: torch tensor of shape (..., state_dim)
        """
        # --- Your code here
        x = torch.cat((state, action), dim=-1)  
        next_state = self.linear1(x)
        next_state = self.act1(next_state)
        next_state = self.linear2(next_state)
        next_state = self.act2(next_state)
        next_state = self.linear3(next_state)
        # ---
        return next_state


class ResidualDynamicsModel(nn.Module):
    """
    Model the residual dynamics s_{t+1} = s_{t} + f(s_{t}, u_{t})

    The network predicts the state difference (Δs) given the state and action.
    """

    def __init__(self, state_dim, action_dim, hidden_dim=100):
        super(ResidualDynamicsModel, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Define neural network layers
        self.linear1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.act1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.act2 = nn.ReLU()
        self.linear3 = nn.Linear(hidden_dim, state_dim)  # Outputs Δs

    def forward(self, state, action):
        """
        Compute next_state by adding residual prediction to current state
        :param state: torch tensor of shape (..., state_dim)
        :param action: torch tensor of shape (..., action_dim)
        :return: next_state: torch tensor of shape (..., state_dim)
        """
        x = torch.cat((state, action), dim=-1)  
        delta_state = self.act1(self.linear1(x))
        delta_state = self.act2(self.linear2(delta_state))
        delta_state = self.linear3(delta_state)

        next_state = state + delta_state  # Residual connection
        return next_state


def free_pushing_cost_function(state, action):
    """
    Compute the state cost for MPPI on a setup without obstacles.
    :param state: torch tensor of shape (B, state_size)
    :param action: torch tensor of shape (B, state_size)
    :return: cost: torch tensor of shape (B,) containing the costs for each of the provided states
    """
    target_pose = TARGET_POSE_FREE_TENSOR  # torch tensor of shape (3,) containing (pose_x, pose_y, pose_theta)
    # cost = None
    # --- Your code here
    Q = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 0.1]], dtype=torch.float32)
    diff = state - target_pose
    cost = torch.einsum('bi,ij,bj -> b', diff, Q, diff)
    # for item in state:
    #     cost += (state[item] - target_pose).T * Q *(state [item] - target_pose)
    # ---
    return cost


def collision_detection(state):
    """
    Checks if the state is in collision with the obstacle.
    The obstacle geometry is known and provided in obstacle_centre and obstacle_halfdims.
    :param state: torch tensor of shape (B, state_size)
    :return: in_collision: torch tensor of shape (B,) containing 1 if the state is in collision and 0 if not.
    """
    obstacle_centre = OBSTACLE_CENTRE_TENSOR  # torch tensor of shape (2,) consisting of obstacle centre (x, y)
    obstacle_dims = 2 * OBSTACLE_HALFDIMS_TENSOR  # torch tensor of shape (2,) consisting of (w_obs, l_obs)
    box_size = BOX_SIZE  # scalar for parameter w
    in_collision = None
    # --- Your code here
    state_xy = state[:,:2]
    max = obstacle_centre + obstacle_dims/2
    min = obstacle_centre - obstacle_dims/2
    in_collision = ((state_xy >= min) & (state_xy <= max)).all(dim = 1).float()
    return in_collision


def obstacle_avoidance_pushing_cost_function(state, action):
    """
    Compute the state cost for MPPI on a setup with obstacles.
    :param state: torch tensor of shape (B, state_size)
    :param action: torch tensor of shape (B, state_size)
    :return: cost: torch tensor of shape (B,) containing the costs for each of the provided states
    """
    target_pose = TARGET_POSE_OBSTACLES_TENSOR  # torch tensor of shape (3,) containing (pose_x, pose_y, pose_theta)
    cost = None
    # --- Your code here
    Q = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 0.1]], dtype=torch.float32)
    diff = state - target_pose
    cost = torch.einsum('bi, ij, bj -> b', diff, Q, diff)

    in_collision = 100 * collision_detection(state)
    cost += in_collision
    # ---
    return cost


class PushingController(object):
    """
    MPPI-based controller
    Since you implemented MPPI on HW2, here we will give you the MPPI for you.
    You will just need to implement the dynamics and tune the hyperparameters and cost functions.
    """

    def __init__(self, env, model, cost_function, num_samples=100, horizon=10):
        self.env = env
        self.model = model
        self.target_state = None
        # MPPI Hyperparameters:
        # --- You may need to tune them
        state_dim = env.observation_space.shape[0]
        u_min = torch.from_numpy(env.action_space.low)
        u_max = torch.from_numpy(env.action_space.high)
        noise_sigma = 0.65 * torch.eye(env.action_space.shape[0])
        lambda_value = 0.01
        # ---
        from mppi import MPPI
        self.mppi = MPPI(self._compute_dynamics,
                         cost_function,
                         nx=state_dim,
                         num_samples=num_samples,
                         horizon=horizon,
                         noise_sigma=noise_sigma,
                         lambda_=lambda_value,
                         u_min=u_min,
                         u_max=u_max)

    def _compute_dynamics(self, state, action):
        """
        Compute next_state using the dynamics model self.model and the provided state and action tensors
        :param state: torch tensor of shape (B, state_size)
        :param action: torch tensor of shape (B, action_size)
        :return: next_state: torch tensor of shape (B, state_size) containing the predicted states from the learned model.
        """
        # next_state = None
        # --- Your code here
        next_state = self.model(state, action)
        # ---
        return next_state

    def control(self, state):
        """
        Query MPPI and return the optimal action given the current state <state>
        :param state: numpy array of shape (state_size,) representing current state
        :return: action: numpy array of shape (action_size,) representing optimal action to be sent to the robot.
        TO DO:
         - Prepare the state so it can be send to the mppi controller. Note that MPPI works with torch tensors.
         - Unpack the mppi returned action to the desired format.
        """
        # action = None
        # state_tensor = None
        # --- Your code here
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0)
        # ---
        action_tensor = self.mppi.command(state_tensor)
        # --- Your code here
        action = action_tensor.squeeze(dim=0).detach().cpu().numpy()
        # ---
        return action

# =========== AUXILIARY FUNCTIONS AND CLASSES HERE ===========
# --- Your code here
def train_step(model,loss,train_loader, optimizer) -> float:
    """
    Performs an epoch train step.
    :param model: Pytorch nn.Module
    :param train_loader: Pytorch DataLoader
    :param optimizer: Pytorch optimizer
    :return: train_loss <float> representing the average loss among the different mini-batches.
        Loss needs to be MSE loss.
    """
    model.train()
    train_loss = 0. # TODO: Modify the value
    for batch_idx, batch in enumerate(train_loader):
        state = batch['state']
        action = batch['action']
        next_state = batch['next_state']
        # --- Your code here
        optimizer.zero_grad()
        # next_pred = model(state, action)  
        mse = loss(model, state, action, next_state)
        mse.backward()
        optimizer.step()
        # ---
        train_loss += mse.item()
    return train_loss/len(train_loader)


def val_step(model,loss, val_loader) -> float:
    """
    Perfoms an epoch of model performance validation
    :param model: Pytorch nn.Module
    :param train_loader: Pytorch DataLoader
    :param optimizer: Pytorch optimizer
    :return: val_loss <float> representing the average loss among the different mini-batches
    """
    model.eval()
    val_loss = 0. # TODO: Modify the value
    # ---
    for batch_idx, batch in enumerate(val_loader):
        state = batch['state']
        action = batch['action']
        next_state = batch['next_state']
        # --- Your code here
        mse = loss(model, state, action, next_state)
        # ---
        val_loss += mse.item()
    return val_loss/len(val_loader)

def train_model(model, loss, train_dataloader, val_dataloader, num_epochs=100, lr=1e-3):
    """
    Trains the given model for `num_epochs` epochs. Use SGD as an optimizer.
    You may need to use `train_step` and `val_step`.
    :param model: Pytorch nn.Module.
    :param train_dataloader: Pytorch DataLoader with the training data.
    :param val_dataloader: Pytorch DataLoader with the validation data.
    :param num_epochs: <int> number of epochs to train the model.
    :param lr: <float> learning rate for the weight update.
    :return:
    """
    # Initialize the optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr)
    # --- Your code here
    # model = LinearRegressor()
    # ---
    pbar = tqdm(range(num_epochs))
    train_losses = []
    val_losses = []
    for epoch_i in pbar:
        # --- Your code here
        train_loss_i = train_step(model, loss, train_dataloader, optimizer)
        val_loss_i = val_step(model, loss, val_dataloader)
        # ---
        pbar.set_description(f'Train Loss: {train_loss_i:.4f} | Validation Loss: {val_loss_i:.4f}')
        train_losses.append(train_loss_i)
        val_losses.append(val_loss_i)
    
    return train_losses, val_losses


    # # helper function
    # def create_loader(data, batch_size):
    #     loader = []
    #     num_batches = data.__len__() // batch_size
    #     # looping for first n-1 batches
    #     for idx in range(num_batches):
    #         # define empty batch
    #         batch = []
    #         # get items
    #         for item in range(idx * batch_size, (idx + 1) * batch_size):
    #             batch.append(data[item])
    #         loader.append(batch)
    #     # loop for last batch
    #     if data.__len__() % batch_size != 0:
    #         batch = []
    #         for item in range(num_batches * batch_size, data.__len__()):
    #             batch.append(data[item])
    #         loader.append(batch)
    #     return loader
    
    # train_loader = create_loader(train_data, batch_size)
    # val_loader = create_loader(val_data, batch_size)
    # train_loader = DataLoader(train_loader, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_loader, batch_size=batch_size, shuffle=False)

    # train_loader = LoaderWrapper(train_loader, train_data)
    # val_loader = LoaderWrapper(val_loader, val_data)

# ---
# ============================================================
