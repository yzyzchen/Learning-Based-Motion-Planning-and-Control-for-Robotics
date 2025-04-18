import numpy as np
from conditional_unet1d import ConditionalUnet1D
from diffusion_scheduler import DDPMScheduler
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import yaml
import pickle

# normalize data
def get_data_stats(data):
    data = data.reshape(-1,data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats


def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata, stats):
    # ndata = np.clip(ndata, -1, 1)
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data


class MultiStepDynamicsDataset(Dataset):
    """
    Dataset containing multi-step dynamics data.
    Each data sample is a dictionary containing (state, action, next_state) in the form:
    {'states':[x_{t}, x_{t+1},..., x_{t+num_steps} ] -- initial state of the multistep torch.float32 tensor of shape (state_size,)
     'actions': [u_t,..., u_{t+num_steps-1}] -- actions applied in the multi-step.
                torch.float32 tensor of shape (num_steps, action_size)
    }

    Observation: If num_steps=1, this dataset is equivalent to SingleStepDynamicsDataset.
    """

    def __init__(self, 
                 data,
                 pred_horizon,
                 obs_horizon,
                 action_horizon,
                 ):
        self.data = data
        self.trajectory_length = self.data['actions'].shape[1] - pred_horizon + 1
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.pred_horizon = pred_horizon
        self.num_steps = pred_horizon
        self.stats = dict()
        self.normalized_data = dict()
        for k, v in self.data.items():
            self.stats[k] = get_data_stats(v)
            self.normalized_data[k] = normalize_data(v, self.stats[k])



    def __len__(self):
        return len(self.data['states']) * self.trajectory_length

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def __getitem__(self, item):
        """
        Return the data sample corresponding to the index <item>.
        :param item: <int> index of the data sample to produce.
            It can take any value in range 0 to self.__len__().
        :return: data sample corresponding to encoded as a dictionary with keys (states, actions).
        The class description has more details about the format of this data sample.
        """
        sample = {
            'states': None,
            'actions': None,
        }
        # --- Your code here
        trajectory_indx = item // self.trajectory_length
        step_indx = item % self.trajectory_length 
        sample['states'] = torch.tensor(
            self.normalized_data['states'][trajectory_indx, step_indx:step_indx + self.obs_horizon + 1]).to(torch.float32)
        sample['actions'] = torch.tensor(
            self.normalized_data['actions'][trajectory_indx, step_indx:step_indx + self.num_steps]
        ).to(torch.float32)
   
        return sample

class DiffusionPolicy:
    def __init__(
            self,
            obs_dim,
            action_dim,
            obs_horizon,
            pred_horizon,
            action_horizon,
            num_timesteps,
            device
            ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.num_timesteps = num_timesteps
        self.device = device
        self.model = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=obs_dim*obs_horizon
        )
        self.model.to(device)
        self.noise_schedule = DDPMScheduler(
            num_train_timesteps=num_timesteps,
            beta_schedule='squaredcos_cap_v2',
        )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=2e-4)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_timesteps
        )
    
    def train_step(self, nbatch):
        """
        Perform a single training step.
        Args
        """
        nobs = nbatch['states'].to(self.device)
        naction = nbatch['actions'].to(self.device)
        B = nobs.shape[0]

        # observation as FiLM conditioning
        # (B, obs_horizon, obs_dim)
        obs_cond = nobs[:, :self.obs_horizon, :]
        # (B, obs_horizon * obs_dim)
        obs_cond = obs_cond.flatten(start_dim=1)

        loss = None
        ################################
        #######  Your code here  #######
        # 1. sample noise to add to actions
        # 2. sample a diffusion iteration for each data point
        # 3. add noise to the actions according to the noise magnitude at each diffusion iteration
        # (this is the forward diffusion process)
        # 4. predict the noise residual
        # 5. calculate the loss (MSE on noise)

        # 1. Sample random noise
        noise = torch.randn_like(naction)

        # 2. Sample random timesteps (B,)
        timesteps = torch.randint(0, self.noise_schedule.num_train_timesteps, (B,), device=self.device).long()

        # 3. Add noise to the actions using the forward process
        noisy_actions = self.noise_schedule.add_noise(naction, noise, timesteps)

        # 4. Predict the noise using the model
        noise_pred = self.model(noisy_actions, timesteps, global_cond=obs_cond)

        # 5. Compute loss
        loss_fn = nn.MSELoss(reduction='mean')
        loss = loss_fn(noise_pred, noise)
        #######  Your code finish  #######
        ##################################

        # optimize
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.lr_scheduler.step()
        return loss.item()
    
    def predict(self, nstates):
        """
        """
        with torch.no_grad():
            obs_cond = nstates.unsqueeze(0).flatten(start_dim=1)
            noisy_action = torch.randn(
                (1, self.pred_horizon, self.action_dim), device=self.device)
            naction = noisy_action

            #################################
            #######  Your code here  ########
            for t in reversed(range(self.noise_schedule.num_train_timesteps)):
                t_tensor = torch.full((1,), t, device=self.device, dtype=torch.long)
                # Predict the noise residual
                pred_noise = self.model(naction, t_tensor, global_cond=obs_cond)
                # Reverse diffusion step
                naction = self.noise_schedule.step(pred_noise, t, naction, add_noise=True)

            #######  Your code finish  #######
            ##################################
        # (B, pred_horizon, action_dim)
        naction = naction.detach().to('cpu').numpy()
        naction = naction[0]

        # only take action_horizon number of actions
        start = self.obs_horizon - 1
        end = start + self.action_horizon
        action = naction[start:end,:]
        return action
    

def save_diffusion_policy(policy, path="dp_model.pt"):
    """
    Save diffusion policy parameters and weights
    
    Args:
        policy: DiffusionPolicy instance
        save_dir: Directory to save the files
    """
    
    # Save parameters as YAML
    params = {
        'obs_dim': policy.obs_dim,
        'action_dim': policy.action_dim,
        'obs_horizon': policy.obs_horizon,
        'pred_horizon': policy.pred_horizon,
        'action_horizon': policy.action_horizon,
        'num_timesteps': policy.num_timesteps
    }
    
    # Save parameters
    with open('config.yaml', 'w') as f:
        yaml.dump(params, f, default_flow_style=False)
    
    # Save model weights
    torch.save(policy.model.state_dict(), path)

def load_diffusion_policy(path="dp_model.pt", device='cuda'):
    """
    Load diffusion policy from saved files
    
    Args:
        save_dir: Directory containing saved files
        device: Device to load the model to
    
    Returns:
        policy: Loaded DiffusionPolicy instance
    """
    # Load parameters
    with open('config.yaml', 'r') as f:
        params = yaml.safe_load(f)
    stats = pickle.load(open('stats.pkl', 'rb'))
    # Create policy
    policy = DiffusionPolicy(
        obs_dim=params['obs_dim'],
        action_dim=params['action_dim'],
        obs_horizon=params['obs_horizon'],
        pred_horizon=params['pred_horizon'],
        action_horizon=params['action_horizon'],
        num_timesteps=params['num_timesteps'],
        device=device
    )
    
    # Load weights
    checkpoint = torch.load(path, map_location=device)
    policy.model.load_state_dict(checkpoint)
    return policy, stats