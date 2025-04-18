# Adapted from diffusers
import math
import numpy as np
import torch


def betas_for_alpha_bar(num_diffusion_timesteps, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
    (1-beta) over time from t = [0,1].

    Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
    to that part of the diffusion process.


    Args:
        num_diffusion_timesteps (`int`): the number of betas to produce.
        max_beta (`float`): the maximum beta to use; use values lower than 1 to
                     prevent singularities.

    Returns:
        betas (`np.ndarray`): the betas used by the scheduler to step the model outputs
    """

    def alpha_bar(time_step):
        return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


class DDPMScheduler:
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "squaredcos_cap_v2",
    ):
        if beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, dtype=torch.float32)
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide cosine schedule
            self.betas = betas_for_alpha_bar(num_train_timesteps)

        self.num_train_timesteps = num_train_timesteps
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.one = torch.tensor(1.0)

        self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy())


    def _get_variance(self, t):
        prev_t = self.previous_timestep(t)

        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev

        # For t > 0, compute predicted variance βt (see formula (6) and (7) from https://arxiv.org/pdf/2006.11239.pdf)
        # and sample from it to get previous sample
        # x_{t-1} ~ N(pred_prev_sample, variance) == add variance to pred_sample
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t

        # we always take the log of variance, so clamp it to ensure it's not 0
        variance = torch.clamp(variance, min=1e-20)

        variance = variance

        return variance


    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        """
        Diffusion kernel that transforms x_0 to x_t given noise.

            x_t = \sqrt{\bar{\alpha_t}}x_0 + \sqrt{1-\bar{\alpha_t}}\epsilon

        Args:
            original_samples: x_0, shape: (B, H, D)
            noise: \epsilon, shape: (B, H, D)
            timesteps: t, shape: (B,)

        Returns:
            noisy_samples: x_t, shape: (B, H, D)
        """
        # Make sure alphas_cumprod and timestep have same device and dtype as original_samples
        alphas_cumprod = self.alphas_cumprod.to(device=original_samples.device, dtype=original_samples.dtype)
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = None
        sqrt_one_minus_alpha_prod = None
        noisy_samples = None
        ################################
        #######  Your code here  #######
        alphas_cumprod = alphas_cumprod[timesteps]
        sqrt_alpha_prod = torch.sqrt(alphas_cumprod).view(-1, 1, 1)
        sqrt_one_minus_alpha_prod = torch.sqrt(1 - alphas_cumprod).view(-1, 1, 1)
        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise

        #######  Your code finish  #######
        ##################################
        return noisy_samples
    
    def step(
        self,
        pred_noise: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        add_noise: bool = True,
    ) -> torch.FloatTensor:
        """
        Predicts x_{t-1} from x_t using the model's noise prediction. This reverses one step of the
        forward diffusion process.

        Args:
            pred_noise (torch.FloatTensor): Model's predicted noise ε_θ(x_t, t), shape: (B, H, D)
            timestep (int): Current timestep t in the diffusion chain, shape: (B,)
            sample (torch.FloatTensor): Current noisy sample x_t, shape: (B, H, D)

        Returns:
            prev_sample: Predicted x_{t-1}, shape: (B, H, D)
    
        """
        t = timestep
        prev_t = self.previous_timestep(t)

        # compute alphas, betas
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t = alpha_prod_t.to(device=pred_noise.device, dtype=pred_noise.dtype)

        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        alpha_prod_t_prev = alpha_prod_t_prev.to(device=pred_noise.device, dtype=pred_noise.dtype)

        alpha_prod_t = alpha_prod_t.view(-1, 1, 1)
        alpha_prod_t_prev = alpha_prod_t_prev.view(-1, 1, 1)
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # compute predicted original sample, pred_original_sample_coeff and current_sample_coeff
        pred_original_sample = None
        pred_original_sample_coeff = None
        current_sample_coeff = None

        ################################
        #######  Your code here  #######
        # Ensure shapes and types
        # Step 1: Predict x0 from xt and noise
        sqrt_alpha_prod_t = torch.sqrt(alpha_prod_t)
        sqrt_one_minus_alpha_prod_t = torch.sqrt(1 - alpha_prod_t)
        pred_original_sample = (sample - sqrt_one_minus_alpha_prod_t * pred_noise) / sqrt_alpha_prod_t

        # Step 2: Compute coefficients
        sqrt_alpha_prod_t_prev = torch.sqrt(alpha_prod_t_prev)
        denom = torch.clamp(1 - alpha_prod_t, min=1e-10)

        # Use the already computed current_beta_t
        pred_original_sample_coeff = (sqrt_alpha_prod_t_prev * current_beta_t) / denom
        current_sample_coeff = (current_alpha_t * (1 - alpha_prod_t_prev)) / denom
        #######  Your code finish  #######
        ##################################
        pred_original_sample = torch.clamp(pred_original_sample, -1, 1)
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample
        variance = 0
        if t > 0 and add_noise:
            variance_noise = torch.randn_like(pred_noise)
            variance = (self._get_variance(t) ** 0.5) * variance_noise

        pred_prev_sample = pred_prev_sample + variance
        return pred_prev_sample



    def __len__(self):
        return self.num_train_timesteps

    def previous_timestep(self, timestep):
        prev_t = timestep - 1
        return prev_t
