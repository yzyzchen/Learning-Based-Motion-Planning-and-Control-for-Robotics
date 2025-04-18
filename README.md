# Robot Learning

![Robot learning](https://img.shields.io/badge/robot-learning-orange)
![Primary language](https://img.shields.io/badge/Python-100.0%25-red)
![License](https://img.shields.io/badge/license-MIT-green)

- **Learning-Based Robot Planning with PPO and Diffusion Policy**

Implemented and compared Proximal Policy Optimization (PPO) and Diffusion Policy for robot motion planning and control, focusing on continuous action spaces in manipulation tasks.
<table>
  <tr>
    <td align="center">
      <b>PPO</b><br>
      <img src="assets/PPO.gif" width="300px">
    </td>
    <td align="center">
      <b>Diffusion Policy</b><br>
      <img src="assets/DP.gif" width="300px">
    </td>
  </tr>
</table>

- **Vision-Based Robot Control in Latent Space:**

Encoded state images into a latent space using a Variational Autoencoder (VAE), then applied Model Predictive Path Integral (MPPI) control in the latent space to drive the robot arm toward the desired state.

<!-- ![Demo](assets/VAE.gif)
![Diagram](assets/VAE.png) -->
<table>
  <tr>
    <td align="center">
      <b>VAE</b><br>
      <img src="assets/VAE.gif" width="300px">
    </td>
    <td align="center">
      <b>state image</b><br>
      <img src="assets/VAE.png" width="300px">
    </td>
  </tr>
</table>


- **Gaussian Process Based Robot pushing with Obstacle Avoidance**
Implemented Gaussian Process to model uncertain dynamics and applied Model Predictive Path Integral (MPPI) control for obstacle-aware object pushing.

![Demo](assets/0.gif)

