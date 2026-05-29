import mujoco
import gymnasium
import torch
import stable_baselines3
import tensorboard

print("MuJoCo:", mujoco.__version__)
print("PyTorch:", torch.__version__)
print("Stable-Baselines3:", stable_baselines3.__version__)
print("TensorBoard:", tensorboard.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Everything works!")