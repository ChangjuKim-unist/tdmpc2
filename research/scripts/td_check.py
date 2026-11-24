import torch,mujoco,dm_control,tensordict,torchrl,hydra
print(torch.__version__, torch.__file__, torch.cuda.is_available(), mujoco.__version__, tensordict.__version__, torchrl.__version__)
from dm_control import suite
e=suite.load("walker","run")
print(e.reset().observation.keys())
