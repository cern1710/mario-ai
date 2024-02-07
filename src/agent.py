import torch
import torch.nn as nn
import torch.nn.functional as F

from nes_py.wrappers import JoypadSpace
from torch import Tensor

class Agent:
    """TODO: finish this

    Attributes:
        env (JoypadSpace):
        discount (float):
        learning_rate (float):
        entropy_coeff (float):
        epsilon (float):
        tau (float):
        num_epochs (int):
        batch_size (int):
        save_dir (str):
    """
    def __init__(
            self,
            env: JoypadSpace,
            discount: float,
            learning_rate: float,
            entropy_coeff: float,
            epsilon: float,
            tau: float,
            num_epochs: int,
            batch_size: int,
            save_dir: str,
        ) -> None:
        self.env = env
        self.discount = discount
        self.learning_rate = learning_rate
        self.entropy_coeff = entropy_coeff
        self.epsilon = epsilon
        self.tau = tau
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.save_dir = save_dir