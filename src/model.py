import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Tuple

class Model(nn.Module):
    """Neural network model optimised for Proximal Policy Optimisation (PPO)

    This model features a combination of convolutional layer for input processing,
    followed by fully connected layers for policy and value estimation, allowing
    both action policies and state-value predictions to be learnt from inputs.

    Attributes:
        conv (nn.ModuleList): List of convolutional layers for feature extraction
        linear (nn.Linear): Linear layer processing flattened output of conv layers
        critic (nn.Linear): Outputs a single value representing state value
        actor (nn.Linear): Outputs a vector of action probabilities
    """
    def __init__(self, num_inputs: int, num_actions: int) -> None:
        """Initializes the model with convolutional and linear layers

        Parameters:
            num_inputs (int): Number of input channels in the input image
            num_actions (int): Number of possible actions the agent can take
        """
        super().__init__()
        self._init_conv_layers(num_inputs)

        input_size = torch.zeros(1, num_inputs, 84, 84)
        output_size = self._forward_conv(input_size).view(-1).shape[0]

        self.linear = nn.Linear(output_size, 512)
        self.critic = nn.Linear(512, 1)
        self.actor = nn.Linear(512, num_actions)

        # TODO: initialise weights

    def _init_conv_layers(self, num_inputs: int) -> None:
        """Initialise convolutional layers as ModuleList object

        Currently based on a modified version of DeepMind's deep Q-Network
        https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf

        Parameters:
            num_inputs (int): Number of input channels in the input image
        """
        self.conv = nn.ModuleList([
            nn.Conv2d(num_inputs, 32, kernel_size=8, stride=4),
            nn.Conv2d(32, 32, kernel_size=4, stride=2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1)
        ])

    def _forward_conv(self, x: Tensor) -> Tensor:
        """Apply ReLU activation after each convolutional layer

        Parameters:
            x (Tensor): Input tensor of the model

        Returns:
            Tensor: Output tensor after processing through convolutional layers
        """
        for layer in self.conv:
            x = F.relu(layer(x))
        return x

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass

        Parameters:
            x (Tensor): Input tensor of the model

        Returns:
            tuple[Tensor, Tensor]: Output of actor and critic layers
        """
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)   # Flatten output for linear layer
        x = F.relu(self.linear(x))
        return self.actor(x), self.critic(x)