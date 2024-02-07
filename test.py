import torch
import numpy as np
import pygame
import gym_super_mario_bros
import sys
sys.path.append('./src')

from model import Model
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

# Source:
# https://stackoverflow.com/questions/287871/how-do-i-print-colored-text-to-the-terminal
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def test_model(model: Model):
    """Tests the output shapes of the model against expected shapes

    Parameters:
        model (Model): Instance of the Model class being tested
    """
    batch_size = 5  # Dummy input tensor for batch size 5
    dummy_input = torch.randn(batch_size, num_inputs, 84, 84)  # Random tensor
    actor_output, critic_output = model(dummy_input)

    assert actor_output.shape == (batch_size, num_actions), "Incorrect actor output shape!"
    assert critic_output.shape == (batch_size, 1), "Incorrect critic output shape!"

    print(bcolors.OKGREEN + "PASSED Test 1: Model's output shapes are correct." + bcolors.OKGREEN)

if __name__ == "__main__":
    np.bool8 = np.bool_ # Avoid warnings
    num_inputs = 3
    num_actions = 10
    model = Model(num_inputs, num_actions)
    test_model(model)

    pygame.init()
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0',
            apply_api_compatibility=True, render_mode="human")
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    key_action_mapping = {
        pygame.K_RIGHT: 1,  # Walk right
        pygame.K_SPACE: 2,  # Jump + right
        pygame.K_DOWN: 3,   # Run + right
        pygame.K_a: 4,      # Run / shoot fireballs
        pygame.K_UP: 5,     # Jump up directly
        pygame.K_LEFT: 6,   # Walk left
    }

    screen = pygame.display.set_mode((84, 84))  # NES resolution
    clock = pygame.time.Clock()

    done = False
    env.reset()
    last_action = 0

    while not done:
        env.render()
        action = last_action

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key in key_action_mapping:
                    action = key_action_mapping[event.key]
                    last_action = action
            elif event.type == pygame.QUIT:
                done = True
            else:
                last_action = 0

        obs, reward, terminated, truncated, info = env.step(action)
        clock.tick(60)

    pygame.quit()
    env.close()