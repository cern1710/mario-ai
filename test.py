import torch
import sys
sys.path.append('./src')

from model import Model

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
    num_inputs = 3
    num_actions = 10
    model = Model(num_inputs, num_actions)
    test_model(model)