import torch.nn as nn
import torch
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

class RayFFN(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
        
        self.flatten = nn.Flatten(-2, -1)

        self._representation_head = nn.Sequential(
            self.flatten
        )

        self.fc1 = nn.Linear(25, 256)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(256, num_outputs)

        self._action_branch = nn.Sequential(
            self.fc1,
            self.act1,
            self.fc2
        )

        self.fc3 = nn.Linear(25, 1)

        self._value_branch = nn.Sequential(
            self.fc3
        )

    def forward(self, input_dict, state, seq_lens):
        #TODO: still need to make sure obs is correct
        x = input_dict["obs"].to(torch.float32)
        self._hidden_out = self._representation_head(x)
        x = self._action_branch(self._hidden_out)
        return x, []

    def value_function(self):
        assert self._hidden_out is not None, "must call forward first!"
        return torch.reshape(self._value_branch(self._hidden_out), [-1])