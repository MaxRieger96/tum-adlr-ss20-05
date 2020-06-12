import torch
from torch.nn import Linear
from torch.nn import Module
from torch.nn.functional import softmax

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-8


class Model(Module):
    def __init__(self, input_dims: int, outputs: int):
        super(Model, self).__init__()
        self.policy_logits = Linear(input_dims, outputs)
        self.value = Linear(input_dims, 1)

    def forward(self, x):
        # TODO
        value = self.value(x)
        policy_logits = self.policy_logits(x)
        policy = softmax(policy_logits, dim=1)
        policy = policy + EPS
        return policy, value
