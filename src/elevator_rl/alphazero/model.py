import torch
from torch.nn import Linear
from torch.nn import Module
from torch.nn.functional import softmax

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-8


class Model(Module):
    def __init__(self, input_dims: int, outputs: int):
        super(Model, self).__init__()
        self.policy_l1 = Linear(input_dims, 2*outputs)
        self.policy_logits = Linear(2*outputs, outputs)
        self.value_l1 = Linear(input_dims, 30)
        self.value = Linear(30, 1)

    def forward(self, x):
        # TODO
        value_l1 = self.value_l1(x)
        value = self.value(value_l1)
        policy_logits = self.policy_logits(self.policy_l1(x))
        policy = softmax(policy_logits, dim=0) # TODO check all of this
        policy = policy + EPS
        return policy, value
