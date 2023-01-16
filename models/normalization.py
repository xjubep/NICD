import torch
from torch import nn


class L2N(nn.Module):

    def __init__(self, eps=1e-6):
        super(L2N, self).__init__()
        self.eps = eps

    def l2n(self, x, eps=1e-6):
        return x / (torch.norm(x, p=2, dim=1, keepdim=True) + eps).expand_as(x)

    def forward(self, x):
        return self.l2n(x, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'eps=' + str(self.eps) + ')'


if __name__ == '__main__':
    print(L2N())
