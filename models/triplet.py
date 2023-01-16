from torch import nn


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, *x, single=False):
        if single:
            return self.forward_single(x[0])
        else:
            return self.forward_triple(x[0], x[1], x[2])

    def forward_single(self, x):
        output = self.embedding_net(x)
        return output

    def forward_triple(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)

    def __str__(self):
        return f'[{super(TripletNet, self).__str__()}]{self.embedding_net.__str__()}'
