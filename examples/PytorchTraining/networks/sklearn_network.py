import torch


class SklearnNet(torch.nn.Module):
    def __init__(self):
        super(SklearnNet, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(30,50),
            torch.nn.ReLU(),
            torch.nn.Linear(50,50),
            torch.nn.ReLU(),
            torch.nn.Linear(50,1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = torch.div(x, 5000.0)
        # print(x[0,:])
        return self.network(x)