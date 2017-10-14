from torch import nn
import torch
from torch.autograd import Variable


class ContentLoss(nn.Module):
    """
    Must be initialized with a "loss network"
    Example use  VGG relu2_2 as loss network:
    vgg = vgg16(pretrained=True)
    relu2_2 = nn.Sequential(*list(vgg.features)[:9])
    for param in relu2_2.parameters():
        param.requires_grad = False
    relu2_2.eval
    relu2_2.cuda()
    """

    def __init__(self, loss_network):
        super(ContentLoss, self).__init__()
        self.loss_network = loss_network

    def forward(self, input, target):
        features_input = self.loss_network(input)
        features_target = self.loss_network(target)
        return torch.mean((features_input - features_target) ** 2)
