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

    def imagenet_preprocess(self, batch):
        tensortype = type(batch.data)
        mean = tensortype(batch.data.size())
        std = tensortype(batch.data.size())

        mean[:, 0, :, :] = 0.485
        mean[:, 1, :, :] = 0.456
        mean[:, 2, :, :] = 0.406

        std[:, 0, :, :] = 0.229
        std[:, 1, :, :] = 0.224
        std[:, 2, :, :] = 0.225

        return (batch - Variable(mean)) / Variable(std)


    def forward(self, input, target):
        input = self.imagenet_preprocess(input)
        target = self.imagenet_preprocess(target)
        features_input = self.loss_network(input)
        features_target = self.loss_network(target)
        return torch.mean((features_input - features_target) ** 2)
