from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms

class AddGaussianNoise(object):
    # this class was used to add noise to an image, the noise obeys Gaussian distribution with
    # parameters mean and std
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)



class GaborConv(nn.Module):
    def __init__(self, kernel_size, in_channels, channel1):
        super(GaborConv, self).__init__()
        # generate parameters needed for a gabor filter
        self.sigma1, self.theta1, self.Lambda1, self.psi1, self.gamma1, self.bias1 = self.generate_parameters(channel1//2, in_channels)
        self.sigma2, self.theta2, self.Lambda2, self.psi2, self.gamma2, self.bias2 = self.generate_parameters(channel1//2, in_channels)
        # generate real and imaginary part of Gabor filters as a tensor of shape in_channels * channel1//2 * kernel_size * kernel_size
        self.filter_cos = self.whole_filter(in_channels, channel1//2, kernel_size, self.sigma1, self.theta1, self.Lambda1, self.psi1, self.gamma1, True).cuda()
        self.filter_sin = self.whole_filter(in_channels, channel1//2, kernel_size, self.sigma1, self.theta1, self.Lambda1, self.psi1, self.gamma1, False).cuda()
        # the second layer of the network is a conventional CNN layer
        self.conv = nn.Conv2d(20, 50, 5, 1)
        # last two layers of the network are fully connected
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)


    def forward(self, x):
        # building the architecture
        # first convolve the image with real and imaginary part of Gabor filters separately and concatenate the results
        x_cos = F.conv2d(x, self.filter_cos, bias=self.bias1)
        x_sin = F.conv2d(x, self.filter_sin, bias=self.bias2)
        x_comb = torch.cat((x_cos, x_sin), 1)
        # max pooling and non-linear activation(ReLU)
        x_comb = F.max_pool2d(x_comb, 2, 2)
        x_comb = F.relu(self.conv(x_comb))
        x_comb = F.max_pool2d(x_comb, 2, 2)
        # change the tensor to a 1-dim verctor as input of the fully connected layers
        x_comb = x_comb.view(-1, 4*4*50)
        x_comb = F.relu(self.fc1(x_comb))
        x_comb = self.fc2(x_comb)
        return F.log_softmax(x_comb, dim=1)


    def generate_parameters(self, dim_out, dim_in):
        sigma = nn.Parameter(torch.randn(dim_out, dim_in))
        theta = nn.Parameter(torch.randn(dim_out, dim_in))
        Lambda = nn.Parameter(torch.randn(dim_out, dim_in))
        psi = nn.Parameter(torch.randn(dim_out, dim_in))
        gamma = nn.Parameter(torch.randn(dim_out, dim_in))
        bias = nn.Parameter(torch.randn(dim_out))
        return sigma, theta, Lambda, psi, gamma, bias

    def one_filter(self, in_channels, kernel_size, sigma, theta, Lambda, psi, gamma, cos):
        # generate Gabor filters as a tensor of shape in_channels * kernel_size * kernel_size
        result = torch.zeros(in_channels, kernel_size, kernel_size)
        for i in range(in_channels):
            result[i] = self.gabor_fn(sigma[i], theta[i], Lambda[i], psi[i], gamma[i], kernel_size, cos)
        return nn.Parameter(result)

    def whole_filter(self, in_channels, out_channels, kernel_size, sigma_column, theta_column, Lambda_column, psi_column, gamma_column, cos):
        # generate Gabor filters as a tensor of shape out_channels * in_channels * kernel_size * kernel_size
        result = torch.zeros(out_channels, in_channels, kernel_size, kernel_size) # \text{out\_channels} , \frac{\text{in\_channels}}{\text{groups}} , kH , kW
        for i in range(out_channels):
            result[i] = self.one_filter(in_channels, kernel_size, sigma_column[i], theta_column[i], Lambda_column[i], psi_column[i], gamma_column[i], cos)
        return nn.Parameter(result)

    def gabor_fn(self, sigma, theta, Lambda, psi, gamma, kernel_size, cos):
        # generate a single Gabor filter, modified https://en.wikipedia.org/wiki/Gabor_filter#Example_implementations
        sigma_x = sigma
        # sigma_y = float(sigma) / gamma
        sigma_y = sigma / gamma

        # Bounding box
        half_size = (kernel_size - 1) // 2
        ymin, xmin = -half_size, -half_size
        ymax, xmax = half_size, half_size
        (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))
        y = torch.FloatTensor(y)
        x = torch.FloatTensor(x)

        # Rotation
        x_theta = x * torch.cos(theta) + y * torch.sin(theta)
        y_theta = -x * torch.sin(theta) + y * torch.cos(theta)

        if cos:
            gb = torch.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * torch.cos(2 * np.pi / Lambda * x_theta + psi)
        else:
            gb = torch.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * torch.sin(2 * np.pi / Lambda * x_theta + psi)
        return gb



def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward(retain_graph=True)
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--std', type=float, default=1.0, metavar='STD',
                        help='noise-std (default: 1.0)')
    parser.add_argument('--mean', type=float, default=0.5, metavar='MEAN',
                        help='noise-mean (default: 0.5)')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,)),
                           transforms.RandomApply([AddGaussianNoise(args.mean, args.std)], p=0.5)
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                           # AddGaussianNoise(args.mean, args.std)
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)


    model = GaborConv(5, 1, 20).to(device)
    # if torch.cuda.is_available():
    #     torch.set_default_tensor_type('torch.cuda.FloatTensor')
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for param in model.parameters():
        print(type(param.data), param.size())

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)
        # for param in model.parameters():
        #     print(param.size(), param.data)
        # print(model.state_dict())

    if (args.save_model):
        torch.save(model.state_dict(),"mnist_one_layer_gabor.pt")

if __name__ == '__main__':
    main()
