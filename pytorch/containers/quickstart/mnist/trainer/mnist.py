# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the \"License\");
# you may not use this file except in compliance with the License.\n",
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an \"AS IS\" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import argparse
import os
import subprocess
import hypertune
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
from torchvision import transforms

MODEL_FILE_NAME = 'torch.model'


class Net(nn.Module):

  def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
    self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
    self.conv2_drop = nn.Dropout2d()
    self.fc1 = nn.Linear(320, 50)
    self.fc2 = nn.Linear(50, 10)

  def forward(self, x):
    x = F.relu(F.max_pool2d(self.conv1(x), 2))
    x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
    x = x.view(-1, 320)
    x = F.relu(self.fc1(x))
    x = F.dropout(x, training=self.training)
    x = self.fc2(x)
    return F.log_softmax(x, dim=1)


def train(args, model, device, train_loader, optimizer, epoch):
  model.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % args.log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
          epoch, batch_idx * len(data), len(train_loader.dataset),
          100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader, epoch):
  model.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      data, target = data.to(device), target.to(device)
      output = model(data)
      test_loss += F.nll_loss(
          output, target, size_average=False).item()  # sum up batch loss
      pred = output.max(
          1, keepdim=True)[1]  # get the index of the max log-probability
      correct += pred.eq(target.view_as(pred)).sum().item()

  test_loss /= len(test_loader.dataset)
  print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
      test_loss, correct, len(test_loader.dataset),
      100. * correct / len(test_loader.dataset)))

  # Uses hypertune to report metrics for hyperparameter tuning.
  hpt = hypertune.HyperTune()
  hpt.report_hyperparameter_tuning_metric(
      hyperparameter_metric_tag='my_loss',
      metric_value=test_loss,
      global_step=epoch)


def get_args():
  """Argument parser.
  Returns:
    Dictionary of arguments.
  """
  parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
  parser.add_argument(
      '--batch-size',
      type=int,
      default=64,
      metavar='N',
      help='input batch size for training (default: 64)')
  parser.add_argument(
      '--test-batch-size',
      type=int,
      default=1000,
      metavar='N',
      help='input batch size for testing (default: 1000)')
  parser.add_argument(
      '--epochs',
      type=int,
      default=10,
      metavar='N',
      help='number of epochs to train (default: 10)')
  parser.add_argument(
      '--lr',
      type=float,
      default=0.01,
      metavar='LR',
      help='learning rate (default: 0.01)')
  parser.add_argument(
      '--momentum',
      type=float,
      default=0.5,
      metavar='M',
      help='SGD momentum (default: 0.5)')
  parser.add_argument(
      '--no-cuda',
      action='store_true',
      default=False,
      help='disables CUDA training')
  parser.add_argument(
      '--seed',
      type=int,
      default=1,
      metavar='S',
      help='random seed (default: 1)')
  parser.add_argument(
      '--log-interval',
      type=int,
      default=10,
      metavar='N',
      help='how many batches to wait before logging training status')
  parser.add_argument(
      '--model-dir',
      default=None,
      help='The directory to store the model')

  args = parser.parse_args()
  return args

def main():
  # Training settings
  args = get_args()
  use_cuda = not args.no_cuda and torch.cuda.is_available()

  torch.manual_seed(args.seed)

  device = torch.device('cuda' if use_cuda else 'cpu')

  kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
  train_loader = torch.utils.data.DataLoader(
      datasets.MNIST(
          '../data',
          train=True,
          download=True,
          transform=transforms.Compose([
              transforms.ToTensor(),
              # Normalize a tensor image with mean and standard deviation
              transforms.Normalize(mean=(0.1307,), std=(0.3081,))
          ])),
      batch_size=args.batch_size,
      shuffle=True,
      **kwargs)
  test_loader = torch.utils.data.DataLoader(
      datasets.MNIST(
          '../data',
          train=False,
          transform=transforms.Compose([
              transforms.ToTensor(),
              # Normalize a tensor image with mean and standard deviation              
              transforms.Normalize(mean=(0.1307,), std=(0.3081,))
          ])),
      batch_size=args.test_batch_size,
      shuffle=True,
      **kwargs)

  model = Net().to(device)
  optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

  for epoch in range(1, args.epochs + 1):
    train(args, model, device, train_loader, optimizer, epoch)
    test(args, model, device, test_loader, epoch)

  if args.model_dir:
    tmp_model_file = os.path.join('/tmp', MODEL_FILE_NAME)
    torch.save(model.state_dict(), tmp_model_file)
    subprocess.check_call([
        'gsutil', 'cp', tmp_model_file,
        os.path.join(args.model_dir, MODEL_FILE_NAME)])

if __name__ == '__main__':
  main()