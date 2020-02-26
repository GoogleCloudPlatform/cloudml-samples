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

import argparse
import hypertune
import torch
import torch.optim as optim
import torch.nn as nn

import data_utils
import model


def train(net, train_loader, optimizer):
    """Create the training loop"""
    net.train()
    criterion = nn.BCELoss()

    for batch_index, data in enumerate(train_loader):
        features = data['features']
        target = data['target']

        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(features)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()


def test(net, test_loader):
    """Test the DNN"""
    net.eval()
    criterion = nn.BCELoss()  # https://pytorch.org/docs/stable/nn.html#bceloss
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            features = data['features']
            target = data['target']
            output = net(features)
            # Binarize the output
            pred = output.apply_(lambda x: 0.0 if x < 0.5 else 1.0)
            test_loss += criterion(output, target)  # sum up batch loss
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    total = len(test_loader) * test_loader.batch_size
    accuracy = 100. * correct / total
    return accuracy


def train_model(args):
    """Load the data, train the model, test the model, export / save the model
    """
    torch.manual_seed(args.seed)

    # Open our dataset
    train_loader, test_loader = data_utils.load_data(
        args.test_split, args.seed, args.batch_size)

    # Create the model
    net = model.SonarDNN().double()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=args.momentum, nesterov=False)

    # Train / Test the model
    latest_accuracy = 0.0
    for epoch in range(1, args.epochs + 1):
        train(net, train_loader, optimizer)
        latest_accuracy = test(net, test_loader)

    # The default name of the metric is training/hptuning/metric.
    # We recommend that you assign a custom name. The only functional
    # difference is that if you use a custom name, you must set the
    # hyperparameterMetricTag value in the HyperparameterSpec object in your
    # job request to match your chosen name.
    # https://cloud.google.com/ml-engine/reference/rest/v1/projects.jobs#HyperparameterSpec
    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag='my_accuracy_tag',
        metric_value=latest_accuracy,
        global_step=args.epochs)

    # Export the trained model
    torch.save(net.state_dict(), args.model_name)

    if args.job_dir:
        # Save the model to GCS
        data_utils.save_model(args.job_dir, args.model_name)
    else:
        print('Accuracy: {:.0f}%'.format(latest_accuracy))


def get_args():
    """Argument parser.
    Returns:
        Dictionary of arguments.
    """
    parser = argparse.ArgumentParser(description='PyTorch Sonar Example')
    parser.add_argument('--job-dir',  # handled automatically by AI Platform
                        help='GCS location to write checkpoints and export ' \
                             'models')
    parser.add_argument('--model-name',
                        type=str,
                        default="hptuning_sonar_model",
                        help='What to name the saved model file')
    parser.add_argument('--batch-size',
                        type=int,
                        default=4,
                        help='input batch size for training (default: 4)')
    parser.add_argument('--test-split',
                        type=float,
                        default=0.2,
                        help='split size for training / testing dataset')
    parser.add_argument('--epochs',
                        type=int,
                        default=10,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr',  # Specified in the config file
                        type=float,
                        default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum',  # Specified in the config file
                        type=float,
                        default=0.5,
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help='random seed (default: 42)')
    args = parser.parse_args()

    return args


def main():
    args = get_args()
    train_model(args)


if __name__ == '__main__':
    main()
