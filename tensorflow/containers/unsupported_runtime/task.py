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

import data_utils
import model


def train_model(args):
    train_features, test_features, train_labels, test_labels = \
        data_utils.load_data(args)

    sonar_model = model.sonar_model()

    sonar_model.fit(train_features, train_labels, epochs=args.epochs,
                    batch_size=args.batch_size)

    score = sonar_model.evaluate(test_features, test_labels,
                                 batch_size=args.batch_size)
    print(score)

    # Export the trained model
    sonar_model.save(args.model_name)

    if args.model_dir:
        # Save the model to GCS
        data_utils.save_model(args.model_dir, args.model_name)


def get_args():
    parser = argparse.ArgumentParser(description='Keras Sonar Example')
    parser.add_argument('--model-dir',
                        type=str,
                        help='Where to save the model')
    parser.add_argument('--model-name',
                        type=str,
                        default='sonar_model.h5',
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
