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

import torch.nn as nn


# Create the Deep Neural Network
class SonarDNN(nn.Module):
    def __init__(self):
        super(SonarDNN, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(60, 60),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(60, 30),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(30, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
