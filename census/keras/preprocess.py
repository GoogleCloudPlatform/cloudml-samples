# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""Pre-processing script to generate a sample for SavedModel prediction."""

import json
import sys
from trainer import model

if __name__=='__main__':
  gen = model.generator_input(['adult.data.csv'], chunk_size=5000)
  sample = gen.next()

  input_sample = {}

  input_sample['input'] = sample[0].values[0].tolist()
  print('Produced sample with label {}'.format(sample[1].values[0].tolist()))

  with open(sys.argv[1], 'w') as outfile:
    json.dump(input_sample, outfile)
