# Copyright 2019 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Helper function for requesting an online prediction."""

import googleapiclient.discovery


def predict_json(project, model, datas, version=None):
  """Send json data to a deployed model for prediction.

  Args:
      project: (str), project where the Cloud ML Engine Model is deployed.
      model: (str), model name.
      datas: ([[any]]), list of input instances, where each input
         instance is a list of attributes.
      version: str, version of the model to target.
  Returns:
      Mapping[str: any]: dictionary of prediction results defined by the
          model.
  """
  # Create the ML Engine service object.
  # To authenticate set the environment variable
  # GOOGLE_APPLICATION_CREDENTIALS=<path_to_service_account_file>
  service = googleapiclient.discovery.build('ml', 'v1')
  name = 'projects/{}/models/{}'.format(project, model)

  if version is not None:
    name += '/versions/{}'.format(version)

  response = service.projects().predict(
      name=name,
      body={'instances': datas}
  ).execute()

  if 'error' in response:
    raise RuntimeError(response['error'])

  return response['predictions']


if __name__ == '__main__':
  project_id = 'YOUR_PROJECT_ID'
  model_name = 'YOUR_MODEL_NAME'
  data_list = [[]]
  version_name = 'YOUR_VERSION_NAME'
  print(predict_json(project=project_id,
                     model=model_name,
                     datas=data_list,
                     version=version_name))
