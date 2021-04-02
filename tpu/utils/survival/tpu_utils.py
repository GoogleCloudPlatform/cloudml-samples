# Copyright 2018 Google LLC
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

from googleapiclient import discovery
from oauth2client.client import GoogleCredentials

credentials = GoogleCredentials.get_application_default()


def list_tpus(project, location):
    """List existing TPU nodes in the project/location.

    Args:
    project: (str) GCP project id.
    location: (str) GCP compute location, such as "us-central1-b".

    Returns
    A Python dictionary with keys 'nodes' and 'nextPageToken'.
    """
    service = discovery.build('tpu', 'v1', credentials=credentials)

    parent = 'projects/{}/locations/{}'.format(project, location)

    request = service.projects().locations().nodes().list(parent=parent)

    return request.execute()


def create_tpu(project, location, tpu_name, accelerator_type='v2-8',
               tensorflow_version='1.11', cidr_block='10.0.101.0',
               preemptible=False):
    """Create a TPU node.

    Args:
    project: (str) GCP project id.
    location: (str) GCP compute location, such as "us-central1-b".
    tpu_name: (str) The ID of the TPU node.
    accelerator_type: (str) The type of the TPU node, such as "v2-8".
    tensorflow_version: (str) The TensorFlow version, such as "1.11".
    cidr_block: (str) The CIDR block used by the TPU node,
    such as "10.0.101.0".
    preemptible: (bool) Whether the node should be created as preemptible.

    Returns
    A TPU node creation operation object.
    """
    service = discovery.build('tpu', 'v1', credentials=credentials)

    parent = 'projects/{}/locations/{}'.format(project, location)

    node = {
        'acceleratorType': accelerator_type,
        'tensorflowVersion': tensorflow_version,
        'network': 'default',
        'cidrBlock': cidr_block,
        'schedulingConfig': {
            'preemptible': preemptible
        }
    }

    # NOTE: in docs and samples nodeId is often referred to as tpu_name
    request = service.projects().locations().nodes().create(
        parent=parent, body=node, nodeId=tpu_name)

    return request.execute()


def get_tpu(project, location, tpu_name):
    """List existing TPU nodes in the project/location.

    Args:
    project: (str) GCP project id.
    location: (str) GCP compute location, such as "us-central1-b".
    tpu_name: (str) The ID of the TPU node.

    Returns
    A TPU node object.
    """
    service = discovery.build('tpu', 'v1', credentials=credentials)

    name = 'projects/{}/locations/{}/nodes/{}'.format(
        project, location, tpu_name)

    request = service.projects().locations().nodes().get(name=name)

    return request.execute()


def delete_tpu(project, location, tpu_name):
    """List existing TPU nodes in the project/location.

    Args:
    project: (str) GCP project id.
    location: (str) GCP compute location, such as "us-central1-b".
    tpu_name: (str) The ID of the TPU node.

    Returns
    A TPU node deletion operation object.
    """
    service = discovery.build('tpu', 'v1', credentials=credentials)

    name = 'projects/{}/locations/{}/nodes/{}'.format(
        project, location, tpu_name)

    request = service.projects().locations().nodes().delete(
        name=name)

    return request.execute()
