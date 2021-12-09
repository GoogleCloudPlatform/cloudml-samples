import glob
import os
import re
import yaml

root = '..'
BASE_URL = 'https://github.com/GoogleCloudPlatform/cloudml-samples/blob/main/tpu/templates/{}/{}'

with open('samples.yaml', 'r') as f:
    samples = yaml.load(f.read())

interface_mapping = {
    'estimator': 'TPUEstimator',
    'keras': 'Keras',
    'rewrite': 'low-level TensorFlow',
}

name_mapping = {
    'triplet-loss': 'Triplet Loss',
    'gan': 'GAN',
    'dense': 'fully connected neural network',
    'cnn': 'convolutional neural network',
    'lstm': 'Long Short Term Memory',
    'grl': 'Gradient Reversal Layer',
    'film': 'Feature Wise Linear Modulation'
}

manifest = []

for name, info_dict in samples.items():
    for interface, sample_name in info_dict.items():
        out_path = os.path.join(root, sample_name)

        ipynb_filenames = glob.glob(os.path.join(out_path, '*.ipynb'))

        for ipynb_filename in ipynb_filenames:
            part = ipynb_filename.split('/')[-1].split('.')[-2]
            sub_readme_filename = part + '_readme.md'

            mapped_sample_name = name_mapping[name]
            mapped_interface = interface_mapping[interface]

            with open('SUB_README_BASE.md', 'r') as f:
                content = f.read()

            content = content.format(sample_name=mapped_sample_name, interface=mapped_interface)

            with open(os.path.join(out_path, sub_readme_filename), 'w') as f:
                f.write(content)

            # urls to the notebooks and the readmes
            ipynb_url = BASE_URL.format(sample_name, part+'.ipynb')
            sub_readme_url = BASE_URL.format(sample_name, sub_readme_filename)

            manifest.append({
                'ipynb_url': ipynb_url,
                'sub_readme_url': sub_readme_url,
                'mapped_sample_name': mapped_sample_name,
                'mapped_interface': mapped_interface
            })
