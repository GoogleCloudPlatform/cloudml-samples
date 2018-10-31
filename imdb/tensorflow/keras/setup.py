"""Cloud ML Engine package configuration."""
from setuptools import find_packages
from setuptools import setup

with open('requirements.txt') as f:
  requirements = [l.strip('\n') for l in f if
                  l.strip('\n') and not l.startswith('#')]

setup(
  name='imdb',
  version='0.1',
  install_requires=requirements,
  packages=find_packages(),
  include_package_data=True,
  description='CMLE samples'
)