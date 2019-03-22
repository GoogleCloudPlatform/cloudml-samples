"""Cloud ML Engine package configuration."""
from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['requests==2.19.1']

setup(name='imdb',
      version='1.0',
      install_requires=REQUIRED_PACKAGES,
      include_package_data=True,
      packages=find_packages(),
      description='IMDB CMLE samples'
)
