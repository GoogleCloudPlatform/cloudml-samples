"""Cloud ML Engine package configuration."""
from setuptools import find_packages
from setuptools import setup


setup(
  name='boston',
  version='0.1',
  install_requires=requirements,
  packages=find_packages(),
  include_package_data=True,
  description='CMLE samples'
)
