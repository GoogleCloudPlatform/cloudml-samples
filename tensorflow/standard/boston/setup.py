"""AI Platform package configuration."""
from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = []

setup(
  name='boston',
  version='1.0',
  install_requires=REQUIRED_PACKAGES,
  packages=find_packages(),
  include_package_data=True,
  description='AI Platform Boston samples'
)
