"""Cloud ML Engine package configuration."""
from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['requests==2.19.1',
										 'google-api-python-client>=1.6.7'
										 'mlperf_compliance==0.0.10'
										 'oauth2client>=4.1.2'
										 'pandas'
										 'psutil>=5.4.3'
										 'py-cpuinfo>=3.3.0'
										 'typing']

setup(name='mnist',
      version='1.0',
      install_requires=REQUIRED_PACKAGES,
      include_package_data=True,
      packages=find_packages(),
      description='MNIST CMLE samples'
)
