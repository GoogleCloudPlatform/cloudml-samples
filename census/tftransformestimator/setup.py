from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['tensorflow-transform==1.5.0', 'apache-beam[gcp]==2.34.0']

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Classifier test'
)
