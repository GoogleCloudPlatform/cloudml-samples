from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'tensorflow==1.13.1',
    'tensorflow-model-analysis==0.13.0'
]

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='babyweight trainer package.'
)
