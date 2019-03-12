from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'tensorflow==1.12.0',
    'tensorflow-model-analysis==0.11.0'
]

setup(
    name='',
    version='',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description=''
)
