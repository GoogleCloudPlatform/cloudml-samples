import setuptools

setuptools.setup(
    name='resnet-tpu',
    version='0.0.1',
    install_requires=['cloudml-hypertune'],
    packages=setuptools.find_packages()
)