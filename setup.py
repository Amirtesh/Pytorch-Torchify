from setuptools import setup, find_packages

setup(
    name='torchify',
    version='0.1.0',
    description='A Keras-like API for PyTorch with support for image and tabular data',
    author='Amirtesh Raghuram',
    url='https://github.com/Amirtesh/Pytorch-Torchify/tree/main',
    packages=find_packages(),
    install_requires=[
        'torch>=1.10.0',
        'numpy',
        'matplotlib',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
