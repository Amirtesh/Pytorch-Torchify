from setuptools import setup, find_packages

setup(
    name='torchkit',
    version='0.1.0',
    description='A Keras-like API for PyTorch for image and tabular data.',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/Amirtesh/Pytorch-TorchKit',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'matplotlib',
        'numpy'
       ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)
