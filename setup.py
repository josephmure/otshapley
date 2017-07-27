#!/usr/bin/env python
"""
"""

# Always prefer setuptools over distutils
from setuptools import setup
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='shapley-indices',
    version='0.0.1',
    description='Shapley indices',
    url='',
    author='Nazih BENOUMECHIARA & Kevin ELIE-DIT-COSAQUE',
    author_email='nazih.benoumechiara@gmail.com',
    license='MIT',
    keywords='sensitivity analysis shapley',
    packages=['shapley'],
    install_requires=['numpy', 'matplotlib', 'seaborn', 'pandas']
)