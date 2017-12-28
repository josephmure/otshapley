# Always prefer setuptools over distutils
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='shapley-effects',
    version='0.1',
    description='Estimation of Shapley effects for Sensitivity Analysis of Model Output.',
    long_description=open('README.md').read(),
    url='https://gitlab.com/CEMRACS17/shapley-effects',
    author='Nazih BENOUMECHIARA & Kevin ELIE-DIT-COSAQUE',
    license='MIT',
    keywords='sensitivity analysis shapley effects depedencies',
    packages=['shapley'],
    install_requires=required
)