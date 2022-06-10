# Always prefer setuptools over distutils

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='otshapley',
    version='0.2',
    description='Estimation of Shapley effects for Sensitivity Analysis of Model Output.',
    long_description=open('README.md').read(),
    url='https://github.com/josephmure/otshapley',
    author='Nazih BENOUMECHIARA & Kevin ELIE-DIT-COSAQUE & Joseph MURÉ',
    author_email = 'joseph.mure@edf.fr',
    license='MIT',
    keywords=['sensitivity analysis', 'shapley', 'effects', 'dependencies'],
    packages=['otshapley', 'otshapley.tests'],
    install_requires=required
)
