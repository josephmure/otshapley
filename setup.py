# Always prefer setuptools over distutils
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

from pip.req import parse_requirements
install_reqs = parse_requirements('requirements.txt')
reqs = [str(ir.req) for ir in install_reqs]

setup(
    name='shapley-indices',
    version='0.0.1',
    description='Estimation of Shapley Indices for Sensitivity Analysis.',
    long_description=open('README.md').read(),
    url='https://gitlab.com/CEMRACS17/shapley-indices',
    author='Nazih BENOUMECHIARA & Kevin ELIE-DIT-COSAQUE',
    license='MIT',
    keywords='sensitivity analysis shapley',
    packages=['shapley'],
    install_requires=reqs
)