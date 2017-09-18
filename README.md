[![build status](https://gitlab.com/CEMRACS17/shapley-indices/badges/master/build.svg)](https://gitlab.com/CEMRACS17/shapley-indices/commits/master)
[![coverage report](https://gitlab.com/CEMRACS17/shapley-indices/badges/master/coverage.svg)](https://gitlab.com/CEMRACS17/shapley-indices/commits/master)
# Shapley indices library

This python library estimates the Shapley indices in the field of sensitivity analysis of model output. The module also propose to estimate the well known Sobol' indices. For costly computational model, the module also propose to approximate with a surrogate model (or meta-model). The available meta-model are the Gaussian Process and the Random Forest.

## Installation

The package has various dependencies and we strongly recommend the use of Anaconda for the installation. The dependencies are :

- Numpy,
- Scipy,
- Pandas,
- Scikit-Learn,
- OpenTURNS,
- Tensorflow,
- GPy,
- GPflow.

Scikit-learn is used to build random-forest and Kriging models. OpenTURNS is very convenient tool to create probabilistic distributions. Tensorflow and GPy are two dependencies of GPflow which generates Kriging models from GPy using Tensorflow.

Optional dependencies are also necessary for various task like plotting or tuning the model:

- Matplotlib,
- Seaborn,
- Scikit-Optimize.

These libraries can easily be installed using Anaconda and pip. Execute the following commands:

```
conda install numpy pandas scikit-learn tensorflow matplotlib seaborn scikit-optimize
conda install -c conda-forge openturns gpy
```

The package GPflow is not availaible on Anaconda or PyPi. Thus it must be installed from the source. First clone the GitHub repository:

```
git clone https://github.com/GPflow/GPflow.git
```

Then, inside the GPflow folder, execute the command:

```
pip install .
```