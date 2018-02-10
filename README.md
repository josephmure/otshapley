[![build status](https://gitlab.com/CEMRACS17/shapley-effects/badges/master/build.svg)](https://gitlab.com/CEMRACS17/shapley-effects/commits/master)

[![coverage report](https://gitlab.com/CEMRACS17/shapley-effects/badges/master/coverage.svg)](https://gitlab.com/CEMRACS17/shapley-effects/commits/master)

# Shapley effects

Shapley-effects, or `shapley`, is a Python library that estimates the Shapley effects for the field of Sensitivity Analysis of Model Output [[1]](http://epubs.siam.org/doi/pdf/10.1137/16M1097717). Several features are available in the library. For a given probabilistic model and numerical function, it is possible to:

- compute the Shapley effects,
- compute the Sobol' indices for dependent and independent inputs,
- build a surrogate model to substitute the numerical function.

The library is mainly built on top of NumPy, OpenTURNS and other libraries. It is also validated and compared to the [`sensitivity`](https://github.com/cran/sensitivity/) package from the R software. 

## Important links

- Example notebooks are available in the [example directory](https://gitlab.com/CEMRACS17/shapley-effects/tree/dev/examples).
- Issues: [https://gitlab.com/CEMRACS17/shapley-effects/issues](https://gitlab.com/CEMRACS17/shapley-effects/issues)

## Installation

### Via PyPi

```
>>> pip install shapley-effects
```

### Via GitHub for the latest development version

```
>>> pip install git+https://gitlab.com/CEMRACS17/shapley-effects/tree/master
```

### Dependencies

Various dependencies are necessary in this library and we strongly recommend the use of [Anaconda](https://anaconda.org/) for the installation. The dependencies are:

- Numpy,
- Scipy,
- Pandas,
- OpenTURNS,
- Scikit-Learn,
- GPflow.

Scikit-learn is used to build kriging and random-forest models. OpenTURNS is a very convenient tool to define probabilistic distributions. GPflow which generates kriging models from GPy using Tensorflow.

Optional dependencies are also necessary for various task like plotting or tuning the model:

- Matplotlib,
- Seaborn,
- Scikit-Optimize.

These libraries can easily be installed using Anaconda and pip. Execute the following commands:

```
>>> conda install numpy pandas scikit-learn tensorflow matplotlib seaborn scikit-optimize
>>> conda install -c conda-forge openturns gpy
```

The package GPflow is not available on Anaconda or PyPi. Thus it must be installed from the source. First clone the GitHub repository:

```
>>> git clone https://github.com/GPflow/GPflow.git
```

Then, inside the GPflow folder, execute the command:

```
>>> pip install .
```

## Acknowledgements

The library has been developed at the [CEMRACS 2017](http://smai.emath.fr/cemracs/cemracs17/) with the help of Bertrand Iooss, Roman Sueur, Veronique Maume-Deschamps and Clementine Prieur.

## References

[1] Owen, A. B., & Prieur, C. (2017). On Shapley value for measuring importance of dependent inputs. SIAM/ASA Journal on Uncertainty Quantification, 5(1), 986-1002.

[2] Song, E., Nelson, B. L., & Staum, J. (2016). Shapley effects for global sensitivity analysis: Theory and computation. SIAM/ASA Journal on Uncertainty Quantification, 4(1), 1060-1083.