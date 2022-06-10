# Shapley effects

The *otshapley* library is a fork of [*shapley-effects*](https://gitlab.com/CEMRACS17/shapley-effects), updated to be compatible with Python 3.9+.
Surrogate model-related functionalities are removed to focus on the actual computation of Shapley effects.

Its purpose is to estimate the Shapley effects for Sensitivity Analysis of Model Output [[1]](http://epubs.siam.org/doi/pdf/10.1137/16M1097717).
Several features are available in the library. For a given probabilistic model and numerical function, it is possible to:

- compute the Shapley effects,
- compute the Sobol' indices for dependent and independent inputs.

The library is mainly built on top of NumPy and OpenTURNS. It is also validated and compared to the [`sensitivity`](https://github.com/cran/sensitivity/) package from the R software. 

## Important links

- Example notebooks are available in the [example directory](https://gitlab.com/CEMRACS17/shapley-effects/tree/dev/examples).

## Installation


### Via GitHub for the latest development version

```
>>> pip install git+https://github.com/josephmure/otshapley
```

### Dependencies

Various dependencies are necessary in this library and we strongly recommend the use of [Anaconda](https://anaconda.org/) for the installation. The dependencies are:

- Numpy,
- Scipy,
- Pandas,
- OpenTURNS.

Optional dependencies are also necessary for various task like plotting or tuning the model:

- Matplotlib,
- Seaborn.


## Acknowledgements

The library has been developed at the [CEMRACS 2017](http://smai.emath.fr/cemracs/cemracs17/) with the help of Bertrand Iooss, Roman Sueur, Veronique Maume-Deschamps and Clementine Prieur.

## References

[1] Owen, A. B., & Prieur, C. (2017). On Shapley value for measuring importance of dependent inputs. SIAM/ASA Journal on Uncertainty Quantification, 5(1), 986-1002.

[2] Song, E., Nelson, B. L., & Staum, J. (2016). Shapley effects for global sensitivity analysis: Theory and computation. SIAM/ASA Journal on Uncertainty Quantification, 4(1), 1060-1083.
