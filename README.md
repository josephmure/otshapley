[![build status](https://gitlab.com/CEMRACS17/shapley-indices/badges/master/build.svg)](https://gitlab.com/CEMRACS17/shapley-indices/commits/master)
[![coverage report](https://gitlab.com/CEMRACS17/shapley-indices/badges/master/coverage.svg)](https://gitlab.com/CEMRACS17/shapley-indices/commits/master)
# Shapley indices library


## TODO list
- Implement all the indices
	- Total order Sobol
	- Second order indices
	- Ind sobol indices
- Try to estimate the sobol indices from Mara & Tarantola, using a classical pick and freeze estimator.
- Cythonize some parts?
- Unique compute_indices_function in Indices class that works for kriging too.
- Create few unitest functions
	- On additive Gaussian model with correlation

- Should we normalize the model output?