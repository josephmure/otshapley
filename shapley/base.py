import openturns as ot


class Base(object):
    """Base class.

    Parameters
    ----------
    input_distribution : ot.DistributionImplementation,
        And OpenTURNS distribution object.
    """
    def __init__(self, input_distribution):
        self.input_distribution = input_distribution
        self.first_order_indice_func = None
        self.total_indice_func = None

    @property
    def input_distribution(self):
        """The OpenTURNS input distribution.
        """
        return self._input_distribution

    @input_distribution.setter
    def input_distribution(self, dist):
        assert isinstance(dist, ot.DistributionImplementation), \
            "The distribution should be an OpenTURNS Distribution object. Given %s" % (type(dist))
        self._input_distribution = dist

    @property
    def dim(self):
        """The input dimension.
        """
        return self._input_distribution.getDimension()

    @property
    def first_order_indice_func(self):
        """Function for 1st indice computation.
        """
        return self._first_order_indice_func

    @first_order_indice_func.setter
    def first_order_indice_func(self, func):
        assert callable(func) or func is None, \
            "First order indice function should be callable or None."

        self._first_order_indice_func = func

    @property
    def total_indice_func(self):
        """Function for 1st indice computation.
        """
        return self._first_order_indice_func

    @total_indice_func.setter
    def total_indice_func(self, func):
        assert callable(func) or func is None, \
            "Total indice function should be callable or None."

        self._first_order_indice_func = func